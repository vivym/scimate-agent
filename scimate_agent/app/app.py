from dataclasses import dataclass
from typing import Any, Literal

from langgraph.types import Command
from socketio import AsyncServer

from scimate_agent.agent import scimate_agent_graph
from scimate_agent.event import EventEmitter
from scimate_agent.interrupt import ExitCommand, Interruption
from scimate_agent.nodes.code_executor import get_session_client
from scimate_agent.state import AgentState, Post, Round, load_plugins

SessionState = Literal["idle", "running", "interrupted"]


@dataclass
class Session:
    session_id: str
    env_id: str
    env_dir: str
    ce_session_id: str | None
    state: SessionState


class SciMateAgentApp:
    def __init__(self, sio: AsyncServer):
        self.sio = sio
        self.sessions: dict[str, Session] = {}

    async def on_connect(self, sid, environ, auth):
        self.sessions[sid] = Session(
            session_id=sid,
            env_id="debug",
            env_dir="tmp/workspace",
            ce_session_id=None,
            state="idle",
        )

        async def on_event(event_name: str, data: Any):
            if event_name == "code_executor_start":
                session = self.sessions[sid]
                session.env_id = data["env_id"]
                session.env_dir = data["env_dir"]
                session.ce_session_id = data["session_id"]

            await self.sio.emit(
                "event",
                data={"event_name": event_name, "data": data},
                to=sid,
            )

        event_emitter = EventEmitter.get_instance(sid)
        event_emitter.on("*", on_event)

    async def on_disconnect(self, sid):
        EventEmitter.remove_instance(sid)

        session = self.sessions[sid]

        # Stop the execution session
        if session.ce_session_id is not None:
            assert session.env_id is not None, "Internal error: env_id is None"
            assert session.env_dir is not None, "Internal error: env_dir is None"

            client = await get_session_client(
                env_id=session.env_id,
                env_dir=session.env_dir,
                session_id=session.ce_session_id,
                create_if_not_exists=False,
            )
            await client.stop()

        del self.sessions[sid]

    async def on_user_query(self, sid, user_query: str):
        session = self.sessions[sid]

        if session.state == "running":
            # TODO: normalize the error message
            self.sio.emit("error", "Session is already running")
            return

        thread_config = {
            "configurable": {
                "thread_id": session.session_id,
                "env_id": session.env_id,
                "env_dir": session.env_dir,
                "session_id": session.session_id,
                "event_handle": session.session_id,
            }
        }

        event_emitter = EventEmitter.get_instance(session.session_id)

        if session.state == "interrupted":
            if user_query == "":
                graph_input = Command(resume=ExitCommand())
            else:
                graph_input = Command(resume=user_query)
        elif session.state == "idle":
            graph_input = AgentState(
                rounds=[
                    Round.new(
                        user_query=user_query,
                        posts=[
                            Post.new(
                                send_from="User",
                                send_to="Planner",
                                message=user_query,
                            )
                        ],
                    )
                ],
                plugins=load_plugins(["scimate_agent/plugins/builtins"]),
                env_id=session.env_id,
                env_dir=session.env_dir,
                session_id=session.session_id,
            )
        else:
            raise ValueError(f"Invalid session state: {session.state}")

        session.state = "running"

        async for event in scimate_agent_graph.astream(
            input=graph_input,
            config=thread_config,
            stream_mode="updates",
            subgraphs=True,
        ):
            if "__interrupt__" in event[1]:
                for interrupt_event in event[1]["__interrupt__"]:
                    assert isinstance(interrupt_event.value, Interruption), (
                        "Interrupt event must be an instance of Interruption."
                    )

                    await event_emitter.emit(
                        "interrupt",
                        interrupt_event.value.model_dump(mode="json"),
                    )

                session.state = "interrupted"

        if session.state == "running":
            session.state = "idle"

    async def stop(self):
        for sid in self.sessions.keys():
            await self.sio.disconnect(sid)

        await self.sio.shutdown()
