from dataclasses import dataclass
from typing import Any,Literal

from socketio import AsyncServer

from scimate_agent.agent import scimate_agent_graph
from scimate_agent.event import EventEmitter
from scimate_agent.interrupt import Interruption
from scimate_agent.state import AgentState, Post, Round

SessionState = Literal["idle", "running"]


@dataclass
class Session:
    session_id: str
    env_id: str
    env_dir: str
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
            state="idle",
        )

        async def on_event(event_name: str, data: Any):
            print(f"event_name: {event_name}, data: {data}")

            await self.sio.emit(
                "event",
                data={"event_name": event_name, "data": data},
                to=sid,
            )

        event_emitter = EventEmitter.get_instance(sid)
        event_emitter.on("*", on_event)

    async def on_disconnect(self, sid):
        EventEmitter.remove_instance(sid)

        # TODO: clean up the session
        del self.sessions[sid]

    async def on_user_query(self, sid, user_query: str):
        session = self.sessions[sid]

        if session.state == "running":
            # TODO: normalize the error message
            self.sio.emit("error", "Session is already running")
            return

        session.state = "running"

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

        input_state = AgentState(
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
            plugins=[],
            env_id=session.env_id,
            env_dir=session.env_dir,
            session_id=session.session_id,
        )

        async for event in scimate_agent_graph.astream(
            input=input_state,
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

        session.state = "idle"

    async def stop(self):
        ...
