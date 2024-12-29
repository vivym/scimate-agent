from functools import lru_cache
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from scimate_agent.state import CodeInterpreterState, Post, RoundUpdate
from .session import SessionManager, SessionClient


# This is a temporary function to get session manager.
# TODO: Seperate code executor to its own service.
@lru_cache
def _get_session_mgr(env_id: str, env_dir: str) -> SessionManager:
    return SessionManager(env_id=env_id, env_dir=env_dir)


@lru_cache
def _get_session_client(session_mgr: SessionManager, thread_id: str) -> SessionClient:
    return session_mgr.get_session_client(session_id=thread_id)


def code_executor_node(state: CodeInterpreterState, config: RunnableConfig) -> dict[str, Any]:
    rounds = state.get_rounds()
    assert len(rounds) > 0, "No round found for CodeExecutor."

    last_round = rounds[-1]
    assert len(last_round.posts) > 0, "No post found for CodeExecutor."
    last_post = last_round.posts[-1]

    assert last_post.send_to == "CodeExecutor", "The latest post is not sent to CodeExecutor."
    assert last_post.send_from == "CodeVerifier", "The latest post is not from CodeVerifier."

    code = last_post.message

    if state.code_executor_session_mgr is None:
        env_id = config["configurable"].get("env_id", None)
        env_dir = config["configurable"].get("env_dir", None)
    else:
        env_id, env_dir = state.code_executor_session_mgr

    session_mgr = _get_session_mgr(env_id, env_dir)
    env_id = session_mgr.env_id
    env_dir = session_mgr.env_dir

    if state.code_executor_session_client is None:
        session_id = config["configurable"].get("thread_id", None)
        if session_id is None:
            raise ValueError("Thread ID is required.")
        session_id = str(session_id)
        session_client = _get_session_client(session_mgr, session_id)
        session_client.start()
    else:
        session_id = state.code_executor_session_client
        session_client = _get_session_client(session_mgr, session_id)

    result = session_client.execute_code(exec_id=f"{session_id}-{last_round.id}", code=code)

    self_correction_count = state.self_correction_count

    if result.is_success:
        post = Post.new(
            send_from="CodeExecutor",
            send_to="Planner",
            message=result.code,
            original_messages=last_post.original_messages,
        )
        self_correction_count = None
    else:
        # Self-correct the code
        post = Post.new(
            send_from="CodeExecutor",
            send_to="CodeGenerator",
            message=result.error,
            original_messages=last_post.original_messages,
        )
        self_correction_count = self_correction_count + 1 if self_correction_count is not None else 1

    return {
        "rounds": RoundUpdate(
            id=last_round.id,
            posts=[post],
        ),
        "self_correction_count": self_correction_count,
        "code_executor_session_mgr": (env_id, env_dir),
        "code_executor_session_client": session_id,
    }


def code_executor_router_edge(state: CodeInterpreterState) -> str:
    rounds = state.get_rounds()
    assert len(rounds) > 0, "No round found for CodeExecutor."

    last_round = rounds[-1]
    if len(last_round.posts) == 0:
        raise ValueError("No post found for CodeExecutor.")
    last_post = last_round.posts[-1]

    assert last_post.send_from == "CodeExecutor", "Last post is not from CodeExecutor."

    if last_post.send_to == "Planner":
        return END
    elif last_post.send_to == "CodeGenerator":
        if state.self_correction_count is None or state.self_correction_count <= 3:
            return "code_generator_node"
        else:
            return END
    else:
        raise ValueError(f"Invalid post to: {last_post.send_to}")
