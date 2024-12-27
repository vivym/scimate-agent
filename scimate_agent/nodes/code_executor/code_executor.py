from typing import Any

from langchain_core.runnables import RunnableConfig

from scimate_agent.state import AgentState
from .session import SessionManager


def code_executor_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    rounds = state.get_rounds("CodeExecutor")
    assert len(rounds) > 0, "No round found for CodeExecutor."

    last_round = rounds[-1]
    assert len(last_round.posts) > 0, "No post found for CodeExecutor."
    last_post = last_round.posts[-1]

    assert last_post.send_to == "CodeExecutor", "The latest post is not sent to CodeExecutor."

    code = last_post.message

    if state.code_executor_session_mgr is None:
        env_id = config["configurable"].get("env_id", None)
        env_dir = config["configurable"].get("env_dir", None)
        session_mgr = SessionManager(env_id=env_id, env_dir=env_dir)
        session_mgr.initialize()
        state.code_executor_session_mgr = session_mgr

    session_mgr = state.code_executor_session_mgr

    if state.code_executor_session_client is None:
        thread_id = config["configurable"].get("thread_id", None)
        if thread_id is None:
            raise ValueError("Thread ID is required.")
        session_client = session_mgr.get_session_client(session_id=thread_id)
        session_client.start()
        state.code_executor_session_client = session_client

    session_client = state.code_executor_session_client

    result = session_client.execute_code(exec_id=thread_id, code=code)

    print("-" * 100)
    print("CodeExecutor Node")
    print(result)
    print("-" * 100)

    return {}


def code_executor_router_edge(state: AgentState) -> str:
    ...
