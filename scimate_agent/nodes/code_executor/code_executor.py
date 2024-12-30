import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from scimate_agent.state import Attachment, AttachmentType, CodeInterpreterState, Post, RoundUpdate
from .session import ExecutionResult, SessionManager, SessionClient

TRUNCATE_CHAR_LENGTH = 1000


# This is a temporary function to get session manager.
# TODO: Seperate code executor to its own service.
@lru_cache
def _get_session_mgr(env_id: str, env_dir: str) -> SessionManager:
    return SessionManager(env_id=env_id, env_dir=env_dir)


@lru_cache
def _get_session_client(session_mgr: SessionManager, thread_id: str) -> SessionClient:
    return session_mgr.get_session_client(session_id=thread_id)


def get_artifact_uri(execution_id: str, file: str, use_local_uri: bool) -> str:
    return (
        Path(os.path.join("workspace", execution_id, file)).as_uri() if use_local_uri else f"http://artifact-ref/{file}"
    )


def format_execution_result(
    result: ExecutionResult,
    indent: int = 0,
    with_code: bool = True,
    code_mask: str | None = None,
    use_local_uri: bool = False,
) -> str:
    lines: list[str] = []

    if with_code:
        if code_mask is not None and len(code_mask) > 0:
            display_code = result.code.replace(code_mask, "")
        else:
            display_code = result.code

        lines.append(
            f"The following python code has been executed:\n"
            "```python\n"
            f"{display_code}\n"
            "```\n"
        )

    lines.append(
        f"The execution of the generated python code above has"
        f" {'succeeded' if result.is_success else 'failed'}.\n"
    )

    if result.output:
        output = result.output
        if isinstance(output, list) and len(output) > 0:
            lines.append(
                "The values of variables of the above Python code after execution are:\n",
            )
            for o in output:
                lines.append(f"{str(o)}")
            lines.append("")
        else:
            lines.append(
                "The result of above Python code after execution is:\n" + str(output),
            )
    elif result.is_success:
        if len(result.stdout) > 0:
            lines.append(
                "The stdout is:",
            )
            lines.append("\n".join(result.stdout)[:TRUNCATE_CHAR_LENGTH])
        else:
            lines.append(
                "The execution is successful but no output is generated.",
            )

    # console output when execution failed
    if not result.is_success:
        lines.append(
            "During execution, the following messages were logged:",
        )
        if len(result.logs) > 0:
            lines.extend([f"- [(l{1})]{ln[0]}: {ln[2]}" for ln in result.logs])
        if result.error is not None:
            lines.append(result.error[:TRUNCATE_CHAR_LENGTH])
        if len(result.stdout) > 0:
            lines.append("\n".join(result.stdout)[:TRUNCATE_CHAR_LENGTH])
        if len(result.stderr) > 0:
            lines.append("\n".join(result.stderr)[:TRUNCATE_CHAR_LENGTH])
        lines.append("")

    # artifacts
    if len(result.artifacts) > 0:
        lines.append("The following artifacts were generated:")
        lines.extend(
            [
                f"- type: {a.type} ; uri: "
                + (
                    get_artifact_uri(
                        execution_id=result.execution_id,
                        file=(
                            a.file_name
                            if os.path.isabs(a.file_name) or not use_local_uri
                            else os.path.join(result.cwd, a.file_name)
                        ),
                        use_local_uri=use_local_uri,
                    )
                )
                + f" ; description: {a.preview}"
                for a in result.artifacts
            ],
        )
        lines.append("")

    return "\n".join([" " * indent + line for line in lines])


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
            message=f"Your code has been executed successfully with the following result:\n{result.output}",
            attachments=last_post.attachments + [
                Attachment.new(
                    type=AttachmentType.CODE_EXECUTION_RESULT,
                    content=format_execution_result(result, with_code=False),
                    extra=result,
                )
            ],
            original_messages=last_post.original_messages,
        )
        self_correction_count = None
    else:
        # Self-correct the code
        post = Post.new(
            send_from="CodeExecutor",
            send_to="CodeGenerator",
            message=result.error,
            attachments=last_post.attachments + [
                Attachment.new(
                    type=AttachmentType.CODE_EXECUTION_RESULT,
                    content=format_execution_result(result, with_code=False),
                    extra=result,
                )
            ],
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
