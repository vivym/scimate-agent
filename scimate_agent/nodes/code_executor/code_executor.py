import asyncio
import base64
import os
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from scimate_agent.event import EventEmitter
from scimate_agent.plugins import ArtifactType
from scimate_agent.state import Attachment, AttachmentType, CodeInterpreterState, Post, RoundUpdate
from .session import ExecutionResult, SessionManager, SessionClient
from .utils import get_id

TRUNCATE_CHAR_LENGTH = 1000

SESSION_CLIENT_CACHE = {}


async def get_session_client(
    env_id: str | None,
    env_dir: str | None,
    session_id: str | None,
    create_if_not_exists: bool = True,
) -> SessionClient:
    cache_key = (env_id, env_dir, session_id)

    if cache_key not in SESSION_CLIENT_CACHE:
        if not create_if_not_exists:
            raise ValueError("Session client not found.")

        session_mgr = SessionManager(env_id=env_id, env_dir=env_dir)
        if session_id is None:
            session_id = get_id(prefix="sess")

        session_client = session_mgr.get_session_client(session_id=session_id)
        await session_client.start()

        cache_key = (session_mgr.env_id, session_mgr.env_dir, session_id)

        SESSION_CLIENT_CACHE[cache_key] = session_client


    return SESSION_CLIENT_CACHE[cache_key]


def get_artifact_uri(file_path: str, use_local_uri: bool) -> str:
    if use_local_uri:
        assert os.path.isabs(file_path)

    return (
        Path(file_path).as_uri() if use_local_uri else f"http://artifact-ref/{file_path}"
    )


def get_default_artifact_name(artifact_type: ArtifactType, mime_type: str) -> str:
    if artifact_type == "file":
        return "artifact"
    if artifact_type == "image":
        if mime_type == "image/png":
            return "image.png"
        if mime_type == "image/jpeg":
            return "image.jpg"
        if mime_type == "image/gif":
            return "image.gif"
        if mime_type == "image/svg+xml":
            return "image.svg"
    if artifact_type == "chart":
        return "chart.json"
    if artifact_type == "svg":
        return "svg.svg"
    return "file"


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
        lines.append("")

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
                        file_path=(
                            a.file_name
                            if os.path.isabs(a.file_name) or not use_local_uri
                            else os.path.join(result.cwd, "artifacts", a.file_name)
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


async def code_executor_node(state: CodeInterpreterState, config: RunnableConfig) -> dict[str, Any]:
    rounds = state.get_rounds()
    assert len(rounds) > 0, "No round found for CodeExecutor."

    last_round = rounds[-1]
    assert len(last_round.posts) > 0, "No post found for CodeExecutor."
    last_post = last_round.posts[-1]

    assert last_post.send_to == "CodeExecutor", "The latest post is not sent to CodeExecutor."
    assert last_post.send_from == "CodeVerifier", "The latest post is not from CodeVerifier."

    code = last_post.message

    if state.env_id is None:
        # Initialize session manager and session client from `config`
        env_id = config["configurable"].get("env_id", None)
        env_dir = config["configurable"].get("env_dir", None)
        session_id = config["configurable"].get("session_id", None)
        if session_id is None:
            raise ValueError("Session ID is required.")
        session_id = str(session_id)
    else:
        env_id = state.env_id
        env_dir = state.env_dir
        session_id = state.session_id

        assert env_id is not None, "Environment ID is required."
        assert env_dir is not None, "Environment directory is required."
        assert session_id is not None, "Session ID is required."

    session_client = await get_session_client(
        env_id=env_id,
        env_dir=env_dir,
        session_id=session_id,
    )

    event_handle = config["configurable"].get("event_handle", None)
    event_emitter = EventEmitter.get_instance(event_handle)

    for plugin in state.plugins:
        if plugin.enabled:
            session_client.load_plugin(
                plugin_name=plugin.name,
                plugin_loader=plugin.load_plugin_package,
                plugin_config=plugin.spec.configurations,
                plugin_hashsum=plugin.hashsum,
            )

    await event_emitter.emit(
        "code_executor_start",
        {
            "env_id": env_id,
            "env_dir": env_dir,
            "session_id": session_id,
            "exec_id": f"{session_id}-{last_round.id}",
            "code": code,
        },
    )

    result = await session_client.execute_code(
        exec_id=f"{session_id}-{last_round.id}",
        code=code,
    )

    for artifact in result.artifacts:
        if artifact.file_name is None:
            original_name = (
                artifact.original_name
                if artifact.original_name is not None
                else get_default_artifact_name(
                    artifact.type,
                    artifact.mime_type,
                )
            )
            file_name = f"{artifact.name}_{original_name}"
            file_path = os.path.join(result.cwd, "artifacts", file_name)
            os.makedirs(os.path.dirname(file_path), exist_ok=True)

            if artifact.file_content_encoding == "base64":
                def sync_write():
                    with open(file_path, "wb") as f:
                        f.write(base64.b64decode(artifact.file_content))

                await asyncio.to_thread(sync_write)
            else:
                def sync_write():
                    with open(file_path, "w") as f:
                        f.write(artifact.file_content)

                await asyncio.to_thread(sync_write)

            artifact.file_name = file_name

    await event_emitter.emit("code_executor_result", result.model_dump(mode="json"))

    self_correction_count = state.self_correction_count

    if result.is_success:
        post = Post.new(
            send_from="CodeExecutor",
            send_to="Planner",
            message=format_execution_result(result, with_code=True, use_local_uri=True),
            attachments=last_post.attachments + [
                Attachment.new(
                    type=AttachmentType.CODE_EXECUTION_RESULT,
                    content=format_execution_result(result, with_code=False, use_local_uri=True),
                    extra=result,
                )
            ],
            original_messages=last_post.original_messages,
        )
        # Reset self-correction count
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
                    content=format_execution_result(result, with_code=False, use_local_uri=True),
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
        "env_id": env_id,
        "env_dir": env_dir,
        "session_id": session_id,
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
