import base64
import json
import logging
import os
import site
import sys
from ast import literal_eval
from typing import Any, Callable, Literal, Union

from jupyter_client.blocking.client import BlockingKernelClient
from jupyter_client.kernelspec import KernelSpec, KernelSpecManager
from jupyter_client.manager import KernelManager
from jupyter_client.multikernelmanager import MultiKernelManager

from ..utils import get_id, time_usage
from .common import (
    DisplayData,
    ExecType,
    ExecutionArtifact,
    ExecutionResult,
    ExecutionResultInternal,
    Plugin,
    Session,
    ResultMimeType,
)

logger = logging.getLogger(__name__)


class KernelSpecProvider(KernelSpecManager):
    def get_kernel_spec(self, kernel_name: str) -> KernelSpec:
        if kernel_name == "scimate":
            return KernelSpec(
                argv=[
                    "python",
                    "-m",
                    "scimate_agent.nodes.code_executor.kernel.launcher",
                    "-f",
                    "{connection_file}",
                ],
                display_name="SciMate",
                language="python",
                metadata={"debugger": True},
            )
        return super().get_kernel_spec(kernel_name)


class SciMateMultiKernelManager(MultiKernelManager):
    def pre_start_kernel(
        self,
        kernel_name: str | None,
        kwargs: Any,
    ) -> tuple[KernelManager, str, str]:
        env: dict[str, str] | None = kwargs.get("env", None)

        km, kernel_name, kernel_id = super().pre_start_kernel(kernel_name, kwargs)
        if env is not None:
            if "CONNECTION_FILE" in env:
                km.connection_file = env["CONNECTION_FILE"]
        return km, kernel_name, kernel_id


class Environment:
    def __init__(self, env_id: str, env_dir: str) -> None:
        self.env_id = env_id
        self.env_dir = env_dir

        self.session_dict: dict[str, Session] = {}

        self.kernel_manager = SciMateMultiKernelManager(
            default_kernel_name="scimate",
            kernel_spec_manager=KernelSpecProvider(),
        )

        logger.info(f"Environment initialized with id: {self.env_id}")

    def get_default_session_dir(self, session_id: str) -> str:
        session_dir = os.path.join(self.env_dir, "sessions", session_id)
        os.makedirs(session_dir, exist_ok=True)
        return session_dir

    def start_session(
        self,
        session_id: str,
        session_dir: str | None = None,
        cwd: str | None = None,
    ) -> None:
        session = self._get_session(session_id, session_dir, cwd)
        session_dir = os.path.realpath(session.session_dir)
        cwd = os.path.realpath(session.cwd)

        kernel_session_dir = os.path.join(session_dir, "kernel")
        os.makedirs(kernel_session_dir, exist_ok=True)

        new_kernel_id = get_id(prefix="knl")

        python_paths = os.pathsep.join(
            [
                os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
            ]
            + site.getsitepackages()
            + sys.path
        )

        # inherit current environment variables
        # TODO: filter out sensitive environment information
        kernel_env = os.environ.copy()
        kernel_env.update(
            {
                "SCIMATE_ENV_ID": self.env_id,
                "SCIMATE_SESSION_ID": session.session_id,
                "SCIMATE_SESSION_DIR": session_dir,
                "SCIMATE_LOGGING_FILE_PATH": os.path.join(kernel_session_dir, "logging.log"),
                "CONNECTION_FILE": self._get_connection_file(session.session_id, new_kernel_id),
                "PYTHONPATH": python_paths,
            }
        )

        session.kernel_id = self.kernel_manager.start_kernel(
            kernel_id=new_kernel_id,
            cwd=cwd,
            env=kernel_env,
        )

        self._cmd_session_init(session)

        session.kernel_status = "ready"

    def stop_session(self, session_id: str) -> None:
        self._cleanup_client(session_id)

        session = self._get_session(session_id)
        if session is None:
            return
        if session.kernel_status == "stopped":
            return
        if session.kernel_status == "pending":
            session.kernel_status = "stopped"
            return

        try:
            if session.kernel_id is not None:
                kernel = self.kernel_manager.get_kernel(session.kernel_id)
                is_alive = kernel.is_alive()
                if is_alive:
                    kernel.shutdown_kernel(now=True)
                kernel.cleanup_resources()
        except Exception as e:
            logger.error(f"Failed to stop kernel {session.kernel_id}: {e}")

        session.kernel_status = "stopped"

    def update_session_vars(self, session_id: str, vars: dict[str, str]) -> None:
        session = self._get_session(session_id)
        if session is None:
            return
        # Defer applying session vars until code execution
        session.session_vars.update(vars)

    def execute_code(
        self,
        session_id: str,
        code: str,
        exec_id: str | None = None,
    ) -> ExecutionResult:
        exec_id = exec_id or get_id(prefix="exec")
        session = self._get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        session.execution_count += 1
        exec_index = session.execution_count
        self._execute_control_code_on_kernel(
            session_id=session_id,
            code=f"%_scimate_exec_pre_check {exec_id} {exec_index}",
        )
        # update session vars before execution
        if session.session_vars is not None:
            self._cmd_update_session_vars(session)

        # execute code on kernel
        exec_result = self._execute_code_on_kernel(
            session_id=session_id,
            exec_id=exec_id,
            code=code,
        )
        session.execution_dict[exec_id] = exec_result
        exec_extra_result = self._execute_control_code_on_kernel(
            session_id=session_id,
            code=f"%_scimate_exec_post_check {exec_id} {exec_index}",
        )

        return self._parse_exec_result(exec_result, exec_extra_result["data"], session.cwd)

    def load_plugin(
        self,
        session_id: str,
        plugin_name: str,
        plugin_loader: Callable[[], bytes],
        plugin_config: dict[str, str] | None = None,
    ) -> None:
        session = self._get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        # TODO: Use hashsum to check if the plugin is already loaded
        if plugin_name in session.plugins:
            prev_plugin = session.plugins[plugin_name]
            if prev_plugin.loaded:
                self._cmd_unload_plugin(session, prev_plugin)
            del session.plugins[plugin_name]

        package_bytes = plugin_loader()
        package_str = base64.b64encode(package_bytes).decode("utf-8")

        plugin = Plugin(
            name=plugin_name,
            package=package_str,
            config=plugin_config,
        )
        self._cmd_load_plugin(session, plugin)
        plugin.loaded = True
        session.plugins[plugin_name] = plugin

    def test_plugin(self, session_id: str, plugin_name: str) -> None:
        session = self._get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        if plugin_name not in session.plugins:
            logger.warning(f"Tried to test plugin `{plugin_name}` in session `{session_id}` that is not loaded.")
            return
        plugin = session.plugins[plugin_name]
        self._cmd_test_plugin(session, plugin)

    def unload_plugin(self, session_id: str, plugin_name: str) -> None:
        session = self._get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        if plugin_name in session.plugins:
            plugin = session.plugins[plugin_name]
            if plugin.loaded:
                self._cmd_unload_plugin(session, plugin)
            del session.plugins[plugin_name]
        else:
            logger.warning(f"Tried to unload plugin `{plugin_name}` that is not loaded.")

    def download_file(self, session_id: str, url: str, file_path: str) -> str:
        session = self._get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        result = self._execute_control_code_on_kernel(
            session_id=session_id,
            code=f"%%_scimate_convert_path\n{file_path}",
            silent=True,
        )
        return result["data"]

    def _get_session(
        self,
        session_id: str,
        session_dir: str | None = None,
        cwd: str | None = None,
    ) -> Session | None:
        if session_id not in self.session_dict and session_dir is not None:
            session_dir = session_dir or self.get_default_session_dir(session_id)
            session_dir = os.path.abspath(session_dir)
            cwd = cwd or os.path.join(session_dir, "cwd")
            cwd = os.path.abspath(cwd)
            new_session = Session(
                session_id=session_id,
                session_dir=session_dir,
                cwd=cwd,
            )
            os.makedirs(new_session.session_dir, exist_ok=True)
            os.makedirs(new_session.cwd, exist_ok=True)
            self.session_dict[session_id] = new_session

        return self.session_dict.get(session_id, None)

    def _get_connection_file(self, session_id: str, kernel_id: str) -> str:
        return os.path.join(
            self._get_session(session_id).session_dir,
            "kernel",
            f"conn-{session_id}-{kernel_id}.json",
        )

    def _get_client(self, session_id: str) -> BlockingKernelClient:
        session = self._get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")


        if session.client is None:
            connection_file = self._get_connection_file(session_id, session.kernel_id)
            client = BlockingKernelClient(connection_file=connection_file)
            client.load_connection_file()

            client.wait_for_ready(timeout=30)
            client.start_channels()

            session.client = client

        return session.client

    def _cleanup_client(self, session_id: str) -> None:
        session = self._get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")

        if session.client is not None:
            session.client.stop_channels()
            session.client = None

    def _execute_code_on_kernel(
        self,
        session_id: str,
        exec_id: str,
        code: str,
        silent: bool = False,
        store_history: bool = True,
        exec_type: ExecType = "user",
    ) -> ExecutionResultInternal:
        exec_result = ExecutionResultInternal(exec_id=exec_id, code=code, exec_type=exec_type)

        client = self._get_client(session_id)
        result_msg_id = client.execute(
            code=code,
            silent=silent,
            store_history=store_history,
            allow_stdin=False,
            stop_on_error=True,
        )

        try:
            # TODO: interrupt kernel if it takes too long
            while True:
                with time_usage() as time_msg:
                    message = client.get_iopub_msg(timeout=180)
                logger.debug(f"Time: {time_msg.total:.2f} \t MsgType: {message['msg_type']} \t Code: {code}")
                logger.debug(json.dumps(message, indent=2, default=str, ensure_ascii=False))

                if message["parent_header"]["msg_id"] != result_msg_id:
                    # skip messages not related to the current execution
                    continue
                msg_type = message["msg_type"]
                if msg_type == "status":
                    if message["content"]["execution_state"] == "idle":
                        break
                elif msg_type == "stream":
                    stream_name = message["content"]["name"]
                    stream_text = message["content"]["text"]
                    if stream_name == "stdout":
                        exec_result.stdout.append(stream_text)
                    elif stream_name == "stderr":
                        exec_result.stderr.append(stream_text)
                    else:
                        logger.warning(f"Unknown stream name: {stream_name}")
                elif msg_type == "execute_result":
                    exec_result.result = message["content"]["data"]
                elif msg_type == "error":
                    error_traceback_lines = message["content"]["traceback"]
                    if error_traceback_lines is None:
                        error_name = message["content"]["ename"]
                        error_value = message["content"]["evalue"]
                        error_traceback_lines = [f"{error_name}: {error_value}"]
                    error_traceback = "\n".join(error_traceback_lines)
                    exec_result.error = error_traceback
                elif msg_type == "display_data":
                    data: dict[ResultMimeType, Any] = message["content"]["data"]
                    metadata: dict[str, Any] = message["content"]["metadata"]
                    transient: dict[str, Any] = message["content"]["transient"]
                    exec_result.displays.append(
                        DisplayData(
                            data=data,
                            metadata=metadata,
                            transient=transient,
                        )
                    )
                elif msg_type == "update_display_data":
                    data: dict[ResultMimeType, Any] = message["content"]["data"]
                    metadata: dict[str, Any] = message["content"]["metadata"]
                    transient: dict[str, Any] = message["content"]["transient"]
                    exec_result.displays.append(
                        DisplayData(
                            data=data,
                            metadata=metadata,
                            transient=transient,
                        )
                    )
                else:
                    logger.debug(f"Unhandled message type: {msg_type}")
        finally:
            ...

        return exec_result

    def _execute_control_code_on_kernel(
        self,
        session_id: str,
        code: str,
        silent: bool = False,
        store_history: bool = False,
    ) -> dict[Literal["is_success", "message", "data"], Union[bool, str, Any]]:
        exec_result = self._execute_code_on_kernel(
            session_id=session_id,
            exec_id=get_id(prefix="ctl"),
            code=code,
            silent=silent,
            store_history=store_history,
            exec_type="control",
        )
        if exec_result.error is not None:
            raise Exception(exec_result.error)
        if exec_result.result is None or "text/plain" not in exec_result.result:
            raise Exception("No text output returned from control code.")
        result = literal_eval(exec_result.result["text/plain"])
        if not result["is_success"]:
            raise Exception(result["message"])
        return result

    def _cmd_session_init(self, session: Session) -> None:
        self._execute_control_code_on_kernel(
            session_id=session.session_id,
            code=f"%_scimate_session_init {session.session_id}",
        )

    def _cmd_update_session_vars(self, session: Session) -> None:
        self._execute_control_code_on_kernel(
            session_id=session.session_id,
            code=f"%%_scimate_update_session_vars\n{json.dumps(session.session_vars)}",
        )

    def _cmd_load_plugin(self, session: Session, plugin: Plugin) -> None:
        self._execute_control_code_on_kernel(
            session_id=session.session_id,
            code=f"%%_scimate_register_plugin {plugin.name}\n{plugin.package}",
        )
        self._execute_control_code_on_kernel(
            session_id=session.session_id,
            code=f"%%_scimate_configure_plugin {plugin.name}\n{json.dumps(plugin.config or {})}",
        )

    def _cmd_test_plugin(self, session: Session, plugin: Plugin) -> None:
        self._execute_control_code_on_kernel(
            session_id=session.session_id,
            code=f"%_scimate_test_plugin {plugin.name}",
        )

    def _cmd_unload_plugin(self, session: Session, plugin: Plugin) -> None:
        self._execute_control_code_on_kernel(
            session_id=session.session_id,
            code=f"%_scimate_unload_plugin {plugin.name}",
        )

    def _parse_exec_result(
        self,
        exec_result: ExecutionResultInternal,
        extra_result: dict[str, Any] | None = None,
        cwd: str | None = None,
    ) -> ExecutionResult:
        result = ExecutionResult(
            exec_id=exec_result.exec_id,
            code=exec_result.code,
            cwd=cwd,
            is_success=exec_result.error is None,
            error=exec_result.error,
            output=None,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            logs=[],
            artifacts=[],
        )

        for mime_type in exec_result.result.keys():
            if mime_type.startswith("text/"):
                text_result = exec_result.result[mime_type]
                try:
                    parsed_result = literal_eval(text_result)
                    result.output = parsed_result
                except Exception:
                    result.output = text_result

        display_artifact_count = 0
        for display in exec_result.displays:
            display_artifact_count += 1
            artifact = ExecutionArtifact(
                name=f"{exec_result.exec_id}-display-{display_artifact_count}"
            )
            has_svg = False
            has_pic = False
            for mime_type in display.data.keys():
                if mime_type.startswith("image/"):
                    if mime_type == "image/svg+xml":
                        if has_pic and has_svg:
                            continue
                        has_svg = True
                        has_pic = True
                        artifact.type = "svg"
                        artifact.file_content_encoding = "str"
                    else:
                        if has_pic:
                            continue
                        has_pic = True
                        artifact.type = "image"
                        artifact.file_content_encoding = "base64"

                    artifact.mime_type = mime_type
                    artifact.file_content = display.data[mime_type]
                elif mime_type.startswith("text/"):
                    artifact.preview = display.data[mime_type]

            if has_pic:
                result.artifacts.append(artifact)

        if isinstance(extra_result, dict):
            for key, value in extra_result.items():
                if key == "logs":
                    result.logs = value
                elif key == "artifacts":
                    for artifact_dict in value:
                        artifact = ExecutionArtifact(
                            name=artifact_dict["name"],
                            type=artifact_dict["type"],
                            original_name=artifact_dict["original_name"],
                            file_name=artifact_dict["file_name"],
                            preview=artifact_dict["preview"],
                        )
                        result.artifacts.append(artifact)

        return result
