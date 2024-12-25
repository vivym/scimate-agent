import os
from typing import TYPE_CHECKING, Any

from scimate_agent.plugins import PluginContext, ArtifactType, LogErrorLevel

if TYPE_CHECKING:
    from .executor import Executor


class RuntimePluginContext(PluginContext):
    def __init__(self, executor: "Executor"):
        self.executor = executor

        self.artifacts: list[dict[str, str]] = []
        self.log_messages: list[tuple[LogErrorLevel, str, str]] = []
        self.outputs: list[tuple[str, str]] = []

    @property
    def execution_id(self) -> str:
        return self.executor.cur_execution_id

    @property
    def session_id(self) -> str:
        return self.executor.session_id

    @property
    def env_id(self) -> str:
        return self.executor.env_id

    @property
    def execution_idx(self) -> int:
        return self.executor.cur_execution_count

    def add_artifact(
        self,
        name: str,
        file_name: str,
        type: ArtifactType,
        val: Any,
        desc: str | None = None,
    ) -> str:
        desc_preview = desc if desc is not None else self._get_preview_by_type(type, val)

        artifact_id, artifact_path = self.create_artifact_path(name, file_name, type, desc=desc_preview)
        if type in ["chart", "file", "txt", "svg", "html"]:
            with open(artifact_path, "w") as f:
                f.write(val)
        elif type == "df":
            val.to_csv(artifact_path, index=False)
        else:
            raise Exception(f"Unsupported artifact type: {type}")

        return artifact_id

    def _get_preview_by_type(self, type: str, val: Any) -> str:
        if type == "chart":
            preview = "chart"
        elif type == "df":
            preview = f"DataFrame in shape {val.shape} with columns {list(val.columns)}"
        elif type == "file" or type == "txt":
            preview = str(val)[:100]
        elif type == "html":
            preview = "Web Page"
        else:
            preview = str(val)
        return preview

    def create_artifact_path(
        self,
        name: str,
        file_name: str,
        type: ArtifactType,
        desc: str,
    ) -> tuple[str, str]:
        id = f"obj_{self.execution_idx}_{type}_{len(self.artifacts):04x}"

        file_name = f"{id}_{file_name}"
        full_path = self._get_obj_path(file_name)

        self.artifacts.append(
            {
                "name": name,
                "type": type,
                "original_name": file_name,
                "file_name": file_name,
                "preview": desc,
            }
        )
        return id, full_path

    def _get_obj_path(self, file_name: str) -> str:
        return os.path.join(self.executor.session_dir, "cwd", file_name)

    def get_session_var(
        self,
        variable_name: str,
        default: str | None,
    ) -> str | None:
        if variable_name in self.executor.session_vars:
            return self.executor.session_vars[variable_name]
        return default

    def log(self, level: LogErrorLevel, tag: str, msg: str) -> None:
        self.log_messages.append((level, tag, msg))

    def get_env(self, plugin_name: str, variable_name: str) -> str:
        name = f"PLUGIN_{plugin_name}_{variable_name}"
        if name in os.environ:
            return os.environ[name]
        raise Exception(f"Environment variable `{name}` not found.")

    def get_normalized_output(self) -> list[tuple[str, str]]:
        def to_str(v: Any) -> str:
            # TODO: configure/tune value length limit
            # TODO: handle known/common data types explicitly
            return str(v)[:5000]

        def normalize_tuple(i: int, v: Any) -> tuple[str, str]:
            default_name = f"execution_result_{i + 1}"
            if isinstance(v, tuple) or isinstance(v, list):
                list_value: Any = v
                name = to_str(list_value[0]) if len(list_value) > 0 else default_name
                if len(list_value) <= 2:
                    val = to_str(list_value[1]) if len(list_value) > 1 else to_str(None)
                else:
                    val = to_str(list_value[1:])
                return (name, val)
            else:
                return (default_name, to_str(v))

        return [normalize_tuple(i, v) for i, v in enumerate(self.outputs)]

    def add_outputs(self, outputs: list[tuple[str, str]]) -> None:
        if isinstance(outputs, list):
            self.outputs.extend(outputs)
        else:
            self.outputs.append((str(outputs), ""))
