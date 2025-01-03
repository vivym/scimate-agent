from dataclasses import dataclass, field
from typing import Any, Literal, Union

from jupyter_client.blocking.client import BlockingKernelClient

from scimate_agent.plugins import ArtifactType

ExecType = Literal["user", "control"]
ResultMimeType = Union[
    Literal["text/plain", "text/html", "text/markdown", "text/latex"],
    str,
]


@dataclass
class Plugin:
    name: str
    package: str
    config: dict[str, str] | None
    loaded: bool = False


@dataclass
class DisplayData:
    data: dict[ResultMimeType, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    transient: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResultInternal:
    exec_id: str
    code: str
    exec_type: ExecType = "user"

    # streaming outputs
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)
    displays: list[DisplayData] = field(default_factory=list)

    # final outputs
    result: dict[ResultMimeType, str] = field(default_factory=dict)
    error: str | None = None


@dataclass
class ExecutionArtifact:
    name: str | None = None
    type: ArtifactType = "file"
    mime_type: str | None = None
    original_name: str | None = None
    file_name: str | None = None
    file_content: str | None = None
    file_content_encoding: Literal["str", "base64"] = "str"
    preview: str | None = None


@dataclass
class ExecutionResult:
    exec_id: str
    code: str

    cwd: str | None = None

    is_success: bool = False
    error: str | None = None

    output: Union[str, list[tuple[str, str]]] | None = None
    stdout: list[str] = field(default_factory=list)
    stderr: list[str] = field(default_factory=list)

    logs: list[tuple[str, str, str]] = field(default_factory=list)
    artifacts: list[ExecutionArtifact] = field(default_factory=list)


@dataclass
class Session:
    session_id: str
    session_dir: str

    cwd: str

    session_vars: dict[str, str] = field(default_factory=dict)

    kernel_id: str | None = None
    kernel_status: Literal[
        "pending",
        "ready",
        "running",
        "stopped",
        "error",
    ] = "pending"

    plugins: dict[str, Plugin] = field(default_factory=dict)

    execution_count: int = 0
    execution_dict: dict[str, ExecutionResultInternal] = field(default_factory=dict)

    client: BlockingKernelClient | None = None
