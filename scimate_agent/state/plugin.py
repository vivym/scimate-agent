import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import structlog
from pydantic import BaseModel, Field

from scimate_agent.utils import read_yaml, write_yaml

if TYPE_CHECKING:
    from structlog.stdlib import AsyncBoundLogger

logger: "AsyncBoundLogger" = structlog.get_logger()


class PluginMetadata(BaseModel):
    name: str
    path: str
    hashsum: str | None = None
    embeddings: dict[str, list[float]] | None = None


class PluginParameter(BaseModel):
    name: str
    type: str
    required: bool = True
    description: str | None = None
    choices: list[Any] | None = None
    default: Any | None = None

    def format_prompt(self, indent: int = 0) -> str:
        lines: list[str] = []

        def line(text: str) -> None:
            lines.append(" " * indent + text)

        line(f"- name: {self.name}")
        line(f"  type: {self.type}")
        line(f"  required: {self.required}")

        if self.description:
            line(f"  description: {self.description}")
        if self.choices:
            line(f"  choices: {', '.join(self.choices)}")
        if self.default:
            line(f"  default: {self.default}")

        return "\n".join(lines)

    def normalize_description(self) -> str:
        return self.description.strip().replace("\n", "\n# ")

    def normalize_type(self) -> str:
        typ = self.type

        if typ.lower() == "string":
            typ = "str"
        elif typ.lower() == "integer":
            typ = "int"
        elif typ.lower() == "boolean":
            typ = "bool"

        if self.choices:
            choice_strs = []
            for choice in self.choices:
                if typ == "str":
                    assert isinstance(choice, str), (
                        "Choice must be a string, "
                        f"but got {type(choice)}."
                    )
                    choice_strs.append(f'"{choice}"')
                elif typ == "int":
                    assert isinstance(choice, int), (
                        "Choice must be an integer, "
                        f"but got {type(choice)}."
                    )
                    choice_strs.append(f"{choice}")
                elif typ == "float":
                    assert isinstance(choice, float), (
                        "Choice must be a float, "
                        f"but got {type(choice)}."
                    )
                    choice_strs.append(f"{choice}")
                elif typ == "bool":
                    assert isinstance(choice, bool), (
                        "Choice must be a boolean, "
                        f"but got {type(choice)}."
                    )
                    choice_strs.append(f"{choice}")
                else:
                    raise ValueError(
                        f"Invalid choice type: {type(choice)}. "
                        f"Expected one of: str, int, float, bool."
                    )
            typ = f"Literal[{', '.join(choice_strs)}]"

        if not self.required and self.default is None:
            typ = f"Optional[{typ}]"
        return typ

    def normalize_default(self) -> str:
        if self.default is None:
            return "None"

        if self.type == "str":
            return f'"{self.default}"'
        else:
            return str(self.default)


class PluginSpec(BaseModel):
    name: str
    description: str
    enabled: bool = True
    examples: list[str] = Field(default_factory=list)
    parameters: list[PluginParameter] = Field(default_factory=list)
    returns: list[PluginParameter] = Field(default_factory=list)
    configurations: dict[str, Any] = Field(default_factory=dict)

    def format_description(self, indent: int = 0) -> str:
        desc = f"{' ' * indent}- {self.name}: {self.description}"

        required_params = [
            f"{p.name}: {p.type}"
            for p in self.parameters
            if p.required
        ]

        if required_params:
            desc += f" Required parameters: {', '.join(required_params)}"

        return desc

    def format_prompt(self) -> str:
        prompt = f"`{self.name}`: {self.description}\n\n"
        prompt += f"```python\ndef {self.name}(\n"

        for param in self.parameters:
            if param.required:
                prompt += f"    # {param.normalize_description()}\n"
            else:
                prompt += f"    # (Optional) {param.normalize_description()}\n"

            if param.required:
                assert param.default is None, "Required parameter cannot have a default value."
                prompt += f"    {param.name}: {param.normalize_type()},\n"
            else:
                prompt += f"    {param.name}: {param.normalize_type()} = {param.normalize_default()},\n"

        if len(self.returns) == 0:
            return_type = "None"
        elif len(self.returns) == 1:
            return_type = self.returns[0].normalize_type()
        else:
            return_type = f"Tuple[{', '.join(r.normalize_type() for r in self.returns)}]"

        prompt += f"): -> {return_type}:\n"
        if len(self.returns) > 0:
            prompt += "    \"\"\"\n"
            prompt += "    Returns:\n"
            for i, r in enumerate(self.returns):
                desc = r.description.strip().replace("\n", "\n            ")
                prefix = f"{i + 1}. " if len(self.returns) > 1 else ""
                prompt += f"        {prefix}{r.name}: {desc}\n"

            if self.examples:
                prompt += "\n"
                prompt += "    Examples:\n"
                example_strs = []
                for example in self.examples:
                    example_strs.append(f"        {example}")
                prompt += "\n\n".join(example_strs)
                prompt += "\n"

            prompt += "    \"\"\"\n"

        prompt += "    ...\n```"

        return prompt


class PluginEntry(BaseModel):
    name: str
    spec: PluginSpec
    metadata: PluginMetadata

    @property
    def enabled(self) -> bool:
        return self.spec.enabled

    @property
    def hashsum(self) -> str:
        return self.metadata.hashsum

    @classmethod
    def from_local_path(cls, path: str | Path) -> Optional["PluginEntry"]:
        path = Path(path)

        if not path.exists() or not path.is_dir():
            return None

        spec_path = path / "spec.yaml"

        if not spec_path.exists():
            return None

        return cls.from_local_spec_path(str(spec_path))

    @classmethod
    def from_local_spec_path(cls, spec_path: str | Path) -> Optional["PluginEntry"]:
        spec_path = Path(spec_path)
        plugin_path = spec_path.parent

        obj = read_yaml(spec_path)

        meta_path = plugin_path / ".metadata.yaml"

        if meta_path.exists():
            metadata = PluginMetadata(**read_yaml(meta_path))
            metadata_write_back = False
        else:
            metadata = PluginMetadata(
                name=obj["name"],
                path=str(plugin_path.resolve()),
            )
            metadata_write_back = True

        return cls.from_spec_dict(obj, metadata=metadata, metadata_write_back=metadata_write_back)

    @classmethod
    def from_spec_dict(
        cls,
        obj: dict[str, Any],
        metadata: PluginMetadata,
        metadata_write_back: bool = False,
    ) -> "PluginEntry":
        spec = PluginSpec(**obj)

        plugin = cls(
            name=spec.name,
            spec=spec,
            metadata=metadata,
        )

        if plugin.metadata.hashsum is None:
            plugin_package_bytes = plugin.load_plugin_package()
            config_bytes = json.dumps(spec.configurations, sort_keys=True).encode("utf-8")
            hashsum = hashlib.sha256(plugin_package_bytes + config_bytes).hexdigest()
            plugin.metadata.hashsum = hashsum
            metadata_write_back = True

        if metadata_write_back:
            meta_path = Path(plugin.metadata.path) / ".metadata.yaml"
            write_yaml(meta_path, plugin.metadata.model_dump(mode="json"))

        return plugin

    def format_description(self, indent: int = 0) -> str:
        return self.spec.format_description(indent)

    def format_prompt(self) -> str:
        return self.spec.format_prompt()

    def load_plugin_package(self) -> bytes:
        plugin_path = Path(self.metadata.path)

        def filter_fn(tarinfo: tarfile.TarInfo) -> tarfile.TarInfo | None:
            if tarinfo.name in [".metadata.yaml", "spec.yaml"]:
                return None
            return tarinfo

        # Package the plugin into a tar.gz
        with io.BytesIO() as tar_buffer:
            with tarfile.open(fileobj=tar_buffer, mode="w:gz") as tar:
                tar.add(plugin_path, arcname="plugin", filter=filter_fn)
            return tar_buffer.getvalue()


def load_plugins(search_paths: list[str | Path]) -> list[PluginEntry]:
    plugins: list[PluginEntry] = []
    for path in search_paths:
        path = Path(path)
        if not path.exists():
            continue

        for entry in path.iterdir():
            if entry.is_dir() and (entry / "spec.yaml").exists():
                try:
                    plugin = PluginEntry.from_local_path(entry)
                    if plugin:
                        plugins.append(plugin)
                except Exception as e:
                    logger.error("Error loading plugin", path=entry, error=e)

    return plugins
