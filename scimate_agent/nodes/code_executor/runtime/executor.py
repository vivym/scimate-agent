import os
import tempfile
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

from scimate_agent.plugins import Plugin, PluginContext, LogErrorLevel
from .context import RuntimePluginContext


@dataclass
class PluginTestEntry:
    name: str
    description: str
    test: Callable[[Plugin], None]


@dataclass
class PluginRuntime:
    name: str
    code: str
    config: dict[str, Any] | None = None
    loaded: bool = False

    initializer: type[Plugin] | None = None
    test_cases: list[PluginTestEntry] = field(default_factory=list)

    @property
    def module_name(self) -> str:
        return f"scimate_agent.plugins.{self.name}"

    def load(self) -> None:
        if self.loaded:
            return

        def register_plugin(impl):
            if self.initializer is not None:
                raise Exception(
                    f"Duplicate plugin implementation found for `{self.name}`."
                )

            self.initializer = impl

        def register_plugin_test(
            test_name: str,
            test_desc: str,
            test_impl: Callable[[Plugin], None],
        ):
            self.test_cases.append(
                PluginTestEntry(
                    test_name,
                    test_desc,
                    test_impl,
                ),
            )

        try:
            import importlib
            import os
            import sys

            from scimate_agent.plugins import register

            module_name = self.module_name
            with tempfile.TemporaryDirectory() as tmpdir:
                module_path = os.path.join(tmpdir, f"{self.name}.py")
                with open(module_path, "w") as f:
                    f.write(self.code)

                spec = importlib.util.spec_from_file_location(module_name, module_path)
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module

                register.register_plugin_inner = register_plugin
                register.register_plugin_test_inner = register_plugin_test
                spec.loader.exec_module(module)  # type: ignore
                register.register_plugin_inner = None
                register.register_plugin_test_inner = None

                if self.initializer is None:
                    raise Exception(
                        f"Plugin `{self.name}` not registered. Please check the plugin code."
                    )

        except Exception as e:
            traceback.print_exc()
            raise Exception(f"Failed to load plugin `{self.name}`: {e}")

    def unload(self) -> None:
        if not self.loaded:
            return

        # attempt to unload the module, though it is not guaranteed to work
        # there might be some memory leak or other issues there are still some references to
        # certain code inside of the original module
        try:
            self.initializer = None
            import sys

            del sys.modules[self.module_name]
        except Exception:
            pass
        self.loaded = False

    def get_instance(self, ctx: PluginContext) -> Plugin:
        if self.initializer is None:
            raise Exception(f"Plugin `{self.name}` not initialized.")

        try:
            return self.initializer(self.name, ctx, self.config or {})
        except Exception as e:
            raise Exception(f"Failed to create instance of plugin `{self.name}`: {e}")

    def test(self) -> tuple[bool, list[str]]:
        error_messages = []

        from scimate_agent.plugins.context import temp_context

        for test_case in self.test_cases:
            try:
                with temp_context() as ctx:
                    print("=====================================================")
                    print("Test Name:", test_case.name)
                    print("Test Description:", test_case.description)
                    print("Running Test...")
                    instance = self.get_instance(ctx)
                    test_case.test(instance)
                    print("Test Passed")
                    print("=====================================================")
                    print()
            except Exception as e:
                traceback.print_exc()
                error_messages.append(f"Test case `{test_case.name}` in plugin `{self.name}` failed:\n{e}")

        return len(error_messages) == 0, error_messages


class Executor:
    def __init__(self, env_id: str, session_id: str, session_dir: str) -> None:
        self.env_id = env_id
        self.session_id = session_id
        self.session_dir = session_dir

        self.session_vars: dict[str, str] = {}

        self.plugin_registry: dict[str, PluginRuntime] = {}

        self.cur_execution_count: int = 0
        self.cur_execution_id: str = ""

        # if not os.path.exists(self.session_dir):
        #     os.makedirs(self.session_dir, exist_ok=True)

        self.ctx = RuntimePluginContext(self)

    def log(self, level: LogErrorLevel, msg: str) -> None:
        self.ctx.log(level, "Executor", msg)

    def preload_libs(self, local_ns: dict[str, Any]) -> None:
        try:
            pd = __import__("pandas")
            # customize pandas display options
            pd.set_option("display.html.table_schema", False)
            pd.set_option("display.notebook_repr_html", False)
            pd.set_option("display.max_rows", 4)
            pd.set_option("display.expand_frame_repr", False)
            local_ns["pd"] = pd
        except ImportError:
            self.log(
                "warning",
                "Recommended package 'pandas' is not installed. Certain functions may not work as expected.",
            )

        try:
            local_ns["np"] = __import__("numpy")
        except ImportError:
            self.log(
                "warning",
                "Recommended package 'numpy' is not installed. Certain functions may not work as expected.",
            )

        try:
            local_ns["plt"] = __import__("matplotlib.pyplot")
        except ImportError:
            self.log(
                "warning",
                "Recommended package 'matplotlib' is not installed. Certain functions may not work as expected.",
            )

    def update_session_vars(self, session_vars: dict[str, str]) -> None:
        self.session_vars = {str(k): str(v) for k, v in session_vars.items()}

    def pre_execution(self, exec_id: int, exec_idx: int) -> None:
        self.cur_execution_id = exec_id
        self.cur_execution_count = exec_idx

        self.ctx.artifacts = []
        self.ctx.log_messages = []
        self.ctx.outputs = []

    def get_post_execution_state(self) -> dict[str, Any]:
        return {
            "artifacts": self.ctx.artifacts,
            "logs": self.ctx.log_messages,
            "outputs": self.ctx.get_normalized_output(),
        }

    def register_plugin(self, plugin_name: str, plugin_code: str) -> None:
        if plugin_name in self.plugin_registry:
            self.log("warning", f"Plugin `{plugin_name}` already registered.")

        plugin = PluginRuntime(name=plugin_name, code=plugin_code)
        plugin.load()

        self.plugin_registry[plugin_name] = plugin

    def configure_plugin(self, plugin_name: str, config: dict[str, Any]) -> None:
        if plugin_name not in self.plugin_registry:
            self.log("error", f"Plugin `{plugin_name}` not registered.")
            return

        self.plugin_registry[plugin_name].config = config

    def test_plugin(self, plugin_name: str) -> tuple[bool, list[str]]:
        if plugin_name not in self.plugin_registry:
            self.log("error", f"Plugin `{plugin_name}` not registered.")
            return False, ["Plugin not registered."]

        return self.plugin_registry[plugin_name].test()

    def get_plugin_instance(self, plugin_name: str) -> Plugin:
        if plugin_name not in self.plugin_registry:
            self.log("error", f"Plugin `{plugin_name}` not registered.")
            raise Exception(f"Plugin `{plugin_name}` not registered.")

        return self.plugin_registry[plugin_name].get_instance(self.ctx)

    def unload_plugin(self, plugin_name: str) -> None:
        if plugin_name not in self.plugin_registry:
            self.log("error", f"Plugin `{plugin_name}` not registered.")
            return

        self.plugin_registry[plugin_name].unload()
        del self.plugin_registry[plugin_name]
