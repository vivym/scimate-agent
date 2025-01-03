import json
from typing import Any

from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import (
    Magics,
    line_cell_magic,
    line_magic,
    magics_class,
    needs_local_scope,
)

from scimate_agent.nodes.code_executor.runtime import Executor
from .ctx_magic import fmt_response


@magics_class
class SciMatePluginMagic(Magics):
    def __init__(self, shell: InteractiveShell, executor: Executor, **kwargs: Any):
        super().__init__(shell, **kwargs)
        self.executor = executor

    @line_cell_magic
    def _scimate_register_plugin(self, line: str, cell: str) -> dict[str, Any]:
        plugin_name = line.strip()
        plugin_package = cell
        try:
            self.executor.register_plugin(plugin_name, plugin_package)
            return fmt_response(
                True,
                f"Plugin `{plugin_name}` registered successfully.",
            )
        except Exception as e:
            return fmt_response(False, f"Failed to register plugin `{plugin_name}`: {e}")

    @line_magic
    def _scimate_test_plugin(self, line: str):
        plugin_name = line.strip()
        is_success, messages = self.executor.test(plugin_name)
        if is_success:
            return fmt_response(
                True,
                f"Plugin `{plugin_name}` passed all tests:\n" + "\n".join(messages),
            )
        else:
            return fmt_response(
                False,
                f"Plugin `{plugin_name}` failed some tests:\n" + "\n".join(messages),
            )
    @needs_local_scope
    @line_cell_magic
    def _scimate_configure_plugin(self, line: str, cell: str, local_ns: dict[str, Any]):
        plugin_name = line.strip()
        plugin_config = json.loads(cell)
        try:
            self.executor.configure_plugin(plugin_name, plugin_config)
            local_ns[plugin_name] = self.executor.get_plugin_instance(plugin_name)
            return fmt_response(
                True,
                f"Plugin `{plugin_name}` configured successfully.",
            )
        except Exception as e:
            return fmt_response(False, f"Failed to configure plugin `{plugin_name}`: {e}")

    @needs_local_scope
    @line_magic
    def _scimate_unload_plugin(self, line: str, local_ns: dict[str, Any]):
        plugin_name = line.strip()
        if plugin_name not in local_ns:
            return fmt_response(
                True,
                f"Plugin `{plugin_name}` is not loaded. Skipping...",
            )

        del local_ns[plugin_name]
        self.executor.unload_plugin(plugin_name)
        return fmt_response(True, f"Plugin `{plugin_name}` unloaded successfully.")
