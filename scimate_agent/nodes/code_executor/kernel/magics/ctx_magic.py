import json
from typing import Any

from IPython.core.interactiveshell import InteractiveShell
from IPython.core.magic import (
    Magics,
    cell_magic,
    line_cell_magic,
    line_magic,
    magics_class,
    needs_local_scope,
)

from scimate_agent.nodes.code_executor.runtime import Executor
from .utils import fmt_response


@magics_class
class SciMateContextMagic(Magics):
    def __init__(self, shell: InteractiveShell, executor: Executor, **kwargs: Any):
        super().__init__(shell, **kwargs)
        self.executor = executor

    @needs_local_scope
    @line_magic
    def _scimate_session_init(self, line: str, local_ns: dict[str, Any]) -> dict[str, Any]:
        self.executor.preload_libs(local_ns)
        return fmt_response(True, "SciMate context initialized.")

    @cell_magic
    def _scimate_update_session_vars(self, line: str, cell: str) -> dict[str, Any]:
        session_vars = json.loads(cell)
        self.executor.update_session_vars(session_vars)
        return fmt_response(True, "SciMate session vars updated.", data=self.executor.session_vars)

    @line_magic
    def _scimate_check_session_vars(self, line: str) -> dict[str, Any]:
        return fmt_response(True, "SciMate session vars printed.", data=self.executor.session_vars)

    @line_magic
    def _scimate_exec_pre_check(self, line: str) -> dict[str, Any]:
        exec_id, exec_idx = line.split(" ")
        exec_idx = int(exec_idx)
        self.executor.pre_execution(exec_id, exec_idx)
        return fmt_response(True, "SciMate exec pre-check executed.")

    @needs_local_scope
    @line_magic
    def _scimate_exec_post_check(self, line: str, local_ns: dict[str, Any]) -> dict[str, Any]:
        if "_" in local_ns:
            self.executor.ctx.add_outputs(local_ns["_"])
        data = self.executor.get_post_execution_state()
        return fmt_response(True, "SciMate exec post-check executed.", data=data)

    @cell_magic
    def _scimate_convert_path(self, line: str, cell: str) -> dict[str, Any]:
        import os

        full_path = os.path.abspath(os.path.expanduser(cell))
        return fmt_response(True, "SciMate path converted.", data=full_path)

    @line_cell_magic
    def _scimate_write_and_run(self, line: str, cell: str) -> dict[str, Any]:
        file_path = line.strip()
        if not file_path:
            return fmt_response(False, "File path is not provided.")

        with open(file_path, "w") as f:
            f.write(cell)

        self.shell.run_cell(cell)
