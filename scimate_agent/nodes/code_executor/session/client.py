from .common import ExecutionResult
from .environment import Environment


class SessionClient:
    def __init__(
        self,
        env: Environment,
        session_id: str,
        session_dir: str,
        cwd: str,
    ) -> None:
        self.env = env
        self.session_id = session_id
        self.session_dir = session_dir
        self.cwd = cwd

    def start(self) -> None:
        self.env.start_session(self.session_id, self.session_dir, self.cwd)

    def stop(self) -> None:
        self.env.stop_session(self.session_id)

    def load_plugin(
        self,
        plugin_name: str,
        plugin_code: str,
        plugin_config: dict[str, str] | None = None,
    ) -> None:
        self.env.load_plugin(
            session_id=self.session_id,
            plugin_name=plugin_name,
            plugin_code=plugin_code,
            plugin_config=plugin_config,
        )

    def test_plugin(self, plugin_name: str) -> None:
        self.env.test_plugin(self.session_id, plugin_name)

    def unload_plugin(self, plugin_name: str) -> None:
        self.env.unload_plugin(self.session_id, plugin_name)

    def update_session_vars(self, vars: dict[str, str]) -> None:
        self.env.update_session_vars(self.session_id, vars)

    def execute_code(self, exec_id: str, code: str) -> ExecutionResult:
        return self.env.execute_code(self.session_id, code=code, exec_id=exec_id)
