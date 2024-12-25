import os

from ..utils import get_id
from .client import SessionClient
from .environment import Environment


class SessionManager:
    def __init__(self, env_id: str | None = None, env_dir: str | None = None) -> None:
        self.env_id = env_id or get_id(prefix="env")
        self.env_dir = env_dir or os.environ.get(
            "SCIMATE_ENV_DIR", os.path.realpath(os.getcwd())
        )

        self.env = Environment(env_id=self.env_id, env_dir=self.env_dir)

    def initialize(self) -> None:
        # Nothing to do here
        ...

    def cleanup(self) -> None:
        # Nothing to do here
        ...

    def get_session_client(
        self,
        session_id: str,
        session_dir: str | None = None,
        cwd: str | None = None,
    ) -> SessionClient:
        session_dir = session_dir or self.env.get_default_session_dir(session_id)
        return SessionClient(
            env=self.env,
            session_id=session_id,
            session_dir=session_dir,
            cwd=cwd,
        )
