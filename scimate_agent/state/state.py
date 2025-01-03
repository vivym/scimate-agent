from typing import Annotated

from pydantic import BaseModel

from scimate_agent.role import Role
from .plugin import PluginEntry
from .round import Round, update_rounds


class AgentState(BaseModel):
    rounds: Annotated[list[Round], update_rounds]

    plugins: list[PluginEntry]

    planner_self_correction_count: int | None = None

    @classmethod
    def new_initial_state(
        cls,
        user_query: str,
        plugins: list[PluginEntry] | None = None,
    ) -> "AgentState":
        return cls(
            rounds=[Round.new(user_query)],
            plugins=[] if plugins is None else plugins,
        )

    def get_rounds(self, role: Role | None = None, include_failure_rounds: bool = False) -> list[Round]:
        rounds: list[Round] = []

        for round in self.rounds:
            round: Round
            if round.status == "failed" and not include_failure_rounds:
                continue

            new_round = round.model_copy()
            new_round.posts = [
                post
                for post in round.posts
                if role is None or post.send_from == role or post.send_to == role
            ]

            rounds.append(new_round)

        return rounds


class CodeInterpreterState(BaseModel):
    rounds: Annotated[list[Round], update_rounds]

    plugins: list[PluginEntry]

    self_correction_count: int | None = None

    # Execution kernel settings
    env_id: str | None = None
    env_dir: str | None = None
    session_id: str | None = None

    def get_rounds(self, role: Role | None = None, include_failure_rounds: bool = False) -> list[Round]:
        rounds: list[Round] = []

        for round in self.rounds:
            round: Round
            if round.status == "failed" and not include_failure_rounds:
                continue

            new_round = round.model_copy()
            new_round.posts = [
                post
                for post in round.posts
                if role is None or post.send_from == role or post.send_to == role
            ]

            rounds.append(new_round)

        return rounds

    def cleanup(self):
        if self.env_id is not None:
            from scimate_agent.nodes.code_executor import get_session_client

            assert self.env_dir is not None, "Internal error: env_dir is None"
            assert self.session_id is not None, "Internal error: session_id is None"
            client = get_session_client(env_id=self.env_id, env_dir=self.env_dir, session_id=self.session_id)
            client.stop()

        self.env_id = None
        self.env_dir = None
        self.session_id = None
