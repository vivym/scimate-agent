from typing import Annotated

from pydantic import BaseModel

from scimate_agent.role import Role
from .round import Round, update_rounds


class AgentState(BaseModel):
    rounds: Annotated[list[Round], update_rounds]

    plugins: list

    planner_self_correction_count: int | None = None

    @classmethod
    def new_initial_state(
        cls,
        user_query: str,
        plugins: list | None = None,
    ) -> "AgentState":
        return cls(
            rounds=[Round.new(user_query)],
            plugins=[] if plugins is None else plugins,
            planner_self_correction_count=None,
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