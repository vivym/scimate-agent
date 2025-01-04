from typing import Any, Literal

from langgraph.types import interrupt
from pydantic import BaseModel

INTERRUPTION_REASON = Literal["greeting", "ask_user"]


class Interruption(BaseModel):
    reason: INTERRUPTION_REASON

    message: str

    @classmethod
    def greeting(cls, message: str) -> "Interruption":
        return cls(reason="greeting", message=message)

    @classmethod
    def ask_user(cls, message: str) -> "Interruption":
        return cls(reason="ask_user", message=message)

    def interrupt(self) -> Any:
        return interrupt(self)
