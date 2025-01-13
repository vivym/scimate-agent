import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel


class AttachmentType(Enum):
    # Planner
    THOUGHT = "thought"
    INIT_PLAN = "init_plan"
    PLAN = "plan"
    CURRENT_PLAN_STEP = "current_plan_step"

    # CodeInterpreter
    PLAN_ENRICHMENT = "plan_enrichment"
    CODE_GENERATION_RESULT = "code_generation_result"
    CODE_VERIFICATION_RESULT = "code_verification_result"
    CODE_EXECUTION_RESULT = "code_execution_result"


class Attachment(BaseModel):
    id: str
    type: AttachmentType
    content: str
    extra: Any | None = None

    @classmethod
    def new(
        cls,
        type: AttachmentType,
        content: str,
        extra: Any | None = None,
        id: str | None = None,
    ) -> "Attachment":
        if id is None:
            id = str(uuid.uuid4())
        return cls(
            id=id,
            type=type,
            content=content,
            extra=extra,
        )
