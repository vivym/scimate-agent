import uuid

from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from .attachment import Attachment


class Post(BaseModel):
    id: str
    send_from: str
    send_to: str
    message: str
    attachments: list[Attachment]

    original_messages: list[BaseMessage] | None

    @classmethod
    def new(
        cls,
        send_from: str,
        send_to: str,
        message: str,
        # Optional fields
        id: str | None = None,
        attachments: list[Attachment] | None = None,
        original_messages: list[BaseMessage] | None = None,
    ) -> "Post":
        id = id if id is not None else str(uuid.uuid4())
        attachments = attachments if attachments is not None else []
        return cls(
            id=id,
            send_from=send_from,
            send_to=send_to,
            message=message,
            attachments=attachments,
            original_messages=original_messages,
        )
