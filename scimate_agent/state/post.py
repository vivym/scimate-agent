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

    def update(self, update: "PostUpdate") -> "Post":
        """CAUTION: This method does not mutate the post in place. It returns a new post."""

        if update.id != self.id:
            raise ValueError("PostUpdate.id does not match the post's id")

        post = Post.new(
            id=self.id,
            send_from=update.send_from if update.send_from is not None else self.send_from,
            send_to=update.send_to if update.send_to is not None else self.send_to,
            message=update.message if update.message is not None else self.message,
            attachments=self.attachments,
            original_messages=self.original_messages,
        )

        if update.attachments is not None:
            # Do not use `extend` because it mutates the list in place
            post.attachments = post.attachments + update.attachments

        if update.original_messages is not None:
            if post.original_messages is None:
                post.original_messages = []
            # Do not use `extend` because it mutates the list in place
            post.original_messages = post.original_messages + update.original_messages

        return post


class PostUpdate(BaseModel):
    id: str
    send_from: str | None = None
    send_to: str | None = None
    message: str | None = None
    attachments: list[Attachment] | None = None
    original_messages: list[BaseMessage] | None = None

    def to_post(self) -> Post:
        if self.send_from is None:
            raise ValueError("send_from is required")
        if self.send_to is None:
            raise ValueError("send_to is required")
        if self.message is None:
            raise ValueError("message is required")

        if self.attachments is None:
            self.attachments = []

        return Post.new(
            id=self.id,
            send_from=self.send_from,
            send_to=self.send_to,
            message=self.message,
            attachments=self.attachments,
            original_messages=self.original_messages,
        )
