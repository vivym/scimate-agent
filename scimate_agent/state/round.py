import uuid
from typing import Literal

from pydantic import BaseModel

from .post import Post, PostUpdate

RoundStatus = Literal["created", "finished", "failed"]


class Round(BaseModel):
    id: str
    user_query: str
    posts: list[Post]
    status: RoundStatus

    @classmethod
    def new(
        cls,
        user_query: str,
        id: str | None = None,
        posts: list[Post] | None = None,
        status: RoundStatus = "created",
    ) -> "Round":
        id = id if id is not None else str(uuid.uuid4())
        if posts is None or len(posts) == 0:
            posts = [
                Post.new(
                    send_from="User",
                    send_to="Planner",
                    message=user_query,
                )
            ]
        return cls(id=id, user_query=user_query, posts=posts, status=status)


class RoundUpdate(BaseModel):
    id: str | None = None
    user_query: str | None = None
    posts: list[Post] | None = None
    status: RoundStatus | None = None


def update_rounds(
    rounds: list[Round], updates: RoundUpdate | list[RoundUpdate]
) -> list[Round]:
    # Make a shallow copy of the rounds list, so we can modify it in place
    rounds = rounds.copy()

    if isinstance(updates, (RoundUpdate, Round)):
        updates = [updates]
    assert isinstance(updates, (list, tuple)), f"updates must be a list or tuple, got {type(updates)}"

    for update in updates:
        assert isinstance(update, (RoundUpdate, Round)), f"updates must be a list of RoundUpdate or Round, got {type(update)}"

        if update.id is None:
            round = None
        else:
            round_idx, round = next(
                ((i, r) for i, r in enumerate(rounds) if r.id == update.id), (None, None)
            )

        if round is None:
            # Start a new round
            if update.user_query is None:
                raise ValueError("user_query is required")

            new_round = Round(
                id=update.id if update.id is not None else str(uuid.uuid4()),
                user_query=update.user_query,
                posts=update.posts if update.posts is not None else [],
                status=update.status if update.status is not None else "created",
            )
            rounds.append(new_round)
        else:
            # Update an existing round
            new_round = round.model_copy()

            if update.user_query is not None:
                new_round.user_query = update.user_query

            if update.posts is not None:
                # Do not use `extend` because it mutates the list in place
                new_posts = [p for p in new_round.posts]
                existing_posts = {p.id: p for p in new_posts}
                for post in update.posts:
                    if isinstance(post, PostUpdate):
                        if post.id in existing_posts:
                            existing_posts[post.id] = existing_posts[post.id].update(post)
                        else:
                            new_posts.append(post.to_post())
                    elif isinstance(post, Post):
                        new_posts.append(post)
                    else:
                        raise ValueError(f"Invalid post: {post}")

                new_round.posts = new_posts

            if update.status is not None:
                new_round.status = update.status

            rounds[round_idx] = new_round

    return rounds
