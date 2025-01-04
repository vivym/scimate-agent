from langgraph.graph import END
from langgraph.types import Command

from scimate_agent.interrupt import ExitCommand, Interruption
from scimate_agent.state import AgentState, Post, Round


def human_node(state: AgentState):
    current_round = state.get_rounds("User")[-1]
    assert len(current_round.posts) > 0, "No post found for User."

    current_post = current_round.posts[-1]
    assert current_post.send_to == "User", "Invalid post, send_to must be User."

    user_query = Interruption.ask_user(current_post.message).interrupt()

    if isinstance(user_query, ExitCommand):
        return Command(goto=END)
    else:
        return Command(
            goto="planner_node",
            update={
                "rounds": Round.new(
                    user_query=user_query,
                    posts=[
                        Post.new(
                            send_from="User",
                            send_to="Planner",
                            message=user_query,
                        )
                    ],
                )
            },
        )
