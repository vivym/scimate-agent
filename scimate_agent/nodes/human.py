from scimate_agent.interrupt import Interruption
from scimate_agent.state import AgentState, Post, Round


def human_node(state: AgentState):
    current_round = state.get_rounds("User")[-1]
    assert len(current_round.posts) > 0, "No post found for User."

    current_post = current_round.posts[-1]
    assert current_post.send_to == "User", "Invalid post, send_to must be User."

    user_input = Interruption.ask_user(current_post.message).interrupt()

    return {
        "rounds": Round.new(
            user_query=user_input,
            posts=[
                Post.new(
                    send_from="User",
                    send_to="Planner",
                    message=user_input,
                )
            ],
        )
    }
