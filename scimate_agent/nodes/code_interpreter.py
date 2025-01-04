from typing import Any

from langchain_core.runnables import RunnableConfig

from scimate_agent.state import AgentState, CodeInterpreterState, Round, RoundUpdate


def code_interpreter_node(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    rounds = state.get_rounds("CodeInterpreter")
    assert len(rounds) > 0, "No round found for CodeInterpreter."

    current_round = rounds[-1]
    if len(current_round.posts) == 0:
        raise ValueError("No post found for CodeInterpreter.")

    current_post = current_round.posts[-1]

    assert current_post.send_from == "Planner", "CodeInterpreter must receive a post from Planner."
    assert current_post.send_to == "CodeInterpreter", "Invalid post, send_to must be CodeInterpreter."

    new_post = current_post.model_copy()
    new_post.send_to = "CodeGenerator"

    ci_state = CodeInterpreterState(
        rounds=[
            Round.new(
                user_query=current_round.user_query,
                posts=[new_post],
            )
        ],
        plugins=state.plugins,
        self_correction_count=None,
        env_id=state.env_id,
        env_dir=state.env_dir,
        session_id=state.session_id,
    )

    from scimate_agent.agent import code_interpreter_graph

    result = code_interpreter_graph.invoke(
        ci_state,
        config=config,
    )
    final_state = CodeInterpreterState(**result)

    ci_rounds = final_state.rounds
    assert len(ci_rounds) > 0, "No round found for CodeInterpreter."

    ci_current_round = ci_rounds[-1]
    assert len(ci_current_round.posts) > 0, "No post found for CodeInterpreter."

    ci_current_post = ci_current_round.posts[-1]
    assert ci_current_post.send_to == "Planner", (
        f"Invalid post, send_to must be Planner: {ci_current_post.send_from} -> {ci_current_post.send_to}.\n"
        f"{current_post}"
    )

    new_post = ci_current_post.model_copy()
    new_post.send_from = "CodeInterpreter"

    return {
        "rounds": RoundUpdate(
            id=current_round.id,
            posts=[new_post],
        ),
        "plugins": final_state.plugins,
        "env_id": final_state.env_id,
        "env_dir": final_state.env_dir,
        "session_id": final_state.session_id,
    }
