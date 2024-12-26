from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from .nodes import (
    code_generator_node,
    code_generator_router_edge,
    planner_node,
    planner_router_edge,
)
from .state import AgentState


def code_executor_node(state: AgentState) -> dict[str, Any]:
    print("Dummy CodeExecutor Node")

    rounds = state.get_rounds("CodeExecutor")
    assert len(rounds) > 0, "No round found for CodeExecutor."

    last_round = rounds[-1]
    assert len(last_round.posts) > 0, "No post found for CodeExecutor."
    last_post = last_round.posts[-1]

    print("-" * 100)
    print(last_post.message)
    print("-" * 100)

    return {}


def human_node(state: AgentState) -> dict[str, Any]:
    print("Dummy Human Node")
    return {}


def create_agent_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("planner_node", planner_node)
    graph_builder.add_node("code_generator_node", code_generator_node)
    graph_builder.add_node("code_executor_node", code_executor_node)
    graph_builder.add_node("human_node", human_node)


    graph_builder.add_edge(START, "planner_node")
    graph_builder.add_conditional_edges(
        "planner_node",
        planner_router_edge,
        {
            "code_generator_node": "code_generator_node",
            "human_node": "human_node",
        },
    )
    graph_builder.add_conditional_edges(
        "code_generator_node",
        code_generator_router_edge,
        {
            "planner_node": "planner_node",
            "code_executor_node": "code_executor_node",
        },
    )
    graph_builder.add_edge("code_executor_node", END)
    graph_builder.add_edge("human_node", END)

    checkpointer = MemorySaver()

    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph


graph = create_agent_graph()
