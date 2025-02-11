from typing import Any

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from .nodes import (
    code_executor_node,
    code_executor_router_edge,
    code_generator_node,
    code_generator_router_edge,
    code_interpreter_node,
    code_verifier_node,
    code_verifier_router_edge,
    human_node,
    planner_node,
    planner_router_edge,
)
from .state import AgentState, CodeInterpreterState


def create_scimate_agent_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("planner_node", planner_node)
    graph_builder.add_node("code_interpreter_node", code_interpreter_node)
    graph_builder.add_node("human_node", human_node)


    graph_builder.add_edge(START, "planner_node")
    graph_builder.add_conditional_edges(
        "planner_node",
        planner_router_edge,
        {
            "code_interpreter_node": "code_interpreter_node",
            "human_node": "human_node",
        },
    )
    graph_builder.add_edge("code_interpreter_node", "planner_node")

    checkpointer = MemorySaver()

    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph


def create_code_interpreter_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(CodeInterpreterState)

    graph_builder.add_node("code_generator_node", code_generator_node)
    graph_builder.add_node("code_verifier_node", code_verifier_node)
    graph_builder.add_node("code_executor_node", code_executor_node)

    graph_builder.add_edge(START, "code_generator_node")
    graph_builder.add_conditional_edges(
        "code_generator_node",
        code_generator_router_edge,
        {
            "code_verifier_node": "code_verifier_node",
            "code_generator_node": "code_generator_node",
            END: END,
        },
    )
    graph_builder.add_conditional_edges(
        "code_verifier_node",
        code_verifier_router_edge,
        {
            "code_executor_node": "code_executor_node",
            "code_generator_node": "code_generator_node",
            END: END,
        },
    )
    graph_builder.add_conditional_edges(
        "code_executor_node",
        code_executor_router_edge,
        {
            "code_generator_node": "code_generator_node",
            END: END,
        },
    )

    checkpointer = MemorySaver()

    graph = graph_builder.compile(checkpointer=checkpointer)

    return graph


scimate_agent_graph = create_scimate_agent_graph()
code_interpreter_graph = create_code_interpreter_graph()
