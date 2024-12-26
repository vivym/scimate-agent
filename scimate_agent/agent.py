from langgraph.graph import START, END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from .nodes import planner_node
from .state import AgentState


def create_agent_graph() -> CompiledStateGraph:
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("planner_node", planner_node)

    graph_builder.add_edge(START, "planner_node")
    graph_builder.add_edge("planner_node", END)

    return graph_builder.compile()


graph = create_agent_graph()
