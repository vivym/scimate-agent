from typing import Annotated, Optional, TYPE_CHECKING

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from scimate_agent.nodes.planner import Plan


class AgentState(TypedDict):
    user_initial_query: str

    planner_messages: Annotated[list[BaseMessage], add_messages]

    current_plan: Optional[Plan]

    code_generator_messages: Annotated[list[BaseMessage], add_messages]
