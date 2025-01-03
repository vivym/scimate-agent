from scimate_agent.interrupt import Interruption
from scimate_agent.state import AgentState


def human_node(state: AgentState):
    if state["human_state"] == "greeting":
        user_query = Interruption.greeting("Hello! How can I help you today?").interrupt()
