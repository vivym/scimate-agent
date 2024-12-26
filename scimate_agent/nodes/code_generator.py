from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from scimate_agent.state import AgentState
from scimate_agent.prompts.prompt import get_prompt_template
from scimate_agent.utils.env import get_env_context


def code_generator_node(state: AgentState, config: RunnableConfig):
    new_state = {}
    new_messages = []

    if state["cg_state"] == "initial":
        new_state["cg_state"] = "running"

        system_message = get_prompt_template("code_generator_system_message").format(
            ENVIRONMENT_CONTEXT=get_env_context(),
            ROLE_NAME="CodeGenerator",
        )

        # TODO: add experiences to the system message

        new_messages.append(SystemMessage(content=system_message))

        # TODO: add examples

    return new_state
