from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.graph import END
from pydantic import BaseModel, Field

from scimate_agent.prompts import get_prompt_template
from scimate_agent.utils.env import get_env_context
if TYPE_CHECKING:
    from scimate_agent.state import AgentState


class Plan(BaseModel):
    init_plan: str = Field(
        description=(
            "The initial plan to decompose the User's task into subtasks and list them as the detailed subtask steps. "
            "The initial plan must contain dependency annotations for sequential and interactive dependencies."
        )
    )

    plan: str = Field(
        description=(
            "The refined plan by merging adjacent steps that have sequential dependency or no dependency. "
            "The final plan must not contain dependency annotations."
        )
    )

    current_plan_step: str = Field(
        description="The current step Planner is executing."
    )

    review: str = Field(
        description=(
            "The review of the current step. If the Worker's response is incorrect or incomplete, "
            "Planner should provide feedback to the Worker."
        )
    )

    send_to: str = Field(
        description="The name of character (User or name of the Worker) that Planner wants to speak to."
    )

    message: str = Field(
        description="The message of Planner sent to the receipt Character. If there is any file path in the message, it should be formatted as links in Markdown, i.e., [file_name](file_path)"
    )


@lru_cache
def _get_planner_llm(llm_vendor: str, llm_model: str, llm_temperature: float):
    if llm_vendor == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
    elif llm_vendor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=llm_model, temperature=llm_temperature)
    else:
        raise ValueError(f"Unsupported LLM vendor: {llm_vendor}")

    return llm.with_structured_output(Plan)


def planner_start_node(state: AgentState, config: RunnableConfig):
    planner_messages = state["planner_messages"]
    user_initial_query = state["user_initial_query"]

    messages = []
    if len(planner_messages) == 0:
        function_manager = config["configurable"].get("function_manager", None)

        system_message = get_prompt_template("planner_system_message").format(
            environment_context=get_env_context(),
            function_description=function_manager.get_function_description_for_planner(),
        )

        messages.append(SystemMessage(content=system_message))

    messages.append(HumanMessage(content=user_initial_query))

    return {"planner_messages": messages}


def planner_thinking_node(state: AgentState, config: RunnableConfig):
    llm_vendor = config["configurable"].get("llm_vendor", "openai")
    llm_model = config["configurable"].get("llm_model", "gpt-4o-mini")
    llm_temperature = config["configurable"].get("llm_temperature", 0)
    llm = _get_planner_llm(llm_vendor, llm_model, llm_temperature)

    plan: Plan = llm.invoke(state["planner_messages"])

    return {"current_plan": plan}


def planner_router_edge(state: AgentState) -> str:
    plan = state["current_plan"]

    assert plan is not None, "Plan is not generated."

    if plan.send_to == "User":
        return "planner_start_node"
    elif plan.send_to == "CodeGenerator":
        return "code_generator_start_node"
    else:
        raise ValueError(f"Unsupported send_to: {plan.send_to}")
