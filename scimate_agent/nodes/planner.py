from functools import lru_cache
from typing import Any, Literal

from langchain_core.load import load as lc_load
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from pydantic import BaseModel, Field

from scimate_agent.prompts import get_prompt_template
from scimate_agent.state import (
    AgentState,
    Attachment,
    AttachmentType,
    PluginEntry,
    Post,
    Round,
    RoundUpdate,
)
from scimate_agent.utils.env import get_env_context


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

    def to_post(
        self,
        id: str | None = None,
        original_messages: list[BaseMessage] | None = None,
    ) -> Post:
        attachments = [
            Attachment.new(
                type=AttachmentType.INIT_PLAN,
                content=self.init_plan,
            ),
            Attachment.new(
                type=AttachmentType.PLAN,
                content=self.plan,
            ),
            Attachment.new(
                type=AttachmentType.CURRENT_PLAN_STEP,
                content=self.current_plan_step,
            ),
            Attachment.new(
                type=AttachmentType.REVIEW,
                content=self.review,
            ),
        ]

        return Post.new(
            id=id,
            send_from="Planner",
            send_to=self.send_to,
            message=self.message,
            attachments=attachments,
            original_messages=original_messages,
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

    return llm.with_structured_output(Plan, include_raw=True)


def get_planner_llm(config: RunnableConfig):
    llm_vendor = config["configurable"].get("llm_vendor", "openai")
    llm_model = config["configurable"].get("llm_model", "gpt-4o-mini")
    llm_temperature = config["configurable"].get("llm_temperature", 0)
    return _get_planner_llm(llm_vendor, llm_model, llm_temperature)


def format_messages(rounds: list[Round], plugins: list[PluginEntry]) -> list[BaseMessage]:
    messages = []

    if plugins:
        plugins_desc = "\n".join([p.format_description(indent=4) for p in plugins])
    else:
        plugins_desc = "None"

    system_message = get_prompt_template("planner_system_message").format(
        ENVIRONMENT_CONTEXT=get_env_context(),
        PLUGINS_DESCRIPTION=plugins_desc,
    )

    # TODO: add experiences to the system message

    messages.append(SystemMessage(content=system_message))

    # TODO: add examples

    # TODO: compress history rounds if needed

    conv_prefix = "Let's start the new conversation!"
    for rnd_idx, round in enumerate(rounds):
        for post_idx, post in enumerate(round.posts):
            if rnd_idx == 0 and post_idx == 0:
                assert post.send_from == "User", "The first post must be from User."
                assert post.send_to == "Planner", "The first post must be sent to Planner."

            if post.send_from == "Planner":
                assert post.original_messages is not None, "Original messages are required for Planner."
                original_messages = [
                    lc_load(msg)
                    for msg in post.original_messages
                ]
                messages += original_messages
            else:
                if rnd_idx == 0 and post_idx == 0:
                    message = f"{conv_prefix}\n{post.message}"
                else:
                    message = post.message
                message = f"From: {post.send_from}\nMessage: {message}"
                messages.append(HumanMessage(content=message))

    return messages


def planner_node(state: AgentState, config: RunnableConfig):
    rounds = state.get_rounds("Planner")
    assert len(rounds) > 0, "No round found for Planner."

    current_round = rounds[-1]

    messages = format_messages(rounds, state.plugins)

    llm = get_planner_llm(config)
    result: dict[Literal["raw", "parsed", "parsing_error"], Any] = llm.invoke(messages)

    raw_message: AIMessage = result["raw"]
    plan: Plan = result["parsed"]
    parsing_error: BaseException | None = result["parsing_error"]

    revise_message = None

    if parsing_error is not None:
        revise_message = f"Parsing error:\n{parsing_error}\n\nPlease regenerate the plan."

    if plan.send_to not in ["User", "CodeInterpreter"]:
        revise_message = (
            f"Unsupported send_to: `{plan.send_to}`. Please check the `send_to` field. "
            "Only `User` and `CodeInterpreter` are supported."
        )

    assert len(raw_message.tool_calls) == 1, f"Invalid tool call count: {len(raw_message.tool_calls)}"
    posts = [
        plan.to_post(
            original_messages=[
                raw_message,
                ToolMessage(content="", tool_call_id=raw_message.tool_calls[0]["id"]),
            ]
        )
    ]

    self_correction_count = state.self_correction_count

    if revise_message is not None:
        # Self-correction. Max 3 times.
        posts.append(
            Post.new(
                send_from="Reviser",
                send_to="Planner",
                message=revise_message,
            )
        )
        self_correction_count = self_correction_count + 1 if self_correction_count is not None else 1
    else:
        # Reset self-correction count when the plan is correct.
        self_correction_count = None

    return {
        "rounds": RoundUpdate(
            id=current_round.id,
            posts=posts,
        ),
        "self_correction_count": self_correction_count,
    }


def planner_router_edge(state: AgentState) -> str:
    rounds = state.rounds
    assert len(rounds) > 0, "No round found for Planner."

    last_round = rounds[-1]
    if len(last_round.posts) == 0:
        raise ValueError("No post found for Planner.")
    last_post = last_round.posts[-1]

    assert last_post.send_from in ["Planner", "Reviser"], (
        "Last post is not from Planner or Reviser."
    )

    if last_post.send_from == "Planner":
        if last_post.send_to == "User":
            return "human_node"
        elif last_post.send_to == "CodeInterpreter":
            return "code_interpreter_node"
        else:
            raise ValueError(f"Unsupported send_to: {last_post.send_to}")
    else:
        # From `Reviser`, self-correction
        assert last_post.send_to == "Planner", (
            f"Reviser must send to Planner, but got `{last_post.send_to}`."
        )

        self_correction_count = state.self_correction_count
        if self_correction_count is None or self_correction_count <= 3:
            return "planner_node"
        else:
            return END
