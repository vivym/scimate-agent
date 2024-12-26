from functools import lru_cache
from typing import Any, Literal

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

from scimate_agent.state import AgentState
from scimate_agent.prompts.prompt import get_prompt_template
from scimate_agent.state import (
    AgentState,
    Attachment,
    AttachmentType,
    Post,
    Round,
    RoundUpdate,
)
from scimate_agent.utils.env import get_env_context


class CodeGenerationResult(BaseModel):
    thought: str = Field(description="The thoughts before generating the code.")

    reply_type: Literal["python", "text"] = Field(
        description=(
            "The type of the reply, which can be 'python' or 'text'. "
            "Select 'text' if the response is not a executable code snippet."
        )
    )

    reply_content: str = Field(
        description=(
            "The actual content of the response. If the reply_type is 'python', the content should be a valid python code snippet. "
            "Make sure escaping the special characters (e.g., '\\', '/', and '\"') in the strings for JSON format."
        )
    )

    def to_post(
        self,
        id: str | None = None,
        original_messages: list[BaseMessage] | None = None,
    ) -> Post:
        send_to = "CodeExecutor" if self.reply_type == "python" else "Planner"
        return Post.new(
            id=id,
            send_from="CodeGenerator",
            send_to=send_to,
            message=self.reply_content,
            original_messages=original_messages,
        )


@lru_cache
def _get_code_generator_llm(llm_vendor: str, llm_model: str, llm_temperature: float):
    if llm_vendor == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=llm_model, temperature=llm_temperature)
    elif llm_vendor == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(model=llm_model, temperature=llm_temperature)
    else:
        raise ValueError(f"Unsupported LLM vendor: {llm_vendor}")

    return llm.with_structured_output(CodeGenerationResult, include_raw=True)


def get_code_generator_llm(config: RunnableConfig):
    llm_vendor = config["configurable"].get("llm_vendor", "openai")
    llm_model = config["configurable"].get("llm_model", "gpt-4o-mini")
    llm_temperature = config["configurable"].get("llm_temperature", 0)
    return _get_code_generator_llm(llm_vendor, llm_model, llm_temperature)


def format_messages(rounds: list[Round]):
    messages = []

    system_message = get_prompt_template("code_generator_system_message").format(
        ENVIRONMENT_CONTEXT=get_env_context(),
        ROLE_NAME="CodeGenerator",
    )

    # TODO: add experiences to the system message

    messages.append(SystemMessage(content=system_message))

    # TODO: add examples

    # TODO: compress history rounds if needed

    conv_prefix = get_prompt_template("code_generator_conv_head").format(
        SUMMARY="",  # TODO: add summary
        PLUGINS="",  # TODO: add plugins
        ROLE_NAME="CodeGenerator",
    )

    for rnd_idx, round in enumerate(rounds):
        for post_idx, post in enumerate(round.posts):
            is_first_post = rnd_idx == 0 and post_idx == 0
            is_final_post = rnd_idx == len(rounds) - 1 and post_idx == len(round.posts) - 1

            if post.send_from == "Planner" and post.send_to == "CodeGenerator":
                if is_final_post:
                    enrichment = f"The user request is: {round.user_query}\n\n"
                else:
                    enrichment = ""

                if is_first_post:
                    message = conv_prefix + "\n"
                else:
                    message = ""

                message += get_prompt_template("code_generator_user_message").format(
                    FEEDBACK="None",  # TODO: add feedback
                    MESSAGE=f"{enrichment}The task for this specific step is: {post.message}",
                )

                if is_final_post:
                    message += "\n" + get_prompt_template("code_generator_requirements").format(
                        ROLE_NAME="CodeGenerator",
                        CODE_GENERATION_REQUIREMENTS="",  # TODO: add code generation requirements
                    )

                messages.append(HumanMessage(content=message))
            elif post.send_from == "CodeGenerator" and post.send_to == "CodeGenerator":
                # Self-correction
                messages += post.original_messages
            elif post.send_from == "CodeGenerator" and post.send_to == "Planner":
                messages += post.original_messages
            else:
                raise ValueError(f"Invalid post: {post}")

    return messages


def code_generator_node(state: AgentState, config: RunnableConfig):
    rounds = state.get_rounds("CodeGenerator")
    assert len(rounds) > 0, "No round found for CodeGenerator."

    current_round = rounds[-1]

    messages = format_messages(rounds)

    llm = get_code_generator_llm(config)
    result: dict[Literal["raw", "parsed", "parsing_error"], Any] = llm.invoke(messages)

    raw_message: AIMessage = result["raw"]
    cg_result: CodeGenerationResult = result["parsed"]
    parsing_error: BaseException | None = result["parsing_error"]

    revise_message = None

    if parsing_error is not None:
        revise_message = f"Parsing error:\n{parsing_error}\n\nPlease regenerate the code."

    if cg_result.reply_type not in ["python", "text"]:
        revise_message = (
            f"Unsupported reply_type: `{cg_result.reply_type}`. Please check the `reply_type` field. "
            "Only `python` and `text` are supported."
        )

    # TODO: Code Verification

    assert len(raw_message.tool_calls) == 1, f"Invalid tool call count: {len(raw_message.tool_calls)}"
    posts = [
        cg_result.to_post(
            original_messages=[
                raw_message,
                ToolMessage(content="", tool_call_id=raw_message.tool_calls[0]["id"]),
            ]
        )
    ]

    self_correction_count = state.code_generator_self_correction_count

    if revise_message is not None:
        # Self-correction. Max 3 times.
        posts.append(
            Post.new(
                send_from="Validator",  # TODO: CG Validator?
                send_to="CodeGenerator",
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
        "code_generator_self_correction_count": self_correction_count,
    }


def code_generator_router_edge(state: AgentState) -> str:
    rounds = state.rounds
    assert len(rounds) > 0, "No round found for CodeGenerator."

    last_round = rounds[-1]
    if len(last_round.posts) == 0:
        raise ValueError("No post found for CodeGenerator.")
    last_post = last_round.posts[-1]

    assert last_post.send_from in ["CodeGenerator", "Validator"], (
        "Last post is not from CodeGenerator or Validator."
    )

    if last_post.send_from == "CodeGenerator":
        if last_post.send_to == "Planner":
            return "planner_node"
        elif last_post.send_to == "CodeExecutor":
            return "code_executor_node"
        else:
            raise ValueError(f"Unsupported send_to: {last_post.send_to}")
    else:
        # From `Validator`, self-correction
        assert last_post.send_to == "CodeGenerator", (
            f"Validator must send to CodeGenerator, but got `{last_post.send_to}`."
        )

        self_correction_count = state.code_generator_self_correction_count
        if self_correction_count is None or self_correction_count <= 3:
            return "code_generator_node"
        else:
            return END
