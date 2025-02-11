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

from scimate_agent.config import AgentConfig
from scimate_agent.event import EventEmitter
from scimate_agent.prompts.prompt import get_prompt_template
from scimate_agent.state import (
    Attachment,
    AttachmentType,
    CodeInterpreterState,
    PluginEntry,
    Post,
    Round,
    RoundUpdate,
)
from scimate_agent.utils.env import get_env_context

ROLE_NAME = "CodeGenerator"


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
        send_to = "CodeVerifier" if self.reply_type == "python" else "Planner"
        return Post.new(
            id=id,
            send_from="CodeGenerator",
            send_to=send_to,
            message=self.reply_content,
            attachments=[
                Attachment.new(
                    type=AttachmentType.CODE_GENERATION_RESULT,
                    content=self.reply_content,
                    extra=self,
                )
            ],
            original_messages=original_messages,
        )


class Example(BaseModel):
    rounds: list[Round]
    plugins: list[str]


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


def get_code_generator_llm(config: AgentConfig):
    return _get_code_generator_llm(
        config.llm_vendor,
        config.llm_model,
        config.llm_temperature,
    )


def format_feedback(post: Post | None) -> str:
    if post is None:
        return "None"

    assert (post.send_from, post.send_to) in [
        ("Reviser", "CodeGenerator"),
        ("CodeVerifier", "CodeGenerator"),
        ("CodeExecutor", "CodeGenerator"),
        ("CodeExecutor", "Planner"),
    ], f"Invalid post: {post.send_from} -> {post.send_to}"

    feedback_items: dict[str, Attachment] = {}

    for attachment in reversed(post.attachments):
        if len(feedback_items) >= 2:
            break

        if attachment.type == AttachmentType.CODE_VERIFICATION_RESULT:
            if "verification_result" not in feedback_items:
                feedback_items["verification_result"] = attachment
        elif attachment.type == AttachmentType.CODE_EXECUTION_RESULT:
            if "execution_result" not in feedback_items:
                feedback_items["execution_result"] = attachment

    feedback = ""
    if "verification_result" in feedback_items:
        verification_result = feedback_items["verification_result"].extra
        if verification_result is not None:
            assert isinstance(verification_result, list), "Verification result must be a list."
            error_msg = "\n".join(verification_result)
            feedback += f"## Verification\nCode verification detected the following issues:\n{error_msg}\n"
        else:
            feedback += f"## Verification\nCode verification has been passed.\n"

    if "execution_result" in feedback_items:
        execution_result = feedback_items["execution_result"]
        if isinstance(execution_result.extra, dict):
            is_success = execution_result.extra.get("is_success", False)
        else:
            is_success = execution_result.extra.is_success

        if is_success:
            feedback += f"## Execution\nThe code has been executed successfully with the following result:\n"
        else:
            feedback += f"## Execution\nThe code has failed to execute with the following error:\n"
        feedback += f"{execution_result.content}\n"

    if feedback == "":
        feedback = "None"

    return feedback


def format_code_generation_requirements(role_name: str, agent_config: AgentConfig) -> str:
    allowed_modules = agent_config.allowed_modules
    blocked_modules = agent_config.blocked_modules
    allowed_functions = agent_config.allowed_functions
    blocked_functions = agent_config.blocked_functions
    allowed_variables = agent_config.allowed_variables
    blocked_variables = agent_config.blocked_variables

    requirements: list[str] = []

    if allowed_modules is not None:
        if len(allowed_modules) > 0:
            requirements.append(
                f"- {role_name} can only import the following Python modules: "
                + ", ".join([f"{module}" for module in allowed_modules]),
            )
        else:
            requirements.append(f"- {role_name} cannot import any Python modules.")

    if blocked_modules is not None:
        if len(blocked_modules) > 0:
            requirements.append(
                f"- {role_name} cannot use the following Python modules: "
                + ", ".join([f"{module}" for module in blocked_modules]),
            )

    if allowed_functions is not None:
        if len(allowed_functions) > 0:
            requirements.append(
                f"- {role_name} can only use the following Python functions: "
                + ", ".join([f"{func}" for func in allowed_functions]),
            )
        else:
            requirements.append(f"- {role_name} cannot use any Python functions.")

    if blocked_functions is not None:
        if len(blocked_functions) > 0:
            requirements.append(
                f"- {role_name} cannot use the following Python functions: "
                + ", ".join([f"{func}" for func in blocked_functions]),
            )

    if allowed_variables is not None:
        if len(allowed_variables) > 0:
            requirements.append(
                f"- {role_name} can only use the following variables: "
                + ", ".join([f"{var}" for var in allowed_variables]),
            )
        else:
            requirements.append(f"- {role_name} cannot use any variables.")

    if blocked_variables is not None:
        if len(blocked_variables) > 0:
            requirements.append(
                f"- {role_name} cannot use the following variables: "
                + ", ".join([f"{var}" for var in blocked_variables]),
            )

    return "\n".join(requirements)


def format_conversation(
    rounds: list[Round],
    agent_config: AgentConfig,
    plugins: list[PluginEntry] | None = None,
    add_requirements: bool = False,
    summary: str | None = None,
):
    messages = []

    conv_prefix = get_prompt_template("code_generator_conv_head").format(
        SUMMARY=summary if summary is not None else "None",
        PLUGINS="\n".join([p.format_prompt() for p in plugins]) if plugins else "None",
        ROLE_NAME=ROLE_NAME,
    )

    last_post = None
    for rnd_idx, round in enumerate(rounds):
        for post_idx, post in enumerate(round.posts):
            is_first_post = rnd_idx == 0 and post_idx == 0
            is_final_post = rnd_idx == len(rounds) - 1 and post_idx == len(round.posts) - 1

            if post.send_from == "Planner" and post.send_to == "CodeGenerator":
                if is_final_post:
                    enrichment = f"The user request is: {round.user_query}\n\n"

                    plan_enrichments = post.get_attachments(AttachmentType.PLAN_ENRICHMENT)
                    if len(plan_enrichments) > 0:
                        enrichment += "Additional context:\n" + "\n".join(
                            [e.content for e in plan_enrichments]
                        ) + "\n\n"
                else:
                    enrichment = ""

                if is_first_post:
                    message = conv_prefix + "\n"
                else:
                    message = ""

                feedback = "None"
                if last_post is not None:
                    feedback = format_feedback(last_post)

                message += get_prompt_template("code_generator_user_message").format(
                    FEEDBACK=feedback,
                    MESSAGE=f"{enrichment}The task for this specific step is: {post.message}",
                )

                if is_final_post and add_requirements:
                    message += "\n\n" + get_prompt_template("code_generator_requirements").format(
                        ROLE_NAME=ROLE_NAME,
                        CODE_GENERATION_REQUIREMENTS=format_code_generation_requirements(
                            ROLE_NAME, agent_config
                        ),
                    )

                messages.append(HumanMessage(content=message))
            elif post.send_from == "Reviser" and post.send_to == "CodeGenerator":
                # Self-correction
                assert not is_first_post, "Reviser should not be the first post."

                message = get_prompt_template("code_generator_user_message").format(
                    FEEDBACK=format_feedback(post),
                    MESSAGE=post.message,  # revise message
                )

                if is_final_post and add_requirements:
                    message += "\n\n" + get_prompt_template("code_generator_requirements").format(
                        ROLE_NAME=ROLE_NAME,
                        CODE_GENERATION_REQUIREMENTS=format_code_generation_requirements(
                            ROLE_NAME, agent_config
                        ),
                    )

                messages.append(HumanMessage(content=message))
            elif post.send_from in ["CodeVerifier", "CodeExecutor"] and post.send_to == "CodeGenerator":
                # Self-correction
                assert not is_first_post, "CodeVerifier and CodeExecutor should not be the first post."

                if post.send_from == "CodeVerifier":
                    message = (
                        "The generated code has been verified and some errors are found. "
                        "If you think you can fix the problem by rewriting the code, "
                        "please do it and try again.\n"
                        "Otherwise, please explain the problem to me."
                    )
                else:
                    message = (
                        "The execution of the previous generated code has failed. "
                        "If you think you can fix the problem by rewriting the code, "
                        "please generate code and run it again.\n"
                        "Otherwise, please explain the problem to me."
                    )

                message = get_prompt_template("code_generator_user_message").format(
                    FEEDBACK=format_feedback(post),
                    MESSAGE=message,
                )

                if is_final_post and add_requirements:
                    message += "\n\n" + get_prompt_template("code_generator_requirements").format(
                        ROLE_NAME=ROLE_NAME,
                        CODE_GENERATION_REQUIREMENTS=format_code_generation_requirements(
                            ROLE_NAME, agent_config
                        ),
                    )

                messages.append(HumanMessage(content=message))
            elif post.send_from == "CodeGenerator" and post.send_to in [
                "CodeVerifier",
                "Planner",
                "Reviser",
            ]:
                assert post.original_messages is not None, "Original messages are required."
                original_messages = [
                    lc_load(msg)
                    for msg in post.original_messages
                ]
                messages += original_messages

                if is_final_post:
                    # This human message is added only for examples and context summarization
                    message = get_prompt_template("code_generator_user_message").format(
                        FEEDBACK=format_feedback(post),
                        MESSAGE="This is the feedback.",
                    )
                    messages.append(HumanMessage(content=message))
            elif post.send_from == "CodeVerifier" and post.send_to == "CodeExecutor":
                # Ignore this post.
                pass
            elif post.send_from == "CodeExecutor" and post.send_to == "Planner":
                # Ignore this post.
                pass
            else:
                raise ValueError(f"Invalid post ({post.send_from} -> {post.send_to}): {post}")

            last_post = post

    return messages


def format_messages(
    rounds: list[Round],
    agent_config: AgentConfig,
    plugins: list[PluginEntry] | None = None,
    examples: list[Example] | None = None,
) -> list[BaseMessage]:
    messages = []

    system_message = get_prompt_template("code_generator_system_message").format(
        ENVIRONMENT_CONTEXT=get_env_context(),
        ROLE_NAME=ROLE_NAME,
    )

    # TODO: add experiences to the system message

    messages.append(SystemMessage(content=system_message))

    if examples is not None:
        for example in examples:
            messages += format_conversation(
                example.rounds, agent_config, plugins=example.plugins, add_requirements=False
            )

    # TODO: compress history rounds if needed
    summary = None

    messages += format_conversation(
        rounds, agent_config, plugins=plugins, add_requirements=True, summary=summary
    )

    return messages


async def code_generator_node(state: CodeInterpreterState, config: RunnableConfig):
    rounds = state.get_rounds()
    assert len(rounds) > 0, "No round found for CodeGenerator."

    current_round = rounds[-1]

    agent_config: AgentConfig = config["configurable"]["agent_config"]
    assert isinstance(agent_config, AgentConfig), (
        f"Agent config is not an instance of AgentConfig: {type(agent_config)}"
    )

    messages = format_messages(rounds, agent_config, plugins=state.plugins)

    llm = get_code_generator_llm(agent_config)
    result: dict[Literal["raw", "parsed", "parsing_error"], Any] = await llm.ainvoke(messages)

    raw_message: AIMessage = result["raw"]
    cg_result: CodeGenerationResult = result["parsed"]
    parsing_error: BaseException | None = result["parsing_error"]

    event_emitter = EventEmitter.get_instance(agent_config.event_handle)
    await event_emitter.emit("cg_result", cg_result.model_dump(mode="json"))

    revise_message = None

    if parsing_error is not None:
        revise_message = f"Parsing error:\n{parsing_error}\n\nPlease try again."

    if cg_result.reply_type not in ["python", "text"]:
        revise_message = (
            f"Unsupported reply_type: `{cg_result.reply_type}`. Please check the `reply_type` field. "
            "Only `python` and `text` are supported. Please try again."
        )

    assert len(raw_message.tool_calls) == 1, f"Invalid tool call count: {len(raw_message.tool_calls)}"
    posts = [
        cg_result.to_post(
            original_messages=[
                raw_message,
                ToolMessage(content="", tool_call_id=raw_message.tool_calls[0]["id"]),
            ]
        )
    ]

    self_correction_count = state.self_correction_count

    if revise_message is not None:
        # Self-correction. Max 3 times.
        posts[-1].send_to = "Reviser"
        posts.append(
            Post.new(
                send_from="Reviser",
                send_to="CodeGenerator",
                message=revise_message,
            )
        )
        self_correction_count = self_correction_count + 1 if self_correction_count is not None else 1

        await event_emitter.emit("cg_revise_message", revise_message)

    return {
        "rounds": RoundUpdate(
            id=current_round.id,
            posts=posts,
        ),
        "self_correction_count": self_correction_count,
    }


def code_generator_router_edge(state: CodeInterpreterState) -> str:
    rounds = state.rounds
    assert len(rounds) > 0, "No round found for CodeGenerator."

    last_round = rounds[-1]
    if len(last_round.posts) == 0:
        raise ValueError("No post found for CodeGenerator.")
    last_post = last_round.posts[-1]

    if last_post.send_from == "CodeGenerator":
        if last_post.send_to == "Planner":
            return END
        elif last_post.send_to == "CodeVerifier":
            return "code_verifier_node"
        else:
            raise ValueError(f"Unsupported send_to: {last_post.send_to}")
    elif last_post.send_from == "Reviser":
        assert last_post.send_to == "CodeGenerator", (
            f"Reviser must send to CodeGenerator, but got `{last_post.send_to}`."
        )

        self_correction_count = state.code_generator_self_correction_count
        if self_correction_count is None or self_correction_count <= 3:
            return "code_generator_node"
        else:
            return END
    else:
        raise ValueError("Last post is not from CodeGenerator or Reviser.")
