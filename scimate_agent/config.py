from pydantic import BaseModel, ConfigDict


class AgentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    # LLM
    llm_vendor: str = "openai"
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.0

    # Event
    event_handle: str | None = None

    # Execution environment
    env_id: str | None = None
    env_dir: str | None = None
    session_id: str | None = None

    # Code Verification
    allowed_modules: list[str] | None = None
    blocked_modules: list[str] | None = None
    allowed_functions: list[str] | None = None
    blocked_functions: list[str] | None = None
    allowed_variables: list[str] | None = None
    blocked_variables: list[str] | None = None
