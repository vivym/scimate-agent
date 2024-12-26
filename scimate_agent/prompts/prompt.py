from functools import lru_cache
from pathlib import Path
from typing import Literal

PromptName = Literal[
    "planner_system_message",
    "code_generator_system_message",
]


@lru_cache
def get_prompt_template(name: PromptName) -> str:
    prompt_path = Path(__file__).parent / "assets" / f"{name}.txt"
    assert prompt_path.exists(), f"Prompt `{name}` not found."

    with open(prompt_path, "r") as f:
        return f.read()
