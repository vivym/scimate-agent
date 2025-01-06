import ast
import asyncio
import re
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END

from scimate_agent.event import EventEmitter
from scimate_agent.state import Attachment, AttachmentType, CodeInterpreterState, Post, RoundUpdate

LINE_MAGIC_PATTERN = re.compile(r"^\s*%\s*[a-zA-Z_]\w*")
CELL_MAGIC_PATTERN = re.compile(r"^\s*%%\s*[a-zA-Z_]\w*")
SHELL_COMMAND_PATTERN = re.compile(r"^\s*!")


def seperate_code_lines(code: str) -> tuple[list[str], str, list[str]]:
    magics = []
    python_lines = []
    shell_lines = []

    inside_cell_magic = False
    for line in code.splitlines():
        if not line.strip() or line.strip().startswith("#"):
            continue

        if inside_cell_magic:
            magics.append(line)
            if not line.strip():
                inside_cell_magic = False
        elif LINE_MAGIC_PATTERN.match(line):
            magics.append(line)
        elif CELL_MAGIC_PATTERN.match(line):
            magics.append(line)
            inside_cell_magic = True
        elif SHELL_COMMAND_PATTERN.match(line):
            shell_lines.append(line)
        else:
            python_lines.append(line)

    return magics, "\n".join(python_lines), shell_lines


class FunctionCallValidator(ast.NodeVisitor):
    def __init__(
        self,
        lines: list[str],
        allowed_modules: list[str] | None = None,
        blocked_modules: list[str] | None = None,
        allowed_functions: list[str] | None = None,
        blocked_functions: list[str] | None = None,
        allowed_variables: list[str] | None = None,
        blocked_variables: list[str] | None = None,
    ):
        self.lines = lines
        self.errors = []

        self.allowed_modules = allowed_modules
        self.blocked_modules = blocked_modules
        assert (
            allowed_modules is None or blocked_modules is None
        ), "Only one of allowed_modules or blocked_modules can be set."

        self.blocked_functions = blocked_functions
        self.allowed_functions = allowed_functions
        assert (
            allowed_functions is None or blocked_functions is None
        ), "Only one of allowed_functions or blocked_functions can be set."

        self.allowed_variables = allowed_variables
        self.blocked_variables = blocked_variables
        assert (
            allowed_variables is None or blocked_variables is None
        ), "Only one of allowed_variables or blocked_variables can be set."

    def _is_allowed_function_call(self, func_name: str) -> bool:
        if self.allowed_functions is not None and func_name in self.allowed_functions:
            return True
        if self.blocked_functions is not None and func_name not in self.blocked_functions:
            return False
        return True

    def visit_Call(self, node: ast.Call):
        if self.allowed_functions is None and self.blocked_functions is None:
            return

        if isinstance(node.func, ast.Name):
            function_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            function_name = node.func.attr
        else:
            raise ValueError(f"Unsupported function call: {node.func}")

        if not self._is_allowed_function_call(function_name):
            self.errors.append(
                f"Error on line {node.lineno}: {self.lines[node.lineno - 1]} "
                f"=> Function '{function_name}' is not allowed."
            )

    def _is_allowed_module_import(self, mod_name: str) -> bool:
        if self.allowed_modules is not None and mod_name in self.allowed_modules:
            return True
        if self.blocked_modules is not None and mod_name not in self.blocked_modules:
            return False
        return True

    def visit_Import(self, node: ast.Import):
        if self.allowed_modules is None and self.blocked_modules is None:
            return

        for alias in node.names:
            if "." in alias.name:
                module_name = alias.name.split(".")[0]
            else:
                module_name = alias.name

            if not self._is_allowed_module_import(module_name):
                self.errors.append(
                    f"Error on line {node.lineno}: {self.lines[node.lineno - 1]} "
                    f"=> Importing module '{module_name}' is not allowed."
                )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if self.allowed_modules is None and self.blocked_modules is None:
            return

        if "." in node.module:
            module_name = node.module.split(".")[0]
        else:
            module_name = node.module

        if not self._is_allowed_module_import(module_name):
            self.errors.append(
                f"Error on line {node.lineno}: {self.lines[node.lineno - 1]} "
                f"=> Importing from module '{module_name}' is not allowed."
            )

    def _is_allowed_variable_assignment(self, var_name: str) -> bool:
        if self.allowed_variables is not None and var_name in self.allowed_variables:
            return True
        if self.blocked_variables is not None and var_name not in self.blocked_variables:
            return False
        return True

    def visit_Assign(self, node: ast.Assign):
        if self.allowed_variables is None and self.blocked_variables is None:
            return

        for target in node.targets:
            variable_names = []
            if isinstance(target, ast.Name):
                variable_names.append(target.id)
            else:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        variable_names.append(name.id)

            for var_name in variable_names:
                if not self._is_allowed_variable_assignment(var_name):
                    self.errors.append(
                        f"Error on line {node.lineno}: {self.lines[node.lineno - 1]} "
                        f"=> Assigning to variable '{var_name}' is not allowed."
                    )


def apply_code_verification(
    code: str,
    allowed_modules: list[str] | None = None,
    blocked_modules: list[str] | None = None,
    allowed_functions: list[str] | None = None,
    blocked_functions: list[str] | None = None,
    allowed_variables: list[str] | None = None,
    blocked_variables: list[str] | None = None,
) -> list[str] | None:
    """Verify the code snippet. Return the list of errors if any."""
    errors = []

    try:
        magics, python_code, _ = seperate_code_lines(code)
        if len(magics) > 0:
            errors.append("Magic commands are not allowed.")

        tree = ast.parse(python_code)

        validator = FunctionCallValidator(
            lines=[
                line.strip()
                for line in python_code.splitlines()
                if line.strip() and not line.strip().startswith("#")
            ],
            allowed_modules=allowed_modules,
            blocked_modules=blocked_modules,
            allowed_functions=allowed_functions,
            blocked_functions=blocked_functions,
            allowed_variables=allowed_variables,
            blocked_variables=blocked_variables,
        )
        validator.visit(tree)
        errors.extend(validator.errors)
    except Exception as e:
        errors.append(f"Syntax error: {e}")

    return errors


async def code_verifier_node(state: CodeInterpreterState, config: RunnableConfig) -> dict[str, Any]:
    rounds = state.get_rounds()
    assert len(rounds) > 0, "No round found for CodeVerifier."

    last_round = rounds[-1]

    if len(last_round.posts) == 0:
        raise ValueError("No post found for CodeVerifier.")

    last_post = last_round.posts[-1]
    assert last_post.send_from == "CodeGenerator", "Last post is not from CodeGenerator."
    assert last_post.send_to == "CodeVerifier", "Last post is not to CodeVerifier."

    code = last_post.message

    errors = await asyncio.to_thread(apply_code_verification, code)

    event_handle = config["configurable"].get("event_handle", None)
    event_emitter = EventEmitter.get_instance(event_handle)
    await event_emitter.emit("cv_result", errors)

    self_correction_count = state.self_correction_count

    if len(errors) > 0:
        error_message = "\n".join(errors)
        post = Post.new(
            send_from="CodeVerifier",
            send_to="CodeGenerator",
            message=f"The code has the following errors:\n{error_message}",
            attachments=last_post.attachments + [
                Attachment.new(
                    type=AttachmentType.CODE_VERIFICATION_RESULT,
                    content=error_message,
                    extra=errors,
                )
            ],
            original_messages=last_post.original_messages,
        )
        self_correction_count = self_correction_count + 1 if self_correction_count is not None else 1
    else:
        # Code verified successfully
        post = Post.new(
            send_from="CodeVerifier",
            send_to="CodeExecutor",
            message=code,
            original_messages=last_post.original_messages,
            attachments=last_post.attachments + [
                Attachment.new(
                    type=AttachmentType.CODE_VERIFICATION_RESULT,
                    content="Code verification has been passed.",
                )
            ],
        )

    return {
        "rounds": RoundUpdate(
            id=last_round.id,
            posts=[post],
        ),
        "self_correction_count": self_correction_count,
    }

def code_verifier_router_edge(state: CodeInterpreterState) -> str:
    rounds = state.get_rounds()
    assert len(rounds) > 0, "No round found for CodeVerifier."

    last_round = rounds[-1]
    if len(last_round.posts) == 0:
        raise ValueError("No post found for CodeVerifier.")
    last_post = last_round.posts[-1]

    assert last_post.send_from == "CodeVerifier", "Last post is not from CodeVerifier."

    if last_post.send_to == "CodeGenerator":
        if state.self_correction_count is None or state.self_correction_count <= 3:
            return "code_generator_node"
        else:
            return END
    elif last_post.send_to == "CodeExecutor":
        return "code_executor_node"
    else:
        raise ValueError(f"Invalid post to: {last_post.send_to}")
