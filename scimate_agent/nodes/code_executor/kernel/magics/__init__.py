import os

from IPython.core.interactiveshell import InteractiveShell

from scimate_agent.nodes.code_executor.runtime import Executor
from .ctx_magic import SciMateContextMagic
from .plugin_magic import SciMatePluginMagic


def load_ipython_extension(ipython: InteractiveShell):
    env_id = os.environ.get("SCIMATE_ENV_ID", "local")
    session_id = os.environ.get("SCIMATE_SESSION_ID", "session_temp")
    session_dir = os.environ.get(
        "SCIMATE_SESSION_DIR",
        os.path.realpath(os.getcwd()),
    )

    executor = Executor(
        env_id=env_id,
        session_id=session_id,
        session_dir=session_dir,
    )

    ctx_magic = SciMateContextMagic(ipython, executor)
    plugin_magic = SciMatePluginMagic(ipython, executor)

    print("Loaded magics")
    print("*" * 80)

    ipython.register_magics(ctx_magic)
    ipython.register_magics(plugin_magic)
    ipython.InteractiveTB.set_mode(mode="Plain")
