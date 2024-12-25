import os
import sys

from .logging import logger


def start_kernel_app():
    from ipykernel.displayhook import ZMQShellDisplayHook
    from ipykernel.kernelapp import IPKernelApp
    from ipykernel.zmqshell import ZMQInteractiveShell

    ZMQInteractiveShell.displayhook_class = ZMQShellDisplayHook

    app = IPKernelApp.instance()
    app.name = "scimate_kernel"
    app.config_file_name = os.path.join(os.path.dirname(__file__), "config.py")
    app.extensions = ["scimate_agent.nodes.code_executor.kernel.magics"]
    app.language = "python"

    logger.info("Initializing kernel app...")
    app.initialize()

    logger.info("Starting kernel app...")
    app.start()


if __name__ == "__main__":
    if sys.path[0] == "":
        del sys.path[0]

    logger.info("Starting process...")
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Current sys.path: {sys.path}")
    start_kernel_app()
