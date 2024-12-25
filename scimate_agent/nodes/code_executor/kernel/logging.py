import logging
import os

logging.basicConfig(
    filename=os.environ.get("SCIMATE_LOGGING_FILE_PATH", "kernel-runtime.log"),
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

logger = logging.getLogger(__name__)
