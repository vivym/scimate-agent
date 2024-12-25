from datetime import datetime


def get_env_context() -> str:
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return f"- Current time: {current_time}\n"
