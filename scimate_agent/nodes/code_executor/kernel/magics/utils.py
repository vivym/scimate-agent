from typing import Any


def fmt_response(is_success: bool, message: str, data: Any = None):
    return {
        "is_success": is_success,
        "message": message,
        "data": data,
    }
