import fnmatch
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

Callback = Callable[[str, Any], Awaitable[None]]

_INSTANCES: dict[str, "EventEmitter"] = {}

_DUMMY_EVENT_EMITTER = None


@dataclass
class Listener:
    event_name: str
    callback: Callback
    once: bool = False


class EventEmitter:
    def __init__(self):
        self.listeners: list[Listener] = []

    @classmethod
    def get_instance(cls, handle: str | None) -> "EventEmitter":
        if handle is None:
            global _DUMMY_EVENT_EMITTER
            if _DUMMY_EVENT_EMITTER is None:
                _DUMMY_EVENT_EMITTER = EventEmitter()
            return _DUMMY_EVENT_EMITTER

        assert handle, "handle can not be empty."

        if handle not in _INSTANCES:
            _INSTANCES[handle] = EventEmitter()
        return _INSTANCES[handle]

    @classmethod
    def remove_instance(cls, handle: str | None):
        if handle in _INSTANCES:
            del _INSTANCES[handle]

    def on(self, name: str, callback: Callback):
        assert name, "event name can not be empty."
        self.listeners.append(
            Listener(
                event_name=name,
                callback=callback,
                once=False,
            )
        )

    def once(self, name: str, callback: Callback):
        assert name, "event name can not be empty."
        self.listeners.append(
            Listener(
                event_name=name,
                callback=callback,
                once=True,
            )
        )

    async def emit(self, name: str, data: Any):
        assert name, "event name can not be empty."

        for c in ["*", "?"]:
            assert c not in name, f"event name can not contain wildcard: {c}."

        listeners = []
        for listener in self.listeners:
            if fnmatch.fnmatch(name, listener.event_name):
                await listener.callback(name, data)
                if not listener.once:
                    listeners.append(listener)

        self.listeners = listeners
