from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

import socketio
import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scimate_agent.utils.logging import setup_logging
from .middlewares import CorrelationMiddleware
from .websocket import SciMateAgentWebsocketHandler

if TYPE_CHECKING:
    from structlog.stdlib import BoundLogger

logger: "BoundLogger" = structlog.get_logger()

websocket_handler: SciMateAgentWebsocketHandler | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()

    global websocket_handler
    websocket_handler = SciMateAgentWebsocketHandler(sio)
    yield
    await logger.ainfo("Shutting down...")
    await websocket_handler.stop()


app = FastAPI(lifespan=lifespan)

app.add_middleware(CorrelationMiddleware)
app.add_middleware(
    CORSMiddleware,
    # TODO: add restrictions
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
combined_asgi_app = socketio.ASGIApp(sio, app)


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@sio.event
async def connect(sid, environ, auth):
    await logger.adebug("connected", auth=auth, sid=sid)

    await websocket_handler.on_connect(sid=sid, environ=environ, auth=auth)


@sio.event
async def disconnect(sid):
    await logger.adebug("disconnected", sid=sid)

    await websocket_handler.on_disconnect(sid=sid)


@sio.event
async def user_query(sid, data):
    await logger.adebug("user_query", sid=sid, data=data)

    await websocket_handler.on_user_query(sid=sid, user_query=data)
