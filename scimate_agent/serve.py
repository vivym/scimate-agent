from contextlib import asynccontextmanager

import socketio
from fastapi import FastAPI

from .app import SciMateAgentApp


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_app
    agent_app = SciMateAgentApp(sio)
    yield
    await agent_app.stop()
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

sio = socketio.AsyncServer(async_mode="asgi", cors_allowed_origins="*")
combined_asgi_app = socketio.ASGIApp(sio, app)

agent_app = None


@app.get("/")
def read_root():
    return {"message": "Hello World"}


@sio.event
async def connect(sid, environ, auth):
    # TODO: use logger
    print(f"connected auth={auth} sid={sid}")

    await agent_app.on_connect(sid=sid, environ=environ, auth=auth)


@sio.event
async def disconnect(sid):
    # TODO: use logger
    print(f"disconnected sid={sid}")

    await agent_app.on_disconnect(sid=sid)


@sio.event
async def user_query(sid, data):
    # TODO: use logger
    print(f"user_query sid={sid} data={data}")

    await agent_app.on_user_query(sid=sid, user_query=data)
