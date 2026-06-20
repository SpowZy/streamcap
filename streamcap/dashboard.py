"""Live metrics dashboard, served locally.

A single dark page (stat tiles + live latency sparkline + local/cloud split)
fed over a WebSocket at ~2 Hz. Runs on a daemon thread so the capture loop
stays in charge of the process lifecycle.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import threading
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse

_STATIC = Path(__file__).parent / "static" / "index.html"

Provider = Callable[[], dict]


def build_app(provider: Provider) -> FastAPI:
    app = FastAPI(title="streamcap dashboard")

    @app.get("/")
    def index() -> HTMLResponse:
        return HTMLResponse(_STATIC.read_text(encoding="utf-8"))

    @app.get("/stats.json")
    def stats() -> JSONResponse:
        return JSONResponse(provider())

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        with contextlib.suppress(WebSocketDisconnect):
            while True:
                await socket.send_text(json.dumps(provider()))
                await asyncio.sleep(0.5)

    return app


def start_dashboard(provider: Provider, host: str = "127.0.0.1", port: int = 8765) -> str:
    import uvicorn

    config = uvicorn.Config(build_app(provider), host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    threading.Thread(target=server.run, daemon=True).start()
    return f"http://{host}:{port}"
