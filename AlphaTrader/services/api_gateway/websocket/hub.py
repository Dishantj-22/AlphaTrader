"""
WebSocket hub: pushes live signals and tick updates to subscribed clients.
"""

import asyncio
import json
import logging
from typing import Optional
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
from shared.config.settings import get_settings
from ..auth.jwt_handler import decode_token

log = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter()


class ConnectionManager:
    """Manages active WebSocket connections keyed by user_id."""

    def __init__(self):
        self._connections: dict[str, list[WebSocket]] = {}
        self._symbol_subs: dict[str, set[str]] = {}

    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        await websocket.accept()
        self._connections.setdefault(user_id, []).append(websocket)
        log.info("WS connected | user=%s total=%d", user_id, sum(len(v) for v in self._connections.values()))

    def disconnect(self, websocket: WebSocket, user_id: str) -> None:
        conns = self._connections.get(user_id, [])
        if websocket in conns:
            conns.remove(websocket)
        log.info("WS disconnected | user=%s", user_id)

    def subscribe_symbol(self, user_id: str, symbol: str) -> None:
        self._symbol_subs.setdefault(user_id, set()).add(symbol.upper())

    async def broadcast_signal(self, signal_dict: dict) -> None:
        symbol = signal_dict.get("symbol", "")
        payload = json.dumps({"type": "SIGNAL", "data": signal_dict})
        for user_id, sockets in list(self._connections.items()):
            if symbol in self._symbol_subs.get(user_id, set()) or not self._symbol_subs.get(user_id):
                for ws in list(sockets):
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        sockets.remove(ws)

    async def broadcast_tick(self, tick_dict: dict) -> None:
        symbol = tick_dict.get("symbol", "")
        payload = json.dumps({"type": "TICK", "data": tick_dict})
        for user_id, sockets in list(self._connections.items()):
            if symbol in self._symbol_subs.get(user_id, set()):
                for ws in list(sockets):
                    try:
                        await ws.send_text(payload)
                    except Exception:
                        sockets.remove(ws)


manager = ConnectionManager()


@router.websocket("/signals")
async def websocket_signals(
    websocket: WebSocket,
    token: Optional[str] = Query(None),
):
    if not token:
        await websocket.close(code=4001)
        return
    try:
        payload = decode_token(token)
        user_id = payload["sub"]
    except Exception:
        await websocket.close(code=4001)
        return

    await manager.connect(websocket, user_id)
    try:
        while True:
            msg = await websocket.receive_text()
            try:
                data = json.loads(msg)
                if data.get("action") == "subscribe" and data.get("symbol"):
                    manager.subscribe_symbol(user_id, data["symbol"])
                    await websocket.send_text(json.dumps({"type": "SUBSCRIBED", "symbol": data["symbol"]}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket, user_id)
