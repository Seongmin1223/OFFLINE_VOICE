from fastapi import WebSocket, WebSocketDisconnect
from typing import Set
import json


_clients: Set[WebSocket] = set()


async def avatar_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    print("[Avatar WS] 브라우저 연결됨")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("[Avatar WS] 브라우저 연결 끊김")
    finally:
        _clients.discard(websocket)


async def broadcast(message: dict) -> None:
    if not _clients:
        return
    data = json.dumps(message, ensure_ascii=False)
    disconnected = set()
    for ws in _clients:
        try:
            await ws.send_text(data)
        except Exception:
            disconnected.add(ws)
    _clients.difference_update(disconnected)
