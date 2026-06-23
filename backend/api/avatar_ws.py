from fastapi import WebSocket, WebSocketDisconnect
from typing import Set
import json


_clients: Set[WebSocket] = set()

# warmup 등 시스템 상태 — 브라우저가 워밍업 도중/이후 언제 접속해도 현재 상태를
# 받을 수 있도록 마지막 상태를 보관한다(워밍업은 브라우저 연결 전에 시작될 수 있음).
_status: dict = {"type": "status", "state": "ready", "message": ""}


async def avatar_websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    _clients.add(websocket)
    print("[Avatar WS] 브라우저 연결됨")
    # 접속 직후 현재 시스템 상태를 즉시 전송 (워밍업 중이면 "시스템 가동 중" 표시됨)
    try:
        await websocket.send_text(json.dumps(_status, ensure_ascii=False))
    except Exception:
        pass
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        print("[Avatar WS] 브라우저 연결 끊김")
    finally:
        _clients.discard(websocket)


async def set_status(state: str, message: str) -> None:
    """시스템 상태 갱신 + 전체 브로드캐스트. state: 'warming' | 'ready'."""
    _status["state"] = state
    _status["message"] = message
    await broadcast(dict(_status))


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
