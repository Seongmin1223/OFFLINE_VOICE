from fastapi import WebSocket, WebSocketDisconnect
from typing import Set
import json


_clients: Set[WebSocket] = set()

# 시스템 상태 — 브라우저가 언제 접속해도 현재 상태를 받을 수 있도록 마지막 상태를 보관.
# 'ready'는 두 조건이 모두 충족돼야 한다: (1) 모델 워밍업 완료, (2) 마이크 사용 가능.
# 기본값은 'warming' — 명시적으로 ready가 될 때까지 프론트엔드 로딩 화면 유지.
_status: dict = {"type": "status", "state": "warming", "message": "시스템 가동 중..."}
_models_ready: bool = False
_mic_ready: bool = True  # mic 모드에서 마이크 미연결 시 mic_loop이 False로 내린다


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
    """워밍업 진행 메시지 등 임시 상태 갱신 + 브로드캐스트.
    최종 ready 판정은 set_models_ready/set_mic_ready를 통한 게이트로만 한다."""
    _status["state"] = state
    _status["message"] = message
    await broadcast(dict(_status))


async def _recompute_status() -> None:
    """모델·마이크 준비 상태를 합쳐 최종 status를 결정·브로드캐스트."""
    if _models_ready and _mic_ready:
        _status["state"] = "ready"
        _status["message"] = "준비 완료"
    elif not _mic_ready:
        _status["state"] = "no_mic"
        _status["message"] = "마이크 연결을 기다리는 중..."
    else:
        _status["state"] = "warming"
        if not _status.get("message"):
            _status["message"] = "시스템 가동 중..."
    await broadcast(dict(_status))


async def set_models_ready(value: bool = True) -> None:
    global _models_ready
    _models_ready = value
    await _recompute_status()


async def set_mic_ready(value: bool) -> None:
    global _mic_ready
    _mic_ready = value
    await _recompute_status()


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
