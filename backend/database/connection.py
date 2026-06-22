"""
aiosqlite 연결 관리.

connect()   - 앱 시작 시 호출
disconnect()- 앱 종료 시 호출
get_db()    - 연결 객체 반환 (리포지터리에서 사용)
get_session_id() - 현재 세션 UUID 반환
"""

from __future__ import annotations

import uuid
import logging
import aiosqlite
from pathlib import Path

from config import config

logger = logging.getLogger(__name__)

_db: aiosqlite.Connection | None = None
_session_id: str = ""


def get_session_id() -> str:
    return _session_id


async def get_db() -> aiosqlite.Connection:
    if _db is None:
        raise RuntimeError("DB 미초기화. connect() 먼저 호출 필요")
    return _db


async def connect() -> None:
    """DB 열기 + 스키마 초기화 + 세션 ID 생성. lifespan 또는 CLI에서 1회 호출."""
    global _db, _session_id

    _session_id = str(uuid.uuid4())
    logger.info(f"세션 시작: {_session_id}")

    db_path = Path(config.DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    _db = await aiosqlite.connect(str(db_path))
    _db.row_factory = aiosqlite.Row

    # WAL: 읽기/쓰기 동시성 향상, 배치 작업과 대화 처리 충돌 방지
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.execute("PRAGMA foreign_keys=ON")

    from database.schema import init_schema
    await init_schema(_db)

    logger.info(f"DB 연결: {db_path}")


async def disconnect() -> None:
    """연결 종료. lifespan 또는 CLI 종료 시 호출."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("DB 연결 종료")
