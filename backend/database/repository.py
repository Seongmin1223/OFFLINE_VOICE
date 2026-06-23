from __future__ import annotations

import aiosqlite
from datetime import datetime, timezone


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#Tier 1: turns

async def insert_turn(
    db: aiosqlite.Connection,
    session_id: str,
    turn_num: int,
    role: str,
    content: str,
    token_count: int = 0,
) -> None:
    await db.execute(
        """
        INSERT INTO turns (session_id, turn_num, role, content, token_count, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, turn_num, role, content, token_count, _now()),
    )


async def get_recent_turns(db: aiosqlite.Connection, session_id: str, n: int) -> list[dict]:
    """최근 n턴 반환 (오름차순)."""
    async with db.execute(
        """
        SELECT role, content, token_count, created_at
        FROM turns
        WHERE session_id = ?
        ORDER BY turn_num DESC
        LIMIT ?
        """,
        (session_id, n),
    ) as cursor:
        rows = await cursor.fetchall()
    return [dict(r) for r in reversed(rows)]


#Tier 2: facts

async def insert_fact(
    db: aiosqlite.Connection,
    category: str,
    content: str,
    importance: int,
    source_session: str | None = None,
) -> int:
    """저장된 fact의 id 반환."""
    now = _now()
    cursor = await db.execute(
        """
        INSERT INTO facts (category, content, importance, created_at, last_accessed, source_session)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (category, content, importance, now, now, source_session),
    )
    return cursor.lastrowid


async def get_facts(db: aiosqlite.Connection, min_importance: int = 1, limit: int = 20) -> list[dict]:
    """카테고리별 균형 분포 + 랜덤 타이브레이커로 anchoring 완화.

    각 카테고리에서 importance DESC + RANDOM 으로 상위 PER_CATEGORY개를 뽑은 뒤,
    합쳐서 다시 importance DESC + RANDOM 으로 최종 limit개 반환.
    last_accessed는 정렬에 더 이상 사용하지 않음
    (같은 fact만 계속 선택→touch→재선택되는 self-anchoring 루프 차단).
    """
    PER_CATEGORY = 3
    async with db.execute(
        """
        WITH ranked AS (
            SELECT id, category, content, importance, last_accessed,
                   ROW_NUMBER() OVER (
                       PARTITION BY category
                       ORDER BY importance DESC, RANDOM()
                   ) AS rn
            FROM facts
            WHERE importance >= ?
        )
        SELECT id, category, content, importance, last_accessed
        FROM ranked
        WHERE rn <= ?
        ORDER BY importance DESC, RANDOM()
        LIMIT ?
        """,
        (min_importance, PER_CATEGORY, limit),
    ) as cursor:
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


async def touch_facts(db: aiosqlite.Connection, ids: list[int]) -> None:
    """last_accessed 갱신 — LRU."""
    if not ids:
        return
    placeholders = ",".join("?" * len(ids))
    await db.execute(
        f"UPDATE facts SET last_accessed = ? WHERE id IN ({placeholders})",
        [_now(), *ids],
    )


async def get_fact_contents(db: aiosqlite.Connection) -> list[dict]:
    """전체 facts의 category + content 반환 — 중복 검사용."""
    async with db.execute("SELECT category, content FROM facts") as cursor:
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]


# Tier 3: session_summaries

async def insert_summary(
    db: aiosqlite.Connection,
    session_id: str,
    summary: str,
    token_count: int,
    started_at: str,
    ended_at: str | None = None,
) -> None:
    await db.execute(
        """
        INSERT OR REPLACE INTO session_summaries
            (session_id, summary, token_count, started_at, ended_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (session_id, summary, token_count, started_at, ended_at),
    )


async def get_recent_summaries(db: aiosqlite.Connection, limit: int = 5) -> list[dict]:
    """최신 세션 요약 반환 (최신순)."""
    async with db.execute(
        """
        SELECT session_id, summary, token_count, started_at, ended_at
        FROM session_summaries
        ORDER BY started_at DESC
        LIMIT ?
        """,
        (limit,),
    ) as cursor:
        rows = await cursor.fetchall()
    return [dict(r) for r in rows]
