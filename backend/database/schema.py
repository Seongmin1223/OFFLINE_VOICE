"""
DB 스키마 초기화.

init_schema()는 connect() 내부에서 호출됨.
테이블 추가 순서:
  Tier 3: session_summaries
  Tier 2: facts
  Tier 1: turns
  cases: 장애 이력 회상 (3-Tier 대화 메모리와 별도 경로)
"""

from __future__ import annotations

import aiosqlite


async def init_schema(db: aiosqlite.Connection) -> None:
    """모든 테이블/인덱스를 CREATE IF NOT EXISTS로 초기화."""

    # Tier 3: 세션 요약
    # 과거 세션 압축본, 배치 작업에서 세션 종료 또는 idle 1시간 시 생성
    await db.execute("""
        CREATE TABLE IF NOT EXISTS session_summaries (
            session_id  TEXT     PRIMARY KEY,
            summary     TEXT     NOT NULL,
            token_count INTEGER  NOT NULL DEFAULT 0,
            started_at  DATETIME NOT NULL,
            ended_at    DATETIME
        )
    """)
    # 최신 세션부터 역순 조회 (Tier 3 조립 시 최근 N개 선택)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_summaries_started
        ON session_summaries(started_at DESC)
    """)

    # Tier 2: 장기 사용자 사실
    # 세션 경계 없이 영구 보존, 배치 작업에서 LLM이 대화 분석 후 저장
    await db.execute("""
        CREATE TABLE IF NOT EXISTS facts (
            id             INTEGER  PRIMARY KEY AUTOINCREMENT,
            category       TEXT     NOT NULL
                               CHECK(category IN ('preference','personal','event','relation')),
            content        TEXT     NOT NULL,
            importance     INTEGER  NOT NULL CHECK(importance BETWEEN 1 AND 3),
            created_at     DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            last_accessed  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
            source_session TEXT
        )
    """)
    # importance 높고 최근 접근한 fact 우선 조회
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_facts_rank
        ON facts(importance DESC, last_accessed DESC)
    """)

    # Tier 1: 대화 턴 원본
    # 현재 세션 대화 기록, 프롬프트 조립 시 최근 N턴 사용
    await db.execute("""
        CREATE TABLE IF NOT EXISTS turns (
            id          INTEGER  PRIMARY KEY AUTOINCREMENT,
            session_id  TEXT     NOT NULL,
            turn_num    INTEGER  NOT NULL,
            role        TEXT     NOT NULL CHECK(role IN ('user', 'assistant')),
            content     TEXT     NOT NULL,
            token_count INTEGER  NOT NULL DEFAULT 0,
            created_at  DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # 세션별 조회 + 최신순 정렬
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_turns_session
        ON turns(session_id, turn_num DESC)
    """)

    # cases: 장애 이력 회상 (3-Tier 대화 메모리와 별도 경로, CSV 임포터로 적재)
    await db.execute("""
        CREATE TABLE IF NOT EXISTS cases (
            case_id     TEXT     PRIMARY KEY,
            scenario    TEXT,
            case_date   TEXT     NOT NULL,
            solution    TEXT     NOT NULL,
            assignee    TEXT     NOT NULL,
            work_type   TEXT,
            work_status TEXT,
            description TEXT     NOT NULL
        )
    """)
    # 솔루션/담당자 필터 검색 + 동일 담당자 횡단 패턴 탐지용
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_cases_solution
        ON cases(solution)
    """)
    await db.execute("""
        CREATE INDEX IF NOT EXISTS idx_cases_assignee
        ON cases(assignee, case_date)
    """)

    await db.commit()
