"""
Phase 6 합성 테스트 — LLM 의존 없이 메모리 시스템의 정량 지표 측정.

지표:
1. DB throughput (turns/facts/summaries insert·select 시간)
2. 누적 데이터 따른 조회 시간 변화
3. build_messages 토큰 예산 준수
4. _is_duplicate 정확도
5. 조립된 messages의 Tier별 분포

실행:
    python synthetic_test.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile
import time
from datetime import datetime, timezone

from config import config
from core.context_builder import build_messages
from core.session_batch import _is_duplicate
from core.tokenizer import count_tokens
from database.connection import connect, disconnect, get_db, get_session_id
import database.repository as repo
from domains.soul.memory import MemoryManager
from domains.soul.soul_container import SoulContainer, SoulConfig


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _db_size_kb() -> float:
    """WAL 모드라 메인 DB만 보면 0이 나오는 경우 있어 -wal/-shm 합산."""
    p = config.DB_PATH
    total = 0
    for ext in ("", "-wal", "-shm"):
        path = p + ext
        if os.path.exists(path):
            total += os.path.getsize(path)
    return total / 1024


# ── 시나리오 데이터 ──────────────────────────────────────────────────────

_SCENARIO_10 = [
    ("user",      "안녕! 나는 김민준이야."),
    ("assistant", "안녕 민준! 만나서 반가워. 오늘 어떻게 지내?"),
    ("user",      "잘 지내고 있어. 나는 24살이고 서울에 살아."),
    ("assistant", "서울 어디쯤 사는데?"),
    ("user",      "강남구야. 직장도 강남에 있어서 출퇴근이 편해."),
    ("assistant", "직장은 어디서 일해?"),
    ("user",      "스타트업에서 백엔드 개발자로 일하고 있어."),
    ("assistant", "어떤 분야의 스타트업이야?"),
    ("user",      "헬스케어 쪽이야. 운동 추천 앱 만들고 있어."),
    ("assistant", "재밌겠다! 너도 운동 좋아해?"),
]
SCENARIO_50 = _SCENARIO_10 * 5


# ── 시나리오 1: 50턴 throughput ──────────────────────────────────────────

async def scenario_1_throughput() -> None:
    print("\n" + "=" * 60)
    print("  시나리오 1: 50턴 단일 세션 throughput")
    print("=" * 60)

    memory = MemoryManager()
    size_before = _db_size_kb()

    t0 = time.perf_counter()
    for role, content in SCENARIO_50:
        await memory.add_turn(role, content)
    elapsed = time.perf_counter() - t0

    avg_ms     = elapsed * 1000 / len(SCENARIO_50)
    size_after = _db_size_kb()

    print(f"  50턴 insert 총 {elapsed*1000:.1f}ms (평균 {avg_ms:.2f}ms/턴)")
    print(f"  DB 크기: {size_before:.1f}KB → {size_after:.1f}KB (+{size_after-size_before:.1f}KB)")

    t0 = time.perf_counter()
    turns = await memory.get_recent_turns(n=20)
    elapsed = time.perf_counter() - t0
    print(f"  get_recent_turns(20) {elapsed*1000:.2f}ms ({len(turns)}턴 반환)")


# ── 시나리오 2: 누적 facts/summaries 조회 ────────────────────────────────

async def scenario_2_accumulation() -> None:
    print("\n" + "=" * 60)
    print("  시나리오 2: 누적 데이터 따른 조회 시간")
    print("=" * 60)

    db = await get_db()

    # 100 facts + 50 summaries 주입
    t0 = time.perf_counter()
    for i in range(100):
        await repo.insert_fact(
            db,
            category=("preference", "personal", "event", "relation")[i % 4],
            content=f"테스트 fact {i}: 어떤 정보가 들어있는 항목",
            importance=(i % 3) + 1,
            source_session=f"synth-{i // 10}",
        )
    for i in range(50):
        await repo.insert_summary(
            db,
            session_id=f"synth-{i}",
            summary=f"테스트 세션 {i}: 사용자가 다양한 주제로 대화함",
            token_count=20,
            started_at=_now_iso(),
            ended_at=_now_iso(),
        )
    await db.commit()
    insert_elapsed = time.perf_counter() - t0

    print(f"  facts 100 + summaries 50 insert: {insert_elapsed*1000:.1f}ms")
    print(f"  DB 크기: {_db_size_kb():.1f}KB")

    t0 = time.perf_counter()
    facts = await repo.get_facts(db, min_importance=1, limit=20)
    print(f"  get_facts(limit=20)              {(time.perf_counter()-t0)*1000:.2f}ms ({len(facts)}개)")

    t0 = time.perf_counter()
    sums = await repo.get_recent_summaries(db, limit=10)
    print(f"  get_recent_summaries(limit=10)   {(time.perf_counter()-t0)*1000:.2f}ms ({len(sums)}개)")

    t0 = time.perf_counter()
    contents = await repo.get_fact_contents(db)
    print(f"  get_fact_contents() (전체)       {(time.perf_counter()-t0)*1000:.2f}ms ({len(contents)}개)")


# ── 시나리오 3: build_messages 토큰 예산 준수 ────────────────────────────

async def scenario_3_budget_compliance() -> None:
    print("\n" + "=" * 60)
    print("  시나리오 3: build_messages 토큰 예산 준수")
    print("=" * 60)

    memory = MemoryManager()
    soul   = SoulContainer(SoulConfig.from_preset("fix_assistant"))
    sys_p  = soul.build_system_prompt()

    print(f"  {'budget':>8} | {'messages':>8} | {'추정 사용':>10} | 결과")
    print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*10}-+-----")
    for budget in [200, 500, 1000, 2000]:
        msgs = await build_messages(
            system_prompt=sys_p,
            memory=memory,
            user_text="오늘 점심 뭐 먹었어?",
            token_budget=budget,
        )
        total = sum(count_tokens(m["content"]) for m in msgs)
        ok = "✓ pass" if total <= budget else "✗ FAIL"
        print(f"  {budget:>8} | {len(msgs):>8} | {total:>10} | {ok}")


# ── 시나리오 4: _is_duplicate 정확도 ─────────────────────────────────────

def scenario_4_dedup_accuracy() -> None:
    print("\n" + "=" * 60)
    print("  시나리오 4: _is_duplicate 정확도")
    print("=" * 60)

    existing = [
        "고양이를 키운다",
        "이름은 김민준",
        "서울 강남구 거주",
        "백엔드 개발자",
    ]

    cases = [
        # (입력, 예상값, 설명)
        ("고양이를 키운다",        True,  "exact match"),
        ("고양이를  키운다",        True,  "공백 차이"),
        ("고양이를 키운다 ",        True,  "trailing space"),
        ("이름은 김민준",          True,  "exact"),
        ("서울 강남구",            False, "substring은 더 이상 중복 아님"),
        ("강남구",                 False, "substring은 더 이상 중복 아님"),
        ("부산 해운대구 거주",      False, "다른 정보"),
        ("프론트엔드 개발자",       False, "다른 정보"),
        ("황금색 고양이를 키운다",  False, "기존이 substring (새 것이 풍부)"),
        ("",                       True,  "빈 문자열 차단"),
    ]

    correct = 0
    for content, expected, desc in cases:
        actual = _is_duplicate(content, existing)
        ok = actual == expected
        correct += int(ok)
        mark = "✓" if ok else "✗"
        print(f"  {mark} {desc:30s} | 예상={expected!s:>5} 실제={actual!s:>5}")

    rate = correct / len(cases) * 100
    print(f"\n  정확도: {correct}/{len(cases)} ({rate:.0f}%)")


# ── 시나리오 5: Tier별 분포 ──────────────────────────────────────────────

async def scenario_5_tier_distribution() -> None:
    print("\n" + "=" * 60)
    print("  시나리오 5: 조립된 messages의 Tier별 분포")
    print("=" * 60)

    memory = MemoryManager()
    soul   = SoulContainer(SoulConfig.from_preset("fix_assistant"))

    msgs = await build_messages(
        system_prompt=soul.build_system_prompt(),
        memory=memory,
        user_text="요즘 어때?",
        token_budget=1500,
    )

    sys_msgs   = [m for m in msgs if m["role"] == "system"]
    hist_msgs  = msgs[len(sys_msgs):-1]   # system 이후, 마지막 user 직전
    current    = msgs[-1]

    sys_tok  = sum(count_tokens(m["content"]) for m in sys_msgs)
    hist_tok = sum(count_tokens(m["content"]) for m in hist_msgs)
    cur_tok  = count_tokens(current["content"])
    total    = sys_tok + hist_tok + cur_tok

    print(f"  Tier 2/3 주입 system 블록  : {len(sys_msgs):2d}개 — {sys_tok:4d}토큰 ({sys_tok/total*100:.1f}%)")
    print(f"  Tier 1 history 턴          : {len(hist_msgs):2d}개 — {hist_tok:4d}토큰 ({hist_tok/total*100:.1f}%)")
    print(f"  현재 user 발화             : {1:2d}개 — {cur_tok:4d}토큰 ({cur_tok/total*100:.1f}%)")
    print(f"  합계                        :          {total:4d}토큰")


# ── 메인 ─────────────────────────────────────────────────────────────────

async def main() -> None:
    # 임시 DB로 격리 — 매 실행마다 깨끗한 환경
    tmp_dir = tempfile.mkdtemp(prefix="airi_synth_")
    config.DB_PATH = os.path.join(tmp_dir, "test.db")

    print("\n" + "=" * 60)
    print("  Phase 6 합성 데이터 정량 테스트")
    print("=" * 60)
    print(f"  임시 DB: {config.DB_PATH}")

    await connect()
    print(f"  세션 ID: {get_session_id()}")

    try:
        await scenario_1_throughput()
        await scenario_2_accumulation()
        await scenario_3_budget_compliance()
        scenario_4_dedup_accuracy()
        await scenario_5_tier_distribution()
    finally:
        final_size = _db_size_kb()
        await disconnect()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        print(f"\n  최종 DB 크기: {final_size:.1f}KB")
        print("  임시 DB 정리 완료")


if __name__ == "__main__":
    asyncio.run(main())
