import asyncio
from datetime import datetime
from database.connection import connect, disconnect, get_db, get_session_id


# ── Phase 1: 스키마 + 직접 SQL ─────────────────────────────────────────────

async def run_phase1_test():
    print("=" * 50)
    print("  Phase 1 테스트 (직접 SQL)")
    print("=" * 50)

    print("\n1. DB 연결 및 스키마 초기화")
    await connect()

    current_session = get_session_id()
    print(f"세션 ID: {current_session}")

    db = await get_db()

    print("\n각 Tier별 더미 데이터 삽입")
    try:
        await db.execute(
            "INSERT INTO turns (session_id, turn_num, role, content, token_count) VALUES (?, ?, ?, ?, ?)",
            (current_session, 1, "user", "안녕! 나는 티라노사우루스가 좋아.", 15)
        )
        await db.execute(
            "INSERT INTO turns (session_id, turn_num, role, content, token_count) VALUES (?, ?, ?, ?, ?)",
            (current_session, 2, "assistant", "안녕! 티라노사우루스를 좋아하는구나. 고기가 먹고 싶니?", 20)
        )
        await db.execute(
            "INSERT INTO facts (category, content, importance, source_session) VALUES (?, ?, ?, ?)",
            ("preference", "공룡 중 티라노사우루스를 가장 좋아함", 3, current_session)
        )
        now_str = datetime.now().isoformat()
        await db.execute(
            "INSERT INTO session_summaries (session_id, summary, token_count, started_at, ended_at) VALUES (?, ?, ?, ?, ?)",
            (current_session, "아이가 공룡(티라노사우루스)을 좋아한다고 말하며 첫인사를 나눔.", 35, now_str, now_str)
        )
        await db.commit()
        print("데이터 삽입 완료!")
    except Exception as e:
        print(f"데이터 삽입 중 에러 발생: {e}")

    print("\n삽입된 데이터 조회 테스트")
    print("--- [Tier 1] Turns ---")
    async with db.execute("SELECT * FROM turns ORDER BY turn_num ASC") as cursor:
        async for row in cursor:
            print(dict(row))

    print("\n--- [Tier 2] Facts ---")
    async with db.execute("SELECT * FROM facts") as cursor:
        async for row in cursor:
            print(dict(row))

    print("\n--- [Tier 3] Session Summaries ---")
    async with db.execute("SELECT * FROM session_summaries") as cursor:
        async for row in cursor:
            print(dict(row))

    print("\nDB 연결 정상 종료...")
    await disconnect()


# ── Phase 2: Repository / MemoryManager / context_builder ─────────────────

async def run_phase2_test():
    print("\n" + "=" * 50)
    print("  Phase 2 테스트")
    print("=" * 50)

    await connect()
    db      = await get_db()
    session = get_session_id()

    # 1. Repository 직접 호출
    import database.repository as repo

    print("\n1. Repository 레이어")

    # Tier 1: turns
    await repo.insert_turn(db, session, 1, "user",      "나는 고양이를 키워.",          10)
    await repo.insert_turn(db, session, 2, "assistant", "귀엽겠다! 이름이 뭐야?",       8)
    await repo.insert_turn(db, session, 3, "user",      "치즈야. 3살이고 삼색이야.",     12)
    await db.commit()

    turns = await repo.get_recent_turns(db, session, n=5)
    print(f"  turns {len(turns)}개:")
    for t in turns:
        print(f"    [{t['role']}] {t['content']}")

    # Tier 2: facts
    await repo.insert_fact(db, "personal",   "고양이 이름은 치즈, 3살 삼색",  2, session)
    await repo.insert_fact(db, "preference", "삼색고양이를 좋아함",           1, session)
    await db.commit()

    facts = await repo.get_facts(db, min_importance=1, limit=10)
    print(f"\n  facts {len(facts)}개:")
    for f in facts:
        print(f"    [{f['category']}] {f['content']}  (importance={f['importance']})")

    # touch_facts — last_accessed 갱신
    await repo.touch_facts(db, [f["id"] for f in facts])
    await db.commit()
    print("\n  touch_facts 완료")

    # Tier 3: session_summaries
    now = datetime.now().isoformat()
    await repo.insert_summary(
        db, session,
        "사용자가 치즈라는 3살 삼색 고양이를 키운다고 말함.",
        20, now, now,
    )
    await db.commit()

    summaries = await repo.get_recent_summaries(db, limit=5)
    print(f"\n  summaries {len(summaries)}개:")
    for s in summaries:
        print(f"    {s['summary']}")

    # ── 2. MemoryManager ──────────────────────────────────────────────────
    from domains.soul.memory import MemoryManager

    print("\n2. MemoryManager")
    memory = MemoryManager()

    await memory.add_turn("user",      "제일 좋아하는 음식은 라멘이야.")
    await memory.add_turn("assistant", "라멘 맛있지~! 어떤 종류를 좋아해?")
    await memory.add_turn("user",      "진한 돈코츠 라멘 최고!")

    recent = await memory.get_recent_turns(n=3)
    print(f"  get_recent_turns {len(recent)}개:")
    for t in recent:
        print(f"    [{t['role']}] {t['content']}")

    facts_via_mgr = await memory.get_facts(min_importance=1, limit=5)
    print(f"\n  get_facts {len(facts_via_mgr)}개 (touch_facts 자동 호출됨)")

    # ── 3. build_messages ────────────────────────────────────────────────
    from core.context_builder import build_messages
    from domains.soul.soul_container import SoulContainer, SoulConfig

    print("\n3. build_messages (token_budget=400)")
    soul = SoulContainer(SoulConfig.from_preset("pobi"))

    messages = await build_messages(
        system_prompt=soul.build_system_prompt(),
        memory=memory,
        user_text="오늘 뭐 먹었어?",
        token_budget=400,
    )
    print(f"  조립된 messages {len(messages)}개:")
    for m in messages:
        preview = m["content"][:60].replace("\n", " ")
        print(f"    [{m['role']}] {preview}{'...' if len(m['content']) > 60 else ''}")

    # ── 4. _parse_facts 파싱 단위 테스트 ─────────────────────────────────
    from core.session_batch import _parse_facts

    print("\n4. _parse_facts 파싱 테스트")
    cases = [
        # 정상 케이스
        '[{"category": "preference", "content": "고양이를 좋아함", "importance": 1}]',
        # LLM이 앞뒤에 텍스트를 붙이는 케이스
        '추출 결과입니다:\n[{"category": "personal", "content": "이름은 민준", "importance": 3}]',
        # 잘못된 카테고리가 섞인 케이스 → 유효한 것만 통과
        '[{"category": "wrong", "content": "xxx", "importance": 1}, {"category": "personal", "content": "나이 25세", "importance": 2}]',
        # 빈 배열
        '[]',
        # JSON이 아예 없는 케이스
        '추출할 사실이 없습니다.',
    ]
    for i, raw in enumerate(cases, 1):
        result = _parse_facts(raw)
        print(f"  case {i}: {result}")

    await disconnect()
    print("\nPhase 2 테스트 완료!")


if __name__ == "__main__":
    asyncio.run(run_phase2_test())
