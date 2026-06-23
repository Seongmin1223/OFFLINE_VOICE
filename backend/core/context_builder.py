from __future__ import annotations

from core.tokenizer import count_tokens
from database.connection import get_db
import database.repository as repo
from domains.soul.memory import MemoryManager


# 시스템 프롬프트에 주입되는 블록 prefix — cutoff 로직에 동일하게 반영해야 함
_SUMMARY_HEADER = "\n\n[이전 대화 요약]"
_FACT_HEADER    = "\n\n[사용자 정보]"
_SUMMARY_LIMIT   = 2
_FACT_LIMIT      = 2
# Tier-1 최근 턴은 매 요청마다 프롬프트 중간에 들어가 prefill(→TTFT)을 키운다.
# 짧은 대화 코히런스에 필요한 최소(직전 1교환=2턴)만 유지해 첫 토큰을 빠르게.
_TURN_LIMIT      = 2
_BULLET         = "\n- "


def _summary_block(summaries: list[dict]) -> str:
    if not summaries:
        return ""
    lines = ["[이전 대화 요약]"]
    for s in summaries:
        lines.append(f"- {s['summary']}")
    return "\n".join(lines)


def _fact_block(facts: list[dict]) -> str:
    if not facts:
        return ""
    lines = ["[사용자 정보]"]
    for f in facts:
        lines.append(f"- {f['content']}")
    return "\n".join(lines)


async def build_messages(
    system_prompt: str,
    memory: MemoryManager,
    user_text: str,
    token_budget: int,
) -> list[dict]:
    """
    Tier 3(요약) → Tier 2(사실) → Tier 1(최근 턴) 순서로 컨텍스트 조립.
    token_budget 초과 시 Tier 3(오래된 것), 낮은 importance facts, 오래된 턴 순 컷오프.
    """
    db = await get_db()

    summaries = await repo.get_recent_summaries(db, limit=_SUMMARY_LIMIT)
    facts     = await memory.get_facts(min_importance=1, limit=_FACT_LIMIT)
    turns     = await memory.get_recent_turns(n=_TURN_LIMIT)

    # 토큰 수 일괄 계산 — len 기반 in-memory 근사라 비용 무시 가능
    texts = (
        [system_prompt, user_text, _SUMMARY_HEADER, _FACT_HEADER, _BULLET]
        + [s["summary"]   for s in summaries]
        + [f["content"]   for f in facts]
        + [t["content"]   for t in turns]
    )
    counts = [count_tokens(t) for t in texts]

    sys_tok, user_tok = counts[0], counts[1]
    summary_header_tok, fact_header_tok, bullet_tok = counts[2], counts[3], counts[4]
    o = 5
    summary_toks = counts[o : o + len(summaries)]; o += len(summaries)
    fact_toks    = counts[o : o + len(facts)];     o += len(facts)
    turn_toks    = counts[o : o + len(turns)]

    used = sys_tok + user_tok

    # Tier 3: 첫 항목 추가 시 헤더 비용, 모든 항목에 bullet 비용
    selected_summaries: list[dict] = []
    for i, (s, cost) in enumerate(zip(summaries, summary_toks)):
        item_cost = cost + bullet_tok + (summary_header_tok if i == 0 else 0)
        if used + item_cost > token_budget:
            break
        selected_summaries.append(s)
        used += item_cost

    # Tier 2: 동일 패턴
    selected_facts: list[dict] = []
    for i, (f, cost) in enumerate(zip(facts, fact_toks)):
        item_cost = cost + bullet_tok + (fact_header_tok if i == 0 else 0)
        if used + item_cost > token_budget:
            break
        selected_facts.append(f)
        used += item_cost

    # 결합된 system content를 먼저 만든 뒤 실제 토큰을 다시 측정
    # (fragment 합산과 결합 측정 사이의 floor 오차 누적 보정)
    full_system = system_prompt
    sb = _summary_block(selected_summaries)
    fb = _fact_block(selected_facts)
    if sb:
        full_system += f"\n\n{sb}"
    if fb:
        full_system += f"\n\n{fb}"

    actual_sys_tok = count_tokens(full_system)
    used = actual_sys_tok + user_tok

    # Tier 1: 보정된 예산 안에서 오래된 턴부터 제거
    budget_for_turns = max(0, token_budget - used)
    turn_total = sum(turn_toks)
    while turns and turn_total > budget_for_turns:
        turn_total -= turn_toks[0]
        turns      = turns[1:]
        turn_toks  = turn_toks[1:]

    messages: list[dict] = [{"role": "system", "content": full_system}]
    messages.extend({"role": t["role"], "content": t["content"]} for t in turns)
    messages.append({"role": "user", "content": user_text})

    return messages
