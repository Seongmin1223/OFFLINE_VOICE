from __future__ import annotations

import re

from core.case_search import search_cases
from core.tokenizer import count_tokens
from database.connection import get_db
import database.repository as repo
from domains.soul.memory import MemoryManager


# 시스템 프롬프트에 주입되는 블록 prefix — cutoff 로직에 동일하게 반영해야 함
_SUMMARY_HEADER = "\n\n[이전 대화 요약]"
_FACT_HEADER    = "\n\n[사용자 정보]"
_SUMMARY_LIMIT   = 2
_FACT_LIMIT      = 2
_TURN_LIMIT      = 4
_CASE_LIMIT      = 3
_CASE_HEADER    = (
    "\n\n[관련 케이스] 아래는 STT 오인식 가능성을 감안해 검색한 과거 장애 케이스입니다. "
    "사용자 발화가 깨져 보여도 이 후보 목록을 근거로 의도를 추론하세요.\n"
    "케이스 번호 규칙(엄수):\n"
    "- 케이스 번호는 오직 아래 목록에서만 가져옵니다. 목록에 적힌 [case:#NNNN]를 네 자리 숫자 그대로 복사하세요.\n"
    "- 목록에 없는 번호는 한 글자도 만들지 마세요. #025처럼 자릿수를 바꾸지도 마세요.\n"
    "- 인용할 만한 케이스가 없으면 번호를 쓰지 말고 설명만 하세요. 억지로 번호를 붙이지 마세요."
)
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


def _summarize_description(description: str, max_len: int = 48) -> str:
    """장문 작업 로그에서 첫 의미 있는 줄을 증상 요약으로 사용.
    리스트 마커(-, 1., a))만 제거하고, 솔루션명 같은 영문 대문자 헤딩 줄은 건너뛴다.
    (기존엔 마커 제거 정규식이 'FIREWATCHER' 같은 영문 단어를 통째로 삼켜 요약이 비었음.)
    """
    for raw in description.splitlines():
        line = raw.strip()
        if not line:
            continue
        # 영문 대문자/숫자/기호만으로 된 헤딩 줄(예: 'FIREWATCHER')은 건너뜀
        if re.fullmatch(r"[A-Z0-9 ()_./\-]+", line):
            continue
        # 리스트 마커만 제거(마커 뒤 공백 요구 — 영문 단어 통째 삭제 방지)
        line = re.sub(r"^(?:[-*•]+|\d+[.)]|[A-Za-z][.)])\s+", "", line).strip()
        if len(line) < 3:
            continue
        if len(line) > max_len:
            line = line[:max_len].rstrip() + "..."
        return line
    return ""


def _case_line(case: dict) -> str:
    summary = _summarize_description(case["description"])
    return f"[case:{case['case_id']}] {case['case_date']} {case['solution']} - {summary}"


def _case_block(cases: list[dict]) -> str:
    if not cases:
        return ""
    lines = [_CASE_HEADER.lstrip("\n")] + [_case_line(c) for c in cases]
    return "\n".join(lines)


async def build_messages(
    system_prompt: str,
    memory: MemoryManager,
    user_text: str,
    token_budget: int,
    candidate_cases: list[dict] | None = None,
) -> list[dict]:
    """
    Tier 3 → Tier 2 → Tier 1 순서로 컨텍스트 조립.
    token_budget 초과 시 Tier 3(오래된 것), 낮은 importance facts, 오래된 턴 순 컷오프.
    """
    db = await get_db()

    # facts: 한 번에 너무 많이 주입하면 LLM이 특정 사실에 과몰입(anchoring).
    # 5세 챗봇 응답에 참고할 사용자 정보는 6개로 제한.
    summaries = await repo.get_recent_summaries(db, limit=_SUMMARY_LIMIT)
    facts     = await memory.get_facts(min_importance=1, limit=_FACT_LIMIT)
    turns     = await memory.get_recent_turns(n=_TURN_LIMIT)
    # 케이스 회상은 3-Tier 대화 메모리와 별도 경로 — pipeline에서 미리 구해 재사용 가능
    cases     = candidate_cases if candidate_cases is not None else (
        await search_cases(user_text, limit=_CASE_LIMIT) if user_text.strip() else []
    )

    # 토큰 수 일괄 계산 — len 기반 in-memory 근사라 비용 무시 가능
    # 시스템 프롬프트에 들어가는 블록 prefix(헤더, bullet)도 함께 측정
    texts = (
        [system_prompt, user_text, _SUMMARY_HEADER, _FACT_HEADER, _CASE_HEADER, _BULLET]
        + [s["summary"]   for s in summaries]
        + [f["content"]   for f in facts]
        + [_case_line(c)  for c in cases]
        + [t["content"]   for t in turns]
    )
    counts = [count_tokens(t) for t in texts]

    sys_tok, user_tok = counts[0], counts[1]
    summary_header_tok, fact_header_tok, case_header_tok, bullet_tok = counts[2], counts[3], counts[4], counts[5]
    o = 6
    summary_toks = counts[o : o + len(summaries)]; o += len(summaries)
    fact_toks    = counts[o : o + len(facts)];     o += len(facts)
    case_toks    = counts[o : o + len(cases)];     o += len(cases)
    turn_toks    = counts[o : o + len(turns)]

    used = sys_tok + user_tok

    # 현재 발화의 근거 후보이므로 장기 메모리보다 먼저 토큰 예산을 배정한다.
    selected_cases: list[dict] = []
    for i, (c, cost) in enumerate(zip(cases, case_toks)):
        item_cost = cost + (case_header_tok if i == 0 else 0)
        if used + item_cost > token_budget:
            break
        selected_cases.append(c)
        used += item_cost

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
    cb = _case_block(selected_cases)
    sb = _summary_block(selected_summaries)
    fb = _fact_block(selected_facts)
    if cb:
        full_system += f"\n\n{cb}"
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
