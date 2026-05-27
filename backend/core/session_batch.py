"""세션 종료 배치 — facts 추출 + 세션 요약 생성."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from core.tokenizer import count_tokens
from database.connection import get_db
import database.repository as repo
from domains.llm.llama_engine import LlamaEngine
from domains.soul.memory import MemoryManager

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Facts 추출 ───────────────────────────────────────────────────────────

_EXTRACTION_PROMPT = """\
당신은 대화 기록에서 사용자 정보를 추출하는 분석 도구입니다.
역할 연기나 인사말 없이 아래 기준에 따라 JSON만 출력하세요.

추출 기준:
- "사용자:" 로 시작하는 발화에서만 정보 추출
- "포비:" 발화는 AI 측 발화이므로 모든 정보 제외
- 추측·유추·일시적 감정(오늘 피곤함, 지금 배고픔 등)은 제외
- 대화에서 반복되거나 사용자가 강조한 정보를 우선시

카테고리:
- preference : 좋아하는 것/싫어하는 것 (음식, 음악, 활동, 취미 등)
- personal   : 이름, 나이, 직업, 거주지 등 개인 신상 정보
- event      : 사용자가 언급한 과거 경험 또는 예정된 중요 사건
- relation   : 가족, 친구, 연인, 반려동물 등 인간관계

importance 기준:
- 1 : 가벼운 취향 (예: 아이스 아메리카노를 즐겨 마심, 고양이를 좋아함)
- 2 : 중요한 개인 정보 (예: 직업이 개발자, 서울 거주, 고양이를 키움)
- 3 : 핵심 정체성 정보 (예: 이름, 나이, 가족 구성원)

주의사항:
- 동일한 사실을 중복 추출하지 말 것
- 확실하지 않은 정보는 포함하지 말 것
- AI에 대한 정보가 아니라 사용자에 대한 정보만 추출

응답은 반드시 순수 JSON 배열만 출력하세요. 설명·마크다운·코드블록 없이.
추출할 사실이 없으면 [] 를 출력하세요.

출력 형식:
[
  {{"category": "preference", "content": "매운 음식을 싫어함", "importance": 1}},
  {{"category": "personal",   "content": "이름은 김민준",      "importance": 3}}
]

대화 기록:
{conversation}"""


_SUMMARY_PROMPT = """\
아래는 사용자와 AI(포비)의 대화 기록입니다.
이 대화를 1~2문장으로 짧게 요약해주세요.

요약 기준:
- 대화의 핵심 화제와 분위기를 압축
- 새로 알게 된 사용자 정보나 의미 있는 사건 위주로 서술
- 추측·확장·감상 없이 실제 대화 내용만 정리
- "사용자가 ~했다" 형식으로 객관적으로 작성

응답은 요약 문장만 출력하세요. 설명·따옴표·번호·머리글 없이.

대화 기록:
{conversation}"""


def _format_turns(turns: list[dict]) -> str:
    lines = []
    for t in turns:
        label = "사용자" if t["role"] == "user" else "포비"
        lines.append(f"{label}: {t['content']}")
    return "\n".join(lines)


_IMPORTANCE_KO = {
    "낮음": 1, "보통": 2, "중간": 2, "높음": 3, "매우 높음": 3, "매우높음": 3,
}


def _coerce_importance(v) -> int | None:
    """LLM이 정수 대신 문자열·실수·한국어 단어로 줘도 1~3으로 흡수."""
    if isinstance(v, bool):
        return None  # bool은 int 서브클래스라 명시 차단
    if isinstance(v, int) and 1 <= v <= 3:
        return v
    if isinstance(v, float) and v == int(v):
        n = int(v)
        return n if 1 <= n <= 3 else None
    if isinstance(v, str):
        s = v.strip()
        if s.isdigit():
            n = int(s)
            return n if 1 <= n <= 3 else None
        return _IMPORTANCE_KO.get(s)
    return None


def _parse_facts(response: str) -> list[dict]:
    match = re.search(r"\[.*\]", response, re.DOTALL)
    if not match:
        return []
    try:
        raw_facts = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    valid: list[dict] = []
    for f in raw_facts:
        if not isinstance(f, dict):
            continue
        cat = f.get("category")
        if cat not in ("preference", "personal", "event", "relation"):
            continue
        content = f.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        importance = _coerce_importance(f.get("importance"))
        if importance is None:
            continue
        valid.append({
            "category":   cat,
            "content":    content.strip(),
            "importance": importance,
        })
    return valid


def _normalize(s: str) -> str:
    return s.strip().lower().replace(" ", "")


def _is_duplicate(content: str, existing: list[str]) -> bool:
    """정규화 후 exact 일치만 중복으로 처리. substring 매칭은 false positive 위험."""
    n = _normalize(content)
    if not n:
        return True
    return any(n == _normalize(ex) for ex in existing)


async def _extract_facts(llm: LlamaEngine, memory: MemoryManager, turns: list[dict]) -> None:
    """대화 → LLM → JSON facts 파싱 → 중복 필터 → DB 저장."""
    conversation = _format_turns(turns)
    messages = [{"role": "user", "content": _EXTRACTION_PROMPT.format(conversation=conversation)}]

    try:
        response = await llm.generate(messages)
    except Exception as e:
        logger.warning(f"facts 추출 실패: {e}")
        return

    facts = _parse_facts(response)
    if not facts:
        logger.info("추출된 facts 없음")
        return

    db = await get_db()

    # 카테고리별 기존 contents 캐시 (중복 검사용)
    by_cat: dict[str, list[str]] = {}
    for row in await repo.get_fact_contents(db):
        by_cat.setdefault(row["category"], []).append(row["content"])

    inserted = 0
    skipped  = 0
    try:
        for fact in facts:
            cat = fact["category"]
            if _is_duplicate(fact["content"], by_cat.get(cat, [])):
                skipped += 1
                continue
            await repo.insert_fact(
                db,
                category=cat,
                content=fact["content"],
                importance=fact["importance"],
                source_session=memory.session_id,
            )
            by_cat.setdefault(cat, []).append(fact["content"])
            inserted += 1
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.warning(f"facts 저장 실패, 롤백: {e}")
        return

    if skipped:
        logger.info(f"facts {inserted}개 저장, {skipped}개 중복 스킵")
    else:
        logger.info(f"facts {inserted}개 저장")


# ── 세션 요약 ────────────────────────────────────────────────────────────

async def _generate_summary(llm: LlamaEngine, memory: MemoryManager, turns: list[dict]) -> None:
    """대화 → 1~2문장 요약 → 토큰 수 계산 → DB 저장."""
    conversation = _format_turns(turns)
    messages = [{"role": "user", "content": _SUMMARY_PROMPT.format(conversation=conversation)}]

    try:
        response = await llm.generate(messages)
    except Exception as e:
        logger.warning(f"세션 요약 실패: {e}")
        return

    summary = response.strip()
    if len(summary) < 5:
        logger.info("세션 요약 내용 없음")
        return

    token_count = count_tokens(summary)
    started_at  = turns[0].get("created_at") or _now()

    db = await get_db()
    try:
        await repo.insert_summary(
            db,
            session_id=memory.session_id,
            summary=summary,
            token_count=token_count,
            started_at=started_at,
            ended_at=_now(),
        )
        await db.commit()
    except Exception as e:
        await db.rollback()
        logger.warning(f"세션 요약 저장 실패: {e}")
        return

    logger.info(f"세션 요약 저장 ({len(summary)}자, {token_count}토큰)")


# ── 진입점 ───────────────────────────────────────────────────────────────

async def run_session_batch(llm: LlamaEngine, memory: MemoryManager) -> None:
    """disconnect() 호출 전 실행. facts 추출 + 세션 요약을 순차 처리."""
    turns = await memory.get_recent_turns(n=200)
    if not turns:
        return

    logger.info(f"세션 종료 배치 시작 (세션 {memory.session_id[:8]}, {len(turns)}턴)")
    await _extract_facts(llm, memory, turns)
    await _generate_summary(llm, memory, turns)
