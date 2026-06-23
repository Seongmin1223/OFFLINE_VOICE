"""LLM 응답의 [case:#NNNN] 인용을 실제 cases 테이블과 대조 — 환각 방지.

목표: README 성능표의 H_rate(환각률) 유지/개선. 존재하지 않는 case_id는
fallback 문구로 치환한다.
"""

from __future__ import annotations

import re

_CASE_TOKEN_RE   = re.compile(r"\[case:#([^\]]+)\]")
_FALLBACK        = ""


def strip_unknown_cases(text: str, valid_ids: set[str]) -> str:
    """text 안의 [case:#NNNN] 중 valid_ids에 없는 것을 fallback으로 치환."""
    def _replace(m: re.Match) -> str:
        raw_id = m.group(1)
        if not re.fullmatch(r"\d{4}", raw_id):
            return _FALLBACK
        case_id = f"#{raw_id}"
        return m.group(0) if case_id in valid_ids else _FALLBACK
    return _CASE_TOKEN_RE.sub(_replace, text)


def strip_case_tokens(text: str) -> str:
    """TTS 음성 출력용 — [case:#NNNN] 토큰 자체를 읽지 않도록 제거."""
    text = re.sub(r"\[case:[^\]]*\]", "", text)
    text = re.sub(r"\[case:\S*", "", text)
    return text
