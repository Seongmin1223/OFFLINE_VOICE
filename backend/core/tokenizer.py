"""LLM 토큰 수 근사 — 한국어 기준 2자 ≈ 1토큰, 보수적 추정.

HTTP 호출 없는 in-memory 계산. 정확도보다 응답속도 우선이라 길이 기반 근사 사용.
정확한 토큰 수가 필요하면 transformers AutoTokenizer 같은 in-memory 토크나이저로 교체.
"""

from __future__ import annotations


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, len(text) // 2)
