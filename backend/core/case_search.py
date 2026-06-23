"""장애 이력 케이스 검색 — 3-Tier 대화 메모리와 별도 경로.

임베딩 없이 키워드 중심으로 동작:
1차: 단어 + 글자 2-gram 겹침에 코퍼스 내 IDF(역문서빈도)를 곱해 점수 산정.
     "서버"·"오류"·"수정"처럼 거의 모든 케이스에 나오는 흔한 단어는 거의 0점,
     "인천교통공사"·"Map2D"처럼 소수 케이스에만 나오는 단어는 고득점 —
     장문 케이스가 흔한 단어 누적만으로 짧은 케이스를 이기는 길이 편향을 막음
     (raw count로 더했을 때 실제 테스트에서 이 편향이 확인됨).
2차: 1차 적중이 희소할 때만(임계 미달) 같은 담당자+솔루션 케이스를 날짜
     인접도로 보강 — "그 전후로 진행된 조치 기록 더 있어?" 같은 후속 질문 대응.
"""

from __future__ import annotations

import math
import re
from datetime import datetime

from database.connection import get_db
import database.repository as repo

_CASE_REF_RE = re.compile(r"#(\d{4})")
_TOKEN_RE    = re.compile(r"[가-힣]+|[a-zA-Z0-9_.]+")

_MIN_SCORE          = 8.0   # 이 미만은 흔한 단어 하나로 인한 우연한 겹침으로 보고 버림
_SPARSE_THRESHOLD   = 2     # 1차 적중이 이 미만일 때만 날짜 인접 보강 시도
_EXPAND_WINDOW_DAYS = 60
_WORD_WEIGHT        = 2.0
_BIGRAM_WEIGHT      = 1.0

# 유지보수 로그 전반에 거의 항상 나오는 범용어 — 13건처럼 코퍼스가 작으면
# IDF만으로는 충분히 깎이지 않아 "서버"/"오류" 한 단어 매칭이 무관한 케이스를
# 끌어올리는 문제가 실측에서 확인됨. 696건 도착 후 코퍼스가 커지면 IDF가
# 자연히 이 단어들을 낮게 평가하게 되므로 이 목록의 영향력은 줄어든다.
_STOPWORDS = {
    "서버", "오류", "수정", "테스트", "진행", "완료", "문제", "확인", "설정",
    "관련", "작업", "실행", "발생", "처리", "대응", "이슈", "현상", "기능",
    "적용", "변경", "등록", "사례", "결과", "분석", "점검", "있어", "있나",
    "했어", "그냥", "지금", "비슷한", "과거에",
    # 유지보수 로그 어디에나 나오는 범용어 — 단일 매칭으로 무관 케이스를
    # 끌어올리는 게 실측(#0458이 03·04·05에 노이즈로 끼어듦)에서 확인됨
    "기록", "문서화", "내용", "히스토리", "특이사항", "참고",
}


def _word_tokens(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) >= 2}


def _query_word_tokens(text: str) -> set[str]:
    return _word_tokens(text) - _STOPWORDS


def _char_bigrams(text: str) -> set[str]:
    stripped = re.sub(r"\s+", "", text)
    return {stripped[i:i + 2] for i in range(len(stripped) - 1)}


def _case_text(case: dict) -> str:
    return f"{case['solution']} {case['assignee']} {case['description']}"


def _build_doc_freq(cases: list[dict]) -> tuple[dict[str, int], dict[str, int]]:
    """코퍼스 전체에서 단어/2-gram이 등장하는 케이스 수(document frequency)."""
    word_df: dict[str, int] = {}
    bigram_df: dict[str, int] = {}
    for c in cases:
        text = _case_text(c)
        for w in _word_tokens(text):
            word_df[w] = word_df.get(w, 0) + 1
        for b in _char_bigrams(text):
            bigram_df[b] = bigram_df.get(b, 0) + 1
    return word_df, bigram_df


def _idf(freq: int, n_docs: int) -> float:
    return math.log((n_docs + 1) / (freq + 1)) + 1


def _score(
    query_words: set[str],
    query_bigrams: set[str],
    case: dict,
    word_df: dict[str, int],
    bigram_df: dict[str, int],
    n_docs: int,
) -> float:
    text = _case_text(case)
    matched_words   = query_words & _word_tokens(text)
    matched_bigrams = query_bigrams & _char_bigrams(text)

    word_score   = sum(_idf(word_df.get(w, 1), n_docs) for w in matched_words)
    bigram_score = sum(_idf(bigram_df.get(b, 1), n_docs) for b in matched_bigrams)
    return word_score * _WORD_WEIGHT + bigram_score * _BIGRAM_WEIGHT


def _parse_date(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except (ValueError, TypeError):
        return None


def _siblings_by_date(cases: list[dict], anchor: dict, exclude_ids: set[str]) -> list[dict]:
    """anchor와 같은 담당자+솔루션인 케이스를 날짜 인접도로 정렬해 반환."""
    anchor_date = _parse_date(anchor["case_date"])

    def _proximity(c: dict) -> float:
        d = _parse_date(c["case_date"])
        if anchor_date is None or d is None:
            return float("inf")
        return abs((d - anchor_date).days)

    siblings = [
        c for c in cases
        if c["case_id"] not in exclude_ids
        and c["assignee"] == anchor["assignee"]
        and c["solution"] == anchor["solution"]
        and _proximity(c) <= _EXPAND_WINDOW_DAYS
    ]
    siblings.sort(key=_proximity)
    return siblings


async def search_cases(query: str, limit: int = 5) -> list[dict]:
    """쿼리와 관련된 케이스 top-N. 결과 없으면 빈 리스트."""
    db = await get_db()
    cases = await repo.get_all_cases(db)
    if not cases:
        return []

    word_df, bigram_df = _build_doc_freq(cases)
    n_docs = len(cases)

    query_words   = _query_word_tokens(query)
    query_bigrams = _char_bigrams(query)

    scored = [(c, _score(query_words, query_bigrams, c, word_df, bigram_df, n_docs)) for c in cases]
    scored = [(c, s) for c, s in scored if s >= _MIN_SCORE]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    primary = [c for c, _ in scored[:limit]]

    if len(primary) >= _SPARSE_THRESHOLD:
        return primary

    # 1차 적중이 희소할 때만 보강. 쿼리에 #NNNN이 명시되면 그 케이스를 앵커로 우선.
    refs = _CASE_REF_RE.findall(query)
    anchor = None
    if refs:
        anchor = next((c for c in cases if c["case_id"] == f"#{refs[0]}"), None)
    if anchor is None and primary:
        anchor = primary[0]
    if anchor is None:
        return primary

    primary_ids = {c["case_id"] for c in primary}
    siblings = _siblings_by_date(cases, anchor, exclude_ids=primary_ids | {anchor["case_id"]})

    result = list(primary)
    if anchor["case_id"] not in primary_ids:
        result.append(anchor)
    result.extend(siblings)

    seen: set[str] = set()
    deduped: list[dict] = []
    for c in result:
        if c["case_id"] not in seen:
            deduped.append(c)
            seen.add(c["case_id"])
    return deduped[:limit]
