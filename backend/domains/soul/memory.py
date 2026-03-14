"""
Memory System
-------------
단기 기억(대화 히스토리) + 장기 기억(중요 사실 영구 저장)을 관리합니다.
장기 기억은 JSON 파일로 로컬에 저장됩니다 (외부 DB 불필요).
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from dataclasses import dataclass, field, asdict


MEMORY_FILE = "memory/long_term.json"


@dataclass
class MemoryEntry:
    """장기 기억 항목."""
    content:    str
    category:   str          = "general"   # name / preference / fact / event
    importance: int          = 1           # 1~5
    timestamp:  float        = field(default_factory=time.time)
    access_count: int        = 0

    def to_dict(self) -> dict:
        return asdict(self)


class MemorySystem:
    """
    단기 + 장기 기억을 통합 관리합니다.

    단기 기억: 현재 대화 세션의 최근 N턴
    장기 기억: 사용자 이름, 취향, 중요 사실 등 영구 저장
    """

    def __init__(self,
                 memory_file: str = MEMORY_FILE,
                 short_term_limit: int = 10):
        self.memory_file      = Path(memory_file)
        self.short_term_limit = short_term_limit
        self.short_term: list[dict]        = []   # 현재 세션 대화
        self.long_term:  list[MemoryEntry] = []   # 영구 기억
        self._load()

    # ── 단기 기억 ──────────────────────────────────────────

    def add_turn(self, role: str, content: str) -> None:
        self.short_term.append({"role": role, "content": content})
        if len(self.short_term) > self.short_term_limit * 2:
            self.short_term = self.short_term[-(self.short_term_limit * 2):]

    def get_recent_turns(self, n: int | None = None) -> list[dict]:
        if n is None:
            return list(self.short_term)
        return list(self.short_term[-(n * 2):])

    # ── 장기 기억 ──────────────────────────────────────────

    def remember(self, content: str,
                 category: str = "general",
                 importance: int = 1) -> MemoryEntry:
        """중요한 사실을 장기 기억에 저장합니다."""
        entry = MemoryEntry(content=content,
                            category=category,
                            importance=importance)
        self.long_term.append(entry)
        self._save()
        return entry

    def recall(self, category: str | None = None,
               min_importance: int = 1) -> list[MemoryEntry]:
        """장기 기억을 조회합니다."""
        results = [
            m for m in self.long_term
            if m.importance >= min_importance
            and (category is None or m.category == category)
        ]
        # 자주 접근된 기억일수록 앞에
        results.sort(key=lambda m: (m.importance, m.access_count), reverse=True)
        for m in results:
            m.access_count += 1
        return results

    def forget(self, content_keyword: str) -> int:
        """키워드가 포함된 기억을 삭제합니다."""
        before = len(self.long_term)
        self.long_term = [
            m for m in self.long_term
            if content_keyword.lower() not in m.content.lower()
        ]
        self._save()
        return before - len(self.long_term)

    # ── 프롬프트 주입용 컨텍스트 생성 ─────────────────────

    def build_memory_context(self) -> str:
        """LLM 시스템 프롬프트에 삽입할 기억 요약을 생성합니다."""
        important = self.recall(min_importance=2)
        if not important:
            return ""

        lines = ["[기억하고 있는 정보]"]
        for m in important[:5]:  # 상위 5개만
            lines.append(f"- ({m.category}) {m.content}")
        return "\n".join(lines)

    # ── 영구 저장/로드 ─────────────────────────────────────

    def _save(self) -> None:
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        data = [m.to_dict() for m in self.long_term]
        self.memory_file.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _load(self) -> None:
        if not self.memory_file.exists():
            return
        try:
            data = json.loads(self.memory_file.read_text(encoding="utf-8"))
            self.long_term = [MemoryEntry(**d) for d in data]
        except Exception:
            self.long_term = []

    def stats(self) -> dict:
        return {
            "short_term_turns": len(self.short_term) // 2,
            "long_term_count":  len(self.long_term),
            "categories": list({m.category for m in self.long_term}),
        }