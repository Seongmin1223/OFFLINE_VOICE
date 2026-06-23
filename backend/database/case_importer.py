"""장애 이력 CSV → cases 테이블 임포터.

13건 샘플(seed_cases.csv)과 향후 도착할 696건 CSV 모두 같은 컬럼 체계
(case_id, 시나리오/scenario, date, 솔루션/solution, 담당자/name, workType,
workStatus, description)를 쓰므로 헤더명만 한글/영문 양쪽으로 매핑해
동일한 임포터로 처리한다.
"""

from __future__ import annotations

import csv

import aiosqlite

_HEADER_ALIASES: dict[str, str] = {
    "case_id":      "case_id",
    "시나리오":      "scenario",
    "scenario":      "scenario",
    "date":          "case_date",
    "솔루션":        "solution",
    "solution":      "solution",
    "담당자":        "assignee",
    "name":          "assignee",
    "assignee":      "assignee",
    "worktype":      "work_type",
    "workstatus":    "work_status",
    "description":   "description",
}


def _normalize_row(row: dict[str, str]) -> dict[str, str] | None:
    mapped: dict[str, str] = {}
    for key, value in row.items():
        field = _HEADER_ALIASES.get(key.strip().lower()) or _HEADER_ALIASES.get(key.strip())
        if field:
            mapped[field] = (value or "").strip()

    if not mapped.get("case_id") or not mapped.get("description"):
        return None

    mapped.setdefault("scenario", "")
    mapped.setdefault("case_date", "")
    mapped.setdefault("solution", "")
    mapped.setdefault("assignee", "")
    mapped.setdefault("work_type", "")
    mapped.setdefault("work_status", "")
    return mapped


async def import_cases_from_csv(db: aiosqlite.Connection, csv_path: str) -> int:
    """CSV를 읽어 cases 테이블에 upsert. 반환값은 처리된 행 수."""
    count = 0
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = _normalize_row(raw_row)
            if row is None:
                continue
            await db.execute(
                """
                INSERT INTO cases
                    (case_id, scenario, case_date, solution, assignee, work_type, work_status, description)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(case_id) DO UPDATE SET
                    scenario    = excluded.scenario,
                    case_date   = excluded.case_date,
                    solution    = excluded.solution,
                    assignee    = excluded.assignee,
                    work_type   = excluded.work_type,
                    work_status = excluded.work_status,
                    description = excluded.description
                """,
                (
                    row["case_id"], row["scenario"], row["case_date"], row["solution"],
                    row["assignee"], row["work_type"], row["work_status"], row["description"],
                ),
            )
            count += 1
    await db.commit()
    return count
