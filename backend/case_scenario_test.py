"""SphereAX_WeMeet2026_적용시나리오_실데이터예시_v0.1.md 시나리오 A~D 재현 테스트.

키워드+필터 기반 search_cases()가 md 문서의 쿼리·예상 case_id를 얼마나
재현하는지 확인한다. 임베딩 없는 1차 구현이므로 100% 일치를 요구하지 않고
hit/miss를 그대로 보여준다.

실행:
    python case_scenario_test.py
"""

from __future__ import annotations

import asyncio
import os
import shutil
import tempfile

from config import config
from core.case_search import search_cases
from database.connection import connect, disconnect


SCENARIOS = [
    {
        "name": "시나리오 D — 솔루션 횡단 재발",
        "query": "[담당자10] Safewatcher 위험감지 통계 엑셀 주간별 순서 또 안맞아. 비슷한 거 처리한 적 있나?",
        "expected": {"#0626", "#0352", "#0252"},
    },
    {
        "name": "시나리오 A — 출장 전 사전 회상",
        "query": "내일 인천교통공사로 시큐워처 망연계 신규 구축 출장을 가는데, 혹시 현장에서 막힐 만한 Tomcat 포트나 자바 버전 관련해서 참고할 만한 과거 이력이 있어?",
        "expected": {"#0446", "#0458", "#0393", "#0433"},
    },
    {
        "name": "시나리오 B — 현장 1차 진단 (md 표의 fact 후보 3건)",
        "query": "지금 파이어워처가 설치된 지자체 현장인데 메인 서버가 갑자기 셧다운됐어. 소프트웨어 오류인지 원인을 잘 모르겠는데, 과거에 비슷한 지자체 서버 다운 사례가 있었나?",
        "expected": {"#0035", "#0033", "#0025"},
    },
    {
        # 같은 "케이스 번호 명시 → 체인 추적" 기능을 검증하기 위해 포함
        "name": "후속 질문 — #0035 체인 추적",
        "query": "방금 말한 전남도청 메인서버 셧다운 건([case:#0035]) 말이야. 그 전후로 진행된 현장 조치나 복구 작업 기록이 더 남아있어?",
        "expected": {"#0035", "#0025"},
    },
    {
        "name": "시나리오 C — 인수인계",
        "query": "이번에 Firewatcher Map2D 컴포넌트 운영 담당을 새로 넘겨받았는데, 전임자들이 남긴 특이사항이나 인수인계 관련 히스토리가 있으면 요약해 줘.",
        "expected": {"#0084", "#0095"},
    },
]


async def main() -> None:
    tmp_dir = tempfile.mkdtemp(prefix="airi_case_test_")
    config.DB_PATH = os.path.join(tmp_dir, "test.db")

    print("=" * 70)
    print("  케이스 회상 시나리오 재현 테스트")
    print("=" * 70)

    await connect()  # connect() 내부에서 cases 테이블이 비어있으면 seed_cases.csv 자동 적재

    try:
        total_hit = 0
        total_expected = 0
        for sc in SCENARIOS:
            results = await search_cases(sc["query"], limit=5)
            got_ids = {c["case_id"] for c in results}
            hit = got_ids & sc["expected"]
            miss = sc["expected"] - got_ids
            extra = got_ids - sc["expected"]

            total_hit += len(hit)
            total_expected += len(sc["expected"])

            print(f"\n[{sc['name']}]")
            print(f"  쿼리: {sc['query'][:60]}{'...' if len(sc['query']) > 60 else ''}")
            print(f"  검색 결과: {[c['case_id'] for c in results]}")
            print(f"  기대 ID  : {sorted(sc['expected'])}")
            print(f"  hit={sorted(hit)} miss={sorted(miss)} extra={sorted(extra)}")

        rate = (total_hit / total_expected * 100) if total_expected else 0
        print("\n" + "=" * 70)
        print(f"  전체 recall: {total_hit}/{total_expected} ({rate:.0f}%)")
        print("=" * 70)
    finally:
        await disconnect()
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main())
