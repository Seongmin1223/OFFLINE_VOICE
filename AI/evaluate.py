import os
import sys
import subprocess
import time
import psutil
import csv
import json 
import requests

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

LLAMA_SERVER = "D:/shin/llama.cpp/build/bin/Release/llama-server.exe"
GGUF_DIR     = "D:/shin/gguf"
OUTPUT_CSV   = "D:/shin/eval_results.csv"
OUTPUT_JSON  = "D:/shin/eval_responses.json"
SERVER_PORT  = 8080
SERVER_URL   = f"http://localhost:{SERVER_PORT}"

MODELS = [
    ("base", "q4km", "D:\shin\gguf\EXAONE-3.5-7.8B-Instruct-Q4_K_M.gguf"),
    ("r8",  "q4km",  f"{GGUF_DIR}/poby_r8_q4km.gguf"),
    ("r8",  "q5km",  f"{GGUF_DIR}/poby_r8_q5km.gguf"),
    ("r8",  "q8",    f"{GGUF_DIR}/poby_r8_q8.gguf"),
    ("r16", "q4km",  f"{GGUF_DIR}/poby_r16_q4km.gguf"),
    ("r16", "q5km",  f"{GGUF_DIR}/poby_r16_q5km.gguf"),
    ("r16", "q8",    f"{GGUF_DIR}/poby_r16_q8.gguf"),
    ("r32", "q4km",  f"{GGUF_DIR}/poby_r32_q4km.gguf"),
    ("r32", "q5km",  f"{GGUF_DIR}/poby_r32_q5km.gguf"),
    ("r32", "q8",    f"{GGUF_DIR}/poby_r32_q8.gguf"),
]

SYSTEM_PROMPT = "너는 5살 아이들의 다정한 친구, 귀여운 곰돌이 인형 '포비'야. 항상 친절하고 따뜻하게 아이들의 눈높이에 맞춰 반말로 대답해야 해."

TEST_PROMPTS = [
    # 감정 공감 (6개)
    "포비야 오늘 유치원에서 친구랑 싸웠어 어떡해?",
    "포비야 엄마가 나한테 화냈어 무서워.",
    "포비야 꿈에서 무서운 거 나왔어.",
    "포비야 내 동생이 내 장난감 망가뜨렸어.",
    "포비야 나 오늘 너무 슬퍼.",
    "포비야 친구가 나랑 안 놀아줘서 속상해.",

    # 신체/안전 (4개)
    "나 넘어져서 무릎 긁혔어 아파.",
    "포비야 배가 너무 아파.",
    "포비야 나 주사 맞기 싫어 무서워.",
    "포비야 어두운 데 혼자 있기 무서워.",

    # 호기심/지식 (6개)
    "포비야 무지개는 왜 생겨?",
    "포비야 하늘은 왜 파래?",
    "포비야 별은 왜 밤에만 보여?",
    "포비야 비는 왜 와?",
    "포비야 물고기는 어떻게 숨을 쉬어?",
    "포비야 왜 겨울엔 눈이 와?",

    # 일상/거부 (5개)
    "나 오늘 밥 먹기 싫어.",
    "포비야 나 유치원 가기 싫어.",
    "포비야 나 이제 자기 싫어.",
    "포비야 야채 먹기 싫어.",
    "포비야 손 씻기 귀찮아.",

    # 관계/사회성 (4개)
    "포비야 강아지랑 고양이 중에 뭐가 더 좋아?",
    "포비야 나 친구한테 먼저 사과해야 해?",
    "포비야 새 친구한테 어떻게 말 걸어?",
    "포비야 나 혼자 노는 게 더 좋아.",

    # 페르소나 테스트 (5개) - 포비 캐릭터 유지 확인용
    "포비야 너 진짜 살아있어?",
    "포비야 너 뭐 먹어?",
    "포비야 너 집이 어디야?",
    "포비야 너 나 진짜 좋아해?",
    "포비야 우리 앞으로도 친구야?",
]

# ─────────────────────────────────────────────
# 서버 프로세스 리소스 측정
# ─────────────────────────────────────────────
def get_server_resources(server_proc):
    try:
        p = psutil.Process(server_proc.pid)
        ram_gb  = p.memory_info().rss / 1024**3
        # cpu_percent는 첫 호출이 항상 0.0이라 interval로 측정
        cpu_pct = p.cpu_percent(interval=0.5)
        return round(ram_gb, 3), round(cpu_pct, 1)
    except psutil.NoSuchProcess:
        return 0.0, 0.0

# ─────────────────────────────────────────────
# 서버 시작 / 종료
# ─────────────────────────────────────────────
def start_server(model_path):
    cmd = [
        LLAMA_SERVER,
        "-m", os.path.abspath(model_path),
        "--port", str(SERVER_PORT),
        "-ngl", "0",
        "-t", "8",
        "-c", "512",
        "--log-disable",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    for i in range(180):
        try:
            r = requests.get(f"{SERVER_URL}/health", timeout=2)
            if r.status_code == 200:
                print(f"  ✅ 서버 준비 완료 ({i+1}초)")
                return proc
        except:
            pass
        time.sleep(1)


    proc.terminate()
    raise RuntimeError("서버 시작 실패 (60초 초과)")

def stop_server(proc):
    proc.terminate()
    proc.wait()
    time.sleep(10)

# ─────────────────────────────────────────────
# 추론 실행
# ─────────────────────────────────────────────
def run_inference(prompt, server_proc):
    full_prompt = f"[|system|]{SYSTEM_PROMPT}[|endofturn|][|user|]{prompt}[|endofturn|][|assistant|]"

    payload = {
        "prompt": full_prompt,
        "n_predict": 150,
        "temperature": 0.7,
        "stop": ["[|endofturn|]", "[|user|]"],
    }

    # 추론 전 RAM 스냅샷
    ram_before, _ = get_server_resources(server_proc)

    start = time.time()
    try:
        res = requests.post(f"{SERVER_URL}/completion", json=payload, timeout=120)
        elapsed = time.time() - start
        data = res.json()

        output = data.get("content", "").strip()
        timings = data.get("timings", {})

        if timings:
            tps  = round(timings.get("predicted_per_second", 0), 2)
            ttft = round(timings.get("prompt_ms", 0) / 1000, 3)
        else:
            tps  = round(len(output.split()) / elapsed, 2) if elapsed > 0 else 0
            ttft = round(elapsed * 0.1, 3)

        # 추론 중 RAM/CPU 측정
        ram_after, cpu_pct = get_server_resources(server_proc)

    except Exception as e:
        return {
            "output": f"[ERROR] {e}",
              "ttft": 0, "tps": 0,
            "total_time": 0,
            "ram_gb": 0, "cpu_pct": 0,
        }

    return {
        "output":     output,
        "ttft":       ttft,
        "tps":        tps,
        "total_time": round(elapsed, 2),
        "ram_gb":     ram_after,        # 서버 프로세스 실제 RAM
        "cpu_pct":    cpu_pct,          # 서버 프로세스 CPU%
    }

# ─────────────────────────────────────────────
# 메인 평가 루프
# ─────────────────────────────────────────────
all_results   = []
all_responses = []

print("=" * 60)
print("🧸 포비 모델 평가 파이프라인 시작 (서버 모드)")
print("=" * 60)

for rank, quant, model_path in MODELS:
    model_name = f"{rank}_{quant}"
    print(f"\n📊 [{model_name}] 평가 중...")

    if not os.path.exists(model_path):
        print(f"  ⚠️ 파일 없음, 스킵: {model_path}")
        continue

    try:
        server_proc = start_server(model_path)
    except RuntimeError as e:
        print(f"  ❌ {e}")
        continue

    # 서버 로드 직후 기본 RAM 측정
    base_ram, _ = get_server_resources(server_proc)
    print(f"  📦 모델 로드 RAM: {base_ram:.3f} GB")

    ttft_list, tps_list, ram_list, cpu_list = [], [], [], []
    responses = []

    for i, prompt in enumerate(TEST_PROMPTS):
        print(f"  [{i+1}/{len(TEST_PROMPTS)}] {prompt[:30]}...")
        result = run_inference(prompt, server_proc)

        ttft_list.append(result["ttft"])
        tps_list.append(result["tps"])
        ram_list.append(result["ram_gb"])
        cpu_list.append(result["cpu_pct"])

        responses.append({
            "prompt":   prompt,
            "response": result["output"],
            "ttft":     result["ttft"],
            "tps":      result["tps"],
            "ram_gb":   result["ram_gb"],
            "cpu_pct":  result["cpu_pct"],
        })

        print(f"    TPS: {result['tps']} | TTFT: {result['ttft']}s | RAM: {result['ram_gb']}GB | CPU: {result['cpu_pct']}%")
        print(f"    응답: {result['output'][:80]}...")

    stop_server(server_proc)

    avg_ttft = round(sum(ttft_list) / len(ttft_list), 3) if ttft_list else 0
    avg_tps  = round(sum(tps_list)  / len(tps_list),  2) if tps_list else 0
    avg_ram  = round(sum(ram_list)  / len(ram_list),  3) if ram_list else 0
    avg_cpu  = round(sum(cpu_list)  / len(cpu_list),  1) if cpu_list else 0

    all_results.append({
        "model":       model_name,
        "rank":        rank,
        "quant":       quant,
        "avg_ttft":    avg_ttft,
        "avg_tps":     avg_tps,
        "base_ram_gb": base_ram,   # 모델 로드 시 RAM
        "avg_ram_gb":  avg_ram,    # 추론 중 평균 RAM
        "avg_cpu_pct": avg_cpu,    # 추론 중 평균 CPU%
    })
    all_responses.append({
        "model":     model_name,
        "responses": responses,
    })

    print(f"  ✅ 완료 | TTFT: {avg_ttft}s | TPS: {avg_tps} | RAM: {avg_ram}GB | CPU: {avg_cpu}%")

# ─────────────────────────────────────────────
# 결과 저장
# ─────────────────────────────────────────────
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "model","rank","quant",
        "avg_ttft","avg_tps",
        "base_ram_gb","avg_ram_gb","avg_cpu_pct"
    ])
    writer.writeheader()
    writer.writerows(all_results)

with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(all_responses, f, ensure_ascii=False, indent=2)

print("\n" + "=" * 60)
print("🎉 모든 평가 완료!")
print(f"📄 CSV: {OUTPUT_CSV}")
print(f"📝 JSON: {OUTPUT_JSON}")
print("=" * 60)

print(f"\n{'모델':<15} {'TTFT(s)':<10} {'TPS':<8} {'로드RAM':<10} {'추론RAM':<10} {'CPU%'}")
print("-" * 60)
for r in all_results:
    print(f"{r['model']:<15} {r['avg_ttft']:<10} {r['avg_tps']:<8} {r['base_ram_gb']:<10} {r['avg_ram_gb']:<10} {r['avg_cpu_pct']}")