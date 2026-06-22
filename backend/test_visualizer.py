"""
비쥬얼라이저 단독 테스트 서버.

실제 TTS 모델(MeloTTS/OpenVoice) 없이 합성 오디오로 FFT → 막대 시퀀스 →
WebSocket → 브라우저 렌더링 전체 파이프라인을 검증한다.

실행:
    cd backend
    python test_visualizer.py

브라우저:
    http://localhost:8000/

트리거 (브라우저에서 자동) 또는 다른 PowerShell:
    Invoke-RestMethod -Method POST -Uri "http://localhost:8000/test/play?duration=3"
"""
import asyncio
import threading
import numpy as np
import sounddevice as sd
import uvicorn
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse

from api.avatar_ws import avatar_websocket_endpoint, broadcast
from domains.tts.visualizer import compute_bar_frames, VIS_FPS, VIS_BAR_COUNT


FRONTEND = Path(__file__).resolve().parent.parent / "frontend" / "avatar.html"
SR = 22050


def synthesize_speech_like(duration_sec: float = 3.0, sr: int = SR) -> np.ndarray:
    """음성 비슷한 합성 신호 생성.

    - 기본 톤 220Hz에 시간에 따른 비브라토와 화음(2/3/4배음)
    - 음절 단위 ADSR amplitude envelope (음성 리듬 모방)
    - 약간의 noise
    """
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False, dtype=np.float32)

    # 시간 변동 베이스 주파수 (회화의 인토네이션 곡선 모방)
    f_base = 180 + 40 * np.sin(2 * np.pi * 0.3 * t) + 20 * np.sin(2 * np.pi * 1.1 * t)
    phase = 2 * np.pi * np.cumsum(f_base) / sr

    # 화음 + 형식적 포먼트 (다양한 주파수 분포)
    signal = (
        0.45 * np.sin(phase)
        + 0.30 * np.sin(2 * phase)
        + 0.20 * np.sin(3 * phase + 0.5)
        + 0.10 * np.sin(5 * phase + 1.0)
    )

    # 음절 envelope: ~3.5 syl/sec, 각 음절 살짝 다른 크기
    syl_rate = 3.5
    syl_phase = 2 * np.pi * syl_rate * t
    envelope = (0.5 * (1 + np.sin(syl_phase - np.pi / 2))) ** 1.5
    # 가끔 침묵 구간
    silence_mask = ((np.sin(2 * np.pi * 0.4 * t + 1.7) + 1) / 2) ** 4
    envelope = envelope * (0.4 + 0.6 * silence_mask)

    signal = signal * envelope

    # 약간의 노이즈
    signal = signal + 0.02 * np.random.RandomState(42).randn(len(t)).astype(np.float32)

    # 정규화
    peak = float(np.max(np.abs(signal))) or 1.0
    signal = (signal / peak * 0.7).astype(np.float32)
    return signal


def _play_in_thread(samples: np.ndarray, sr: int) -> threading.Thread:
    def _run():
        try:
            sd.play(samples, sr)
            sd.wait()
        except Exception as e:
            print(f"[Test] 재생 오류: {e}")
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    return th


app = FastAPI(title="Visualizer Test")


@app.get("/")
async def index():
    return FileResponse(FRONTEND)


@app.post("/test/play")
async def trigger_play(duration: float = 3.0):
    samples = synthesize_speech_like(duration_sec=duration)
    frames = compute_bar_frames(samples, SR)
    print(f"[Test] frames={len(frames)} duration={duration:.2f}s")

    await broadcast({"type": "speaking", "speaking": True,
                     "text": f"테스트 합성 오디오 {duration:.1f}s"})
    await broadcast({
        "type": "audio_frames",
        "fps": VIS_FPS,
        "bars": VIS_BAR_COUNT,
        "frames": frames,
    })

    # 약간 지연 후 재생 → 브라우저 프레임 큐 적재와 동기화
    await asyncio.sleep(0.02)
    th = _play_in_thread(samples, SR)

    async def _finish():
        # 재생 완료까지 대기
        await asyncio.get_event_loop().run_in_executor(None, th.join)
        await broadcast({"type": "speaking", "speaking": False})
    asyncio.create_task(_finish())

    return {"status": "playing", "duration": duration, "frames": len(frames)}


app.add_api_websocket_route("/ws/avatar", avatar_websocket_endpoint)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
