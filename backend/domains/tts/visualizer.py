"""TTS 출력 PCM을 비쥬얼라이저용 프레임 시퀀스로 변환.

외부 무거운 의존성(MeloTTS, OpenVoice) 없이 numpy 만으로 동작하므로
test 또는 다른 백엔드(piper, kokoro 등)와도 자유롭게 재사용 가능.
"""
import numpy as np


VIS_BAR_COUNT = 56
VIS_FPS = 30
VIS_FREQ_MIN = 80      # Hz
VIS_FREQ_MAX = 8000    # Hz


def compute_bar_frames(samples: np.ndarray, sr: int,
                       num_bars: int = VIS_BAR_COUNT,
                       fps: int = VIS_FPS) -> list[list[float]]:
    """TTS PCM을 받아 프레임당 [num_bars] 길이의 0~1 정규화 막대 높이 리스트로 변환.

    로그 스케일 주파수 밴드, dB 변환, -60dB~0dB 클리핑.
    """
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = samples.astype(np.float32, copy=False)

    samples_per_frame = max(1, int(sr / fps))
    num_frames = max(1, len(samples) // samples_per_frame)

    nyquist = sr / 2
    fmax = min(VIS_FREQ_MAX, nyquist - 1)
    edges = np.logspace(np.log10(VIS_FREQ_MIN), np.log10(fmax), num_bars + 1)

    window = np.hanning(samples_per_frame).astype(np.float32)

    frames: list[list[float]] = []
    for i in range(num_frames):
        chunk = samples[i * samples_per_frame:(i + 1) * samples_per_frame]
        if len(chunk) < 4:
            frames.append([0.0] * num_bars)
            continue
        if len(chunk) != len(window):
            w = np.hanning(len(chunk)).astype(np.float32)
        else:
            w = window
        spec = np.abs(np.fft.rfft(chunk * w))
        freqs = np.fft.rfftfreq(len(chunk), 1.0 / sr)

        bands = np.zeros(num_bars, dtype=np.float32)
        for j in range(num_bars):
            mask = (freqs >= edges[j]) & (freqs < edges[j + 1])
            if mask.any():
                bands[j] = spec[mask].mean()

        # dB 스케일 (-60dB~0dB → 0~1)
        bands_db = 20.0 * np.log10(bands + 1e-6)
        bands_norm = np.clip((bands_db + 60.0) / 60.0, 0.0, 1.0)

        frames.append([round(float(b), 3) for b in bands_norm])

    # 시퀀스 전체 피크로 재정규화: 작은 진폭의 TTS도 막대가 충분히 솟게.
    # 너무 조용한 구간(전부 0 근처)은 그대로 두어 idle 파동을 가리지 않음.
    peak = max((max(f) for f in frames), default=0.0)
    if peak > 0.05:
        frames = [[min(round(b / peak, 3), 1.0) for b in f] for f in frames]

    return frames
