import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import threading
import sys

sys.stdout.reconfigure(encoding='utf-8')

SAMPLE_RATE = 16000
SAVE_PATH = "recorded.wav"
DEVICE = 1  # 마이크 배열(인텔 스마트 사운드)

print("Whisper 모델 로딩 중...")
model = whisper.load_model("small", download_root="./models/whisper")

while True:
    input("\n[엔터] 녹음 시작")
    print("녹음 중...")

    audio_chunks = []
    recording = True

    def record():
        while recording:
            chunk = sd.rec(
                int(0.1 * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=DEVICE
            )
            sd.wait()
            audio_chunks.append(chunk)

    thread = threading.Thread(target=record)
    thread.start()

    input("[엔터] 녹음 중지")
    recording = False
    thread.join()

    print("인식 중...")
    audio = np.concatenate(audio_chunks, axis=0)
    audio_int16 = (audio.squeeze() * 32767).astype(np.int16)
    wavfile.write(SAVE_PATH, SAMPLE_RATE, audio_int16)

    result = model.transcribe(SAVE_PATH, language="ko")
    print("\n인식 결과:", result['text'])

    again = input("\n다시 할까요? (y/n): ")
    if again.lower() != "y":
        break

print("종료")