"""포비 TTS 단독 테스트 (LLM/STT 없이).

사용법:
    python test_poby_tts.py "안녕하세요 포비예요"
    python test_poby_tts.py "오늘 날씨 정말 좋네요" happy
"""
import sys
import sounddevice as sd
from domains.tts.piper_engine import get_engine


def main():
    args = sys.argv[1:]
    if not args:
        text = "안녕하세요 포비예요 오늘 같이 재밌게 놀아요"
        emotion = "neutral"
    else:
        text = args[0]
        emotion = args[1] if len(args) > 1 else "neutral"

    print(f"[Poby TTS] text='{text}'  emotion='{emotion}'")
    engine = get_engine()
    audio, sr = engine.synthesize(text, emotion)
    print(f"[Poby TTS] 합성 완료: {len(audio)/sr:.1f}sec @ {sr}Hz, 재생 중...")
    sd.play(audio, sr)
    sd.wait()
    print("[Poby TTS] 재생 완료")


if __name__ == "__main__":
    main()
