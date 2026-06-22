"""포비 TTS 단독 테스트 (LLM/STT 없이).

사용법:
    python test_poby_tts.py "안녕하세요 포비예요"
"""
import sys
import sounddevice as sd
from domains.tts.piper_engine import get_engine


def main():
    args = sys.argv[1:]
    text = args[0] if args else "안녕하세요 포비예요 오늘 같이 재밌게 놀아요"

    print(f"[Poby TTS] text='{text}'")
    engine = get_engine()
    audio, sr = engine.synthesize(text)
    print(f"[Poby TTS] 합성 완료: {len(audio)/sr:.1f}sec @ {sr}Hz, 재생 중...")
    sd.play(audio, sr)
    sd.wait()
    print("[Poby TTS] 재생 완료")


if __name__ == "__main__":
    main()
