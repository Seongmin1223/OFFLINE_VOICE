import os
os.add_dll_directory(r"C:\Program Files\MeCab\bin")
os.environ["PATH"] = r"C:\ffmpeg\bin;" + os.environ["PATH"]

from melo.api import TTS

model = TTS(
    language='KR',
    device='cuda',
    config_path=r'C:\MeloTTS-Windows\melo\logs\data\example\config\config.json',
    ckpt_path=r'C:\MeloTTS-Windows\melo\logs\data\example\config\G_64000.pth'
)

speaker_ids = model.hps.data.spk2id
print("스피커:", speaker_ids)

model.tts_to_file(
    "안녕하세요! 오늘 날씨가 정말 좋네요.",
    speaker_ids['F2001'],
    "output_finetune.wav",
    speed=1.0
)
print("완료!")