# C:\MeloTTS-Windows\melo\test_openvoice.py
import sys
import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, r'C:\OpenVoice')
sys.path.insert(0, r'C:\MeloTTS-Windows')

from melo.api import TTS
from melo import utils
import onnxruntime as ort
from openvoice import se_extractor
from openvoice.api import ToneColorConverter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')

# 1단계: MeloTTS ONNX로 음성 생성
tts = TTS(language='KR', device='cpu')
hps = tts.hps
text = '안녕하세요, 저는 아이리입니다. 오늘도 좋은 하루 보내세요. 무엇이든 도와드릴게요. 함께라면 무엇이든 할 수 있어요. 당신과 이야기하는 것이 즐거워요.'
bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(text, 'KR', hps, 'cpu', tts.symbol_to_id)

inputs = {
    'x':             phones.numpy()[np.newaxis, :],
    'x_lengths':     np.array([phones.shape[0]], dtype=np.int64),
    'tones':         tones.numpy()[np.newaxis, :],
    'ja_bert':       ja_bert.numpy()[np.newaxis, :, :],
    'sid':           np.array([0], dtype=np.int64),
    'noise_scale':   np.array([0.667], dtype=np.float32),
    'length_scale': np.array([0.8], dtype=np.float32),
    'noise_scale_w': np.array([0.8],   dtype=np.float32),
}
sess = ort.InferenceSession(r'C:\MeloTTS-Windows\melo\melotts_base_kr_v2.onnx')
out = sess.run(None, inputs)
audio = out[0].squeeze()
sf.write('temp_tts.wav', audio, 44100)
print('1단계 TTS 완료')

# 2단계: OpenVoice로 톤 변환
ckpt_path = r'C:\OpenVoice\checkpoints_v2\converter'
converter = ToneColorConverter(f'{ckpt_path}/config.json', device=device)
converter.load_ckpt(f'{ckpt_path}/checkpoint.pth')
ref_wav = r'C:\MeloTTS-Windows\melo\reference_voice.wav'
target_se, _ = se_extractor.get_se(ref_wav, converter, vad=True)
source_se, _ = se_extractor.get_se('temp_tts.wav', converter, vad=True)

converter.convert(
    audio_src_path='temp_tts.wav',
    src_se=source_se,
    tgt_se=target_se,
    output_path='output_openvoice.wav',
)
print('2단계 OpenVoice 변환 완료 → output_openvoice.wav')