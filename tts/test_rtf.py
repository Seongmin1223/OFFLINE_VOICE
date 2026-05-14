# C:\MeloTTS-Windows\melo\test_rtf.py
import onnxruntime as ort
import numpy as np
import time
import sys
import soundfile as sf
import torch

sys.path.insert(0, r'C:\MeloTTS-Windows')
from melo.api import TTS
from melo import utils

device = 'cpu'
tts = TTS(language='KR', device=device)
hps = tts.hps

text = '안녕하세요, 저는 아이리입니다.'
bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(text, 'KR', hps, device, tts.symbol_to_id)

x        = phones.numpy()[np.newaxis, :]
xl       = np.array([phones.shape[0]], dtype=np.int64)
t        = tones.numpy()[np.newaxis, :]
ja_bert_in = ja_bert.numpy()[np.newaxis, :, :]
sid      = np.array([0], dtype=np.int64)

inputs = {
    'x': x, 'x_lengths': xl, 'tones': t, 'ja_bert': ja_bert_in, 'sid': sid,
    'noise_scale':   np.array([0.667], dtype=np.float32),
    'length_scale':  np.array([1.0],   dtype=np.float32),
    'noise_scale_w': np.array([0.8],   dtype=np.float32),
}

sess = ort.InferenceSession('melotts_base_kr_v2.onnx')
start = time.time()
out = sess.run(None, inputs)
elapsed = time.time() - start
rtf = elapsed / (out[0].shape[-1] / 44100)
audio = out[0].squeeze()
sf.write('melotts_base_kr_v2_test.wav', audio, 44100)
print(f'RTF: {rtf:.3f} | 저장 완료')