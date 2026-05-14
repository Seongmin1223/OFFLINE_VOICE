# check_model.py
import sys
sys.path.insert(0, r'C:\MeloTTS-Windows\melo')
import os
os.add_dll_directory(r"C:\Program Files\MeCab\bin")
import json
import utils
from models import SynthesizerTrn
from text.symbols import symbols

config = json.load(open(r'C:\MeloTTS-Windows\melo\data\example\config.json'))
hps_data = config['data']
hps_model = config['model']

net_g = SynthesizerTrn(
    len(symbols),
    hps_data['filter_length'] // 2 + 1,
    hps_data['sampling_rate'],
    n_speakers=hps_data['n_speakers'],
    num_languages=hps_data.get('num_languages', 1),
    num_tones=hps_data.get('num_tones', 1),
    **hps_model
)

print('language_emb:', net_g.enc_p.language_emb.weight.shape)
print('emb_g:', net_g.emb_g.weight.shape)
print('emb:', net_g.enc_p.emb.weight.shape)