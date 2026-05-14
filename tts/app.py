import subprocess
from glob import glob

# WebUI by mrfakename <X @realmrfakename / HF @mrfakename>
# Demo also available on HF Spaces: https://huggingface.co/spaces/mrfakename/MeloTTS
import gradio as gr
import time
import uuid
import shutil
import os, torch, io
# os.system('python -m unidic download')
print("Make sure you've downloaded unidic (python -m unidic download) for this WebUI to work.")
from melo.api import TTS
speed = 1.0
import tempfile
import click
import atexit

device = 'auto'
models = {
    'EN': TTS(language='EN', device=device),
    'ES': TTS(language='ES', device=device),
    'FR': TTS(language='FR', device=device),
    'ZH': TTS(language='ZH', device=device),
    'JP': TTS(language='JP', device=device),
    'KR': TTS(language='KR', device=device),
}
speaker_ids = models['EN'].hps.data.spk2id

default_text_dict = {
    'EN': 'The field of text-to-speech has seen rapid development recently.',
    'ES': 'El campo de la conversión de texto a voz ha experimentado un rápido desarrollo recientemente.',
    'FR': 'Le domaine de la synthèse vocale a connu un développement rapide récemment',
    'ZH': 'text-to-speech 领域近年来发展迅速',
    'JP': 'テキスト読み上げの分野は最近急速な発展を遂げています',
    'KR': '최근 텍스트 음성 변환 분야가 급속도로 발전하고 있습니다.',    
}
    
def get_ckpt_files(folder_path):
    return [os.path.basename(f) for f in glob(os.path.join(folder_path, '*.pth'))]

def run_infer(text, ckpt_file, output_path):
    ckpt_path = os.path.join("custom", ckpt_file)
    config_path = os.path.join(os.path.dirname(ckpt_path), 'config.json')
    
    try:
        model = TTS(language="EN", config_path=config_path, ckpt_path=ckpt_path)
        
        for spk_name, spk_id in model.hps.data.spk2id.items():
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            model.tts_to_file(text, spk_id, output_path)
        
        return output_path
    except Exception as e:
        print(f"Error running infer: {e}")
        return None

def synthesize(speaker, text, speed, language, ckpt_file, progress=gr.Progress()):    
    start_time = time.time()
    print(f"Synthesis started at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        if ckpt_file and ckpt_file != "None":
            # Use custom infer function for custom models
            output_file = run_infer(text, ckpt_file, temp_file.name)
        else:
            # Use original MeloTTS synthesis
            models[language].tts_to_file(text, models[language].hps.data.spk2id[speaker], temp_file.name, speed=speed, pbar=progress.tqdm)
            output_file = temp_file.name
    
    end_time = time.time()
    print(f"Synthesis ended at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time:.2f} seconds")
    
    return output_file


def update_speaker_dropdown(language, ckpt_file):
    if ckpt_file and ckpt_file != "None":
        return gr.update(value=None, choices=[], interactive=False)
    else:
        return gr.update(value=list(models[language].hps.data.spk2id.keys())[0], 
                         choices=list(models[language].hps.data.spk2id.keys()), 
                         interactive=True)
    
def cleanup_temp_files():
    temp_dir = tempfile.gettempdir()
    for filename in os.listdir(temp_dir):
        if filename.endswith(".wav"):
            try:
                os.remove(os.path.join(temp_dir, filename))
            except PermissionError:
                print(f"Could not remove {filename}")

atexit.register(cleanup_temp_files)
    
def load_speakers(language, text):
    if text in list(default_text_dict.values()):
        newtext = default_text_dict[language]
    else:
        newtext = text
    return gr.update(value=list(models[language].hps.data.spk2id.keys())[0], choices=list(models[language].hps.data.spk2id.keys())), newtext

with gr.Blocks() as demo:
    gr.Markdown('# MeloTTS WebUI\n\nA WebUI for MeloTTS.')
    with gr.Group():
        language = gr.Radio(['EN', 'ES', 'FR', 'ZH', 'JP', 'KR'], label='Language', value='EN')
        speaker = gr.Dropdown(speaker_ids.keys(), interactive=True, value='EN-US', label='Speaker')
        speed = gr.Slider(label='Speed', minimum=0.1, maximum=10.0, value=1.0, interactive=True, step=0.1)
        text = gr.Textbox(label="Text to speak", value=default_text_dict['EN'])
        
        # Use a relative path to the 'custom' folder
        custom_folder = os.path.join(os.path.dirname(__file__), "custom")
        ckpt_files = ["None"] + get_ckpt_files(custom_folder)
        
        ckpt_dropdown = gr.Dropdown(choices=ckpt_files, value="None", label="Select .pth file (optional)", interactive=True)
        language.input(load_speakers, inputs=[language, text], outputs=[speaker, text])
        ckpt_dropdown.change(update_speaker_dropdown, inputs=[language, ckpt_dropdown], outputs=[speaker])
    
    btn = gr.Button('Synthesize', variant='primary')
    aud = gr.Audio(interactive=False, type="filepath")
    btn.click(synthesize, inputs=[speaker, text, speed, language, ckpt_dropdown], outputs=[aud])
    gr.Markdown('WebUI by [mrfakename](https://twitter.com/realmrfakename).')

@click.command()
@click.option('--share', '-s', is_flag=True, show_default=True, default=False, help="Expose a publicly-accessible shared Gradio link usable by anyone with the link. Only share the link with people you trust.")
@click.option('--host', '-h', default=None)
@click.option('--port', '-p', type=int, default=None)
def main(share, host, port):
    demo.queue(api_open=False).launch(inbrowser=True, server_name=host, server_port=port)

if __name__ == "__main__":
    main()