import os
import time
import wave
import psutil
import numpy as np
import torch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from silero_vad import load_silero_vad, VADIterator

def print_memory_usage(label):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    print(f"[{label}] 현재 RAM 사용량: {mem_mb:.2f} MB")

def get_chunks_from_wav(wav_path, chunk_size=512):
    chunks = []
    with wave.open(wav_path, 'rb') as wf:
        channels = wf.getnchannels() #  채널 수 확인 (1: Mono, 2: Stereo)
        
        while True:
            data = wf.readframes(chunk_size)
            if not data:
                break
                
            # Bytes -> Float32 Tensor 변환
            audio_np = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            
            if channels == 2:
                audio_np = audio_np.reshape(-1, 2).mean(axis=1)
                
            if len(audio_np) != chunk_size:
                break
                
            chunks.append(torch.from_numpy(audio_np))
    return chunks

if __name__ == "__main__":
    # 테스트 wav 경로(assets 폴더)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_wav_path = os.path.join(current_dir, "assets", "test.wav")

    print("Silero VAD 성능 벤치마크 테스트")
    print_memory_usage("VAD 로드 전")

    model = load_silero_vad(onnx=True)
    vad_iterator = VADIterator(model, sampling_rate=16000, threshold=0.8, min_silence_duration_ms=2000)
    
    print_memory_usage("VAD 로드 후")


    if not os.path.exists(test_wav_path):
        print(f"\n 테스트할 파일이 없습니다: {test_wav_path}")
        sys.exit(1)

    test_chunks = get_chunks_from_wav(test_wav_path)
    print(f"\n총 {len(test_chunks)}개의 청크(Chunk)를 테스트")

    # Latency 측정
    start_time = time.time()
    
    for chunk in test_chunks:
        # VAD 추론
        vad_iterator(chunk, return_seconds=False)
        
    end_time = time.time()
    
    total_time = end_time - start_time
    time_per_chunk = (total_time / len(test_chunks)) * 1000 # ms 단위로 변환
    
    print(f"\n 벤치마크 결과")
    print(f"총 처리 시간: {total_time:.4f} 초")
    print(f"청크 1개당 평균 처리 시간: {time_per_chunk:.4f} ms")