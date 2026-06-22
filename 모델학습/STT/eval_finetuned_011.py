#!/usr/bin/env python3
"""
Whisper 베이스라인(순정) 모델 평가 - 071 단독
- VAD on/off 둘 다 측정
- CER, WER, 지연시간, RTF, CPU/RAM
- 실시간 진행 상황 출력
"""
import os
import re
import time
import json
import gc
import numpy as np
import pandas as pd
import soundfile as sf
import psutil
from datetime import datetime, timedelta
from faster_whisper import WhisperModel

# ─────────────────────────────────────
# 설정
# ─────────────────────────────────────
MODEL_PATH   = "./whisper_small_011_ct2"
DEVICE       = "cpu"
COMPUTE_TYPE = "int8"
SAMPLES_DIR  = "./samples_eval_011"


def clean_text(text):
    text = re.sub(r'\((.*?)\)/\((.*?)\)', r'\2', text)
    text = re.sub(r'[a-z]/', '', text)
    text = re.sub(r'[*&/]', '', text)
    text = re.sub(r'[.,!?]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def evaluate_mode(model, wav_files, samples_dir, use_vad, mode_label, jiwer):
    print(f"\n{'='*70}")
    print(f"  🎯 [{mode_label}] 평가 시작")
    print(f"  샘플: {len(wav_files):,}개 | VAD: {'ON' if use_vad else 'OFF'}")
    print(f"  시작: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*70}")

    process = psutil.Process(os.getpid())
    num_cores = psutil.cpu_count()
    process.cpu_percent(None)
    time.sleep(0.1)

    results = []
    skipped = 0
    failed = 0
    start_time = time.time()

    for idx, wav_file in enumerate(wav_files):
        basename = wav_file.replace('.wav', '')
        wav_path = os.path.join(samples_dir, wav_file)
        json_path = os.path.join(samples_dir, f"{basename}.json")

        if not os.path.exists(json_path):
            skipped += 1
            continue

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)
                true_text = label_data.get("text", "")

            if not true_text:
                skipped += 1
                continue

            audio_array, sr = sf.read(wav_path)
            audio_array = audio_array.astype(np.float32)  # VAD ONNX 호환
            audio_duration = len(audio_array) / sr

            if sr != 16000:
                try:
                    import librosa
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                    audio_array = audio_array.astype(np.float32)
                except ImportError:
                    skipped += 1
                    continue

            t_start = time.time()
            segments, info = model.transcribe(
                audio_array,
                language="ko",
                beam_size=5,
                vad_filter=use_vad,
                vad_parameters=dict(min_silence_duration_ms=500) if use_vad else None,
            )
            pred_text = "".join([s.text for s in segments])
            latency = time.time() - t_start
            rtf = latency / audio_duration if audio_duration > 0 else 0

            c_true = clean_text(true_text)
            c_pred = clean_text(pred_text)

            if not c_true:
                skipped += 1
                continue

            cer = jiwer.cer(c_true, c_pred)
            wer = jiwer.wer(c_true, c_pred)

            cpu_pct = process.cpu_percent() / num_cores
            ram_mb = process.memory_info().rss / (1024 * 1024)
            sys_ram_pct = psutil.virtual_memory().percent

            results.append({
                "id": basename,
                "audio_sec": audio_duration,
                "true_text": c_true,
                "pred_text": c_pred,
                "cer": cer,
                "wer": wer,
                "latency_sec": latency,
                "rtf": rtf,
                "cpu_pct": cpu_pct,
                "ram_mb": ram_mb,
                "sys_ram_pct": sys_ram_pct,
            })

            if (idx + 1) % 50 == 0:
                elapsed = time.time() - start_time
                speed = (idx + 1) / elapsed
                remain = (len(wav_files) - idx - 1) / speed
                eta = datetime.now() + timedelta(seconds=remain)
                avg_cer = np.mean([r["cer"] for r in results]) * 100
                avg_lat = np.mean([r["latency_sec"] for r in results])
                avg_rtf = np.mean([r["rtf"] for r in results])

                print(f"\n┌─ [{idx+1:,}/{len(wav_files):,}] ({(idx+1)/len(wav_files)*100:.1f}%)")
                print(f"│  📊 CER: {avg_cer:.2f}% | 지연: {avg_lat:.2f}s | RTF: {avg_rtf:.2f}x")
                print(f"│  💻 CPU: {cpu_pct:.1f}% | RAM: {ram_mb:.0f}MB | 시스템: {sys_ram_pct:.1f}%")
                print(f"│  ⏱  경과: {str(timedelta(seconds=int(elapsed)))} | "
                      f"속도: {speed:.2f}샘플/s | ETA: {eta.strftime('%H:%M:%S')}")
                print(f"└{'─'*68}", flush=True)

        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"  ⚠️ 실패 [{basename}]: {e}", flush=True)
            continue

    elapsed = time.time() - start_time
    print(f"\n  ✅ [{mode_label}] 완료!")
    print(f"     성공: {len(results):,}개 | 스킵: {skipped:,}개 | 실패: {failed:,}개")
    print(f"     소요: {str(timedelta(seconds=int(elapsed)))}")
    return results


def print_summary(df, mode_label):
    print(f"\n{'='*70}")
    print(f"  📊 [{mode_label}] 결과 요약")
    print(f"{'='*70}")
    print(f"  평가 샘플 수:       {len(df):,}개")
    print(f"\n  [정확도]")
    print(f"    평균 CER:         {df['cer'].mean()*100:.2f}%")
    print(f"    평균 WER:         {df['wer'].mean()*100:.2f}%")
    print(f"    중앙값 CER:       {df['cer'].median()*100:.2f}%")
    print(f"    CER 표준편차:     {df['cer'].std()*100:.2f}%")
    print(f"\n  [속도]")
    print(f"    평균 지연시간:    {df['latency_sec'].mean():.3f}초")
    print(f"    중앙값 지연:      {df['latency_sec'].median():.3f}초")
    print(f"    평균 RTF:         {df['rtf'].mean():.3f}x  (1.0 미만이 실시간)")
    print(f"    총 오디오 길이:   {df['audio_sec'].sum():.1f}초")
    print(f"    총 처리 시간:     {df['latency_sec'].sum():.1f}초")
    print(f"\n  [리소스]")
    print(f"    평균 CPU:         {df['cpu_pct'].mean():.1f}%")
    print(f"    최대 CPU:         {df['cpu_pct'].max():.1f}%")
    print(f"    평균 RAM:         {df['ram_mb'].mean():.1f} MB")
    print(f"    최대 RAM:         {df['ram_mb'].max():.1f} MB")
    print(f"    평균 시스템 RAM:  {df['sys_ram_pct'].mean():.1f}%")
    print(f"{'='*70}")


def main():
    try:
        import jiwer
    except ImportError:
        print("❌ pip install jiwer")
        return

    print("="*70)
    print(f"  🚀 Whisper Finetuned Model 평가 (011)")
    print(f"  모델: {MODEL_PATH} | 디바이스: {DEVICE} | Compute: {COMPUTE_TYPE}")
    print(f"  시작: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)

    if not os.path.exists(SAMPLES_DIR):
        print(f"❌ 폴더 없음: {SAMPLES_DIR}")
        return

    wav_files = sorted([f for f in os.listdir(SAMPLES_DIR) if f.endswith('.wav')])
    if not wav_files:
        print(f"❌ WAV 없음")
        return

    print(f"\n   샘플 수: {len(wav_files):,}개")

    print("\n📦 모델 로딩 중...")
    t0 = time.time()
    model = WhisperModel(MODEL_PATH, device=DEVICE, compute_type=COMPUTE_TYPE)
    print(f"  ✅ 로드 완료 ({time.time()-t0:.1f}초)")

    overall_start = time.time()

    # VAD OFF
    results_off = evaluate_mode(model, wav_files, SAMPLES_DIR, False, "VAD OFF", jiwer)
    df_off = pd.DataFrame(results_off)
    df_off.to_csv("results_finetuned_011_vad_off.csv", index=False, encoding="utf-8-sig")
    print(f"\n  💾 results_finetuned_011_vad_off.csv")
    print_summary(df_off, "FINETUNED 011 - VAD OFF")
    gc.collect()

    # VAD ON
    results_on = evaluate_mode(model, wav_files, SAMPLES_DIR, True, "VAD ON", jiwer)
    df_on = pd.DataFrame(results_on)
    df_on.to_csv("results_finetuned_011_vad_on.csv", index=False, encoding="utf-8-sig")
    print(f"\n  💾 results_finetuned_011_vad_on.csv")
    print_summary(df_on, "FINETUNED 011 - VAD ON")

    # 비교
    print(f"\n{'='*70}")
    print(f"  🆚 VAD OFF vs ON")
    print(f"{'='*70}")
    print(f"  {'지표':<20} {'VAD OFF':<15} {'VAD ON':<15} {'차이':<15}")
    print(f"  {'-'*65}")
    metrics = [
        ("평균 CER (%)", df_off['cer'].mean()*100, df_on['cer'].mean()*100),
        ("평균 WER (%)", df_off['wer'].mean()*100, df_on['wer'].mean()*100),
        ("평균 지연 (s)", df_off['latency_sec'].mean(), df_on['latency_sec'].mean()),
        ("평균 RTF (x)", df_off['rtf'].mean(), df_on['rtf'].mean()),
        ("평균 CPU (%)", df_off['cpu_pct'].mean(), df_on['cpu_pct'].mean()),
        ("최대 RAM (MB)", df_off['ram_mb'].max(), df_on['ram_mb'].max()),
    ]
    for name, off_val, on_val in metrics:
        diff = on_val - off_val
        sign = "+" if diff > 0 else ""
        print(f"  {name:<20} {off_val:<15.3f} {on_val:<15.3f} {sign}{diff:.3f}")

    total_elapsed = time.time() - overall_start
    print(f"\n  ⏱  전체: {str(timedelta(seconds=int(total_elapsed)))}")
    print(f"  완료: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()