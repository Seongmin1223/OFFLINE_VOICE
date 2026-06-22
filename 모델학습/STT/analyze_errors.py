import pandas as pd
import difflib
from collections import Counter
import re

def get_errors(true_text, pred_text):
    true_text = str(true_text).replace(' ', '')
    pred_text = str(pred_text).replace(' ', '')
    
    matcher = difflib.SequenceMatcher(None, true_text, pred_text)
    missing = [] # Characters in true_text that were not predicted correctly
    inserted = [] # Characters in pred_text that were not in true_text
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'replace':
            missing.extend(list(true_text[i1:i2]))
            inserted.extend(list(pred_text[j1:j2]))
        elif tag == 'delete':
            missing.extend(list(true_text[i1:i2]))
        elif tag == 'insert':
            inserted.extend(list(pred_text[j1:j2]))
            
    return missing, inserted

def main():
    df_base = pd.read_csv('results_base_011_vad_off.csv')
    df_fine = pd.read_csv('results_finetuned_011_vad_off.csv')
    
    df_merged = pd.merge(df_base[['id', 'true_text', 'pred_text']], 
                         df_fine[['id', 'pred_text']], 
                         on=['id'], suffixes=('_base', '_fine'))
                         
    base_missing = Counter()
    base_inserted = Counter()
    fine_missing = Counter()
    fine_inserted = Counter()
    
    for _, row in df_merged.iterrows():
        t = row['true_text']
        pb = row['pred_text_base']
        pf = row['pred_text_fine']
        
        bm, bi = get_errors(t, pb)
        fm, fi = get_errors(t, pf)
        
        base_missing.update(bm)
        base_inserted.update(bi)
        fine_missing.update(fm)
        fine_inserted.update(fi)
        
    # Find mitigated errors (high in base, low in fine)
    # Only consider Korean characters
    def is_korean(char):
        return bool(re.match(r'[가-힣]', char))
        
    chars = set(list(base_missing.keys()) + list(fine_missing.keys()))
    chars = [c for c in chars if is_korean(c)]
    
    stats = []
    for c in chars:
        b_err = base_missing[c]
        f_err = fine_missing[c]
        diff = b_err - f_err
        stats.append((c, b_err, f_err, diff))
        
    # Sort by absolute improvement
    stats.sort(key=lambda x: x[3], reverse=True)
    
    print("=== 가장 많이 완화된 글자 (Base 오류 -> Fine 오류) ===")
    for c, b, f, d in stats[:20]:
        if d > 0:
            print(f"'{c}': {b}회 -> {f}회 (감소: {d})")
            
    print("\n=== 아직 완화되지 않은 (오류가 여전히 많은) 글자 ===")
    # Sort by remaining errors
    stats.sort(key=lambda x: x[2], reverse=True)
    for c, b, f, d in stats[:20]:
        if f > 0:
            print(f"'{c}': {b}회 -> {f}회 (감소: {d})")
            
    print("\n=== 파인튜닝 후 오류가 오히려 증가한 글자 ===")
    stats.sort(key=lambda x: x[3])
    for c, b, f, d in stats[:20]:
        if d < 0:
            print(f"'{c}': {b}회 -> {f}회 (증가: {-d})")

if __name__ == '__main__':
    main()
