#!/usr/bin/env python3
"""
run_fp_analysis.py — Generate FP analysis and calibration sweep for LaTeX report.

Outputs: results/fp_analysis.json
  - fp_rates: top-15 phonemes by FP rate
  - k_sweep:  Score vs K (tau=0.90 fixed)
  - tau_sweep: Score vs tau (K=50 fixed)
"""
import sys, csv, json, warnings
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import scipy.special

warnings.filterwarnings('ignore')

ROOT        = Path(__file__).parent
CKPT_DIR    = ROOT / 'checkpoint'
VOCAB_PATH  = ROOT / 'splits' / 'phone_vocab.json'
VALID_CSV   = ROOT / 'splits' / 'valid_phones.csv'
TRAIN_AUDIO = ROOT / 'MDD-Challenge-2025-training-set' / 'audio_data' / 'train'
OUT_JSON    = ROOT / 'results' / 'fp_analysis.json'

BATCH_SIZE = 8
MIN_FP     = 5

sys.path.insert(0, str(ROOT))
from utils import load_wav_f32, trim_silence, evaluate_on_valid
from evaluate import _align_pair

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ── Load vocab + model ────────────────────────────────────────────────────────
vocab    = json.load(open(VOCAB_PATH, encoding='utf-8'))
id2phone = vocab['id2phone']
phone2id = vocab['phone2id']

from transformers import Wav2Vec2ForCTC, AutoFeatureExtractor
feat_ext = AutoFeatureExtractor.from_pretrained(str(CKPT_DIR))
model    = Wav2Vec2ForCTC.from_pretrained(str(CKPT_DIR)).to(DEVICE).eval()
print(f'Model loaded. Vocab: {len(id2phone)}')


# ── Inference helpers ─────────────────────────────────────────────────────────
def load_audio(path):
    y, sr = load_wav_f32(str(path))
    return trim_silence(y, 16000)

def greedy_ctc_with_conf(logits_batch):
    all_preds, all_confs = [], []
    for seq in logits_batch:
        probs    = scipy.special.softmax(seq, axis=-1)
        pred_ids = np.argmax(seq, axis=-1)
        out_ph, out_cf, prev = [], [], None
        for t, i in enumerate(pred_ids):
            if i == prev: continue
            prev = int(i)
            if prev == phone2id['<blank>']: continue
            out_ph.append(id2phone[prev] if prev < len(id2phone) else '<unk>')
            out_cf.append(float(probs[t, prev]))
        all_preds.append(' '.join(out_ph))
        all_confs.append(out_cf)
    return all_preds, all_confs

def run_batched(paths, tag=''):
    all_preds, all_confs = [], []
    n = len(paths)
    for start in range(0, n, BATCH_SIZE):
        batch  = [load_audio(p) for p in paths[start:start + BATCH_SIZE]]
        inputs = feat_ext(batch, sampling_rate=16000, return_tensors='pt', padding=True)
        iv   = inputs.input_values.to(DEVICE)
        attn = inputs.get('attention_mask')
        if attn is not None: attn = attn.to(DEVICE)
        with torch.no_grad():
            logits = model(iv, attention_mask=attn).logits.cpu().numpy()
        p, c = greedy_ctc_with_conf(logits)
        all_preds.extend(p)
        all_confs.extend(c)
        print(f'  {tag}: {min(start + BATCH_SIZE, n)}/{n}', end='\r', flush=True)
    print()
    return all_preds, all_confs


# ── Calibration function (suppress_set + threshold) ──────────────────────────
def calibrate(gt_c, preds, confs, suppress_set, thr=0.90):
    out = []
    for c, pred, conf in zip(gt_c, preds, confs):
        if not pred.strip():
            out.append(c); continue
        rs, hs, _ = _align_pair(c, pred)
        ca = rs; pa = hs
        conf_map = dict(enumerate(conf))
        new_tokens, p_idx = [], 0
        for ct, pt in zip(ca, pa):
            if pt == '<eps>': continue
            cur_conf = conf_map.get(p_idx, 1.0)
            p_idx += 1
            if ct != '<eps>' and pt != ct:
                thresh = 1.0 if ct in suppress_set else thr
                new_tokens.append(ct if cur_conf < thresh else pt)
            else:
                new_tokens.append(pt)
        out.append(' '.join(new_tokens))
    return out


# ── Step 1: Validation inference ─────────────────────────────────────────────
print('\n── Step 1: Validation inference ──')
valid_df  = pd.read_csv(VALID_CSV)
VALID_C   = valid_df['c_norm'].tolist()
VALID_T   = valid_df['t_norm'].tolist()
val_paths = [TRAIN_AUDIO / Path(p).name for p in valid_df['path']]
val_preds, val_confs = run_batched(val_paths, 'Valid')
print(f'Done: {len(val_preds)} predictions')


# ── Step 2: Per-phoneme FP rates + confusion ──────────────────────────────────
print('\n── Step 2: Per-phoneme FP rate analysis ──')
occ      = Counter()
fp_count = Counter()
confusion = defaultdict(Counter)

for c in VALID_C:
    for tok in c.replace('*','').replace('$','').split():
        occ[tok] += 1

for c, t, p in zip(VALID_C, VALID_T, val_preds):
    rs, hs, op_rh = _align_pair(c, t)
    hs2, os2, op_ho = _align_pair(t, p)
    flag = 0
    for i in range(len(hs)):
        if hs[i] == '<eps>': continue
        while flag < len(hs2) and hs2[flag] == '<eps>': flag += 1
        if flag < len(hs2) and hs[i] == hs2[flag]:
            if op_rh[i] == 'C' and op_ho[flag] != 'C':
                canon   = rs[i]   if rs[i]   != '<eps>' else '<blank_pos>'
                pred_ph = os2[flag] if os2[flag] != '<eps>' else '<deleted>'
                fp_count[canon] += 1
                confusion[canon][pred_ph] += 1
            flag += 1

fp_rates = {ph: fp_count[ph] / occ[ph] for ph in fp_count if occ.get(ph, 0) > 0}
eligible = [(ph, r) for ph, r in fp_rates.items() if fp_count.get(ph, 0) >= MIN_FP]
sorted_ph = [ph for ph, _ in sorted(eligible, key=lambda x: -x[1])]

print(f'Eligible: {len(eligible)} phonemes (>= {MIN_FP} FP)')
print(f'\n{"Phoneme":14s}  {"FP":5s}  {"Occ":5s}  {"Rate":6s}  Top confusion')
print('-' * 60)
fp_table = []
for ph in sorted_ph[:15]:
    rate = fp_rates[ph]
    fp_c = fp_count[ph]
    oc   = occ[ph]
    top1 = confusion[ph].most_common(1)[0] if confusion[ph] else ('-', 0)
    pct  = top1[1] / fp_c * 100 if fp_c > 0 else 0
    print(f'  {ph:12s}  {fp_c:5d}  {oc:5d}  {rate:6.3f}  {top1[0]} ({pct:.0f}%)')
    fp_table.append({
        'phoneme': ph, 'fp': fp_c, 'occ': oc,
        'fp_rate': round(rate, 4),
        'top_confusion': top1[0], 'top_confusion_pct': round(pct, 1)
    })


# ── Step 3: K sweep (tau=0.90 fixed) ─────────────────────────────────────────
print('\n── Step 3: K sweep (tau=0.90) ──')
K_VALUES = [0, 10, 20, 30, 40, 50, 60, 70, 80]
k_sweep  = []

print(f'{"K":4s}  {"Score":7s}  {"F1":7s}  {"PER":7s}  {"DER":7s}')
print('-' * 42)
for K in K_VALUES:
    sup_set = set(sorted_ph[:K]) if K > 0 else set()
    cal = calibrate(VALID_C, val_preds, val_confs, sup_set, thr=0.90)
    m   = evaluate_on_valid(VALID_C, VALID_T, cal)
    print(f'{K:4d}  {m["score"]:.4f}  {m["f1"]:.4f}  {m["per"]:.4f}  {m["der"]:.4f}')
    k_sweep.append({'K': K, 'score': round(m['score'], 4),
                    'f1': round(m['f1'], 4), 'per': round(m['per'], 4),
                    'der': round(m['der'], 4)})


# ── Step 4: tau sweep (K=50 fixed) ───────────────────────────────────────────
print('\n── Step 4: tau sweep (K=50) ──')
TAU_VALUES = [0.80, 0.82, 0.85, 0.87, 0.90, 0.92, 0.95]
tau_sweep  = []
sup50      = set(sorted_ph[:50])

print(f'{"tau":6s}  {"Score":7s}  {"F1":7s}  {"PER":7s}  {"DER":7s}')
print('-' * 46)
for tau in TAU_VALUES:
    cal = calibrate(VALID_C, val_preds, val_confs, sup50, thr=tau)
    m   = evaluate_on_valid(VALID_C, VALID_T, cal)
    print(f'{tau:6.2f}  {m["score"]:.4f}  {m["f1"]:.4f}  {m["per"]:.4f}  {m["der"]:.4f}')
    tau_sweep.append({'tau': tau, 'score': round(m['score'], 4),
                      'f1': round(m['f1'], 4), 'per': round(m['per'], 4),
                      'der': round(m['der'], 4)})


# ── Save results ──────────────────────────────────────────────────────────────
OUT_JSON.parent.mkdir(exist_ok=True)
results = {
    'fp_table': fp_table,
    'k_sweep':  k_sweep,
    'tau_sweep': tau_sweep,
}
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)
print(f'\nSaved → {OUT_JSON}')
