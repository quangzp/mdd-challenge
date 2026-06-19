#!/usr/bin/env python3
"""
run_inference.py — Local inference for MDD Challenge test sets.

Usage:
  python3.10 run_inference.py public   → result_public.csv  (scored vs ground truth)
  python3.10 run_inference.py private  → result.csv

Pipeline:
  1. Run model on validation set to compute per-phoneme FP rates.
  2. Run model on target test set.
  3. Apply K=50 FP-rate-guided calibration.
  4. Write result CSV (and score if public test).
"""
import sys, csv, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings('ignore')

ROOT         = Path(__file__).parent
CKPT_DIR     = ROOT / 'checkpoint'
VOCAB_PATH   = ROOT / 'splits' / 'phone_vocab.json'
VALID_CSV    = ROOT / 'splits' / 'valid_phones.csv'
TRAIN_AUDIO  = ROOT / 'MDD-Challenge-2025-training-set' / 'audio_data' / 'train'

BATCH_SIZE   = 8
K_SUPPRESS   = 50
DEF_THR      = 0.90
MIN_FP       = 5

MODE = sys.argv[1] if len(sys.argv) > 1 else 'public'

if MODE == 'public':
    TEST_DIR   = ROOT / 'MDD-Challenge-2025-public-test'
    TEST_META  = TEST_DIR / 'metadata' / 'public_test_phones.csv'
    AUDIO_DIR  = TEST_DIR / 'audio_data' / 'public_test'
    OUTPUT_CSV = ROOT / 'result_public.csv'
    HAS_GT     = True   # public test has transcript ground truth
else:
    TEST_DIR   = ROOT / 'MDD-Challenge-2025-private-test'
    TEST_META  = TEST_DIR / 'metadata' / 'private_test_submission.csv'
    AUDIO_DIR  = TEST_DIR / 'audio_data' / 'private_test'
    OUTPUT_CSV = ROOT / 'result.csv'
    HAS_GT     = False

sys.path.insert(0, str(ROOT))
from utils import load_wav_f32, trim_silence
from evaluate import _align_pair

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Mode: {MODE} | Device: {DEVICE}')


# ── Load vocab + model ────────────────────────────────────────────────────────
vocab    = json.load(open(VOCAB_PATH, encoding='utf-8'))
id2phone = vocab['id2phone']
phone2id = vocab['phone2id']

from transformers import Wav2Vec2ForCTC, AutoFeatureExtractor
feat_ext = AutoFeatureExtractor.from_pretrained(str(CKPT_DIR))
model    = Wav2Vec2ForCTC.from_pretrained(str(CKPT_DIR)).to(DEVICE).eval()
print(f'Model loaded. Vocab size: {len(id2phone)}')


# ── Audio + inference helpers ─────────────────────────────────────────────────
import scipy.special

def load_audio(path):
    y, sr = load_wav_f32(str(path))
    assert sr == 16000, f"Expected 16kHz, got {sr}: {path}"
    return trim_silence(y, 16000)


def greedy_ctc_with_conf(logits):
    all_preds, all_confs = [], []
    for seq in logits:
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
        batch = [load_audio(p) for p in paths[start:start + BATCH_SIZE]]
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


# ── Step 1: Validation inference → FP rates ──────────────────────────────────
print('\n── Step 1: Validation inference (FP rate calibration) ──')
valid_df  = pd.read_csv(VALID_CSV)
VALID_C   = valid_df['c_norm'].tolist()
VALID_T   = valid_df['t_norm'].tolist()
val_paths = [TRAIN_AUDIO / Path(p).name for p in valid_df['path']]

val_preds, _ = run_batched(val_paths, 'Valid')

from collections import Counter
occ      = Counter()
fp_count = Counter()
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
                fp_count[rs[i] if rs[i] != '<eps>' else '<blank_pos>'] += 1
            flag += 1

fp_rates  = {ph: fp_count[ph] / occ[ph] for ph in fp_count if occ.get(ph, 0) > 0}
eligible  = [(ph, r) for ph, r in fp_rates.items() if fp_count.get(ph, 0) >= MIN_FP]
sorted_ph = [ph for ph, _ in sorted(eligible, key=lambda x: -x[1])]
suppress  = set(sorted_ph[:K_SUPPRESS])
active    = sorted_ph[K_SUPPRESS:]
print(f'Eligible: {len(eligible)} phonemes | K={K_SUPPRESS} suppressed | active ({len(active)}): {active}')


# ── Step 2: Test set inference ────────────────────────────────────────────────
print(f'\n── Step 2: {MODE} test inference ──')
test_df    = pd.read_csv(TEST_META)
test_paths = [AUDIO_DIR / Path(row['path']).name for _, row in test_df.iterrows()]
test_canon = test_df['canonical'].tolist()

missing = [p for p in test_paths if not p.exists()]
if missing:
    print(f'WARNING: {len(missing)} files not found, e.g. {missing[0]}')

test_preds, test_confs = run_batched(test_paths, MODE.capitalize())


# ── Step 3: K=50 calibration ─────────────────────────────────────────────────
def _align2(s1, s2):
    r = _align_pair(s1, s2)
    return r[0], r[1]


def calibrate(gt_c, preds, confs, suppress_set, default_thr=0.90):
    out = []
    for c, pred, conf in zip(gt_c, preds, confs):
        if not pred.strip():
            out.append(c); continue
        ca, pa = _align2(c, pred)
        conf_map = dict(enumerate(conf))
        new_tokens, p_idx = [], 0
        for ct, pt in zip(ca, pa):
            if pt == '<eps>': continue
            cur_conf = conf_map.get(p_idx, 1.0)
            p_idx += 1
            if ct != '<eps>' and pt != ct:
                thr = 1.0 if ct in suppress_set else default_thr
                new_tokens.append(ct if cur_conf < thr else pt)
            else:
                new_tokens.append(pt)
        out.append(' '.join(new_tokens))
    return out


calibrated = calibrate(test_canon, test_preds, test_confs, suppress, DEF_THR)


# ── Step 4: Score (public test only) ─────────────────────────────────────────
if HAS_GT:
    from utils import evaluate_on_valid
    test_transcript = test_df['transcript'].tolist()
    m = evaluate_on_valid(test_canon, test_transcript, calibrated, tag='Public test (K=50 cal)')
    print(f'\nPublic test score: F1={m["f1"]:.4f}  PER={m["per"]:.4f}  DER={m["der"]:.4f}  Score={m["score"]:.4f}')

    # Also show baseline (no calibration)
    m_base = evaluate_on_valid(test_canon, test_transcript, test_preds, tag='Public test (baseline)')
    print(f'Baseline score:    F1={m_base["f1"]:.4f}  PER={m_base["per"]:.4f}  DER={m_base["der"]:.4f}  Score={m_base["score"]:.4f}')


# ── Step 5: Write CSV ─────────────────────────────────────────────────────────
print(f'\n── Writing {OUTPUT_CSV} ──')
with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['id', 'path', 'predict'])
    for i, (_, row) in enumerate(test_df.iterrows()):
        w.writerow([row['id'], row['path'], calibrated[i]])

n_diff = sum(1 for c, p in zip(test_canon, calibrated) if c != p)
print(f'Done. {len(calibrated)} rows → {OUTPUT_CSV}')
print(f'Flagged as error: {n_diff}/{len(calibrated)} ({n_diff/len(calibrated)*100:.1f}%)')
