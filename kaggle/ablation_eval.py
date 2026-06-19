#!/usr/bin/env python3
"""
ablation_eval.py — Inference-time ablation evaluation using the best checkpoint.

Evaluates each preprocessing variant (A-F) on the validation set WITHOUT retraining.
- A (normalize_amp) and B (trim_silence): true inference-time preprocessing changes
- C (gain) and D (noise): fixed-value approximations (not random like training)
- E (SpecAugment) and F (no_adult): training-time only — skipped, scored as N/A

Saves results to results/ablation_results.json.
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import scipy.special

warnings.filterwarnings('ignore')

ROOT        = Path(__file__).parent
CKPT_DIR    = ROOT.parent / 'checkpoint'
VOCAB_PATH  = ROOT / 'splits' / 'phone_vocab.json'
VALID_CSV   = ROOT / 'splits' / 'valid_phones.csv'
TRAIN_AUDIO = ROOT / 'MDD-Challenge-2025-training-set' / 'audio_data' / 'train'
OUT_JSON    = ROOT / 'results' / 'ablation_results.json'

sys.path.insert(0, str(ROOT))
from utils import load_wav_f32, trim_silence, normalize_amp, evaluate_on_valid

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

vocab    = json.load(open(VOCAB_PATH, encoding='utf-8'))
id2phone = vocab['id2phone']
phone2id = vocab['phone2id']
from transformers import Wav2Vec2ForCTC, AutoFeatureExtractor
feat_ext = AutoFeatureExtractor.from_pretrained(str(CKPT_DIR))
model    = Wav2Vec2ForCTC.from_pretrained(str(CKPT_DIR)).to(DEVICE).eval()
print('Model loaded.')


# ── Preprocessing variants ────────────────────────────────────────────────────
def preproc_baseline(y): return y                                # Baseline: raw audio

def preproc_A(y): return normalize_amp(y)                        # A: normalize_amp only

def preproc_B(y): return trim_silence(y)                         # B: trim_silence only

def preproc_C(y):                                                # C: fixed +3 dB gain (midpoint of ±6)
    return np.clip(y * 10 ** (3.0 / 20.), -1., 1.)

def preproc_D(y):                                                # D: fixed SNR=27.5 dB noise (midpoint of 20-35)
    sp    = np.mean(y ** 2) + 1e-10
    np_   = sp / 10 ** (27.5 / 10.)
    noise = np.random.default_rng(42).normal(0., np_ ** 0.5, y.shape).astype(np.float32)
    return np.clip(y + noise, -1., 1.)

def preproc_final(y): return normalize_amp(trim_silence(y))      # Final inference pipeline


VARIANTS = [
    ('Baseline', preproc_baseline),
    ('A: normalize_amp', preproc_A),
    ('B: trim_silence', preproc_B),
    ('C: gain +3dB (approx)', preproc_C),
    ('D: noise SNR27dB (approx)', preproc_D),
    ('Final pipeline (trim+norm)', preproc_final),
]


# ── Inference for one variant ─────────────────────────────────────────────────
def run_variant(name, preproc_fn, valid_df):
    print(f'  Running {name} ...', flush=True)
    paths = [TRAIN_AUDIO / Path(p).name for p in valid_df['path']]
    preds = []
    for i, p in enumerate(paths):
        y, sr = load_wav_f32(str(p))
        y = preproc_fn(y)
        inp = feat_ext([y], sampling_rate=16000, return_tensors='pt', padding=True)
        with torch.no_grad():
            logits = model(inp.input_values.to(DEVICE)).logits.cpu().numpy()
        ids  = np.argmax(logits[0], axis=-1)
        out, prev = [], None
        for t, j in enumerate(ids):
            if j == prev: continue
            prev = int(j)
            if prev == phone2id['<blank>']: continue
            out.append(id2phone[prev] if prev < len(id2phone) else '<unk>')
        preds.append(' '.join(out))
        if (i + 1) % 50 == 0:
            print(f'    {i+1}/{len(paths)}', flush=True)
    print(f'    {len(paths)}/{len(paths)} done', flush=True)
    m = evaluate_on_valid(
        valid_df['c_norm'].tolist(),
        valid_df['t_norm'].tolist(),
        preds, tag=name
    )
    print(f'    F1={m["f1"]:.4f}  PER={m["per"]:.4f}  DER={m["der"]:.4f}  Score={m["score"]:.4f}')
    return m


# ── Main ──────────────────────────────────────────────────────────────────────
valid_df = pd.read_csv(VALID_CSV)
print(f'Validation set: {len(valid_df)} samples\n')

results = {}
for name, fn in VARIANTS:
    results[name] = run_variant(name, fn, valid_df)
    print()

# E and F cannot be evaluated at inference time
results['E: SpecAugment'] = {
    'note': 'Training-time only. Disabled in final model (ENABLE_SPEC_AUGMENT=False).'
}
results['F: no_adult'] = {
    'note': 'Training data selection. Requires retraining with filtered dataset.'
}

OUT_JSON.parent.mkdir(exist_ok=True)
with open(OUT_JSON, 'w') as f:
    json.dump(results, f, indent=2)

print(f'Results saved to {OUT_JSON}')
print()
print(f'{"Variant":<34} {"Score":>7}')
print('-' * 44)
for name, _ in VARIANTS:
    m = results[name]
    print(f'{name:<34} {m["score"]:7.4f}')
print(f'{"E: SpecAugment":<34} {"N/A":>7}')
print(f'{"F: no_adult":<34} {"N/A":>7}')
