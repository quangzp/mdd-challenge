"""
Stage 0 — Fixed Split + Vocab
Run locally (no GPU needed). Creates artifacts in splits/ used by all later stages.
"""
import json
import pandas as pd
from pathlib import Path
from collections import Counter
from utils import (get_meta_dir, get_speaker_id, norm_phones,
                   tokenize, VALID_SPEAKERS, evaluate_on_valid, save_results)

Path('splits').mkdir(exist_ok=True)
Path('results').mkdir(exist_ok=True)

META = get_meta_dir()

# ── Load raw data ────────────────────────────────────────────────────────────
df_t = pd.read_csv(META / 'train.csv')
df_p = pd.read_csv(META / 'train_phones.csv')
print(f"Loaded {len(df_t)} samples")

# ── Enrich ───────────────────────────────────────────────────────────────────
df_t['speaker_id'] = df_t['path'].map(get_speaker_id)
df_p['speaker_id'] = df_t['speaker_id'].values
df_p['c_norm']     = df_p['canonical'].map(norm_phones)
df_p['t_norm']     = df_p['transcript'].map(norm_phones)
df_p['ph_error']   = (df_p['c_norm'] != df_p['t_norm']).astype(bool)

# ── Split ────────────────────────────────────────────────────────────────────
valid_mask = df_t['speaker_id'].isin(VALID_SPEAKERS)
train_df = df_t[~valid_mask].copy().reset_index(drop=True)
valid_df = df_t[ valid_mask].copy().reset_index(drop=True)
train_ph = df_p[~valid_mask].copy().reset_index(drop=True)
valid_ph = df_p[ valid_mask].copy().reset_index(drop=True)

# ── Hard assertions ──────────────────────────────────────────────────────────
assert set(train_df['speaker_id']).isdisjoint(VALID_SPEAKERS), \
    "FATAL: speaker leak between train and valid!"
assert len(valid_df) == 460,  f"Expected 460 valid, got {len(valid_df)}"
assert len(train_df) == 2720, f"Expected 2720 train, got {len(train_df)}"
assert abs(valid_ph['ph_error'].mean() - 0.100) < 0.005, \
    f"Valid error rate drifted: {valid_ph['ph_error'].mean():.4f}"
assert not train_ph['t_norm'].isna().any(), "NaN in train transcript phones"
assert not valid_ph['t_norm'].isna().any(), "NaN in valid transcript phones"

print(f"Train: {len(train_df)} | Valid: {len(valid_df)}")
print(f"Train error rate: {train_ph['ph_error'].mean():.4f}")
print(f"Valid error rate: {valid_ph['ph_error'].mean():.4f}  "
      f"(S0008: {valid_ph[valid_df['speaker_id']=='S0008']['ph_error'].mean():.4f}, "
      f"S0003: {valid_ph[valid_df['speaker_id']=='S0003']['ph_error'].mean():.4f})")

# ── Save split ───────────────────────────────────────────────────────────────
train_df.to_csv('splits/train_ids.csv',    index=False)
valid_df.to_csv('splits/valid_ids.csv',    index=False)
train_ph.to_csv('splits/train_phones.csv', index=False)
valid_ph.to_csv('splits/valid_phones.csv', index=False)
print("Split CSV files saved.")

# ── Build vocab (from train only) ────────────────────────────────────────────
BLANK, UNK = '<blank>', '<unk>'
counter = Counter()
# Transcript phones from TRAIN only (CTC target — not available at inference)
for s in train_ph['t_norm']:
    counter.update(tokenize(s))
# Canonical phones from ALL data (always available at inference)
for s in df_p['c_norm']:
    counter.update(tokenize(s))

id2phone = [BLANK, UNK] + sorted(counter.keys())
phone2id = {p: i for i, p in enumerate(id2phone)}

# Assertions: zero OOV in both splits
oov_train = sum(1 for s in train_ph['t_norm']
                for t in tokenize(s) if t not in phone2id)
oov_valid = sum(1 for s in valid_ph['t_norm']
                for t in tokenize(s) if t not in phone2id)
oov_canon = sum(1 for s in df_p['c_norm']
                for t in tokenize(s) if t not in phone2id)
assert oov_train == 0, f"OOV tokens in train targets: {oov_train}"
assert oov_valid == 0, f"OOV tokens in valid targets: {oov_valid}"
assert oov_canon == 0, f"OOV tokens in canonical: {oov_canon}"

with open('splits/phone_vocab.json', 'w', encoding='utf-8') as f:
    json.dump({'id2phone': id2phone, 'phone2id': phone2id}, f,
              ensure_ascii=False, indent=2)
print(f"Vocab size: {len(id2phone)}  (blank=0, unk=1, phones=2..{len(id2phone)-1})")

# ── Compute & store naive baselines ─────────────────────────────────────────
valid_c = valid_ph['c_norm'].tolist()
valid_t = valid_ph['t_norm'].tolist()

b0 = evaluate_on_valid(valid_c, valid_t, valid_c,
                       "B0 predict=canonical (no-error baseline)")
b1 = evaluate_on_valid(valid_c, valid_t, valid_t,
                       "B1 oracle=transcript  (upper bound)")
b2 = evaluate_on_valid(valid_c, valid_t, ['' for _ in valid_t],
                       "B2 predict=empty      (blank collapse)")

save_results({
    'naive_B0_canonical': b0,
    'naive_B1_oracle':    b1,
    'naive_B2_empty':     b2,
    'split_info': {
        'valid_speakers': sorted(VALID_SPEAKERS),
        'n_train': len(train_df),
        'n_valid': len(valid_df),
        'train_error_rate': round(float(train_ph['ph_error'].mean()), 4),
        'valid_error_rate': round(float(valid_ph['ph_error'].mean()), 4),
        'vocab_size': len(id2phone),
    }
})

# ── Speaker distribution summary ─────────────────────────────────────────────
print("\n=== Train speaker summary ===")
for spk in sorted(train_df['speaker_id'].unique()):
    mask = train_df['speaker_id'] == spk
    n = mask.sum()
    err = train_ph.loc[mask.values, 'ph_error'].mean()
    print(f"  {spk:12s}  n={n:3d}  err={err:.3f}")

print("\nStage 0 complete. All artifacts in splits/ and results/")
