"""
utils.py — shared utilities, fixed throughout all stages.
Do NOT modify after Stage 0 is executed.
"""
import re, wave, csv, json
import numpy as np
from pathlib import Path
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────
def get_audio_dir():
    """Resolve audio dir for both local and Kaggle environments."""
    candidates = [
        Path('MDD-Challenge-2025-training-set/audio_data/train'),
        Path('/kaggle/input/datasets/cquangnguynl/mdd-challenge/'
             'MDD-Challenge-2025-training-set/audio_data/train'),
        Path('/kaggle/input/mdd-challenge/'
             'MDD-Challenge-2025-training-set/audio_data/train'),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Audio directory not found. Check DATA_ROOT.")

def get_meta_dir():
    candidates = [
        Path('MDD-Challenge-2025-training-set/metadata'),
        Path('/kaggle/input/datasets/cquangnguynl/mdd-challenge/'
             'MDD-Challenge-2025-training-set/metadata'),
        Path('/kaggle/input/mdd-challenge/'
             'MDD-Challenge-2025-training-set/metadata'),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("Metadata directory not found.")

# Fixed valid speakers — never change after Stage 0
VALID_SPEAKERS = frozenset(['S0008', 'S0003'])

# ── Speaker parsing ─────────────────────────────────────────────────────────
def get_speaker_id(path: str) -> str:
    stem = Path(path).stem
    m = re.search(r'(S\d+)', stem)
    if m:
        return m.group(1)
    return 'TUYEN' if 'tuyen' in stem.lower() else 'ADULT'

# ── Phone normalization ─────────────────────────────────────────────────────
def norm_phones(s: str) -> str:
    """Strip $ and *, normalize whitespace. Mirrors evaluate.py exactly."""
    return ' '.join(str(s).replace('*', '').replace('$', '').split())

def tokenize(s: str) -> list:
    return norm_phones(s).split()

# ── Audio loading ───────────────────────────────────────────────────────────
def load_wav_f32(path) -> tuple:
    """Load WAV → float32 array in [-1, 1], returns (array, sample_rate)."""
    with wave.open(str(path), 'rb') as wf:
        sr  = wf.getframerate()
        sw  = wf.getsampwidth()
        nch = wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    scale = {1: 128.0,   2: 32768.0,  4: 2147483648.0}[sw]
    y = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    y = (y - 128.0) / 128.0 if sw == 1 else y / scale
    if nch > 1:
        y = y.reshape(-1, nch).mean(axis=1)
    return y.copy(), sr

# ── Audio preprocessing ─────────────────────────────────────────────────────
def normalize_amp(y: np.ndarray, target_peak: float = 0.9) -> np.ndarray:
    peak = np.abs(y).max()
    return y if peak < 1e-6 else y * (target_peak / peak)

def trim_silence(y: np.ndarray, sr: int = 16000,
                 threshold_db: float = -45.0,
                 frame_ms: int = 25, hop_ms: int = 10,
                 min_dur_sec: float = 0.3) -> np.ndarray:
    fl = int(sr * frame_ms / 1000)
    hl = int(sr * hop_ms  / 1000)
    energies = [np.mean(y[i:i+fl]**2) for i in range(0, max(1, len(y)-fl), hl)]
    db  = 10 * np.log10(np.array(energies) + 1e-10)
    act = db > threshold_db
    if not act.any():
        return y
    s = max(0, int(np.argmax(act)) * hl - fl)
    e = min(len(y), (len(act) - int(np.argmax(act[::-1]))) * hl + fl)
    trimmed = y[s:e]
    return y if len(trimmed) / sr < min_dur_sec else trimmed

# ── Augmentation ────────────────────────────────────────────────────────────
def aug_gain(y: np.ndarray, lo: float = -6., hi: float = 6.) -> np.ndarray:
    return np.clip(y * 10 ** (np.random.uniform(lo, hi) / 20.), -1., 1.)

def aug_noise(y: np.ndarray,
              snr_lo: float = 20., snr_hi: float = 35.,
              prob: float = 0.3) -> np.ndarray:
    if np.random.rand() > prob:
        return y
    sp  = np.mean(y ** 2) + 1e-10
    np_ = sp / 10 ** (np.random.uniform(snr_lo, snr_hi) / 10.)
    noise = np.random.normal(0., np_ ** 0.5, y.shape).astype(np.float32)
    return np.clip(y + noise, -1., 1.)

# ── Evaluation ──────────────────────────────────────────────────────────────
def evaluate_on_valid(gt_canonical: list, gt_transcript: list,
                      predictions: list, tag: str = '',
                      gt_path: str = '/tmp/_mdd_gt.csv',
                      res_path: str = '/tmp/_mdd_res.csv') -> dict:
    """Write temp CSVs, call evaluate.py, return metrics dict."""
    import sys, os
    # ensure evaluate.py is importable
    proj_root = Path(__file__).parent
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))
    import evaluate as ev

    with open(gt_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ['canonical', 'transcript'])
        w.writeheader()
        for c, t in zip(gt_canonical, gt_transcript):
            w.writerow({'canonical': c, 'transcript': t})
    with open(res_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, ['predict'])
        w.writeheader()
        for p in predictions:
            w.writerow({'predict': p})

    f1  = ev.compute_f1( gt_path, res_path)
    per = ev.compute_per(gt_path, res_path)
    der = ev.compute_der(gt_path, res_path)
    sc  = 0.5 * f1 + 0.4 * (1 - der) + 0.1 * (1 - per)
    if tag:
        print(f"{tag:45s}  F1={f1:.4f}  PER={per:.4f}  DER={der:.4f}  Score={sc:.4f}")
    return {'f1': f1, 'per': per, 'der': der, 'score': sc}

# ── CTC decoding ─────────────────────────────────────────────────────────────
def greedy_ctc(logits: np.ndarray, id2phone: list,
               phone2id: dict, blank_penalty: float = 0.0) -> list:
    adj = logits.copy()
    adj[..., phone2id['<blank>']] -= blank_penalty
    pred_ids = np.argmax(adj, axis=-1)
    results = []
    for seq in pred_ids:
        out, prev = [], None
        for i in seq:
            if i == prev:
                continue
            prev = int(i)
            if prev == phone2id['<blank>']:
                continue
            out.append(id2phone[prev] if prev < len(id2phone) else '<unk>')
        results.append(' '.join(out))
    return results

# ── Vocab I/O ────────────────────────────────────────────────────────────────
def load_vocab(path: str = 'splits/phone_vocab.json') -> tuple:
    with open(path, encoding='utf-8') as f:
        d = json.load(f)
    return d['id2phone'], d['phone2id']

def save_results(results_dict: dict, path: str = 'results/results_log.json'):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    existing = {}
    if Path(path).exists():
        with open(path) as f:
            existing = json.load(f)
    existing.update(results_dict)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {path}")
