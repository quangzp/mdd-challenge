#!/usr/bin/env python3
"""
train.py — CLI training script for MDD Challenge 2025.

Usage:
  python train.py
  python train.py --epochs 30 --output-dir checkpoint

Prerequisites:
  1. python prepare_data.py   (creates splits/ and splits/phone_vocab.json)
  2. GPU recommended (CPU will work but is very slow)

Output:
  <output-dir>/   — best checkpoint (model.safetensors, config.json, preprocessor_config.json)
  <output-dir>/history.json  — per-epoch validation metrics
"""
import argparse, json, math, re, warnings, wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.special
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from transformers import (
    AutoFeatureExtractor, TrainerCallback, TrainingArguments, Trainer,
    Wav2Vec2ForCTC, get_linear_schedule_with_warmup,
)
try:
    from transformers.trainer_utils import LengthGroupedSampler
except ImportError:
    from transformers.trainer_pt_utils import LengthGroupedSampler

warnings.filterwarnings('ignore')

# ── Constants ─────────────────────────────────────────────────────────────────
BASE_MODEL   = 'nguyenvulebinh/wav2vec2-base-vietnamese-250h'
EPOCHS       = 30
BATCH_SIZE   = 4
GRAD_ACCUM   = 4          # effective batch = 16
ENCODER_LR   = 2e-5
HEAD_LR      = 2e-3
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.10
SEED         = 42
INFER_BATCH  = 16
ENABLE_SPEC_AUGMENT = False

BLANK = '<blank>'
UNK   = '<unk>'
VALID_SPEAKERS = frozenset(['S0008', 'S0003'])

torch.manual_seed(SEED)
np.random.seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Audio utilities ───────────────────────────────────────────────────────────
def load_wav_f32(path):
    with wave.open(str(path), 'rb') as wf:
        sr, sw, nch = wf.getframerate(), wf.getsampwidth(), wf.getnchannels()
        frames = wf.readframes(wf.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[sw]
    scale = {1: 128.0,   2: 32768.0,  4: 2147483648.0}[sw]
    y = np.frombuffer(frames, dtype=dtype).astype(np.float32)
    y = (y - 128.0) / 128.0 if sw == 1 else y / scale
    if nch > 1:
        y = y.reshape(-1, nch).mean(axis=1)
    return y.copy(), sr


def normalize_amp(y, target_peak=0.9):
    peak = np.abs(y).max()
    return y if peak < 1e-6 else y * (target_peak / peak)


def trim_silence(y, sr=16000, threshold_db=-45.0,
                 frame_ms=25, hop_ms=10, min_dur_sec=0.3):
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


def aug_gain(y, lo=-6., hi=6.):
    return np.clip(y * 10 ** (np.random.uniform(lo, hi) / 20.), -1., 1.)


def aug_noise(y, snr_lo=20., snr_hi=35., prob=0.3):
    if np.random.rand() > prob:
        return y
    sp  = np.mean(y ** 2) + 1e-10
    np_ = sp / 10 ** (np.random.uniform(snr_lo, snr_hi) / 10.)
    return np.clip(y + np.random.normal(0., np_**0.5, y.shape).astype(np.float32), -1., 1.)


def preproc(y):
    return normalize_amp(trim_silence(y))


def augment(y):
    return aug_noise(aug_gain(y))


def norm_phones(s):
    return ' '.join(str(s).replace('*', '').replace('$', '').split())


def tokenize(s):
    return norm_phones(s).split()


def get_speaker_id(path):
    m = re.search(r'(S\d+)', Path(path).stem)
    if m:
        return m.group(1)
    return 'TUYEN' if 'tuyen' in str(path).lower() else 'ADULT'


# ── Evaluation ────────────────────────────────────────────────────────────────
def _align(seq1, seq2):
    GAP = -1; MATCH = 1; MISMATCH = -1
    n, m = len(seq1), len(seq2)
    score = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): score[i][0] = GAP * i
    for j in range(n + 1): score[0][j] = GAP * j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            s = MATCH if seq1[j-1] == seq2[i-1] else \
                GAP if (seq1[j-1] == '<eps>' or seq2[i-1] == '<eps>') else MISMATCH
            score[i][j] = max(score[i-1][j-1]+s, score[i-1][j]+GAP, score[i][j-1]+GAP)
    align1, align2 = [], []
    i, j = m, n
    while i > 0 and j > 0:
        s = MATCH if seq1[j-1] == seq2[i-1] else \
            GAP if (seq1[j-1] == '<eps>' or seq2[i-1] == '<eps>') else MISMATCH
        if score[i][j] == score[i-1][j-1] + s:
            align1.append(seq1[j-1]); align2.append(seq2[i-1]); i -= 1; j -= 1
        elif score[i][j] == score[i][j-1] + GAP:
            align1.append(seq1[j-1]); align2.append('<eps>'); j -= 1
        else:
            align1.append('<eps>'); align2.append(seq2[i-1]); i -= 1
    while j > 0: align1.append(seq1[j-1]); align2.append('<eps>'); j -= 1
    while i > 0: align1.append('<eps>'); align2.append(seq2[i-1]); i -= 1
    align1.reverse(); align2.reverse()
    return align1, align2


def _ops(a1, a2):
    ops = []
    for r, h in zip(a1, a2):
        if   r != '<eps>' and h == '<eps>': ops.append('D')
        elif r == '<eps>' and h != '<eps>': ops.append('I')
        elif r != h:                        ops.append('S')
        else:                               ops.append('C')
    return ops


def _align_pair(s1, s2):
    seq1 = s1.replace('*','').replace('$','').split()
    seq2 = s2.replace('*','').replace('$','').split()
    a1, a2 = _align(seq1, seq2)
    return a1, a2, _ops(a1, a2)


def _score_preds(gt_c, gt_t, preds):
    cor_cor=cor_nocor=0
    sub_sub=sub_sub1=sub_nosub=0
    ins_ins=ins_ins1=ins_noins=0
    del_del=del_del1=del_nodel=0
    total_sub=total_del=total_ins=total_cor_per=0

    for c, t, p in zip(gt_c, gt_t, preds):
        rs, hs, op_rh   = _align_pair(c, t)
        hs2, os2, op_ho = _align_pair(t, p)
        rs3, os3, op_ro = _align_pair(c, p)

        total_sub += op_ho.count('S')
        total_del += op_ho.count('D')
        total_ins += op_ho.count('I')
        total_cor_per += op_ho.count('C')

        flag = 0
        for i in range(len(rs)):
            if rs[i] == '<eps>': continue
            while flag < len(rs3) and rs3[flag] == '<eps>': flag += 1
            if flag < len(rs3) and rs[i] == rs3[flag]:
                if   op_rh[i]=='D' and op_ro[flag]=='D':              del_del  += 1
                elif op_rh[i]=='D' and op_ro[flag] not in ('D','C'): del_del1 += 1
                elif op_rh[i]=='D' and op_ro[flag]=='C':              del_nodel+= 1
                flag += 1

        flag = 0
        for i in range(len(hs)):
            if hs[i] == '<eps>': continue
            while flag < len(hs2) and hs2[flag] == '<eps>': flag += 1
            if flag < len(hs2) and hs[i] == hs2[flag]:
                if   op_rh[i]=='C' and op_ho[flag]=='C':  cor_cor  += 1
                elif op_rh[i]=='C' and op_ho[flag]!='C':  cor_nocor+= 1
                if   op_rh[i]=='S' and op_ho[flag]=='C':  sub_sub  += 1
                elif op_rh[i]=='S' and op_ho[flag]!='C' and rs[i]!=os2[flag]: sub_sub1 += 1
                elif op_rh[i]=='S' and op_ho[flag]!='C' and rs[i]==os2[flag]: sub_nosub+= 1
                if   op_rh[i]=='I' and op_ho[flag]=='C':  ins_ins  += 1
                elif op_rh[i]=='I' and op_ho[flag]!='C' and op_ho[flag]!='D': ins_ins1 += 1
                elif op_rh[i]=='I' and op_ho[flag]=='D':  ins_noins+= 1
                flag += 1

    TR = sub_sub+sub_sub1+del_del+del_del1+ins_ins+ins_ins1
    FR = cor_nocor
    FA = sub_nosub+ins_noins+del_nodel
    DE = sub_sub1+del_del1+ins_ins1
    ref_len = total_sub+total_del+total_cor_per

    prec = TR/(TR+FR) if (TR+FR)>0 else 0.
    rec  = TR/(TR+FA) if (TR+FA)>0 else 0.
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.
    per  = (total_sub+total_del+total_ins)/ref_len if ref_len>0 else 0.
    der  = DE/(TR+FA) if (TR+FA)>0 else 0.
    sc   = 0.5*f1 + 0.4*(1-der) + 0.1*(1-per)
    return {'f1':f1,'prec':prec,'rec':rec,'per':per,'der':der,'score':sc}


def evaluate_on_valid(gt_c, gt_t, preds, tag=''):
    m = _score_preds(gt_c, gt_t, preds)
    if tag:
        print(f'{tag:50s}  F1={m["f1"]:.4f}  PER={m["per"]:.4f}  '
              f'DER={m["der"]:.4f}  Score={m["score"]:.4f}')
    return m


# ── Dataset + Collator + Trainer ──────────────────────────────────────────────
class MDDDataset(Dataset):
    def __init__(self, df, ph_df, audio_dir, phone2id, is_train=True):
        self.df       = df.reset_index(drop=True)
        self.ph       = ph_df.reset_index(drop=True)
        self.audio_dir = Path(audio_dir)
        self.phone2id  = phone2id
        self.is_train  = is_train
        self.lengths   = []
        for _, row in self.df.iterrows():
            p = self.audio_dir / Path(row['path']).name
            try:
                with wave.open(str(p), 'rb') as wf:
                    self.lengths.append(wf.getnframes())
            except Exception:
                self.lengths.append(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        ph  = self.ph.iloc[idx]
        y, _ = load_wav_f32(self.audio_dir / Path(row['path']).name)
        y = preproc(y)
        if self.is_train:
            y = augment(y)
        labels = [self.phone2id.get(t, self.phone2id[UNK]) for t in tokenize(ph['t_norm'])]
        return {'input_values': y.astype(np.float32), 'labels': labels}


@dataclass
class MDDCollator:
    fe: object
    sr: int = 16000
    pad_id: int = -100

    def __call__(self, batch):
        out = self.fe(
            [b['input_values'] for b in batch],
            sampling_rate=self.sr, padding=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        labels = [torch.tensor(b['labels'], dtype=torch.long) for b in batch]
        maxL = max(len(l) for l in labels)
        pad  = torch.full((len(labels), maxL), self.pad_id, dtype=torch.long)
        for i, l in enumerate(labels):
            pad[i, :len(l)] = l
        out['labels'] = pad
        return out


class MDDTrainer(Trainer):
    def _get_train_sampler(self, train_dataset=None):
        ds = train_dataset or self.train_dataset
        if not self.args.group_by_length or not hasattr(ds, 'lengths'):
            return RandomSampler(ds)
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=ds.lengths,
            dataset=ds,
        )

    def _get_eval_sampler(self, eval_dataset=None):
        return SequentialSampler(eval_dataset or self.eval_dataset)


# ── Inference helpers ─────────────────────────────────────────────────────────
def _greedy_ctc(logits, id2phone, blank_id):
    all_preds, all_confs = [], []
    for seq in logits:
        probs    = scipy.special.softmax(seq, axis=-1)
        pred_ids = np.argmax(seq, axis=-1)
        out_ph, out_cf, prev = [], [], None
        for t, i in enumerate(pred_ids):
            if i == prev: continue
            prev = int(i)
            if prev == blank_id: continue
            out_ph.append(id2phone[prev] if prev < len(id2phone) else UNK)
            out_cf.append(float(probs[t, prev]))
        all_preds.append(' '.join(out_ph))
        all_confs.append(out_cf)
    return all_preds, all_confs


def run_batched(model, feat_ext, paths, id2phone, blank_id, tag=''):
    all_preds, all_confs = [], []
    n = len(paths)
    model.eval()
    for start in range(0, n, INFER_BATCH):
        batch = []
        for p in paths[start:start+INFER_BATCH]:
            y, _ = load_wav_f32(str(p))
            batch.append(trim_silence(y, 16000))
        inputs = feat_ext(batch, sampling_rate=16000, return_tensors='pt', padding=True)
        iv   = inputs.input_values.to(DEVICE)
        attn = inputs.get('attention_mask')
        if attn is not None:
            attn = attn.to(DEVICE)
        with torch.no_grad():
            logits = model(iv, attention_mask=attn).logits.cpu().numpy()
        p, c = _greedy_ctc(logits, id2phone, blank_id)
        all_preds.extend(p)
        all_confs.extend(c)
        print(f'  {tag}: {min(start+INFER_BATCH, n)}/{n}', end='\r', flush=True)
    print()
    return all_preds, all_confs


# ── Per-epoch validation callback ─────────────────────────────────────────────
class EpochMetricsCallback(TrainerCallback):
    def __init__(self, val_paths, gt_c, gt_t, feat_ext, id2phone, blank_id, output_dir):
        self._val_paths  = val_paths
        self._gt_c       = gt_c
        self._gt_t       = gt_t
        self._fe         = feat_ext
        self._id2phone   = id2phone
        self._blank_id   = blank_id
        self._output_dir = output_dir
        self._best_score = -1.0
        self._best_epoch = 0
        self.history     = []

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        preds, _ = run_batched(model, self._fe, self._val_paths,
                               self._id2phone, self._blank_id, tag='Valid')
        m = evaluate_on_valid(self._gt_c, self._gt_t, preds,
                              tag=f'Epoch {int(state.epoch):2d}')
        m['epoch'] = int(state.epoch)
        self.history.append(m)

        if m['score'] > self._best_score:
            self._best_score = m['score']
            self._best_epoch = int(state.epoch)
            model.save_pretrained(str(self._output_dir))
            self._fe.save_pretrained(str(self._output_dir))
            print(f'  -> Best  epoch={self._best_epoch}  score={self._best_score:.4f}')

        model.train()


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Train MDD model')
    parser.add_argument('--epochs',      type=int,   default=EPOCHS)
    parser.add_argument('--batch-size',  type=int,   default=BATCH_SIZE)
    parser.add_argument('--grad-accum',  type=int,   default=GRAD_ACCUM)
    parser.add_argument('--encoder-lr',  type=float, default=ENCODER_LR)
    parser.add_argument('--head-lr',     type=float, default=HEAD_LR)
    parser.add_argument('--output-dir',  type=str,   default='checkpoint')
    parser.add_argument('--splits-dir',  type=str,   default='splits')
    parser.add_argument('--audio-dir',   type=str,   default='MDD-Challenge-2025-training-set/audio_data/train')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    splits_dir = Path(args.splits_dir)
    audio_dir  = Path(args.audio_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'Device: {DEVICE}')
    print(f'Epochs: {args.epochs} | Batch: {args.batch_size} | GradAccum: {args.grad_accum}')
    print(f'Encoder LR: {args.encoder_lr} | Head LR: {args.head_lr}')
    print(f'Output: {output_dir}')

    # ── Load vocab ──────────────────────────────────────────────────────────
    vocab_path = splits_dir / 'phone_vocab.json'
    if not vocab_path.exists():
        raise FileNotFoundError(
            f'{vocab_path} not found. Run: python prepare_data.py'
        )
    vocab    = json.load(open(vocab_path, encoding='utf-8'))
    id2phone = vocab['id2phone']
    phone2id = vocab['phone2id']
    blank_id = phone2id[BLANK]
    print(f'Vocab size: {len(id2phone)}')

    # ── Load data ───────────────────────────────────────────────────────────
    # train_phones.csv / valid_phones.csv include path, c_norm, t_norm columns
    train_ph = pd.read_csv(splits_dir / 'train_phones.csv')
    valid_ph = pd.read_csv(splits_dir / 'valid_phones.csv')
    train_df = train_ph   # path column present in phones csv
    valid_df = valid_ph

    print(f'Train: {len(train_df)}  Valid: {len(valid_df)}')

    val_paths = [audio_dir / Path(p).name for p in valid_ph['path']]
    VALID_C   = valid_ph['c_norm'].tolist()
    VALID_T   = valid_ph['t_norm'].tolist()

    # ── Build model ─────────────────────────────────────────────────────────
    fe = AutoFeatureExtractor.from_pretrained(BASE_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(
        BASE_MODEL,
        vocab_size=len(id2phone),
        pad_token_id=blank_id,
        ctc_loss_reduction='mean',
        ctc_zero_infinity=True,
        ignore_mismatched_sizes=True,
    )
    model.config.apply_spec_augment = ENABLE_SPEC_AUGMENT
    model.config.mask_time_prob     = 0.05 if ENABLE_SPEC_AUGMENT else 0.0
    model.config.mask_feature_prob  = 0.0
    model.freeze_feature_encoder()
    model.to(DEVICE)
    print(f'Model loaded: {BASE_MODEL}')

    # ── Optimizer (differential LR: encoder vs CTC head) ───────────────────
    head_ids   = {id(p) for p in model.lm_head.parameters()}
    enc_params = [p for p in model.parameters()
                  if id(p) not in head_ids and p.requires_grad]
    hd_params  = [p for p in model.lm_head.parameters() if p.requires_grad]
    total_steps = math.ceil(len(train_df) / (args.batch_size * args.grad_accum)) * args.epochs
    warmup      = int(total_steps * WARMUP_RATIO)
    opt = torch.optim.AdamW(
        [{'params': enc_params, 'lr': args.encoder_lr},
         {'params': hd_params,  'lr': args.head_lr}],
        weight_decay=WEIGHT_DECAY,
    )
    sch = get_linear_schedule_with_warmup(opt, warmup, total_steps)
    print(f'Steps={total_steps}  Warmup={warmup}')

    # ── Datasets ────────────────────────────────────────────────────────────
    train_ds = MDDDataset(train_df, train_ph, audio_dir, phone2id, is_train=True)
    valid_ds = MDDDataset(valid_df, valid_ph, audio_dir, phone2id, is_train=False)
    collator = MDDCollator(fe=fe)
    cb = EpochMetricsCallback(val_paths, VALID_C, VALID_T, fe, id2phone, blank_id, output_dir)

    # ── Training ────────────────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir                  = str(output_dir / '_checkpoints'),
        num_train_epochs            = args.epochs,
        per_device_train_batch_size = args.batch_size,
        gradient_accumulation_steps = args.grad_accum,
        fp16                        = torch.cuda.is_available(),
        eval_strategy               = 'no',
        save_strategy               = 'no',
        group_by_length             = True,
        remove_unused_columns       = False,
        report_to                   = 'none',
        logging_steps               = 30,
        seed                        = SEED,
        label_names                 = ['labels'],
    )
    trainer = MDDTrainer(
        model         = model,
        args          = training_args,
        train_dataset = train_ds,
        eval_dataset  = valid_ds,
        data_collator = collator,
        callbacks     = [cb],
        optimizers    = (opt, sch),
    )

    print(f'\nTraining {args.epochs} epochs | Base: {BASE_MODEL}')
    trainer.train()

    # ── Save history ────────────────────────────────────────────────────────
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(cb.history, f, indent=2)

    print(f'\nBest: epoch={cb._best_epoch}  score={cb._best_score:.4f}')
    print(f'Checkpoint: {output_dir}')
    print(f'History:    {history_path}')

    # Print summary table
    hist_df = pd.DataFrame(cb.history)[['epoch', 'f1', 'per', 'der', 'score']]
    print('\n' + hist_df.to_string(index=False, float_format=lambda x: f'{x:.4f}'))


if __name__ == '__main__':
    main()
