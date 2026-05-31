"""stage6_utils.py — Stage 6 diagnostic utilities.

Functions:
  audit_fp_sources        — decompose cor_nocor FPs by type and phoneme
  fp_summary              — print audit result table
  mismatch_analysis       — Spearman correlation: PER_noerr/loss vs Precision/F1
  greedy_ctc_with_conf    — greedy CTC decode + per-token softmax confidence
  calibrated_predictions  — filter low-confidence substitutions
  pr_sweep                — sweep confidence threshold, compute P-R-F1-Score table
  gop_scores              — log-likelihood ratio per phoneme alignment position
  gop_calibrated_predictions — filter predictions by GOP threshold
"""
from pathlib import Path
import numpy as np
from evaluate import _align_pair


def _align2(s1, s2):
    """Wrapper: align s1 and s2, returning exactly (aligned1, aligned2).

    Defensive wrapper around evaluate._align_pair that always returns exactly
    2 values regardless of bytecode caching or _align_pair signature changes.
    """
    result = _align_pair(s1, s2)
    return result[0], result[1]


# ── Task 2: FP Audit ──────────────────────────────────────────────────────────

def audit_fp_sources(gt_c, gt_t, preds):
    """Decompose cor_nocor false positives by operation type and phoneme.

    Returns dict:
      total_fp        : int
      by_op           : {'S': int, 'D': int, 'I_other': int}
      by_op_pct       : same keys, fractional
      top_phones      : list[(phoneme, count)] top-20 canonical phonemes causing FP
      fp_per_sample   : list[int]
      mean_fp_per_sample: float
    """
    by_op    = {'S': 0, 'D': 0, 'I_other': 0}
    by_phone = {}
    fp_per   = []

    for c, t, p in zip(gt_c, gt_t, preds):
        rs, hs, op_rh = _align_pair(c, t)
        hs2, os2, op_ho = _align_pair(t, p)

        sample_fp = 0
        flag = 0
        for i in range(len(hs)):
            if hs[i] == '<eps>':
                continue
            while flag < len(hs2) and hs2[flag] == '<eps>':
                flag += 1
            if flag < len(hs2) and hs[i] == hs2[flag]:
                if op_rh[i] == 'C' and op_ho[flag] != 'C':
                    op = op_ho[flag] if op_ho[flag] in ('S', 'D') else 'I_other'
                    by_op[op] += 1
                    phone = rs[i] if rs[i] != '<eps>' else '<blank_pos>'
                    by_phone[phone] = by_phone.get(phone, 0) + 1
                    sample_fp += 1
                flag += 1
        fp_per.append(sample_fp)

    total = sum(by_op.values())
    return {
        'total_fp'          : total,
        'by_op'             : by_op,
        'by_op_pct'         : {k: round(v / total, 4) if total else 0
                               for k, v in by_op.items()},
        'top_phones'        : sorted(by_phone.items(), key=lambda x: -x[1])[:20],
        'fp_per_sample'     : fp_per,
        'mean_fp_per_sample': round(float(np.mean(fp_per)), 2) if fp_per else 0,
    }


def fp_summary(audit_result):
    """Print audit_fp_sources() result as a formatted table."""
    r = audit_result
    print(f'Total FPs (cor_nocor): {r["total_fp"]}')
    print('By operation:')
    for op, cnt in r['by_op'].items():
        pct = r['by_op_pct'][op]
        bar = '█' * int(pct * 40)
        print(f'  {op:8s}  {cnt:5d}  ({pct*100:5.1f}%)  {bar}')
    print('\nTop-10 canonical phonemes generating FPs:')
    for ph, cnt in r['top_phones'][:10]:
        print(f'  {ph:12s}  {cnt:5d}')
    fp_arr = np.array(r['fp_per_sample'])
    q = np.percentile(fp_arr, [0, 25, 50, 75, 100])
    print(f'\nFP per sample — mean: {r["mean_fp_per_sample"]}  '
          f'(min/Q1/med/Q3/max: {q[0]:.0f}/{q[1]:.0f}/{q[2]:.0f}/{q[3]:.0f}/{q[4]:.0f})')


# ── Task 3: Loss–Metric Mismatch ─────────────────────────────────────────────

def mismatch_analysis(epoch_json_path, start_epoch=10):
    """Spearman correlation: PER_noerr/train_loss vs Precision/F1.

    start_epoch: skip rapid early-learning phase.
    """
    import json
    from scipy.stats import spearmanr

    history = json.load(open(epoch_json_path))
    h = [x for x in history if x.get('epoch', 0) >= start_epoch]
    if not h:
        print(f'No epochs >= {start_epoch} found in {epoch_json_path}.')
        return

    prec   = [x['prec']             for x in h]
    f1     = [x['f1']               for x in h]
    per    = [x['per']              for x in h]
    per_ne = [x.get('per_noerr', 0) for x in h]
    tloss  = [x.get('train_loss')   for x in h]
    epochs = [x['epoch']            for x in h]

    print(f'Mismatch analysis — epochs {int(epochs[0])}–{int(epochs[-1])} ({len(h)} points)')
    print(f'{"Metric pair":42s}  Spearman r    p-value')
    print('-' * 65)

    pairs = [
        ('PER_noerr vs Precision',  per_ne, prec),
        ('PER_noerr vs F1',         per_ne, f1),
        ('PER (full) vs Precision', per,    prec),
        ('train_loss vs Precision', tloss,  prec),
        ('train_loss vs F1',        tloss,  f1),
    ]
    for label, x, y in pairs:
        xy = [(xi, yi) for xi, yi in zip(x, y)
              if xi is not None and yi is not None]
        if len(xy) < 5:
            print(f'  {label:42s}  N/A (n={len(xy)} < 5)')
            continue
        xs, ys = zip(*xy)
        r, p = spearmanr(xs, ys)
        sig = '**' if p < 0.05 else ('*' if p < 0.10 else 'ns')
        print(f'  {label:42s}  r={r:+.3f}   p={p:.3f}  {sig}')

    n = len(h)
    h1, h2 = h[:n // 2], h[n // 2:]
    delta = (np.mean([x['prec'] for x in h2])
             - np.mean([x['prec'] for x in h1]))
    print(f'\nPrecision: '
          f'mean(ep{int(h1[0]["epoch"])}-{int(h1[-1]["epoch"])})='
          f'{np.mean([x["prec"] for x in h1]):.4f}  '
          f'mean(ep{int(h2[0]["epoch"])}-{int(h2[-1]["epoch"])})='
          f'{np.mean([x["prec"] for x in h2]):.4f}  '
          f'Δ={delta:+.4f}')
    if abs(delta) < 0.005:
        print('DIAGNOSIS: Precision PLATEAU confirmed — MISMATCH between CTC loss and MDD metric.')
    else:
        print(f'DIAGNOSIS: Precision still shifting ({delta:+.4f}) — some training signal remains.')


# ── Task 4: Calibration ───────────────────────────────────────────────────────

def greedy_ctc_with_conf(logits, id2phone, phone2id, blank_penalty=0.0):
    """Greedy CTC decode, returning per-token max softmax confidence.

    Returns:
      preds       : list[str] — same format as greedy_ctc()
      confidences : list[list[float]] — per-token max softmax prob
    """
    import scipy.special
    adj = logits.copy()
    adj[..., phone2id['<blank>']] -= blank_penalty

    all_preds, all_confs = [], []
    for seq_logits in adj:
        probs    = scipy.special.softmax(seq_logits, axis=-1)
        pred_ids = np.argmax(seq_logits, axis=-1)
        out_ph, out_cf, prev = [], [], None
        for t, i in enumerate(pred_ids):
            if i == prev:
                continue
            prev = int(i)
            if prev == phone2id['<blank>']:
                continue
            ph = id2phone[prev] if prev < len(id2phone) else '<unk>'
            out_ph.append(ph)
            out_cf.append(float(probs[t, prev]))
        all_preds.append(' '.join(out_ph))
        all_confs.append(out_cf)
    return all_preds, all_confs


def calibrated_predictions(gt_c, preds, confidences, threshold):
    """Suppress low-confidence substitutions: restore canonical when confidence < threshold.

    Only substitution positions (predicted ≠ canonical) are filtered.
    Deletions and insertions are unchanged.
    """
    filtered = []
    for c, pred, conf in zip(gt_c, preds, confidences):
        if not pred.strip():
            filtered.append(pred)
            continue

        ca, pa = _align2(c, pred)
        conf_map = dict(enumerate(conf))

        new_tokens, p_idx = [], 0
        for ct, pt in zip(ca, pa):
            if pt == '<eps>':
                continue
            cur_conf = conf_map.get(p_idx, 1.0)
            p_idx += 1
            if ct != '<eps>' and pt != ct and cur_conf < threshold:
                new_tokens.append(ct)   # restore canonical
            else:
                new_tokens.append(pt)
        filtered.append(' '.join(new_tokens))
    return filtered


def _compute_pr_from_preds(gt_c, gt_t, cal_preds):
    """Compute precision/recall from predictions via three-way alignment.
    Returns (precision, recall, f1).
    """
    import csv, tempfile, os
    from evaluate import _align_pair as ap

    cor_nocor = sub_sub = sub_sub1 = sub_nosub = 0
    ins_ins = ins_ins1 = ins_noins = del_del = del_del1 = del_nodel = 0

    for c, t, p in zip(gt_c, gt_t, cal_preds):
        rs, hs, op_rh = ap(c, t)
        hs2, os2, op_ho = ap(t, p)
        rs3, os3, op_ro = ap(c, p)

        flag = 0
        for ii in range(len(rs)):
            if rs[ii] == '<eps>':
                continue
            while flag < len(rs3) and rs3[flag] == '<eps>':
                flag += 1
            if flag < len(rs3) and rs[ii] == rs3[flag]:
                if   op_rh[ii] == 'D' and op_ro[flag] == 'D':               del_del  += 1
                elif op_rh[ii] == 'D' and op_ro[flag] not in ('D', 'C'):    del_del1 += 1
                elif op_rh[ii] == 'D' and op_ro[flag] == 'C':               del_nodel += 1
                flag += 1

        flag = 0
        for ii in range(len(hs)):
            if hs[ii] == '<eps>':
                continue
            while flag < len(hs2) and hs2[flag] == '<eps>':
                flag += 1
            if flag < len(hs2) and hs[ii] == hs2[flag]:
                if   op_rh[ii] == 'C' and op_ho[flag] != 'C':               cor_nocor += 1
                if   op_rh[ii] == 'S' and op_ho[flag] == 'C':               sub_sub   += 1
                elif op_rh[ii] == 'S' and op_ho[flag] != 'C' and rs[ii] != os2[flag]: sub_sub1 += 1
                elif op_rh[ii] == 'S' and op_ho[flag] != 'C' and rs[ii] == os2[flag]: sub_nosub += 1
                if   op_rh[ii] == 'I' and op_ho[flag] == 'C':               ins_ins   += 1
                elif op_rh[ii] == 'I' and op_ho[flag] not in ('C', 'D'):    ins_ins1  += 1
                elif op_rh[ii] == 'I' and op_ho[flag] == 'D':               ins_noins += 1
                flag += 1

    TR = sub_sub + sub_sub1 + del_del + del_del1 + ins_ins + ins_ins1
    FR = cor_nocor
    FA = sub_nosub + ins_noins + del_nodel
    prec = TR / (TR + FR) if (TR + FR) > 0 else 0.0
    rec  = TR / (TR + FA) if (TR + FA) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def pr_sweep(gt_c, gt_t, logits, id2phone, phone2id,
             thresholds=None, blank_penalty=0.0):
    """Sweep confidence threshold, compute P/R/F1/Score table.

    threshold=0.0 → no filtering (baseline greedy CTC)
    threshold=1.0 → all substitutions suppressed (predict canonical everywhere)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import evaluate_on_valid

    if thresholds is None:
        thresholds = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0]

    base_preds, confidences = greedy_ctc_with_conf(
        logits, id2phone, phone2id, blank_penalty=blank_penalty
    )

    results = []
    print(f'{"Thresh":7s}  {"P":7s}  {"R":7s}  {"F1":7s}  {"Score":7s}')
    print('-' * 45)
    for thr in thresholds:
        if thr == 0.0:
            cal_preds = base_preds
        elif thr >= 1.0:
            cal_preds = list(gt_c)
        else:
            cal_preds = calibrated_predictions(gt_c, base_preds, confidences, thr)

        prec, rec, f1 = _compute_pr_from_preds(gt_c, gt_t, cal_preds)
        m = evaluate_on_valid(gt_c, gt_t, cal_preds)
        row = {'threshold': thr, 'precision': prec, 'recall': rec,
               'f1': f1, 'score': m['score'], 'der': m['der'], 'per': m['per']}
        results.append(row)
        print(f'{thr:7.2f}  {prec:.4f}  {rec:.4f}  {f1:.4f}  {m["score"]:.4f}')
    return results


# ── Task 5: GOP Scoring ───────────────────────────────────────────────────────

def gop_scores(logits, gt_c, id2phone, phone2id, blank_penalty=0.0):
    """Compute per-substitution GOP = log P(predicted) - log P(canonical).

    Uses mean frame probability over the phoneme's CTC span.

    Returns:
      gop_preds      : list[str] — greedy CTC predictions (same as greedy_ctc)
      gop_scores_list: list[list[float]] — one score per non-eps predicted token
        float('inf') for correct/insertion positions (non-error → don't suppress)
        float for substitution positions (positive → confident error)
    """
    import scipy.special
    adj = logits.copy()
    adj[..., phone2id['<blank>']] -= blank_penalty

    all_preds, all_gops = [], []
    for seq_logits, c_str in zip(adj, gt_c):
        probs    = scipy.special.softmax(seq_logits, axis=-1)  # (T, V)
        pred_ids = np.argmax(seq_logits, axis=-1)              # (T,)

        # Reconstruct predicted tokens with their frame spans
        ph_list, spans, prev_id, start = [], [], None, 0
        for t, i in enumerate(pred_ids):
            if i != prev_id:
                if prev_id is not None and prev_id != phone2id['<blank>']:
                    ph_list.append(id2phone[prev_id]
                                   if prev_id < len(id2phone) else '<unk>')
                    spans.append((start, t))
                start, prev_id = t, int(i)
        if prev_id is not None and prev_id != phone2id['<blank>']:
            ph_list.append(id2phone[prev_id]
                           if prev_id < len(id2phone) else '<unk>')
            spans.append((start, len(pred_ids)))

        pred_str = ' '.join(ph_list)
        all_preds.append(pred_str)

        ca, pa = _align2(c_str, pred_str)
        gops, p_idx = [], 0
        for ct, pt in zip(ca, pa):
            if pt == '<eps>':
                continue
            if p_idx >= len(spans):
                gops.append(0.0)
                p_idx += 1
                continue
            s, e = spans[p_idx]
            p_idx += 1
            frame_probs = probs[s:e]

            if ct == '<eps>' or ct == pt:
                gops.append(float('inf'))   # insertion or correct → keep as-is
            elif len(frame_probs) == 0:
                gops.append(0.0)
            else:
                p_pred_id  = phone2id.get(pt, 1)
                p_canon_id = phone2id.get(ct, 1)
                log_pred  = float(np.log(frame_probs[:, p_pred_id].mean()  + 1e-10))
                log_canon = float(np.log(frame_probs[:, p_canon_id].mean() + 1e-10))
                gops.append(log_pred - log_canon)

        all_gops.append(gops)

    return all_preds, all_gops


def gop_calibrated_predictions(gt_c, preds, gop_scores_list, gop_threshold):
    """Suppress substitutions where GOP ≤ gop_threshold.

    GOP > threshold → confident mispronunciation → keep (flag as error)
    GOP ≤ threshold → uncertain → restore canonical (suppress FP)
    float('inf') positions are never suppressed.
    """
    filtered = []
    for c, pred, gops in zip(gt_c, preds, gop_scores_list):
        if not pred.strip():
            filtered.append(pred)
            continue
        ca, pa = _align2(c, pred)
        new_tokens, gop_idx = [], 0
        for ct, pt in zip(ca, pa):
            if pt == '<eps>':
                continue
            g = gops[gop_idx] if gop_idx < len(gops) else float('inf')
            gop_idx += 1
            if ct != '<eps>' and pt != ct and g != float('inf') and g <= gop_threshold:
                new_tokens.append(ct)   # suppress: restore canonical
            else:
                new_tokens.append(pt)
        filtered.append(' '.join(new_tokens))
    return filtered


def gop_pr_sweep(gt_c, gt_t, logits, id2phone, phone2id,
                 thresholds=None, blank_penalty=0.0):
    """Sweep GOP threshold, compute P/R/F1/Score table."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils import evaluate_on_valid

    if thresholds is None:
        thresholds = [-5.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]

    gop_preds, gop_scores_list = gop_scores(
        logits, gt_c, id2phone, phone2id, blank_penalty=blank_penalty
    )

    results = []
    print(f'{"GOP_thr":8s}  {"P":7s}  {"R":7s}  {"F1":7s}  {"Score":7s}')
    print('-' * 48)
    for thr in thresholds:
        cal_preds = gop_calibrated_predictions(gt_c, gop_preds, gop_scores_list, thr)
        prec, rec, f1 = _compute_pr_from_preds(gt_c, gt_t, cal_preds)
        m = evaluate_on_valid(gt_c, gt_t, cal_preds)
        row = {'gop_threshold': thr, 'precision': prec, 'recall': rec,
               'f1': f1, 'score': m['score']}
        results.append(row)
        print(f'{thr:8.1f}  {prec:.4f}  {rec:.4f}  {f1:.4f}  {m["score"]:.4f}')
    return results
