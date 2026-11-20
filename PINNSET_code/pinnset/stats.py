"""공용 통계·저장 함수 — 구간 내 순위, F1·혼동행렬, 원형 이동 귀무분포, JSON 정리"""
import math
import numpy as np, pandas as pd


def seg_u(score, seg, rng=None):
    """구간 안 centered mid-rank u=(평균순위-0.5)/n. 모든 점수가 같으면 정확히 0.5 (V07 erratum, 2026-09-30 정정).
    rng가 있으면 동점을 무작위로 깨서 '상위 k%' 선택에 쓴다"""
    s = pd.Series(np.asarray(score, float))
    if rng is not None:
        s = s + rng.uniform(0, 1e-9, len(s)) * (s.abs().max() + 1)
    return s.groupby(np.asarray(seg)).transform(lambda q: (q.rank() - 0.5) / len(q)).values


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def confusion_kr(y, flag):
    """고정 기준 분석(analysis.py)용 한글 요약"""
    y = np.asarray(y).astype(bool); flag = np.asarray(flag).astype(bool)
    tp, fp, fn = int((y & flag).sum()), int((~y & flag).sum()), int((y & ~flag).sum())
    return dict(검사=int(flag.sum()), TP=tp, FP=fp, FN=fn, 정밀도=tp / max(tp + fp, 1), 검출률=tp / max(tp + fn, 1), F1=f1(tp, fp, fn))


def confusion(y, flag):
    """엄격 검증(evaluate.py)용 혼동행렬 (분모가 0이면 None)"""
    y, flag = np.asarray(y, bool), np.asarray(flag, bool)
    tp, fp, fn, tn = [int(x.sum()) for x in [y & flag, ~y & flag, y & ~flag, ~y & ~flag]]
    return dict(n=len(y), defects=int(y.sum()), inspected=int(flag.sum()), TP=tp, FP=fp, FN=fn, TN=tn,
                inspection_rate=float(flag.mean()) if len(y) else None,
                recall=tp / (tp + fn) if tp + fn else None,
                precision=tp / (tp + fp) if tp + fp else None,
                F1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.)


def circular_null(x, groups, rng, n, stat):
    """구간(groups: 위치 배열 목록)마다 x를 무작위로 원형 이동한 n번의 stat 값"""
    null = np.empty(n)
    for i in range(n):
        xs = x.copy()
        for ii in groups:
            if len(ii) > 1: xs[ii] = np.roll(x[ii], rng.integers(1, len(ii)))
        null[i] = stat(xs)
    return null


def json_safe(o):
    """NaN·무한대를 null로 바꿔 엄격한 JSON 파서에서도 읽히게 한다"""
    if isinstance(o, dict): return {k: json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o): return None
    if hasattr(o, "item") and getattr(o, "ndim", 1) == 0:
        v = o.item(); return None if isinstance(v, float) and not math.isfinite(v) else v
    return o
