"""[6] 엄격 검증 (E12, 공식 성능) — 2026-10-04 독립 재검증 코드(validate_core.py)를 제출 패키지 경로에 맞게 이식.
  · 시간 순서(forward): 각 연속 가동 구간 시작 전의 라벨·무라벨만으로 레시피 지문·무라벨 선별·문·척도·보정층을 다시 계산 (전처리 미래 누설 제거)
  · 사건 블록 제외(leave-block-out): 시험 사건을 공유하는 구간 전체(양품·불량·LH/RH)와 앞뒤 5분을 학습에서 제거
  · 같은 분할·부품·예산(CN7 5%, RG3 50%)의 5개 모델 비교, 전체/보정 준비 후/점수 가능 운영표, 사건 블록 bootstrap 2,000회
  · FLR 제외 비교, PINN 처방 운전점의 학습 영역 점검, C 관측식 해석 미분·rank, A 차분 간격 점검
실행 조건은 docs/protocol_frozen.md (결과를 보기 전에 고정). 모델 코드와 가중치는 바꾸지 않는다(SHA256 전후 비교).
"""
from pathlib import Path
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest

from .config import CFG, PROC, BUDGET, HALF_THICK, ALPHA, T_GLASS, T_NF, T_MELT_REF, override
from .data import load_labeled, load_unlabeled, read_labeled_raw, shot_table, pairs_for, cause_class, features
from .model import reference, select_tier1, recipe_gate, deviation, fit_calibration, predict, RiskModel
from .stats import seg_u, confusion
from .pinn import WEIGHTS
from . import request

ORIGINAL = Path(__file__).resolve().parent.parent   # PINNSET_code (코드·가중치 해시 기록 범위)

SEED = 20261004
MODELS = ["constant", "PINNSET", "logistic", "HistGB", "IsolationForest", "PINNSET_no_FLR"]
VARS = list(CFG["VARS"])
KEY = ["TimeStamp", "PART_NAME"]


def dump(out, name, value):
    def convert(x):
        if isinstance(x, np.generic):
            return x.item()
        if isinstance(x, (pd.Timestamp, Path)):
            return str(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        raise TypeError(type(x).__name__)
    (out / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=convert), encoding="utf-8")


def log(s):
    print(s, flush=True)


def budget_flags(score, seg, tie, budget):
    flag = np.zeros(len(score), bool)
    for s in np.unique(seg):
        ii = np.flatnonzero(np.asarray(seg) == s)
        n = int(np.floor(len(ii) * budget + .5))
        order = np.lexsort((np.asarray(tie)[ii], -np.asarray(score)[ii]))
        flag[ii[order[:n]]] = True
    return flag


def event_blocks(parts, S):
    """Union segments touched by the same 30-minute defect event."""
    parent = {int(s): int(s) for s in S.seg.unique()}
    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s
    bad = parts[parts.y == 1].merge(S[["TimeStamp", "seg"]], on="TimeStamp", validate="many_to_one").sort_values("TimeStamp")
    bad["event"] = (bad.TimeStamp.diff() > pd.Timedelta(minutes=30)).cumsum()
    events = []
    for e, q in bad.groupby("event"):
        segs = sorted(map(int, q.seg.unique()))
        for s in segs[1:]:
            parent[find(s)] = find(segs[0])
        events.append(dict(event=int(e), start=str(q.TimeStamp.min()), end=str(q.TimeStamp.max()),
                           defects=len(q), segments=segs))
    roots = {s: find(s) for s in parent}
    ids = {r: i for i, r in enumerate(sorted(set(roots.values())))}
    return {s: ids[r] for s, r in roots.items()}, events


def deviations(S, model, no_flr=False):
    with override(VARS=[v for v in VARS if v != "FLR"] if no_flr else VARS):
        return deviation(S, model.scale, recipe_gate(S, model.gmed, model.gsc))


def fit_reference(prod, trainS, U):
    if trainS.empty or not (trainS.bad == 0).any() or U.empty:
        return None, dict(reason="no historical normal shot or unlabeled row")
    med, sc = reference(trainS, CFG["TIER_VARS"])
    T1, lots = select_tier1(U, prod, med, sc)
    if T1.empty:
        return None, dict(reason="no eligible historical tier1 lot", candidate_lots=len(lots))
    ST = shot_table(T1)
    m = RiskModel(prod)
    m.set_references(trainS, ST)
    return m, dict(tier_rows=len(T1), tier_shots=len(ST), tier_max_time=str(T1.TimeStamp.max()),
                   tier_median=med.to_dict(), tier_scale=sc.to_dict(), gate_median=m.gmed.to_dict(),
                   gate_scale=m.gsc.to_dict(), deviation_scale=m.scale)


def data_audit(L, U, data):
    raw = read_labeled_raw(data)
    keys = ["TimeStamp", "PART_FACT_SERIAL", "PART_NAME"]
    duplicates = raw[raw.duplicated(keys, keep=False)]
    conflict = duplicates.groupby(keys)[["PassOrFail", "Reason"] + PROC].nunique(dropna=False).gt(1).any(axis=1)
    R = dict(raw_rows=len(raw), deduplicated_rows=len(L), removed_duplicates=len(raw) - len(L),
             duplicate_key_conflicts=int(conflict.sum()), non_target_rows=int((~L.PART_NAME.str.startswith(("CN7", "RG3"))).sum()), products={})
    for prod in CFG["PRODUCTS"]:
        p = L[L.PART_NAME.str.startswith(prod)]
        S = shot_table(p)
        blocks, events = event_blocks(p, S)
        pairs = pairs_for(p)
        variation = p.groupby("TimeStamp")[list(dict.fromkeys(VARS + CFG["GATE_VARS"] + CFG["TIER_VARS"]))].nunique(dropna=False)
        joined = p.merge(S[["TimeStamp", "seg"]], on="TimeStamp", validate="many_to_one")
        R["products"][prod] = dict(parts=len(p), shots=len(S), defects=int(p.y.sum()), paired_shots=len(pairs),
                                  unpaired_shots=len(S) - len(pairs), key_duplicates=int(p.duplicated(KEY).sum()),
                                  shared_shot_feature_conflicts=int(variation.gt(1).any(axis=1).sum()),
                                  joined_rows=len(joined), initial_allowance_defects=int((p.Reason == "초기허용불량").sum()),
                                  segments=S.seg.nunique(), independent_blocks=len(set(blocks.values())), defect_events=events,
                                  pass_fail_values=p.PassOrFail.value_counts().to_dict(),
                                  labeled_unlabeled_time_overlap_shots=int(S.TimeStamp.isin(U[(U.EQUIP_NAME == CFG["EQUIP"]) & U.PART_NAME.str.startswith(prod)].TimeStamp).sum()))
        assert not p.duplicated(KEY).any()
        assert len(joined) == len(p)
    assert not conflict.any()
    return R


def evaluate_product(prod, parts, U, mode, out, fit_ref=None, devs=None, log=log):
    """fit_ref·devs: 기준 설정과 이탈 계산을 바꿔 끼울 때(newrecipe.py). 없으면 공식 fit_reference·deviations"""
    fit_ref = fit_ref or fit_reference; devs = devs or deviations
    S = shot_table(parts)
    blockmap, events = event_blocks(parts, S)
    parts = parts.merge(S[["TimeStamp", "seg"]], on="TimeStamp", validate="many_to_one").sort_values(KEY).reset_index(drop=True)
    parts["block"] = parts.seg.map(blockmap)
    parts["tie"] = np.random.default_rng(SEED).random(len(parts))
    U = U[(U.EQUIP_NAME == CFG["EQUIP"]) & U.PART_NAME.str.startswith(prod)].copy()
    outputs, folds, parity = [], [], None
    groups = sorted(S.seg.unique()) if mode == "forward" else sorted(parts.block.unique())
    for group in groups:
        te = parts.seg == group if mode == "forward" else parts.block == group
        testsegs = parts.seg[te].unique()
        testS = S[S.seg.isin(testsegs)]
        start, end = testS.TimeStamp.min(), testS.TimeStamp.max()
        if mode == "forward":
            trS = S[S.TimeStamp < start]
            tr = parts.TimeStamp < start
            pool = U[U.TimeStamp < start]
        else:
            excludeL = np.zeros(len(parts), bool)
            excludeU = np.zeros(len(U), bool)
            for s in testsegs:
                qs = S[S.seg == s]
                a, z = qs.TimeStamp.min() - pd.Timedelta(minutes=5), qs.TimeStamp.max() + pd.Timedelta(minutes=5)
                excludeL |= parts.TimeStamp.between(a, z).values
                excludeU |= U.TimeStamp.between(a, z).values
            tr = ~excludeL
            trS = S[S.TimeStamp.isin(parts.TimeStamp[tr])]
            pool = U[~excludeU]
        train = parts[tr].copy()
        test = parts[te].copy()
        fold = dict(product=prod, mode=mode, fold=int(group), test_start=str(start), test_end=str(end),
                    train_parts=len(train), test_parts=len(test), train_max_time=str(train.TimeStamp.max()),
                    pool_rows=len(pool), pool_max_time=str(pool.TimeStamp.max()), test_segments=list(map(int, testsegs)),
                    labeled_time_overlap=len(set(train.TimeStamp) & set(test.TimeStamp)),
                    unlabeled_test_time_overlap=int(pool.TimeStamp.isin(test.TimeStamp).sum()))
        assert fold["labeled_time_overlap"] == fold["unlabeled_test_time_overlap"] == 0
        if mode == "forward" and len(train):
            assert train.TimeStamp.max() < start
        if mode == "forward" and len(pool):
            assert pool.TimeStamp.max() < start
        m, refs = fit_ref(prod, trS, pool)
        fold.update(refs)
        test["status"] = "unavailable"
        test["D"] = np.nan
        test["D_no_FLR"] = np.nan
        for name in MODELS:
            test[name] = np.nan
        if m is None:
            fold["calibration"] = "unavailable references"
            outputs.append(test)
            folds.append(fold)
            log(f"{prod} {mode} fold {group}: no references")
            continue
        trainD = devs(trS, m)
        trainDN = devs(trS, m, True)
        train = train.merge(trS[["TimeStamp"]].assign(D=trainD, D_no_FLR=trainDN), on="TimeStamp", validate="many_to_one")
        train = train[np.isfinite(train.D)].copy()
        train["gD"] = np.log1p(train.D)
        prefixS = S[S.TimeStamp <= end]
        testD = devs(prefixS, m)
        testDN = devs(prefixS, m, True)
        dmap = prefixS[["TimeStamp"]].assign(D=testD, D_no_FLR=testDN).set_index("TimeStamp")
        test["D"] = test.TimeStamp.map(dmap.D)
        test["D_no_FLR"] = test.TimeStamp.map(dmap.D_no_FLR)
        ready = len(train) > 0 and 0 < train.y.sum() < len(train)
        fold["calibration_train_parts"] = len(train)
        fold["calibration_train_defects"] = int(train.y.sum())
        if not ready:
            fold["calibration"] = "unavailable defects/classes"
            outputs.append(test)
            folds.append(fold)
            log(f"{prod} {mode} fold {group}: no calibration classes")
            continue
        pair = pairs_for(train)
        b, Lc, spec = fit_calibration(train, pair, dict(CFG["MODEL"][prod]))
        trainN = train.copy()
        trainN["gD"] = np.log1p(trainN.D_no_FLR)
        bn, ln, sn = fit_calibration(trainN, pair, dict(CFG["MODEL"][prod]))
        test["status"] = np.where(np.isfinite(test.D), "score", "held")
        valid = test.status == "score"
        T = test[valid].copy()
        T["gD"] = np.log1p(T.D)
        test.loc[valid, "PINNSET"] = predict(T, b, Lc, spec)
        T["gD"] = np.log1p(T.D_no_FLR)
        test.loc[valid, "PINNSET_no_FLR"] = predict(T, bn, ln, sn)
        test.loc[valid, "constant"] = train.y.mean()
        Xtr, Xte = features(train), features(test[valid])
        makers = {
            "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1., max_iter=3000)),
            "HistGB": HistGradientBoostingClassifier(max_iter=200, learning_rate=.05, max_depth=3, min_samples_leaf=20, random_state=0),
            "IsolationForest": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), IsolationForest(n_estimators=200, random_state=0)),
        }
        if valid.any():
            for name, mdl in makers.items():
                if name == "IsolationForest":
                    mdl.fit(Xtr.drop(columns="RH"))
                    pred = -mdl.score_samples(Xte.drop(columns="RH"))
                else:
                    mdl.fit(Xtr, train.y)
                    pred = mdl.predict_proba(Xte)[:, 1]
                test.loc[valid, name] = pred
        fold.update(calibration="ready", coef=b.tolist(), Lc=float(Lc), spec=spec, test_scored=int(valid.sum()),
                    test_scored_defects=int(test.y[valid].sum()),
                    max_D_no_FLR_difference=float(np.nanmax(np.abs(test.D - test.D_no_FLR))) if valid.any() else None)
        # Representative parity with original causal stream: same refs, same calibration.
        if mode == "forward" and parity is None and valid.any():
            m.b, m.Lc, m.used = b, Lc, spec
            stream = m.predict_stream(prefixS, parts[parts.TimeStamp <= end])
            original = stream[stream.TimeStamp.isin(test.TimeStamp)][KEY + ["이탈점수", "불량확률"]]
            check = test.merge(original, on=KEY, validate="one_to_one")
            k = check.status == "score"
            parity = dict(fold=int(group), parts=len(check),
                          D_max_diff=float(np.max(np.abs(check.D[k] - check.이탈점수[k]))),
                          probability_max_diff=float(np.max(np.abs(check.PINNSET[k] - check.불량확률[k]))))
            assert parity["D_max_diff"] < 1e-10 and parity["probability_max_diff"] < 1e-10
            # Disturb future values under fixed current-fold references; prefix must remain identical.
            altered = S.copy()
            fut = altered.TimeStamp > end
            for c in set(VARS + CFG["GATE_VARS"]):
                altered.loc[fut, c] = altered.loc[fut, c].values[::-1]
            da = devs(altered, m)[:len(prefixS)]
            assert np.allclose(da, testD, atol=1e-10, rtol=0, equal_nan=True)
            parity["future_prefix_invariance"] = True
        outputs.append(test)
        folds.append(fold)
        log(f"{prod} {mode} fold {group}: train {len(train)}, test {len(test)}, scored {valid.sum()}")
    E = pd.concat(outputs).sort_values(KEY).reset_index(drop=True)
    assert len(E) == len(parts) and not E.duplicated(KEY).any()
    E.to_csv(out / f"predictions_{mode}_{prod}.csv", index=False, encoding="utf-8-sig")
    return E, folds, parity


def performance(E, prod, mode):
    valid = E[E.status == "score"].copy()
    y, seg, tie = valid.y.values, valid.seg.values, valid.tie.values
    rows, flags, ranks = [], {}, {}
    base = np.mean((y - valid.constant.values) ** 2) if len(y) else np.nan
    budget = BUDGET[prod]
    for name in MODELS:
        score = valid[name].values
        ranks[name] = seg_u(score, seg)
        flags[name] = budget_flags(score, seg, tie, budget)
        r = confusion(y, flags[name])
        r.update(product=prod, mode=mode, model=name, budget=budget,
                 AUC_within_segment=float(roc_auc_score(y, ranks[name])) if len(np.unique(y)) == 2 else None,
                 AUC_global=float(roc_auc_score(y, score)) if len(np.unique(y)) == 2 else None)
        if name != "IsolationForest":
            bs = float(np.mean((y - score) ** 2))
            r.update(Brier=bs, Brier_improvement=1 - bs / base if base > 0 else None)
        rows.append(r)
    ops = []
    for label, p in [("all_parts", E), ("after_calibration_available", E[E.status != "unavailable"]),
                     ("scored_only", valid), ("all_parts_without_initial_allowance", E[E.Reason != "초기허용불량"])]:
        rule = p.D.ge(CFG["ALARM"]) if prod == "CN7" else p.RH.eq(1)
        flag = p.status.ne("score") | rule
        r = confusion(p.y, flag)
        r.update(product=prod, mode=mode, population=label,
                 fallback_inspections=int(p.status.ne("score").sum()),
                 fallback_defects=int(p.y[p.status.ne("score")].sum()),
                 score_inspections=int((p.status.eq("score") & rule).sum()))
        ops.append(r)
    eventrows = []
    for block, q in E.groupby("block"):
        validq = q[q.status == "score"]
        rule = q.D.ge(CFG["ALARM"]) if prod == "CN7" else q.RH.eq(1)
        r = confusion(q.y, q.status.ne("score") | rule)
        r.update(product=prod, mode=mode, block=int(block), start=str(q.TimeStamp.min()), end=str(q.TimeStamp.max()),
                 scored_defects=int(validq.y.sum()), fallback_defects=int(q.y[q.status != "score"].sum()))
        eventrows.append(r)
    # Fixed-prediction paired bootstrap; independent blocks, not individual parts.
    rng = np.random.default_rng(SEED)
    groups = sorted(valid.block.unique())
    idx = {g: np.flatnonzero(valid.block.values == g) for g in groups}
    intervals = []
    for rival in ["constant", "logistic", "HistGB", "IsolationForest", "PINNSET_no_FLR"]:
        vals, diffs, captures, brier_improvements, brier_differences = [], [], [], [], []
        for _ in range(2000):
            ii = np.concatenate([idx[g] for g in rng.choice(groups, len(groups), replace=True)]) if groups else np.array([], int)
            yy = y[ii]
            if len(ii):
                bp = np.mean((yy-valid.PINNSET.values[ii])**2)
                bc = np.mean((yy-valid.constant.values[ii])**2)
                if bc > 0:
                    brier_improvements.append(float(1-bp/bc))
                if rival != "IsolationForest":
                    brier_differences.append(float(np.mean((yy-valid[rival].values[ii])**2)-bp))
            if len(np.unique(yy)) < 2:
                continue
            a = roc_auc_score(yy, ranks["PINNSET"][ii])
            vals.append(a)
            diffs.append(a - roc_auc_score(yy, ranks[rival][ii]))
            captures.append(float(np.sum(yy * (flags["PINNSET"][ii].astype(int) - flags[rival][ii].astype(int))) / yy.sum()))
        ci = lambda a: np.percentile(a, [2.5, 97.5]).tolist() if a else None
        intervals.append(dict(product=prod, mode=mode, rival=rival, valid_iterations=len(vals), requested_iterations=2000,
                              independent_blocks=len(groups), defect_blocks=int(valid.groupby("block").y.sum().gt(0).sum()),
                              PINNSET_AUC_CI95=ci(vals), delta_AUC_CI95=ci(diffs), delta_recall_CI95=ci(captures),
                              PINNSET_Brier_improvement_CI95=ci(brier_improvements), delta_Brier_CI95=ci(brier_differences),
                              Brier_valid_iterations=len(brier_improvements)))
    return rows, ops, eventrows, intervals


def old_event_overlap(L, U):
    R = {}
    for prod in CFG["PRODUCTS"]:
        parts = L[L.PART_NAME.str.startswith(prod)]
        S = shot_table(parts)
        m, _ = fit_reference(prod, S, U)
        D = deviations(S, m)
        P = parts.merge(S[["TimeStamp", "seg"]].assign(D=D), on="TimeStamp")
        P = P[np.isfinite(P.D)].copy()
        P["cause"] = [cause_class(prod, r) for r in P.itertuples()]
        rows = []
        if prod == "CN7":
            for cause in ["A 재가동", "B 고속", "C 캐비티", "설명 불가"]:
                segs = P.seg[P.cause == cause].unique()
                te = P.seg.isin(segs) & P.cause.isin([cause, "양품"])
                tr = P.cause != cause
                rows.append(dict(cause=cause, train_test_shared_parts=int((te & tr).sum()), test_parts=int(te.sum())))
        else:
            d = P[P.y == 1].sort_values("TimeStamp")
            ev = (d.TimeStamp.diff() > pd.Timedelta(minutes=30)).cumsum()
            for e, q in d.groupby(ev.values):
                tr = ~P.TimeStamp.between(q.TimeStamp.min() - pd.Timedelta(minutes=5), q.TimeStamp.max() + pd.Timedelta(minutes=5))
                te = P.seg.isin(q.seg.unique()) & ((P.y == 0) | P.index.isin(q.index))
                rows.append(dict(event=int(e), train_test_shared_parts=int((te & tr).sum()), test_parts=int(te.sum())))
        R[prod] = rows
    return R


def restart_comparison(L, E):
    p = L[L.PART_NAME.str.startswith("CN7")]
    S = shot_table(p)
    S["since"] = S.groupby("seg").cumcount() + 1
    reason = p.groupby("TimeStamp").Reason.apply(lambda x: "|".join(sorted(set(x.dropna()))))
    S["A"] = S.TimeStamp.map(reason).str.contains("초기허용불량")
    sh = E.groupby("TimeStamp").agg(D=("D", "first"), status=("status", "first"))
    S = S.merge(sh, on="TimeStamp", validate="one_to_one")
    q = S[S.seg != S.seg.min()].copy()
    rows = []
    rules = {f"fixed_{n}_shots": q.since.le(n) for n in [3, 5, 10]}
    rules["D_ge5_first10"] = q.since.le(10) & q.D.ge(5) & q.status.eq("score")
    rules["D_ge5_or_fallback_first10"] = q.since.le(10) & (q.D.ge(5) | q.status.ne("score"))
    for name, flag in rules.items():
        rows.append(dict(rule=name, restart_segments=q.seg.nunique(), total_shots=len(q), separated_shots=int(flag.sum()),
                         A_defect_shots=int(q.A.sum()), A_captured=int((flag & q.A).sum()),
                         all_defect_shots_captured=int((flag & q.bad.eq(1)).sum()),
                         good_shots_separated=int((flag & q.bad.eq(0)).sum())))
    return rows


def independent_jacobian_analysis(J, th, sig):
    J = J / np.asarray(sig)[:, None]
    _, sv, vt = np.linalg.svd(J, full_matrices=False)
    sr = np.linalg.svd(J * np.abs(th), compute_uv=False)
    rank = int((sr > sr[0] * 1e-9).sum()) if sr[0] > 0 else 0
    if rank < len(th):
        return dict(rank=rank, relative_singular_values=sr.tolist(), identified=False)
    cov = (vt.T / sv**2) @ vt
    sd = np.sqrt(np.diag(cov))
    return dict(rank=rank, relative_singular_values=sr.tolist(), identified=True, rho=float(cov[0, 1] / np.prod(sd)),
                rel_se=(sd / abs(th)).tolist(), cov=cov.tolist())


def physics_request_audit(results):
    action = json.loads((results / "action_results.json").read_text(encoding="utf-8"))
    inverse = []
    for row in action["역설계"]:
        tf, tw = row["충진(초)"], row["금형(℃)"]
        for name, w, t, melt in [("required_fill", tw, row["필요 충진(초)"], T_MELT_REF),
                                 ("required_mold", row["필요 금형(℃)"], tf, T_MELT_REF),
                                 ("required_melt", tw, tf, row["필요 용융(℃)"])]:
            tg = (T_GLASS - w) / (melt - w)
            tau = t / (CFG["HALF_THICK"]**2 / CFG["ALPHA"])
            inverse.append(dict(shot=row["시각"], lever=name, mold=w, fill=t, melt=melt, theta_g=tg,
                                inside_training_domain=bool(.099 <= tg <= .401 and .002*.999 <= tau <= .65*1.001),
                                feasible_setting_verified=False))
    C = []
    th, cases = request.C_cases(30, 1.2, .005, .01)
    dp = 142 + np.linspace(-.3, .3, 30)
    jt = np.c_[-dp / th[0]**2, -dp / th[1]**2]
    for name, (f, sig, _, _) in cases.items():
        if name.startswith("현재"):
            J = jt
        elif name.startswith("같은"):
            J = np.tile(jt, (10, 1))
        elif "중량" in name:
            J = np.r_[jt, np.tile([-th[1]/th[0]**2, 1/th[0]], (30, 1))]
        else:
            J = np.r_[jt, np.c_[-dp / th[0]**2, np.zeros(30)], np.c_[np.zeros(30), -dp/th[1]**2]]
        numerical = request.jacobian(f, th)
        own = independent_jacobian_analysis(J, th, sig)
        original = request.analyze(f, th, sig)
        C.append(dict(candidate=name, analytic_numeric_J_max_diff=float(np.max(np.abs(J-numerical))),
                      independent=own, original_rank=original["rank"],
                      sensor_mapping="flow-like dP/R, not direct pressure" if "압력" in name else "conditional observation model"))
        assert own["rank"] == original["rank"]
    A = []
    thA = np.array([19., 2.])
    flr0 = request.FLR0_from_pinn()
    for name, (f, sig, _, _) in request.A2_cases(flr0, 1.).items():
        for h in [.025, .05, .1]:
            J = np.zeros((len(f(thA)), 2))
            for j in range(2):
                a, b = thA.copy(), thA.copy()
                a[j] += h
                b[j] -= h
                J[:, j] = (f(a)-f(b))/(2*h)
            own = independent_jacobian_analysis(J, thA, sig)
            own.update(candidate=name, h_K=h,
                       decision=bool(own["identified"] and abs(own.get("rho", 1)) < .9 and own.get("rel_se", [np.inf])[0] < .25))
            A.append(own)
    original = json.loads((results / "risk_v0_results.json").read_text(encoding="utf-8"))["RG3"]["calibration"]
    b0, lc = original["coef"][0], original["cavity_log_ratio"]
    pl, pr = 1/(1+np.exp(-b0)), 1/(1+np.exp(-(b0+lc)))
    return dict(C_analytic_rank=C, A_step_sensitivity=A, inverse_domain=inverse,
                RG3_ratio_definition=dict(log_odds_offset=lc, odds_ratio=float(np.exp(lc)), probability_ratio=float(pr/pl), p_LH=float(pl), p_RH=float(pr)),
                reused_PINN_evidence=json.loads((WEIGHTS / "pinn_checks.json").read_text(encoding="utf-8")),
                physical_assumptions=dict(alpha_mm2_per_s=ALPHA, half_thickness_mm=HALF_THICK, no_flow_celsius=int(T_NF),
                                          Tm="barrel zones 1-3 mean proxy", Tw="mold channels 3-4 mean proxy",
                                          material_and_thickness_measured=False),
                pressure_mapping_issue="C_cases pressure candidate outputs dP/R: these are branch flow-like quantities. A pressure sensor requires a location/pressure-flow observation equation before its sensor precision or power can be claimed.",
                grid_rates_are_success_probabilities=False)


def run(path, outdir="."):
    """결과 폴더의 risk_v0_results.json·predictions_forward_*.csv·action_results.json을 읽고 <결과 폴더>/strict/에 쓴다"""
    DATA = Path(path).resolve(); RESULTS = Path(outdir).resolve(); OUT = RESULTS / "strict"
    OUT.mkdir(exist_ok=True)
    started = time.time()
    original_hashes = {str(p.relative_to(ORIGINAL)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in ORIGINAL.rglob("*") if p.is_file() and p.suffix in {".py", ".npz"} and "__pycache__" not in p.parts}
    log("Load supplied data")
    L, U = load_labeled(str(DATA)), load_unlabeled(str(DATA))
    dump(OUT, "data_audit.json", data_audit(L, U, str(DATA)))
    dump(OUT, "original_event_overlap.json", old_event_overlap(L, U))
    allfolds, parity, tables, ops, evs, cis = [], {}, [], [], [], []
    for prod in CFG["PRODUCTS"]:
        parts = L[L.PART_NAME.str.startswith(prod)].copy()
        for mode in ["forward", "leave_block_out"]:
            E, folds, pa = evaluate_product(prod, parts, U, mode, OUT)
            allfolds.extend(folds)
            parity[f"{prod}_{mode}"] = pa
            a, b, c, d = performance(E, prod, mode)
            tables.extend(a); ops.extend(b); evs.extend(c); cis.extend(d)
            dump(OUT, "folds.json", allfolds)
            dump(OUT, "comparisons.json", tables)
            dump(OUT, "operations.json", ops)
            dump(OUT, "uncertainty.json", cis)
            if mode == "forward" and prod == "CN7":
                dump(OUT, "restart_comparison.json", restart_comparison(L, E))
    # Recalculate frozen forward operating metrics independently, including previously omitted cold start.
    frozen = []
    for prod in CFG["PRODUCTS"]:
        E = pd.read_csv(RESULTS / f"predictions_forward_{prod}.csv", parse_dates=["TimeStamp"])
        E = E.merge(L[L.PART_NAME.str.startswith(prod)][KEY + ["y", "RH", "Reason"]], on=KEY, validate="one_to_one")
        rule = E.이탈점수.ge(5) if prod == "CN7" else E.RH.eq(1)
        for name, k in [("all_parts", np.ones(len(E), bool)), ("after_calibration_available", E.판정.ne("평가불가"))]:
            flag = E.판정.ne("점수") | rule
            r = confusion(E.y[k], flag[k])
            r.update(product=prod, mode="original_frozen_forward", population=name,
                     fallback_inspections=int(E.판정[k].ne("점수").sum()), fallback_defects=int(E.y[k & E.판정.ne("점수")].sum()))
            frozen.append(r)
    dump(OUT, "original_forward_operations.json", frozen)
    dump(OUT, "stream_parity.json", parity)
    dump(OUT, "events.json", evs)
    dump(OUT, "physics_request.json", physics_request_audit(RESULTS))
    for p, sha in original_hashes.items():
        assert hashlib.sha256((ORIGINAL / p).read_bytes()).hexdigest() == sha
    pd.DataFrame(tables).to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(ops + frozen).to_csv(OUT / "operating_comparison.csv", index=False, encoding="utf-8-sig")
    dump(OUT, "run_receipt.json", dict(seed=SEED, elapsed_seconds=time.time()-started,
                                  original_code_weights_unchanged=True, original_sha256=original_hashes,
                                  protocol_sha256=hashlib.sha256((ORIGINAL / "docs/protocol_frozen.md").read_bytes()).hexdigest(),
                                  source_data_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in [DATA/"labeled_data.csv", DATA/"unlabeled_data.csv"]},
                                  all_checks_completed=True))
    log(f"Completed in {time.time()-started:.1f} s")
