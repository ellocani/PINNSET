"""[1][2][3][7] 고정 기준 검증과 추가 분석 — 정상 기준(문·척도)을 전체 기간 무라벨로 고정한 평가 (참고용)
※ 공식 성능은 evaluate.py(엄격 검증, E12)다. 여기 결과는 모델 재현·스트림 점검·해석·강건성 참고값이다.

  run_v0      (v0) 동결 모델 재현: 1층 선별·문·오경보율·보정식, 구간 하나 빼기 검증(S8), 유의성(S9)   → risk_v0_results.json
  run_v1      (v1) 일괄=샷별 일치, 미래 누설 검사, 시간 순서(고정 기준), 원인 태그, 무라벨 태그       → risk_v1_validation.json,
              전체 학습 모델의 부품별 출력 predictions_{제품}.csv(현장 투입용), predictions_forward_{제품}.csv(참고)
  run_extra   E1 진단 · E2 모델 비교+F1 · E3 상호작용 · E4 FN/FP · E5 태그·비교범위 민감도 · E6 보류 규칙
              · E7 C 단서 재현 · E8 보류 포함 확률 평가 · E9 위치 변수                                → extra_results.json
  run_verify  E10a 원인별 사건 안 AUC(E12 예측) · E10c 경보 기준별 운영점(E12 예측) · E10d 변수 빼기·음성 대조
              · E10e 민감도 · E11 다른 설비 적용성                                                     → verify_results.json
선행: run_extra는 run_v1의 예측 CSV, run_verify는 evaluate.py의 strict/ 예측 CSV가 결과 폴더에 있어야 한다.
민감도 계산은 config.override로 동결 파라미터를 잠시 바꾸고 끝나면 원래 값으로 되돌린다.
"""
import json
import numpy as np, pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from .config import CFG, VARS4, PROC, BUDGET, override
from .data import load_labeled, load_unlabeled, read_labeled_raw, read_unlabeled, shot_table, pairs_for, cause_class, features
from .model import (reference, select_tier1, recipe_gate, dev_scale, deviation, irls, fit_calibration, predict,
                    prepare_product)
from .stats import seg_u, f1, confusion_kr, circular_null, json_safe

# ============================================================ S8 검증 (v0)
def evaluate_loso(P, pairs, spec):
    y = P.y.values; g = P.seg.values; oof = np.zeros(len(P))
    for s in np.unique(g):
        tr = g != s; te = g == s
        pair_tr = pairs.loc[pairs.index.isin(P.TimeStamp[tr])]
        b, Lc, sp = fit_calibration(P[tr], pair_tr, dict(spec))
        oof[te] = predict(P[te], b, Lc, sp)
    base = brier_score_loss(y, np.full(len(y), y.mean()))
    wr = seg_u(oof, g)
    order = np.argsort(-oof); top = order[: int(len(y) * 0.05)]
    bins = pd.cut(oof, [0, .003, .01, .03, .1, 1], labels=["<0.3%", "0.3~1%", "1~3%", "3~10%", ">=10%"], include_lowest=True)
    cal = pd.DataFrame(dict(p=oof, y=y, b=bins)).groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    return dict(brier=brier_score_loss(y, oof), const=base, improve=1 - brier_score_loss(y, oof) / base,
                auc_within=roc_auc_score(y, wr), top5=int(y[top].sum()), n_def=int(y.sum()), calib=cal.round(4).to_dict())

# ============================================================ S9 유의성 (v0)
def circular_shift_test(P, rng):
    sh = P.drop_duplicates("TimeStamp")[["TimeStamp", "seg", "D"]].reset_index(drop=True)
    idx = P.TimeStamp.map(pd.Series(np.arange(len(sh)), index=sh.TimeStamp)).values
    y = P.y.values; g = P.seg.values
    auc = lambda s: roc_auc_score(y, seg_u(s, g))
    obs = auc(P.D.values)
    groups = [np.array(list(v)) for v in sh.groupby("seg").groups.values()]
    null = circular_null(sh.D.values, groups, rng, CFG["N_SHIFT"], lambda Ds: auc(Ds[idx]))
    return dict(auc=obs, null_95=float(np.percentile(null, 95)), p=float((np.sum(null >= obs) + 1) / (len(null) + 1)))

def paired_test(pairs, rng):
    k = int(((pairs.RH == 1) & (pairs.LH == 0)).sum()); m = int(((pairs.LH == 1) & (pairs.RH == 0)).sum())
    bad = pairs[(pairs.RH + pairs.LH) > 0].copy()
    bad["ev"] = (bad.index.to_series().diff() > pd.Timedelta(minutes=CFG["EVENT_GAP_MIN"])).cumsum().values
    evs = [e for _, e in bad.groupby("ev")]; share = []
    for _ in range(CFG["N_BOOT"]):
        pick = [evs[i] for i in rng.integers(0, len(evs), len(evs))]
        kk = sum(int(((e.RH == 1) & (e.LH == 0)).sum()) for e in pick); mm = sum(int(((e.LH == 1) & (e.RH == 0)).sum()) for e in pick)
        if kk + mm: share.append(kk / (kk + mm))
    return dict(rh_only=k, lh_only=m, both=int(((pairs.RH == 1) & (pairs.LH == 1)).sum()),
                p_mcnemar=float(binomtest(k, k + m, 0.5).pvalue) if k + m else 1.0,
                boot_share_95=[float(np.percentile(share, 2.5)), float(np.percentile(share, 97.5))], n_events=len(evs))

# ============================================================ v0 실행
def run_v0(path):
    """동결 v0 모델 재현. 문 기준은 라벨 양품 샷(v1 이후는 1층 무라벨)으로 잡는다"""
    rng = np.random.default_rng(CFG["SEED"])
    L, U = load_labeled(path), load_unlabeled(path); out = {}
    for prod in CFG["PRODUCTS"]:
        parts = L[L.PART_NAME.str.startswith(prod)]
        S = shot_table(parts)
        gmed, gsc = reference(S, CFG["GATE_VARS"]); tmed, tsc = reference(S, CFG["TIER_VARS"])
        T1, lots = select_tier1(U, prod, tmed, tsc)
        ST = shot_table(T1)
        scale = {c: dev_scale(ST[c]) for c in CFG["VARS"]}
        rec = recipe_gate(S, gmed, gsc)
        S["D"] = deviation(S, scale, rec)
        held = rec | np.isnan(S.D.values)
        ST["D"] = deviation(ST, scale, recipe_gate(ST, gmed, gsc))
        P = parts.merge(S.loc[~held, ["TimeStamp", "D", "seg"]], on="TimeStamp")
        P["gD"] = np.log1p(P.D)
        pairs = pairs_for(parts)
        spec = dict(CFG["MODEL"][prod])
        b, Lc, sp = fit_calibration(P, pairs, spec)
        du = ST.D.dropna().values; dl = S.D[(S.bad == 0) & ~held].dropna().values
        out[prod] = dict(
            tier1=dict(lots_all=int(len(lots)), lots_tier1=int((lots.d <= CFG["TIER_DIST"]).sum()), shots=int(len(ST))),
            gate=dict(held_shots=int(held.sum()), recipe=int(rec.sum()), not_ready=int((held & ~rec).sum()),
                      held_defect_shots=int(S.bad[held].sum())),
            false_alarm=dict(tier1=float(np.mean(du >= CFG["ALARM"])), labeled_normal=float(np.mean(dl >= CFG["ALARM"])),
                             n_tier1=int(len(du)), n_labeled=int(len(dl))),
            calibration=dict(coef=[float(x) for x in b], cavity_log_ratio=float(Lc), spec=sp),
            loso=evaluate_loso(P, pairs, dict(CFG["MODEL"][prod])),
            significance=dict(process=circular_shift_test(P, rng), cavity=paired_test(pairs, rng)),
        )
        o = out[prod]
        print(f"\n===== {prod} =====")
        print(f" 1층 무라벨: lot {o['tier1']['lots_tier1']}/{o['tier1']['lots_all']}, 샷 {o['tier1']['shots']}")
        print(f" 문: 보류 {o['gate']['held_shots']}샷 (레시피 {o['gate']['recipe']}, 기준 준비 {o['gate']['not_ready']}), 보류된 불량 샷 {o['gate']['held_defect_shots']}")
        print(f" 오경보율: 1층 {o['false_alarm']['tier1']*100:.2f}% ({o['false_alarm']['n_tier1']}샷) / 라벨 양품 {o['false_alarm']['labeled_normal']*100:.2f}% ({o['false_alarm']['n_labeled']}샷)")
        print(f" 보정식: 계수 {np.round(b,2).tolist()}, 캐비티 로그위험비 {Lc:.2f} (x{np.exp(Lc):.2f}), 사용 성분 {sp}")
        e = o["loso"]; print(f" 검증: Brier {e['brier']:.5f} vs 상수 {e['const']:.5f} (개선 {e['improve']*100:+.1f}%), 구간 내 AUC {e['auc_within']:.3f}, 상위 5% {e['top5']}/{e['n_def']}")
        s = o["significance"]; print(f" 유의성: 공정 AUC {s['process']['auc']:.3f} p={s['process']['p']:.4f} | 캐비티 RH만 {s['cavity']['rh_only']} LH만 {s['cavity']['lh_only']} p={s['cavity']['p_mcnemar']:.4f}, 부트스트랩 {np.round(s['cavity']['boot_share_95'],2).tolist()}")
    json.dump(out, open("risk_v0_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return out

# ============================================================ v1 검증
def run_v1(path):
    L, U = load_labeled(path), load_unlabeled(path); report = {}
    for prod in CFG["PRODUCTS"]:
        d = prepare_product(prod, L, U)
        parts, S, ST, pairs, m, Dv0, P = d["parts"], d["S"], d["ST"], d["pairs"], d["m"], d["D"], d["P"]
        R = {}
        # 1) 일괄 계산 = 샷별 계산
        m.fit_calibration(P, pairs); out = m.predict_stream(S, parts)
        s1 = out.drop_duplicates("TimeStamp").set_index("TimeStamp").이탈점수
        a = pd.Series(Dv0, index=S.TimeStamp).reindex(s1.index)
        R["1_일괄=샷별"] = dict(최대차이=float(np.nanmax(np.abs(a.values - s1.values))) if s1.notna().any() else 0.0,
                            보류일치=bool(((a.isna()) == (s1.isna())).all()))
        # 2) 미래 누설 검사: 절반 이후 샷의 공정값을 뒤섞어도 앞 절반의 예측이 그대로인가
        cut = S.TimeStamp.iloc[len(S) // 2]
        S2 = S.copy(); fut = S2.TimeStamp > cut
        rng = np.random.default_rng(1)
        for c in VARS4 + CFG["GATE_VARS"]: S2.loc[fut, c] = rng.permutation(S2.loc[fut, c].values)
        out2 = m.predict_stream(S2, parts)
        k = out.TimeStamp <= cut
        R["2_미래누설"] = dict(앞절반_부품=int(k.sum()),
                          확률_최대차이=float(np.nanmax(np.abs(out.불량확률[k].values - out2.불량확률[k].values))),
                          위험점수_최대차이=float(np.nanmax(np.abs(out.위험점수[k].values - out2.위험점수[k].values))))
        # 3) 시간 순서 (고정 기준, 참고): 보정층만 과거 구간으로 학습. 문·척도는 전체 기간 무라벨 기준이라 엄격한 forward가 아님
        #    → 공식 시간 순서 평가는 evaluate.py가 전처리까지 학습 구간 안에서 다시 계산한다 (2026-10-04)
        calib = {}
        for s in sorted(S.seg.unique()):
            tr = P.seg < s
            if P.y[tr].sum() == 0: calib[s] = None; continue
            b, Lc, used = fit_calibration(P[tr], pairs.loc[pairs.index.isin(P.TimeStamp[tr])], dict(m.spec)); calib[s] = (b, Lc, used)
        outF = m.predict_stream(S, parts, calib_by_seg=calib)
        outF.to_csv(f"predictions_forward_{prod}.csv", index=False, encoding="utf-8-sig")
        E = outF.merge(parts[["TimeStamp", "PART_NAME", "y", "Reason", "RH"]], on=["TimeStamp", "PART_NAME"])
        ev = E[E.판정 == "점수"]; y = ev.y.values
        base = brier_score_loss(y, np.full(len(y), y.mean()))
        R["3_시간순서"] = dict(평가종류="고정 기준(문·척도를 전체 기간 무라벨로 고정) — 스트림 점검용 참고값. 공식 시간 순서 성능은 strict/ (evaluate.py)", 평가부품=int(len(ev)), 불량=int(y.sum()), 평가불가_불량=int(E.y[E.판정 == "평가불가"].sum()),
                          보류_불량=int(E.y[E.판정 == "보류"].sum()),
                          위험점수_AUC=float(roc_auc_score(y, ev.위험점수)), Brier개선=float(1 - brier_score_loss(y, ev.불량확률) / base),
                          위험점수95이상_검출=f"{int(y[ev.위험점수.values >= 95].sum())}/{int(y.sum())} (부품 {int((ev.위험점수 >= 95).sum())}개)",
                          구간내_AUC=float(roc_auc_score(y, seg_u(ev.위험점수.values, ev.seg.values))),
                          태그부품_검사시_검출=f"{int(y[(ev.원인태그 != '-').values].sum())}/{int(y.sum())} (부품 {int((ev.원인태그 != '-').sum())}개)")
        # 4) 원인 태그 검증 (전체로 학습한 모델의 샷별 출력 기준)
        T = out.merge(parts[["TimeStamp", "PART_NAME", "y", "Reason", "RH"]], on=["TimeStamp", "PART_NAME"])
        T["원인"] = [cause_class(prod, r) for r in T.itertuples()]
        T["태그"] = np.where(T.판정 == "보류", "보류", T.원인태그)
        R["4_태그표"] = pd.crosstab(T.원인, T.태그).to_dict()
        # 5) 무라벨 1층에서 경보와 태그 분포 (라벨 없이 현장에서 얼마나 자주 울리나)
        TU = d["T1"].copy(); TU["RH"] = TU.PART_NAME.str.endswith("RH").astype(int)
        outU = m.predict_stream(ST, TU)
        sc = outU[outU.판정 == "점수"]
        R["5_무라벨_태그"] = dict(점수부품=int(len(sc)), 태그=sc.원인태그.value_counts().to_dict(),
                             보류=outU[outU.판정 == "보류"].보류사유.str.split(" ").str[0].value_counts().to_dict())
        report[prod] = R
        out.to_csv(f"predictions_{prod}.csv", index=False, encoding="utf-8-sig")
        print(f"\n===================== {prod} =====================")
        print(" 1) 일괄=샷별:", R["1_일괄=샷별"])
        print(" 2) 미래 누설:", R["2_미래누설"])
        print(" 3) 시간 순서:", R["3_시간순서"])
        print(" 4) 원인 × 태그 (부품 수):"); print(pd.crosstab(T.원인, T.태그).to_string())
        print(" 5) 무라벨 1층:", R["5_무라벨_태그"])
    json.dump(report, open("risk_v1_validation.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return report

# ============================================================ E 공통 준비
def build(path):
    L, U = load_labeled(path), load_unlabeled(path); B = {}
    for prod in CFG["PRODUCTS"]:
        d = prepare_product(prod, L, U)
        S, held = d["S"].copy(), d["held"]
        S["D"] = d["D"]; S["held"] = held; S["recipe_hold"] = d["rec"]
        gap = S.TimeStamp.diff().dt.total_seconds().fillna(1e9).values
        S["since_stop"] = S.groupby("seg").cumcount() + 1                       # 구간 시작(300초 넘는 정지) 후 몇 번째 샷
        S["stop_before"] = pd.Series(np.where(S.since_stop == 1, gap, np.nan)).ffill().values   # 직전 정지 길이(초)
        P = d["parts"].merge(S.loc[~held, ["TimeStamp", "seg", "D", "since_stop", "stop_before"]], on="TimeStamp")
        P["gD"] = np.log1p(P.D)
        P["cause"] = [cause_class(prod, r) for r in P.itertuples()]
        B[prod] = dict(parts=d["parts"], S=S, P=P, pairs=d["pairs"], m=d["m"], prod=prod)
    return B

# ============================================================ E1 데이터 진단
def diagnose(path):
    raw = read_labeled_raw(path)
    dedup = raw.drop_duplicates(["TimeStamp", "PART_FACT_SERIAL", "PART_NAME"])
    R = dict(labeled=dict(원본행=int(len(raw)), 중복행=int(len(raw) - len(dedup)),
                          중복_시작시각=str(raw[raw.duplicated(["TimeStamp", "PART_FACT_SERIAL", "PART_NAME"])].TimeStamp.min())))
    miss = raw[PROC].isna().sum(); R["labeled"]["결측열"] = {k: int(v) for k, v in miss[miss > 0].items()}
    R["labeled"]["결측률_최대"] = float((raw[PROC].isna().mean()).max())
    const = [c for c in PROC if dedup[c].nunique() <= 1]; R["labeled"]["상수열"] = const
    prod = dedup.PART_NAME.str[:3]; R["labeled"]["제품별"] = {}
    for p in ["CN7", "RG3"]:
        q = dedup[prod == p]; y = (q.PassOrFail == "N")
        side = q.PART_NAME.str[-2:]
        key = [c for c in PROC if c not in const]
        g1 = q.assign(y=y).groupby(key, dropna=False).y.agg(["min", "max", "size"])
        g2 = q.assign(y=y, side=side).groupby(key + ["side"], dropna=False).y.agg(["min", "max", "size"])
        R["labeled"]["제품별"][p] = dict(부품=int(len(q)), 불량=int(y.sum()), 불량률=float(y.mean()),
                                     RH불량=int(y[side == "RH"].sum()), LH불량=int(y[side == "LH"].sum()),
                                     라벨충돌_행=int(g1[(g1["min"] != g1["max"])]["size"].sum()),
                                     라벨충돌_행_LHRH구분=int(g2[(g2["min"] != g2["max"])]["size"].sum()),
                                     불량사유=q[y].Reason.value_counts().to_dict())
    # 극단값: 제품별 중앙값에서 로버스트 z 10 초과 (측정 오류 후보)
    ext = {}
    for p in ["CN7", "RG3"]:
        q = dedup[prod == p][PROC]
        med = q.median(); mad = 1.4826 * (q - med).abs().median()
        z = ((q - med) / mad.replace(0, np.nan)).abs()
        cnt = (z > 10).sum(); ext[p] = {k: int(v) for k, v in cnt[cnt > 0].sort_values(ascending=False).head(8).items()}
    R["labeled"]["극단값_z10"] = ext
    U = read_unlabeled(path, ["TimeStamp", "EQUIP_NAME", "PART_NAME", "ERR_FACT_QTY", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL"] + PROC)
    mu = U[PROC + ["ERR_FACT_QTY"]].isna().sum()
    R["unlabeled"] = dict(행=int(len(U)), 설비=int(U.EQUIP_NAME.nunique()), 품목=int(U.PART_NAME.nunique()),
                          작업오더=int(U.groupby(["PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "PART_NAME"]).ngroups),
                          결측열={k: int(v) for k, v in mu[mu > 0].items()},
                          결측률_최대=float(U[PROC].isna().mean().max()),
                          완전중복행=int(U.duplicated(["TimeStamp", "PART_FACT_SERIAL", "PART_NAME", "EQUIP_NAME"]).sum()),
                          기간=[str(U.TimeStamp.min()), str(U.TimeStamp.max())])
    return R

# ============================================================ E2 모델 비교 + F1
def _best_thr(y, s):
    qs = np.unique(np.quantile(s, np.linspace(0.5, 0.999, 120)))
    best = (-1, np.inf)
    for t in qs:
        fl = s >= t; v = f1(int((fl & (y == 1)).sum()), int((fl & (y == 0)).sum()), int((~fl & (y == 1)).sum()))
        if v > best[0]: best = (v, t)
    return best[1]

def _inner_thr(make, X, y, g, unsup):
    """학습 구간 안에서만 3겹(구간 묶음) 교차검증으로 F1 최대 임계값을 정한다 → 시험 구간 라벨 미사용"""
    ug = np.unique(g); fold = {s: i % 3 for i, s in enumerate(ug)}; f = np.array([fold[s] for s in g])
    oof = np.zeros(len(y))
    for k in range(3):
        tr, te = f != k, f == k
        if te.sum() == 0: continue
        mdl = make()
        if unsup: mdl.fit(X[tr]); oof[te] = -mdl.score_samples(X[te])
        else:
            if y[tr].sum() == 0: oof[te] = y[tr].mean(); continue
            mdl.fit(X[tr], y[tr]); oof[te] = mdl.predict_proba(X[te])[:, 1]
    return _best_thr(y, oof)

def compare(B):
    makers = {
        "로지스틱 회귀": (lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)), False),
        "트리 부스팅(HistGB)": (lambda: HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=20, random_state=0), False),
        "이상탐지(Isolation Forest)": (lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), IsolationForest(n_estimators=200, random_state=0)), True),
    }
    OUT = {}
    for prod, b in B.items():
        P, pairs = b["P"].reset_index(drop=True), b["pairs"]
        y = P.y.values; g = P.seg.values; X = features(P); Xu = X.drop(columns=["RH"]).values; Xs = X.values
        scores = {k: np.zeros(len(P)) for k in ["상수(베이스라인)", "PINNSET v0"] + list(makers)}
        flags = {k: np.zeros(len(P), bool) for k in scores}
        for s in np.unique(g):
            tr, te = g != s, g == s
            scores["상수(베이스라인)"][te] = y[tr].mean(); flags["상수(베이스라인)"][te] = True   # 결정 규칙: 전수 검사
            bb, Lc, sp = fit_calibration(P[tr], pairs.loc[pairs.index.isin(P.TimeStamp[tr])], dict(CFG["MODEL"][prod]))
            scores["PINNSET v0"][te] = predict(P[te], bb, Lc, sp)
            flags["PINNSET v0"][te] = (P.D.values[te] >= CFG["ALARM"]) if prod == "CN7" else (P.RH.values[te] == 1)
            for name, (mk, unsup) in makers.items():
                Xm = Xu if unsup else Xs
                mdl = mk()
                if unsup: mdl.fit(Xm[tr]); scores[name][te] = -mdl.score_samples(Xm[te])
                else: mdl.fit(Xm[tr], y[tr]); scores[name][te] = mdl.predict_proba(Xm[te])[:, 1]
                thr = _inner_thr(mk, Xm[tr], y[tr], g[tr], unsup)
                flags[name][te] = scores[name][te] >= thr
        rows = []; rng = np.random.default_rng(0); q = BUDGET[prod]
        base = brier_score_loss(y, scores["상수(베이스라인)"])
        for name, sc in scores.items():
            u = seg_u(sc, g); ub = seg_u(sc, g, rng)
            prob = name != "이상탐지(Isolation Forest)"
            cb = confusion_kr(y, ub > 1 - q); cr = confusion_kr(y, flags[name])
            rows.append({"모델": name, "구간 AUC": roc_auc_score(y, u),
                         "PR-AUC": average_precision_score(y, sc if prob else u),
                         "Brier": brier_score_loss(y, sc) if prob else np.nan,
                         "Brier 개선": 1 - brier_score_loss(y, sc) / base if prob else np.nan,
                         f"예산{int(q*100)}% 검출": f"{cb['TP']}/{int(y.sum())}", f"예산{int(q*100)}% F1": cb["F1"],
                         "결정규칙 검사": cr["검사"], "결정규칙 정밀도": cr["정밀도"], "결정규칙 검출률": cr["검출률"], "결정규칙 F1": cr["F1"]})
        # 사건별(원인별) 사건 안 AUC: 같은 구간 안에서 그 원인 불량 vs 양품
        ev = {}
        for c in [c for c in P.cause.unique() if c != "양품"]:
            segs = P.seg[P.cause == c].unique(); k = P.seg.isin(segs) & P.cause.isin([c, "양품"])
            ev[c] = {name: float(roc_auc_score((P.cause[k] == c).astype(int), seg_u(sc[k.values], g[k.values]))) for name, sc in scores.items()}
        OUT[prod] = dict(표=rows, 원인별_AUC=ev, 부품=int(len(P)), 불량=int(y.sum()), 예산=q)
        P["oof_pinnset"] = scores["PINNSET v0"]; b["P"] = P
    return OUT

# ============================================================ E3 상호작용
def _lrt(X0, X1, y):
    def dev(X):
        bb = irls(X, y, np.zeros(len(y))); p = np.clip(1 / (1 + np.exp(-(X @ bb))), 1e-12, 1 - 1e-12)
        return -2 * np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)), bb
    d0, _ = dev(X0); d1, bb = dev(X1)
    return float(d0 - d1), float(chi2.sf(max(d0 - d1, 0), X1.shape[1] - X0.shape[1])), bb

def interaction(B, preds):
    R = {}
    for prod, b in B.items():
        P = b["P"]; y = P.y.values.astype(float); gD = P.gD.values; rh = P.RH.values.astype(float)
        one = np.ones(len(P))
        X_add = np.c_[one, gD, rh]; X_int = np.c_[one, gD, rh, gD * rh]
        lr_d, p_d, _ = _lrt(np.c_[one, rh], X_add, y)
        lr_r, p_r, _ = _lrt(np.c_[one, gD], X_add, y)
        lr_i, p_i, bi = _lrt(X_add, X_int, y)
        alarm = P.D.values >= CFG["ALARM"]
        tab = pd.crosstab([np.where(alarm, "경보(D≥5)", "평상"), np.where(rh == 1, "RH", "LH")], P.y, margins=False)
        cells = [{"이탈": a, "캐비티": c, "부품": int(tab.loc[(a, c)].sum()), "불량": int(tab.loc[(a, c)].get(1, 0)),
                  "불량률": float(tab.loc[(a, c)].get(1, 0) / tab.loc[(a, c)].sum())} for a, c in tab.index]
        # 재가동 직후 × 경보
        bins = pd.cut(P.since_stop, [0, 10, 50, 1e9], labels=["정지 후 1~10샷", "11~50샷", "51샷 이후"])
        rs = []
        for lab in ["정지 후 1~10샷", "11~50샷", "51샷 이후"]:
            for a in [True, False]:
                k = (bins == lab).values & (alarm == a)
                rs.append({"구간": lab, "경보": "D≥5" if a else "D<5", "부품": int(k.sum()), "불량": int(y[k].sum()),
                           "불량률": float(y[k].mean()) if k.sum() else np.nan})
        R[prod] = dict(가법모형_이탈_LR=[lr_d, p_d], 가법모형_캐비티_LR=[lr_r, p_r], 상호작용항_LR=[lr_i, p_i],
                       상호작용계수=[float(x) for x in bi], 이탈x캐비티=cells, 재가동x경보=rs)
        # 방향 상호작용(CN7): 속도 z의 부호 효과가 사건마다 뒤집히는가
        if prod == "CN7":
            pr = preds[prod]; pr = pr[pr.판정 == "점수"].merge(P[["TimeStamp", "PART_NAME", "y", "cause"]], on=["TimeStamp", "PART_NAME"])
            dirs = {}
            for c in ["A 재가동", "B 고속"]:
                segs = pr.seg[pr.cause == c].unique(); k = pr.seg.isin(segs) & pr.cause.isin([c, "양품"])
                yy = (pr.cause[k] == c).astype(int).values; gg = pr.seg[k].values
                dirs[c] = {"속도 z(+방향) AUC": float(roc_auc_score(yy, seg_u(pr.z_speed[k].values, gg))),
                           "속도 |z| AUC": float(roc_auc_score(yy, seg_u(np.abs(pr.z_speed[k].values), gg))),
                           "FLR z(+방향) AUC": float(roc_auc_score(yy, seg_u(pr.z_flr[k].values, gg))),
                           "이탈점수 D AUC": float(roc_auc_score(yy, seg_u(pr.이탈점수[k].values, gg))),
                           "불량 샷 속도 z 중앙": float(pr.z_speed[k & (pr.cause == c)].median()),
                           "불량 샷 FLR z 중앙": float(pr.z_flr[k & (pr.cause == c)].median())}
            R[prod]["방향x사건"] = dirs
    return R

# ============================================================ E4 FN·FP 조건
def fnfp(B, preds):
    R = {}
    for prod, b in B.items():
        S = b["S"]
        pr = preds[prod].merge(b["parts"][["TimeStamp", "PART_NAME", "y", "Reason", "RH"]], on=["TimeStamp", "PART_NAME"])
        pr = pr.merge(S[["TimeStamp", "since_stop", "stop_before"]], on="TimeStamp", how="left")
        pr["cause"] = [cause_class(prod, r) for r in pr.itertuples()]
        flag = (pr.판정 == "보류") | ((pr.이탈점수 >= CFG["ALARM"]) if prod == "CN7" else (pr.RH == 1))
        pr["검사"] = flag
        by = pr.groupby("cause").agg(부품=("y", "size"), 검사=("검사", "sum")).reset_index()
        # 검사 이유 분해
        why = np.where(pr.판정 == "보류", np.where(pr.보류사유.str.startswith("다른"), "보류: 다른 레시피", "보류: 기준 준비 중"),
                       np.where(flag, "경보(D≥5)" if prod == "CN7" else "RH 우선검사", "통과"))
        pr["이유"] = why
        comp = pd.crosstab(pr.이유, pr.y.map({0: "양품", 1: "불량"})).reset_index().to_dict("records")
        fn = pr[(pr.y == 1) & ~flag]; fp_alarm = pr[(pr.y == 0) & (pr.판정 == "점수") & ((pr.이탈점수 >= CFG["ALARM"]) if prod == "CN7" else False)]
        cols = ["TimeStamp", "PART_NAME", "cause", "Reason", "이탈점수", "주요변수", "since_stop", "stop_before", "원인태그"]
        R[prod] = dict(원인별_검사=by.to_dict("records"), 검사이유x결과=comp,
                       FN=fn[cols].assign(TimeStamp=fn.TimeStamp.astype(str)).round(2).to_dict("records"),
                       FP경보=fp_alarm[cols].assign(TimeStamp=fp_alarm.TimeStamp.astype(str)).round(2).to_dict("records"))
        if prod == "RG3":   # 사건별 RH/LH 불량 (E12 반전 확인)
            d = pr[pr.y == 1].sort_values("TimeStamp").copy()
            d["사건"] = (d.TimeStamp.diff() > pd.Timedelta(minutes=CFG["EVENT_GAP_MIN"])).cumsum() + 1
            R[prod]["사건별"] = d.groupby("사건").agg(시작=("TimeStamp", lambda s: str(s.min())), RH=("RH", "sum"),
                                                   LH=("RH", lambda s: int((s == 0).sum())), 검출=("검사", "sum")).reset_index().to_dict("records")
    return R

# ============================================================ E5 v1 파라미터 민감도 (태그 기준·위험점수 비교 범위)
def _tags(df, t, prod):
    z = df[["z_speed", "z_fill", "z_inj", "z_flr"]].values; D = df.이탈점수.values; out = []
    for i in range(len(df)):
        if df.판정.iat[i] != "점수": out.append("보류" if df.판정.iat[i] == "보류" else "-"); continue
        if D[i] >= CFG["ALARM"]:
            slow = z[i, 0] <= -t or z[i, 1] >= t or z[i, 3] >= t
            fast = z[i, 0] >= t or z[i, 1] <= -t or z[i, 3] <= -t
            out.append("A형" if slow and not fast else "B형" if fast and not slow else "불명")
        else:
            out.append("캐비티" if prod == "RG3" and df.RH.iat[i] == 1 else "-")
    return np.array(out)

def _pct(p, window=None, lot=None):
    res = np.full(len(p), np.nan); hist = []; hl = {}
    for i, x in enumerate(p):
        if lot is not None:
            w = hl.setdefault(lot[i], []); arr = np.array(w + [x]); w.append(x)
        else:
            arr = np.array((hist if window is None else hist[-(window - 1):]) + [x]); hist.append(x)
        res[i] = 100 * (np.sum(arr < x) + 0.5 * np.sum(arr == x)) / len(arr)
    return res

def tag_sensitivity(B, preds, fwd):
    R = {"TAG_Z": {}, "비교범위": {}}
    for prod in B:
        full = preds[prod].merge(B[prod]["parts"][["TimeStamp", "PART_NAME", "y", "Reason", "RH"]], on=["TimeStamp", "PART_NAME"])
        full["cause"] = [cause_class(prod, r) for r in full.itertuples()]
        F = fwd[prod].merge(B[prod]["parts"][["TimeStamp", "PART_NAME", "y", "RH"]], on=["TimeStamp", "PART_NAME"])
        rows = []
        for t in [1.5, 2.0, 2.5, 3.0]:
            tg = _tags(full, t, prod); tf = _tags(F, t, prod); ev = (F.판정 == "점수").values
            r = {"TAG_Z": t}
            if prod == "CN7":
                r.update({"A→A형": int(((full.cause == "A 재가동") & (tg == "A형")).sum()), "B→B형": int(((full.cause == "B 고속") & (tg == "B형")).sum()),
                          "A·B 오분류": int((((full.cause == "A 재가동") & (tg == "B형")) | ((full.cause == "B 고속") & (tg == "A형"))).sum()),
                          "불명 태그": int((tg == "불명").sum()), "양품 A형": int(((full.y == 0) & (tg == "A형")).sum()),
                          "양품 B형": int(((full.y == 0) & (tg == "B형")).sum())})
            tagged = ev & (tf != "-")
            r["시간순서 태그검사"] = f"{int(F.y[tagged].sum())}/{int(F.y[ev].sum())} (부품 {int(tagged.sum())})"
            rows.append(r)
        R["TAG_Z"][prod] = rows
        if prod == "CN7":   # 위험점수 비교 범위 (RG3는 확률이 두 값뿐이라 백분위 지표 부적합)
            E = F[F.판정 == "점수"].reset_index(drop=True); p = E.불량확률.values; y = E.y.values
            lot = B[prod]["S"].set_index("TimeStamp").lot.reindex(E.TimeStamp).values
            rr = []
            for name, kw in [("최근 250부품", dict(window=250)), ("최근 500부품", dict(window=500)), ("최근 1,000부품 (현재)", dict(window=1000)),
                             ("최근 2,000부품", dict(window=2000)), ("과거 전체", dict(window=None)), ("같은 작업오더 과거만", dict(lot=lot))]:
                sc = _pct(p, **kw); k = sc >= 95
                rr.append({"비교 범위": name, "점수95 이상 부품": int(k.sum()), "검출": f"{int(y[k].sum())}/{int(y.sum())}",
                           "구간 AUC": float(roc_auc_score(y, seg_u(sc, E.seg.values)))})
            R["비교범위"][prod] = rr
    return R

# ============================================================ E6 보류 처리 규칙
def hold_policy(B, preds):
    R = {}
    for prod in B:
        pr = preds[prod].merge(B[prod]["parts"][["TimeStamp", "PART_NAME", "y"]], on=["TimeStamp", "PART_NAME"])
        h = pr[pr.판정 == "보류"]; kind = np.where(h.보류사유.str.startswith("다른"), "다른 레시피", "기준 준비 중")
        t = pd.DataFrame({"사유": kind, "y": h.y.values}).groupby("사유").y.agg(부품="size", 불량="sum").reset_index()
        R[prod] = dict(전체부품=int(len(pr)), 보류부품=int(len(h)), 보류비율=float(len(h) / len(pr)),
                       사유별=t.to_dict("records"), 전체불량률=float(pr.y.mean()))
    return R

# ============================================================ E7 C 단서 재현 검정 (2026-09-30 사전 고정)
# 이상탐지(Isolation Forest)가 CN7 캐비티 불량(C) 시점에 호퍼·배럴 6존 온도 변화를 포착 → 가설로 고정한 뒤
# 같은 설비의 다른 제품(RG3) 캐비티 불량에서 재현되는지 단측 검정. 방향(+)과 변수는 RG3를 보기 전에 정함.
C_HYP = {"Hopper_Temperature": "+", "Barrel_Temperature_6": "+"}

def _shots_full(b, prod):
    p = b["parts"]; side = np.where(p.RH == 1, "RH", "LH")
    sh = p.groupby("TimeStamp")[PROC].first()
    yy = p.assign(side=side).pivot_table(index="TimeStamp", columns="side", values="y", aggfunc="max").fillna(0)
    rs = p.groupby("TimeStamp").Reason.apply(lambda r: "+".join(sorted(set(r.dropna()))))
    sh = sh.join(yy).join(rs.rename("R")).reset_index()
    return sh.merge(b["S"][["TimeStamp", "seg", "D", "held"]], on="TimeStamp")

def c_replication(B, N=5000, seed=0):
    rng = np.random.default_rng(seed); R = {"가설": C_HYP, "기준_alpha": 0.05 / len(C_HYP)}
    for prod in ["RG3", "CN7"]:
        d = _shots_full(B[prod], prod); d = d[~d.held.astype(bool)].reset_index(drop=True)
        bad = (d.LH + d.RH) > 0
        cav = (d.R.str.contains("가스") & (d.D < CFG["ALARM"])) if prod == "CN7" else (bad & (d.D < CFG["ALARM"]))
        segs = d.seg.values; idx = [np.where(segs == s)[0] for s in np.unique(segs)]
        rows = []
        for var, sign in C_HYP.items():
            x = d[var].values * (1 if sign == "+" else -1)
            u = seg_u(x, segs); obs = u[cav.values].mean()
            null = circular_null(x, idx, rng, N, lambda xs: seg_u(xs, segs)[cav.values].mean())
            ev = (d.TimeStamp[cav].diff() > pd.Timedelta(minutes=30)).cumsum()
            eu = pd.Series(u[cav.values]).groupby(ev.values).mean(); k = int((eu > 0.5).sum())
            rows.append({"변수": var, "방향": sign, "불량 샷": int(cav.sum()), "평균 순위 u": float(obs),
                         "원형이동 p": float((1 + (null >= obs).sum()) / (N + 1)), "사건 수": int(len(eu)), "위쪽 사건": k,
                         "부호검정 p": float(binomtest(k, len(eu), 0.5, alternative="greater").pvalue)})
        R[prod + (" (재현 검정)" if prod == "RG3" else " (발견 데이터, 참고용·선택편향 있음)")] = rows
    return R

# ============================================================ E8 보류 부품을 포함한 확률 평가 (2026-09-30)
# 동결 모델은 그대로 두고 '평가 방식'만 추가한다. 보류 부품에 확률을 주는 세 방법을 구간 빼기 검증으로 비교.
#   강제 점수   : 문(Gate)을 무시하고 이탈 점수 D를 계산해 모델 확률을 준다 (기준이 없는 샷은 사유별 경험률)
#   사유별 경험률: 학습 구간에서 같은 보류 사유 부품의 불량률 (k+0.5)/(n+1)
#   전체 불량률  : 학습 구간 전체(점수+보류) 불량률
def held_inclusion(B, preds):
    R = {}
    for prod, b in B.items():
        P = b["P"].reset_index(drop=True); pairs = b["pairs"]; S = b["S"]
        pr = preds[prod]
        H = pr[pr.판정 == "보류"].merge(b["parts"][["TimeStamp", "PART_NAME", "y", "RH"]], on=["TimeStamp", "PART_NAME"])
        H["kind"] = np.where(H.보류사유.str.startswith("다른"), "다른 레시피", "기준 준비 중")
        Dforce = deviation(S, b["m"].scale, np.zeros(len(S), bool))
        H = H.merge(pd.DataFrame({"TimeStamp": S.TimeStamp.values, "Dforce": Dforce}), on="TimeStamp", how="left").reset_index(drop=True)
        y = P.y.values; g = P.seg.values; hs = H.seg.values; hy = H.y.values
        oof = np.zeros(len(P)); pA = np.zeros(len(H)); pB = np.zeros(len(H)); pC = np.zeros(len(H))
        for s in np.unique(np.r_[g, hs]):
            tr, te, htr, hte = g != s, g == s, hs != s, hs == s
            bb, Lc, sp = fit_calibration(P[tr], pairs.loc[pairs.index.isin(P.TimeStamp[tr])], dict(CFG["MODEL"][prod]))
            if te.sum(): oof[te] = predict(P[te], bb, Lc, sp)
            prev = np.r_[y[tr], hy[htr]].mean()
            for i in np.where(hte)[0]:
                k = htr & (H.kind.values == H.kind[i])
                pA[i] = prev; pB[i] = (hy[k].sum() + 0.5) / (k.sum() + 1) if k.sum() else prev
                d = H.Dforce[i]
                pC[i] = float(predict(pd.DataFrame({"gD": [np.log1p(d)], "RH": [H.RH[i]]}), bb, Lc, sp)[0]) if not np.isnan(d) else pB[i]
        yy = np.r_[y, hy]; gg = np.r_[g, hs]
        cs_all = np.array([yy[gg != s].mean() for s in gg]); cs_sc = np.array([y[g != s].mean() for s in g])
        rows = [{"보류 부품 처리": "점수 부품만 평가 (공식값)", "평가 부품": int(len(P)), "불량": int(y.sum()),
                 "Brier": brier_score_loss(y, oof), "상수 Brier": brier_score_loss(y, cs_sc),
                 "Brier 개선": 1 - brier_score_loss(y, oof) / brier_score_loss(y, cs_sc), "구간 AUC": roc_auc_score(y, seg_u(oof, g))}]
        for name, ph in [("강제 점수 (문 무시)", pC), ("사유별 경험률 (권장)", pB), ("전체 불량률", pA)]:
            pp = np.r_[oof, ph]; bs = brier_score_loss(yy, pp); ct = brier_score_loss(yy, cs_all)
            rows.append({"보류 부품 처리": name, "평가 부품": int(len(yy)), "불량": int(yy.sum()), "Brier": bs, "상수 Brier": ct,
                         "Brier 개선": 1 - bs / ct, "구간 AUC": roc_auc_score(yy, seg_u(pp, gg))})
        f = ~H.Dforce.isna()
        R[prod] = dict(적용률=float(len(P) / (len(P) + len(H))), 표=rows,
                       보류사유별=H.groupby("kind").agg(부품=("y", "size"), 불량=("y", "sum")).reset_index().to_dict("records"),
                       강제점수_가능=int(f.sum()), 강제점수_D중앙=float(H.Dforce[f].median()) if f.any() else None,
                       강제점수_확률중앙=float(np.median(pC[f.values])) if f.any() else None, 강제점수_불량=int(hy[f.values].sum()))
    return R

# ============================================================ E9 위치 변수 (2026-09-30, C 보강 근거)
# 쿠션 위치 = 사출이 끝난 뒤 스크류 앞 수지의 끝점 → 두 캐비티에 들어간 수지 총량의 지표.
# C(캐비티) 불량 샷의 쿠션이 같은 구간 양품과 같다면 "총량은 정상, 분배만 틀어짐" (미지수 2개 중 합은 관측됨).
# 결과를 본 뒤의 분석이므로 탐색으로 표기. 모델·처방은 바꾸지 않는다.
POS = ["Cushion_Position", "Switch_Over_Position", "Plasticizing_Position", "Clamp_Open_Position"]

def position_check(B, N=3000, seed=1):
    rng = np.random.default_rng(seed); R = {"변수상태": {}, "검정": []}
    for prod in ["CN7", "RG3"]:
        d = _shots_full(B[prod], prod); d = d[~d.held.astype(bool)].reset_index(drop=True)
        R["변수상태"][prod] = [{"변수": c, "고유값 수": int(d[c].nunique()), "중앙": float(d[c].median()),
                               "흔들림(MAD)": float(1.4826 * (d[c] - d[c].median()).abs().median()), "최소": float(d[c].min()), "최대": float(d[c].max())} for c in POS]
        segs = d.seg.values; idx = [np.where(segs == s)[0] for s in np.unique(segs)]
        groups = {"C 캐비티(공정 평상)": (d.R.str.contains("가스") & (d.D < CFG["ALARM"])) if prod == "CN7" else (((d.LH + d.RH) > 0) & (d.D < CFG["ALARM"])),
                  "미성형 포함": d.R.str.contains("미성형"), "초기허용불량": d.R.str.contains("초기")}
        x = d["Cushion_Position"].values; u = seg_u(x, segs)
        for gname, gm in groups.items():
            gm = gm.values
            if gm.sum() == 0: continue
            obs = u[gm].mean()
            null = circular_null(x, idx, rng, N, lambda xs: seg_u(xs, segs)[gm].mean())
            R["검정"].append({"제품": prod, "집단": gname, "샷": int(gm.sum()), "쿠션 평균 순위 u": float(obs),
                             "양측 p (원형이동)": float((1 + (np.abs(null - 0.5) >= abs(obs - 0.5)).sum()) / (N + 1)),
                             "쿠션 최대(mm)": float(d.Cushion_Position[gm].max()), "구간 양품 중앙(mm)": float(np.median(x[~gm]))})
    return R
# ============================================================ E1~E9 실행
def run_extra(path, outdir="."):
    B = build(path)
    preds = {p: pd.read_csv(f"{outdir}/predictions_{p}.csv", parse_dates=["TimeStamp"]) for p in B}
    fwd = {p: pd.read_csv(f"{outdir}/predictions_forward_{p}.csv", parse_dates=["TimeStamp"]) for p in B}
    R = dict(E1_진단=diagnose(path)); print("E1 진단 완료", flush=True)
    R["E2_비교"] = compare(B); print("E2 비교 완료", flush=True)
    R["E3_상호작용"] = interaction(B, preds); print("E3 완료", flush=True)
    R["E4_FNFP"] = fnfp(B, preds); R["E5_민감도"] = tag_sensitivity(B, preds, fwd); R["E6_보류"] = hold_policy(B, preds)
    R["E7_C단서_재현"] = c_replication(B); print("E7 C 단서 재현 검정 완료", flush=True)
    R["E8_보류포함"] = held_inclusion(B, preds); print("E8 보류 포함 평가 완료", flush=True)
    R["E9_위치변수"] = position_check(B); print("E9 위치 변수 완료", flush=True)
    json.dump(json_safe(R), open(f"{outdir}/extra_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str, allow_nan=False)
    return R

# ============================================================ E10·E11 강건성·적용성 (2026-10-02)
# [2026-10-04 정정] 누설·겹침이 있던 평가 경로를 제거했다. E10a·E10c는 E12(evaluate.py) 예측을 쓴다.
# (E10b 구간 부트스트랩은 E12 사건 블록 bootstrap과 중복이라 삭제)
def _rebuild(b, D, held):
    S = b["S"]; P = b["parts"].merge(S.loc[~held, ["TimeStamp", "seg"]].assign(D=D[~held]), on="TimeStamp")
    P["gD"] = np.log1p(P.D); P["cause"] = [cause_class(b["prod"], r) for r in P.itertuples()]
    return P

def _oof(P, pairs, spec):
    g = P.seg.values; oof = np.zeros(len(P))
    for s in np.unique(g):
        tr, te = g != s, g == s
        bb, Lc, sp = fit_calibration(P[tr], pairs.loc[pairs.index.isin(P.TimeStamp[tr])], dict(spec)); oof[te] = predict(P[te], bb, Lc, sp)
    return oof

def _auc(P, score):
    return float(roc_auc_score(P.y.values, seg_u(score, P.seg.values)))

# ---------------------------------------------------------------- E10a 원인별 사건 안 AUC (E12 사건 블록 제외 예측 사용)
def event_out(B, outdir="."):
    R = {}
    for prod in ["CN7", "RG3"]:
        E = pd.read_csv(f"{outdir}/strict/predictions_leave_block_out_{prod}.csv", parse_dates=["TimeStamp"])
        E = E[E.status == "score"].copy(); E["cause"] = [cause_class(prod, r) for r in E.itertuples()]
        rows = []
        if prod == "CN7":
            for c in ["A 재가동", "B 고속", "C 캐비티", "설명 불가"]:
                segs = E.seg[E.cause == c].unique(); T = E[E.seg.isin(segs) & E.cause.isin([c, "양품"])]
                rows.append({"원인": c, "불량 부품": int((E.cause == c).sum()), "사건 안 AUC": float(roc_auc_score((T.cause == c).astype(int), seg_u(T.PINNSET.values, T.seg.values))),
                             "D≥5 경보로 검출": f"{int(((E.cause == c) & (E.D >= CFG['ALARM'])).sum())}/{int((E.cause == c).sum())}"})
        else:
            d = E[E.y == 1].sort_values("TimeStamp"); ev = (d.TimeStamp.diff() > pd.Timedelta(minutes=CFG["EVENT_GAP_MIN"])).cumsum()
            for e, q in d.groupby(ev.values):
                T = E[E.seg.isin(q.seg.unique()) & ((E.y == 0) | E.index.isin(q.index))]
                rows.append({"사건": int(e) + 1, "시작": str(q.TimeStamp.min())[:16], "RH 불량": int((q.RH == 1).sum()), "LH 불량": int((q.RH == 0).sum()),
                             "사건 안 AUC": float(roc_auc_score(T.y.values, seg_u(T.PINNSET.values, T.seg.values)))})
        R[prod] = rows
    return R

# ---------------------------------------------------------------- E10c 경보 기준별 운영점 (E12 엄격 시간 순서 예측 사용)
def operating(B, outdir="."):
    E = pd.read_csv(f"{outdir}/strict/predictions_forward_CN7.csv", parse_dates=["TimeStamp"])
    E = E[E.status != "unavailable"]; y = E.y.values.astype(bool); held = (E.status == "held").values; sc = (E.status == "score").values
    sw = []
    for t in [2, 3, 4, 5, 6, 8, 10, 20]:
        k = sc & (E.D.values >= t); i2 = held | k
        sw.append({"경보 기준 D≥": t, "검사 비율(보정 준비 후, 보류 포함)": float(i2.mean()), "검출": f"{int((y & i2).sum())}/{int(y.sum())}",
                   "경보 F1(점수 부품)": f1(int((y & k).sum()), int((~y & k).sum()), int((y & sc).sum() - (y & k).sum()))})
    return {"CN7 경보 기준별": sw}

# ---------------------------------------------------------------- E10d 변수 빼기·음성 대조
def ablation(B, seed=0):
    rng = np.random.default_rng(seed); R = {}
    b = B["CN7"]; S = b["S"]; m = b["m"]; pairs = b["pairs"]; spec = CFG["MODEL"]["CN7"]; rec = S.recipe_hold.values
    V0 = list(CFG["VARS"]); rows = []
    def rowfor(name, P):
        oof = _oof(P, pairs, spec); r = {"구성": name, "구간 AUC": _auc(P, oof)}
        for c in ["A 재가동", "B 고속"]: r[f"{c[0]} 경보"] = f"{int(((P.cause == c) & (P.D >= CFG['ALARM'])).sum())}/{int((P.cause == c).sum())}"
        return r
    rows.append(rowfor("기본 (변수 4개)", b["P"]))
    for drop in V0:
        with override(VARS=[v for v in V0 if v != drop]):
            D = deviation(S, m.scale, rec); held = rec | np.isnan(D)
            rows.append(rowfor(f"− {drop}", _rebuild(b, D, held)))
    P = b["P"].copy(); P["D"] = P.groupby("seg").D.transform(lambda q: rng.permutation(q.values)); P["gD"] = np.log1p(P.D)
    rows.append({"구성": "음성 대조: D를 구간 안에서 섞기", "구간 AUC": _auc(P, _oof(P, pairs, spec))})
    P = b["P"].copy(); P["D"] = np.abs(rng.standard_normal(len(P))) * 3; P["gD"] = np.log1p(P.D)
    rows.append({"구성": "음성 대조: D를 난수로", "구간 AUC": _auc(P, _oof(P, pairs, spec))})
    R["CN7"] = rows
    b = B["RG3"]; rows = []
    for name, sp in [("캐비티만 (최종)", dict(process=False, cavity="shrink")), ("공정만", dict(process=True, cavity="none")),
                     ("공정 + 캐비티", dict(process=True, cavity="shrink")), ("둘 다 없음 (상수)", dict(process=False, cavity="none"))]:
        r = evaluate_loso(b["P"], b["pairs"], sp); rows.append({"구성": name, "구간 AUC": r["auc_within"], "Brier 개선": r["improve"]})
    R["RG3"] = rows
    return R

# ---------------------------------------------------------------- E10e 민감도
def robustness(B):
    R = {}
    def eval_all(tag, val, prods=("CN7", "RG3")):
        out = {"파라미터": tag, "값": val}
        for prod in prods:
            b = B[prod]; S = b["S"]; m = b["m"]
            rec = recipe_gate(S, m.gmed, m.gsc); D = deviation(S, m.scale, rec); held = rec | np.isnan(D); P = _rebuild(b, D, held)
            r = evaluate_loso(P, b["pairs"], CFG["MODEL"][prod])
            out.update({f"{prod} 구간AUC": r["auc_within"], f"{prod} Brier개선": r["improve"], f"{prod} 적용률": float(len(P) / len(b["parts"])),
                        f"{prod} 보류된 불량": int(b["parts"].y.sum() - P.y.sum())})
            if prod == "CN7": out["CN7 A·B 경보"] = f"{int(((P.cause.isin(['A 재가동', 'B 고속'])) & (P.D >= CFG['ALARM'])).sum())}/16"
        return out
    rows = []
    for g in [2.0, 2.5, 3.0, 3.5, 4.0]:
        with override(GATE_DIST=g): rows.append(eval_all("GATE_DIST", g))
    for a in [4.0, 4.5, 5.0, 5.5, 6.0]:
        with override(ALARM=a): rows.append(eval_all("ALARM", a, ("CN7",)))
    for r_ in [10, 20, 40]:
        with override(REBASE_RUN=r_): rows.append(eval_all("REBASE_RUN", r_, ("CN7",)))
    tau = []
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        with override(SHRINK_TAU=t):
            b = B["RG3"]; r = evaluate_loso(b["P"], b["pairs"], CFG["MODEL"]["RG3"])
            bb, Lc, sp = fit_calibration(b["P"], b["pairs"], dict(CFG["MODEL"]["RG3"]))
            pr = predict(pd.DataFrame({"gD": [0, 0], "RH": [0, 1]}), bb, Lc, sp)
        tau.append({"SHRINK_TAU": t, "RG3 구간AUC": r["auc_within"], "RG3 Brier개선": r["improve"], "RH/LH 확률비": float(pr[1] / pr[0])})
    seg = []; b = B["CN7"]
    for gap in [200, 250, 300, 400, 500, 600]:
        S = b["S"]; newseg = (S.TimeStamp.diff().dt.total_seconds() > gap).cumsum()
        P = b["P"].copy(); P["seg"] = P.TimeStamp.map(pd.Series(newseg.values, index=S.TimeStamp)).values
        r = evaluate_loso(P, b["pairs"], CFG["MODEL"]["CN7"])
        seg.append({"SEG_GAP_S": gap, "구간 수": int(newseg.nunique()), "CN7 구간AUC": r["auc_within"], "CN7 Brier개선": r["improve"]})
    R["문·경보·기준교체"] = rows; R["축소강도"] = tau; R["구간분할"] = seg
    return R

# ---------------------------------------------------------------- E11 다른 설비·품목 적용성
def applicability(path):
    from scipy.stats import spearmanr
    VARS3 = ["Max_Injection_Speed", "Filling_Time", "Injection_Time"]
    U = read_unlabeled(path, ["TimeStamp", "EQUIP_NAME", "PART_NAME", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "ERR_FACT_QTY"] + VARS3)
    rows = []; lots = []
    for (eq, item), u in U.groupby(["EQUIP_NAME", "PART_NAME"]):
        s = u.sort_values("TimeStamp").drop_duplicates("TimeStamp").copy()
        s["lot"] = s.PART_FACT_PLAN_DATE.astype(str) + "_" + s.PART_FACT_SERIAL.astype(str)
        s = s[s.groupby("lot").TimeStamp.transform("size") >= 30]
        if s.lot.nunique() < 3: continue
        s = s.sort_values("TimeStamp").reset_index(drop=True)
        gap = s.TimeStamp.diff().dt.total_seconds().fillna(1e9); newseg = (gap > CFG["SEG_GAP_S"]) | s.lot.ne(s.lot.shift())
        s["since"] = s.groupby(newseg.cumsum()).cumcount() + 1; s["k_in_lot"] = s.groupby("lot").cumcount() + 1
        Z = []
        for c in VARS3:
            r = s[c] - s.groupby("lot")[c].transform("median")
            sc = dev_scale(s[c])                    # v0과 같은 정의: 여러 작업오더에 걸친 원값의 흔들림(기록 눈금 하한)
            Z.append((r / sc).abs())
        s["D"] = pd.concat(Z, axis=1).max(axis=1)
        steady = (s.since > 20) & (s.k_in_lot > CFG["REF_MIN"]); first = (s.since <= 3) & (s.k_in_lot > CFG["REF_MIN"])
        rows.append({"설비": eq, "품목": item, "라벨 제품": "CN7" if "CN7" in item else ("RG3" if "RG3" in item else "-"), "작업오더": int(s.lot.nunique()), "샷": int(len(s)),
                     "정상 운전 경보율": float((s.D[steady] >= CFG["ALARM"]).mean()) if steady.any() else np.nan,
                     "재가동 직후 3샷 경보율": float((s.D[first] >= CFG["ALARM"]).mean()) if first.any() else np.nan})
        g = s.groupby("lot").agg(경보율=("D", lambda d: float((d >= CFG["ALARM"]).mean())), 샷=("D", "size"), 불량=("ERR_FACT_QTY", "max"))
        g["설비"] = eq; lots.append(g)
    T = pd.DataFrame(rows); L = pd.concat(lots)
    byeq = T.groupby("설비").apply(lambda q: pd.Series({"품목": int(len(q)), "작업오더": int(q.작업오더.sum()), "샷": int(q.샷.sum()),
                                                   "정상 경보율(샷 가중)": float(np.average(q["정상 운전 경보율"].fillna(0), weights=q.샷)),
                                                   "재가동 직후 경보율(샷 가중)": float(np.average(q["재가동 직후 3샷 경보율"].fillna(0), weights=q.샷))}), include_groups=False).reset_index()
    rho, p = spearmanr(L.경보율, L.불량 / L.샷)
    lab = T[T["라벨 제품"] != "-"][["설비", "품목", "작업오더", "샷", "정상 운전 경보율", "재가동 직후 3샷 경보율"]]
    return dict(설비별=byeq.to_dict("records"), 라벨제품_품목=lab.to_dict("records"), 품목수=int(len(T)), 전체_정상경보율_중앙=float(T["정상 운전 경보율"].median()),
                전체_재가동경보율_중앙=float(T["재가동 직후 3샷 경보율"].median()),
                오더경보율_vs_ERR_FACT_QTY=[float(rho), float(p), int(len(L))],
                비고="FLR은 품목별 두께를 몰라 제외하고 시간·속도 3개 변수로만 계산. 척도는 v0과 같은 정의(오더에 걸친 원값 흔들림)이나 레시피 문·1층 선별 없이 품목 전체로 잡음. 기준은 오더 중앙값(사후). ERR_FACT_QTY는 정의가 불확실해 참고용")

def run_verify(path, outdir="."):
    B = build(path)
    R = {"E10a_원인별_사건블록제외": event_out(B, outdir)}; print("E10a", flush=True)
    R["E10c_경보기준별_운영점"] = operating(B, outdir); print("E10c", flush=True)
    R["E10d_변수빼기_고정기준"] = ablation(B); print("E10d", flush=True)
    R["E10e_민감도_고정기준"] = robustness(B); print("E10e", flush=True)
    R["E11_적용성"] = applicability(path); print("E11", flush=True)
    json.dump(R, open(f"{outdir}/verify_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return R
