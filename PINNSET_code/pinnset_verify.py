"""
PINNSET 강건성 검증 (E10)과 다른 설비 적용성 점검 (E11) — 2026-10-02
=====================================================================
보고서의 강건성 수치를 제출 코드만으로 재현하기 위해 독립 재검증(V04·V05·V06·V08·V09·V10)의 핵심을 다시 구현했다.
모델은 바꾸지 않는다. 동결 파라미터를 잠시 바꾸는 민감도 계산은 끝나면 반드시 원래 값으로 되돌린다.
  [2026-10-04 정정] 누설·겹침이 있던 평가 경로를 이 모듈에서 제거했다.
  E10a 원인별 사건 안 AUC  E12 사건 블록 제외 예측(시험 구간 전체 학습 제외, 전처리 학습 내부)에서 원인별로 계산
  (E10b 구간 부트스트랩은 E12 사건 블록 bootstrap과 중복이라 삭제)
  E10c 경보 기준별 운영점  E12 엄격 시간 순서 예측에서 경보 기준 D를 바꿔 검사 비율·검출 계산
  ※ E10d·E10e는 정상 기준·척도를 전체 기간 무라벨로 고정한 '고정 기준' 상대 비교(참고용)다. 공식 성능은 E12.
  E10d 변수 빼기·음성 대조  이탈 변수를 하나씩 빼기, 이탈 점수 섞기·난수 대체, RG3 성분 조합 (V09)
  E10e 민감도          문 거리, 경보 기준, 기준 교체, 축소 강도, 구간 분할 (V10)
  E11  적용성 점검      무라벨 13개 설비·품목에 이탈 점수를 적용했을 때 정상 운전 경보율과 재가동 반응
"""
import json, copy
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import pinnset_risk_v0 as v0
import pinnset_risk_v1 as v1
from pinnset_risk_v0 import CFG
from pinnset_extra import seg_u, f1, build

def _rebuild(b, D, held):
    S = b["S"]; P = b["parts"].merge(S.loc[~held, ["TimeStamp", "seg"]].assign(D=D[~held]), on="TimeStamp")
    P["gD"] = np.log1p(P.D); P["cause"] = [v1.cause_class(b["prod"], r) for r in P.itertuples()]
    return P

def _oof(P, pairs, spec):
    y = P.y.values; g = P.seg.values; oof = np.zeros(len(P))
    for s in np.unique(g):
        tr, te = g != s, g == s
        bb, Lc, sp = v0.fit_calibration(P[tr], pairs.loc[pairs.index.isin(P.TimeStamp[tr])], dict(spec)); oof[te] = v0.predict(P[te], bb, Lc, sp)
    return oof

def _auc(P, score):
    return float(roc_auc_score(P.y.values, seg_u(score, P.seg.values)))

# ---------------------------------------------------------------- E10a 원인별 사건 안 AUC (E12 사건 블록 제외 예측 사용)
def event_out(B, outdir="."):
    R = {}
    for prod in ["CN7", "RG3"]:
        E = pd.read_csv(f"{outdir}/strict/predictions_leave_block_out_{prod}.csv", parse_dates=["TimeStamp"])
        E = E[E.status == "score"].copy(); E["cause"] = [v1.cause_class(prod, r) for r in E.itertuples()]
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
    try:
        for drop in V0:
            CFG["VARS"] = [v for v in V0 if v != drop]; D = v0.deviation(S, m.scale, rec); held = rec | np.isnan(D)
            rows.append(rowfor(f"− {drop}", _rebuild(b, D, held)))
    finally:
        CFG["VARS"] = V0
    P = b["P"].copy(); P["D"] = P.groupby("seg").D.transform(lambda q: rng.permutation(q.values)); P["gD"] = np.log1p(P.D)
    rows.append({"구성": "음성 대조: D를 구간 안에서 섞기", "구간 AUC": _auc(P, _oof(P, pairs, spec))})
    P = b["P"].copy(); P["D"] = np.abs(rng.standard_normal(len(P))) * 3; P["gD"] = np.log1p(P.D)
    rows.append({"구성": "음성 대조: D를 난수로", "구간 AUC": _auc(P, _oof(P, pairs, spec))})
    R["CN7"] = rows
    b = B["RG3"]; rows = []
    for name, sp in [("캐비티만 (최종)", dict(process=False, cavity="shrink")), ("공정만", dict(process=True, cavity="none")),
                     ("공정 + 캐비티", dict(process=True, cavity="shrink")), ("둘 다 없음 (상수)", dict(process=False, cavity="none"))]:
        r = v0.evaluate_loso(b["P"], b["pairs"], sp); rows.append({"구성": name, "구간 AUC": r["auc_within"], "Brier 개선": r["improve"]})
    R["RG3"] = rows
    return R

# ---------------------------------------------------------------- E10e 민감도
def sensitivity(B):
    R = {}; keep = copy.deepcopy(CFG)
    def eval_all(tag, val, prods=("CN7", "RG3")):
        out = {"파라미터": tag, "값": val}
        for prod in prods:
            b = B[prod]; S = b["S"]; m = b["m"]
            rec = v0.recipe_gate(S, m.gmed, m.gsc); D = v0.deviation(S, m.scale, rec); held = rec | np.isnan(D); P = _rebuild(b, D, held)
            r = v0.evaluate_loso(P, b["pairs"], CFG["MODEL"][prod])
            out.update({f"{prod} 구간AUC": r["auc_within"], f"{prod} Brier개선": r["improve"], f"{prod} 적용률": float(len(P) / len(b["parts"])),
                        f"{prod} 보류된 불량": int(b["parts"].y.sum() - P.y.sum())})
            if prod == "CN7": out["CN7 A·B 경보"] = f"{int(((P.cause.isin(['A 재가동', 'B 고속'])) & (P.D >= CFG['ALARM'])).sum())}/16"
        return out
    rows = []
    try:
        for g in [2.0, 2.5, 3.0, 3.5, 4.0]: CFG["GATE_DIST"] = g; rows.append(eval_all("GATE_DIST", g))
        CFG["GATE_DIST"] = keep["GATE_DIST"]
        for a in [4.0, 4.5, 5.0, 5.5, 6.0]: CFG["ALARM"] = a; rows.append(eval_all("ALARM", a, ("CN7",)))
        CFG["ALARM"] = keep["ALARM"]
        for r_ in [10, 20, 40]: CFG["REBASE_RUN"] = r_; rows.append(eval_all("REBASE_RUN", r_, ("CN7",)))
        CFG["REBASE_RUN"] = keep["REBASE_RUN"]
        tau = []
        for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            CFG["SHRINK_TAU"] = t; b = B["RG3"]; r = v0.evaluate_loso(b["P"], b["pairs"], CFG["MODEL"]["RG3"])
            bb, Lc, sp = v0.fit_calibration(b["P"], b["pairs"], dict(CFG["MODEL"]["RG3"]))
            pr = v0.predict(pd.DataFrame({"gD": [0, 0], "RH": [0, 1]}), bb, Lc, sp)
            tau.append({"SHRINK_TAU": t, "RG3 구간AUC": r["auc_within"], "RG3 Brier개선": r["improve"], "RH/LH 확률비": float(pr[1] / pr[0])})
        CFG["SHRINK_TAU"] = keep["SHRINK_TAU"]
        seg = []; b = B["CN7"]
        for gap in [200, 250, 300, 400, 500, 600]:
            S = b["S"]; newseg = (S.TimeStamp.diff().dt.total_seconds() > gap).cumsum()
            P = b["P"].copy(); P["seg"] = P.TimeStamp.map(pd.Series(newseg.values, index=S.TimeStamp)).values
            r = v0.evaluate_loso(P, b["pairs"], CFG["MODEL"]["CN7"])
            seg.append({"SEG_GAP_S": gap, "구간 수": int(newseg.nunique()), "CN7 구간AUC": r["auc_within"], "CN7 Brier개선": r["improve"]})
    finally:
        for k_, v_ in keep.items(): CFG[k_] = v_
    R["문·경보·기준교체"] = rows; R["축소강도"] = tau; R["구간분할"] = seg
    return R

# ---------------------------------------------------------------- E11 다른 설비·품목 적용성
def applicability(path):
    from scipy.stats import spearmanr
    VARS3 = ["Max_Injection_Speed", "Filling_Time", "Injection_Time"]
    U = pd.read_csv(f"{path}/unlabeled_data.csv", usecols=["TimeStamp", "EQUIP_NAME", "PART_NAME", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "ERR_FACT_QTY"] + VARS3, parse_dates=["TimeStamp"])
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
            sc = v0.dev_scale(s[c])                    # v0과 같은 정의: 여러 작업오더에 걸친 원값의 흔들림(기록 눈금 하한)
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

def run(path, outdir="."):
    B = build(path)
    for p in B: B[p]["prod"] = p
    R = {"E10a_원인별_사건블록제외": event_out(B, outdir)}; print("E10a", flush=True)
    R["E10c_경보기준별_운영점"] = operating(B, outdir); print("E10c", flush=True)
    R["E10d_변수빼기_고정기준"] = ablation(B); print("E10d", flush=True)
    R["E10e_민감도_고정기준"] = sensitivity(B); print("E10e", flush=True)
    R["E11_적용성"] = applicability(path); print("E11", flush=True)
    json.dump(R, open(f"{outdir}/verify_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return R
