"""
PINNSET 추가 분석 (서면평가 문항 대응) — 2026-09-30
※ 이 모듈의 비교·민감도·보류 평가는 정상 기준을 전체 기간 무라벨로 고정한 '고정 기준' 평가(참고용)다. 공식 성능은 pinnset_strict.py(E12).
================================================
동결된 v0 모델(구조·파라미터)은 바꾸지 않는다. 여기서는 평가·해석만 추가한다.

  E1 diagnose        데이터 진단: 결측·중복·상수열·극단값·불균형·라벨 충돌          (문항 1)
  E2 compare         같은 구간 빼기(LOSO) 조건 모델 비교 + F1                      (문항 2)
  E3 interaction     변수 간 상호작용: 방향×사건, 이탈×캐비티, 재가동×경보          (문항 3)
  E4 fnfp            FN·FP가 몰리는 공정조건                                        (문항 3)
  E5 sensitivity     v1 전용 파라미터 민감도: TAG_Z, 위험점수 비교 범위            (남은 작업)
  E6 hold_policy     보류 부품 처리 규칙과 비용                                     (남은 작업)
  E7 c_replication   C 단서(호퍼·배럴6) 사전 고정 가설의 RG3 재현 검정              (문항 3)
  E8 held_inclusion  보류 부품을 포함한 확률 평가: 강제 점수 vs 사유별 경험률 vs 전체 불량률  (문항 2·5)
  E9 position_check  위치 변수 상태와 쿠션 위치 검정 (C 총량 정상 → 분배 문제 보강, 탐색)    (문항 1·3)

선행: pinnset_risk_v1.py 실행 결과 predictions_{제품}.csv, predictions_forward_{제품}.csv 가 작업 폴더에 있어야 한다.
"""
import json, warnings
import numpy as np, pandas as pd
from scipy.stats import chi2
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
warnings.filterwarnings("ignore")
import pinnset_risk_v0 as v0
import pinnset_risk_v1 as v1
from pinnset_risk_v0 import CFG

PROC = ["Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time", "Clamp_Close_Time", "Cushion_Position",
        "Switch_Over_Position", "Plasticizing_Position", "Clamp_Open_Position", "Max_Injection_Speed", "Max_Screw_RPM",
        "Average_Screw_RPM", "Max_Injection_Pressure", "Max_Switch_Over_Pressure", "Max_Back_Pressure",
        "Average_Back_Pressure"] + [f"Barrel_Temperature_{i}" for i in range(1, 8)] + ["Hopper_Temperature"] + \
       [f"Mold_Temperature_{i}" for i in range(1, 13)]
BUDGET = {"CN7": 0.05, "RG3": 0.50}          # V07과 같은 검사 예산


def _json_safe(o):
    """NaN·무한대를 null로 바꿔 엄격한 JSON 파서에서도 읽히게 한다"""
    import math
    if isinstance(o, dict): return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_json_safe(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o): return None
    if hasattr(o, "item") and getattr(o, "ndim", 1) == 0:
        v = o.item(); return None if isinstance(v, float) and not math.isfinite(v) else v
    return o

def seg_u(score, seg, rng=None):
    """구간 안 centered mid-rank. rng가 있으면 동점을 무작위로 깨서 '상위 k%' 선택에 쓴다"""
    s = pd.Series(np.asarray(score, float))
    if rng is not None:
        s = s + rng.uniform(0, 1e-9, len(s)) * (s.abs().max() + 1)
    return s.groupby(np.asarray(seg)).transform(lambda q: (q.rank() - 0.5) / len(q)).values

def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0

def confusion(y, flag):
    y = np.asarray(y).astype(bool); flag = np.asarray(flag).astype(bool)
    tp, fp, fn = int((y & flag).sum()), int((~y & flag).sum()), int((y & ~flag).sum())
    return dict(검사=int(flag.sum()), TP=tp, FP=fp, FN=fn, 정밀도=tp / max(tp + fp, 1), 검출률=tp / max(tp + fn, 1), F1=f1(tp, fp, fn))

# ============================================================ 공통 준비
def build(path):
    data = v1.prepare(path); B = {}
    for prod, d in data.items():
        parts, S, ST, pairs = d["parts"], d["S"], d["ST"], d["pairs"]
        m = v1.RiskModel(prod); m.set_references(S, ST)
        rec = v0.recipe_gate(S, m.gmed, m.gsc); D = v0.deviation(S, m.scale, rec)
        held = rec | np.isnan(D)
        S = S.copy(); S["D"] = D; S["held"] = held; S["recipe_hold"] = rec
        gap = S.TimeStamp.diff().dt.total_seconds().fillna(1e9).values
        S["since_stop"] = S.groupby("seg").cumcount() + 1                       # 구간 시작(300초 넘는 정지) 후 몇 번째 샷
        S["stop_before"] = pd.Series(np.where(S.since_stop == 1, gap, np.nan)).ffill().values   # 직전 정지 길이(초)
        P = parts.merge(S.loc[~held, ["TimeStamp", "seg", "D", "since_stop", "stop_before"]], on="TimeStamp")
        P["gD"] = np.log1p(P.D)
        P["cause"] = [v1.cause_class(prod, r) for r in P.itertuples()]
        B[prod] = dict(parts=parts, S=S, P=P, pairs=pairs, m=m)
    return B

# ============================================================ E1 데이터 진단
def diagnose(path):
    raw = pd.read_csv(f"{path}/labeled_data.csv", parse_dates=["TimeStamp"])
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
    U = pd.read_csv(f"{path}/unlabeled_data.csv", usecols=["TimeStamp", "EQUIP_NAME", "PART_NAME", "ERR_FACT_QTY",
                                                            "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL"] + PROC)
    mu = U[PROC + ["ERR_FACT_QTY"]].isna().sum()
    R["unlabeled"] = dict(행=int(len(U)), 설비=int(U.EQUIP_NAME.nunique()), 품목=int(U.PART_NAME.nunique()),
                          작업오더=int(U.groupby(["PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "PART_NAME"]).ngroups),
                          결측열={k: int(v) for k, v in mu[mu > 0].items()},
                          결측률_최대=float(U[PROC].isna().mean().max()),
                          완전중복행=int(U.duplicated(["TimeStamp", "PART_FACT_SERIAL", "PART_NAME", "EQUIP_NAME"]).sum()),
                          기간=[str(U.TimeStamp.min()), str(U.TimeStamp.max())])
    return R

# ============================================================ E2 모델 비교 + F1
def _xy(P):
    X = P[PROC].copy()
    X["Tm"], X["Tw"], X["FLR"], X["RH"] = P.Tm, P.Tw, P.FLR, P.RH
    return X

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
        y = P.y.values; g = P.seg.values; X = _xy(P); Xu = X.drop(columns=["RH"]).values; Xs = X.values
        scores = {k: np.zeros(len(P)) for k in ["상수(베이스라인)", "PINNSET v0"] + list(makers)}
        flags = {k: np.zeros(len(P), bool) for k in scores}
        for s in np.unique(g):
            tr, te = g != s, g == s
            scores["상수(베이스라인)"][te] = y[tr].mean(); flags["상수(베이스라인)"][te] = True   # 결정 규칙: 전수 검사
            bb, Lc, sp = v0.fit_calibration(P[tr], pairs.loc[pairs.index.isin(P.TimeStamp[tr])], dict(CFG["MODEL"][prod]))
            scores["PINNSET v0"][te] = v0.predict(P[te], bb, Lc, sp)
            flags["PINNSET v0"][te] = (P.D.values[te] >= CFG["ALARM"]) if prod == "CN7" else (P.RH.values[te] == 1)
            for name, (mk, unsup) in makers.items():
                Xm = Xu if unsup else Xs
                mdl = mk()
                if unsup: mdl.fit(Xm[tr]); sc_tr = None; scores[name][te] = -mdl.score_samples(Xm[te])
                else: mdl.fit(Xm[tr], y[tr]); scores[name][te] = mdl.predict_proba(Xm[te])[:, 1]
                thr = _inner_thr(mk, Xm[tr], y[tr], g[tr], unsup)
                flags[name][te] = scores[name][te] >= thr
        rows = []; rng = np.random.default_rng(0); q = BUDGET[prod]
        base = brier_score_loss(y, scores["상수(베이스라인)"])
        for name, sc in scores.items():
            u = seg_u(sc, g); ub = seg_u(sc, g, rng)
            prob = name != "이상탐지(Isolation Forest)"
            cb = confusion(y, ub > 1 - q); cr = confusion(y, flags[name])
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
        bb = v0.irls(X, y, np.zeros(len(y))); p = np.clip(1 / (1 + np.exp(-(X @ bb))), 1e-12, 1 - 1e-12)
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
        P = b["P"]; S = b["S"]
        pr = preds[prod].merge(b["parts"][["TimeStamp", "PART_NAME", "y", "Reason", "RH"]], on=["TimeStamp", "PART_NAME"])
        pr = pr.merge(S[["TimeStamp", "since_stop", "stop_before"]], on="TimeStamp", how="left")
        pr["cause"] = [v1.cause_class(prod, r) for r in pr.itertuples()]
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

# ============================================================ E5 v1 파라미터 민감도
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

def sensitivity(B, preds, fwd):
    R = {"TAG_Z": {}, "비교범위": {}}
    for prod in B:
        full = preds[prod].merge(B[prod]["parts"][["TimeStamp", "PART_NAME", "y", "Reason", "RH"]], on=["TimeStamp", "PART_NAME"])
        full["cause"] = [v1.cause_class(prod, r) for r in full.itertuples()]
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

def run(path, outdir="."):
    B = build(path)
    preds = {p: pd.read_csv(f"{outdir}/predictions_{p}.csv", parse_dates=["TimeStamp"]) for p in B}
    fwd = {p: pd.read_csv(f"{outdir}/predictions_forward_{p}.csv", parse_dates=["TimeStamp"]) for p in B}
    R = dict(E1_진단=diagnose(path)); print("E1 진단 완료", flush=True)
    R["E2_비교"] = compare(B); print("E2 비교 완료", flush=True)
    R["E3_상호작용"] = interaction(B, preds); print("E3 완료", flush=True)
    R["E4_FNFP"] = fnfp(B, preds); R["E5_민감도"] = sensitivity(B, preds, fwd); R["E6_보류"] = hold_policy(B, preds)
    R["E7_C단서_재현"] = c_replication(B); print("E7 C 단서 재현 검정 완료", flush=True)
    R["E8_보류포함"] = held_inclusion(B, preds); print("E8 보류 포함 평가 완료", flush=True)
    R["E9_위치변수"] = position_check(B); print("E9 위치 변수 완료", flush=True)
    json.dump(_json_safe(R), open(f"{outdir}/extra_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str, allow_nan=False)
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
    from scipy.stats import binomtest
    rng = np.random.default_rng(seed); R = {"가설": C_HYP, "기준_alpha": 0.05 / len(C_HYP)}
    for prod in ["RG3", "CN7"]:
        d = _shots_full(B[prod], prod); d = d[~d.held.astype(bool)].reset_index(drop=True)
        bad = (d.LH + d.RH) > 0
        cav = (d.R.str.contains("가스") & (d.D < CFG["ALARM"])) if prod == "CN7" else (bad & (d.D < CFG["ALARM"]))
        segs = d.seg.values; idx = [np.where(segs == s)[0] for s in np.unique(segs)]
        rows = []
        for var, sign in C_HYP.items():
            x = d[var].values * (1 if sign == "+" else -1)
            u = seg_u(x, segs); obs = u[cav.values].mean(); null = np.empty(N)
            for i in range(N):
                xs = x.copy()
                for ii in idx:
                    if len(ii) > 1: xs[ii] = np.roll(x[ii], rng.integers(1, len(ii)))
                null[i] = seg_u(xs, segs)[cav.values].mean()
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
        Dforce = v0.deviation(S, b["m"].scale, np.zeros(len(S), bool))
        H = H.merge(pd.DataFrame({"TimeStamp": S.TimeStamp.values, "Dforce": Dforce}), on="TimeStamp", how="left").reset_index(drop=True)
        y = P.y.values; g = P.seg.values; hs = H.seg.values; hy = H.y.values
        oof = np.zeros(len(P)); pA = np.zeros(len(H)); pB = np.zeros(len(H)); pC = np.zeros(len(H))
        for s in np.unique(np.r_[g, hs]):
            tr, te, htr, hte = g != s, g == s, hs != s, hs == s
            bb, Lc, sp = v0.fit_calibration(P[tr], pairs.loc[pairs.index.isin(P.TimeStamp[tr])], dict(CFG["MODEL"][prod]))
            if te.sum(): oof[te] = v0.predict(P[te], bb, Lc, sp)
            prev = np.r_[y[tr], hy[htr]].mean()
            for i in np.where(hte)[0]:
                k = htr & (H.kind.values == H.kind[i])
                pA[i] = prev; pB[i] = (hy[k].sum() + 0.5) / (k.sum() + 1) if k.sum() else prev
                d = H.Dforce[i]
                pC[i] = float(v0.predict(pd.DataFrame({"gD": [np.log1p(d)], "RH": [H.RH[i]]}), bb, Lc, sp)[0]) if not np.isnan(d) else pB[i]
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
            obs = u[gm].mean(); null = np.empty(N)
            for i in range(N):
                xs = x.copy()
                for ii in idx:
                    if len(ii) > 1: xs[ii] = np.roll(x[ii], rng.integers(1, len(ii)))
                null[i] = seg_u(xs, segs)[gm].mean()
            R["검정"].append({"제품": prod, "집단": gname, "샷": int(gm.sum()), "쿠션 평균 순위 u": float(obs),
                             "양측 p (원형이동)": float((1 + (np.abs(null - 0.5) >= abs(obs - 0.5)).sum()) / (N + 1)),
                             "쿠션 최대(mm)": float(d.Cushion_Position[gm].max()), "구간 양품 중앙(mm)": float(np.median(x[~gm]))})
    return R
