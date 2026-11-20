"""
PINNSET 위험점수 모델 v0 — 재현 스크립트
================================================
설계도 문서(위험점수_모델_설계보고서.pdf)의 단계 번호와 함수가 1:1로 대응한다.

  S1 load_labeled / load_unlabeled   데이터 읽기, 중복 제거, 파생량(Tm, Tw, FLR)
  S2 shot_table                      부품 행 → 샷 단위 표, 작업오더 키, 연속 가동 구간
  S3 reference                       라벨 양품 샷의 기준값(중앙값)과 흔들림 폭
  S4 select_tier1                    무라벨 1층 선별 (레시피 지문 거리)
  S5 recipe_gate                     적용 범위 문 (샷 단위 설정 거리)
  S6 deviation                       이탈 점수 D (과거 샷 기준, lot 재설정, 20샷 연속 경보 재설정)
  S7 fit_calibration                 보정층 (제품별 설정: CN7 공정 / RG3 캐비티 축소 추정)
  S8 evaluate_loso                   구간 하나 빼고 검증 → Brier, AUC, 보정표
  S9 significance                    원형 이동 검정, 짝지은 비교(맥니마), 사건 단위 부트스트랩

실행:  python pinnset_risk_v0.py --data /경로/dataset
필요:  numpy, pandas, scipy, scikit-learn  (난수 시드 고정 → 같은 결과)
"""
import argparse, json, warnings
import numpy as np, pandas as pd
from scipy.special import erfinv
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, brier_score_loss
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # Windows(CP949) 콘솔에서도 기호 출력 오류가 나지 않게
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
warnings.filterwarnings("ignore")

# ============================================================ 파라미터 (전부 여기서만 바꾼다)
CFG = dict(
    EQUIP="650톤-우진2호기",
    PRODUCTS=["CN7", "RG3"],
    # 물성 (ASA 문헌 대표값) — FLR 계산용
    ALPHA=0.080, T_NF=110.0, HALF_THICK=4.37 / 2,
    # 변수 묶음
    VARS=["Max_Injection_Speed", "Filling_Time", "Injection_Time", "FLR"],          # 이탈 점수: 원인 A·B 변수
    GATE_VARS=["Tm", "Tw", "Hopper_Temperature", "Plasticizing_Position"],          # 문: 샷마다 안 변하는 설정
    TIER_VARS=["Tm", "Tw", "Cycle_Time", "Plasticizing_Position", "Average_Back_Pressure"],  # 1층: lot 레시피 지문
    # 기준값
    SEG_GAP_S=300,          # 연속 가동 구간을 나누는 정지 시간(초)
    GATE_DIST=3.0,          # 문: 설정 거리 기준
    TIER_DIST=3.0,          # 1층: 지문 거리 기준
    TIER_MIN_ROWS=30,       # 1층: lot 최소 행 수
    REF_MIN=5,              # 이탈 점수 계산에 필요한 최소 기준 샷
    REF_WIN=200,            # 기준 중앙값 창(샷)
    ALARM=5.0,              # 경보 기준 (이탈 점수)
    REBASE_RUN=20,          # 연속 경보가 이만큼이면 기준 재설정
    EVENT_GAP_MIN=30,       # 불량 사건을 나누는 시간(분)
    # 보정층: 제품별 성분 (유의성 검정 결과로 결정)
    MODEL={"CN7": dict(process=True, cavity="none"),
           "RG3": dict(process=False, cavity="shrink")},
    SHRINK_TAU=0.7,         # 캐비티 로그 위험비 사전분포 N(0, τ²)
    N_SHIFT=3000, N_BOOT=5000, SEED=0,
)

# ============================================================ S1 데이터
BZ = ["Barrel_Temperature_1", "Barrel_Temperature_2", "Barrel_Temperature_3"]

def add_derived(df):
    df = df.copy()
    df["Tm"] = df[BZ].mean(axis=1)                                            # 노즐측 용융온도
    df["Tw"] = df[["Mold_Temperature_3", "Mold_Temperature_4"]].mean(axis=1)  # 금형온도
    r = np.clip((CFG["T_NF"] - df.Tw) / (df.Tm - df.Tw), 1e-9, 1 - 1e-9)
    df["FLR"] = 2 * np.sqrt(CFG["ALPHA"] * df.Filling_Time.clip(lower=1e-3)) * erfinv(r) / CFG["HALF_THICK"]
    return df

def load_labeled(path):
    d = pd.read_csv(f"{path}/labeled_data.csv", parse_dates=["TimeStamp"])
    d = d.drop_duplicates(["TimeStamp", "PART_FACT_SERIAL", "PART_NAME"])      # 10-27 이후 이중 삽입 제거
    d["y"] = (d.PassOrFail == "N").astype(int)
    d["RH"] = d.PART_NAME.str.endswith("RH").astype(int)
    return add_derived(d)

def load_unlabeled(path):
    cols = ["TimeStamp", "PART_NAME", "EQUIP_NAME", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "ERR_FACT_QTY",
            "Cycle_Time", "Hopper_Temperature", "Plasticizing_Position", "Average_Back_Pressure",
            "Max_Injection_Speed", "Filling_Time", "Injection_Time", "Mold_Temperature_3", "Mold_Temperature_4"] + BZ
    return add_derived(pd.read_csv(f"{path}/unlabeled_data.csv", usecols=cols, parse_dates=["TimeStamp"]))

# ============================================================ S2 샷 표
def shot_table(parts):
    g = parts.groupby("TimeStamp")
    keep = list(dict.fromkeys(CFG["VARS"] + CFG["GATE_VARS"] + CFG["TIER_VARS"]))
    S = g[keep].first()
    S["lot"] = g.PART_FACT_PLAN_DATE.first().astype(str) + "_" + g.PART_FACT_SERIAL.min().astype(str)
    if "y" in parts:
        S["bad"] = g.y.max()
    S = S.sort_index().reset_index()
    S["seg"] = (S.TimeStamp.diff().dt.total_seconds() > CFG["SEG_GAP_S"]).cumsum()
    return S

# ============================================================ S3 기준값
def reference(S, cols):
    n = S[S.bad == 0]
    med = n[cols].median()
    sc = np.maximum(1.4826 * (n[cols] - med).abs().median(), np.maximum(0.01 * med.abs(), 0.5))
    return med, sc

def dist(df, med, sc):
    return np.sqrt((((df[med.index] - med) / sc) ** 2).mean(axis=1))

# ============================================================ S4 무라벨 1층
def select_tier1(U, prod, med, sc):
    q = U[(U.EQUIP_NAME == CFG["EQUIP"]) & U.PART_NAME.str.startswith(prod)]
    key = ["PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "PART_NAME"]
    lots = q.groupby(key)[CFG["TIER_VARS"]].median()
    lots["rows"] = q.groupby(key).size()
    lots = lots[lots.rows >= CFG["TIER_MIN_ROWS"]]
    lots["d"] = dist(lots, med, sc)
    ok = lots.index[lots.d <= CFG["TIER_DIST"]]
    return q.set_index(key).loc[ok].reset_index(), lots

# ============================================================ S5 적용 범위 문
def recipe_gate(S, med, sc):
    return (dist(S, med, sc) > CFG["GATE_DIST"]).values

# ============================================================ S6 이탈 점수
def dev_scale(s):
    v = np.unique(s.round(4)); q = np.diff(v).min() if len(v) > 1 else 1e-3
    return max(1.4826 * np.median(np.abs(s - s.median())), q, 1e-3)       # 측정 눈금보다 작아지지 않게

def deviation(S, scale, held):
    V = CFG["VARS"]; X = S[V].values; lot = S.lot.values
    ref = {c: [] for c in V}; D = np.full(len(S), np.nan); prev = None; run = []
    for i in range(len(S)):
        if lot[i] != prev:                                   # 작업오더가 바뀌면 기준 재설정
            ref = {c: [] for c in V}; run = []
        prev = lot[i]
        if held[i]:
            continue
        if len(ref[V[0]]) < CFG["REF_MIN"]:                  # 기준 준비 중 → D 없음(보류)
            for j, c in enumerate(V): ref[c].append(X[i, j])
            continue
        z = [(X[i, j] - np.median(ref[c][-CFG["REF_WIN"]:])) / scale[c] for j, c in enumerate(V)]
        D[i] = max(map(abs, z))
        if D[i] < CFG["ALARM"]:                              # 경보가 아니면 기준에 추가
            run = []
            for j, c in enumerate(V): ref[c].append(X[i, j])
        else:
            run.append(i)
            if len(run) >= CFG["REBASE_RUN"]:                # 긴 연속 경보 = 설정 변경으로 보고 재설정
                ref = {c: list(X[run, j]) for j, c in enumerate(V)}; run = []
    return D

# ============================================================ S7 보정층
def irls(X, y, off, it=60):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ b + off))); W = p * (1 - p) + 1e-9
        b = b + np.linalg.solve(X.T @ (W[:, None] * X) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    return b

def shrink_log_ratio(pair):
    """짝지은 비교(같은 샷 LH/RH)에서 로그 위험비를 추정하고 사건 뭉침만큼 분산을 키워 0 쪽으로 축소"""
    k = int(((pair.RH == 1) & (pair.LH == 0)).sum()); m = int(((pair.LH == 1) & (pair.RH == 0)).sum())
    one = pair[(pair.RH + pair.LH) == 1]
    nev = int((one.index.to_series().diff() > pd.Timedelta(minutes=CFG["EVENT_GAP_MIN"])).sum() + 1) if len(one) else 1
    L = np.log((k + 0.5) / (m + 0.5)); v = (1 / (k + 0.5) + 1 / (m + 0.5)) * max(1, (k + m) / nev)
    return L * CFG["SHRINK_TAU"] ** 2 / (CFG["SHRINK_TAU"] ** 2 + v)

def design(P, spec):
    X = np.ones((len(P), 1))
    if spec["process"]:
        X = np.c_[X, P.gD.values]
    return X

def fit_calibration(P, pair, spec):
    off = np.zeros(len(P)); Lc = 0.0
    if spec["cavity"] == "shrink":
        Lc = shrink_log_ratio(pair); off = Lc * P.RH.values
    b = irls(design(P, spec), P.y.values, off)
    if spec["process"] and b[1] < 0:                       # 물리 제약: 이탈이 위험을 줄일 수 없음
        spec = dict(spec, process=False); b = irls(design(P, spec), P.y.values, off)
    return b, Lc, spec

def predict(P, b, Lc, spec):
    return 1 / (1 + np.exp(-(design(P, spec) @ b + Lc * P.RH.values)))

# ============================================================ S8 검증
# AUC 정의(2026-09-30 정정): 구간 안 centered mid-rank u=(평균순위-0.5)/n. 모든 점수가 같으면 정확히 0.5 (V07 erratum)
def evaluate_loso(P, pairs, spec):
    y = P.y.values; g = P.seg.values; oof = np.zeros(len(P))
    for s in np.unique(g):
        tr = g != s; te = g == s
        pair_tr = pairs.loc[pairs.index.isin(P.TimeStamp[tr])]
        b, Lc, sp = fit_calibration(P[tr], pair_tr, dict(spec))
        oof[te] = predict(P[te], b, Lc, sp)
    base = brier_score_loss(y, np.full(len(y), y.mean()))
    wr = pd.Series(oof).groupby(g).transform(lambda q: (q.rank() - 0.5) / len(q)).values
    order = np.argsort(-oof); top = order[: int(len(y) * 0.05)]
    bins = pd.cut(oof, [0, .003, .01, .03, .1, 1], labels=["<0.3%", "0.3~1%", "1~3%", "3~10%", ">=10%"], include_lowest=True)
    cal = pd.DataFrame(dict(p=oof, y=y, b=bins)).groupby("b", observed=True).agg(n=("y", "size"), pred=("p", "mean"), actual=("y", "mean"))
    return dict(brier=brier_score_loss(y, oof), const=base, improve=1 - brier_score_loss(y, oof) / base,
                auc_within=roc_auc_score(y, wr), top5=int(y[top].sum()), n_def=int(y.sum()), calib=cal.round(4).to_dict())

# ============================================================ S9 유의성
def circular_shift_test(P, rng):
    sh = P.drop_duplicates("TimeStamp")[["TimeStamp", "seg", "D"]].reset_index(drop=True)
    idx = P.TimeStamp.map(pd.Series(np.arange(len(sh)), index=sh.TimeStamp)).values
    y = P.y.values; g = P.seg.values
    auc = lambda s: roc_auc_score(y, pd.Series(s).groupby(g).transform(lambda q: (q.rank() - 0.5) / len(q)).values)
    obs = auc(P.D.values); null = []
    groups = [np.array(list(v)) for v in sh.groupby("seg").groups.values()]
    for _ in range(CFG["N_SHIFT"]):
        Ds = sh.D.values.copy()
        for gi in groups:
            if len(gi) > 1: Ds[gi] = np.roll(Ds[gi], rng.integers(1, len(gi)))
        null.append(auc(Ds[idx]))
    null = np.array(null)
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

# ============================================================ 실행
def run(path):
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
        pairs = parts.pivot_table(index="TimeStamp", columns=parts.PART_NAME.str[-2:], values="y", aggfunc="max").dropna()
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

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="dataset"); a = ap.parse_args()
    run(a.data)
