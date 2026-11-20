"""PINNSET 새 레시피·새 수지 적용성 검증 (E13, 별도 실행) — 2026-10-07

    python run_newrecipe.py --data <KAMP 데이터 폴더> --out results            # 전체 (약 60분)
    python run_newrecipe.py --data <KAMP 데이터 폴더> --out results --fast     # 엄격 평가는 공식·자동+PINN 2가지만 (약 30분)

질문: 새 레시피(또는 새 설비·품목)가 들어오면, 라벨 없이 적은 공정 변수만으로 위험점수 모델이 자동으로 준비되는가?
     수지 물성이 바뀌면 PINN이 FLR을 직접 계산해 쓸 수 있는가?

① 라벨 없는 레시피 자동 등록 (auto)
   공식 모델은 라벨 데이터의 정상 샷으로 '지금 레시피'의 지문을 잡는다. 자동 모드는 라벨을 전혀 쓰지 않고, 무라벨 작업오더의
   공정 설정 지문(용융·금형온도, 사이클, 계량 위치, 배압)을 서로 비교해 레시피 묶음을 스스로 찾는다(같은 거리·같은 기준 3.0).
   묶음마다 적용 범위 기준과 이탈 척도를 따로 만들고, 샷은 설정이 가장 가까운 레시피로 보낸다. 어느 레시피와도 멀면 보류.
② PINN FLR 직접 사용 (pinn)
   위험점수 모델의 FLR을 반무한·물성 일정 해석식 대신 학습된 PINN(실험 2, 온도 의존 열확산율)으로 샷마다 계산한다(이분법).
   PINN 학습 범위(θg 0.10~0.40, 시간 0.12~38.8초) 밖 행은 해석식을 그대로 쓴다.
③ 엄격 평가 비교: evaluate.evaluate_product를 그대로 쓰고 기준 설정(fit_reference)과 FLR 계산만 바꾼다
   (official / auto / pinn / pinn_auto). 동결 파라미터·보정층·평가 방식은 그대로.
④ 무라벨 전체 설비·품목: 앞 절반 기간으로 레시피를 등록하고 뒤 절반을 샷 순서대로 판정(frozen), 또는 매일 0시에
   그때까지의 기록으로 다시 등록(daily). 판정률·처음 보는 레시피 보류율·정상 운전 경보율·재가동 직후 경보율.
   두께·수지를 모르는 품목은 FLR을 빼고 3변수로 계산(라벨 부품 CN7·RG3 + 라벨 설비만 FLR 포함).
⑤ 새 수지 무차원 변환: 학습된 PINN을 재학습 없이 다른 두께·용융 열확산율·Tg·무유동 온도에 쓰고 실단위 FDM과 대조.
결과: <out>/newrecipe/newrecipe_results.json, 표 CSV. 모델 코드·가중치·기존 결과는 바꾸지 않는다.
"""
from pathlib import Path
import argparse, json, subprocess, sys, time

import numpy as np
import pandas as pd

from .config import CFG, override, HALF_THICK, ALPHA, T_GLASS, T_NF
from .data import load_labeled, load_unlabeled, shot_table
from .model import dev_scale, deviation, RiskModel
from .pinn import PINN, WEIGHTS, T0, T1, LG
from . import evaluate as st

RUNNER = Path(__file__).resolve().parent.parent / "run_newrecipe.py"
TV, GATE = CFG["TIER_VARS"], CFG["GATE_VARS"]
LOTKEY = ["PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "PART_NAME"]
FULL = ["Max_Injection_Speed", "Filling_Time", "Injection_Time", "FLR"]


# ================================================================ ① 라벨 없는 레시피 자동 등록
def width(R, med, cols):
    return np.maximum(1.4826 * (R[cols] - med).abs().median(), np.maximum(0.01 * med.abs(), 0.5))


def build_bank(U, min_rows=300):
    """U: 같은 설비·품목의 무라벨 행. 반환: 레시피 목록 (큰 묶음부터). 라벨 열은 쓰지 않는다"""
    if U.empty: return []
    groups = {k: g for k, g in U.groupby(LOTKEY)}
    lots = pd.DataFrame({k: g[TV].median() for k, g in groups.items()}).T
    lots["rows"] = pd.Series({k: len(g) for k, g in groups.items()})
    lots = lots[lots.rows >= CFG["TIER_MIN_ROWS"]]
    remaining = list(lots.index); bank = []
    while remaining:
        L = lots.loc[remaining]; best = None
        for s in remaining:                                   # 씨앗 작업오더: 자기 지문·폭으로 거리 3 이내 작업오더를 모음
            fp = L.loc[[s], TV].iloc[0]; sc = width(groups[s], fp, TV)
            d = np.sqrt((((L[TV] - fp) / sc) ** 2).mean(axis=1)); mem = list(d.index[d <= CFG["TIER_DIST"]])
            w = int(L.loc[mem, "rows"].sum())
            if best is None or w > best[0]: best = (w, mem)
        if best[0] < min_rows: break                          # 300행(약 한 교대분) 미만 묶음은 아직 등록하지 않음
        R = pd.concat([groups[k] for k in best[1]])
        med = R[TV].median(); sc = width(R, med, TV)            # 묶음 전체로 지문을 다시 잡고 1층을 다시 고름 (select_tier1과 같은 거리)
        d = np.sqrt((((L[TV] - med) / sc) ** 2).mean(axis=1)); mem = list(d.index[d <= CFG["TIER_DIST"]]) or best[1]
        T1 = pd.concat([groups[k] for k in mem]); ST = shot_table(T1)
        G = ST[GATE]; gmed = G.median()
        gsc = np.maximum(1.4826 * (G - gmed).abs().median(), np.maximum(0.01 * gmed.abs(), 0.5))
        bank.append(dict(tier_med=med, tier_sc=sc, gmed=gmed, gsc=gsc, scale={c: dev_scale(ST[c]) for c in CFG["VARS"]},
                         lots=len(mem), rows=len(T1), shots=len(ST)))
        remaining = [k for k in remaining if k not in set(mem) | set(best[1])]
    return bank


def route(S, bank):
    G = S[GATE].values.astype(float)
    d = np.stack([np.sqrt((((G - b["gmed"].values) / b["gsc"].values) ** 2).mean(1)) for b in bank], 1)
    r = d.argmin(1); return r, d[np.arange(len(S)), r]


def effective_recipe(r, held):
    """보류된 샷(어느 레시피와도 먼 샷)은 레시피를 바꾸지 않는다: 직전 판정 레시피를 이어 쓴다.
    그래야 다른 레시피 샷이 한두 개 끼어들어도 지금 레시피의 기준이 처음부터 다시 쌓이지 않는다(공식 모델과 같은 동작)."""
    r = np.asarray(r).copy(); cur = None
    for i in range(len(r)):
        if held[i]:
            if cur is not None: r[i] = cur
        else: cur = r[i]
    return r


def multi_deviation(S, bank):
    """model.deviation과 같은 계산. 다른 점: 샷마다 가장 가까운 레시피의 척도를 쓰고, 레시피가 바뀌면 작업오더가 바뀐 것처럼 기준을 새로 잡는다"""
    r, gd = route(S, bank); held = gd > CFG["GATE_DIST"]
    r = effective_recipe(r, held)
    lot = (S.lot.astype(str) + "|" + pd.Series(r, index=S.index).astype(str)).values
    return deviation(S, [bank[k]["scale"] for k in r], held, lots=lot), r, gd


class MultiModel(RiskModel):
    """실시간 경로: RiskModel.step을 그대로 쓰되, 샷마다 가장 가까운 레시피의 기준으로 바꿔 끼운다"""
    def __init__(self, prod, bank):
        super().__init__(prod); self.bank = bank; self._use(0); self.cur = None
    def reset_stream(self):
        super().reset_stream(); self.cur = None
    def _use(self, k):
        b = self.bank[k]; self.gmed, self.gsc, self.scale = b["gmed"], b["gsc"], b["scale"]
    def step(self, shot, parts):
        g = np.array([shot[c] for c in GATE], float)
        d = [np.sqrt(np.mean(((g - b["gmed"].values) / b["gsc"].values) ** 2)) for b in self.bank]; k = int(np.argmin(d))
        if d[k] > CFG["GATE_DIST"] and self.cur is not None: k = self.cur        # 보류될 샷은 레시피를 바꾸지 않음 (effective_recipe와 같은 규칙)
        elif d[k] <= CFG["GATE_DIST"]: self.cur = k
        self._use(k)
        sh = dict(shot); sh["lot"] = f"{shot['lot']}|{k}"
        return super().step(sh, parts)


# ================================================================ ② PINN FLR 직접 사용
class PinnFLR:
    """학습된 PINN으로 샷마다 FLR: θ(x̂, t) = θ_nf인 x̂를 이분법으로 찾는다.
    수지·두께가 바뀌면 tg·tnf·half_mm·a_melt만 바꾼다(무차원 변환). 실험 4 가중치(입력 4개)는 ratio(α_유리/α_용융)도 넣는다"""
    def __init__(self, path=WEIGHTS / "pinn_exp2_map.npz", tg=T_GLASS, tnf=T_NF, half_mm=HALF_THICK, a_melt=ALPHA, ratio=None):
        self.m = PINN(str(path)); self.tg, self.tnf, self.tref, self.ratio = tg, tnf, half_mm ** 2 / a_melt, ratio
        self.T0, self.T1, self.LG = T0, T1, LG
    def _theta(self, x, th, g):
        f = [2 * x - 1, 2 * np.log(th / self.T0) / self.LG - 1, 2 * (g - 0.10) / 0.30 - 1]
        if self.m.n_in == 4: f.append(np.full_like(x, 2 * (self.ratio - 1.0) - 1))
        h = np.stack(f, 1)
        for W, b in zip(self.m.W, self.m.b): h = np.tanh(h @ W + b)
        return (h @ self.m.Wo + self.m.bo).ravel()
    def valid(self, Tw, Tm, tfill):
        g = (self.tg - Tw) / (Tm - Tw); th = tfill / self.tref
        return (g >= 0.10) & (g <= 0.40) & (th >= self.T0) & (th <= self.T1) & (Tm > self.tnf) & (Tw < self.tnf)
    def __call__(self, Tw, Tm, tfill, iters=45):
        Tw, Tm, tfill = (np.asarray(a, float).ravel() for a in (Tw, Tm, tfill))
        with np.errstate(divide="ignore", invalid="ignore"):
            g = (self.tg - Tw) / (Tm - Tw); nf = (self.tnf - Tw) / (Tm - Tw); th = tfill / self.tref
            ok = self.valid(Tw, Tm, tfill)
        out = np.full(len(Tw), np.nan)
        if not ok.any(): return out
        g, nf, th = g[ok], nf[ok], th[ok]
        full = self._theta(np.ones_like(g), th, g) < nf              # 중심까지 θ_nf 아래 = 완전 고화
        lo, hi = np.zeros_like(g), np.ones_like(g)
        for _ in range(iters):
            mid = (lo + hi) / 2; up = self._theta(mid, th, g) >= nf
            hi = np.where(up, mid, hi); lo = np.where(up, lo, mid)
        r = (lo + hi) / 2; r[full] = 1.0; out[ok] = r; return out


# ================================================================ ③ 엄격 평가 비교 (변형 하나를 별도 프로세스로)
def auto_reference(prod, trainS, U):
    """auto: 라벨(trainS)은 쓰지 않고 무라벨 작업오더로 레시피 묶음을 등록한다"""
    bank = build_bank(U)
    if not bank: return None, dict(reason="no recipe cluster")
    return MultiModel(prod, bank), dict(recipes=len(bank), recipe_Tm=[round(float(b["tier_med"]["Tm"]), 1) for b in bank])


def auto_deviations(S, model, no_flr=False):
    VARS = st.VARS
    with override(VARS=[v for v in VARS if v != "FLR"] if no_flr else VARS):
        return multi_deviation(S, model.bank)[0]


def strict_variant(data, out, tag):
    D = Path(data); O = Path(out) / "newrecipe" / tag; O.mkdir(parents=True, exist_ok=True)
    info = {}
    L, U = load_labeled(str(D)), load_unlabeled(str(D)); R = dict(info=info)
    if "pinn" in tag:                                        # FLR을 해석식 대신 PINN으로 (학습 범위 밖 행은 해석식 유지)
        PF = PinnFLR()
        def pinn_flr(df):
            df = df.copy(); f = PF(df.Tw.values, df.Tm.values, df.Filling_Time.clip(lower=1e-3).values)
            ok = np.isfinite(f); info.setdefault("PINN_FLR_행", []).append([int(ok.sum()), len(df)])
            df.loc[ok, "FLR"] = f[ok]; return df
        L, U = pinn_flr(L), pinn_flr(U)
    fit_ref, devs = (auto_reference, auto_deviations) if "auto" in tag else (None, None)
    for prod in CFG["PRODUCTS"]:
        parts = L[L.PART_NAME.str.startswith(prod)].copy()
        for mode in ["forward", "leave_block_out"]:
            E, folds, pa = st.evaluate_product(prod, parts, U, mode, O, fit_ref, devs, log=lambda s: None)
            rows, ops, evs, cis = st.performance(E, prod, mode)
            pin = [r for r in rows if r["model"] == "PINNSET"][0]; op = {o["population"]: o for o in ops}
            sc = E.status.eq("score")
            R[f"{prod}_{mode}"] = dict(status=E.status.value_counts().to_dict(), budget=pin["budget"], TP=pin["TP"], defects=pin["defects"],
                                       AUC_within_segment=pin["AUC_within_segment"], Brier_improvement=pin.get("Brier_improvement"),
                                       ops_after_calibration=op["after_calibration_available"], ops_all=op["all_parts"],
                                       alarms=int((sc & E.D.ge(CFG["ALARM"])).sum()), alarm_defects=int((sc & E.D.ge(CFG["ALARM"]) & E.y.eq(1)).sum()),
                                       recipes_per_fold=[f.get("recipes") for f in folds] if "auto" in tag else None)
    json.dump(R, open(O / "summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)


# ================================================================ ④ 무라벨 전체 설비·품목
def onboarding(data, mode="frozen", min_shots=2000):
    U = load_unlabeled(str(data)); rows = []
    for (eq, item), u in U.groupby(["EQUIP_NAME", "PART_NAME"]):
        u = u.drop_duplicates(["TimeStamp", "PART_NAME"]).sort_values("TimeStamp")
        if len(u) < min_shots: continue
        cut = u.TimeStamp.quantile(0.5); tr, te = u[u.TimeStamp < cut], u[u.TimeStamp >= cut]
        lab = item.startswith(tuple(CFG["PRODUCTS"])) and eq == CFG["EQUIP"]
        with override(VARS=FULL if lab else FULL[:3]):     # 두께·수지를 모르는 품목은 FLR 빼고 3변수
            S = shot_table(te)
            if mode == "daily":                                # 매일 0시에 그때까지의 기록으로 다시 등록
                parts, nb = [], []
                for day, s_ in S.groupby(S.TimeStamp.dt.normalize(), sort=True):
                    b_ = build_bank(u[u.TimeStamp < day]); nb.append(len(b_))
                    if not b_: parts.append((np.full(len(s_), np.nan), np.full(len(s_), -1), np.full(len(s_), 99.))); continue
                    d_, r_, g_ = multi_deviation(s_, b_); parts.append((d_, r_ + 1000 * len(parts), g_))
                D = np.concatenate([p[0] for p in parts]); r = np.concatenate([p[1] for p in parts]); gd = np.concatenate([p[2] for p in parts]); nrec = max(nb)
            else:
                bank = build_bank(tr); nrec = len(bank)
                if not bank: rows.append(dict(설비=eq, 품목=item, 레시피=0, 시험샷=len(S))); continue
                D, r, gd = multi_deviation(S, bank)
            gap = S.TimeStamp.diff().dt.total_seconds().fillna(1e9).values
            lotk = (S.lot.astype(str) + "|" + pd.Series(r, index=S.index).astype(str)).values
            newrun = (gap > CFG["SEG_GAP_S"]) | (lotk != np.r_[["_"], lotk[:-1]])
            since = pd.Series(1, index=S.index).groupby(np.cumsum(newrun)).cumsum().values
            sc = np.isfinite(D); steady = sc & (since > 20); first = sc & (since <= 3)
            rows.append(dict(설비=eq, 품목=item, 라벨부품=bool(lab), 변수=len(CFG["VARS"]), 레시피=int(nrec), 시험샷=len(S), 판정률=float(sc.mean()),
                             처음보는레시피_보류=float((gd > CFG["GATE_DIST"]).mean()),
                             정상운전_경보율=float((D[steady] >= CFG["ALARM"]).mean()) if steady.any() else np.nan,
                             재가동3샷_경보율=float((D[first] >= CFG["ALARM"]).mean()) if first.any() else np.nan,
                             정상운전샷=int(steady.sum()), 재가동샷=int(first.sum())))
    T = pd.DataFrame(rows); ok = T.dropna(subset=["정상운전_경보율"])
    summ = dict(방식=mode, 품목=len(T), 평가품목=len(ok), 설비=int(ok.설비.nunique()), 등록레시피합=int(ok.레시피.sum()),
                판정률_중앙=float(ok.판정률.median()), 판정률_샷가중=float(np.average(ok.판정률, weights=ok.시험샷)),
                정상경보율_중앙=float(ok.정상운전_경보율.median()), 정상경보율_샷가중=float(np.average(ok.정상운전_경보율, weights=ok.정상운전샷)),
                재가동경보율_샷가중=float(np.average(ok.재가동3샷_경보율.fillna(0), weights=ok.재가동샷)),
                정상경보율_1퍼센트초과_품목=int((ok.정상운전_경보율 > 0.01).sum()))
    return T, summ


# ================================================================ ⑤ 새 수지: 무차원 변환만으로 PINN 사용 → 실단위 FDM 대조
def new_resin_check():
    from scipy.integrate import solve_ivp
    from scipy.special import erf, erfinv
    RATIO, W = 0.115 / 0.080, 0.03
    def fdm(h, a_m, Tg, Tnf, Tm, Tw, tf, n=600):
        x = np.linspace(0, h, n + 1); dx = x[1] - x[0]; span = Tm - Tw; t0 = 0.002 * h ** 2 / a_m
        T0 = Tw + span * erf(x / (2 * np.sqrt(a_m * t0)))
        def f(t, T):
            T = T.copy(); T[0] = Tw; lap = np.empty_like(T)
            lap[1:-1] = (T[2:] - 2 * T[1:-1] + T[:-2]) / dx ** 2; lap[-1] = 2 * (T[-2] - T[-1]) / dx ** 2; lap[0] = 0
            s = 1 / (1 + np.exp(-((T - Tg) / span) / W)); return a_m * (RATIO + (1 - RATIO) * s) * lap
        T = solve_ivp(f, (t0, tf), T0, method="BDF", t_eval=[tf], rtol=1e-8, atol=1e-8).y[:, 0]; i = np.argmax(T >= Tnf)
        return 1.0 if not (T >= Tnf).any() else float(np.interp(Tnf, [T[i - 1], T[i]], [x[i - 1], x[i]]) / h)
    cases = [("본 과제 ASA (기준)", 2.185, 0.080, 105, 110, 275.5, 22.8, 4.45), ("가상 수지 A: 얇은 부품", 1.0, 0.080, 105, 110, 260, 40, 1.2),
             ("가상 수지 B: PC 유사 (Tg 145)", 1.5, 0.100, 145, 150, 300, 85, 2.0), ("가상 수지 C: ABS 유사 (Tg 100)", 2.0, 0.090, 100, 105, 240, 50, 3.0),
             ("가상 수지 D: 두꺼운 부품", 3.0, 0.070, 110, 115, 280, 60, 10.0)]
    rows = []
    for name, h, a, Tg, Tnf, Tm, Tw, tf in cases:
        p = float(PinnFLR(tg=Tg, tnf=Tnf, half_mm=h, a_melt=a)(Tw, Tm, tf)[0]); q = fdm(h, a, Tg, Tnf, Tm, Tw, tf)
        an = float(2 * np.sqrt(a * tf) * erfinv((Tnf - Tw) / (Tm - Tw)) / h)
        rows.append(dict(조건=name, 반두께_mm=h, 용융열확산율=a, Tg=Tg, 무유동온도=Tnf, 용융=Tm, 금형=Tw, 충진초=tf,
                         PINN=round(p, 4), FDM=round(q, 4), PINN차이=round(p - q, 4), 해석식=round(an, 4), 해석식차이=round(an - q, 4)))
    return rows


# ================================================================ 실행
def main():
    """run_newrecipe.py에서 호출. --variant는 내부용(엄격 평가 변형 하나를 별도 프로세스로)"""
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", default="results")
    ap.add_argument("--fast", action="store_true", help="엄격 평가는 official, pinn_auto 두 가지만")
    ap.add_argument("--variant", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.variant: strict_variant(a.data, a.out, a.variant); return
    O = Path(a.out) / "newrecipe"; O.mkdir(parents=True, exist_ok=True); t0 = time.time(); R = {}
    print("[1/4] 새 수지 무차원 변환 검산 (PINN vs 실단위 FDM)", flush=True); R["새수지_무차원변환"] = new_resin_check()
    tags = ["official", "pinn_auto"] if a.fast else ["official", "auto", "pinn", "pinn_auto"]
    print(f"[2/4] 엄격 평가 비교 {tags} (변형마다 별도 프로세스)", flush=True)
    for i in range(0, len(tags), 2):                       # 메모리를 아끼려고 두 개씩
        procs = [subprocess.Popen([sys.executable, str(RUNNER), "--data", a.data, "--out", a.out, "--variant", t]) for t in tags[i:i + 2]]
        for p in procs:
            if p.wait() != 0: raise SystemExit(f"엄격 평가 변형 실패: {p.args[-1]}")
    R["엄격평가"] = {t: json.load(open(O / t / "summary.json", encoding="utf-8")) for t in tags}
    print("[3/4] 무라벨 전체 설비·품목 (앞 절반 등록 → 뒤 절반 판정)", flush=True)
    T, R["무라벨_고정등록"] = onboarding(a.data, "frozen"); T.to_csv(O / "onboarding_frozen.csv", index=False, encoding="utf-8-sig")
    print("[4/4] 무라벨 전체 설비·품목 (매일 다시 등록)", flush=True)
    T, R["무라벨_매일등록"] = onboarding(a.data, "daily"); T.to_csv(O / "onboarding_daily.csv", index=False, encoding="utf-8-sig")
    R["소요초"] = time.time() - t0
    json.dump(R, open(O / "newrecipe_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
    print(f"완료 {R['소요초']:.0f}초 → {O / 'newrecipe_results.json'}")
