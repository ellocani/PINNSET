"""
[5] 데이터 요청 엔진 v2 (층 5) (2026-09-30)
=================================================
v1 프로토타입(request_engine.py)의 결함 5가지를 고친 버전.
  1. 측정오차를 관측 채널마다 따로 넣는다 (계측기 사양, 모르면 범위로 스윕)
  2. 점수 하나 대신 4단 판정: ① 식별 ② 엉킴 ③ 결정 가능 ④ 비용·기간
  3. 임의 상수(6자릿수, 비용 1·2·4 나누기) 제거 → ①~③ 통과 후보를 ④로 정렬
  4. B는 정정된 관측(60.5 양품 / 60.7 불량, 다른 날 조건)으로 "조건 고정 실험"을 설계
  5. 모르는 값의 가정치를 격자로 훑어 통과 비율(강건성)을 낸다
PINN 연결: (단계 1) A의 정상 고화층 비율 FLR0, (단계 2) A의 관측 함수 전체를 학습된 PINN 지도(weights/pinn_exp2_map.npz)로 계산한다.

[사전 고정] 아래 DECISIONS, SIGMA, 격자는 계산 결과를 보기 전에 정했다(2026-09-30). 결과를 보고 바꾸지 않는다.
"""
import json
import numpy as np, pandas as pd
from scipy.stats import norm
from .config import T_MOLD_REF, FILL_REF, T_MELT_REF
from .pinn import PINN, WEIGHTS
from .stats import json_safe

# ============================================================ 사전 고정: 결정 질문과 판정 기준
DECISIONS = {
    "C": dict(질문="어느 캐비티의 벤트·러너를 먼저 고칠 것인가",
              기준="저항비 R_RH/R_LH > 1 을 양측 95%로 판정할 검정력 ≥ 80%"),
    "A": dict(질문="재가동 대처를 수지 온도(퍼지·예열) 쪽으로 할지, 고화층(금형) 쪽으로 할지",
              기준="수지 온도 부족 ΔT와 추가 고화층 Δδ의 엉킴 |ρ| < 0.9 이고 ΔT 상대 불확실성 < 25%"),
    "B": dict(질문="설비 사출속도 상한을 얼마로 걸 것인가",
              기준="같은 조건에서 임계속도 구간 폭 ≤ 1 mm/s, 벤트 청소 전후 각각 확보"),
    "C_CN7통계": dict(질문="CN7에서도 RH가 불리하다고 말할 수 있는가",
                    기준="한쪽만 불량인 샷 쌍에서 RH 비중 검정 (위험비 1.8배 가정, 양측 α 0.05, 검정력 80%)"),
}
RHO_MAX, REL_SE_MAX, POWER_MIN, ROBUST_MIN = 0.9, 0.25, 0.80, 0.80   # 엉킴 상한, ΔT 불확실성 상한, 검정력 하한, 격자 통과 비율 하한

# 관측 채널별 측정오차 (상대값). 사양을 모르는 채널은 범위로 스윕한다.
SIGMA = {
    "총주입량": 0.01,                       # 사출속도×충진시간 또는 쿠션 기반 총량 (1%)
    "중량비": [0.002, 0.005, 0.01],          # 캐비티 중량비: 0.01 g 저울 기준 0.2~1% (부품 무게 미상이라 범위)
    "캐비티압력": [0.005, 0.01, 0.02],       # 캐비티 압력 센서 (0.5~2%)
    "유동저항비": 0.01,                     # 압력/속도로 구한 겉보기 유동저항 (1%)
    "노즐온도K": [0.5, 1.0, 2.0],            # 노즐 수지 온도 센서 (K, 절대값)
}
# [결과를 본 뒤 수정, 2026-09-30] 사전 설계는 ② 엉킴 기준을 모든 원인에 일괄 적용했다.
# 실행해 보니 C(결정 대상이 '저항비'라는 조합 하나)에서 두 저항의 엉킴은 결정과 무관했다(비율의 불확실성은 ③에서 공분산으로 이미 반영).
# 그래서 ②는 결정이 '두 값을 따로 가르는 것'인 원인(A)에만 적용하는 '수정 판정'을 추가한다. 사전 판정도 함께 보고한다.
RHO_APPLIES = {"C": False, "A": True}
TIER_ORDER = {"A": 0, "B": 1, "C": 2}

# ============================================================ 공통 계산

def jacobian(f, th, rel=1e-6):
    th = np.asarray(th, float); y0 = f(th); J = np.zeros((len(y0), len(th)))
    for j in range(len(th)):
        h = rel * max(abs(th[j]), 1e-9); tp, tm = th.copy(), th.copy(); tp[j] += h; tm[j] -= h
        J[:, j] = (f(tp) - f(tm)) / (2 * h)
    return J

def analyze(f, th, sig):
    """관측 함수 f, 가정치 th, 관측별 표준편차 sig → 식별·엉킴·공분산"""
    J = jacobian(f, th) / np.asarray(sig)[:, None]
    Js = J * np.abs(th)[None, :]                                   # 파라미터를 자기 크기로 나눈 상대 단위
    s = np.linalg.svd(Js, compute_uv=False)
    rank = int(np.sum(s > 1e-9 * s.max())) if s.max() > 0 else 0
    out = dict(rank=rank, n_param=len(th), sv=s.tolist())
    if rank < len(th):
        out.update(identified=False, rho=1.0, rel_se=[np.inf] * len(th), cov=None); return out
    F = J.T @ J; C = np.linalg.inv(F); sd = np.sqrt(np.diag(C))
    rho = float(C[0, 1] / (sd[0] * sd[1])) if len(th) == 2 else 0.0
    out.update(identified=True, rho=rho, rel_se=(sd / np.abs(th)).tolist(), cov=C.tolist()); return out

# ============================================================ 원인 C: 캐비티 분배 (병렬 유로)
# 모르는 값 θ = (R_LH, R_RH). 샷마다 사출압 ΔP_i가 조금씩 다르다.
def C_cases(n_shot, ratio, s_w, s_p):
    dP = 142.0 + np.linspace(-0.3, 0.3, n_shot)                   # 샷별 사출압 (정상 흔들림 수준)
    th = np.array([1.0, ratio])
    tot = lambda t: dP / t[0] + dP / t[1]
    cases = {
        "현재 (총량만)": (tot, np.full(n_shot, SIGMA["총주입량"] * tot(th).mean()), None, 0),
        "같은 데이터 10배 (대조)": (lambda t: np.tile(tot(t), 10), np.full(10 * n_shot, SIGMA["총주입량"] * tot(th).mean()), "A", 0),
        "캐비티별 부품 중량": (lambda t: np.r_[tot(t), np.full(n_shot, t[1] / t[0])],
                         np.r_[np.full(n_shot, SIGMA["총주입량"] * tot(th).mean()), np.full(n_shot, s_w * ratio)], "A", 7),
        # [2026-10-04 정정] 이 관측식의 dP/R은 압력이 아니라 캐비티별 유량이다. 실제 압력 센서의 관측식(위치, 압력-유량 관계)은 미정이므로
        # 이 후보는 "캐비티별 유량을 직접 잴 수 있다면"이라는 이상적 상한으로만 쓰고 요청 카드의 정량 판정에서 뺀다.
        "캐비티별 유량 직접 관측 (이상적 가정)": (lambda t: np.r_[tot(t), dP / t[0], dP / t[1]],
                            np.r_[np.full(n_shot, SIGMA["총주입량"] * tot(th).mean()), s_p * dP / th[0], s_p * dP / th[1]], "C", 60),
    }
    return th, cases

def C_power(res, ratio):
    if not res["identified"]: return 0.0
    C = np.array(res["cov"]); g = np.array([-ratio / 1.0 ** 2, 1 / 1.0])  # ∂(R_RH/R_LH)/∂θ at R_LH=1
    se = float(np.sqrt(g @ C @ g)); return float(norm.cdf(abs(ratio - 1) / se - 1.96))

def run_C():
    rows = []; grid = [(r, sw, sp) for r in [1.05, 1.1, 1.2, 1.3, 1.5] for sw in SIGMA["중량비"] for sp in SIGMA["캐비티압력"]]
    for r, sw, sp in grid:
        th, cases = C_cases(30, r, sw, sp)
        for name, (f, sig, tier, lead) in cases.items():
            res = analyze(f, th, sig); pw = C_power(res, r)
            rows.append(dict(후보=name, 저항비=r, 중량오차=sw, 압력오차=sp, 식별=res["identified"], rank=res["rank"],
                             엉킴=abs(res["rho"]), 검정력=pw, tier=tier, lead=lead))
    df = pd.DataFrame(rows)
    # 30샷 기준 대표값(저항비 1.2, 중량 0.5%, 압력 1%)과 필요 샷 수
    rep = df[(df.저항비 == 1.2) & (df.중량오차 == 0.005) & (df.압력오차 == 0.01)].set_index("후보")
    need = {}
    for name in ["캐비티별 부품 중량", "캐비티별 유량 직접 관측 (이상적 가정)"]:
        for n in [1, 2, 3, 5, 10, 20, 30, 50, 100]:
            th, cases = C_cases(n, 1.05, 0.01, 0.02)                 # 가장 불리한 격자점(차이 5%, 오차 최대)
            f, sig, _, _ = cases[name]
            if C_power(analyze(f, th, sig), 1.05) >= POWER_MIN: need[name] = n; break
    return df, rep, need

# ============================================================ 원인 A: 재가동 과도응답
# 모르는 값 θ = (ΔT0: 첫 샷 수지 온도 부족 K, Δδ0: 첫 샷 추가 고화층 비율). 둘 다 시상수 τ로 회복.
TAU, K = 2.9, np.arange(6); EA, RG, T0 = 52_000, 8.314, 548.0       # ASA 겉보기 활성화에너지, 기준 용융온도(K)

def FLR0_from_pinn():
    return float(_pinn().flr(T_MOLD_REF, FILL_REF, T_MELT_REF))       # 정상 운전점 (금형 22.8℃, 충진 4.45초, 용융 275.5℃)

def A_cases(flr0, sT):
    def resist(t):
        dT, dd = t[0] * np.exp(-K / TAU), t[1] * np.exp(-K / TAU)
        return np.exp(EA / RG * (1 / (T0 - dT) - 1 / T0)) * ((1 - flr0) / (1 - flr0 - dd)) ** 3
    s_r = SIGMA["유동저항비"]
    return {
        "현재 (유동저항 6샷)": (resist, np.full(6, s_r), None, 0),
        "재가동 사건 5회 추가 (대조)": (lambda t: np.tile(resist(t), 6), np.full(36, s_r), "A", 30),
        "노즐 수지 온도 센서": (lambda t: np.r_[resist(t), t[0] * np.exp(-K / TAU)], np.r_[np.full(6, s_r), np.full(6, sT)], "C", 60),
    }

def run_A():
    flr0 = FLR0_from_pinn(); rows = []
    for dT in [5, 10, 19, 30]:
        for dd in [0.005, 0.01, 0.02, 0.04]:
            for sT in SIGMA["노즐온도K"]:
                th = np.array([dT, dd])
                for name, (f, sig, tier, lead) in A_cases(flr0, sT).items():
                    res = analyze(f, th, sig)
                    ok = res["identified"] and abs(res["rho"]) < RHO_MAX and res["rel_se"][0] < REL_SE_MAX
                    rows.append(dict(후보=name, ΔT0=dT, Δδ0=dd, 노즐오차K=sT, 식별=res["identified"], 엉킴=abs(res["rho"]),
                                     ΔT불확실성=res["rel_se"][0], Δδ불확실성=res["rel_se"][1], 결정가능=ok, tier=tier, lead=lead))
    return pd.DataFrame(rows), flr0

def run_A_need(flr0):
    """[추가 분석] 노즐 온도 센서에 더해 추가 고화층 Δδ를 직접 재는 측정(예: 금형 벽면 온도 센서 → 고화층 추정)이
    있다면, Δδ를 얼마나 정밀하게 재야 A 기준을 통과하나. 기준은 바꾸지 않고 '필요 사양'을 역으로 구한다."""
    rows = []
    for s_rel in [0.1, 0.2, 0.3, 0.5, 1.0]:
        ok = []
        for dT in [5, 10, 19, 30]:
            for dd in [0.005, 0.01, 0.02, 0.04]:
                for sT in SIGMA["노즐온도K"]:
                    th = np.array([dT, dd]); f0, sig0, _, _ = A_cases(flr0, sT)["노즐 수지 온도 센서"]
                    f = lambda t, f0=f0: np.r_[f0(t), t[1] * np.exp(-K / TAU)]
                    sig = np.r_[sig0, s_rel * dd * np.exp(-K / TAU)]
                    res = analyze(f, th, sig)
                    ok.append(res["identified"] and abs(res["rho"]) < RHO_MAX and res["rel_se"][0] < REL_SE_MAX)
        rows.append({"Δδ 측정 상대오차": s_rel, "A 기준 통과 비율": float(np.mean(ok))})
    return rows


# ============================================================ 단계 2: 원인 A를 PINN 관측 함수로 (2026-09-30, 계산 전 고정)
# 손 수식 모델은 '추가 고화층 Δδ'를 독립 미지수로 두었다. 실제로 고화층은 충진시간·금형 표면 온도·수지 온도로 정해지는 결과이므로
# PINN 지도(실험 2)로 계산한다. 그러면 미지수는 θ = (ΔT0: 수지 온도 부족 K, ΔTs0: 금형 표면 온도 부족 K)가 된다.
#   FLR_k = PINN(금형 실측 Tw_k − ΔTs_k, 충진 실측 tf_k, 용융 Tm − ΔT_k),  ΔT_k = ΔT0·e^(−k/τ), ΔTs_k = ΔTs0·e^(−k/τ)
#   관측 = 유동저항비 R_k = 점도항(ΔT_k) × ((1 − FLR0)/(1 − FLR_k))^3
# 입력(실측): 10-27 재가동 6샷 (A 불량 5샷 + 복귀 1샷)의 충진시간과 금형온도 (labeled_data, 00:56~01:07)
# 판정 기준: 사전 고정 A 기준을 그대로 사용 (|ρ| < 0.9, ΔT 상대 불확실성 < 25%). 격자·센서 오차도 계산 전 고정.
A_SHOTS = dict(tf=np.array([8.27, 5.72, 6.37, 4.87, 4.58, 4.52]), Tw=np.array([21.05, 21.20, 21.15, 21.35, 21.25, 21.30]), Tm=T_MELT_REF)
GRID_A2 = dict(ΔT0=[5, 10, 19, 30], ΔTs0=[0.5, 1, 2, 4], 센서오차K=[0.5, 1.0, 2.0])
_PINN = None

def _pinn():
    global _PINN
    if _PINN is None:
        _PINN = PINN(WEIGHTS / "pinn_exp2_map.npz")
    return _PINN

def A2_obs_resist(t, flr0):
    m = _pinn(); dT, dTs = t[0] * np.exp(-K / TAU), t[1] * np.exp(-K / TAU)
    flr = np.array([m.flr(A_SHOTS["Tw"][k] - dTs[k], A_SHOTS["tf"][k], A_SHOTS["Tm"] - dT[k]) for k in range(6)])
    visc = np.exp(EA / RG * (1 / (T0 - dT) - 1 / T0))
    return visc * ((1 - flr0) / (1 - flr)) ** 3

def A2_cases(flr0, s):
    r = lambda t: A2_obs_resist(t, flr0); sr = np.full(6, SIGMA["유동저항비"])
    nz = lambda t: t[0] * np.exp(-K / TAU); wl = lambda t: t[1] * np.exp(-K / TAU)
    return {
        "현재 (유동저항 6샷)": (r, sr, None, 0),
        "재가동 사건 5회 추가 (대조)": (lambda t: np.tile(r(t), 6), np.tile(sr, 6), "A", 30),
        "노즐 수지 온도 센서": (lambda t: np.r_[r(t), nz(t)], np.r_[sr, np.full(6, s)], "C", 60),
        "금형 표면 온도 센서": (lambda t: np.r_[r(t), wl(t)], np.r_[sr, np.full(6, s)], "C", 60),
        "노즐 + 금형 표면 센서": (lambda t: np.r_[r(t), nz(t), wl(t)], np.r_[sr, np.full(6, s), np.full(6, s)], "C", 60),
    }

def jacobian_pinn(f, th, h=(0.05, 0.05)):
    """PINN 지도 위 중심차분 (FLR이 801점 보간이라 상대 1e-6 차분은 잡음에 묻힘 → 0.05 K 고정 간격)"""
    y0 = f(th); J = np.zeros((len(y0), len(th)))
    for j in range(len(th)):
        tp, tm = th.copy(), th.copy(); tp[j] += h[j]; tm[j] -= h[j]; J[:, j] = (f(tp) - f(tm)) / (2 * h[j])
    return J

def analyze_pinn(f, th, sig):
    J = jacobian_pinn(f, th) / np.asarray(sig)[:, None]; F = J.T @ J
    s = np.linalg.svd(J * np.abs(th)[None, :], compute_uv=False); rank = int(np.sum(s > 1e-9 * s.max()))
    if rank < 2: return dict(identified=False, rho=1.0, rel_se=[np.inf, np.inf], se_abs=[np.inf, np.inf])
    C = np.linalg.inv(F); sd = np.sqrt(np.diag(C))
    return dict(identified=True, rho=float(C[0, 1] / (sd[0] * sd[1])), rel_se=(sd / np.abs(th)).tolist(), se_abs=sd.tolist())

def run_A2():
    flr0 = FLR0_from_pinn(); rows = []
    for dT in GRID_A2["ΔT0"]:
        for dTs in GRID_A2["ΔTs0"]:
            th = np.array([float(dT), float(dTs)])
            for s in GRID_A2["센서오차K"]:
                for name, (f, sig, tier, lead) in A2_cases(flr0, s).items():
                    if s != GRID_A2["센서오차K"][0] and name.startswith(("현재", "재가동")): continue   # 센서 없는 후보는 센서 오차와 무관
                    r = analyze_pinn(f, th, sig)
                    ok = r["identified"] and abs(r["rho"]) < RHO_MAX and r["rel_se"][0] < REL_SE_MAX
                    rows.append(dict(후보=name, ΔT0=dT, ΔTs0=dTs, 센서오차K=s if not name.startswith(("현재", "재가동")) else None,
                                     식별=r["identified"], 엉킴=abs(r["rho"]), ΔT불확실성=r["rel_se"][0], ΔTs불확실성K=r["se_abs"][1], 결정가능=ok, tier=tier, lead=lead))
    df = pd.DataFrame(rows)
    # 실측 FLR 재현 확인: 수지·표면 부족이 0일 때 PINN이 주는 A 샷의 FLR (충진 지연만의 효과)
    m = _pinn(); flr_obs = [m.flr(A_SHOTS["Tw"][k], A_SHOTS["tf"][k], A_SHOTS["Tm"]) for k in range(6)]
    return df, flr_obs

# ============================================================ 원인 B: 임계속도 (이진 관측 → 실험 설계)
def run_B():
    obs = dict(양품최고=60.5, 양품날="10-20 (금형 24.0℃)", 불량최저=60.7, 불량날="10-16 (금형 26.4℃)")
    return dict(관측=obs, 현재="같은 조건 관측 없음 → 구간 정의 불가 (두 관측이 다른 날·조건)",
                설계=[{"계단 폭(mm/s)": s, "단계 수(55.5→61.5)": int(np.ceil(6 / s)) + 1, "샷 수(단계당 20)": (int(np.ceil(6 / s)) + 1) * 20,
                       "구간 폭": s, "기준 충족": s <= 1.0} for s in [2.0, 1.0, 0.5]],
                비고="벤트 청소 전·후 각각 수행 → 총 샷 수 2배. 불량이 처음 나온 단계에서 중단, 시험 부품 전량 격리")

# ============================================================ 통계형: CN7 캐비티 라벨 기간
def run_C_cn7(v0_json, pred_csv):
    cav = json.load(open(v0_json, encoding="utf-8"))["CN7"]["significance"]["cavity"]; k = cav["rh_only"] + cav["lh_only"]
    t = pd.read_csv(pred_csv, parse_dates=["TimeStamp"]).TimeStamp; days = (t.max() - t.min()).total_seconds() / 86400
    p1 = 1.8 / 2.8; z = 1.96; zb = norm.ppf(POWER_MIN)
    n = int(np.ceil(((z * 0.5 + zb * np.sqrt(p1 * (1 - p1))) / (p1 - 0.5)) ** 2))
    return dict(현재_한쪽불량샷=int(k), 라벨기간_일=round(days, 1), 필요_한쪽불량샷=n,
                필요_개월=round(n / (k / days) / 30.4, 1), 가정="위험비 1.8배, 현재 발생 속도 유지")

# ============================================================ 정성 요청 (식별 계산 대상 아님: 정의·로그)
QUAL = [
    dict(요청="캐비티 압력 센서 2채널 (관측식 확정 후 재계산)", 원인="C", 푸는질문="센서 위치·압력-유량 관계를 정하면 식별성 재계산. 현재 정량 결과는 '캐비티별 유량을 직접 잰다면'의 이상적 상한일 뿐", tier="C", lead=60),
    dict(요청="사출 설정값 로그 (속도·압력·절환 설정)", 원인="A·B", 푸는질문="A에서 속도가 설정을 못 따라간 이유 (압력 147.4 < 상한 169)", tier="A", lead=7),
    dict(요청="정지 사유·가동 절차 로그", 원인="A", 푸는질문="짧은 정지(5~15분) 뒤 식은 수지 = 절차 생략인가", tier="A", lead=7),
    dict(요청="초기허용불량 판정 기준", 원인="라벨", 푸는질문="A 라벨이 물리 불량인가, 초기 구간 일괄 폐기인가", tier="A", lead=3),
    dict(요청="절환 위치 실측 기록", 원인="B·미성형", 푸는질문="V/P 전환 지점 (현재 0 또는 상수로 미기록)", tier="A", lead=7),
    dict(요청="레시피·자재 코드, 설정 변경 로그", 원인="문(Gate)", 푸는질문="다른 레시피 혼입을 직접 식별", tier="A", lead=7),
    dict(요청="ERR_FACT_QTY 정의, 오더별 생산 수량", 원인="무라벨", 푸는질문="부분 > 전체 모순 해소", tier="A", lead=3),
    dict(요청="수지 데이터시트 (허용 가공온도·점도·PVT)", 원인="A·PINN", 푸는질문="열분해 한계, 점도 모델 검증", tier="B", lead=7),
    dict(요청="검사 결과 즉시 회신 (인라인 검사)", 원인="모델", 푸는질문="쌍둥이 감시 (CN7 Brier 개선 +32% → +45%)", tier="C", lead=90),
]

# ============================================================ 실행
def run(out="."):
    from . import report  # noqa: F401  한글 폰트 등록 재사용
    import matplotlib.pyplot as plt
    dfC, repC, needC = run_C(); dfA, flr0 = run_A(); B = run_B(); needA = run_A_need(flr0); dfA2, flr_delay = run_A2()
    C7 = run_C_cn7(f"{out}/risk_v0_results.json", f"{out}/predictions_CN7.csv")
    def summ(df, ok_col):
        g = df.groupby("후보"); return pd.DataFrame({"식별 비율": g.식별.mean(), "엉킴<0.9 비율": g.엉킴.apply(lambda x: (x < RHO_MAX).mean()),
                                                   "결정 가능 비율": g[ok_col].mean(), "tier": g.tier.first(), "lead": g.lead.first()})
    dfC["결정가능"] = dfC.검정력 >= POWER_MIN
    sC = summ(dfC, "결정가능"); sA = summ(dfA, "결정가능"); sA2 = summ(dfA2, "결정가능")
    specA2 = (dfA2[dfA2.센서오차K.notna()].groupby(["후보", "센서오차K"]).결정가능.mean().unstack()).reset_index()
    cards = []
    for cause, s in [("C", sC), ("A", sA2)]:
        for name, r in s.iterrows():
            if name.startswith("노즐 + ") or name == "금형 표면 온도 센서" or "이상적 가정" in name: continue   # 조합은 표면 단독과 같고, 표면 센서는 사양 카드로 대체
            if r.tier is None or (isinstance(r.tier, float) and np.isnan(r.tier)): continue
            p1, p2, p3 = r["식별 비율"] >= ROBUST_MIN, r["엉킴<0.9 비율"] >= ROBUST_MIN, r["결정 가능 비율"] >= ROBUST_MIN
            pre = ("①" if p1 else "✗①") + ("②" if p2 else "✗②") + ("③" if p3 else "✗③")
            post = ("①" if p1 else "✗①") + (("②" if p2 else "✗②") if RHO_APPLIES[cause] else "(②해당없음)") + ("③" if p3 else "✗③")
            cards.append(dict(요청=name, 원인=cause, 판정=post, 통과=bool(p1 and p3 and (p2 or not RHO_APPLIES[cause])), 사전판정=pre, 사전통과=bool(p1 and p2 and p3),
                              tier=r.tier, lead=int(r.lead), 근거=f"격자 통과 비율: 식별 {r['식별 비율']:.0%}, 엉킴 {r['엉킴<0.9 비율']:.0%}, 결정 {r['결정 가능 비율']:.0%}"))
    cards.append(dict(요청="조건 고정 사출속도 계단 시험 (1 mm/s, 벤트 청소 전후)", 원인="B", 판정="③(설계로 충족)", 통과=True, 사전판정="③(설계로 충족)", 사전통과=True, tier="B", lead=14,
                      근거=f"현재 같은 조건 관측 없음. 1 mm/s 계단 {B['설계'][1]['샷 수(단계당 20)']}샷 × 2회로 구간 폭 1 mm/s"))
    cards.append(dict(요청=f"CN7 라벨 {C7['필요_개월']}개월 추가", 원인="C(CN7)", 판정="통계형", 통과=True, 사전판정="통계형", 사전통과=True, tier="A", lead=int(C7["필요_개월"] * 30),
                      근거=f"한쪽 불량 샷 {C7['필요_한쪽불량샷']}개 필요 (현재 {C7['현재_한쪽불량샷']}개, {C7['라벨기간_일']}일)"))
    sp = dfA2[(dfA2.후보 == "금형 표면 온도 센서") & dfA2.센서오차K.notna()]
    ok_s = [e for e in GRID_A2["센서오차K"] if sp[sp.센서오차K == e].결정가능.mean() >= ROBUST_MIN]
    if ok_s:
        emax = max(ok_s); rate = sp[sp.센서오차K <= emax].결정가능.mean()
        cards.append(dict(요청=f"금형 표면 온도 센서 (오차 {emax:g} K 이내)", 원인="A", 판정="①②③ (사양 조건부)", 통과=True, 사전판정="단계 2 추가", 사전통과=None,
                          tier="C", lead=60, 근거=f"PINN 모델: 오차 {emax:g} K 이내 격자 {rate:.0%} 통과(오차 2 K 포함 시 {sp.결정가능.mean():.0%}). 노즐 센서를 더해도 이득 거의 없음. 손 수식 모델도 '금형 쪽 측정 필요'로 같은 결론"))
    for q in QUAL: cards.append(dict(요청=q["요청"], 원인=q["원인"], 판정="정성(정의·로그)", 통과=None, 사전판정="정성(정의·로그)", 사전통과=None, tier=q["tier"], lead=q["lead"], 근거=q["푸는질문"]))
    cd = pd.DataFrame(cards)
    cd = cd[~cd.요청.str.contains("대조")].copy()
    cd["순위키"] = cd.통과.map({True: 0, None: 1, False: 2}) * 10 + cd.tier.map(TIER_ORDER)
    cd = cd.sort_values(["순위키", "lead"]).drop(columns="순위키").reset_index(drop=True); cd.index += 1
    # 그림: 후보별 4단 통과 비율
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 3.4))
    for a, (s, t) in zip(ax, [(sC, "원인 C (캐비티 분배)"), (sA, "원인 A — 손 수식 (대조)"), (sA2, "원인 A — PINN 관측 함수 (주)")]):
        s2 = s[["식별 비율", "엉킴<0.9 비율", "결정 가능 비율"]] * 100
        s2.plot.barh(ax=a, color=["#9aa3ad", "#6b8fc9", "#1d3b6e"], width=0.75, legend=(a is ax[0]))
        a.axvline(ROBUST_MIN * 100, ls="--", c="#c92a2a", lw=1); a.set_xlim(0, 105); a.set_xlabel("가정치 격자 중 통과 비율 (%)"); a.set_title(t, loc="left"); a.set_ylabel("")
    if ax[0].get_legend(): ax[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(f"{out}/요청_1_4단판정.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    R = dict(사전고정=dict(결정질문=DECISIONS, 기준=dict(엉킴상한=RHO_MAX, 불확실성상한=REL_SE_MAX, 검정력하한=POWER_MIN, 강건비율=ROBUST_MIN), 측정오차=SIGMA),
             PINN_FLR0=flr0, C_요약=sC.reset_index().to_dict("records"), C_대표값=repC.reset_index()[["후보", "식별", "엉킴", "검정력"]].to_dict("records"),
             C_필요샷수_최악조건=needC, A_요약=sA.reset_index().to_dict("records"),
             A_대표값=dfA[(dfA.ΔT0 == 19) & (dfA.Δδ0 == 0.02) & (dfA.노즐오차K == 1.0)][["후보", "식별", "엉킴", "ΔT불확실성", "Δδ불확실성", "결정가능"]].to_dict("records"),
             A_필요사양=needA, A_PINN_요약=sA2.reset_index().to_dict("records"), A_PINN_센서오차별=specA2.to_dict("records"),
             A_PINN_대표값=dfA2[(dfA2.ΔT0 == 19) & (dfA2.ΔTs0 == 1) & ((dfA2.센서오차K == 1.0) | dfA2.센서오차K.isna())][["후보", "엉킴", "ΔT불확실성", "ΔTs불확실성K", "결정가능"]].to_dict("records"),
             A_PINN_충진지연FLR=[float(x) for x in flr_delay],
             A_모델비교="주 모델 = PINN 관측 함수(고화층을 충진시간·금형 표면·수지 온도로 계산). 손 수식(Δδ를 자유 미지수로 둠)은 대조. 공통 결론: 노즐 단독 불충분, 금형 쪽 측정 필요", B=B, C_CN7통계=C7, 수정사유="② 엉킴 기준은 결정이 두 값을 따로 가르는 원인(A)에만 적용 (결과를 본 뒤 수정, 사전 판정 병기)", 요청카드=cd.reset_index().rename(columns={"index": "순위"}).to_dict("records"))
    json.dump(json_safe(R), open(f"{out}/request_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda x: None if x is None or (isinstance(x, float) and np.isinf(x)) else str(x), allow_nan=False)
    write_cards(R, out)
    return R

# ============================================================ 단계 3: 요청서 자동 생성 (요청 카드 + 실행 계획)
# 카드 세부 내용(무엇·얼마나·판정 후 행동·선행 조건)은 요청 이름 앞부분으로 찾는다. 정량 수치는 request_results.json에서 가져온다.
CARD_DETAIL = {
    "캐비티별 부품 중량": dict(무엇="샷마다 LH·RH 부품 중량 기록 (0.01 g 저울). 연속 30샷 이상, 불량이 나는 구간 포함",
                        얼마나="계산상 최소 1샷(측정오차만 고려한 하한). 샷 간 흔들림 추정을 위해 30샷 이상",
                        판정후="R_RH/R_LH > 1이 95%로 확인되면 RH 벤트·러너 우선 점검. 차이가 없으면 캐비티 압력 센서로 넘어감", 선행="저울 1대"),
    "CN7 라벨": dict(무엇="현재 방식의 불량 판정을 계속하되 LH/RH 구분과 판정 시각을 함께 기록",
                   얼마나="같은 샷에서 한쪽만 불량인 샷 94개 (현재 하루 약 0.66개 → 약 4.7개월)",
                   판정후="RH 비중이 0.5보다 유의하게 크면 CN7 위험점수에 캐비티 성분 추가(재학습은 새 데이터로만)", 선행="없음 (현행 업무 유지)"),
    "조건 고정 사출속도 계단 시험": dict(무엇="금형온도·레시피 고정, 사출속도 55.5→61.5 mm/s를 1 mm/s씩 올림. 단계당 20샷. 벤트 청소 전과 후 각 1회",
                              얼마나="140샷 × 2회 = 280샷 (생산 약 5시간분)",
                              판정후="첫 불량 단계 직전 속도를 설비 상한 후보로. 청소 후 임계속도가 오르면 벤트 청소 주기를 처방에 추가",
                              선행="생산 계획에 시험 시간 확보, 첫 불량 시 즉시 중단·시험 부품 전량 격리"),
    "캐비티 압력 센서": dict(무엇="LH·RH 캐비티에 압력 센서 각 1채널", 얼마나="관측식 확정 후 계산 (현재 정량 결과는 유량 직접 관측이라는 이상적 상한)",
                       판정후="센서 위치·압력-유량 관계를 정한 뒤 식별성 재계산. 중량 결과가 모호할 때의 2차 수단", 선행="금형 가공(센서 장착), 센서 위치·러너 구조 자료"),
    "금형 표면 온도 센서": dict(무엇="캐비티 표면(또는 표면 근처) 온도를 재가동 직후 6샷 동안 기록. 정확도 1 K 이내",
                         얼마나="재가동 사건 1회부터 판정 가능 (PINN 모델 기준)",
                         판정후="금형 표면 부족 ΔTs가 작으면 대처를 수지 온도(퍼지·예열)로 확정, 크면 재가동 시 금형 예열 절차 추가",
                         선행="센서 정확도 1 K 이내 사양 확인 (2 K급이면 대부분 판정 불가)"),
    "초기허용불량 판정 기준": dict(무엇="'초기허용불량'을 몇 샷까지, 어떤 외관 기준으로 분류하는지 현장 기준서",
                           얼마나="문서 1부", 판정후="물리 불량이면 A 성능을 '재가동 불량 검출'로, 일괄 폐기면 '가동 초기 폐기 구간 검출'로 표현",
                           선행="없음"),
    "ERR_FACT_QTY 정의": dict(무엇="무라벨 불량 수의 집계 단위·시점·분모와 오더별 생산 수량", 얼마나="정의서 1부 + 오더별 수량표",
                             판정후="정의가 확인되면 무라벨 오더 단위 약한 정답으로 사용 가능 여부 재검토", 선행="없음"),
    "사출 설정값 로그": dict(무엇="샷 또는 작업오더별 사출속도·압력 한계·절환 위치 설정값", 얼마나="라벨 기간과 같은 기간",
                       판정후="A에서 속도가 설정을 못 따라간 이유 확인. '설정 대비 실제 속도'를 관측 함수에 넣어 A의 미지수를 정량 분석으로 전환",
                       선행="없음 (설비 로그 내보내기)"),
    "정지 사유·가동 절차 로그": dict(무엇="정지마다 사유, 재가동 시 퍼지·예열 실시 여부", 얼마나="최소 1개월",
                             판정후="퍼지·예열 여부별 식은 수지 지문 비율 비교 → 처방 A-3(짧은 정지 뒤 절차) 확정", 선행="없음"),
    "절환 위치 실측 기록": dict(무엇="V/P 절환 위치 실측값 (현재 0 또는 상수로 기록됨)", 얼마나="설정 변경 1회",
                          판정후="미성형·플래시 원인 분석에 절환 위치 사용", 선행="데이터 수집 설정 변경"),
    "레시피·자재 코드": dict(무엇="작업오더별 레시피 코드, 수지 로트, 설정 변경 이력", 얼마나="라벨 기간과 같은 기간",
                       판정후="다른 레시피 혼입을 용융온도 역추정 대신 코드로 직접 식별 → 보류 판정 정확도 향상", 선행="없음"),
    "수지 데이터시트": dict(무엇="ASA 허용 가공온도, 점도 곡선, PVT 데이터", 얼마나="자재 공급사 자료 1부",
                      판정후="열분해 한계로 '용융 +35℃ 불가' 판단 확정, 점도 활성화에너지(가정 52 kJ/mol) 검증", 선행="자재 등급 확인"),
    "검사 결과 즉시 회신": dict(무엇="검사 결과를 수 샷 이내에 모델에 회신 (인라인 검사)", 얼마나="상시",
                         판정후="쌍둥이 감시로 CN7 확률 개선 +32% → +45% 기대 (검사 즉시 회신 가정)", 선행="인라인 검사 설비"),
    "노즐 수지 온도 센서": dict(무엇="(비권장) 노즐 내부 수지 온도", 얼마나="-",
                         판정후="손 수식·PINN 두 모델 모두 단독으로는 판정 불가. 0.5 K급 정밀도일 때만 일부 통과", 선행="-"),
}
PHASES = [("즉시 (1주 이내)", 0, 7), ("2주", 8, 14), ("2~3개월", 15, 90), ("5개월", 91, 10_000)]

def _detail(name):
    for k, v in CARD_DETAIL.items():
        if name.startswith(k): return v
    return dict(무엇="-", 얼마나="-", 판정후="-", 선행="-")

def write_cards(R, out="."):
    cards = R["요청카드"]
    css = ("body{font-family:'Malgun Gothic','Apple SD Gothic Neo','Noto Sans KR','Noto Sans CJK KR','NanumGothic',sans-serif;max-width:900px;margin:24px auto;padding:0 16px;color:#1d2330;line-height:1.6;font-size:14px}"
           "h1{font-size:24px}h2{color:#1d3b6e;border-bottom:2px solid #1d3b6e;margin-top:28px}.card{border:1px solid #d5dae3;border-radius:10px;margin:12px 0;overflow:hidden;page-break-inside:avoid}"
           ".hd{background:#1d3b6e;color:#fff;padding:8px 12px;font-weight:700}.hd.q{background:#6b7a90}.hd.f{background:#9aa3ad}"
           "table{border-collapse:collapse;width:100%;font-size:13px}td{border-top:1px solid #eef0f4;padding:5px 10px;vertical-align:top}td:first-child{width:110px;color:#555;font-weight:700;background:#fafbfc}"
           ".small{font-size:12px;color:#666}")
    H = [f"<html><head><meta charset='utf-8'><title>PINNSET 데이터 요청서</title><style>{css}</style></head><body>",
         "<h1>PINNSET 데이터 요청서</h1>",
         "<p class='small'>request.py가 request_results.json에서 자동 생성. 정량 판정 = 야코비안 식별성(4단 판정), 실험 설계, 검정력. "
         "정렬 = 정량 통과 → 정성 → 미통과, 같은 묶음 안에서 비용 등급(A→B→C) → 기간. 격자 통과 비율은 명시한 가정·오차 격자에서 기준을 넘은 비율이며 실제 성공 확률이 아니다.</p>", "<h2>1. 실행 계획</h2>"]
    for ph, lo, hi in PHASES:
        items = [c for c in cards if lo <= int(c["lead"]) <= hi and c["통과"] is not False]
        if items: H.append(f"<p><b>{ph}</b>: " + " · ".join(f"#{c['순위']} {c['요청']}" for c in items) + "</p>")
    H.append("<p class='small'>순위는 비용 우선 규칙이지만 실제로는 병행한다. A급(기록만 켜면 됨)은 오늘 시작하고, 라벨 누적은 그동안 저절로 진행된다.</p>")
    H.append("<h2>2. 요청 카드</h2>")
    for c in cards:
        d = _detail(c["요청"]); cls = "hd" if c["통과"] is True else ("hd q" if c["통과"] is None else "hd f")
        tier = {"A": "A급 (기록만 켜면 됨)", "B": "B급 (문서·계획 실험)", "C": "C급 (센서·설비)"}.get(c["tier"], c["tier"])
        H.append(f"<div class='card'><div class='{cls}'>#{c['순위']} {c['요청']} <span style='float:right'>원인 {c['원인']}</span></div><table>"
                 f"<tr><td>판정</td><td>{c['판정']} <span class='small'>(사전 기준: {c['사전판정']})</span></td></tr>"
                 f"<tr><td>무엇을</td><td>{d['무엇']}</td></tr><tr><td>근거·정량 효과</td><td>{c['근거']}</td></tr>"
                 f"<tr><td>얼마나</td><td>{d['얼마나']}</td></tr><tr><td>판정 후 행동</td><td>{d['판정후']}</td></tr>"
                 f"<tr><td>비용·기간</td><td>{tier} · 약 {c['lead']}일</td></tr><tr><td>선행 조건</td><td>{d['선행']}</td></tr></table></div>")
    H.append("</body></html>")
    open(f"{out}/PINNSET_데이터요청서.html", "w", encoding="utf-8").write("\n".join(H))
