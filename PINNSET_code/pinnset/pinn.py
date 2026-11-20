"""
PINN 성형 윈도우 지도 — 학습된 가중치 불러오기와 지도 계산
(학습: pinn_torch/pinn_torch_train.py, PyTorch 2.11·float64. 여기서는 학습된 가중치로 값만 계산하므로 PyTorch 불필요)
가중치 파일은 PINNSET_code/weights/ 에 있다 (WEIGHTS).
================================================================
가중치
  pinn_exp2_map.npz   : 실험 2. 온도에 따라 변하는 열확산율(ASA: 유리 0.115 → 용융 0.080 mm²/s, Tg 105℃)
                        입력 3개 (위치, 시간, 유리전이 위치 θg) → 무차원 온도 θ
  pinn_exp1_const.npz : 실험 1. 물성 일정(α = 0.080). 입력 2개 (위치, 시간) → θ
  pinn_exp4_resin.npz : 실험 4. 수지 일반화. 입력 4개 (위치, 시간, θg, 열확산율 비 r = α_유리/α_용융 ∈ [1, 2]) → θ

신경망: 입력 → [tanh 40] × 4층 → 출력 1 (완전연결)

무차원화 (학습 때와 똑같이 넣어야 한다)
  x̂ = 금형 벽에서의 거리 / 반두께(2.185 mm)          ∈ [0, 1]   (0 = 벽, 1 = 두께 중심)
  t̂ = 시간(초) / t_ref,  t_ref = h²/α_용융 = 59.68 초  ∈ [0.002, 0.65]  (≈ 0.12 ~ 38.8 초)
  θ  = (T − 금형온도) / (용융온도 − 금형온도)            (벽 0, 녹은 수지 1)
  θg = (105 − 금형온도) / (용융온도 − 금형온도)          ∈ [0.10, 0.40]  (유리전이 위치)
특징 (신경망 입력)
  f0 = 2x̂ − 1
  f1 = 2·log(t̂/0.002)/log(0.65/0.002) − 1              (로그 시간: 초기 얇은 층을 잘 보려고)
  f2 = 2·(θg − 0.10)/0.30 − 1                           (실험 2·4)
  f3 = 2·(r − 1.0)/1.0 − 1                              (실험 4만)
경계·초기조건 (학습 조건): 벽 θ = 0(금형온도 고정), 중심 대칭, t̂ = 0.002에서 erf 초기 분포
FLR = 고화층 두께 / 반두께 = 벽에서부터 θ가 θ_nf = (110 − 금형온도)/(용융온도 − 금형온도)에 처음 닿는 x̂
      (두께 전체가 θ_nf 아래면 1.0 = 완전 고화)

검증 기준값 (weights/학습기록/verify_report_exp1_2.md)
  라벨 운전점(금형 22.8℃, 용융 275.5℃, 충진 4.45초): 실험 2 FLR 0.1763 (FDM 0.1751), 실험 1 FLR 0.1722 (푸리에 급수 정답 0.1726)
  FDM 대비 FLR 오차: 중앙 0.0009, 최대 0.0021 (금형 20~80℃, 1.5~8초, 15개 조건)
"""
from pathlib import Path
import numpy as np
from .config import HALF_THICK as H_MM, ALPHA as A_MELT, T_GLASS, T_NF, T_MELT_REF

WEIGHTS = Path(__file__).resolve().parent.parent / "weights"
T0, T1 = 0.002, 0.65
TREF = H_MM ** 2 / A_MELT
LG = np.log(T1 / T0)

class PINN:
    def __init__(self, path):
        w = np.load(path); self.W = [w[f"W{i}"] for i in range(4)]; self.b = [w[f"b{i}"] for i in range(4)]
        self.Wo, self.bo = w["Wo"], w["bo"]; self.n_in = self.W[0].shape[0]

    def theta(self, xhat, t_sec, theta_g=None, ratio=None, tref=TREF):
        """무차원 온도 θ(x̂, t). 배열 브로드캐스트 가능. tref = 반두께²/용융 열확산율(초): 수지·두께가 바뀌면 이것만 바꾼다"""
        xhat, t_sec = np.broadcast_arrays(np.asarray(xhat, float), np.asarray(t_sec, float))
        th = t_sec / tref
        if np.any((th < T0 * 0.999) | (th > T1 * 1.001)):
            raise ValueError(f"시간은 {T0*tref:.2f}~{T1*tref:.1f}초 범위만 학습됨")
        f = [2 * xhat - 1, 2 * np.log(th / T0) / LG - 1]
        if self.n_in >= 3:
            g = np.broadcast_to(np.asarray(theta_g, float), xhat.shape)
            if np.any((g < 0.099) | (g > 0.401)):
                raise ValueError("θg는 0.10~0.40 범위만 학습됨 (금형·용융온도 조합을 확인)")
            f.append(2 * (g - 0.10) / 0.30 - 1)
        if self.n_in == 4:
            if ratio is None: raise ValueError("실험 4 가중치는 열확산율 비 ratio(α_유리/α_용융)가 필요함")
            if not (0.999 <= ratio <= 2.001): raise ValueError("ratio는 1.0~2.0 범위만 학습됨")
            f.append(np.full(xhat.shape, 2 * (ratio - 1.0) / 1.0 - 1))
        h = np.stack([a.ravel() for a in f], 1)
        for W, b in zip(self.W, self.b):
            h = np.tanh(h @ W + b)
        return (h @ self.Wo + self.bo).reshape(xhat.shape)

    def temperature(self, x_mm, t_sec, T_mold, T_melt):
        """실단위 온도(℃)"""
        g = (T_GLASS - T_mold) / (T_melt - T_mold) if self.n_in >= 3 else None
        return T_mold + (T_melt - T_mold) * self.theta(np.asarray(x_mm) / H_MM, t_sec, g)

    def flr(self, T_mold, t_fill, T_melt=T_MELT_REF, n=801, Tg=T_GLASS, Tnf=T_NF, ratio=None, half_mm=H_MM, a_melt=A_MELT):
        """고화층 비율 FLR (0~1). 기본값 = 본 과제 ASA 부품. 다른 수지·두께는 Tg·Tnf·half_mm·a_melt(·ratio, 실험 4)만 바꿔 넣는다"""
        x = np.linspace(0, 1, n)
        g = (Tg - T_mold) / (T_melt - T_mold) if self.n_in >= 3 else None
        th = self.theta(x, np.full(n, t_fill), g, ratio=ratio, tref=half_mm ** 2 / a_melt)
        nf = (Tnf - T_mold) / (T_melt - T_mold)
        if not (th >= nf).any(): return 1.0
        i = int(np.argmax(th >= nf))
        return 0.0 if i == 0 else float(np.interp(nf, [th[i - 1], th[i]], [x[i - 1], x[i]]))

    def flr_map(self, T_molds, t_fills, T_melt=T_MELT_REF):
        """지도: 행 = 충진시간, 열 = 금형온도"""
        return np.array([[self.flr(Tw, tf, T_melt) for Tw in T_molds] for tf in t_fills])

if __name__ == "__main__":
    m2, m1 = PINN(WEIGHTS / "pinn_exp2_map.npz"), PINN(WEIGHTS / "pinn_exp1_const.npz")
    print(f"라벨 운전점 FLR: 실험2 {m2.flr(22.8, 4.45):.4f} (기대 0.1763) | 실험1 {m1.flr(22.8, 4.45):.4f} (기대 0.1722)")
    print(f"FDM 대조점 (금형 50℃, 4.45초): {m2.flr(50, 4.45):.4f} (PINN 기록 0.1322, FDM 0.1324)")
    print(f"FDM 대조점 (금형 80℃, 8초):    {m2.flr(80, 8.0):.4f} (PINN 기록 0.1012, FDM 0.1007)")
    M = m2.flr_map(np.linspace(20, 80, 7), [1, 4.45, 10])
    print("지도 예 (행: 충진 1 / 4.45 / 10초, 열: 금형 20~80℃ 7점):"); print(np.round(M, 3))
