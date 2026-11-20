import numpy as np
from scipy.special import erf
h_mm, a_m = 4.37/2, 0.080            # 반두께(mm), 용융 상태 열확산율(mm²/s) — ASA 기준
tref = h_mm**2/a_m                     # 시간 기준 (초)
t0, t1 = 0.002, 0.65                   # 무차원 시간 범위 (≈0.12 s ~ 38.8 s)
LG = np.log(t1/t0)
def feats(x, t, p=None):
    """입력 특징과 그 미분: x̂∈[0,1] → 2x̂−1, t̂ → 로그시간 정규화 (초기 가파른 층을 잘 보려고)"""
    s = np.log(t/t0)/LG
    cols=[2*x-1, 2*s-1]; dx=[np.full_like(x,2.0), np.zeros_like(x)]; dt=[np.zeros_like(x), 2/(t*LG)]
    if p is not None: cols.append(p); dx.append(np.zeros_like(x)); dt.append(np.zeros_like(x))
    F=np.hstack(cols); return F, np.hstack(dx), np.hstack(dt), np.zeros_like(F)
def series(x, t, n=400):
    """유한 두께 평판 정확해 (벽 θ=0, 중심 대칭, 초기 θ=1)"""
    x=np.asarray(x,float); t=np.asarray(t,float); shp=np.broadcast(x,t).shape
    xx=np.broadcast_to(x,shp).ravel()[None,:]; tt=np.broadcast_to(t,shp).ravel()[None,:]
    k=np.arange(n)[:,None]; lam=(2*k+1)*np.pi/2
    return (4/((2*k+1)*np.pi)*np.sin(lam*xx)*np.exp(-lam**2*tt)).sum(0).reshape(shp)
def erf_semi(x, t):
    """반무한 평판 해석해"""
    return erf(x/(2*np.sqrt(t)))
def sample(n, rng):
    x=rng.random((n,1))**1.6; t=t0*np.exp(rng.random((n,1))*LG); return x,t
