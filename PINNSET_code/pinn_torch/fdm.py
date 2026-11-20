import numpy as np
from scipy.integrate import solve_ivp
from common import t0, erf_semi
AG, AM = 0.115, 0.080                      # 유리 상태 / 용융 상태 열확산율 (mm²/s, ASA 대표값)
def a_of(th, thg, w=0.03, ratio=None):
    s=1/(1+np.exp(-(th-thg)/w))
    if ratio is None: return (AG+(AM-AG)*s)/AM
    return ratio+(1-ratio)*s                 # 실험 4: 유리/용융 열확산율 비 r을 바꿔 가며 (r=1이면 물성 일정)
def solve(thg=None, Bi=None, tend=0.65, n=500, ic='erf', teval=None, ratio=None):
    """비선형 확산 FDM. 벽: θ=0 (Bi=None) 또는 Robin θ_x=Bi·θ. 중심 대칭."""
    x=np.linspace(0,1,n+1); dx=x[1]-x[0]
    th0=erf_semi(x,t0) if ic=='erf' else np.ones_like(x)
    def f(t,y):
        th=y.copy()
        if Bi is None: th[0]=0.0
        gw=th[1]-2*dx*Bi*th[0] if Bi is not None else 0.0     # 벽 유령점
        lap=np.empty_like(th)
        lap[1:-1]=(th[2:]-2*th[1:-1]+th[:-2])/dx**2
        lap[-1]=2*(th[-2]-th[-1])/dx**2
        lap[0]=(th[1]-2*th[0]+gw)/dx**2 if Bi is not None else 0.0
        a=a_of(th,thg,ratio=ratio) if thg is not None else 1.0
        return a*lap
    s=solve_ivp(f,(t0,tend),th0,method='BDF',t_eval=teval,rtol=1e-7,atol=1e-9)
    return x,s.t,s.y
def flr_from_profile(x,th,thnf):
    i=np.argmax(th>=thnf); return np.interp(thnf,[th[i-1],th[i]],[x[i-1],x[i]]) if i>0 else 0.0
