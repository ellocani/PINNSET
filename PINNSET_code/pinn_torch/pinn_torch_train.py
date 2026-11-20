"""
PINNSET PINN 실험 1·2·3·3b·4 — PyTorch 학습 스크립트
=================================================
제출 코드 weights/의 가중치(pinn_exp1_const.npz, pinn_exp2_map.npz, pinn_exp3*_bi.npz)를 만든 학습 코드.
미분(u_t, u_x, u_xx)은 torch.autograd, 최적화는 Adam, 정밀도 float64. 초기 가중치·표본점은 시드 고정 난수로 만든다.

실행
  pip install -r requirements_torch.txt
  python pinn_torch_train.py --exp 1 2 3 3b 4 --out weights_torch   # 전체 (실험 4 포함, CPU 약 1.5~2시간 / --exp all은 실험 4 제외)
  python pinn_torch_train.py --exp 1 2 --out weights_torch          # 지도용 가중치만
  python pinn_torch_train.py --verify-only --out ../weights         # PyTorch 없이 기존 가중치 검증만
  python pinn_torch_train.py --exp 4 --out weights_torch --device cuda   # 실험 4: 수지 일반화 지도 (GPU 권장, CPU 약 40~60분)

산출물 (--out 폴더)
  pinn_exp1_const.npz, pinn_exp2_map.npz   : 키 W0~W3, b0~b3, Wo, bo (float64) → weights/pinn_map.py와 제출 파이프라인이 읽음
  pinn_exp3_bi.npz, pinn_exp3b_bi.npz       : 실험 3·3b 가중치 + 역산한 log(Bi)
  pinn_exp4_resin.npz                       : 실험 4 가중치 (입력 4개: 위치, 시간, 유리전이 위치 θg, 열확산율 비 r)
  P1.pkl, P2.pkl, R3.pkl, R3b.pkl           : 그림 스크립트 fig1~3.py 입력
  train_log.json                            : 실험별 손실 기록, 역산 Bi, 소요 시간
  verify_report.json / verify_report.md     : 기준점·푸리에 급수·FDM 대조 결과와 통과 여부

재현성
  · 초기 가중치: 시드 고정 난수 default_rng(seed).normal(0, √(1/입력 수)), 편향 0
  · 표본점: 시드 고정 난수로 뽑고 500회마다 새로 뽑음
  · Adam: β=(0.9, 0.999), ε=1e-8, 학습률 감쇠 시점 고정
  · 정밀도: float64
  CPU 종류·스레드 수에 따라 부동소수점 합산 순서가 달라 가중치 숫자가 비트 단위로 같지는 않을 수 있다.
  그래서 같은 온도장을 배웠는지를 verify_report로 판정한다: 기준점 FLR, 정답(푸리에 급수·FDM) 대비 오차.
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path
import numpy as np
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # Windows(CP949) 콘솔에서도 기호 출력 오류가 나지 않게
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import t0, t1, LG, tref, erf_semi, series, sample          # 상수·정답 함수(푸리에 급수, 반무한 해)
from fdm import solve, flr_from_profile

AG, AM = 0.115, 0.080      # 유리 / 용융 상태 열확산율 (mm²/s, ASA 대표값)
WG = 0.03                  # 유리전이 전환 폭 (θ 단위)


# ============================================================ PyTorch 부분 (학습할 때만 import)
def _torch():
    import torch
    torch.set_default_dtype(torch.float64)
    return torch


def init_params(torch, n_in, width, depth, seed):
    """시드 고정 난수로 초기화 (층 순서대로 W, b, 마지막에 출력층)"""
    rng = np.random.default_rng(seed); P = {}
    sizes = [n_in] + [width] * depth
    for l in range(depth):
        P[f"W{l}"] = rng.normal(0, np.sqrt(1 / sizes[l]), (sizes[l], sizes[l + 1]))
        P[f"b{l}"] = np.zeros((1, sizes[l + 1]))
    P["Wo"] = rng.normal(0, np.sqrt(1 / width), (width, 1)); P["bo"] = np.zeros((1, 1))
    return {k: torch.tensor(v, requires_grad=True) for k, v in P.items()}


def mlp(torch, P, F, depth):
    h = F
    for l in range(depth):
        h = torch.tanh(h @ P[f"W{l}"] + P[f"b{l}"])
    return h @ P["Wo"] + P["bo"]


def jet(torch, P, x, t, depth, lg=LG, p=None, need=("u", "ux", "ut", "uxx")):
    """u와 x·t 미분. 특징: 2x−1, 2·log(t/t0)/lg − 1, (θg 정규화)"""
    x = torch.as_tensor(x).clone().requires_grad_(True); t = torch.as_tensor(t).clone().requires_grad_(True)
    cols = [2 * x - 1, 2 * torch.log(t / t0) / lg - 1]
    if p is not None: cols.append(torch.as_tensor(p))
    u = mlp(torch, P, torch.cat(cols, 1), depth); out = {"u": u}
    if "ux" in need or "uxx" in need:
        ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]; out["ux"] = ux
        if "uxx" in need: out["uxx"] = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    if "ut" in need: out["ut"] = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    return out


def save_npz(P, path, extra=None):
    d = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in P.items()}
    if extra: d.update(extra)
    np.savez(path, **d); return {k: v for k, v in d.items() if k[0] in "Wb"}


def adam(torch, params, lr):
    return torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999), eps=1e-8)


# ------------------------------------------------------------ 실험 1: 물성 일정 (exp1.py)
def exp1(out, iters=8001, log=print):
    torch = _torch(); D = 4; rng = np.random.default_rng(0)
    P = init_params(torch, 2, 40, D, 1); opt = adam(torch, list(P.values()), 2e-3)
    xb_t = t0 * np.exp(np.linspace(0, 1, 120)[:, None] * LG)
    xi = np.linspace(0, 1, 200)[:, None] ** 1.5; ti = np.full_like(xi, t0); th_i = torch.tensor(erf_semi(xi, t0))
    hist = []; T0 = time.time()
    for it in range(iters):
        if it % 500 == 0: xr, tr = sample(2500, rng)
        o = jet(torch, P, xr, tr, D); r = (o["ut"] - o["uxx"]) * torch.tensor(tr)
        uw = jet(torch, P, np.zeros_like(xb_t), xb_t, D, need=("u",))["u"]
        uc = jet(torch, P, np.ones_like(xb_t), xb_t, D, need=("ux",))["ux"]
        u0 = jet(torch, P, xi, ti, D, need=("u",))["u"]
        L = (r ** 2).mean() + 10 * (uw ** 2).mean() + (uc ** 2).mean() + 10 * ((u0 - th_i) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step()
        if it in (4000, 6000):
            for g in opt.param_groups: g["lr"] *= 0.3
        if it % 2000 == 0: hist.append((it, float(L))); log(f"[실험1] {it} 손실 {float(L):.2e} {time.time()-T0:.0f}s")
    W = save_npz(P, out / "pinn_exp1_const.npz"); pickle.dump(W, open(out / "P1.pkl", "wb"))
    return dict(손실기록=hist, 소요초=time.time() - T0)


# ------------------------------------------------------------ 실험 2: 온도 의존 물성 → 성형 윈도우 지도 (exp2.py)
def exp2(out, iters=9001, log=print):
    torch = _torch(); D = 4; rng = np.random.default_rng(1)
    P = init_params(torch, 3, 40, D, 2); opt = adam(torch, list(P.values()), 2e-3)
    pn = lambda g: (g - 0.10) / 0.30 * 2 - 1
    tb = t0 * np.exp(np.linspace(0, 1, 100)[:, None] * LG); xi = np.linspace(0, 1, 150)[:, None] ** 1.5
    th_i = torch.tensor(erf_semi(xi, t0)); hist = []; T0 = time.time()
    for it in range(iters):
        if it % 500 == 0:
            xr, tr = sample(3000, rng); gr = 0.10 + 0.30 * rng.random((3000, 1))
            gb = 0.10 + 0.30 * rng.random((100, 1)); gi = 0.10 + 0.30 * rng.random((150, 1))
        o = jet(torch, P, xr, tr, D, p=pn(gr))
        a = (AG + (AM - AG) * torch.sigmoid((o["u"] - torch.tensor(gr)) / WG)) / AM
        r = (o["ut"] - a * o["uxx"]) * torch.tensor(tr)
        uw = jet(torch, P, np.zeros_like(tb), tb, D, p=pn(gb), need=("u",))["u"]
        uc = jet(torch, P, np.ones_like(tb), tb, D, p=pn(gb), need=("ux",))["ux"]
        u0 = jet(torch, P, xi, np.full_like(xi, t0), D, p=pn(gi), need=("u",))["u"]
        L = (r ** 2).mean() + 10 * (uw ** 2).mean() + (uc ** 2).mean() + 10 * ((u0 - th_i) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step()
        if it in (5000, 7000):
            for g in opt.param_groups: g["lr"] *= 0.3
        if it % 2000 == 0: hist.append((it, float(L))); log(f"[실험2] {it} 손실 {float(L):.2e} {time.time()-T0:.0f}s")
    W = save_npz(P, out / "pinn_exp2_map.npz"); pickle.dump(W, open(out / "P2.pkl", "wb"))
    return dict(손실기록=hist, 소요초=time.time() - T0)


# ------------------------------------------------------------ 실험 3·3b: 벽면 센서로 접촉 열저항 Bi 역산 (exp3.py, exp3b.py)
BI_TRUE, TEND = 20.0, 0.20

def _sensor():
    ts = np.geomspace(0.5 / tref, TEND, 25)[:, None]
    x, tt, Y = solve(Bi=BI_TRUE, tend=TEND * 1.001, ic="ones", teval=ts.ravel())
    rng0 = np.random.default_rng(99); th_s = Y[0][:, None] + rng0.normal(0, 0.002, ts.shape)
    return ts, th_s

def _inverse_run(torch, seed, use_data, ts, th_s, *, width, depth, iters, n_col, xpow, w_res, w_data, decay, traj_every=None, log=print, tag=""):
    rng = np.random.default_rng(seed); P = init_params(torch, 2, width, depth, 10 + seed)
    q = torch.tensor([[np.log(rng.uniform(3, 100))]], requires_grad=True)          # 모르는 값: log(Bi)
    opt = adam(torch, list(P.values()) + [q], 3e-3); LGe = np.log(TEND / t0)
    tb = t0 * np.exp(np.linspace(0, 1, 80)[:, None] * LGe); xi = np.linspace(0, 1, 120)[:, None]
    T_s = torch.tensor(th_s); traj = []; T0 = time.time()
    for it in range(iters):
        if it % 500 == 0: xr = rng.random((n_col, 1)) ** xpow; tr = t0 * np.exp(rng.random((n_col, 1)) * LGe)
        o = jet(torch, P, xr, tr, depth, lg=LGe); r = (o["ut"] - o["uxx"]) * torch.tensor(tr)
        w = jet(torch, P, np.zeros_like(tb), tb, depth, lg=LGe, need=("u", "ux"))
        rob = w["u"] - w["ux"] * torch.exp(-q)                                            # 벽 Robin 조건: θ_x = Bi·θ
        uc = jet(torch, P, np.ones_like(tb), tb, depth, lg=LGe, need=("ux",))["ux"]
        u0 = jet(torch, P, xi, np.full_like(xi, t0), depth, lg=LGe, need=("u",))["u"]
        L = w_res * (r ** 2).mean() + 10 * (rob ** 2).mean() + (uc ** 2).mean() + 10 * ((u0 - 1.0) ** 2).mean()
        if use_data:
            us = jet(torch, P, np.zeros_like(ts), ts, depth, lg=LGe, need=("u",))["u"]; L = L + w_data * ((us - T_s) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step()
        if it in decay:
            for g in opt.param_groups: g["lr"] *= 0.3
        if traj_every and it % traj_every == 0:
            traj.append((it, float(torch.exp(q)))); log(f"[{tag}] 시드 {seed} {it} Bi={traj[-1][1]:.2f} 손실 {float(L):.2e}")
    res = dict(data=use_data, seed=seed, Bi=float(torch.exp(q)), loss=float(L), res=float(torch.sqrt((r.detach() ** 2).mean())))
    if traj_every: res["traj"] = traj
    log(f"[{tag}] 센서 {'있음' if use_data else '없음'} 시드 {seed}: Bi = {res['Bi']:.2f} (정답 {BI_TRUE}), 손실 {res['loss']:.2e}, {time.time()-T0:.0f}s")
    return res, P, q

def exp3(out, log=print, iters=6001):
    torch = _torch(); ts, th_s = _sensor(); res = []; keep = {}
    for use_data in (True, False):
        for seed in range(3):
            r_, P, q = _inverse_run(torch, seed, use_data, ts, th_s, width=32, depth=3, iters=iters, n_col=1500, xpow=1.6,
                                    w_res=1.0, w_data=100.0, decay=(4500,), log=log, tag="실험3")
            res.append(r_); keep[f"{'sensor' if use_data else 'nosensor'}_seed{seed}"] = (P, q)
            pickle.dump(dict(res=res, ts=ts, th_s=th_s, BI=BI_TRUE), open(out / "R3.pkl", "wb"))
    P, q = keep["sensor_seed0"]; save_npz(P, out / "pinn_exp3_bi.npz", {"logBi": q.detach().numpy()})
    return dict(결과=res)

def exp3b(out, log=print, iters=10001):
    torch = _torch(); ts, th_s = _sensor(); res = []
    for seed in range(2):
        r_, P, q = _inverse_run(torch, seed, True, ts, th_s, width=48, depth=4, iters=iters, n_col=2500, xpow=2.2,
                                w_res=3.0, w_data=30.0, decay=(6000, 8500), traj_every=1000, log=log, tag="실험3b")
        res.append(r_)
        pickle.dump(dict(res=res, ts=ts, th_s=th_s, BI=BI_TRUE), open(out / "R3b.pkl", "wb"))
        if seed == 0: save_npz(P, out / "pinn_exp3b_bi.npz", {"logBi": q.detach().numpy()})
    return dict(결과=res)


# ------------------------------------------------------------ 실험 4: 수지 일반화 지도 (열확산율 비 r도 입력)
# 실험 2는 ASA 한 가지(유리 0.115 / 용융 0.080 mm²/s, 비 1.4375)만 배웠다. 실험 4는 비 r = α_유리/α_용융을 1.0~2.0에서 함께 배워,
# 수지가 바뀌어도(열확산율 비·Tg·무유동 온도·두께·용융 열확산율이 달라도) 다시 학습하지 않고 FLR을 계산하게 한다.
R_LO, R_HI = 1.0, 2.0

def exp4(out, iters=20001, log=print, device="cpu", width=48, depth=4, n_col=4000):
    torch = _torch(); dev = torch.device(device); rng = np.random.default_rng(4)
    P = {k: v.detach().to(dev).requires_grad_(True) for k, v in init_params(torch, 4, width, depth, 4).items()}
    opt = adam(torch, list(P.values()), 2e-3)
    T = lambda a: torch.tensor(a, device=dev)
    def jet4(x, t, g, r, need=("u", "ux", "ut", "uxx")):
        x = T(x).requires_grad_(True); t = T(t).requires_grad_(True); g = T(g); r = T(r)
        F = torch.cat([2 * x - 1, 2 * torch.log(t / t0) / LG - 1, 2 * (g - 0.10) / 0.30 - 1, 2 * (r - R_LO) / (R_HI - R_LO) - 1], 1)
        u = mlp(torch, P, F, depth); o = {"u": u}
        if "ux" in need or "uxx" in need:
            ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]; o["ux"] = ux
            if "uxx" in need: o["uxx"] = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
        if "ut" in need: o["ut"] = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
        return o, g, r
    G = lambda n: 0.10 + 0.30 * rng.random((n, 1)); Rr = lambda n: R_LO + (R_HI - R_LO) * rng.random((n, 1))
    tb = t0 * np.exp(np.linspace(0, 1, 100)[:, None] * LG); xi = np.linspace(0, 1, 150)[:, None] ** 1.5; th_i = T(erf_semi(xi, t0))
    hist = []; T0 = time.time()
    for it in range(iters):
        if it % 500 == 0:
            xr, tr = sample(n_col, rng); gr, rr = G(n_col), Rr(n_col)
            gb, rb, gi, ri = G(100), Rr(100), G(150), Rr(150)
        o, g, r = jet4(xr, tr, gr, rr)
        a = r + (1 - r) * torch.sigmoid((o["u"] - g) / WG)                 # 무차원 열확산율 a(θ) = α(θ)/α_용융
        res = (o["ut"] - a * o["uxx"]) * T(tr)
        uw = jet4(np.zeros_like(tb), tb, gb, rb, need=("u",))[0]["u"]
        uc = jet4(np.ones_like(tb), tb, gb, rb, need=("ux",))[0]["ux"]
        u0 = jet4(xi, np.full_like(xi, t0), gi, ri, need=("u",))[0]["u"]
        L = (res ** 2).mean() + 10 * (uw ** 2).mean() + (uc ** 2).mean() + 10 * ((u0 - th_i) ** 2).mean()
        opt.zero_grad(); L.backward(); opt.step()
        if it in (11000, 15000, 18000):
            for g_ in opt.param_groups: g_["lr"] *= 0.3
        if it % 2000 == 0: hist.append((it, float(L))); log(f"[실험4] {it} 손실 {float(L):.2e} {time.time()-T0:.0f}s")
    save_npz({k: v.cpu() for k, v in P.items()}, out / "pinn_exp4_resin.npz", {"r_range": np.array([R_LO, R_HI])})
    return dict(손실기록=hist, 소요초=time.time() - T0, 장치=device)


# ============================================================ 검증 (numpy·scipy만, PyTorch 불필요)
ANCHORS = [(1, 22.8, 4.45, 0.1722), (2, 22.8, 4.45, 0.1763), (2, 50.0, 4.45, 0.1322), (2, 80.0, 8.0, 0.1012)]   # 제출 가중치 기록값 (2026-10-06 학습)

def _pinn_cls():
    sys.path.insert(0, str(HERE.parent / "weights")); from pinn_map import PINN; return PINN

REQUIRED = {"1": "pinn_exp1_const.npz", "2": "pinn_exp2_map.npz", "3": "pinn_exp3_bi.npz", "3b": "pinn_exp3b_bi.npz", "4": "pinn_exp4_resin.npz"}


def verify(wdir, ref_dir=None, tol_anchor=0.002, tol_ref=0.003, tol_fdm=0.003, exps=("1", "2", "3", "3b")):
    PINN = _pinn_cls(); wdir = Path(wdir); R = {"가중치 폴더": str(wdir), "통과": True}
    # 0) 검증할 실험의 필수 가중치가 있는지 먼저 확인 (없으면 실패)
    missing = [REQUIRED[e] for e in exps if e in REQUIRED and not (wdir / REQUIRED[e]).exists()]
    R["⓪ 필수 가중치"] = [dict(검증할_실험=" ".join(exps), 누락=", ".join(missing) or "없음", 통과=not missing)]
    if missing: R["통과"] = False
    n_checks = 0
    m = {e: PINN(str(wdir / f)) for e, f in [(1, "pinn_exp1_const.npz"), (2, "pinn_exp2_map.npz")] if (wdir / f).exists()}
    # ① 기준점 (제출 가중치 기록값)
    rows = []
    for e, Tw, tf, exp_ in ANCHORS:
        if e not in m: continue
        v = m[e].flr(Tw, tf); ok = abs(v - exp_) <= tol_anchor
        rows.append(dict(실험=e, 금형=Tw, 충진초=tf, 기록값=exp_, 계산값=round(v, 5), 차이=round(v - exp_, 5), 통과=ok)); R["통과"] &= ok; n_checks += 1
    R["① 기준점 (허용 ±%.3f)" % tol_anchor] = rows
    # ② 비교용 가중치 폴더(--ref)와의 FLR 차이 (지도 전체, 같은 폴더면 생략)
    if ref_dir and Path(ref_dir).resolve() != wdir.resolve():
        rr = []
        for e, f in [(1, "pinn_exp1_const.npz"), (2, "pinn_exp2_map.npz")]:
            if e in m and (Path(ref_dir) / f).exists():
                ref = PINN(str(Path(ref_dir) / f)); Tws = np.linspace(20, 80, 7) if e == 2 else [22.8]; tfs = [1.5, 3, 4.45, 6, 8, 10]
                d = [abs(m[e].flr(Tw, tf) - ref.flr(Tw, tf)) for Tw in Tws for tf in tfs]
                ok = max(d) <= tol_ref; rr.append(dict(실험=e, 비교점=len(d), 최대차이=round(max(d), 5), 중앙차이=round(float(np.median(d)), 5), 통과=ok)); R["통과"] &= ok; n_checks += 1
        R["② 비교 가중치 대비 FLR (허용 ±%.3f)" % tol_ref] = rr
    # ③ 실험 1 vs 유한 두께 정답(푸리에 급수)
    if 1 in m:
        Tw, Tm = 22.8, 275.5; nf = (110 - Tw) / (Tm - Tw); x = np.linspace(0, 1, 801); rows = []
        for ts_ in [1.5, 4.45, 8.0, 11.2]:
            a = flr_from_profile(x, series(x, ts_ / tref), nf); p = m[1].flr(Tw, ts_)
            rows.append(dict(시간초=ts_, 정답FLR=round(a, 5), PINN_FLR=round(p, 5), 상대오차=round(abs(p - a) / a, 4)))
        R["③ 실험1 vs 푸리에 급수 정답"] = rows
    # ④ 실험 2 vs FDM 수치해 (금형 20~80℃ × 충진 1.5·4.45·8초 = 15조건)
    if 2 in m:
        Tm = 275.5; rows = []
        for Tw in [20.0, 35.0, 50.0, 65.0, 80.0]:
            g = (105 - Tw) / (Tm - Tw); nf = (110 - Tw) / (Tm - Tw)
            for tf in [1.5, 4.45, 8.0]:
                x, tt, Y = solve(thg=g, tend=tf / tref * 1.0005, teval=[tf / tref]); f_ = flr_from_profile(x, Y[:, 0], nf)
                rows.append(dict(금형=Tw, 충진초=tf, FDM=round(f_, 5), PINN=round(m[2].flr(Tw, tf, Tm), 5)))
        err = [abs(r["PINN"] - r["FDM"]) for r in rows]; ok = float(np.median(err)) <= tol_fdm
        R["④ 실험2 vs FDM (허용 중앙 ±%.3f)" % tol_fdm] = dict(조건=rows, 중앙오차=round(float(np.median(err)), 5), 최대오차=round(max(err), 5), 통과=ok); R["통과"] &= ok; n_checks += 1
    # ⑤ 실험 3·3b 역산 결과 (있으면)
    for k, f in [("실험3", "R3.pkl"), ("실험3b", "R3b.pkl")]:
        if (wdir / f).exists():
            d = pickle.load(open(wdir / f, "rb"))
            R[f"⑤ {k} 역산 Bi (정답 {d['BI']})"] = [dict(센서="있음" if r["data"] else "없음", 시드=r["seed"], Bi=round(r["Bi"], 2), 손실=float(f"{r['loss']:.3e}")) for r in d["res"]]
    # ⑥ 실험 4 (수지 일반화): 열확산율 비 r × 금형온도 × 충진시간을 FDM 정답과 대조, r=1은 푸리에 급수 정답, r=1.4375는 실험 2와 비교
    if (wdir / "pinn_exp4_resin.npz").exists():
        from scipy.special import erfinv
        m4 = PINN(str(wdir / "pinn_exp4_resin.npz")); Tm = 275.5; rows = []
        m2 = m.get(2) or (PINN(str(Path(ref_dir) / "pinn_exp2_map.npz")) if ref_dir and (Path(ref_dir) / "pinn_exp2_map.npz").exists() else None)
        for r in [1.0, 1.25, 1.4375, 1.7, 2.0]:
            for Tw in [20.0, 50.0, 80.0]:
                g = (105 - Tw) / (Tm - Tw); nf = (110 - Tw) / (Tm - Tw)
                for tf in [1.5, 4.45, 8.0]:
                    x, tt, Y = solve(thg=g, tend=tf / tref * 1.0005, teval=[tf / tref], ratio=r); f_ = flr_from_profile(x, Y[:, 0], nf)
                    an = 2 * np.sqrt(tf / tref) * erfinv(nf)                    # 반무한·물성 일정 해석식 (비교용)
                    row = dict(비r=r, 금형=Tw, 충진초=tf, FDM=round(f_, 5), PINN4=round(m4.flr(Tw, tf, Tm, ratio=r), 5), 해석식=round(float(an), 5))
                    if r == 1.4375 and m2 is not None: row["실험2"] = round(m2.flr(Tw, tf, Tm), 5)
                    rows.append(row)
        err = [abs(r_["PINN4"] - r_["FDM"]) for r_ in rows]; aerr = [abs(r_["해석식"] - r_["FDM"]) for r_ in rows]
        d2 = [abs(r_["PINN4"] - r_["실험2"]) for r_ in rows if "실험2" in r_]
        x = np.linspace(0, 1, 801); s1 = []
        for ts_ in [1.5, 4.45, 8.0]:                                             # r = 1(물성 일정)은 푸리에 급수 정답이 있다
            a_ = flr_from_profile(x, series(x, ts_ / tref), (110 - 22.8) / (275.5 - 22.8)); s1.append(abs(m4.flr(22.8, ts_, 275.5, ratio=1.0) - a_))
        ok = float(np.median(err)) <= tol_fdm and max(err) <= 2 * tol_fdm
        R["⑥ 실험4 수지 일반화 vs FDM (허용 중앙 ±%.3f, 최대 ±%.3f)" % (tol_fdm, 2 * tol_fdm)] = dict(
            조건=rows, 중앙오차=round(float(np.median(err)), 5), 최대오차=round(max(err), 5), 해석식_중앙오차=round(float(np.median(aerr)), 5),
            해석식_최대오차=round(max(aerr), 5), 실험2와_최대차이=round(max(d2), 5) if d2 else None, r1_푸리에_최대오차=round(max(s1), 5), 통과=ok)
        R["통과"] &= ok; n_checks += 1
    # 수치 검증 항목이 하나도 실행되지 않았으면 통과로 볼 수 없다
    R["검증 항목 수"] = n_checks
    if n_checks == 0: R["통과"] = False
    json.dump(R, open(wdir / "verify_report.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    md = [f"# PINN 가중치 검증 보고 — {'통과' if R['통과'] else '실패'}", ""]
    for k, v in R.items():
        if isinstance(v, (list, dict)) and k not in ("통과",):
            md.append(f"## {k}"); rows = v["조건"] if isinstance(v, dict) and "조건" in v else v
            if isinstance(v, dict) and "조건" in v: md.append(", ".join(f"{a} {b}" for a, b in v.items() if a != "조건"))
            if rows:
                md.append("| " + " | ".join(rows[0].keys()) + " |"); md.append("|" + "---|" * len(rows[0]))
                md += ["| " + " | ".join(str(x) for x in r.values()) + " |" for r in rows]
            md.append("")
    open(wdir / "verify_report.md", "w", encoding="utf-8").write("\n".join(md))
    return R


# ============================================================ 실행
def main():
    ap = argparse.ArgumentParser(description="PINNSET PINN 실험을 PyTorch로 재학습")
    ap.add_argument("--exp", nargs="+", default=["all"], help="all = 1 2 3 3b (실험 4 제외). 전체는 1 2 3 3b 4")
    ap.add_argument("--out", default="weights_torch"); ap.add_argument("--ref", default=str(HERE.parent / "weights"), help="비교할 가중치 폴더 (기본: 제출 weights/)")
    ap.add_argument("--quick", action="store_true", help="동작 확인용: 반복 수를 1/20로 (결과 검증은 통과하지 않음)")
    ap.add_argument("--verify-only", action="store_true", help="학습 없이 --out 폴더의 가중치만 검증 (PyTorch 불필요)")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU 스레드 수 (0 = 기본)")
    ap.add_argument("--device", default="cpu", help="실험 4 학습 장치: cpu 또는 cuda")
    ap.add_argument("--iters4", type=int, default=20001, help="실험 4 반복 수 (검증 ⑥을 못 넘으면 늘린다)")
    ap.add_argument("--figures", action="store_true", help="검증 뒤 원본 그림 스크립트(fig1~3)로 PINN 그림 3장을 --out 폴더에 생성")
    ap.add_argument("--data", default=os.environ.get("PINNSET_DATA", ""), help="그림 2용 labeled_data.csv가 있는 폴더")
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    if not a.verify_only:
        torch = _torch()
        if a.threads: torch.set_num_threads(a.threads)
        exps = ["1", "2", "3", "3b"] if "all" in a.exp else a.exp; f = 20 if a.quick else 1; logs = {}
        lf = open(out / "train.log", "a", encoding="utf-8")
        def log(s): print(s, flush=True); lf.write(s + "\n"); lf.flush()
        log(f"torch {torch.__version__}, float64, 실험 {exps}, quick={a.quick}")
        if "1" in exps: logs["실험1"] = exp1(out, iters=8001 // f + 1, log=log)
        if "2" in exps: logs["실험2"] = exp2(out, iters=9001 // f + 1, log=log)
        if "3" in exps: logs["실험3"] = exp3(out, log=log, iters=6001 // f + 1)
        if "3b" in exps: logs["실험3b"] = exp3b(out, log=log, iters=10001 // f + 1)
        if "4" in exps: logs["실험4"] = exp4(out, iters=a.iters4 // f + 1, log=log, device=a.device)
        json.dump(logs, open(out / "train_log.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    exps_v = ["1", "2", "3", "3b"] if "all" in a.exp else [e for e in a.exp if e != "all"]
    R = verify(out, a.ref, exps=exps_v)
    print(open(out / "verify_report.md", encoding="utf-8").read())
    if a.figures: make_figures(out, a.data)
    missing = [m for m in R["⓪ 필수 가중치"][0]["누락"].split(", ") if m != "없음"]
    if R["통과"]:
        print("결과: 통과"); sys.exit(0)
    if missing or R["검증 항목 수"] == 0:
        print("결과: 실패 — 가중치 누락 또는 검증 항목 없음:", missing or "검증 항목 0개"); sys.exit(1)
    if a.quick:
        print("결과: 동작 확인 완료 (--quick은 반복 1/20이라 수렴 검증은 하지 않는다. 수치 기준 미달은 정상)"); sys.exit(0)
    print("결과: 실패 — 위 보고서에서 '통과: False' 항목을 확인"); sys.exit(1)


def make_figures(out, data):
    """원본 그림 스크립트를 --out 폴더에서 실행 (P1.pkl·P2.pkl·R3.pkl·R3b.pkl을 읽음)"""
    import subprocess
    env = dict(os.environ, PYTHONPATH=str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""), PINNSET_DATA=data or ".")
    for f, need in [("fig1.py", "P1.pkl"), ("fig2.py", "P2.pkl"), ("fig3.py", "R3.pkl")]:
        if not (Path(out) / need).exists(): print(f"{f}: {need} 없음 → 건너뜀"); continue
        if f == "fig2.py" and not (Path(data or ".") / "labeled_data.csv").exists(): print("fig2.py: --data에 labeled_data.csv 없음 → 건너뜀"); continue
        r = subprocess.run([sys.executable, str(HERE / f)], cwd=out, env=env, capture_output=True, text=True)
        print(f"{f}: {'완료' if r.returncode == 0 else '실패'}"); print(r.stdout[-600:] if r.returncode == 0 else r.stderr[-1500:])


if __name__ == "__main__":
    main()
