"""
[4] 원인별 대처 분석 (층 4) — PINN 성형 윈도우 지도 위 역설계
====================================================================
  C1 shots        CN7 샷 표 + 원인 분류 + 샷별 PINN FLR (학습 가중치 weights/pinn_exp2_map.npz)
  C2 levers       운전점에서 레버 효율: 충진시간·금형온도·용융온도 1단위당 FLR 변화 (PINN 지도 위 수치미분)
  C3 inverse      A 불량 샷을 정상 FLR로 되돌리는 데 필요한 조건 (역설계: 각 레버 하나만 움직일 때)
  C4 speed_window 사출속도 창: A(느림)·B(빠름) 쪽 관측 경계
  C5 restart_rule 재가동 처리 규칙 비교: 고정 폐기 N샷 vs 이탈 점수(D≥5) 기반 분리
  C6 fingerprint  무라벨 재가동에서 '식은 수지 지문'(첫 샷 속도↓·압력↑·다음 샷 회복)과 정지 시간의 관계
그림 5장 + action_results.json 을 작업 폴더에 저장.
"""
import json
import numpy as np, pandas as pd
from scipy.optimize import brentq
from .config import CFG, B_WINDOW
from .data import load_labeled, load_unlabeled, read_unlabeled, shot_table
from .model import reference, select_tier1, dev_scale, deviation, recipe_gate
from .pinn import PINN, WEIGHTS
from . import report  # noqa: F401  한글 폰트 등록 재사용
import matplotlib.pyplot as plt

def shots(path):
    L = load_labeled(path); parts = L[L.PART_NAME.str.startswith("CN7")]
    S = shot_table(parts)
    gmed, gsc = reference(S, CFG["GATE_VARS"]); tmed, tsc = reference(S, CFG["TIER_VARS"])
    U = load_unlabeled(path); T1, _ = select_tier1(U, "CN7", tmed, tsc); ST = shot_table(T1)
    scale = {c: dev_scale(ST[c]) for c in CFG["VARS"]}
    S["D"] = deviation(S, scale, recipe_gate(S, gmed, gsc))
    R = parts.groupby("TimeStamp").Reason.apply(lambda r: "+".join(sorted(set(r.dropna())))).rename("R")
    S = S.merge(R.reset_index(), on="TimeStamp")
    S["since"] = S.groupby("seg").cumcount() + 1
    S["stop"] = S.TimeStamp.diff().dt.total_seconds().where(S.since == 1).ffill()
    b_win = S.TimeStamp.between(*B_WINDOW) & (S.R != "")
    S["cls"] = np.select([S.Tm < 265, S.R == "초기허용불량", b_win, S.R.str.contains("가스"), S.R == ""],
                         ["다른 레시피", "A", "B", "C", "정상"], "U")
    m = PINN(WEIGHTS / "pinn_exp2_map.npz")
    k = S.cls != "다른 레시피"
    S.loc[k, "FLR_pinn"] = [m.flr(r.Tw, r.Filling_Time, r.Tm) for r in S[k].itertuples()]
    return S, m

def analyze(S, m):
    n = S[S.cls == "정상"]; ref = float(n.FLR_pinn.median())
    Tw0, Tm0, tf0 = float(n.Tw.median()), float(n.Tm.median()), float(n.Filling_Time.median())
    f = lambda Tw, tf, Tm: m.flr(Tw, tf, Tm)
    lev = {"충진시간 +1초": (f(Tw0, tf0 + .05, Tm0) - f(Tw0, tf0 - .05, Tm0)) / .1,
           "금형온도 +1℃": f(Tw0 + .5, tf0, Tm0) - f(Tw0 - .5, tf0, Tm0),
           "용융온도 +1℃": (f(Tw0, tf0, Tm0 + 1) - f(Tw0, tf0, Tm0 - 1)) / 2}
    inv = []
    for r in S[S.cls == "A"].itertuples():
        inv.append({"시각": r.TimeStamp.strftime("%m-%d %H:%M"), "재가동 후": int(r.since), "속도": round(r.Max_Injection_Speed, 1),
                    "충진(초)": round(r.Filling_Time, 2), "금형(℃)": round(r.Tw, 1), "PINN FLR": round(r.FLR_pinn, 4),
                    "필요 충진(초)": round(brentq(lambda t: f(r.Tw, t, r.Tm) - ref, 0.2, r.Filling_Time), 2),
                    "필요 금형(℃)": round(brentq(lambda w: f(w, r.Filling_Time, r.Tm) - ref, r.Tw, 85), 1),
                    "필요 용융(℃)": round(brentq(lambda T: f(r.Tw, r.Filling_Time, T) - ref, r.Tm, 700), 1)})
    sp = S.Max_Injection_Speed
    win = {"정상 정속 평균": float(n.Max_Injection_Speed.mean()), "정상 표준편차": float(n.Max_Injection_Speed.std()),
           "양품 관측 최고 속도": float(S.loc[S.cls == "정상", "Max_Injection_Speed"].max()),
           "B 불량 최저 속도": float(sp[S.cls == "B"].min()), "A 불량 최고 속도": float(sp[S.cls == "A"].max()),
           "재가동 양품 최저 속도(10-30)": float(sp[(S.cls == "정상") & (S.since <= 3) & (S.stop < 3600)].min())}
    rr = []
    rs = S[S.since == 1].iloc[1:]
    q = S[S.seg != S.seg.min()]                     # 자료 첫 구간은 실제 재가동인지 알 수 없어 제외 (2026-10-04 정정)
    A_, G_ = (q.cls == "A"), q.cls.isin(["정상", "다른 레시피"])   # 양품 = 불량이 아닌 샷 전부 (E12와 같은 정의)
    for N in [3, 5, 10]:
        k = q.since <= N; rr.append({"규칙": f"정지 후 {N}샷 고정 폐기", "폐기 샷": int(k.sum()), "잡은 A 불량 샷": int((k & A_).sum()),
                                     "잘못 폐기한 양품 샷": int((k & G_).sum())})
    k = (q.since <= 10) & (q.D >= CFG["ALARM"])
    rr.append({"규칙": "이탈 점수 D≥5인 동안만 분리", "폐기 샷": int(k.sum()), "잡은 A 불량 샷": int((k & A_).sum()), "잘못 폐기한 양품 샷": int((k & G_).sum())})
    k = (q.since <= 10) & ((q.D >= CFG["ALARM"]) | q.D.isna())
    rr.append({"규칙": "D≥5 또는 판정 불가(보류)도 분리", "폐기 샷": int(k.sum()), "잡은 A 불량 샷": int((k & A_).sum()), "잘못 폐기한 양품 샷": int((k & G_).sum())})
    flr_tbl = S[S.cls != "다른 레시피"].groupby("cls").FLR_pinn.agg(["count", "median", "min", "max"]).round(4).reset_index()
    return dict(기준=dict(FLR=ref, 금형=Tw0, 용융=Tm0, 충진=tf0, 재가동횟수=int(len(rs))), 레버=lev,
                보상비={"금형(℃/초)": -lev["충진시간 +1초"] / lev["금형온도 +1℃"], "용융(℃/초)": -lev["충진시간 +1초"] / lev["용융온도 +1℃"]},
                역설계=inv, 속도창=win, 재가동규칙=rr, 원인별FLR=flr_tbl.to_dict("records"))

COL = {"정상": "#9aa3ad", "A": "#d6362b", "B": "#f08c00", "C": "#12877b", "U": "#6741d9"}

def fig_map(S, m, R, out):
    tfs = np.linspace(4.0, 8.6, 47); Tws = np.linspace(20, 56, 37)
    M = np.array([[m.flr(w, t, R["기준"]["용융"]) for t in tfs] for w in Tws])
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    cs = ax.contourf(tfs, Tws, M, levels=np.linspace(0.12, 0.26, 29), cmap="Blues")
    cl = ax.contour(tfs, Tws, M, levels=[0.14, 0.16, 0.20, 0.22, 0.24], colors="#0b3d91", linewidths=.7)
    ax.clabel(cl, fmt="%.2f", fontsize=8)
    c0 = ax.contour(tfs, Tws, M, levels=[R["기준"]["FLR"]], colors="#111", linewidths=2.2)
    ax.clabel(c0, fmt=lambda v: f"정상 FLR {v:.3f}", fontsize=8.5, manual=[(7.3, 47)])
    q = S[S.cls == "정상"]; ax.scatter(q.Filling_Time, q.Tw, s=6, c=COL["정상"], label=f"정상 샷 ({len(q)})", zorder=3)
    for c, mk, lab in [("C", "o", "C 캐비티"), ("B", "^", "B 고속")]:
        q = S[S.cls == c]; ax.scatter(q.Filling_Time, q.Tw, s=48, c=COL[c], marker=mk, edgecolor="white", label=f"{lab} ({len(q)})", zorder=4)
    a = S[(S.TimeStamp >= "2020-10-27 00:56") & (S.TimeStamp <= "2020-10-27 01:08")]
    ax.plot(a.Filling_Time, a.Tw, "-", c=COL["A"], lw=1, zorder=4)
    for i, r in enumerate(a.itertuples(), 1):
        ax.scatter(r.Filling_Time, r.Tw, s=70, marker="s", c=COL["A"] if r.cls == "A" else "white", edgecolor=COL["A"], zorder=5)
        ax.annotate(str(i), (r.Filling_Time, r.Tw), xytext=(0, -13), textcoords="offset points", ha="center", fontsize=8.5, color=COL["A"])
    for r in R["역설계"][:3]:
        t = r["충진(초)"]; ax.annotate("", xy=(t, r["필요 금형(℃)"]), xytext=(t, r["금형(℃)"] + 0.8),
                                      arrowprops=dict(arrowstyle="->", color="#c92a2a", lw=1.4, ls="--"))
        ax.text(t - 0.08, r["필요 금형(℃)"] + 0.3, f"{r['필요 금형(℃)']:.0f}℃ 필요", color="#c92a2a", fontsize=8.5, ha="right", bbox=dict(fc="white", ec="none", alpha=.8, pad=1))
    b = S[(S.TimeStamp >= "2020-10-30 02:11") & (S.TimeStamp <= "2020-10-30 02:13")]
    ax.scatter(b.Filling_Time, b.Tw, s=70, marker="D", facecolor="white", edgecolor="#e8590c", lw=1.6, zorder=5, label="10-30 재가동 (양품)")
    ax.scatter([], [], s=70, marker="s", c=COL["A"], label="10-27 재가동 A 불량 (번호=순서)")
    ax.set_xlabel("충진시간 (초)"); ax.set_ylabel("금형온도 (℃)"); ax.set_xlim(4.0, 8.6); ax.set_ylim(20, 56)
    ax.set_title("PINN 성형 윈도우 지도 (색 = 고화층 비율 FLR, 용융온도 275.5℃)", loc="left", fontsize=11)
    ax.legend(loc="upper left", fontsize=8, framealpha=.92); fig.colorbar(cs, ax=ax, label="FLR", shrink=.8)
    fig.savefig(f"{out}/대처_1_PINN지도_역설계.png", dpi=170, bbox_inches="tight"); plt.close(fig)

def fig_levers(R, out):
    r = R["역설계"][1]
    items = [("충진시간", r["충진(초)"], r["필요 충진(초)"], "초"), ("금형온도", r["금형(℃)"], r["필요 금형(℃)"], "℃"), ("용융온도", 275.5, r["필요 용융(℃)"], "℃")]
    fig, ax = plt.subplots(1, 3, figsize=(8.6, 2.9))
    for a, (nm, now, need, u) in zip(ax, items):
        a.bar(["현재", "필요"], [now, need], color=["#9aa3ad", "#1d3b6e" if nm == "충진시간" else "#c92a2a"], width=.6)
        a.set_title(f"{nm}: {need - now:+.1f}{u}", fontsize=10.5); a.set_ylim(0, max(now, need) * 1.18)
        for i, v in enumerate([now, need]): a.text(i, v * 1.02, f"{v:.1f}", ha="center", fontsize=9)
    fig.suptitle(f"A 2번째 샷(충진 {r['충진(초)']}초)을 정상 FLR로 되돌리려면 — 레버 하나씩", x=0.02, ha="left", fontsize=11)
    fig.tight_layout(); fig.savefig(f"{out}/대처_2_레버비교.png", dpi=170, bbox_inches="tight"); plt.close(fig)

def fig_speed(S, R, out):
    rows = [("정상 정속 운전", S[(S.cls == "정상") & (S.D < 5)], COL["정상"], "o"),
            ("10-27 재가동 (598초 정지)", S[(S.TimeStamp >= "2020-10-27 00:56") & (S.TimeStamp <= "2020-10-27 01:12")], None, "s"),
            ("10-30 재가동 (634초 정지)", S[(S.TimeStamp >= "2020-10-30 02:11") & (S.TimeStamp <= "2020-10-30 02:16")], "#e8590c", "D"),
            ("10-20 속도 튐 (가동 중)", S[(S.TimeStamp >= "2020-10-20 06:18") & (S.TimeStamp <= "2020-10-20 06:21")], "#e8590c", "D"),
            ("10-16 B 고속 사건", S[S.cls == "B"], COL["B"], "^")]
    fig, ax = plt.subplots(figsize=(8.4, 3.9)); rng = np.random.default_rng(0)
    w = R["속도창"]
    ax.axvspan(w["양품 관측 최고 속도"], w["B 불량 최저 속도"], color="#ffe8cc", zorder=0)
    ax.axvspan(w["B 불량 최저 속도"], 63, color="#ffd8d8", zorder=0)
    ax.axvline(55.8, c="#555", ls=":", lw=1); ax.text(55.9, 4.55, "기존 권고 상한 55.8", fontsize=8, color="#555")
    for i, (nm, q, c, mk) in enumerate(rows):
        y = len(rows) - 1 - i + (rng.uniform(-.12, .12, len(q)) if i == 0 else np.zeros(len(q)))
        cc = [COL["A"] if cl == "A" else "#9aa3ad" for cl in q.cls] if c is None else c
        ax.scatter(q.Max_Injection_Speed, y, s=14 if i == 0 else 60, c=cc, marker=mk, edgecolor="white" if i else "none", zorder=3)
        if i in (1, 2):
            for j, r in enumerate(q.itertuples(), 1):
                if j <= 5: ax.annotate(str(j), (r.Max_Injection_Speed, len(rows) - 1 - i), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=9)
    ax.set_xlabel("최대 사출속도 (mm/s)"); ax.set_xlim(36, 63)
    ax.text(w["B 불량 최저 속도"] + .15, -0.55, "B 불량 관측", fontsize=8, color="#c92a2a")
    ax.text(w["양품 관측 최고 속도"] - .1, -0.85, "회색지대", fontsize=8, color="#b35c00", ha="right")
    ax.set_title("사출속도 창: 빨간 사각 = A 불량 샷, 번호 = 재가동 후 순서", loc="left", fontsize=11)
    fig.savefig(f"{out}/대처_3_사출속도창.png", dpi=170, bbox_inches="tight"); plt.close(fig)

def fig_restart(R, out):
    rr = pd.DataFrame(R["재가동규칙"])
    fig, ax = plt.subplots(figsize=(7.6, 2.9))
    ax.barh(rr.규칙.str.replace("고정 폐기", "고정 분리"), rr["폐기 샷"], color=["#9aa3ad"] * 3 + ["#1d3b6e", "#6b8fc9"])   # 표시만 '분리'(검사 대기). 결과 키 이름은 호환을 위해 유지
    for i, r in rr.iterrows(): ax.text(r["폐기 샷"] + 1.5, i, f"분리 {r['폐기 샷']}샷 · A 불량 {r['잡은 A 불량 샷']}/5, 양품 {r['잘못 폐기한 양품 샷']}", va="center", fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, rr["폐기 샷"].max() * 1.6); ax.set_xlabel("분리 샷 수 — CN7 라벨 기간 재가동 %d회 (자료 첫 구간 제외)" % R["기준"]["재가동횟수"])
    ax.set_title("재가동 처리 규칙 비교 (분리 = 폐기가 아니라 검사 대기)", loc="left", fontsize=11)
    fig.savefig(f"{out}/대처_4_재가동규칙.png", dpi=170, bbox_inches="tight"); plt.close(fig)


# ============================================================ C6 재가동 '식은 수지 지문' (라벨 없는 데이터로 메커니즘 검증)
# 지문 = 같은 작업오더 안에서 300초 넘게 멈춘 뒤 첫 샷의 사출속도가 오더 중앙값보다 5% 이상 낮고, 사출압력은 1% 이상 높으며,
#        다음 샷에서 속도 저하가 절반 이하로 회복. (속도·압력이 같이 내려가면 설정 변경으로 보고 제외)
def restart_fingerprint(path):
    from scipy.stats import spearmanr, fisher_exact
    U = read_unlabeled(path, ["TimeStamp", "EQUIP_NAME", "PART_NAME", "PART_FACT_PLAN_DATE",
                              "PART_FACT_SERIAL", "Max_Injection_Speed", "Max_Injection_Pressure"])
    rows = []
    for eq, u in U.groupby("EQUIP_NAME"):
        s = u.sort_values("TimeStamp").drop_duplicates("TimeStamp")
        s = s.assign(lot=s.PART_FACT_PLAN_DATE.astype(str) + "_" + s.PART_FACT_SERIAL.astype(str) + "_" + s.PART_NAME)
        s = s[s.groupby("lot").Max_Injection_Speed.transform("size") >= 30].reset_index(drop=True)
        gap = s.TimeStamp.diff().dt.total_seconds().values; lot = s.lot.values
        vr = (s.Max_Injection_Speed / s.groupby("lot").Max_Injection_Speed.transform("median")).values
        pr = (s.Max_Injection_Pressure / s.groupby("lot").Max_Injection_Pressure.transform("median")).values
        for i in np.where((pd.Series(lot).eq(pd.Series(lot).shift())) & (gap > 300))[0]:
            nx = [j for j in range(i + 1, min(i + 6, len(s))) if lot[j] == lot[i]]
            v2 = 1 - vr[i + 1] if nx and nx[0] == i + 1 else np.nan
            rows.append(dict(설비=eq, 정지초=gap[i], 속도저하=1 - vr[i], 둘째샷저하=v2, 압력비=pr[i],
                             재정지=int(any(gap[j] > 120 for j in nx))))
    R = pd.DataFrame(rows); R = R[R.속도저하.abs() < 0.9]
    R["지문"] = (R.속도저하 > 0.05) & (R.압력비 > 1.01) & (R.둘째샷저하 < R.속도저하 * 0.5)
    R["정지"] = pd.cut(R.정지초, [300, 600, 900, 1800, 3600, 6 * 3600, 1e9], labels=["5-10분", "10-15분", "15-30분", "30-60분", "1-6시간", "6시간+"])
    tb = R.groupby("정지", observed=True).지문.agg(재가동="size", 지문="sum", 비율="mean").reset_index()
    rho, p = spearmanr(np.log10(R.정지초), R.지문)
    ct = pd.crosstab(R.재정지, R.지문); odds, pf = fisher_exact(ct.values)
    return dict(재가동=int(len(R)), 설비=int(R.설비.nunique()), 지문=int(R.지문.sum()), 정지별=tb.to_dict("records"),
                정지길이_spearman=[float(rho), float(p)], 재정지_fisher=[float(odds), float(pf)],
                지문_재정지있음=int(ct.loc[1, True]) if (1 in ct.index and True in ct.columns) else 0,
                지문_재정지없음=int(ct.loc[0, True]) if (0 in ct.index and True in ct.columns) else 0)

def labeled_restarts(S):
    """CN7 라벨 기간 재가동(300초+ 정지) 첫 샷의 속도 이탈과 불량"""
    n = S[(S.cls == "정상") & (S.since > 20)]; med = n.Max_Injection_Speed.median(); sc = 1.4826 * (n.Max_Injection_Speed - med).abs().median()
    f = S[S.since == 1].iloc[1:]
    return [dict(재가동=r.TimeStamp.strftime("%m-%d %H:%M"), 정지초=int(r.stop), 첫샷_속도z=round((r.Max_Injection_Speed - med) / sc, 1),
                 불량=r.cls) for r in f.itertuples() if r.cls != "다른 레시피"]

def fig_fingerprint(F, out):
    tb = pd.DataFrame(F["정지별"])
    fig, ax = plt.subplots(figsize=(7.6, 3.0))
    ax.bar(tb.정지.astype(str), tb.비율 * 100, color=["#c92a2a" if i < 2 else "#9aa3ad" for i in range(len(tb))])
    for i, r in tb.iterrows(): ax.text(i, r.비율 * 100 + 0.08, f"{int(r.지문)}/{int(r.재가동)}", ha="center", fontsize=8.5)
    ax.set_ylabel("식은 수지 지문 비율 (%)"); ax.set_xlabel("재가동 전 정지 시간")
    ax.set_title(f"무라벨 재가동 {F['재가동']:,}회 (설비 {F['설비']}대): 짧게 멈출수록 첫 샷에 식은 수지가 나온다", loc="left", fontsize=10.5)
    ax.set_ylim(0, tb.비율.max() * 100 * 1.35)
    fig.savefig(f"{out}/대처_5_재가동지문.png", dpi=170, bbox_inches="tight"); plt.close(fig)

def run(path, out="."):
    S, m = shots(path); R = analyze(S, m)
    R["라벨_재가동"] = labeled_restarts(S); R["재가동_지문"] = restart_fingerprint(path)
    fig_map(S, m, R, out); fig_levers(R, out); fig_speed(S, R, out); fig_restart(R, out); fig_fingerprint(R["재가동_지문"], out)
    json.dump(R, open(f"{out}/action_results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return R
