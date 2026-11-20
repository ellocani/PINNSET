"""PINNSET 현장 대시보드 생성 (재생 모드 · 실시간 모드 · CSV 불러오기)

    python run_dashboard.py --data <KAMP 데이터 폴더> --results <결과 폴더>

· 공식 성능과 같은 엄격 시간 순서(forward, E12) 조건을 그대로 다시 돌려, 각 샷 시점에 모델이 실제로 알 수 있던
  정보만으로 낸 출력(판정·확률·위험점수·원인태그·변수별 이탈 z·기준값)을 샷 순서대로 모은다.
  - 폴드마다 evaluate.evaluate_product(forward)와 똑같이 기준(fit_reference)과 보정층(fit_calibration)을 다시 맞추고
    RiskModel.predict_stream으로 스트림 출력을 받는다. (strict 코드의 parity 검사와 같은 방식)
  - 결과를 results/strict/predictions_forward_{CN7,RG3}.csv(공식값)와 맞춰 보고, 하나라도 다르면 화면을 만들지 않는다.
· 결과: <결과 폴더>/PINNSET_대시보드.html 파일 하나 (데이터·스크립트 내장, 외부 라이브러리·인터넷 불필요)
· 모델 코드·가중치·기존 결과 숫자는 바꾸지 않는다. 결과 폴더를 읽어 HTML만 새로 만든다.
· CSV 불러오기 모드: 사용자가 넣은 KAMP 형식 CSV를 화면 안의 같은 판정 엔진(RiskModel.step 이식)으로 처리한다.
  과거 파일은 품목별 시간순 정렬 → 같은 시각 행을 한 샷으로 묶어 판정, 실시간 감시는 같은 파일을 주기적으로 다시 확인해 새 행만 판정.
  시연용 늘어나는 CSV: tools/csv_feed_sim.py
· 재생·실시간 모드는 시연 화면이며 설비에 연결된 실시간 시스템이 아니다. 모델이 이미 샷 순서 처리 구조(predict_stream)라 PLC·MES 연결 시 같은 화면을 실시간으로 쓸 수 있다.
"""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from .config import CFG
from .data import load_labeled, load_unlabeled, shot_table, pairs_for
from .model import fit_calibration, prepare_product
from . import evaluate as st

VARS = list(CFG["VARS"])
ZCOL = ["z_speed", "z_fill", "z_inj", "z_flr"]
KEY = ["TimeStamp", "PART_NAME"]
EXPECT = {"CN7": dict(score=3454, held=145, unavailable=375, alarm_parts=20, alarm_defects=10),
          "RG3": dict(score=1022, held=24, unavailable=210)}


def log(s):
    print(s, flush=True)


# ------------------------------------------------------------------ 1. 엄격 forward 재계산
def recompute(prod, L, U):
    parts = L[L.PART_NAME.str.startswith(prod)].copy()
    S = shot_table(parts)
    parts = parts.merge(S[["TimeStamp", "seg"]], on="TimeStamp", validate="many_to_one").sort_values(KEY).reset_index(drop=True)
    U = U[(U.EQUIP_NAME == CFG["EQUIP"]) & U.PART_NAME.str.startswith(prod)]
    rows = []
    for seg in sorted(S.seg.unique()):
        testS = S[S.seg == seg]
        start, end = testS.TimeStamp.min(), testS.TimeStamp.max()
        trS, pool = S[S.TimeStamp < start], U[U.TimeStamp < start]
        train = parts[parts.TimeStamp < start].copy()
        m, _ = st.fit_reference(prod, trS, pool)
        test_ts = set(testS.TimeStamp)
        if m is None:                       # 과거 기록 없음 → 기준도 없음
            continue
        trainD = st.deviations(trS, m)
        train = train.merge(trS[["TimeStamp"]].assign(D=trainD), on="TimeStamp", validate="many_to_one")
        train = train[np.isfinite(train.D)].copy(); train["gD"] = np.log1p(train.D)
        ready = len(train) > 0 and 0 < train.y.sum() < len(train)
        if ready:
            b, Lc, spec = fit_calibration(train, pairs_for(train), dict(CFG["MODEL"][prod]))
        else:                               # 보정층 없음: 이탈 z만 보기 위해 임시 계수로 돌리고 확률은 버린다
            b, Lc, spec = np.zeros(1 + int(m.spec["process"])), 0.0, dict(m.spec)
        m.b, m.Lc, m.used = b, Lc, spec
        prefixS = S[S.TimeStamp <= end]
        stream = m.predict_stream(prefixS, parts[parts.TimeStamp <= end])
        stream = stream[stream.TimeStamp.isin(test_ts)].copy()
        if not ready:
            stream[["불량확률", "위험점수"]] = np.nan
        stream["calib_ready"] = ready
        sc = pd.Series(m.scale)
        stream["fold_scale"] = [sc.values.tolist()] * len(stream)
        rows.append(stream)
    R = pd.concat(rows, ignore_index=True)
    return parts, S, R


def merge_and_check(prod, parts, S, R, strict_csv):
    E = pd.read_csv(strict_csv, parse_dates=["TimeStamp"])
    E = E[KEY + ["seg", "block", "status", "D", "PINNSET", "y", "RH", "Reason"]]
    M = E.merge(R.drop(columns=["seg"]), on=KEY, how="left", validate="one_to_one")
    # ① status 일치
    st_stream = np.where(M.판정.eq("보류"), "held", np.where(M.판정.eq("점수"), "score", "unavailable"))
    st_stream = np.where(M.calib_ready.fillna(False).astype(bool) | M.판정.eq("보류"), st_stream, "unavailable")
    # 보정층이 없는 구간은 strict에서 모두 unavailable (보류 여부와 무관)
    st_stream = np.where(M.calib_ready.fillna(False).astype(bool), st_stream, "unavailable")
    assert (st_stream == M.status.values).all(), f"{prod}: status 불일치 {int((st_stream != M.status.values).sum())}"
    # ② score 부품의 D·확률
    k = M.status.eq("score")
    dD = float(np.max(np.abs(M.D[k] - M.이탈점수[k]))); dP = float(np.max(np.abs(M.PINNSET[k] - M.불량확률[k])))
    assert dD < 1e-9 and dP < 1e-9, f"{prod}: D {dD}, P {dP}"
    # ③④ 개수
    cnt = M.status.value_counts().to_dict()
    ex = EXPECT[prod]
    for s in ["score", "held", "unavailable"]:
        assert cnt.get(s, 0) == ex[s], f"{prod} {s}: {cnt.get(s, 0)} != {ex[s]}"
    if prod == "CN7":
        al = k & M.D.ge(CFG["ALARM"])
        assert al.sum() == ex["alarm_parts"] and M.y[al].sum() == ex["alarm_defects"], (al.sum(), M.y[al].sum())
    log(f"  {prod} 검증 통과: status 일치, D 최대차 {dD:.1e}, 확률 최대차 {dP:.1e}, {cnt}")
    return M, dict(D_max_diff=dD, P_max_diff=dP, counts=cnt)


# ------------------------------------------------------------------ 2. 화면 데이터
def shot_payload(prod, M, S):
    S = S.set_index("TimeStamp")
    shots = []
    for ts, g in M.sort_values(KEY).groupby("TimeStamp", sort=True):
        g = g.sort_values("RH")
        r0 = g.iloc[0]
        x = [float(S.at[ts, c]) for c in VARS]
        z = [None if pd.isna(r0[c]) else float(r0[c]) for c in ZCOL] if r0.get("판정") == "점수" else None
        sc = r0["fold_scale"] if isinstance(r0["fold_scale"], list) else None
        base = [x[j] - z[j] * sc[j] for j in range(4)] if (z is not None and sc is not None) else None
        reason = r0["보류사유"] if isinstance(r0.get("보류사유"), str) and r0["보류사유"] != "-" else ""
        D = None if pd.isna(r0.D) else float(r0.D)
        Dstream = None if pd.isna(r0.get("이탈점수")) else float(r0["이탈점수"])
        ps = []
        for _, r in g.iterrows():
            ps.append(dict(n=r.PART_NAME[-2:], y=int(r.y),
                           p=None if pd.isna(r.PINNSET) else round(float(r.PINNSET), 6),
                           pct=None if pd.isna(r.get("위험점수")) or r.status != "score" else round(float(r["위험점수"]), 2),
                           tag=(r["원인태그"] if isinstance(r.get("원인태그"), str) and r.status == "score" else "-"),
                           rs=str(r.Reason) if isinstance(r.Reason, str) else ""))
        shots.append(dict(t=str(ts)[5:16].replace("-", "/"), T=str(ts), seg=int(r0.seg), s=r0.status,
                          D=None if D is None else round(D, 4),
                          Dv=None if Dstream is None else round(Dstream, 4),  # 보정층 없는 구간 참고용 이탈(시연 표시만)
                          x=[round(v, 4) for v in x], z=None if z is None else [round(v, 4) for v in z],
                          m=None if base is None else [round(v, 4) for v in base],
                          sc=None if sc is None else [round(v, 5) for v in sc],
                          rsn=reason, P=ps))
    return shots


def counters_check(prod, shots):
    """대기열 규칙(4.2절)으로 끝까지 재생했을 때의 누적값 → 운영표와 같아야 한다."""
    out = {}
    def tally(sel):
        n = ins = d = tp = 0
        for s in sel:
            for p in s["P"]:
                n += 1; d += p["y"]
                flag = s["s"] != "score" or (s["D"] >= 5 if prod == "CN7" else p["n"] == "RH")
                ins += flag; tp += flag and p["y"]
        return dict(n=n, inspected=ins, rate=ins / n, defects=d, caught=tp)
    out["all"] = tally(shots)
    out["after_calib"] = tally([s for s in shots if s["s"] != "unavailable"])
    out["scored"] = tally([s for s in shots if s["s"] == "score"])
    return out


# ------------------------------------------------------------------ 3. 실시간 모드 (무라벨 실데이터 입력 → 브라우저 안에서 샷마다 판정)
# 입력 구간: 정상 기준(1층 작업오더)에 쓰지 않은 작업오더이고, 라벨 기간보다 앞선 시기라 라벨과 겹치지 않는 연속 1,200샷.
#            재가동이 들어 있는 구간을 골랐다(시연용 선택). 무라벨이므로 성능 평가가 아니라 처리 시연이다.
LIVE = {"CN7": ("2020-09-07 03:10", 1200), "RG3": ("2020-03-29 13:32", 1200)}
LIVE_MODEL_EXPECT = {"CN7": dict(b=[-6.91, 2.59], Lc=0.0), "RG3": dict(b=[-4.07], Lc=0.74)}   # 보고서 수식과 같은 최종 모델


def final_model(prod, L, U):
    """현장 투입용 최종 모델 = analysis.run_v1의 전체 학습 모델과 같은 절차."""
    d = prepare_product(prod, L, U)
    m, parts, T1 = d["m"], d["parts"], d["T1"]
    m.fit_calibration(d["P"], d["pairs"])
    ex = LIVE_MODEL_EXPECT[prod]
    assert np.allclose(m.b, ex["b"], atol=.006) and abs(m.Lc - ex["Lc"]) < .006, (m.b, m.Lc)
    return m, parts, T1


def live_payload(prod, L, U, injections):
    m, parts, T1 = final_model(prod, L, U)
    u = U[(U.EQUIP_NAME == CFG["EQUIP"]) & U.PART_NAME.str.startswith(prod)].drop_duplicates(KEY).copy()
    u["RH"] = u.PART_NAME.str.endswith("RH").astype(int)
    SU = shot_table(u)
    start, n = LIVE[prod]
    W = SU[SU.TimeStamp >= pd.Timestamp(start)].head(n).reset_index(drop=True)
    assert len(W) == n
    assert not W.TimeStamp.isin(set(T1.TimeStamp)).any(), "입력 구간이 정상 기준 계산에 쓰였다"
    assert W.TimeStamp.max() < parts.TimeStamp.min(), "입력 구간이 라벨 기간과 겹친다"
    uw = u[u.TimeStamp.isin(set(W.TimeStamp))]
    o = m.predict_stream(W, uw)                       # Python 원본 계산 (브라우저 계산의 정답지)
    G = CFG["GATE_VARS"]
    shots, exp = [], []
    og = {t: g for t, g in o.groupby("TimeStamp", sort=False)}
    for r in W.itertuples():
        g = og[r.TimeStamp]
        shots.append(dict(T=str(r.TimeStamp), lot=r.lot, x=[float(getattr(r, c)) for c in VARS], g=[float(getattr(r, c)) for c in G],
                          P=[[n[-2:], int(n.endswith("RH"))] for n in g.PART_NAME]))
        if (g.판정 == "보류").all():
            exp.append(dict(r=g.보류사유.iloc[0], D=None))
        else:
            exp.append(dict(r="", D=float(g.이탈점수.iloc[0]), P=[[float(a), float(b), t] for a, b, t in zip(g.불량확률, g.위험점수, g.원인태그)]))
    cnt = o.drop_duplicates("TimeStamp").판정.value_counts().to_dict()
    al = o.drop_duplicates("TimeStamp").이탈점수.ge(CFG["ALARM"]).sum()
    log(f"  {prod} 실시간 입력 {W.TimeStamp.min()} ~ {W.TimeStamp.max()} {n}샷, 판정 {cnt}, 경보 {int(al)}샷 (정상 기준 미사용 작업오더, 라벨 기간 이전)")
    model = dict(gmed=[float(m.gmed[c]) for c in G], gsc=[float(m.gsc[c]) for c in G], scale=[float(m.scale[c]) for c in VARS],
                 b=[float(v) for v in m.b], Lc=float(m.Lc), process=bool(m.used["process"]))
    inj = dict(injections); inj["recipe_dTm"] = round(8 * model["gsc"][0], 1)
    return dict(model=model, shots=shots, exp=exp, inj=inj,
                range=[str(W.TimeStamp.min())[:16], str(W.TimeStamp.max())[:16]], counts={k: int(v) for k, v in cnt.items()}, alarms=int(al))


def injection_ratios(cn7_shots):
    """모의 주입용 변화 비율 = 실제 사건 샷의 (현재값 ÷ 평소값). 숫자를 지어내지 않고 실제 사건에서 가져온다."""
    by = {s["T"]: s for s in cn7_shots}
    def ratios(ts):
        return [[round(x / m, 5) for x, m in zip(by[t]["x"], by[t]["m"])] for t in ts]
    A = ["2020-10-27 00:56:21", "2020-10-27 00:59:00", "2020-10-27 01:03:34", "2020-10-27 01:04:41", "2020-10-27 01:05:41"]
    B = ["2020-10-20 06:18:16", "2020-10-20 06:19:14", "2020-10-20 06:20:14"]
    return dict(A=ratios(A), B=ratios(B))


def load_cards(results):
    """요청 엔진이 만든 데이터 요청서 html에서 카드 문구를 그대로 가져온다 (문구를 새로 쓰지 않는다)."""
    html = (results / "PINNSET_데이터요청서.html").read_text(encoding="utf-8")
    strip = lambda t: re.sub(r"<[^>]+>", "", t).strip()
    cards = {}
    for cls, title, cause, body in re.findall(r"<div class='card'><div class='hd([^']*)'>(.*?) <span style='float:right'>(.*?)</span></div><table>(.*?)</table></div>", html, re.S):
        num = re.match(r"#(\d+)", title).group(1)
        cards[num] = dict(title=strip(title), cause=strip(cause), kind=cls.strip() or "n",
                          rows=[[strip(k), strip(v)] for k, v in re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td></tr>", body)])
    plan = [[strip(k), strip(v)] for k, v in re.findall(r"<p><b>(.*?)</b>: (.*?)</p>", html)]
    assert len(cards) == 14 and len(plan) >= 4, (len(cards), len(plan))
    return dict(cards=cards, plan=plan)


def action_numbers(action):
    rr = {r["규칙"]: r for r in action["재가동규칙"]}
    f5 = rr["정지 후 5샷 고정 폐기"]; dr = rr["이탈 점수 D≥5인 동안만 분리"]; dh = rr["D≥5 또는 판정 불가(보류)도 분리"]
    inv = action["역설계"][0]; sp = action["속도창"]
    return dict(fix5=f5["폐기 샷"], fix5_good=f5["잘못 폐기한 양품 샷"], drule=dr["폐기 샷"], drule_A=dr["잡은 A 불량 샷"],
                dheld=dh["폐기 샷"], n_restart=action["기준"]["재가동횟수"], mold_need=inv["필요 금형(℃)"], mold_now=inv["금형(℃)"],
                mold_up=round(inv["필요 금형(℃)"] - inv["금형(℃)"]), melt=round(action["기준"]["용융"]),
                v_mean=round(sp["정상 정속 평균"], 1), v_sd=round(sp["정상 표준편차"], 1),
                v_good_max=round(sp["양품 관측 최고 속도"], 1), v_B_min=round(sp["B 불량 최저 속도"], 1))


def render(payload, checks, cards, action, live):
    cfg = {k: CFG[k] for k in ["ALARM", "TAG_Z", "GATE_DIST", "REF_MIN", "REF_WIN", "REBASE_RUN", "PCT_WINDOW", "EQUIP", "ALPHA", "T_NF", "HALF_THICK"]}
    js = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    html = TEMPLATE
    for k, v in [("__DATA__", js(payload)), ("__CARDS__", js(cards)), ("__ACT__", js(action_numbers(action))),
                 ("__CFG__", js(cfg)), ("__LIVE__", js(live)), ("__CHECKS__", js({p: dict(D_max_diff=c["D_max_diff"], P_max_diff=c["P_max_diff"]) for p, c in checks.items()}))]:
        html = html.replace(k, v)
    return html


TEMPLATE = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PINNSET 현장 화면</title>
<style>
:root{--navy:#1D3B6E;--navy2:#2E5590;--bg:#F4F6F9;--line:#D5DAE3;--ink:#1D2330;--sub:#5A6475;--red:#C0392B;--redbg:#FBEAE8;
--amber:#B9770E;--amberbg:#FDF3E1;--gray:#6B7A90;--graybg:#ECEFF3;--ok:#1E7D5A;--okbg:#E6F4EE;--card:#fff}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:'Malgun Gothic','맑은 고딕','Apple SD Gothic Neo','Noto Sans KR','Noto Sans CJK KR','NanumGothic',sans-serif;font-size:13px;line-height:1.45}
header{background:var(--navy);color:#fff;padding:10px 18px;display:flex;align-items:center;gap:16px;flex-wrap:wrap}
header h1{font-size:18px;margin:0;font-weight:700;letter-spacing:-.2px}
header .demo{font-size:12px;background:rgba(255,255,255,.14);padding:3px 9px;border-radius:12px}
header .prod{margin-left:auto;display:flex;gap:4px}
.seg-btn{border:1px solid rgba(255,255,255,.5);background:transparent;color:#fff;padding:5px 14px;border-radius:6px;font:inherit;font-weight:700;cursor:pointer}
.seg-btn.on{background:#fff;color:var(--navy)}
.modes{display:flex;background:rgba(255,255,255,.12);border-radius:8px;padding:3px;gap:3px}
.mode-btn{border:0;background:transparent;color:#fff;padding:6px 14px;border-radius:6px;font:inherit;font-weight:700;cursor:pointer}
.mode-btn.on{background:#fff;color:var(--navy)}
.livedot{display:inline-block;width:8px;height:8px;border-radius:50%;background:#2ECC71;margin-right:6px;animation:blink 1s infinite}
@keyframes blink{50%{opacity:.25}}
.inj button{border-color:var(--red);color:var(--red)}.inj button:hover{background:var(--red);color:#fff}
.injflag{background:#F3E8FA;color:#6C3483;border:1px solid #D7BDE2}
.hide{display:none!important}
.ctrl{background:#fff;border-bottom:1px solid var(--line);padding:8px 18px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
.ctrl button{font:inherit;cursor:pointer;border:1px solid var(--line);background:#fff;border-radius:6px;padding:5px 10px;color:var(--ink)}
.ctrl button.on{background:var(--navy);color:#fff;border-color:var(--navy)}
.ctrl .play{background:var(--navy);color:#fff;border-color:var(--navy);min-width:84px;font-weight:700}
.ctrl input[type=range]{flex:1;min-width:200px;accent-color:var(--navy)}
.ctrl .lbl{color:var(--sub);font-size:12px}
.ev{display:flex;gap:6px;flex-wrap:wrap;padding:6px 18px 8px;background:#fff;border-bottom:1px solid var(--line);align-items:center}
.ev button{font:inherit;font-size:12px;cursor:pointer;border:1px solid var(--navy);color:var(--navy);background:#fff;border-radius:14px;padding:3px 11px}
.ev button:hover{background:var(--navy);color:#fff}
main{display:grid;grid-template-columns:minmax(0,2.15fr) minmax(0,1fr);gap:12px;padding:12px 18px 24px}
.col{display:flex;flex-direction:column;gap:12px;min-width:0}
.pane{background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden}
.pane>h2{margin:0;font-size:13px;color:#fff;background:var(--navy);padding:6px 12px;font-weight:700;display:flex;justify-content:space-between;align-items:center}
.pane>h2 small{font-weight:400;opacity:.85;font-size:11px}
.pb{padding:10px 12px}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
/* ① status */
.status{display:grid;grid-template-columns:200px 1fr;gap:12px;align-items:stretch}
.verdict{border-radius:8px;padding:12px;display:flex;flex-direction:column;justify-content:center;align-items:flex-start}
.verdict .big{font-size:30px;font-weight:800;letter-spacing:-1px}
.verdict .why{font-size:12px;margin-top:4px}
.v-ok{background:var(--okbg);color:var(--ok)}.v-alarm{background:var(--red);color:#fff}.v-held{background:var(--amberbg);color:var(--amber)}.v-na{background:var(--graybg);color:var(--gray)}
.meta{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--sub);margin-bottom:6px}
.meta b{color:var(--ink)}
.parts{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.part{border:1px solid var(--line);border-radius:8px;padding:8px 10px}
.part .ph{display:flex;justify-content:space-between;align-items:center;font-weight:700;margin-bottom:4px}
.kv{display:grid;grid-template-columns:78px 1fr;row-gap:2px;font-size:12px}
.kv>span:nth-child(odd){color:var(--sub)}
.num{font-variant-numeric:tabular-nums}
.badge{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;font-weight:700;white-space:nowrap}
.b-red{background:var(--red);color:#fff}.b-amber{background:var(--amberbg);color:var(--amber);border:1px solid #E8C788}.b-gray{background:var(--graybg);color:var(--gray)}.b-ok{background:var(--okbg);color:var(--ok)}.b-navy{background:#E5EBF5;color:var(--navy)}
.frozen{margin-top:8px;font-size:11px;color:var(--sub);border-top:1px dashed var(--line);padding-top:6px}
/* ② */
table.t{border-collapse:collapse;width:100%;font-size:12px;margin-top:6px}
table.t th{background:var(--navy);color:#fff;font-weight:700;padding:4px 6px;text-align:right;font-size:11px}
table.t th:first-child,table.t td:first-child{text-align:left}
table.t td{padding:4px 6px;border-bottom:1px solid #E6E9EF;text-align:right}
table.t tr:nth-child(even) td{background:var(--bg)}
.explain{margin-top:8px;padding:7px 9px;border-left:3px solid var(--navy);background:#F7F9FC;font-size:12px}
.explain.red{border-color:var(--red);background:var(--redbg)}
.tagline{font-size:14px;font-weight:700;margin-bottom:4px}
/* ③ */
.act ol{margin:4px 0 0;padding-left:18px}.act li{margin:3px 0}
.act .head{font-weight:700;font-size:14px;margin-bottom:2px}
.note{font-size:11px;color:var(--sub);margin-top:6px}
/* ④ */
canvas{width:100%;display:block}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--sub);padding:0 12px 8px}
.sw{display:inline-block;width:12px;height:10px;vertical-align:-1px;margin-right:4px;border-radius:2px}
/* ⑤ */
.cnt{width:100%;border-collapse:collapse;font-size:12px}
.cnt th{font-size:11px;color:var(--sub);font-weight:400;text-align:right;padding:2px 4px;border-bottom:1px solid var(--line)}
.cnt td{text-align:right;padding:4px;border-bottom:1px solid #EEF0F4}
.cnt th:first-child,.cnt td:first-child{text-align:left}
.q{list-style:none;margin:6px 0 0;padding:0}
.q li{display:grid;grid-template-columns:auto 1fr auto;gap:8px;align-items:center;padding:4px 6px;border-bottom:1px solid #EEF0F4;font-size:12px}
.q li .tm{color:var(--sub);font-variant-numeric:tabular-nums}
.qh{font-size:12px;font-weight:700;margin-top:10px;color:var(--navy)}
.more{font-size:11px;color:var(--sub);padding:3px 6px}
/* ⑥ */
.banner{padding:7px 9px;border-radius:6px;font-size:12px;margin-bottom:8px}
.banner.red{background:var(--redbg);color:var(--red);border:1px solid #EBC2BD}
.banner.info{background:#EEF2F8;color:var(--navy)}
.rc{border:1px solid var(--line);border-radius:8px;margin:7px 0;overflow:hidden}
.rc .rh{background:var(--navy);color:#fff;padding:5px 9px;font-weight:700;font-size:12px;display:flex;justify-content:space-between;gap:6px}
.rc .rh.q{background:var(--gray)}.rc .rh.f{background:#9AA3AD}
.rc .rb{padding:6px 9px;font-size:12px}
.rc .rb div{margin:2px 0}.rc .rb b{color:var(--sub);font-weight:700;margin-right:4px}
.rc details{font-size:11px;color:var(--sub);margin-top:3px}.rc details summary{cursor:pointer}
.plan{font-size:11.5px;margin-top:8px;border-top:1px dashed var(--line);padding-top:6px}
.plan div{margin:2px 0}.plan b{color:var(--navy)}
.fintro{margin:14px 18px 24px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:18px 20px}
.fintro h3{margin:0 0 4px;font-size:16px;color:var(--navy)}
.drop{border:2px dashed #AAB4C3;border-radius:10px;padding:26px 16px;text-align:center;color:var(--sub);cursor:pointer;margin:12px 0;font-size:14px}
.drop b{color:var(--navy);font-size:15px}.drop.over{border-color:var(--navy);background:#EEF2F8;color:var(--navy)}
.fgrid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.fcard{border:1px solid var(--line);border-radius:8px;padding:10px 12px;font-size:12.5px}
.fcard h4{margin:0 0 4px;font-size:13px;color:var(--navy)}.fcard ul{margin:4px 0 0;padding-left:18px}.fcard li{margin:2px 0}
.cols{font-size:11.5px;color:var(--sub);margin-top:10px;line-height:1.6}.cols code{background:var(--bg);border:1px solid var(--line);border-radius:4px;padding:0 4px;color:var(--ink);font-size:11px}
.fbar{display:inline-block;width:220px;height:8px;background:var(--graybg);border-radius:4px;overflow:hidden;vertical-align:middle}.fbar i{display:block;height:100%;width:0;background:var(--navy)}
.fstat{font-size:12px}.fstat.bad{color:var(--red);font-weight:700}
.ctrl select{font:inherit;border:1px solid var(--line);border-radius:6px;padding:4px 6px;background:#fff}
.ctrl label{display:inline-flex;align-items:center;gap:4px;cursor:pointer}
.ctrl button.on{background:var(--ok);border-color:var(--ok);color:#fff}
@media (max-width:760px){.fgrid{grid-template-columns:1fr}}
.foot{grid-column:1/-1;font-size:11px;color:var(--sub);padding:4px 2px}
@media (max-width:1100px){main{grid-template-columns:1fr}.row2{grid-template-columns:1fr}}
@media (max-width:640px){.status{grid-template-columns:1fr}.parts{grid-template-columns:1fr}header,.ctrl,.ev{padding-left:12px;padding-right:12px}main{padding:10px 12px}}
</style></head><body>
<header>
  <h1>PINNSET 현장 화면</h1>
  <div class="modes"><button class="mode-btn on" data-m="replay">재생 모드</button><button class="mode-btn" data-m="live">실시간 모드</button><button class="mode-btn" data-m="file">CSV 불러오기</button></div>
  <span class="demo" id="demo"></span>
  <div class="prod"><button class="seg-btn on" data-p="CN7">CN7</button><button class="seg-btn" data-p="RG3">RG3</button></div>
</header>
<div class="ctrl">
  <button class="play" id="play">▶ 재생</button>
  <span class="lbl" id="spdl">속도</span><span id="spd"><button data-s="1">1샷/초</button><button data-s="10" class="on">10샷/초</button><button data-s="100">100샷/초</button></span>
  <span id="rctl" style="display:contents"><button id="prev" title="한 샷 뒤로">◀ 1샷</button><button id="next" title="한 샷 앞으로">1샷 ▶</button>
  <input type="range" id="slider" min="0" max="10" value="0">
  <span class="lbl num" id="pos"></span>
  <span class="lbl" id="dlyl">검사 결과 회신 지연 가정 <b id="dly">10</b>샷</span></span>
  <span id="fctl" class="hide" style="display:contents"><button id="fpick">CSV 열기 (한 번에 처리)</button><button id="fwatch" title="Chrome·Edge: 같은 파일에 새 행이 붙을 때마다 바로 판정">실시간 감시 시작</button><button id="fstop" class="hide">감시 중지</button><button id="fsave" class="hide" title="CN7·RG3 판정 결과를 부품 단위 CSV로 저장 (엑셀에서 바로 열림)">결과 CSV 저장</button>
  <span class="lbl">확인 주기</span><select id="fint"><option value="1000" selected>1초</option><option value="2000">2초</option><option value="5000">5초</option></select>
  <label class="lbl"><input type="checkbox" id="ffollow" checked>최신 샷 따라가기</label>
  <label class="lbl" title="EQUIP_NAME이 학습 설비(650톤-우진2호기)가 아니어도 넣는다. 설정이 다르면 적용 범위 문에서 보류된다."><input type="checkbox" id="fany">설비명 무시</label>
  <input type="file" id="ffile" accept=".csv,.txt,text/csv" class="hide"></span>
  <span id="lctl" class="hide" style="flex:1;display:flex;gap:12px;align-items:center;flex-wrap:wrap"><button id="lreset">처음부터</button><span class="lbl num" id="lpos"></span></span>
</div>
<div class="ctrl hide" id="fline" style="padding-top:4px;padding-bottom:6px"><span id="fstat" class="lbl fstat"></span><span id="fbar" class="fbar hide"><i></i></span></div>
<div class="ev hide" id="fev"></div>
<div class="ev" id="events"><span class="lbl" style="color:var(--sub);font-size:12px">사건 바로 가기</span></div>
<div class="ev inj hide" id="inj"><span class="lbl" style="color:var(--sub);font-size:12px">모의 사건 주입 (다음 샷부터, 실제 사건의 변화 비율 적용)</span>
 <button data-k="A">재가동 A형 (5샷)</button><button data-k="B">고속 B형 (3샷)</button><button data-k="R">레시피 변경 (8샷)</button>
 <span class="lbl" style="font-size:11px;color:var(--sub)" id="injnote"></span></div>
<section class="fintro hide" id="fintro">
 <h3>CSV 불러오기 — 과거 데이터 한 번에 처리 · 계속 늘어나는 파일 실시간 감시</h3>
 <div class="note" style="margin:0">파일은 이 브라우저 안에서만 읽고 어디로도 보내지 않는다. 판정은 실시간 모드와 같은 엔진(현장 투입용 최종 모델)으로 샷마다 계산한다.</div>
 <div class="drop" id="fdrop"><b>여기에 CSV를 끌어다 놓거나 눌러서 선택</b><br>과거 데이터 전체를 한 번에 판정 (수백 MB도 가능, 처리 중 진행률 표시)</div>
 <div class="banner red hide" id="fmsgw"></div><div id="fmsg" class="banner info" style="display:none"></div>
 <div class="fgrid">
  <div class="fcard"><h4>한 번에 처리 (모든 브라우저)</h4><ul><li>파일 전체를 읽어 품목별(CN7·RG3)로 시간순 정렬 → 같은 시각 행(LH·RH)을 한 샷으로 묶음 → 샷마다 판정</li>
   <li>끝나면 슬라이더·재생·<b>경보 바로 가기</b>로 결과를 훑어봄</li><li>PassOrFail 열이 있으면 실제 불량과 '검사 대상에 들어간 불량' 수를 같이 보여 줌</li></ul></div>
  <div class="fcard"><h4>실시간 감시 (Chrome·Edge)</h4><ul><li>'실시간 감시 시작' → 파일 선택. 지금까지의 내용을 먼저 판정하고, 이후 1~5초마다 파일을 다시 확인</li>
   <li>새로 붙은 행만 읽어 새 샷이 완성되는 즉시 판정 (같은 시각 LH·RH가 다 들어올 때까지 대기, 일정 시간 새 행이 없으면 그대로 판정)</li>
   <li>파일이 통째로 다시 쓰여도 이미 판정한 시각 이후 행만 반영</li></ul></div>
 </div>
 <div class="cols"><b>필요한 열</b> (KAMP 사출성형 데이터 열 이름, 순서 무관, 다른 열은 무시): <code>TimeStamp</code> <code>PART_NAME</code> <code>PART_FACT_PLAN_DATE</code> <code>PART_FACT_SERIAL</code> <code>Max_Injection_Speed</code> <code>Filling_Time</code> <code>Injection_Time</code> <code>Barrel_Temperature_1~3</code> <code>Mold_Temperature_3·4</code> <code>Hopper_Temperature</code> <code>Plasticizing_Position</code><br>
 <b>있으면 쓰는 열</b>: <code>EQUIP_NAME</code> (학습 설비 650톤-우진2호기 행만 판정, '설비명 무시'로 해제) · <code>PassOrFail</code> <code>Reason</code> (실제 불량 표시) · 글자 인코딩 UTF-8·CP949 자동 판별 · 품목은 PART_NAME 앞글자 CN7/RG3로 구분</div>
</section>
<main id="main">
 <div class="col">
  <section class="pane"><h2>① 현재 샷 상태 <small>1샷 = 부품 2개 (LH·RH: 왼쪽·오른쪽 캐비티)</small></h2><div class="pb" id="p1"></div></section>
  <div class="row2">
   <section class="pane"><h2>② 원인과 근거 <small>z = (현재값 − 평소값) ÷ 평소 흔들림</small></h2><div class="pb" id="p2"></div></section>
   <section class="pane"><h2>③ 대처 방법</h2><div class="pb act" id="p3"></div></section>
  </div>
  <section class="pane"><h2>④ 추이 — 최근 200샷 <small id="h4s">실제 불량은 검사 결과 회신 후에만 표시</small></h2>
   <canvas id="cv" height="300"></canvas>
   <div class="legend">
    <span><span class="sw" style="background:#1D3B6E"></span>D(이탈 점수, 세로축 로그 눈금)</span>
    <span><span class="sw" style="background:#C0392B;height:2px"></span>경보선 D = 5</span>
    <span><span class="sw" style="background:#2E86C1"></span>불량 확률</span>
    <span><span class="sw" style="background:#FDF3E1;border:1px solid #E8C788"></span>보류</span>
    <span><span class="sw" style="background:#E3E7ED"></span>평가 불가</span>
    <span style="color:#C0392B" id="lg1">▲ 회신된 실제 불량</span>
    <span id="lg2">┆ 재가동(연속 가동 구간 시작)</span>
    <span id="lg3" class="hide"><span class="sw" style="background:#6C3483"></span>모의 주입 샷</span>
    <span id="lg4">그래프를 누르면 그 샷으로 이동</span>
   </div>
  </section>
 </div>
 <div class="col">
  <section class="pane"><h2>⑤ 검사 대기열 <small>운영 규칙 = 4.2절</small></h2><div class="pb" id="p5"></div></section>
  <section class="pane"><h2>⑥ 데이터 요청 <small>요청서 카드 문구 그대로</small></h2><div class="pb" id="p6"></div></section>
 </div>
 <div class="foot" id="foot"></div>
</main>
<script>
const DATA=__DATA__, REQ=__CARDS__, ACT=__ACT__, CFG=__CFG__, CHK=__CHECKS__, LIVE=__LIVE__;
let MODE="replay";
const VN=["최대 사출속도","충진시간","사출시간","FLR(고화층 비율)"], VU=["mm/s","초","초",""], VD=[1,2,2,3];
const EVENTS={
 CN7:[{l:"10/16 05:21~05:58 첫 구간: 캐비티 가스 11 + 고속 6 (평가 불가 → 전수 대체검사)",T:"2020-10-16 05:57:20"},
      {l:"10/20 06:19 B형 경보 (고속, 양품이었음)",T:"2020-10-20 06:19:14"},
      {l:"10/27 00:56~01:05 재가동 A 불량 10개 (A형 경보)",T:"2020-10-27 00:56:21"},
      {l:"11/03 04:44 원인 불명 불량 1개 (놓친 1개, 회신 시점)",T:"2020-11-03 04:44:17",after:true}],
 RG3:[{l:"10/22 00:51~01:26 첫 구간 (평가 불가)",T:"2020-10-22 01:09:23"},
      {l:"10/22 04:46~07:37 RH 불량 집중 (캐비티 주의)",T:"2020-10-22 04:57:56"},
      {l:"11/04 05:19~05:33 LH 불량 (RH 우선 검사에서 빠짐, 회신 시점)",T:"2020-11-04 05:21:53",after:true}]};
let prod="CN7",S=[],N=0,i=0,playing=false,speed=10,DELAY=10,acc=0,last=0,PRE=null;
const $=id=>document.getElementById(id);
const fmt=(v,d)=>v==null||!isFinite(v)?"–":Number(v).toFixed(d);
const pct=v=>v==null?"–":(v*100<1?(v*100).toFixed(2):(v*100).toFixed(1))+"%";
function inspect(s,p){return s.s!=="score"||(prod==="CN7"?s.D>=CFG.ALARM:p.n==="RH");}
function kind(s,p){ if(s.s==="unavailable")return["대체검사 (평가 불가)","b-gray"]; if(s.s==="held")return["대체검사 (보류)","b-amber"];
 if(s.D>=CFG.ALARM)return["경보 "+(p.tag.split(" ")[0]||""),"b-red"]; if(prod==="RG3"&&p.n==="RH")return["RH 우선 검사","b-navy"]; return["통과","b-ok"];}
function prep(){
 S=DATA[prod];N=S.length;const pops=["all","after","scored"];PRE={};
 for(const k of pops)PRE[k]={n:new Int32Array(N),ins:new Int32Array(N),fb:new Int32Array(N),def:new Int32Array(N),cau:new Int32Array(N)};
 let segStart=0,prevSeg=-1;
 S.forEach((s,j)=>{ if(s.seg!==prevSeg){segStart=j;prevSeg=s.seg;} s.k=j-segStart+1;
  for(const k of pops){const A=PRE[k];const inP=k==="all"||(k==="after"&&s.s!=="unavailable")||(k==="scored"&&s.s==="score");
   let n=0,ins=0,fb=0,d=0,c=0; if(inP)for(const p of s.P){n++;const f=inspect(s,p);ins+=f;fb+=(s.s!=="score");d+=p.y;c+=(f&&p.y);}
   A.n[j]=(j?A.n[j-1]:0)+n;A.ins[j]=(j?A.ins[j-1]:0)+ins;A.fb[j]=(j?A.fb[j-1]:0)+fb;A.def[j]=(j?A.def[j-1]:0)+d;A.cau[j]=(j?A.cau[j-1]:0)+c;}});
 $("slider").max=N-1;
 const ev=$("events");ev.querySelectorAll("button").forEach(b=>b.remove());
 for(const e of EVENTS[prod]){const b=document.createElement("button");b.textContent=e.l;b.onclick=()=>{let j=S.findIndex(s=>s.T===e.T);if(j<0)return;if(e.after)j=Math.min(N-1,j+DELAY);stop();go(j);};ev.appendChild(b);}
}
function revealed(){if(MODE==="live")return -1;if(MODE==="file")return i;return i>=N-1?N-1:i-DELAY;}
function go(j){i=Math.max(0,Math.min(N-1,j));$("slider").value=i;draw();}
function stop(){playing=false;$("play").textContent="▶ 재생";}
// ---------- ① 현재 샷
function verdict(s){ if(s.s==="unavailable")return{c:"v-na",t:"평가 불가",w:"과거 기록이 없어 정상 기준·보정층을 만들 수 없음 → 전수 대체검사"};
 if(s.s==="held")return{c:"v-held",t:"보류",w:s.rsn};
 if(s.D>=CFG.ALARM)return{c:"v-alarm",t:"경보",w:"D(이탈 점수) "+fmt(s.D,1)+" ≥ 경보 기준 "+CFG.ALARM};
 return{c:"v-ok",t:"정상",w:"D(이탈 점수) "+fmt(s.D,2)+" < "+CFG.ALARM+(prod==="RG3"?" · 공정 이탈 없음":"")};}
function p1(s){const v=verdict(s),r=revealed();
 let h=MODE==="live"?`<div class="meta"><span><span class="livedot"></span>수신 시각 <b class="num">${s.recv}</b></span><span>원본 기록 시각 <b class="num">${s.T}</b></span><span>작업오더 <b>${s.lotn}</b> · 오더 내 <b>${s.k}</b>번째 샷</span><span>누적 수신 <b class="num">${LST.shots.toLocaleString()}</b>샷</span></div>`
 :MODE==="file"?`<div class="meta"><span>시각 <b class="num">${s.T}</b></span><span>작업오더 <b>${s.lotn}</b> · 오더 내 <b>${s.k}</b>번째 샷</span><span>위치 <b class="num">${i+1}/${N}</b></span>${s.recv?`<span>${WATCH?'<span class="livedot"></span>':""}판정 시각 <b class="num">${s.recv}</b></span>`:""}<span>파일 <b>${FS.dname}</b></span></div>`
 :`<div class="meta"><span>시각 <b class="num">${s.T}</b></span><span>연속 가동 구간 <b>${s.seg}</b> · 재가동 후 <b>${s.k}</b>번째 샷</span><span>재생 위치 <b class="num">${i+1}/${N}</b></span></div>`;
 if(s.inj)h+=`<div class="banner injflag" style="margin-bottom:8px"><b>모의 주입 샷</b> — ${s.inj}</div>`;
 h+=`<div class="status"><div class="verdict ${v.c}"><div class="big">${v.t}</div><div class="why">${v.w}</div></div><div class="parts">`;
 for(const p of s.P){const [kt,kc]=kind(s,p);const rep=MODE==="live"?`<span class="badge b-gray">라벨 없음 (무라벨 입력)</span>`:MODE==="file"?(p.y==null?`<span class="badge b-gray">라벨 없음</span>`:p.y?`<span class="badge b-red">불량${p.rs?" ("+p.rs+")":""}</span>`:`<span class="badge b-ok">양품</span>`):i<=r?(p.y?`<span class="badge b-red">불량 (${p.rs||"-"})</span>`:`<span class="badge b-ok">양품</span>`):`<span class="badge b-gray">회신 대기</span>`;
  h+=`<div class="part"><div class="ph"><span>${p.n} 부품</span><span class="badge ${kc}">${kt}</span></div><div class="kv">
  <span>불량 확률</span><span class="num"><b>${s.s==="score"?pct(p.p):"–"}</b>${s.s==="unavailable"?" (보정층 없음)":s.s==="held"?" (보류)":""}</span>
  <span>위험점수</span><span class="num">${p.pct==null?"–":fmt(p.pct,1)+" <span style='color:var(--sub)'>(최근 "+CFG.PCT_WINDOW+"부품 중 백분위)</span>"}</span>
  <span>원인 태그</span><span>${p.tag==="-"?"–":"<b>"+p.tag+"</b>"}</span>
  <span>검사 결과</span><span>${rep}</span></div></div>`;}
 h+=`</div></div><div class="frozen">동결 파라미터: 경보 D ≥ ${CFG.ALARM} · 원인 태그 |z| ≥ ${CFG.TAG_Z} · 레시피 설정 거리 > ${CFG.GATE_DIST}이면 보류 · 정상 기준 최소 ${CFG.REF_MIN}샷 · 기준 창 ${CFG.REF_WIN}샷 · 연속 경보 ${CFG.REBASE_RUN}샷이면 기준 재설정 · 위험점수 창 ${CFG.PCT_WINDOW}부품</div>`;
 $("p1").innerHTML=h;}
// ---------- ② 원인과 근거
function zbars(z){const W=520,H=148,L=128,R=W-12,mid=(L+R)/2,lim=10,sx=v=>mid+Math.max(-lim,Math.min(lim,v))/lim*(R-mid)*.92;
 let g=`<svg viewBox="0 0 ${W} ${H}" width="100%" role="img" aria-label="변수별 이탈 z">`;
 for(const [v,c,dash] of [[-5,"#C0392B","4 3"],[5,"#C0392B","4 3"],[-2,"#8A94A6","2 3"],[2,"#8A94A6","2 3"]])g+=`<line x1="${sx(v)}" x2="${sx(v)}" y1="14" y2="${H-14}" stroke="${c}" stroke-dasharray="${dash}"/>`;
 g+=`<line x1="${mid}" x2="${mid}" y1="14" y2="${H-14}" stroke="#1D2330"/>`;
 for(const v of [-10,-5,-2,0,2,5,10])g+=`<text x="${sx(v)}" y="${H-2}" font-size="10" text-anchor="middle" fill="#5A6475">${v===10?"≥+10":v===-10?"≤−10":v>0?"+"+v:v}</text>`;
 g+=`<text x="${sx(5)+3}" y="11" font-size="10" fill="#C0392B">경보 ±5</text><text x="${sx(2)+3}" y="11" font-size="10" fill="#5A6475" text-anchor="end" dx="-6">태그 ±2</text>`;
 z.forEach((v,j)=>{const y=22+j*29,a=sx(Math.min(v,0)),b=sx(Math.max(v,0)),big=Math.abs(v)>=CFG.ALARM,tg=Math.abs(v)>=CFG.TAG_Z;
  const col=big?"#C0392B":tg?"#B9770E":"#1D3B6E";
  g+=`<text x="${L-8}" y="${y+12}" font-size="11.5" text-anchor="end" fill="#1D2330">${VN[j]}</text><rect x="${a}" y="${y+2}" width="${Math.max(1.5,b-a)}" height="14" fill="${col}" rx="2"/>`;
  const tx=v>=0?b+4:a-4,clip=Math.abs(v)>lim,lab=(v>=0?"+":"")+v.toFixed(1),txt=clip?(v>=0?lab+" ▶":"◀ "+lab):lab;
  g+=`<text x="${clip?(v>=0?R-3:sx(-lim)+3):tx}" y="${y+13}" font-size="11" font-weight="700" text-anchor="${clip?(v>=0?"end":"start"):(v>=0?"start":"end")}" fill="${clip?"#fff":col}">${txt}</text>`;});
 return g+"</svg>";}
const PHR=[["최대 사출속도가 평소보다 빨라짐","금형 속 공기가 빠져나갈 시간이 부족 (공기 갇힘 위험)"],["최대 사출속도가 평소보다 느려짐","수지가 늦게 들어가 흐르는 동안 식음 (재가동 직후 식은 수지 의심)"],
 ["충진시간이 평소보다 길어짐","수지가 금형을 채우는 동안 더 많이 굳음"],["충진시간이 평소보다 짧아짐","빠른 충전 → 공기 갇힘 위험"],
 ["사출시간이 평소보다 길어짐","수지가 잘 안 밀려 들어감 (점도 상승 의심)"],["사출시간이 평소보다 짧아짐","빠른 충전 쪽으로 이탈"],
 ["FLR(고화층 비율)이 평소보다 커짐","흐를 통로가 좁아져 끝까지 못 채울 위험 (미성형)"],["FLR(고화층 비율)이 평소보다 작아짐","빠른 충전 쪽으로 이탈"]];
function p2(s){let h="";const tags=[...new Set(s.P.map(p=>p.tag).filter(t=>t!=="-"))];
 if(s.s==="score"&&tags.length)h+=`<div class="tagline" style="color:${s.D>=CFG.ALARM?"var(--red)":"var(--navy)"}">원인 태그: ${tags.join(" / ")}</div>`;
 else if(s.s==="score")h+=`<div class="tagline" style="color:var(--ok)">원인 태그 없음 (경보 아님)</div>`;
 if(!s.z){h+=`<div class="explain">${s.s==="held"?"보류 샷: "+s.rsn+" → 평소값(기준)이 아직 없거나 다른 레시피라 z를 계산하지 않음.":"평가 불가: 이 시점 이전의 기록이 없어 정상 기준을 만들 수 없음."}</div>`;
  h+=`<table class="t"><tr><th>변수</th><th>현재값</th><th>평소값</th><th>평소 흔들림</th><th>z</th></tr>`;s.x.forEach((x,j)=>h+=`<tr><td>${VN[j]}</td><td class="num">${fmt(x,VD[j])} ${VU[j]}</td><td>–</td><td>–</td><td>–</td></tr>`);
  $("p2").innerHTML=h+"</table>";return;}
 if(s.s==="unavailable")h+=`<div class="note" style="margin:0 0 4px">참고: 정상 기준은 있으나 앞선 불량 사건이 없어 확률 보정층이 없음 → 판정은 평가 불가</div>`;
 h+=zbars(s.z)+`<table class="t"><tr><th>변수</th><th>현재값</th><th>평소값 m</th><th>평소 흔들림 s</th><th>z</th></tr>`;
 s.z.forEach((z,j)=>h+=`<tr><td>${VN[j]}</td><td class="num">${fmt(s.x[j],VD[j])} ${VU[j]}</td><td class="num">${fmt(s.m[j],VD[j])}</td><td class="num">${fmt(s.sc[j],VD[j]+1)}</td><td class="num" style="font-weight:700;color:${Math.abs(z)>=CFG.ALARM?"var(--red)":Math.abs(z)>=CFG.TAG_Z?"var(--amber)":"inherit"}">${z>=0?"+":""}${z.toFixed(1)}</td></tr>`);
 h+="</table>";
 const j=s.z.map(Math.abs).indexOf(Math.max(...s.z.map(Math.abs))),z=s.z[j],ph=PHR[j*2+(z>=0?0:1)];
 if(Math.abs(z)>=CFG.TAG_Z)h+=`<div class="explain ${s.D>=CFG.ALARM?"red":""}">${ph[0]} (평소 흔들림의 <b>${z>=0?"+":""}${z.toFixed(1)}배</b>) → ${ph[1]}</div>`;
 else h+=`<div class="explain">네 변수 모두 평소 흔들림의 ${CFG.TAG_Z}배 안 — 공정값은 평소와 같음${prod==="RG3"?". RG3 불량은 대부분 공정값에 흔적이 없는 캐비티 불량이라 RH 부품을 우선 검사":""}.</div>`;
 h+=`<div class="note">D(이탈 점수) = 네 변수 |z| 중 최댓값 = ${fmt(s.D,2)}. 평소값 = 같은 작업오더 과거 정상 샷 최근 ${CFG.REF_WIN}개 중앙값, 평소 흔들림 = 기준 작업오더의 1.4826·MAD(Median Absolute Deviation, 중앙값 절대편차)</div>`;
 $("p2").innerHTML=h;}
// ---------- ③ 대처
function situation(s){ if(s.s==="unavailable")return"na"; if(s.s==="held")return s.rsn.startsWith("다른")?"recipe":"ready";
 if(s.D>=CFG.ALARM){const t=s.P[0].tag;return t.startsWith("A")?"A":t.startsWith("B")?"B":"U";} if(prod==="RG3")return"C"; return"ok";}
function p3(s){const a=ACT,sit=situation(s);let h="";
 if(sit==="A")h=`<div class="head" style="color:var(--red)">A형 느린 충전·고화 — 재가동 직후 식은 수지 의심</div><ol>
 <li><b>생산품 분리</b>: 이탈 점수가 ${CFG.ALARM} 아래로 돌아올 때까지 분리. 재가동 ${a.n_restart}회 기준 검사 대기 분리 대상은 고정 5샷 규칙 ${a.fix5}샷(양품 ${a.fix5_good}샷 포함), 이탈 기반 규칙 ${a.drule}샷으로 A 불량 ${a.drule_A}샷을 모두 포함 (판정 보류 샷까지 분리하면 ${a.dheld}샷). 폐기가 아니라 검사 대기.</li>
 <li><b>수지 온도 먼저 회복</b>: 금형온도로 보상하려면 첫 샷 기준 ${a.mold_need}℃(+${a.mold_up}℃)가 필요해 비현실적 → 재가동 전 퍼지·예열로 수지 온도를 먼저 회복.</li>
 <li><b>배럴 센서만 보고 안심 금지</b>: 이 사건 동안 배럴 온도는 ${a.melt}℃ 그대로였음.</li></ol>`;
 else if(sit==="B"){const v=s.x[0];h=`<div class="head" style="color:var(--red)">B형 고속·공기 갇힘</div><ol>
 <li><b>생산품 분리</b> 후 사출속도 설정 확인 (현재 최대 사출속도 <b>${fmt(v,1)} mm/s</b>).</li>
 <li><b>사출속도 상한 후보</b>: 정상 정속 ${a.v_mean}±${a.v_sd} mm/s, 양품 관측 최고 ${a.v_good_max}, B 불량 최저 ${a.v_B_min} → 상한 60 mm/s 부근 설정 검토. 경계 확정은 요청 #3(계단 시험).</li>
 <li><b>벤트 점검</b>: 반복되면 벤트 청소.</li></ol>`;}
 else if(sit==="U")h=`<div class="head" style="color:var(--red)">공정 이탈 (유형 불명)</div><ol><li>생산품 분리, 이탈 점수가 ${CFG.ALARM} 아래로 돌아올 때까지 유지.</li><li>속도·시간이 같은 방향으로 움직이지 않음 → 설정 변경 여부를 사출 설정값 로그(#7)로 확인.</li></ol>`;
 else if(sit==="ready")h=`<div class="head" style="color:var(--amber)">보류 — 기준 준비 중</div><ol><li>정상 기준 샷 ${CFG.REF_MIN}개가 쌓일 때까지 <b>대체검사</b>.</li><li>작업오더가 바뀌면 기준을 새로 잡음 (다른 오더의 기준을 빌리지 않음).</li></ol>`;
 else if(sit==="recipe")h=`<div class="head" style="color:var(--amber)">보류 — 다른 레시피</div><ol><li><b>대체검사</b> + 레시피 코드 확인.</li><li>학습 범위 밖 설정이라 모델 점수를 내지 않음 (적용 범위 문).</li></ol>`;
 else if(sit==="na")h=`<div class="head" style="color:var(--gray)">평가 불가 — 도입 초기</div><ol><li><b>전수 대체검사</b>.</li><li>정상 기준과 보정층은 이 구간 기록이 쌓인 뒤부터 만들어짐.</li></ol>${prod==="CN7"&&s.seg===0?`<div class="note">이 첫 구간 불량 17개(캐비티 가스 11, 고속 6)는 모델이 아니라 대체검사가 잡는다. 모델의 검출 성능은 사건 블록 제외 평가에서 따로 확인(고속 6/6, 캐비티 가스 1/11).</div>`:""}`;
 else if(sit==="C")h=`<div class="head" style="color:var(--navy)">RG3 캐비티 주의 (RH)</div><ol><li><b>RH 부품 우선 검사</b> (RH 쪽 불량 오즈비 2.09).</li><li>RH 안에서 더 가를 정보는 없음 → 요청 #1(캐비티별 부품 중량).</li><li>LH 쪽 불량이 회신되면 벤트·러너 점검 요청.</li></ol>`;
 else h=`<div class="head" style="color:var(--ok)">조치 없음 — 정상 생산 계속</div><ol><li>검사 여력이 있으면 위험점수 상위부터 검사 (표 4-2).</li></ol>`;
 $("p3").innerHTML=h;}
// ---------- ⑤ 대기열
function p5(s){if(MODE==="live")return p5live(s);if(MODE==="file")return p5file(s);const r=revealed();let h=`<table class="cnt"><tr><th>누적 (지금까지)</th><th>검사 / 부품</th><th>검사 비율</th><th>대체검사</th><th>선정된 불량<br>(회신 기준)</th></tr>`;
 for(const [k,lab] of [["all","전체"],["after","첫 구간 제외"],["scored","점수 부품만"]]){const A=PRE[k],n=A.n[i],ins=A.ins[i],d=r>=0?A.def[r]:0,c=r>=0?A.cau[r]:0;
  h+=`<tr><td>${lab}</td><td class="num">${ins.toLocaleString()} / ${n.toLocaleString()}</td><td class="num"><b>${n?(ins/n*100).toFixed(1):"–"}%</b></td><td class="num">${A.fb[i].toLocaleString()}</td><td class="num"><b>${c}/${d}</b></td></tr>`;}
 h+=`</table><div class="note">${i>=N-1?"재생 종료: 모든 검사 결과 회신 완료 → 표 4-2와 같은 값":"규칙: "+(prod==="CN7"?"경보 부품(D ≥ 5) → 확률 높은 순":"RH 부품 우선 (RG3는 공정값에 흔적이 없는 캐비티 불량이 대부분)")+", 보류·평가 불가 부품 → 대체검사"}</div>`;
 const pend=[];for(let j=Math.max(r+1,0);j<=i;j++){const t=S[j];for(const p of t.P)if(inspect(t,p))pend.push({j,t,p});}
 pend.sort((a,b)=>{const ra=a.t.s!=="score"?2:0,rb=b.t.s!=="score"?2:0;if(ra!==rb)return ra-rb;if(ra===0)return(b.p.p||0)-(a.p.p||0);return b.j-a.j;});
 h+=`<div class="qh">검사 대기 (회신 전) ${pend.length}건</div><ul class="q">`;
 pend.slice(0,7).forEach(o=>{const [kt,kc]=kind(o.t,o.p);h+=`<li><span class="tm">${o.t.t} ${o.p.n}</span><span><span class="badge ${kc}">${kt}</span></span><span class="num">${o.t.s==="score"?pct(o.p.p):"–"}</span></li>`;});
 if(!pend.length)h+=`<li style="grid-template-columns:1fr;color:var(--sub)">대기 중인 검사 없음</li>`;
 h+=`</ul>${pend.length>7?`<div class="more">외 ${pend.length-7}건</div>`:""}`;
 const done=[];for(let j=r;j>=0&&done.length<6&&j>r-400;j--){const t=S[j];for(const p of t.P){const f=inspect(t,p);if(f||p.y)done.push({t,p,f});}}
 h+=`<div class="qh">최근 회신</div><ul class="q">`;
 done.slice(0,6).forEach(o=>{const res=o.p.y?(o.f?`<span class="badge b-red">불량 확인 (${o.p.rs})</span>`:`<span class="badge b-red" style="background:#fff;color:var(--red);border:1px solid var(--red)">놓친 불량 (${o.p.rs})</span>`):`<span class="badge b-ok">양품</span>`;
  h+=`<li><span class="tm">${o.t.t} ${o.p.n}</span><span>${res}</span><span class="badge ${kind(o.t,o.p)[1]}" style="font-weight:400">${o.f?"검사함":"검사 안 함"}</span></li>`;});
 if(!done.length)h+=`<li style="grid-template-columns:1fr;color:var(--sub)">아직 회신 없음</li>`;
 $("p5").innerHTML=h+"</ul>";}
// ---------- ⑥ 요청
function card(n,why){const c=REQ.cards[n];if(!c)return"";const R=Object.fromEntries(c.rows);
 return`<div class="rc"><div class="rh ${c.kind==="n"?"":c.kind}"><span>${c.title}</span><span style="font-weight:400">${c.cause}</span></div><div class="rb">
 ${why?`<div style="color:var(--navy);font-weight:700">${why}</div>`:""}<div><b>무엇을</b>${R["무엇을"]}</div><div><b>판정 후 행동</b>${R["판정 후 행동"]}</div><div><b>비용·기간</b>${R["비용·기간"]}</div>
 <details><summary>카드 원문 전체</summary>${c.rows.map(([k,v])=>`<div><b>${k}</b>${v}</div>`).join("")}</details></div></div>`;}
function lastMissed(){if(MODE==="live")return null;const r=revealed();for(let j=r;j>=0&&j>r-30;j--){const t=S[j];for(const p of t.P)if(p.y&&!inspect(t,p))return{t,p,j};}return null;}
function p6(s){const sit=situation(s);let h="";const m=lastMissed();
 if(m)h+=`<div class="banner red"><b>놓친 불량 회신</b>: ${m.t.t} ${m.p.n} (${m.p.rs}) — 검사하지 않은 부품에서 불량. ${m.t.s==="score"&&m.t.D<CFG.ALARM?"공정값은 평소와 같았음(D "+fmt(m.t.D,2)+") → 공정값에 흔적이 없는 불량. 캐비티별 정보가 필요":""}</div>`+card("1","놓친 불량 → 1순위")+(prod==="CN7"?card("2"):"");
 const map={A:[["4"],["7"],["8"],["14","비권장: 단독으로는 판정 불가"]],B:[["3"],["7"],["9"]],U:[["7"],["8"]],recipe:[["10"]],C:[["1"]]};
 if(map[sit]){if(!m||sit!=="C")h+=`<div class="banner info">현재 상황(${{A:"A형 경보",B:"B형 경보",U:"유형 불명 경보",recipe:"보류 — 다른 레시피",C:"RG3 캐비티 주의"}[sit]})과 관련된 요청</div>`+map[sit].map(([n,w])=>card(n,w)).join("");}
 else if(!m)h+=`<div class="banner info">현재 상황과 직접 연결된 요청 없음${sit==="na"?" (기록이 쌓이면 해소)":""}</div>`;
 h+=`<div class="plan"><b>실행 계획</b>${REQ.plan.map(([k,v])=>`<div title="${v}"><b>${k}</b> ${(v.match(/#\d+/g)||[]).join(" ")}</div>`).join("")}<div><b>비권장</b> #14</div></div>`;
 $("p6").innerHTML=h;}
// ---------- ④ 추이
const cv=$("cv");let geo=null;
function p4(){const dpr=window.devicePixelRatio||1,W=cv.clientWidth,H=300;cv.width=W*dpr;cv.height=H*dpr;const c=cv.getContext("2d");c.setTransform(dpr,0,0,dpr,0,0);c.clearRect(0,0,W,H);
 const L=46,Rr=W-52,top=12,h1=170,top2=top+h1+26,h2=H-top2-22,a=Math.max(0,i-199),n=200,xw=(Rr-L)/n,X=j=>L+(j-a+.5)*xw,r=revealed();geo={L,xw,a};
 c.font="11px 'Malgun Gothic','Apple SD Gothic Neo',sans-serif";
 const ty=d=>top+h1-Math.log1p(Math.min(d,200))/Math.log1p(200)*h1;
 let pm=0;for(let j=a;j<=i;j++)for(const p of S[j].P)if(p.p!=null&&S[j].s==="score")pm=Math.max(pm,p.p);
 const pmax=prod==="CN7"?1:Math.max(.05,Math.ceil(pm*1.25*100)/100),ty2=v=>top2+h2-v/pmax*h2;
 for(let j=a;j<=i;j++){const s=S[j];if(s.s==="held"||s.s==="unavailable"){c.fillStyle=s.s==="held"?"#FDF3E1":"#E3E7ED";c.fillRect(X(j)-xw/2,top,xw+.5,h1);c.fillRect(X(j)-xw/2,top2,xw+.5,h2);}}
 c.strokeStyle="#D5DAE3";c.lineWidth=1;c.fillStyle="#5A6475";c.textAlign="right";
 for(const d of [0,1,2,5,10,20,50,100,200]){const y=ty(d);c.beginPath();c.moveTo(L,y);c.lineTo(Rr,y);c.stroke();c.fillText(d,L-5,y+4);}
 c.save();c.translate(12,top+h1/2);c.rotate(-Math.PI/2);c.textAlign="center";c.fillText("D (이탈 점수)",0,0);c.restore();
 c.strokeStyle="#C0392B";c.setLineDash([5,3]);c.beginPath();c.moveTo(L,ty(5));c.lineTo(Rr,ty(5));c.stroke();c.setLineDash([]);c.fillStyle="#C0392B";c.textAlign="left";c.fillText("경보 5",Rr+4,ty(5)+4);
 let lastLab=-1e9;for(let j=a+1;j<=i;j++)if(S[j].seg!==S[j-1].seg){c.strokeStyle="#8A94A6";c.setLineDash([2,3]);c.beginPath();c.moveTo(X(j)-xw/2,top);c.lineTo(X(j)-xw/2,top2+h2);c.stroke();c.setLineDash([]);if(X(j)-lastLab>60){c.fillStyle="#5A6475";c.textAlign="left";c.fillText(MODE!=="replay"?"오더 변경":"재가동",X(j)-xw/2+2,top+10);lastLab=X(j);}}
 c.strokeStyle="#1D3B6E";c.lineWidth=1.6;c.beginPath();let pen=false;for(let j=a;j<=i;j++){const d=S[j].s==="score"?S[j].D:null;if(d==null){pen=false;continue;}const x=X(j),y=ty(d);pen?c.lineTo(x,y):c.moveTo(x,y);pen=true;}c.stroke();
 for(let j=a;j<=i;j++){const s=S[j];if(s.s==="score"&&s.D>=CFG.ALARM){c.fillStyle="#C0392B";c.beginPath();c.arc(X(j),ty(s.D),3,0,7);c.fill();}}
 for(let j=a;j<=Math.min(i,r);j++){const s=S[j];const k=s.P.filter(p=>p.y).length;if(k){c.fillStyle="#C0392B";const x=X(j),y=top+h1-2;c.beginPath();c.moveTo(x,y-9);c.lineTo(x-5,y);c.lineTo(x+5,y);c.closePath();c.fill();}}
 c.strokeStyle="#D5DAE3";c.lineWidth=1;c.fillStyle="#5A6475";c.textAlign="right";
 for(const f of [0,.5,1]){const v=pmax*f,y=ty2(v);c.beginPath();c.moveTo(L,y);c.lineTo(Rr,y);c.stroke();c.fillText((v*100).toFixed(pmax<.2?1:0)+"%",L-5,y+4);}
 c.save();c.translate(12,top2+h2/2);c.rotate(-Math.PI/2);c.textAlign="center";c.fillText("불량 확률",0,0);c.restore();
 const cols={LH:"#2E86C1",RH:"#7D3C98"};for(const nm of ["LH","RH"]){c.strokeStyle=cols[nm];c.lineWidth=1.4;c.beginPath();let pn=false;for(let j=a;j<=i;j++){const s=S[j];const p=s.P.find(q=>q.n===nm);if(!p||p.p==null||s.s!=="score"){pn=false;continue;}const x=X(j),y=ty2(p.p);pn?c.lineTo(x,y):c.moveTo(x,y);pn=true;}c.stroke();}
 c.textAlign="left";c.fillStyle=cols.LH;c.fillText("LH",Rr+4,top2+12);c.fillStyle=cols.RH;c.fillText("RH",Rr+4,top2+26);
 c.strokeStyle="#1D2330";c.lineWidth=1;c.beginPath();c.moveTo(X(i),top);c.lineTo(X(i),top2+h2);c.stroke();
 c.fillStyle="#5A6475";c.textAlign="left";c.fillText(S[a].t,L,H-6);c.textAlign="right";c.fillText(S[i].t+" (현재)",Rr,H-6);
 for(let j=a;j<=i;j++)if(S[j].inj){c.fillStyle="#6C3483";c.fillRect(X(j)-Math.max(1,xw/2),top,Math.max(2,xw),5);}
 if(MODE==="replay"&&r<i){c.textAlign="center";c.fillStyle="#8A94A6";const xr=X(Math.max(r,a));c.fillText("◀ 회신 완료 | 회신 대기 ▶",(xr+X(i))/2>L+80?Math.min(xr,Rr-80):L+80,top2-8);}}
cv.addEventListener("click",e=>{if(!geo||MODE==="live")return;const rc=cv.getBoundingClientRect();const j=Math.floor((e.clientX-rc.left-geo.L)/geo.xw)+geo.a;if(j>=0&&j<=i){stop();go(j);}});
function draw(){const s=S[i];if(MODE!=="live")$("pos").textContent=`${s.t} · ${i+1}/${N}샷`;else $("lpos").innerHTML=`입력: 무라벨 실제 운전 기록 ${LIVE[prod].range[0]} ~ ${LIVE[prod].range[1]} (${LIVE[prod].shots.length.toLocaleString()}샷, 끝나면 처음부터 반복) · 진행 ${LIDX.toLocaleString()}/${LIVE[prod].shots.length.toLocaleString()}`;p1(s);p2(s);p3(s);p4();p5(s);p6(s);}
function tick(t){if(playing){acc+=(t-last)/1000*speed;let k=Math.floor(acc);if(k>0){acc-=k;if(MODE==="live"){liveAdvance(k);}else if(i+k>=N-1){go(N-1);stop();}else go(i+k);}}last=t;requestAnimationFrame(tick);}
function startPlay(){playing=true;acc=0;$("play").textContent="❚❚ 일시정지";}
$("play").onclick=()=>{if(playing)stop();else{if(MODE==="replay"&&i>=N-1)go(0);startPlay();}};
$("spd").querySelectorAll("button").forEach(b=>b.onclick=()=>{speed=+b.dataset.s;$("spd").querySelectorAll("button").forEach(x=>x.classList.toggle("on",x===b));});
$("slider").oninput=e=>{go(+e.target.value);};$("prev").onclick=()=>{stop();go(i-1);};$("next").onclick=()=>{stop();go(i+1);};
document.querySelectorAll(".seg-btn").forEach(b=>b.onclick=()=>{stop();prod=b.dataset.p;document.querySelectorAll(".seg-btn").forEach(x=>x.classList.toggle("on",x===b));if(MODE==="live"){initLive();startPlay();}else if(MODE==="file"){showFile(true);}else{prep();go(0);}});
window.addEventListener("resize",()=>p4());
document.addEventListener("keydown",e=>{if(MODE==="live"&&e.key!==" ")return;if(e.target.tagName==="INPUT"&&e.key!==" ")return;if(e.key===" "){e.preventDefault();$("play").click();}if(e.key==="ArrowRight"){stop();go(i+1);}if(e.key==="ArrowLeft"){stop();go(i-1);}});
$("foot").textContent=`데이터 출처: 엄격 시간 순서(forward) 평가를 폴드마다 다시 계산한 스트림 출력(RiskModel.predict_stream). 공식 예측 CSV와 대조 — 이탈 점수 최대 차이 CN7 ${CHK.CN7.D_max_diff.toExponential(1)}, RG3 ${CHK.RG3.D_max_diff.toExponential(1)}; 확률 최대 차이 CN7 ${CHK.CN7.P_max_diff.toExponential(1)}, RG3 ${CHK.RG3.P_max_diff.toExponential(1)}. 각 샷 시점에 모델이 알 수 있던 정보만 표시하고, 실제 불량은 검사 결과 회신(지연 ${DELAY}샷 가정) 뒤에만 표시한다. 키보드: 스페이스 재생/정지, ←/→ 한 샷 이동.`;
// ================= 실시간 모드: 브라우저 안의 판정 엔진 (model.RiskModel.step과 같은 계산) =================
const REPLAY_FOOT=$("foot").textContent;
function med(a){const b=a.slice().sort((x,y)=>x-y),n=b.length,h=n>>1;return n%2?b[h]:(b[h-1]+b[h])/2;}
function tagRule(z,D,rh,pr){if(D>=CFG.ALARM){const t=CFG.TAG_Z;const slow=z[0]<=-t||z[1]>=t||z[3]>=t,fast=z[0]>=t||z[1]<=-t||z[3]<=-t;
 if(slow&&!fast)return"A형 느린 충전·고화";if(fast&&!slow)return"B형 고속·공기 갇힘";return"공정 이탈 (유형 불명)";}
 if(pr==="RG3"&&rh===1)return"캐비티 주의 (RH)";return"-";}
function Engine(M,pr){this.M=M;this.pr=pr;this.reset();}
Engine.prototype.reset=function(){this.ref=[[],[],[],[]];this.run=[];this.prev=null;this.win=[];this.k=0;this.lotn=0;};
Engine.prototype.step=function(sh,x,g){const M=this.M,C=CFG;
 if(sh.lot!==this.prev){this.ref=[[],[],[],[]];this.run=[];this.k=0;this.lotn++;}
 this.prev=sh.lot;this.k++;
 let ss=0;for(let j=0;j<4;j++){const d=(g[j]-M.gmed[j])/M.gsc[j];ss+=d*d;}const gd=Math.sqrt(ss/4);
 let reason=null;
 if(gd>C.GATE_DIST)reason="다른 레시피 (설정 거리 "+gd.toFixed(1)+")";
 else if(this.ref[0].length<C.REF_MIN){for(let j=0;j<4;j++)this.ref[j].push(x[j]);reason="기준 준비 중 (작업오더 "+this.ref[0].length+"번째 샷)";}
 if(reason)return{held:true,reason};
 const m=[],z=[];for(let j=0;j<4;j++){m[j]=med(this.ref[j].slice(-C.REF_WIN));z[j]=(x[j]-m[j])/M.scale[j];}
 let D=0;for(const v of z)D=Math.max(D,Math.abs(v));
 if(D<C.ALARM){this.run=[];for(let j=0;j<4;j++)this.ref[j].push(x[j]);}
 else{this.run.push(x.slice());if(this.run.length>=C.REBASE_RUN){this.ref=[0,1,2,3].map(j=>this.run.map(r=>r[j]));this.run=[];}}
 const parts=sh.P.map(([n,rh])=>{const lin=M.b[0]+(M.process?M.b[1]*Math.log1p(D):0)+M.Lc*rh,p=1/(1+Math.exp(-lin));
  const w=this.win.slice(-(C.PCT_WINDOW-1));w.push(p);let lt=0,eq=0;for(const v of w){if(v<p)lt++;else if(v===p)eq++;}
  const pct=100*(lt+.5*eq)/w.length;this.win.push(p);if(this.win.length>4*C.PCT_WINDOW)this.win=this.win.slice(-C.PCT_WINDOW);
  return{n,rh,p,pct,tag:tagRule(z,D,rh,this.pr)};});
 return{held:false,D,z,m,parts};};
// 자체 일치 검사: 같은 입력을 Python 원본(predict_stream)에 넣은 결과와 비교
const SELF={};
function selfCheck(pr){const L=LIVE[pr],E=new Engine(L.model,pr);let dD=0,dP=0,dQ=0,bad=0;
 L.shots.forEach((sh,j)=>{const r=E.step(sh,sh.x.slice(),sh.g.slice()),ex=L.exp[j];
  if(r.held!==(ex.D==null)){bad++;return;}
  if(r.held){if(r.reason!==ex.r)bad++;return;}
  dD=Math.max(dD,Math.abs(r.D-ex.D));r.parts.forEach((p,k)=>{dP=Math.max(dP,Math.abs(p.p-ex.P[k][0]));dQ=Math.max(dQ,Math.abs(p.pct-ex.P[k][1]));if(p.tag!==ex.P[k][2])bad++;});});
 return{dD,dP,dQ,bad,n:L.shots.length};}
let ENG=null,LS=[],LIDX=0,PENDING=[],LST=null;
function hhmmss(d){return d.toTimeString().slice(0,8);}
function initLive(){ENG=new Engine(LIVE[prod].model,prod);LS=[];LIDX=0;PENDING=[];LST={shots:0,parts:0,ins:0,fb:0,al:0,inj:0};
 if(!SELF[prod])SELF[prod]=selfCheck(prod);liveAdvance(1);setLiveFoot();}
function liveStep(){const L=LIVE[prod];if(LIDX>=L.shots.length){LIDX=0;ENG.reset();}
 const sh=L.shots[LIDX];let x=sh.x.slice(),g=sh.g.slice(),inj=null;
 if(PENDING.length){const q=PENDING.shift();inj=q.label;if(q.r)x=x.map((v,j)=>v*q.r[j]);if(q.dTm)g[0]+=q.dTm;LST.inj++;}
 const r=ENG.step(sh,x,g);LIDX++;
 const P=sh.P.map(([n,rh],k)=>r.held?{n,y:null,p:null,pct:null,tag:"-",rs:""}:{n,y:null,p:r.parts[k].p,pct:r.parts[k].pct,tag:r.parts[k].tag,rs:""});
 const rec={t:sh.T.slice(5,16).replace("-","/"),T:sh.T,seg:ENG.lotn,lotn:ENG.lotn,k:ENG.k,s:r.held?"held":"score",D:r.held?null:r.D,x,
  z:r.held?null:r.z,m:r.held?null:r.m,sc:LIVE[prod].model.scale,rsn:r.held?r.reason:"",P,inj,recv:hhmmss(new Date())};
 LST.shots++;for(const p of P){LST.parts++;const f=inspect(rec,p);LST.ins+=f;LST.fb+=rec.s!=="score";}if(rec.s==="score"&&rec.D>=CFG.ALARM)LST.al++;
 LS.push(rec);if(LS.length>600)LS.splice(0,LS.length-400);}
function liveAdvance(k){for(let q=0;q<k;q++)liveStep();S=LS;N=LS.length;i=N-1;draw();}
function p5live(s){let h=`<table class="cnt"><tr><th>누적 (수신 시작부터)</th><th>값</th></tr>
 <tr><td>수신 샷 / 부품</td><td class="num">${LST.shots.toLocaleString()} / ${LST.parts.toLocaleString()}</td></tr>
 <tr><td>검사 대상 부품 (비율)</td><td class="num"><b>${LST.ins.toLocaleString()} (${LST.parts?(LST.ins/LST.parts*100).toFixed(1):"–"}%)</b></td></tr>
 <tr><td>그중 대체검사 (보류)</td><td class="num">${LST.fb.toLocaleString()}</td></tr>
 <tr><td>경보 샷 (D ≥ ${CFG.ALARM})</td><td class="num">${LST.al.toLocaleString()}${LST.inj?` <span style="color:#6C3483">(모의 주입 ${LST.inj}샷 포함)</span>`:""}</td></tr></table>
 <div class="note">무라벨 입력이라 실제 불량 여부와 검출률은 없음 → 검출 성능은 재생 모드(표 4-2). 규칙: ${prod==="CN7"?"경보 부품(D ≥ 5) → 확률 높은 순":"RH 부품 우선"}, 보류 부품 → 대체검사</div>`;
 const pend=[];for(let j=Math.max(0,LS.length-10);j<LS.length;j++){const t=LS[j];for(const p of t.P)if(inspect(t,p))pend.push({j,t,p});}
 pend.sort((a,b)=>{const ra=a.t.s!=="score"?2:0,rb=b.t.s!=="score"?2:0;if(ra!==rb)return ra-rb;if(ra===0)return(b.p.p||0)-(a.p.p||0);return b.j-a.j;});
 h+=`<div class="qh">검사 대기 (최근 10샷) ${pend.length}건</div><ul class="q">`;
 pend.slice(0,9).forEach(o=>{const [kt,kc]=kind(o.t,o.p);h+=`<li><span class="tm">${o.t.recv} ${o.p.n}</span><span><span class="badge ${kc}">${kt}</span>${o.t.inj?` <span class="badge injflag">주입</span>`:""}</span><span class="num">${o.t.s==="score"?pct(o.p.p):"–"}</span></li>`;});
 if(!pend.length)h+=`<li style="grid-template-columns:1fr;color:var(--sub)">대기 중인 검사 없음 — 정상 생산 계속</li>`;
 $("p5").innerHTML=h+`</ul>${pend.length>9?`<div class="more">외 ${pend.length-9}건</div>`:""}`;}
function setLiveFoot(){const c=SELF[prod];$("foot").textContent=`실시간 모드: 무라벨 실제 운전 기록(${prod} ${LIVE[prod].range[0]} ~ ${LIVE[prod].range[1]}, 정상 기준 계산에 쓰지 않은 작업오더, 라벨 기간 이전)을 샷 하나씩 입력하면 이 화면 안의 판정 엔진이 그 자리에서 계산한다. 모델은 현장 투입용 최종 모델(전체 데이터 학습). 브라우저 계산 = Python 원본(RiskModel.predict_stream) 자체 검사: ${c.n.toLocaleString()}샷 중 판정·보류 사유·원인 태그 불일치 ${c.bad}건, 이탈 점수 최대 차이 ${c.dD.toExponential(1)}, 확률 ${c.dP.toExponential(1)}, 위험점수 ${c.dQ.toExponential(1)}. 모의 주입은 실제 사건(CN7 10/27 재가동, 10/20 고속)의 변화 비율을 실제 입력 샷에 곱한 것이며 화면에 '주입'으로 표시한다. 설비 PLC·MES에 연결한 것은 아니다.`;}
const INJ=LIVE.CN7.inj;
$("injnote").textContent=`재가동 A형 = CN7 10/27 00:56~01:05 실제 5샷 · 고속 B형 = CN7 10/20 06:18~06:20 실제 3샷 · 레시피 변경 = 용융온도 설정 +${INJ.recipe_dTm}℃`;
$("inj").querySelectorAll("button").forEach(b=>b.onclick=()=>{const k=b.dataset.k;
 if(k==="A")INJ.A.forEach((r,n)=>PENDING.push({r,label:`재가동 A형 모의 (${n+1}/5샷) — CN7 10/27 재가동 사건의 실제 변화 비율(현재값÷평소값)을 이 입력 샷에 적용`}));
 if(k==="B")INJ.B.forEach((r,n)=>PENDING.push({r,label:`고속 B형 모의 (${n+1}/3샷) — CN7 10/20 고속 사건의 실제 변화 비율을 이 입력 샷에 적용`}));
 if(k==="R")for(let n=0;n<8;n++)PENDING.push({dTm:LIVE[prod].inj.recipe_dTm,label:`레시피 변경 모의 (${n+1}/8샷) — 용융온도 설정 +${LIVE[prod].inj.recipe_dTm}℃`});
 if(!playing)liveAdvance(1);});
$("lreset").onclick=()=>{initLive();};
function setMode(m){stop();MODE=m;document.querySelectorAll(".mode-btn").forEach(x=>x.classList.toggle("on",x.dataset.m===m));
 const live=m==="live",file=m==="file";
 $("h4s").textContent=live?"무라벨 입력 — 실제 불량 표시 없음":file?"▲ = 파일에 기록된 실제 불량 (PassOrFail 열이 있을 때)":"실제 불량은 검사 결과 회신 후에만 표시";
 $("lg1").textContent=file?"▲ 파일에 기록된 실제 불량":"▲ 회신된 실제 불량";
 $("lg1").classList.toggle("hide",live);$("lg4").classList.toggle("hide",live);$("lg3").classList.toggle("hide",!live);
 $("lg2").textContent=live||file?"┆ 작업오더 변경 (기준 재설정)":"┆ 재가동(연속 가동 구간 시작)";$("rctl").style.display=live?"none":"contents";$("dlyl").classList.toggle("hide",file);$("lctl").classList.toggle("hide",!live);$("events").classList.toggle("hide",live||file);$("inj").classList.toggle("hide",!live);
 $("fctl").classList.toggle("hide",!file);$("fline").classList.toggle("hide",!file);$("fev").classList.toggle("hide",!file||!FS);
 if(!file){$("fintro").classList.add("hide");$("main").classList.remove("hide");for(const id of ["play","spdl","spd"])$(id).classList.remove("hide");}
 $("demo").textContent=live?"실시간 처리 시연 · 무라벨 실제 운전 기록을 샷마다 입력 → 화면 안에서 즉시 판정 (설비 미연결)":file?"CSV 불러오기 · 넣은 파일을 같은 판정 엔진으로 처리 (과거 파일 한 번에 / 늘어나는 파일 실시간 감시)":"재생 시연 · 과거 데이터를 엄격 시간 순서 평가 조건 그대로 샷 순서대로 재생 (성능 검증용)";
 if(live){speed=1;$("spd").querySelectorAll("button").forEach(x=>x.classList.toggle("on",x.dataset.s==="1"));initLive();startPlay();}
 else if(file){showFile(true);if(!FS)fstatus("CSV를 열면 여기에 진행 상황이 표시됨");}
 else{$("foot").textContent=REPLAY_FOOT;prep();go(0);}}
document.querySelectorAll(".mode-btn").forEach(b=>b.onclick=()=>setMode(b.dataset.m));

// ================= 파일 모드: 사용자 CSV를 같은 판정 엔진(Engine = RiskModel.step)으로 처리 =================
// · 한 번에 처리: 과거 CSV 전체를 읽어 제품별로 시간순 정렬 → 샷으로 묶기 → 샷마다 판정 (Python 원본과 같은 순서·같은 계산)
// · 실시간 감시: 같은 파일을 주기적으로 다시 확인해 새로 붙은 행만 읽고, 새 샷이 완성되면 바로 판정
const FREQ=["TimeStamp","PART_NAME","PART_FACT_PLAN_DATE","PART_FACT_SERIAL","Barrel_Temperature_1","Barrel_Temperature_2","Barrel_Temperature_3",
 "Mold_Temperature_3","Mold_Temperature_4","Max_Injection_Speed","Filling_Time","Injection_Time","Hopper_Temperature","Plasticizing_Position"];
const SQPI=Math.sqrt(Math.PI);
function erfPos(x){if(x<3){const x2=x*x;let t=x,s=x,n=0;for(;;){n++;t*=2*x2/(2*n+1);s+=t;if(t<1e-17*s||n>300)break;}return 2/SQPI*Math.exp(-x2)*s;}
 let f=x;for(let k=80;k>=1;k--)f=x+(k/2)/f;return 1-Math.exp(-x*x)/(SQPI*f);}
function erfinv(y){if(y<=0)return y===0?0:-erfinv(-y);let w=-Math.log((1-y)*(1+y)),p;
 if(w<5){w-=2.5;p=2.81022636e-08;p=3.43273939e-07+p*w;p=-3.5233877e-06+p*w;p=-4.39150654e-06+p*w;p=0.00021858087+p*w;p=-0.00125372503+p*w;p=-0.00417768164+p*w;p=0.246640727+p*w;p=1.50140941+p*w;}
 else{w=Math.sqrt(w)-3;p=-0.000200214257;p=0.000100950558+p*w;p=0.00134934322+p*w;p=-0.00367342844+p*w;p=0.00573950773+p*w;p=-0.0076224613+p*w;p=0.00943887047+p*w;p=1.00167406+p*w;p=2.83297682+p*w;}
 let x=p*y;for(let k=0;k<4;k++){const d=2/SQPI*Math.exp(-x*x);if(d===0)break;x-=(erfPos(x)-y)/d;}return x;}
function flr(fill,Tm,Tw){const r=Math.min(Math.max((CFG.T_NF-Tw)/(Tm-Tw),1e-9),1-1e-9);return 2*Math.sqrt(CFG.ALPHA*Math.max(fill,1e-3))*erfinv(r)/CFG.HALF_THICK;}
function tsKey(s){const m=/(\d{4})\D(\d{1,2})\D(\d{1,2})\D+(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?/.exec(s);if(!m)return NaN;
 const k=Date.UTC(+m[1],m[2]-1,+m[3],+m[4],+m[5],+(m[6]||0)),d=new Date(k);   // 없는 날짜·시각(2/30, 25시 등)은 거절
 return d.getUTCFullYear()===+m[1]&&d.getUTCMonth()===m[2]-1&&d.getUTCDate()===+m[3]&&d.getUTCHours()===+m[4]&&d.getUTCMinutes()===+m[5]&&d.getUTCSeconds()===+(m[6]||0)?k:NaN;}
function num(t){t=(t==null?"":String(t)).trim();return t===""?NaN:Number(t);}   // "20초"·"1,200"·빈 칸은 NaN (parseFloat처럼 앞 숫자만 읽지 않음)
const esc=t=>String(t).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
function tsStr(k){return new Date(k).toISOString().slice(0,19).replace("T"," ");}
function splitCSV(line){if(line.indexOf('"')<0)return line.split(",");const o=[];let c="",q=false;
 for(let k=0;k<line.length;k++){const ch=line[k];if(q){if(ch==='"'){if(line[k+1]==='"'){c+='"';k++;}else q=false;}else c+=ch;}else if(ch==='"')q=true;else if(ch===","){o.push(c);c="";}else c+=ch;}o.push(c);return o;}
function inspectP(s,p,pr){return s.s!=="score"||(pr==="CN7"?s.D>=CFG.ALARM:p.n==="RH");}
const LRANGE={};for(const p of ["CN7","RG3"])LRANGE[p]=[tsKey(DATA[p][0].T),tsKey(DATA[p][DATA[p].length-1].T)];
async function sniffEnc(file){const b=new Uint8Array(await file.slice(0,262144).arrayBuffer());
 try{new TextDecoder("utf-8",{fatal:true}).decode(b.subarray(0,b.length-4));return"utf-8";}catch(e){return"euc-kr";}}

function FileSession(name,opt){this.name=name;this.dname=esc(name);this.opt=opt;this.col=null;this.labels=false;this.enc="utf-8";this.dec=null;this.rem="";this.offset=0;this.sig=null;
 this.mode="collect";this.cnt={rows:0,used:0,other:0,equip:0,bad:0,dup:0,late:0,ooo:0};this.err=null;this.P={};
 for(const p of ["CN7","RG3"])this.P[p]={pr:p,eng:new Engine(LIVE[p].model,p),S:[],rows:[],pend:null,lastK:-Infinity,alarms:[],
  st:{shots:0,parts:0,score:0,held:0,al:0,ins:0,fb:0,lab:0,def:0,cau:0,overlap:0}};}
FileSession.prototype.header=function(line){const h=splitCSV(line.replace(/^﻿/,"")).map(s=>s.trim());const ix={};h.forEach((n,k)=>{if(!(n in ix))ix[n]=k;});
 const miss=FREQ.filter(c=>!(c in ix));if(miss.length)throw new Error("필수 열이 없음: "+miss.join(", "));
 this.col=ix;this.ncol=h.length;this.labels="PassOrFail" in ix;this.hasEquip="EQUIP_NAME" in ix;};
FileSession.prototype.feedText=function(text,final){text=this.rem+text;let a=0;
 for(;;){const b=text.indexOf("\n",a);if(b<0)break;let line=text.slice(a,b);a=b+1;if(line.endsWith("\r"))line=line.slice(0,-1);if(line.length)this.line(line);}
 this.rem=text.slice(a);if(final&&this.rem.trim().length){const l=this.rem.replace(/\r$/,"");this.rem="";this.line(l);}};
FileSession.prototype.feedBytes=function(u8,final){this.feedText(this.dec.decode(u8,{stream:!final}),final);};
FileSession.prototype.line=function(line){if(!this.col){this.header(line);return;}
 const f=splitCSV(line),C=this.col,rw=this.rewriting;if(!rw)this.cnt.rows++;
 const name=(f[C.PART_NAME]||"").trim();const pr=name.startsWith("CN7")?"CN7":name.startsWith("RG3")?"RG3":null;
 if(!pr){if(!rw)this.cnt.other++;return;}
 if(this.hasEquip&&!this.opt.anyEquip&&(f[C.EQUIP_NAME]||"").trim()!==CFG.EQUIP){if(!rw)this.cnt.equip++;return;}
 const v=c=>num(f[C[c]]);const k=tsKey(f[C.TimeStamp]||"");
 const Tm=(v("Barrel_Temperature_1")+v("Barrel_Temperature_2")+v("Barrel_Temperature_3"))/3,Tw=(v("Mold_Temperature_3")+v("Mold_Temperature_4"))/2;
 const sp=v("Max_Injection_Speed"),fi=v("Filling_Time"),inj=v("Injection_Time"),hop=v("Hopper_Temperature"),pl=v("Plasticizing_Position");
 if(!isFinite(k)||![Tm,Tw,sp,fi,inj,hop,pl].every(isFinite)){if(!rw)this.cnt.bad++;return;}
 let y=null,rs="";if(this.labels){const pf=(f[C.PassOrFail]||"").trim();y=pf==="N"?1:pf==="Y"?0:null;if("Reason" in C)rs=(f[C.Reason]||"").trim();if(rs==="None")rs="";}const rs0=rs;rs=esc(rs);
 const ser=(f[C.PART_FACT_SERIAL]||"").trim();
 const row={k,name,n:name.slice(-2),rh:name.endsWith("RH")?1:0,lotd:(f[C.PART_FACT_PLAN_DATE]||"").trim(),ser,sern:/^-?\d+(\.\d+)?$/.test(ser)?+ser:NaN,
  x:[sp,fi,inj,flr(fi,Tm,Tw)],g:[Tm,Tw,hop,pl],y,rs,rs0};
 const P=this.P[pr];
 if(this.mode==="collect"){if(row.k>P.lastK){P.rows.push(row);if(rw)this.cnt.rows++;}else if(!rw)this.cnt.late++;return;}
 this.stream(P,row);};
FileSession.prototype.stream=function(P,row){
 if(row.k<=P.lastK){row.k===P.lastK?this.cnt.late++:this.cnt.ooo++;return;}
 if(P.pend&&row.k===P.pend.k){if(P.pend.rows.some(r=>r.name===row.name))this.cnt.dup++;else P.pend.rows.push(row);return;}
 if(P.pend&&row.k<P.pend.k){this.cnt.ooo++;return;}
 if(P.pend)this.flushP(P);P.pend={k:row.k,rows:[row]};};
FileSession.prototype.flushP=function(P){const g=P.pend;P.pend=null;if(!g)return;this.shot(P,g.rows);P.lastK=g.k;};
FileSession.prototype.flushAll=function(){let n=0;for(const p in this.P)if(this.P[p].pend){this.flushP(this.P[p]);n++;}return n;};
FileSession.prototype.shot=function(P,rows){const r0=rows[0],pr=P.pr,M=LIVE[pr].model;
 const sers=rows.map(r=>r.sern),lot=r0.lotd+"_"+(sers.every(isFinite)?String(Math.min(...sers)):rows.map(r=>r.ser).sort()[0]);
 const sh={lot,P:rows.map(r=>[r.n,r.rh])};const r=P.eng.step(sh,r0.x.slice(),r0.g.slice());this.cnt.used+=rows.length;
 const T=tsStr(r0.k),recv=this.opt.watch?hhmmss(new Date()):"";
 const Pp=rows.map((q,j)=>r.held?{n:q.n,nm:q.name,y:q.y,p:null,pct:null,tag:"-",rs:q.rs,rs0:q.rs0}:{n:q.n,nm:q.name,y:q.y,p:r.parts[j].p,pct:r.parts[j].pct,tag:r.parts[j].tag,rs:q.rs,rs0:q.rs0});
 const rec={t:T.slice(5,16).replace("-","/"),T,seg:P.eng.lotn,lotn:P.eng.lotn,k:P.eng.k,s:r.held?"held":"score",D:r.held?null:r.D,x:r0.x,
  z:r.held?null:r.z,m:r.held?null:r.m,sc:M.scale,rsn:r.held?r.reason:"",P:Pp,inj:null,recv};
 Pp.sort((a,b)=>a.n<b.n?-1:a.n>b.n?1:0);                     // 화면 표시는 LH·RH 순 (판정 계산은 위에서 이미 끝남)
 const st=P.st,S=P.S;st.shots++;r.held?st.held++:st.score++;
 const al=!r.held&&r.D>=CFG.ALARM;if(al){st.al++;const pv=S[S.length-1];if(!(pv&&pv.s==="score"&&pv.D>=CFG.ALARM))P.alarms.push(S.length);}
 for(const p of Pp){st.parts++;const f=inspectP(rec,p,pr);st.ins+=f;st.fb+=rec.s!=="score";if(p.y!=null){st.lab++;st.def+=p.y;st.cau+=f&&p.y;}}
 if(this.labels&&r0.k>=LRANGE[pr][0]&&r0.k<=LRANGE[pr][1])st.overlap++;
 S.push(rec);};
// 모아 둔 행 → 시간순 정렬 → 샷 묶기 → 판정 (holdLast: 마지막 샷은 다음 행이 올 때까지 보류 = 실시간 감시용)
FileSession.prototype.processCollected=async function(holdLast,onProg){let tot=0,done=0;for(const p in this.P)tot+=this.P[p].rows.length;
 for(const p in this.P){const P=this.P[p],R=P.rows;P.rows=[];R.sort((a,b)=>a.k-b.k);let g=[],gk=null,t0=performance.now();
  for(let j=0;j<=R.length;j++){const row=R[j];
   if(row&&row.k===gk){if(g.some(q=>q.name===row.name))this.cnt.dup++;else g.push(row);continue;}
   if(g.length){if(!row&&holdLast)P.pend={k:gk,rows:g};else{this.shot(P,g);P.lastK=gk;}}
   if(!row)break;g=[row];gk=row.k;done++;
   if(performance.now()-t0>40){t0=performance.now();if(onProg)onProg(done/Math.max(1,tot));await new Promise(r=>setTimeout(r,0));}}
  done=Math.max(done,0);}
 this.mode="stream";};
async function sigAt(file,a,b){const u=new Uint8Array(await file.slice(a,b).arrayBuffer());let s="";for(const x of u)s+=(x<16?"0":"")+x.toString(16);return s;}
async function markTail(sess,file){sess.tailFrom=Math.max(0,sess.offset-4096);sess.tail=await sigAt(file,sess.tailFrom,sess.offset);}   // 이미 읽은 끝 4KB: 다음 확인 때 그대로인지 본다
async function headSig(file,n){const b=new Uint8Array(await file.slice(0,Math.min(n||4096,file.size)).arrayBuffer());let s="";for(const x of b)s+=(x<16?"0":"")+x.toString(16);return s;}
// 파일 전체 읽기 (모으기 모드). filterOld: 이미 판정한 시각 이전 행은 버림(파일이 통째로 다시 쓰였을 때)
async function readAll(sess,file,onProg){sess.enc=await sniffEnc(file);sess.dec=new TextDecoder(sess.enc);sess.rem="";sess.col=null;sess.mode="collect";
 for(const p in sess.P)sess.P[p].pend=null;
 const rd=file.stream().getReader();let got=0,t0=performance.now();
 for(;;){const {value,done}=await rd.read();if(done)break;sess.feedBytes(value,false);got+=value.length;
  if(performance.now()-t0>60){t0=performance.now();onProg&&onProg(got/Math.max(1,file.size));await new Promise(r=>setTimeout(r,0));}}
 sess.feedBytes(new Uint8Array(0),!sess.opt.watch);sess.offset=file.size;sess.sig=await headSig(file);await markTail(sess,file);}

let FS=null,WATCH=null,FOLLOW=true;
function fstatus(h,cls){$("fstat").innerHTML=h;$("fstat").className="lbl fstat "+(cls||"");}
function fprog(f,label){const b=$("fbar");if(f==null){b.classList.add("hide");return;}b.classList.remove("hide");b.firstElementChild.style.width=(100*f).toFixed(1)+"%";if(label)fstatus(label);}
let LOADING=false;
function setBusy(on){LOADING=on;for(const id of ["fpick","fwatch"])$(id).disabled=on;$("fdrop").style.pointerEvents=on?"none":"";}
// 새 파일은 끝까지 성공해야 화면 결과를 바꾼다. 실패하면 이전 결과(파일 이름 표시)를 그대로 둔다.
async function openFile(file,watch){if(LOADING)return null;setBusy(true);stopWatch();
 const opt={watch:!!watch,anyEquip:$("fany").checked};const sess=new FileSession(file.name,opt),t0=performance.now();
 try{await readAll(sess,file,f=>fprog(f*.5,`읽는 중 · ${esc(file.name)} · ${(file.size/1e6).toFixed(1)} MB`));
  if(!sess.col)throw new Error("빈 파일이거나 머리글 줄이 없음");
  await sess.processCollected(!!watch,f=>fprog(.5+f*.5,`판정 중 · ${esc(file.name)}`));}
 catch(e){fprog(null);setBusy(false);fstatus("불러오기 실패: "+esc(e.message)+(FS?` · 이전 결과(${FS.dname}) 그대로 표시 중`:""),"bad");return null;}
 fprog(null);setBusy(false);FS=sess;sess.secs=(performance.now()-t0)/1000;
 if(!sess.P[prod].S.length){const o=prod==="CN7"?"RG3":"CN7";if(sess.P[o].S.length||sess.P[o].pend)selectProd(o,true);}
 showFile(true);return sess;}
function stopWatch(){if(WATCH){clearTimeout(WATCH.timer);WATCH=null;}$("fwatch").classList.remove("on");$("fwatch").textContent="실시간 감시 시작";$("fstop").classList.add("hide");}
async function startWatch(){if(LOADING)return;if(!window.showOpenFilePicker){fstatus("이 브라우저는 파일 감시를 지원하지 않음 → Chrome 또는 Edge에서 열면 됨. ('CSV 열기'로 한 번에 처리는 가능)","bad");return;}
 let h;try{[h]=await window.showOpenFilePicker({types:[{description:"CSV",accept:{"text/csv":[".csv",".txt"]}}]});}catch(e){return;}
 const f=await h.getFile();const s=await openFile(f,true);if(!s)return;
 WATCH={h,size:f.size,mod:f.lastModified,changed:Date.now(),polls:0,last:Date.now(),newAt:Date.now()};
 $("fwatch").classList.add("on");$("fwatch").textContent="감시 중";$("fstop").classList.remove("hide");schedPoll();}
function schedPoll(){if(!WATCH)return;WATCH.timer=setTimeout(poll,+$("fint").value);}
// 감시 한 번: W(이번 감시 세대)가 바뀌었으면(중지·다른 파일) 늦게 끝난 결과는 버리고 다음 예약도 하지 않는다
async function poll(){const W=WATCH;if(!W)return;const s=FS;W.polls++;W.last=Date.now();
 try{const f=await W.h.getFile();if(WATCH!==W)return;
  if(f.size===W.size&&f.lastModified===W.mod){if(Date.now()-W.changed>Math.max(3000,2*+$("fint").value)&&s.flushAll()){W.newAt=Date.now();refreshFile();}}
  else{const before=totShots(s);const hs=await headSig(f,s.sig.length/2);
   const ts=f.size>=s.offset?await sigAt(f,s.tailFrom,s.offset):null;if(WATCH!==W)return;
   if(f.size<s.offset||hs!==s.sig||ts!==s.tail){s.rewrites=(s.rewrites||0)+1;s.rewriting=true;try{await readAll(s,f);}finally{s.rewriting=false;}if(WATCH!==W)return;await s.processCollected(true);}
   else{const buf=new Uint8Array(await f.slice(s.offset).arrayBuffer());if(WATCH!==W)return;s.feedBytes(buf,false);s.offset+=buf.length;await markTail(s,f);}
   W.size=f.size;W.mod=f.lastModified;W.changed=Date.now();W.lastNew=totShots(s)-before;if(W.lastNew>0)W.newAt=Date.now();refreshFile();}
  W.err=null;}
 catch(e){if(WATCH!==W)return;W.err=e.message;}
 watchStatus();if(WATCH===W)schedPoll();}
function totShots(s){return s.P.CN7.S.length+s.P.RG3.S.length;}
function watchStatus(){if(!FS)return;const s=FS,c=s.cnt;let h=`<b>${s.dname}</b> · 읽은 행 ${c.rows.toLocaleString()} · 판정에 쓴 행 ${c.used.toLocaleString()}`;
 if(WATCH){const pend=s.P.CN7.pend||s.P.RG3.pend;h=`<span class="livedot"></span>`+h+` · 마지막 확인 ${hhmmss(new Date(WATCH.last))} (${(+$("fint").value/1000)}초마다)`+(pend?` · 다음 행을 기다리는 샷 1개`:"")+` · 마지막 새 샷 ${hhmmss(new Date(WATCH.newAt))}`+(Date.now()-WATCH.newAt>180000?` · <span style="color:var(--red);font-weight:700">입력 지연: 3분 넘게 새 샷 없음 (화면은 마지막 결과)</span>`:"")+(s.rewrites?` · <span style="color:var(--amber)">이미 읽은 부분이 바뀐 것 ${s.rewrites}회 감지 → 이미 판정한 샷은 그대로, 그 뒤 시각 행만 반영</span>`:"")+(WATCH.err?` · <span style="color:var(--red)">읽기 재시도 중 (${WATCH.err})</span>`:"");}
 else h+=` · 처리 ${s.secs.toFixed(1)}초`;
 fstatus(h);}
function refreshFile(){if(MODE!=="file"||!FS)return;const P=FS.P[prod];S=P.S;N=S.length;$("slider").max=Math.max(0,N-1);
 if(!N){showFile(false);return;}$("fintro").classList.add("hide");$("main").classList.remove("hide");
 if(FOLLOW||i>=N)i=N-1;$("slider").value=i;fevents();draw();}
function showFile(ok){const has=FS&&FS.P[prod].S.length;$("fsave").classList.toggle("hide",!FS||!totShots(FS));for(const id of ["play","spdl","spd"])$(id).classList.toggle("hide",!has);$("rctl").style.display=has?"contents":"none";$("fintro").classList.toggle("hide",!!has);$("main").classList.toggle("hide",!has);$("fev").classList.toggle("hide",!has);$("fmsg").style.display=FS&&!has?"block":"none";
 if(FS&&!has)$("fmsg").innerHTML=`<b>${FS.dname}</b>에서 ${prod} 샷을 찾지 못함 (읽은 행 ${FS.cnt.rows.toLocaleString()}, 다른 품목 ${FS.cnt.other.toLocaleString()}, 다른 설비 ${FS.cnt.equip.toLocaleString()}). 위 품목 버튼으로 바꾸거나 '설비명 무시'를 켜고 다시 열어 보면 됨.`;
 if(has){FOLLOW=true;$("ffollow").checked=true;S=FS.P[prod].S;N=S.length;i=N-1;$("slider").max=N-1;$("slider").value=i;fevents();draw();}
 watchStatus();setFileFoot();}
function fevents(){const P=FS.P[prod],ev=$("fev");const A=P.alarms;
 let h=`<span class="lbl" style="font-size:12px">경보 바로 가기 (${A.length}건${A.length>40?", 최근 40건 표시":""})</span><button data-j="prev">◀ 이전 경보</button><button data-j="next">다음 경보 ▶</button>`;
 for(const j of A.slice(-40)){const s=P.S[j];h+=`<button data-j="${j}" title="D ${fmt(s.D,1)}">${s.t} ${(s.P[0].tag||"").split(" ")[0]}</button>`;}
 if(!A.length)h+=`<span class="lbl" style="font-size:12px">경보 없음</span>`;
 if(ev.dataset.sig!==String(A.length)+prod){ev.innerHTML=h;ev.dataset.sig=String(A.length)+prod;}}
$("fev").addEventListener("click",e=>{const b=e.target.closest("button");if(!b||!FS)return;const A=FS.P[prod].alarms;let j=b.dataset.j;
 if(j==="prev"){const c=A.filter(a=>a<i);if(!c.length)return;j=c[c.length-1];}else if(j==="next"){const c=A.filter(a=>a>i);if(!c.length)return;j=c[0];}
 FOLLOW=false;$("ffollow").checked=false;stop();go(+j);});
function p5file(s){const P=FS.P[prod],t=P.st,lab=FS.labels&&t.lab>0;
 let h=`<table class="cnt"><tr><th>누적 (파일 ${WATCH?"감시 시작부터":"전체"})</th><th>값</th></tr>
 <tr><td>판정한 샷 / 부품</td><td class="num">${t.shots.toLocaleString()} / ${t.parts.toLocaleString()}</td></tr>
 <tr><td>점수 / 보류 샷</td><td class="num">${t.score.toLocaleString()} / ${t.held.toLocaleString()}</td></tr>
 <tr><td>경보 샷 (D ≥ ${CFG.ALARM})</td><td class="num"><b>${t.al.toLocaleString()}</b></td></tr>
 <tr><td>검사 대상 부품 (비율)</td><td class="num"><b>${t.ins.toLocaleString()} (${t.parts?(t.ins/t.parts*100).toFixed(1):"–"}%)</b></td></tr>
 <tr><td>그중 대체검사 (보류)</td><td class="num">${t.fb.toLocaleString()}</td></tr>`;
 if(lab)h+=`<tr><td>파일에 기록된 불량</td><td class="num">${t.def.toLocaleString()}</td></tr><tr><td>검사 대상에 들어간 불량</td><td class="num"><b>${t.cau}/${t.def}</b></td></tr>`;
 h+=`</table>`;
 if(lab&&t.overlap)h+=`<div class="banner red" style="margin-top:6px">이 파일의 ${t.overlap.toLocaleString()}샷은 최종 모델이 학습한 라벨 기간과 겹침 → 위 불량 선정 수는 성능이 아님 (성능은 재생 모드·표 4-2).</div>`;
 const c=FS.cnt,sk=[["다른 품목",c.other],["다른 설비",c.equip],["값 누락·형식 오류",c.bad],["중복 행",c.dup],["늦게 도착한 행",c.late],["시간 역순 행",c.ooo]].filter(a=>a[1]);
 h+=`<div class="note">규칙: ${prod==="CN7"?"경보 부품(D ≥ 5) → 확률 높은 순":"RH 부품 우선"}, 보류 부품 → 대체검사.${sk.length?" 판정에서 뺀 행: "+sk.map(a=>a[0]+" "+a[1].toLocaleString()).join(" · "):""}</div>`;
 const pend=[];for(let j=Math.max(0,i-9);j<=i;j++){const q=S[j];for(const p of q.P)if(inspectP(q,p,prod))pend.push({j,t:q,p});}
 pend.sort((a,b)=>{const ra=a.t.s!=="score"?2:0,rb=b.t.s!=="score"?2:0;if(ra!==rb)return ra-rb;if(ra===0)return(b.p.p||0)-(a.p.p||0);return b.j-a.j;});
 h+=`<div class="qh">검사 대상 (이 샷까지 최근 10샷) ${pend.length}건</div><ul class="q">`;
 pend.slice(0,8).forEach(o=>{const [kt,kc]=kind(o.t,o.p);const yb=o.p.y==null?"":o.p.y?` <span class="badge b-red">불량</span>`:` <span class="badge b-ok">양품</span>`;h+=`<li><span class="tm">${o.t.t} ${o.p.n}</span><span><span class="badge ${kc}">${kt}</span>${yb}</span><span class="num">${o.t.s==="score"?pct(o.p.p):"–"}</span></li>`;});
 if(!pend.length)h+=`<li style="grid-template-columns:1fr;color:var(--sub)">검사 대상 없음 — 정상 생산 계속</li>`;
 $("p5").innerHTML=h+`</ul>${pend.length>8?`<div class="more">외 ${pend.length-8}건</div>`:""}`;}
function setFileFoot(){$("foot").textContent=`파일 모드: 넣은 CSV(KAMP 사출 데이터 형식)를 실시간 모드와 같은 판정 엔진(model.RiskModel.step을 옮긴 것, 현장 투입용 최종 모델)으로 처리한다. 행 → 파생값(용융온도 Tm = 배럴 1~3 평균, 금형온도 Tw = 금형 3·4 평균, FLR(고화층 비율) 해석식) → 같은 시각 행을 한 샷으로 묶음 → 샷마다 판정. 파일은 브라우저 안에서만 읽고 어디로도 보내지 않는다. 실시간 감시는 Chrome·Edge에서 같은 파일을 주기적으로 다시 확인해 새로 붙은 행만 읽는다(파일이 통째로 다시 쓰이면 이미 판정한 시각 이후 행만 반영). 같은 시각의 LH·RH 행이 모두 들어올 때까지 한 샷은 다음 행을 기다리며, 일정 시간 새 행이 없으면 그대로 판정한다.`;}
function selectProd(p,silent){prod=p;document.querySelectorAll(".seg-btn").forEach(x=>x.classList.toggle("on",x.dataset.p===p));}
$("fpick").onclick=()=>$("ffile").click();$("fdrop").onclick=()=>$("ffile").click();
$("ffile").onchange=e=>{const f=e.target.files[0];if(f)openFile(f,false);e.target.value="";};
$("fwatch").onclick=()=>{if(!WATCH)startWatch();};$("fstop").onclick=()=>{stopWatch();if(FS){FS.flushAll();FS.opt.watch=false;refreshFile();watchStatus();}};
$("ffollow").onchange=e=>{FOLLOW=e.target.checked;if(FOLLOW&&FS){i=N-1;go(i);}};
const dz=$("fdrop");["dragover","dragenter"].forEach(k=>dz.addEventListener(k,e=>{e.preventDefault();dz.classList.add("over");}));
["dragleave","drop"].forEach(k=>dz.addEventListener(k,e=>{e.preventDefault();dz.classList.remove("over");}));
dz.addEventListener("drop",e=>{const f=e.dataTransfer.files[0];if(f)openFile(f,false);});
document.addEventListener("dragover",e=>{if(MODE==="file")e.preventDefault();});
document.addEventListener("drop",e=>{if(MODE!=="file")return;e.preventDefault();const f=e.dataTransfer.files[0];if(f)openFile(f,false);});
if(!window.showOpenFilePicker){$("fwatch").title="Chrome·Edge에서만 지원";}
function csvCell(v){if(v==null)return"";if(typeof v==="number")return String(v);let t=String(v);if(/^[=+\-@\t\r]/.test(t))t="'"+t;   // 엑셀 수식으로 읽히지 않게
 return /[",\n\r]/.test(t)?'"'+t.replace(/"/g,'""')+'"':t;}
$("fsave").onclick=()=>{if(!FS)return;const H=["품목","TimeStamp","PART_NAME","판정","보류사유","이탈점수_D","불량확률","위험점수","원인태그","경보(D>=5)","검사대상","z_최대사출속도","z_충진시간","z_사출시간","z_FLR","작업오더순번","오더내샷"].concat(FS.labels?["실제불량(PassOrFail=N)","Reason"]:[]);
 const L=[H.join(",")];for(const pr of ["CN7","RG3"])for(const s of FS.P[pr].S)for(const p of s.P){const sc=s.s==="score";
  const r=[pr,s.T,p.nm,sc?"점수":"보류",s.rsn,sc?s.D:null,p.p,p.pct,p.tag==="-"?"":p.tag,sc&&s.D>=CFG.ALARM?1:0,inspectP(s,p,pr)?1:0].concat(sc?s.z:[null,null,null,null]).concat([s.lotn,s.k]);
  if(FS.labels)r.push(p.y,p.rs0);L.push(r.map(csvCell).join(","));}
 const blob=new Blob(["\uFEFF"+L.join("\r\n")+"\r\n"],{type:"text/csv;charset=utf-8"}),a=document.createElement("a");
 a.href=URL.createObjectURL(blob);a.download=FS.name.replace(/\.[^.]+$/,"")+"_PINNSET판정.csv";document.body.appendChild(a);a.click();setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove();},1000);};
function unf(){if(MODE==="file"){FOLLOW=false;$("ffollow").checked=false;}}
$("slider").addEventListener("input",unf);$("prev").addEventListener("click",unf);$("next").addEventListener("click",unf);cv.addEventListener("click",unf);
prep();setMode("replay");requestAnimationFrame(t=>{last=t;tick(t);});
</script></body></html>"""


def build(data, results, out=None):
    data, results = Path(data).resolve(), Path(results).resolve()
    out = Path(out) if out else results / "PINNSET_대시보드.html"
    log("[대시보드] 데이터 읽기")
    L, U = load_labeled(str(data)), load_unlabeled(str(data))
    payload, checks = {}, {}
    for prod in CFG["PRODUCTS"]:
        log(f"[대시보드] {prod} 엄격 시간 순서 폴드 재계산")
        parts, S, R = recompute(prod, L, U)
        M, chk = merge_and_check(prod, parts, S, R, results / "strict" / f"predictions_forward_{prod}.csv")
        shots = shot_payload(prod, M, S)
        cc = counters_check(prod, shots)
        chk["queue"] = cc
        if prod == "CN7":
            assert (cc["after_calib"]["inspected"], cc["after_calib"]["caught"], cc["after_calib"]["defects"]) == (165, 10, 11)
            assert (cc["all"]["inspected"], cc["all"]["caught"], cc["all"]["defects"]) == (540, 27, 28)
            assert (cc["scored"]["inspected"], cc["scored"]["caught"], cc["scored"]["defects"]) == (20, 10, 11)
        else:
            assert (cc["all"]["inspected"], cc["all"]["caught"]) == (745, 27)
            assert (cc["scored"]["inspected"], cc["scored"]["caught"]) == (511, 22)
        log(f"  {prod} 대기열 누적 일치: {cc}")
        payload[prod] = shots; checks[prod] = chk
    log("[대시보드] 실시간 모드 입력 준비 (최종 모델 + 무라벨 실데이터)")
    inj = injection_ratios(payload["CN7"])
    live = {prod: live_payload(prod, L, U, inj) for prod in CFG["PRODUCTS"]}
    for prod in live:
        checks[prod]["live"] = dict(range=live[prod]["range"], counts=live[prod]["counts"], alarms=live[prod]["alarms"])
    action = json.loads((results / "action_results.json").read_text(encoding="utf-8"))
    html = render(payload, checks, load_cards(results), action, live)
    out.write_text(html, encoding="utf-8")
    (results / "dashboard_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log(f"[대시보드] 완료: {out} ({out.stat().st_size/1e6:.1f} MB)")
    return out
