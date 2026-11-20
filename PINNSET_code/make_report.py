"""결과 JSON들을 읽어 PINNSET_검증리포트.html 한 파일로 만든다 (그림은 파일 안에 포함)"""
import json, base64, io, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # Windows(CP949) 콘솔에서도 기호 출력 오류가 나지 않게
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

KFONT_FILES = ["C:/Windows/Fonts/malgun.ttf", "/System/Library/Fonts/AppleSDGothicNeo.ttc", "/Library/Fonts/AppleGothic.ttf",
               "/System/Library/Fonts/Supplemental/AppleGothic.ttf"]
KFONT_GLOBS = ["/usr/share/fonts/**/*NanumGothic*.ttf", "/usr/share/fonts/**/NotoSansCJK-Regular.ttc", "/usr/share/fonts/**/NotoSansKR*",
               "/usr/share/fonts/**/*CJK*Regular*", "~/.fonts/**/*Nanum*", "~/Library/Fonts/*Nanum*",
               "/usr/local/share/fonts/**/*Nanum*", "/usr/local/share/fonts/**/*CJK*", "~/.local/share/fonts/**/*Nanum*", "~/.local/share/fonts/**/*CJK*"]

def _font():
    """한글 폰트 파일을 직접 찾아 등록한다. 이름으로만 찾으면 CJK 폰트가 'JP'로 등록된 환경에서 한글이 네모로 깨진다."""
    import glob
    cands = [f for f in KFONT_FILES if os.path.exists(f)]
    for g in KFONT_GLOBS: cands += sorted(glob.glob(os.path.expanduser(g), recursive=True))
    for f in cands:
        try:
            font_manager.fontManager.addfont(f)
            name = font_manager.FontProperties(fname=f).get_name()
            plt.rcParams["font.family"] = name; plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    print("경고: 한글 폰트를 찾지 못했습니다. 그림 속 한글이 깨질 수 있습니다 (표·본문은 정상).")
    return None
KFONT = _font()

def img(fig):
    b = io.BytesIO(); fig.savefig(b, format="png", dpi=130, bbox_inches="tight"); plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}">'

# 재가동 규칙 표: 결과 파일의 내부 키(대시보드가 사용)는 그대로 두고 화면 표시 이름만 바꾼다
RESTART_LABEL = {"폐기 샷": "검사 대기 분리 샷", "잘못 폐기한 양품 샷": "분리 대상에 포함된 양품 샷"}

def tbl(rows, fmt=None):
    df = pd.DataFrame(rows)
    def f(v):
        if v is None: return "-"                     # JSON의 null(이전 NaN)
        if isinstance(v, float):
            return "-" if np.isnan(v) else (f"{v:.4f}" if abs(v) < 0.1 and v != 0 else f"{v:.3f}")
        return str(v)
    h = "".join(f"<th>{c}</th>" for c in df.columns)
    b = "".join("<tr>" + "".join(f"<td>{f(v)}</td>" for v in r) + "</tr>" for r in df.itertuples(index=False))
    return f"<table><tr>{h}</tr>{b}</table>"

def box(t, cls="info"): return f'<div class="box {cls}">{t}</div>'

CSS = """body{font-family:'Malgun Gothic','맑은 고딕','Apple SD Gothic Neo','Noto Sans KR','Noto Sans CJK KR','NanumGothic',sans-serif;max-width:980px;margin:24px auto;padding:0 16px;color:#1d2330;line-height:1.6;font-size:14px}
h1{font-size:26px}h2{color:#1d3b6e;border-bottom:2px solid #1d3b6e;padding-bottom:3px;margin-top:34px}h3{margin-top:20px}
table{border-collapse:collapse;width:100%;margin:6px 0 12px;font-size:12.5px}th{background:#1d3b6e;color:#fff;padding:5px;text-align:left}
td{border-bottom:1px solid #dde1e8;padding:4px 5px}tr:nth-child(even) td{background:#f7f8fa}
img{max-width:100%;border:1px solid #e3e6eb}.box{border-radius:8px;padding:10px 14px;margin:10px 0}
.info{background:#eef3fb;border-left:6px solid #1d3b6e}.warn{background:#fdf6e7;border-left:6px solid #b7791f}.small{font-size:12px;color:#666}"""

def build(d="."):
    v0 = json.load(open(f"{d}/risk_v0_results.json", encoding="utf-8")); v1 = json.load(open(f"{d}/risk_v1_validation.json", encoding="utf-8"))
    ex = json.load(open(f"{d}/extra_results.json", encoding="utf-8")); info = json.load(open(f"{d}/run_info.json", encoding="utf-8")) if os.path.exists(f"{d}/run_info.json") else {}
    H = [f"<html><head><meta charset='utf-8'><title>PINNSET 검증 리포트</title><style>{CSS}</style></head><body>",
         "<h1>PINNSET 검증 리포트 (자동 생성)</h1>",
         f"<p class='small'>run_all.py가 만든 결과 파일에서 자동으로 작성됨. 환경: {info.get('env', {})} · 소요 {info.get('elapsed_s', {})}</p>"]
    # ---------------- 요약
    rows = []
    for p in ["CN7", "RG3"]:
        o = v0[p]; t = v1[p]["3_시간순서"]
        cm = {(r["product"], r["mode"], r["model"]): r for r in json.load(open(f"{d}/strict/comparisons.json", encoding="utf-8"))} if os.path.exists(f"{d}/strict/comparisons.json") else {}
        f_ = cm.get((p, "forward", "PINNSET"), {}); l_ = cm.get((p, "leave_block_out", "PINNSET"), {})
        rows.append({"제품": p, "사건 블록 제외 구간 AUC (공식)": l_.get("AUC_within_segment"), "사건 블록 제외 Brier 개선": l_.get("Brier_improvement"),
                     "엄격 시간 순서 구간 AUC (공식)": f_.get("AUC_within_segment"), "엄격 시간 순서 Brier 개선": f_.get("Brier_improvement"),
                     "공정 신호 p": o["significance"]["process"]["p"], "캐비티 짝비교 p": o["significance"]["cavity"]["p_mcnemar"],
                     "고정 기준 구간 빼기 AUC (참고)": o["loso"]["auc_within"]})
    H += ["<h2>0. 요약</h2>", tbl(rows),
          box("공식 성능은 엄격 검증(E12: 전처리까지 학습 구간 안에서 계산)이다. '고정 기준' 표시는 정상 기준을 전체 기간 무라벨로 고정한 참고 평가다. 구간 AUC = 구간별 점수를 순위 백분위로 바꾼 뒤(동점은 평균 순위) 전체 평가 부품을 모아 ROC-AUC를 계산한 값 (0.5=무작위). "
              "Brier 개선 = 불량률 하나로 답하는 상수 모델 대비 확률 오차 감소율. 공정 신호 p = 원형 이동 검정(3,000회).")]
    # ---------------- 1 진단
    e1 = ex["E1_진단"]; L = e1["labeled"]; U = e1["unlabeled"]
    H += ["<h2>1. 데이터 진단</h2>",
          tbl([{"항목": "라벨 원본 행", "값": L["원본행"]}, {"항목": "중복 행 (같은 시각·시리얼·부품)", "값": f"{L['중복행']} (시작 {L['중복_시작시각']})"},
               {"항목": "라벨 결측 최대 비율", "값": f"{L['결측률_최대']:.4%}"}, {"항목": "라벨 결측 열", "값": L["결측열"] or "없음"},
               {"항목": "상수 열", "값": ", ".join(L["상수열"]) or "없음"},
               {"항목": "무라벨 행 / 설비 / 품목 / 작업오더", "값": f"{U['행']:,} / {U['설비']} / {U['품목']} / {U['작업오더']:,}"},
               {"항목": "무라벨 결측 최대 비율", "값": f"{U['결측률_최대']:.4%}"}, {"항목": "무라벨 결측 열", "값": U["결측열"] or "없음"},
               {"항목": "무라벨 기간", "값": " ~ ".join(U["기간"])}]),
          "<h3>제품별 불균형과 라벨 충돌</h3>",
          tbl([{"제품": p, "부품": r["부품"], "불량": r["불량"], "불량률": f"{r['불량률']:.2%}", "RH 불량": r["RH불량"], "LH 불량": r["LH불량"],
                "라벨 충돌 행": r["라벨충돌_행"], "LH/RH 구분 시 충돌": r["라벨충돌_행_LHRH구분"], "불량 사유": r["불량사유"]} for p, r in L["제품별"].items()]),
          box("라벨 충돌 = 36개 공정값이 완전히 같은데 판정이 다른 행. 공정값만으로는 원리적으로 구분 불가한 불량이며 F1 상한을 만든다."),
          "<h3>극단값 (제품별 중앙값에서 로버스트 z 10 초과, 측정 오류 후보)</h3>",
          tbl([{"제품": p, "열별 개수": r} for p, r in L["극단값_z10"].items()])]
    # ---------------- 2 비교
    e2 = ex["E2_비교"]; H += ["<h2>2. 모델 비교 — 고정 기준 참고 평가 (같은 구간 빼기 조건, F1 포함)</h2>", box("이 장은 정상 기준을 전체 기간으로 고정한 참고 평가다. 공식 성능은 strict/ 폴더의 엄격 검증 결과(0장 요약 표, strict/model_comparison.csv)다."),
        box("모든 모델이 같은 시험 부품(PINNSET이 점수를 낸 부품)과 같은 구간 빼기 분할을 쓴다. 일반 ML 입력: 공정센서 36개 + 용융·금형온도·FLR·RH. "
            "결정규칙 F1: PINNSET은 동결 규칙(CN7 D≥5 경보, RG3 RH 우선), 일반 ML은 학습 구간 안 3겹 교차검증으로 F1이 최대인 임계값을 정해 적용(시험 라벨 미사용), 상수는 전수 검사. "
            "예산 F1: 구간마다 점수 상위 k%를 검사했을 때.")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 3.8)); fig.subplots_adjust(wspace=0.75)
    for i, p in enumerate(["CN7", "RG3"]):
        r = e2[p]; H += [f"<h3>{p} — 시험 부품 {r['부품']:,}개, 불량 {r['불량']}개, 예산 {int(r['예산']*100)}%</h3>", tbl(r["표"])]
        H += ["<p>원인별 사건 안 AUC (그 원인 불량 vs 같은 구간 양품)</p>", tbl([{"원인": c, **v} for c, v in r["원인별_AUC"].items()])]
        df = pd.DataFrame(r["표"]); ax[i].barh(df["모델"], df["구간 AUC"], color=["#999", "#1d3b6e", "#6b8fc9", "#b7791f", "#2f7d4f"][:len(df)])
        ax[i].axvline(0.5, ls="--", c="k", lw=0.8); ax[i].set_xlim(0, 1); ax[i].set_title(f"{p} 구간 AUC"); ax[i].invert_yaxis()
    H += [img(fig)]
    # ---------------- 3 보정·유의성·오경보
    H += ["<h2>3. 확률 보정표 · 유의성 · 경보율 — 고정 기준 참고 평가 (v0, 구간 빼기)</h2>", box("이 장의 보정표는 고정 기준 참고 평가다. 공식 성능은 strict/ 폴더의 엄격 검증 결과다. 무라벨 데이터에는 품질 판정이 없으므로 무라벨 경보 비율은 오경보율이 아니라 경보율로 표시한다.")]
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for i, p in enumerate(["CN7", "RG3"]):
        o = v0[p]; c = o["loso"]["calib"]; bins = list(c["n"].keys())
        H += [f"<h3>{p}</h3>", tbl([{"예측 구간": b, "부품": c["n"][b], "평균 예측": c["pred"][b], "실제 불량률": c["actual"][b]} for b in bins])]
        s = o["significance"]; fa = o["false_alarm"]; g = o["gate"]; cal = o["calibration"]
        H += [tbl([{"항목": "보정식 계수 / 캐비티 오즈비", "값": f"{np.round(cal['coef'], 2).tolist()} / 오즈비 {np.exp(cal['cavity_log_ratio']):.2f} / 성분 {cal['spec']}"},
                   {"항목": "공정 신호 원형 이동 검정", "값": f"AUC {s['process']['auc']:.3f}, 귀무 95% {s['process']['null_95']:.3f}, p={s['process']['p']:.4f}"},
                   {"항목": "캐비티 짝지은 비교", "값": f"RH만 {s['cavity']['rh_only']} vs LH만 {s['cavity']['lh_only']}, p={s['cavity']['p_mcnemar']:.4f}, 사건 부트스트랩 RH 비중 {np.round(s['cavity']['boot_share_95'], 2).tolist()} (사건 {s['cavity']['n_events']}개)"},
                   {"항목": "경보율 (D≥5)", "값": f"무라벨 기준 작업오더 경보율 {fa['tier1']:.2%} ({fa['n_tier1']:,}샷) / 라벨 양품 샷 오경보율 {fa['labeled_normal']:.2%} ({fa['n_labeled']:,}샷)"},
                   {"항목": "보류 샷", "값": f"{g['held_shots']} (다른 레시피 {g['recipe']}, 기준 준비 {g['not_ready']}), 보류된 불량 샷 {g['held_defect_shots']}"},
                   {"항목": "상위 5% 검사 검출 (전 구간 통합 순위)", "값": f"{o['loso']['top5']}/{o['loso']['n_def']}"}])]
        pr = [c["pred"][b] for b in bins]; ac = [c["actual"][b] for b in bins]
        ax[i].plot(pr, ac, "o-", c="#1d3b6e"); m = max(max(pr), max(ac)) * 1.1; ax[i].plot([0, m], [0, m], "--", c="grey")
        ax[i].set_xscale("symlog", linthresh=1e-3); ax[i].set_yscale("symlog", linthresh=1e-3)
        ax[i].set_xlabel("평균 예측 불량확률"); ax[i].set_ylabel("실제 불량률"); ax[i].set_title(f"{p} 보정 곡선 (점선=완벽 보정)")
    H += [img(fig)]
    # ---------------- 4 v1
    H += ["<h2>4. 샷별 예측 함수 검증 (v1)</h2>"]
    for p in ["CN7", "RG3"]:
        R = v1[p]; tag = pd.DataFrame(R["4_태그표"]).fillna(0).astype(int)
        H += [f"<h3>{p}</h3>", tbl([{"검증": "일괄 계산 = 샷별 계산", "결과": R["1_일괄=샷별"]}, {"검증": "미래 누설 (뒤 절반 뒤섞기)", "결과": R["2_미래누설"]},
                                   {"검증": "시간 순서 (앞선 구간만 학습)", "결과": R["3_시간순서"]}, {"검증": "무라벨 1층 태그 분포", "결과": R["5_무라벨_태그"]}]),
              "<p>원인 × 태그 (부품 수)</p>", tbl(tag.reset_index().rename(columns={"index": "원인"}).to_dict("records"))]
    # ---------------- 5 상호작용·FN/FP
    e3 = ex["E3_상호작용"]; e4 = ex["E4_FNFP"]; H += ["<h2>5. 영향변수 상호작용과 FN·FP 조건</h2>"]
    for p in ["CN7", "RG3"]:
        r = e3[p]
        H += [f"<h3>{p} — 이탈(D)과 캐비티(RH)</h3>",
              tbl([{"검정 (우도비)": "이탈 D 추가", "LR": r["가법모형_이탈_LR"][0], "p": r["가법모형_이탈_LR"][1]},
                   {"검정 (우도비)": "RH 추가", "LR": r["가법모형_캐비티_LR"][0], "p": r["가법모형_캐비티_LR"][1]},
                   {"검정 (우도비)": "이탈×RH 상호작용항 추가", "LR": r["상호작용항_LR"][0], "p": r["상호작용항_LR"][1]}]),
              tbl(r["이탈x캐비티"]), "<p>정지 후 경과 × 경보</p>", tbl(r["재가동x경보"])]
        if "방향x사건" in r: H += ["<p>방향 × 사건 (같은 구간 안 AUC)</p>", tbl([{"사건": k, **v} for k, v in r["방향x사건"].items()])]
        q = e4[p]
        H += [f"<h3>{p} — 운영 규칙(보류 + {'D≥5' if p == 'CN7' else 'RH'}) 적용 시 결과</h3>", tbl(q["원인별_검사"]), tbl(q["검사이유x결과"]),
              "<p>놓친 불량 (FN)</p>", tbl(q["FN"]) if q["FN"] else "<p>없음</p>"]
        if q["FP경보"]: H += ["<p>경보가 울렸지만 양품 (FP)</p>", tbl(q["FP경보"])]
        if "사건별" in q: H += ["<p>사건별 RH/LH 불량과 검출</p>", tbl(q["사건별"])]
    # ---------------- 6 민감도·보류
    e5 = ex["E5_민감도"]; e6 = ex["E6_보류"]
    H += ["<h2>6. v1 파라미터 민감도와 보류 규칙</h2>", "<h3>원인 태그 기준 TAG_Z</h3>"]
    for p in ["CN7", "RG3"]: H += [f"<p>{p}</p>", tbl(e5["TAG_Z"][p])]
    H += ["<h3>위험점수 비교 범위 (CN7, 시간 순서 예측)</h3>", tbl(e5["비교범위"]["CN7"]),
          box("RG3는 예측확률이 LH/RH 두 값뿐이라 백분위 점수가 두 값으로 뭉쳐 이 지표가 맞지 않는다.", "warn"), "<h3>보류 부품</h3>"]
    H += [tbl([{"제품": p, "전체 부품": r["전체부품"], "보류 부품": r["보류부품"], "보류 비율": f"{r['보류비율']:.2%}", "사유별": r["사유별"], "전체 불량률": f"{r['전체불량률']:.2%}"} for p, r in e6.items()])]
    if "E8_보류포함" in ex:
        e8 = ex["E8_보류포함"]
        H += ["<h2>6-2. 보류 부품을 포함한 확률 평가</h2>",
              box("동결 모델은 그대로 두고 평가 방식만 추가. 강제 점수 = 문을 무시하고 모델 확률을 줌(기준 없는 샷은 사유별 경험률). "
                  "사유별 경험률 = 학습 구간에서 같은 보류 사유 부품의 불량률 (k+0.5)/(n+1). 전체 불량률 = 학습 구간 전체 불량률. 모두 구간 빼기 검증.")]
        for p in ["CN7", "RG3"]:
            r = e8[p]
            H += [f"<h3>{p} — 적용률 {r['적용률']:.1%}</h3>", tbl(r["표"]), tbl(r["보류사유별"]),
                  f"<p class='small'>강제 점수가 가능한 보류 부품 {r['강제점수_가능']}개: 이탈 점수 중앙 {r['강제점수_D중앙']}, 강제 확률 중앙 {r['강제점수_확률중앙']}, 그중 불량 {r['강제점수_불량']}개</p>"]
    if "E9_위치변수" in ex:
        e9 = ex["E9_위치변수"]
        H += ["<h2>6-3. 위치 변수 (쿠션 위치로 본 총 주입량, 탐색)</h2>",
              box("쿠션 위치 = 사출이 끝난 뒤 스크류 앞 수지의 끝점(두 캐비티 총 주입량의 지표). C 불량 샷의 쿠션이 같은 구간 양품과 같으면 총량은 정상이고 분배만 틀어진 것. 결과를 본 뒤의 탐색 분석.")]
        for p in ["CN7", "RG3"]: H += [f"<p>{p} 위치 변수 상태</p>", tbl(e9["변수상태"][p])]
        H += ["<p>쿠션 위치 검정 (구간 안 순위, 원형 이동 3,000회)</p>", tbl(e9["검정"])]
    if "E7_C단서_재현" in ex:
        e7 = ex["E7_C단서_재현"]; H += ["<h2>7. C 단서 재현 검정 (사전 고정 가설)</h2>",
            box(f"가설: 캐비티 불량 샷은 같은 구간 양품보다 {', '.join(k + '(' + v + ')' for k, v in e7['가설'].items())}. 방향은 CN7에서 정하고 RG3에서 단측 검정. 기준 α={e7['기준_alpha']}.")]
        for k, v in e7.items():
            if isinstance(v, list): H += [f"<p>{k}</p>", tbl(v)]
    if os.path.exists(f"{d}/action_results.json"):
        A = json.load(open(f"{d}/action_results.json", encoding="utf-8"))
        def png(p): return f'<img src="data:image/png;base64,{base64.b64encode(open(p, "rb").read()).decode()}">' if os.path.exists(p) else ""
        H += ["<h2>8. 원인별 대처 (PINN 지도 역설계)</h2>", tbl([{"레버": k, "FLR 변화": v} for k, v in A["레버"].items()]),
              tbl([{"항목": k, "값": v} for k, v in A["보상비"].items()]), png(f"{d}/대처_1_PINN지도_역설계.png"),
              "<h3>A 불량 샷 역설계 (레버 하나씩)</h3>", tbl(A["역설계"]), png(f"{d}/대처_2_레버비교.png"),
              "<h3>원인별 PINN FLR</h3>", tbl(A["원인별FLR"]), "<h3>사출속도 창</h3>", tbl([{"항목": k, "값": v} for k, v in A["속도창"].items()]),
              png(f"{d}/대처_3_사출속도창.png"), "<h3>재가동 처리 규칙 (분리 = 폐기가 아니라 검사 대기)</h3>", tbl([{RESTART_LABEL.get(k, k).replace("고정 폐기", "고정 분리") if k == "규칙" else RESTART_LABEL.get(k, k): (v.replace("고정 폐기", "고정 분리") if isinstance(v, str) else v) for k, v in r.items()} for r in A["재가동규칙"]]), png(f"{d}/대처_4_재가동규칙.png")]
        if "재가동_지문" in A:
            F = A["재가동_지문"]
            H += ["<h3>CN7 라벨 기간 재가동 첫 샷</h3>", tbl(A["라벨_재가동"]),
                  "<h3>무라벨 재가동의 식은 수지 지문 (라벨과 독립된 메커니즘 검증)</h3>",
                  box("지문 = 같은 작업오더 안에서 300초 넘게 멈춘 뒤 첫 샷 속도가 오더 중앙값보다 5% 이상 낮고 압력은 1% 이상 높으며, 다음 샷에서 절반 이상 회복."),
                  tbl(F["정지별"]), tbl([{"검정": "정지 길이(log) vs 지문 Spearman", "값": F["정지길이_spearman"]},
                                        {"검정": "재가동 직후 5샷 안 재정지 vs 지문 Fisher (오즈비, p)", "값": F["재정지_fisher"]}]),
                  png(f"{d}/대처_5_재가동지문.png")]
    if os.path.exists(f"{d}/request_results.json"):
        Q = json.load(open(f"{d}/request_results.json", encoding="utf-8"))
        def png2(p): return f'<img src="data:image/png;base64,{base64.b64encode(open(p, "rb").read()).decode()}">' if os.path.exists(p) else ""
        pre = Q["사전고정"]
        H += ["<h2>9. 데이터 요청 엔진 v2</h2>",
              box("결정 질문과 판정 기준은 계산 전에 고정. 4단 판정: ① 식별(관측이 모르는 값을 정하나) ② 엉킴 |ρ| < " + str(pre["기준"]["엉킴상한"]) +
                  " ③ 결정 가능(사전 기준) ④ 비용·기간. 가정치 격자 중 " + f"{pre['기준']['강건비율']:.0%}" + " 이상에서 통과하면 강건. " + Q.get("수정사유", "")),
              tbl([{"원인": k, **v} for k, v in pre["결정질문"].items()]),
              f"<p>PINN 정상 FLR0 = {Q['PINN_FLR0']:.4f} (원인 A 관측 함수에 사용)</p>",
              "<h3>원인 C 후보 (격자 통과 비율)</h3>", tbl(Q["C_요약"]), tbl(Q["C_대표값"]),
              f"<p>최악 조건(저항비 1.05, 오차 최대)에서 결정에 필요한 샷 수: {Q['C_필요샷수_최악조건']}</p>",
              "<h3>원인 A 후보</h3>", tbl(Q["A_요약"]), tbl(Q["A_대표값"]), "<p>A 필요 사양: Δδ 측정 정밀도별 통과 비율 (노즐 센서와 조합, 손 수식 대조)</p>", tbl(Q["A_필요사양"]),
              "<h3>원인 A — PINN 관측 함수 (주 모델)</h3>", box(Q.get("A_모델비교", "")), tbl(Q["A_PINN_요약"]), tbl(Q["A_PINN_대표값"]),
              "<p>센서 오차별 결정 가능 비율</p>", tbl(Q["A_PINN_센서오차별"]),
              "<h3>원인 B 실험 설계</h3>", tbl(Q["B"]["설계"]), f"<p class='small'>{Q['B']['현재']} · {Q['B']['비고']}</p>",
              "<h3>CN7 캐비티 통계형 요청</h3>", tbl([Q["C_CN7통계"]]), png2(f"{d}/요청_1_4단판정.png"),
              "<h3>요청 카드 (순위)</h3>", tbl(Q["요청카드"])]
    if os.path.exists(f"{d}/verify_results.json"):
        W = json.load(open(f"{d}/verify_results.json", encoding="utf-8"))
        H += ["<h2>10. 원인별·경보 기준별 결과 (E12 예측), 변수 빼기·민감도 (고정 기준, 참고), 다른 설비 적용성 (E11)</h2>",
              "<h3>원인별 사건 안 AUC (사건 블록 제외 엄격 예측)</h3>", tbl(W["E10a_원인별_사건블록제외"]["CN7"]), tbl(W["E10a_원인별_사건블록제외"]["RG3"]),
              "<h3>경보 기준별 운영점 (엄격 시간 순서, 보정 준비 후)</h3>", tbl(W["E10c_경보기준별_운영점"]["CN7 경보 기준별"]),
              "<h3>변수 빼기·음성 대조 (고정 기준, 참고)</h3>", tbl(W["E10d_변수빼기_고정기준"]["CN7"]), tbl(W["E10d_변수빼기_고정기준"]["RG3"]),
              "<h3>민감도 (고정 기준, 참고)</h3>", tbl(W["E10e_민감도_고정기준"]["문·경보·기준교체"]), tbl(W["E10e_민감도_고정기준"]["축소강도"]), tbl(W["E10e_민감도_고정기준"]["구간분할"]),
              "<h3>다른 설비·품목 적용성 (무라벨)</h3>", box(W["E11_적용성"]["비고"]), tbl(W["E11_적용성"]["설비별"]), tbl(W["E11_적용성"]["라벨제품_품목"]),
              tbl([{"항목": "품목 수", "값": W["E11_적용성"]["품목수"]}, {"항목": "정상 운전 경보율 중앙", "값": W["E11_적용성"]["전체_정상경보율_중앙"]},
                   {"항목": "재가동 직후 경보율 중앙", "값": W["E11_적용성"]["전체_재가동경보율_중앙"]}, {"항목": "오더 경보율 vs ERR_FACT_QTY (ρ, p, n)", "값": W["E11_적용성"]["오더경보율_vs_ERR_FACT_QTY"]}])]
    sd = f"{d}/strict"
    if os.path.exists(f"{sd}/operations.json"):
        ops = json.load(open(f"{sd}/operations.json", encoding="utf-8")); cmp_ = json.load(open(f"{sd}/comparisons.json", encoding="utf-8")); unc = json.load(open(f"{sd}/uncertainty.json", encoding="utf-8"))
        H += ["<h2>11. 엄격 검증 (E12: 전처리까지 학습 구간 안에서, 사건 블록 전체 제외)</h2>",
              box("시간 순서(forward)는 각 구간 시작 전 자료만으로 레시피 지문·무라벨 선별·문·척도·보정층을 다시 계산. 사건 블록 제외는 시험 사건 구간 전체와 앞뒤 5분을 학습에서 제거. 실행 조건: docs/protocol_frozen.md"),
              "<h3>운영표 (보류·평가불가 = 전수 대체검사 가정)</h3>", tbl(ops), "<h3>같은 조건 모델 비교</h3>", tbl(cmp_), "<h3>사건 블록 bootstrap</h3>", tbl(unc)]
        if os.path.exists(f"{sd}/restart_comparison.json"): H += ["<h3>재가동 정책 (같은 재가동 대상)</h3>", tbl(json.load(open(f"{sd}/restart_comparison.json", encoding="utf-8")))]
    H += ["</body></html>"]
    open(f"{d}/PINNSET_검증리포트.html", "w", encoding="utf-8").write("\n".join(H))

if __name__ == "__main__":
    import sys; build(sys.argv[1] if len(sys.argv) > 1 else ".")
