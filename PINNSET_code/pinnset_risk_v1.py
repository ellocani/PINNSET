"""
PINNSET 위험점수 모델 v1
================================================
v0(pinnset_risk_v0.py)에 세 가지를 더했다.
  - 원인 태그 (tag_rule)            : 경보 샷의 이탈 방향으로 A형/B형 구분, RG3 RH는 캐비티 주의
  - 예측 함수 (RiskModel.predict_stream) : 샷이 들어오는 순서대로, 그 시점까지의 정보만으로 답을 낸다
  - 시간 순서 검증 (evaluate_forward)    : 앞선 조각만으로 학습해 다음 조각을 예측

실행:  python pinnset_risk_v1.py --data /경로/dataset
"""
import argparse, json, warnings
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # Windows(CP949) 콘솔에서도 기호 출력 오류가 나지 않게
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
warnings.filterwarnings("ignore")
import pinnset_risk_v0 as v0
from pinnset_risk_v0 import CFG

CFG.update(dict(
    TAG_Z=2.0,            # 원인 태그: 한 변수가 평소의 2배 넘게 벗어나면 그 방향으로 본다
    PCT_WINDOW=1000,      # 위험점수: 최근 몇 개 부품 중 몇 등인지
))
V = CFG["VARS"]  # [사출속도, 충진시간, 사출시간, FLR]
IS, IF, II, IR = 0, 1, 2, 3

# ============================================================ 원인 태그
def tag_rule(z, D, RH, prod):
    """z: 부호 있는 이탈(변수별), D: 이탈 점수, RH: 0/1"""
    if D is not None and not np.isnan(D) and D >= CFG["ALARM"]:
        t = CFG["TAG_Z"]
        slow = z[IS] <= -t or z[IF] >= t or z[IR] >= t      # 느린 충전 → 고화층 두꺼워짐
        fast = z[IS] >= t or z[IF] <= -t or z[IR] <= -t     # 빠른 충전 → 공기 갇힘
        if slow and not fast: return "A형 느린 충전·고화"
        if fast and not slow: return "B형 고속·공기 갇힘"
        return "공정 이탈 (유형 불명)"
    if prod == "RG3" and RH == 1:
        return "캐비티 주의 (RH)"
    return "-"

# ============================================================ 예측 모델
class RiskModel:
    def __init__(self, prod):
        self.prod = prod; self.spec = dict(CFG["MODEL"][prod])

    # ---- 학습: 기준값(문·척도)은 고정, 보정층은 주어진 학습 부품으로
    def set_references(self, S_lab, T1_shots):
        """문 기준값과 이탈 척도는 모두 1층 무라벨 운전 기록으로 잡는다 (라벨 불필요, 여러 lot을 포함해 폭이 현실적)"""
        G = T1_shots[CFG["GATE_VARS"]]; self.gmed = G.median()
        self.gsc = np.maximum(1.4826 * (G - self.gmed).abs().median(), np.maximum(0.01 * self.gmed.abs(), 0.5))
        self.scale = {c: v0.dev_scale(T1_shots[c]) for c in V}

    def fit_calibration(self, P_train, pairs_train):
        self.b, self.Lc, self.used = v0.fit_calibration(P_train, pairs_train, dict(self.spec))

    def prob(self, D, RH):
        x = [1.0] + ([np.log1p(D)] if self.used["process"] else [])
        return float(1 / (1 + np.exp(-(np.dot(self.b, x) + self.Lc * RH))))

    # ---- 예측 함수: 샷을 순서대로 받아 부품별 결과를 낸다
    def reset_stream(self):
        self.ref = {c: [] for c in V}; self.run = []; self.prev_lot = None; self.window = []

    def step(self, shot, parts):
        """shot: 샷 하나의 공정값(dict/Series), parts: 그 샷의 부품들 [(PART_NAME, RH), ...]"""
        out = []; x = np.array([shot[c] for c in V], dtype=float)
        if shot["lot"] != self.prev_lot:                          # 작업오더가 바뀌면 이탈 기준 재설정
            self.ref = {c: [] for c in V}; self.run = []
        self.prev_lot = shot["lot"]
        gd = float(np.sqrt(np.mean(((pd.Series({c: shot[c] for c in CFG["GATE_VARS"]}) - self.gmed) / self.gsc) ** 2)))
        if gd > CFG["GATE_DIST"]:
            reason = "다른 레시피 (설정 거리 %.1f)" % gd
        elif len(self.ref[V[0]]) < CFG["REF_MIN"]:
            for j, c in enumerate(V): self.ref[c].append(x[j])
            reason = "기준 준비 중 (작업오더 %d번째 샷)" % len(self.ref[V[0]])
        else:
            reason = None
        if reason:
            return [dict(PART_NAME=n, 판정="보류", 보류사유=reason, 불량확률=np.nan, 위험점수=np.nan,
                         원인태그="-", 이탈점수=np.nan, 주요변수="-") for n, _ in parts]
        z = np.array([(x[j] - np.median(self.ref[c][-CFG["REF_WIN"]:])) / self.scale[c] for j, c in enumerate(V)])
        D = float(np.abs(z).max()); top = V[int(np.abs(z).argmax())]
        if D < CFG["ALARM"]:
            self.run = []
            for j, c in enumerate(V): self.ref[c].append(x[j])
        else:
            self.run.append(x)
            if len(self.run) >= CFG["REBASE_RUN"]:
                arr = np.array(self.run); self.ref = {c: list(arr[:, j]) for j, c in enumerate(V)}; self.run = []
        for n, rh in parts:
            p = self.prob(D, rh)
            w = self.window[-(CFG["PCT_WINDOW"] - 1):] + [p]           # 과거 부품 + 현재 부품만
            arr = np.array(w); pct = 100 * (np.sum(arr < p) + 0.5 * np.sum(arr == p)) / len(arr)
            self.window.append(p)
            out.append(dict(PART_NAME=n, 판정="점수", 보류사유="-", 불량확률=p, 위험점수=pct,
                            원인태그=tag_rule(z, D, rh, self.prod), 이탈점수=D, 주요변수=f"{top} {z[V.index(top)]:+.1f}",
                            z_speed=z[IS], z_fill=z[IF], z_inj=z[II], z_flr=z[IR]))
        return out

    def predict_stream(self, S, parts_df, calib_by_seg=None):
        """S: 샷 표(시간순), parts_df: 부품 표. calib_by_seg가 있으면 조각마다 보정층을 바꿔 끼운다(시간 순서 검증용)"""
        self.reset_stream(); rows = []
        byts = {t: g for t, g in parts_df.groupby("TimeStamp")}
        for _, sh in S.iterrows():
            if calib_by_seg is not None:
                c = calib_by_seg.get(sh.seg)
                if c is None:
                    self.b = None
                else:
                    self.b, self.Lc, self.used = c
            g = byts[sh.TimeStamp]
            parts = list(zip(g.PART_NAME, g.RH))
            if calib_by_seg is not None and self.b is None:        # 아직 학습할 불량이 없던 시기
                self._advance_only(sh); res = [dict(PART_NAME=n, 판정="평가불가", 보류사유="-", 불량확률=np.nan, 위험점수=np.nan,
                                                  원인태그="-", 이탈점수=np.nan, 주요변수="-") for n, _ in parts]
            else:
                res = self.step(sh, parts)
            for r in res: r.update(TimeStamp=sh.TimeStamp, seg=sh.seg)
            rows += res
        return pd.DataFrame(rows)

    def _advance_only(self, sh):
        """보정층이 없는 시기에도 이탈 기준 상태는 똑같이 진행시킨다"""
        b, Lc, used = getattr(self, "b", None), getattr(self, "Lc", 0.0), getattr(self, "used", self.spec)
        self.b, self.Lc, self.used = np.zeros(1 + int(self.spec["process"])), 0.0, dict(self.spec)
        self.step(sh, [("_", 0)]); self.window.pop() if self.window else None
        self.b, self.Lc, self.used = None, Lc, used

# ============================================================ 준비
def prepare(path):
    L, U = v0.load_labeled(path), v0.load_unlabeled(path); data = {}
    for prod in CFG["PRODUCTS"]:
        parts = L[L.PART_NAME.str.startswith(prod)].copy()
        S = v0.shot_table(parts)
        tmed, tsc = v0.reference(S, CFG["TIER_VARS"]); T1, _ = v0.select_tier1(U, prod, tmed, tsc)
        ST = v0.shot_table(T1)
        pairs = parts.pivot_table(index="TimeStamp", columns=parts.PART_NAME.str[-2:], values="y", aggfunc="max").dropna()
        data[prod] = dict(parts=parts, S=S, ST=ST, T1=T1, pairs=pairs)
    return data

def cause_class(prod, r):
    if r.y == 0: return "양품"
    if prod == "CN7":
        if r.Reason == "초기허용불량": return "A 재가동"
        if pd.Timestamp("2020-10-16 05:55") <= r.TimeStamp <= pd.Timestamp("2020-10-16 05:58"): return "B 고속"
        if r.Reason == "가스": return "C 캐비티"
        return "설명 불가"
    return "C 캐비티"

# ============================================================ 검증
def validate(path):
    data = prepare(path); report = {}
    for prod, d in data.items():
        parts, S, ST, pairs = d["parts"], d["S"], d["ST"], d["pairs"]
        m = RiskModel(prod); m.set_references(S, ST)
        rec = v0.recipe_gate(S, m.gmed, m.gsc); Dv0 = v0.deviation(S, m.scale, rec)
        held = rec | np.isnan(Dv0)
        P = parts.merge(S.loc[~held, ["TimeStamp", "seg"]].assign(D=Dv0[~held]), on="TimeStamp"); P["gD"] = np.log1p(P.D)
        R = {}
        # 1) 일괄 계산 = 샷별 계산
        m.fit_calibration(P, pairs); out = m.predict_stream(S, parts)
        s1 = out.drop_duplicates("TimeStamp").set_index("TimeStamp").이탈점수
        a = pd.Series(Dv0, index=S.TimeStamp).reindex(s1.index)
        R["1_일괄=샷별"] = dict(최대차이=float(np.nanmax(np.abs(a.values - s1.values))) if s1.notna().any() else 0.0,
                            보류일치=bool(((a.isna()) == (s1.isna())).all()))
        # 2) 미래 누설 검사: 절반 이후 샷의 공정값을 뒤섞어도 앞 절반의 예측이 그대로인가
        cut = S.TimeStamp.iloc[len(S) // 2]
        S2 = S.copy(); fut = S2.TimeStamp > cut
        rng = np.random.default_rng(1)
        for c in V + CFG["GATE_VARS"]: S2.loc[fut, c] = rng.permutation(S2.loc[fut, c].values)
        out2 = m.predict_stream(S2, parts)
        k = out.TimeStamp <= cut
        R["2_미래누설"] = dict(앞절반_부품=int(k.sum()),
                          확률_최대차이=float(np.nanmax(np.abs(out.불량확률[k].values - out2.불량확률[k].values))),
                          위험점수_최대차이=float(np.nanmax(np.abs(out.위험점수[k].values - out2.위험점수[k].values))))
        # 3) 시간 순서 (고정 기준, 참고): 보정층만 과거 구간으로 학습. 문·척도는 전체 기간 무라벨 기준이라 엄격한 forward가 아님
        #    → 공식 시간 순서 평가는 pinnset_strict.py가 전처리까지 학습 구간 안에서 다시 계산한다 (2026-10-04)
        calib = {}
        for s in sorted(S.seg.unique()):
            tr = P.seg < s
            if P.y[tr].sum() == 0: calib[s] = None; continue
            b, Lc, used = v0.fit_calibration(P[tr], pairs.loc[pairs.index.isin(P.TimeStamp[tr])], dict(m.spec)); calib[s] = (b, Lc, used)
        outF = m.predict_stream(S, parts, calib_by_seg=calib)
        outF.to_csv(f"predictions_forward_{prod}.csv", index=False, encoding="utf-8-sig")
        E = outF.merge(parts[["TimeStamp", "PART_NAME", "y", "Reason", "RH"]], on=["TimeStamp", "PART_NAME"])
        ev = E[E.판정 == "점수"]; y = ev.y.values
        base = brier_score_loss(y, np.full(len(y), y.mean()))
        order = np.argsort(-ev.불량확률.values)
        R["3_시간순서"] = dict(평가종류="고정 기준(문·척도를 전체 기간 무라벨로 고정) — 스트림 점검용 참고값. 공식 시간 순서 성능은 strict/ (pinnset_strict.py)", 평가부품=int(len(ev)), 불량=int(y.sum()), 평가불가_불량=int(E.y[E.판정 == "평가불가"].sum()),
                          보류_불량=int(E.y[E.판정 == "보류"].sum()),
                          위험점수_AUC=float(roc_auc_score(y, ev.위험점수)), Brier개선=float(1 - brier_score_loss(y, ev.불량확률) / base),
                          위험점수95이상_검출=f"{int(y[ev.위험점수.values >= 95].sum())}/{int(y.sum())} (부품 {int((ev.위험점수 >= 95).sum())}개)",
                          구간내_AUC=float(roc_auc_score(y, ev.groupby("seg").위험점수.transform(lambda q: (q.rank() - 0.5) / len(q)))),
                          태그부품_검사시_검출=f"{int(y[(ev.원인태그 != '-').values].sum())}/{int(y.sum())} (부품 {int((ev.원인태그 != '-').sum())}개)")
        # 4) 원인 태그 검증 (전체로 학습한 모델의 샷별 출력 기준)
        T = out.merge(parts[["TimeStamp", "PART_NAME", "y", "Reason", "RH"]], on=["TimeStamp", "PART_NAME"])
        T["원인"] = [cause_class(prod, r) for r in T.itertuples()]
        T["태그"] = np.where(T.판정 == "보류", "보류", T.원인태그)
        R["4_태그표"] = pd.crosstab(T.원인, T.태그).to_dict()
        # 5) 무라벨 1층에서 경보와 태그 분포 (라벨 없이 현장에서 얼마나 자주 울리나)
        TU = d["T1"].copy(); TU["RH"] = TU.PART_NAME.str.endswith("RH").astype(int)
        outU = m.predict_stream(ST, TU)
        sc = outU[outU.판정 == "점수"]
        R["5_무라벨_태그"] = dict(점수부품=int(len(sc)), 태그=sc.원인태그.value_counts().to_dict(),
                             보류=outU[outU.판정 == "보류"].보류사유.str.split(" ").str[0].value_counts().to_dict())
        report[prod] = R
        out.to_csv(f"predictions_{prod}.csv", index=False, encoding="utf-8-sig")
        print(f"\n===================== {prod} =====================")
        print(" 1) 일괄=샷별:", R["1_일괄=샷별"])
        print(" 2) 미래 누설:", R["2_미래누설"])
        print(" 3) 시간 순서:", R["3_시간순서"])
        print(" 4) 원인 × 태그 (부품 수):"); print(pd.crosstab(T.원인, T.태그).to_string())
        print(" 5) 무라벨 1층:", R["5_무라벨_태그"])
    json.dump(report, open("risk_v1_validation.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=str)
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--data", default="dataset"); a = ap.parse_args()
    validate(a.data)
