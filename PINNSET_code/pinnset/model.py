"""위험점수 모델 — 설계도 문서(위험점수_모델_설계보고서.pdf)의 단계 번호와 대응한다.

  S3 reference                       라벨 양품 샷의 기준값(중앙값)과 흔들림 폭
  S4 select_tier1                    무라벨 1층 선별 (레시피 지문 거리)
  S5 recipe_gate                     적용 범위 문 (샷 단위 설정 거리)
  S6 deviation / Baseline            이탈 점수 D (과거 샷 기준, lot 재설정, 20샷 연속 경보 재설정)
  S7 fit_calibration                 보정층 (제품별 설정: CN7 공정 / RG3 캐비티 축소 추정)
  RiskModel                          샷이 들어오는 순서대로, 그 시점까지의 정보만으로 판정·확률·위험점수·원인 태그
(S1·S2 데이터·샷 표는 data.py, S8·S9 검증·유의성은 analysis.py)
"""
import warnings
import numpy as np, pandas as pd
from .config import CFG, VARS4 as V
from .data import shot_table, pairs_for
warnings.filterwarnings("ignore")

IS, IF, II, IR = 0, 1, 2, 3   # V = [사출속도, 충진시간, 사출시간, FLR]


# ============================================================ S3 기준값
def reference(S, cols):
    n = S[S.bad == 0]
    med = n[cols].median()
    sc = np.maximum(1.4826 * (n[cols] - med).abs().median(), np.maximum(0.01 * med.abs(), 0.5))
    return med, sc

def dist(df, med, sc):
    return np.sqrt((((df[med.index] - med) / sc) ** 2).mean(axis=1))

# ============================================================ S4 무라벨 1층
def select_tier1(U, prod, med, sc):
    q = U[(U.EQUIP_NAME == CFG["EQUIP"]) & U.PART_NAME.str.startswith(prod)]
    key = ["PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "PART_NAME"]
    lots = q.groupby(key)[CFG["TIER_VARS"]].median()
    lots["rows"] = q.groupby(key).size()
    lots = lots[lots.rows >= CFG["TIER_MIN_ROWS"]]
    lots["d"] = dist(lots, med, sc)
    ok = lots.index[lots.d <= CFG["TIER_DIST"]]
    return q.set_index(key).loc[ok].reset_index(), lots

# ============================================================ S5 적용 범위 문
def recipe_gate(S, med, sc):
    return (dist(S, med, sc) > CFG["GATE_DIST"]).values

# ============================================================ S6 이탈 점수
def dev_scale(s):
    v = np.unique(s.round(4)); q = np.diff(v).min() if len(v) > 1 else 1e-3
    return max(1.4826 * np.median(np.abs(s - s.median())), q, 1e-3)       # 측정 눈금보다 작아지지 않게


class Baseline:
    """작업오더 안의 이탈 기준. 일괄 계산(deviation)과 샷별 스트림(RiskModel)이 같은 규칙을 쓴다."""
    def __init__(self, V):
        self.V = V; self.lot = None; self.ref = [[] for _ in V]; self.run = []

    def step(self, x, lot, held, scale):
        """x: 이 샷의 변수값, scale: 변수별 척도. 반환: None(문 보류) / 정수(기준 준비 중, 쌓인 샷 수) / (D, z)"""
        if lot != self.lot:                                  # 작업오더가 바뀌면 기준 재설정
            self.ref = [[] for _ in self.V]; self.run = []
        self.lot = lot
        if held:
            return None
        if len(self.ref[0]) < CFG["REF_MIN"]:                # 기준 준비 중 → D 없음(보류)
            for j in range(len(self.V)): self.ref[j].append(x[j])
            return len(self.ref[0])
        z = np.array([(x[j] - np.median(self.ref[j][-CFG["REF_WIN"]:])) / scale[c] for j, c in enumerate(self.V)])
        D = float(np.abs(z).max())
        if D < CFG["ALARM"]:                                 # 경보가 아니면 기준에 추가
            self.run = []
            for j in range(len(self.V)): self.ref[j].append(x[j])
        else:
            self.run.append(x)
            if len(self.run) >= CFG["REBASE_RUN"]:           # 긴 연속 경보 = 설정 변경으로 보고 재설정
                self.ref = [list(col) for col in zip(*self.run)]; self.run = []
        return D, z


def deviation(S, scale, held, lots=None):
    """샷 표 전체의 이탈 점수 D (보류·기준 준비 중이면 NaN). scale은 변수별 척도 dict 또는 샷마다의 dict 목록,
    lots를 주면 S.lot 대신 그 값이 바뀔 때 기준을 재설정한다."""
    V = CFG["VARS"]; X = S[V].values; lot = S.lot.values if lots is None else lots
    base = Baseline(V); D = np.full(len(S), np.nan)
    for i in range(len(S)):
        r = base.step(X[i], lot[i], held[i], scale if isinstance(scale, dict) else scale[i])
        if isinstance(r, tuple): D[i] = r[0]
    return D

# ============================================================ S7 보정층
def irls(X, y, off, it=60):
    b = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(X @ b + off))); W = p * (1 - p) + 1e-9
        b = b + np.linalg.solve(X.T @ (W[:, None] * X) + 1e-6 * np.eye(len(b)), X.T @ (y - p))
    return b

def shrink_log_ratio(pair):
    """짝지은 비교(같은 샷 LH/RH)에서 로그 위험비를 추정하고 사건 뭉침만큼 분산을 키워 0 쪽으로 축소"""
    k = int(((pair.RH == 1) & (pair.LH == 0)).sum()); m = int(((pair.LH == 1) & (pair.RH == 0)).sum())
    one = pair[(pair.RH + pair.LH) == 1]
    nev = int((one.index.to_series().diff() > pd.Timedelta(minutes=CFG["EVENT_GAP_MIN"])).sum() + 1) if len(one) else 1
    L = np.log((k + 0.5) / (m + 0.5)); v = (1 / (k + 0.5) + 1 / (m + 0.5)) * max(1, (k + m) / nev)
    return L * CFG["SHRINK_TAU"] ** 2 / (CFG["SHRINK_TAU"] ** 2 + v)

def design(P, spec):
    X = np.ones((len(P), 1))
    if spec["process"]:
        X = np.c_[X, P.gD.values]
    return X

def fit_calibration(P, pair, spec):
    off = np.zeros(len(P)); Lc = 0.0
    if spec["cavity"] == "shrink":
        Lc = shrink_log_ratio(pair); off = Lc * P.RH.values
    b = irls(design(P, spec), P.y.values, off)
    if spec["process"] and b[1] < 0:                       # 물리 제약: 이탈이 위험을 줄일 수 없음
        spec = dict(spec, process=False); b = irls(design(P, spec), P.y.values, off)
    return b, Lc, spec

def predict(P, b, Lc, spec):
    return 1 / (1 + np.exp(-(design(P, spec) @ b + Lc * P.RH.values)))

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

# ============================================================ 샷별 예측 모델
class RiskModel:
    def __init__(self, prod):
        self.prod = prod; self.spec = dict(CFG["MODEL"][prod]); self.b = None

    # ---- 학습: 기준값(문·척도)은 고정, 보정층은 주어진 학습 부품으로
    def set_references(self, S_lab, T1_shots):
        """문 기준값과 이탈 척도는 모두 1층 무라벨 운전 기록으로 잡는다 (라벨 불필요, 여러 lot을 포함해 폭이 현실적)"""
        G = T1_shots[CFG["GATE_VARS"]]; self.gmed = G.median()
        self.gsc = np.maximum(1.4826 * (G - self.gmed).abs().median(), np.maximum(0.01 * self.gmed.abs(), 0.5))
        self.scale = {c: dev_scale(T1_shots[c]) for c in V}

    def fit_calibration(self, P_train, pairs_train):
        self.b, self.Lc, self.used = fit_calibration(P_train, pairs_train, dict(self.spec))

    def prob(self, D, RH):
        x = [1.0] + ([np.log1p(D)] if self.used["process"] else [])
        return float(1 / (1 + np.exp(-(np.dot(self.b, x) + self.Lc * RH))))

    # ---- 예측 함수: 샷을 순서대로 받아 부품별 결과를 낸다
    def reset_stream(self):
        self.base = Baseline(V); self.window = []

    def _advance(self, shot):
        """이탈 기준 상태를 한 샷 진행. 반환: (보류 사유 또는 None, D, z)"""
        x = np.array([shot[c] for c in V], dtype=float)
        gd = float(np.sqrt(np.mean(((pd.Series({c: shot[c] for c in CFG["GATE_VARS"]}) - self.gmed) / self.gsc) ** 2)))
        r = self.base.step(x, shot["lot"], gd > CFG["GATE_DIST"], self.scale)
        if r is None: return "다른 레시피 (설정 거리 %.1f)" % gd, None, None
        if not isinstance(r, tuple): return "기준 준비 중 (작업오더 %d번째 샷)" % r, None, None
        return None, r[0], r[1]

    def step(self, shot, parts):
        """shot: 샷 하나의 공정값(dict/Series), parts: 그 샷의 부품들 [(PART_NAME, RH), ...]"""
        reason, D, z = self._advance(shot)                         # 보정층이 없는 시기에도 이탈 기준은 똑같이 진행
        if self.b is None:                                         # 아직 학습할 불량이 없던 시기 (시간 순서 검증)
            return [dict(PART_NAME=n, 판정="평가불가", 보류사유="-", 불량확률=np.nan, 위험점수=np.nan,
                         원인태그="-", 이탈점수=np.nan, 주요변수="-") for n, _ in parts]
        if reason:
            return [dict(PART_NAME=n, 판정="보류", 보류사유=reason, 불량확률=np.nan, 위험점수=np.nan,
                         원인태그="-", 이탈점수=np.nan, 주요변수="-") for n, _ in parts]
        top = V[int(np.abs(z).argmax())]; out = []
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
        """S: 샷 표(시간순), parts_df: 부품 표. calib_by_seg가 있으면 조각마다 보정층을 바꿔 끼운다(시간 순서 검증용, 없으면 None)"""
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
            res = self.step(sh, list(zip(g.PART_NAME, g.RH)))
            for r in res: r.update(TimeStamp=sh.TimeStamp, seg=sh.seg)
            rows += res
        return pd.DataFrame(rows)

# ============================================================ 제품별 준비 (전체 기간 고정 기준)
def prepare_product(prod, L, U):
    """라벨 부품·샷 표·1층 무라벨과, 1층 무라벨로 기준을 잡은 모델의 문·이탈 점수. 보정층은 아직 맞추지 않는다."""
    parts = L[L.PART_NAME.str.startswith(prod)].copy()
    S = shot_table(parts)
    tmed, tsc = reference(S, CFG["TIER_VARS"]); T1, _ = select_tier1(U, prod, tmed, tsc)
    ST = shot_table(T1)
    m = RiskModel(prod); m.set_references(S, ST)
    rec = recipe_gate(S, m.gmed, m.gsc); D = deviation(S, m.scale, rec)
    held = rec | np.isnan(D)
    P = parts.merge(S.loc[~held, ["TimeStamp", "seg"]].assign(D=D[~held]), on="TimeStamp"); P["gD"] = np.log1p(P.D)
    return dict(parts=parts, S=S, ST=ST, T1=T1, pairs=pairs_for(parts), m=m, rec=rec, D=D, held=held, P=P)
