"""데이터 — 원본 CSV 읽기, 파생량(Tm, Tw, FLR), 샷 표, LH/RH 쌍, 원인 분류

무라벨 CSV(약 225MB)는 프로세스마다 한 번만 읽고 재사용한다. 돌려받은 표는 고치지 말고 복사해서 쓴다.
"""
import os
import numpy as np, pandas as pd
from scipy.special import erfinv
from .config import CFG, PROC, B_WINDOW

BZ = ["Barrel_Temperature_1", "Barrel_Temperature_2", "Barrel_Temperature_3"]
UNLABELED_COLS = ["TimeStamp", "PART_NAME", "EQUIP_NAME", "PART_FACT_PLAN_DATE", "PART_FACT_SERIAL", "ERR_FACT_QTY",
                  "Cycle_Time", "Hopper_Temperature", "Plasticizing_Position", "Average_Back_Pressure",
                  "Max_Injection_Speed", "Filling_Time", "Injection_Time", "Mold_Temperature_3", "Mold_Temperature_4"] + BZ
_CACHE = {}


def add_derived(df):
    df = df.copy()
    df["Tm"] = df[BZ].mean(axis=1)                                            # 노즐측 용융온도
    df["Tw"] = df[["Mold_Temperature_3", "Mold_Temperature_4"]].mean(axis=1)  # 금형온도
    r = np.clip((CFG["T_NF"] - df.Tw) / (df.Tm - df.Tw), 1e-9, 1 - 1e-9)
    df["FLR"] = 2 * np.sqrt(CFG["ALPHA"] * df.Filling_Time.clip(lower=1e-3)) * erfinv(r) / CFG["HALF_THICK"]
    return df


def read_labeled_raw(path):
    """라벨 원본 (중복 제거 전)"""
    return pd.read_csv(f"{path}/labeled_data.csv", parse_dates=["TimeStamp"])


def load_labeled(path):
    d = read_labeled_raw(path)
    d = d.drop_duplicates(["TimeStamp", "PART_FACT_SERIAL", "PART_NAME"])      # 10-27 이후 이중 삽입 제거
    d["y"] = (d.PassOrFail == "N").astype(int)
    d["RH"] = d.PART_NAME.str.endswith("RH").astype(int)
    return add_derived(d)


def read_unlabeled(path, cols):
    """무라벨 원본에서 열 cols만 (원본 열 순서 유지). 파일은 한 번만 읽는다"""
    key = ("raw", os.path.realpath(path))
    if key not in _CACHE:
        _CACHE[key] = pd.read_csv(f"{path}/unlabeled_data.csv", parse_dates=["TimeStamp"])
    U = _CACHE[key]; cols = set(cols)
    return U[[c for c in U.columns if c in cols]]


def load_unlabeled(path):
    key = ("derived", os.path.realpath(path))
    if key not in _CACHE:
        _CACHE[key] = add_derived(read_unlabeled(path, UNLABELED_COLS))
    return _CACHE[key]


def shot_table(parts):
    """부품 행 → 샷 단위 표 (작업오더 키 lot, 연속 가동 구간 seg)"""
    g = parts.groupby("TimeStamp")
    keep = list(dict.fromkeys(CFG["VARS"] + CFG["GATE_VARS"] + CFG["TIER_VARS"]))
    S = g[keep].first()
    S["lot"] = g.PART_FACT_PLAN_DATE.first().astype(str) + "_" + g.PART_FACT_SERIAL.min().astype(str)
    if "y" in parts:
        S["bad"] = g.y.max()
    S = S.sort_index().reset_index()
    S["seg"] = (S.TimeStamp.diff().dt.total_seconds() > CFG["SEG_GAP_S"]).cumsum()
    return S


def pairs_for(parts):
    """같은 샷의 LH/RH 불량 여부 (둘 다 있는 샷만)"""
    return parts.pivot_table(index="TimeStamp", columns=parts.PART_NAME.str[-2:], values="y", aggfunc="max").dropna()


def cause_class(prod, r):
    if r.y == 0: return "양품"
    if prod == "CN7":
        if r.Reason == "초기허용불량": return "A 재가동"
        if pd.Timestamp(B_WINDOW[0]) <= r.TimeStamp <= pd.Timestamp(B_WINDOW[1]): return "B 고속"
        if r.Reason == "가스": return "C 캐비티"
        return "설명 불가"
    return "C 캐비티"


def features(P):
    """비교 모델(로지스틱·HistGB·Isolation Forest) 입력: 공정 변수 전체 + Tm, Tw, FLR, RH"""
    X = P[PROC].copy()
    X["Tm"], X["Tw"], X["FLR"], X["RH"] = P.Tm, P.Tw, P.FLR, P.RH
    return X
