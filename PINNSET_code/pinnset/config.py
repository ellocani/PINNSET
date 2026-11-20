"""PINNSET 설정 — 동결 파라미터와 물성 상수 (전부 여기서만 바꾼다)"""
import contextlib

# ============================================================ 물성 (ASA 문헌 대표값)과 라벨 정상 운전점
HALF_THICK = 4.37 / 2      # 반두께 (mm)
ALPHA = 0.080              # 용융 상태 열확산율 (mm²/s)
T_GLASS = 105.0            # 유리전이 온도 (℃)
T_NF = 110.0               # 무유동 온도 (℃)
T_MELT_REF, T_MOLD_REF, FILL_REF = 275.5, 22.8, 4.45   # 용융온도(℃)·금형온도(℃)·충진시간(초)

# ============================================================ 위험점수 모델 파라미터
CFG = dict(
    EQUIP="650톤-우진2호기",
    PRODUCTS=["CN7", "RG3"],
    # 물성 — FLR 계산용
    ALPHA=ALPHA, T_NF=T_NF, HALF_THICK=HALF_THICK,
    # 변수 묶음
    VARS=["Max_Injection_Speed", "Filling_Time", "Injection_Time", "FLR"],          # 이탈 점수: 원인 A·B 변수
    GATE_VARS=["Tm", "Tw", "Hopper_Temperature", "Plasticizing_Position"],          # 문: 샷마다 안 변하는 설정
    TIER_VARS=["Tm", "Tw", "Cycle_Time", "Plasticizing_Position", "Average_Back_Pressure"],  # 1층: lot 레시피 지문
    # 기준값
    SEG_GAP_S=300,          # 연속 가동 구간을 나누는 정지 시간(초)
    GATE_DIST=3.0,          # 문: 설정 거리 기준
    TIER_DIST=3.0,          # 1층: 지문 거리 기준
    TIER_MIN_ROWS=30,       # 1층: lot 최소 행 수
    REF_MIN=5,              # 이탈 점수 계산에 필요한 최소 기준 샷
    REF_WIN=200,            # 기준 중앙값 창(샷)
    ALARM=5.0,              # 경보 기준 (이탈 점수)
    REBASE_RUN=20,          # 연속 경보가 이만큼이면 기준 재설정
    EVENT_GAP_MIN=30,       # 불량 사건을 나누는 시간(분)
    # 보정층: 제품별 성분 (유의성 검정 결과로 결정)
    MODEL={"CN7": dict(process=True, cavity="none"),
           "RG3": dict(process=False, cavity="shrink")},
    SHRINK_TAU=0.7,         # 캐비티 로그 위험비 사전분포 N(0, τ²)
    N_SHIFT=3000, N_BOOT=5000, SEED=0,
    # 샷별 출력
    TAG_Z=2.0,              # 원인 태그: 한 변수가 평소의 2배 넘게 벗어나면 그 방향으로 본다
    PCT_WINDOW=1000,        # 위험점수: 최근 몇 개 부품 중 몇 등인지
)
VARS4 = list(CFG["VARS"])   # 샷별 스트림(RiskModel)이 쓰는 4변수 [사출속도, 충진시간, 사출시간, FLR]

# ============================================================ 평가·비교 모델 공통
BUDGET = {"CN7": 0.05, "RG3": 0.50}          # 검사 예산 (V07과 같음)
B_WINDOW = ("2020-10-16 05:55", "2020-10-16 05:58")   # CN7 B(고속) 불량 사건 시각
PROC = ["Injection_Time", "Filling_Time", "Plasticizing_Time", "Cycle_Time", "Clamp_Close_Time", "Cushion_Position",
        "Switch_Over_Position", "Plasticizing_Position", "Clamp_Open_Position", "Max_Injection_Speed", "Max_Screw_RPM",
        "Average_Screw_RPM", "Max_Injection_Pressure", "Max_Switch_Over_Pressure", "Max_Back_Pressure",
        "Average_Back_Pressure"] + [f"Barrel_Temperature_{i}" for i in range(1, 8)] + ["Hopper_Temperature"] + \
       [f"Mold_Temperature_{i}" for i in range(1, 13)]


@contextlib.contextmanager
def override(**kw):
    """민감도·변수 빼기 계산용: CFG 값을 잠시 바꾸고 끝나면 반드시 원래 값으로 되돌린다"""
    old = {k: CFG[k] for k in kw}
    CFG.update(kw)
    try:
        yield
    finally:
        CFG.update(old)
