"""
PINNSET 전체 실행 — 전처리부터 학습·추론·검증·리포트 생성까지 명령 하나

    python run_all.py --data <KAMP 사출성형 데이터 폴더> --out <결과 폴더>

데이터 폴더에는 labeled_data.csv, unlabeled_data.csv 가 있어야 한다.
결과 폴더에 만들어지는 것:
    risk_v0_results.json          모델 재현·구간 빼기 검증·유의성 (v0, 동결 모델)
    risk_v1_validation.json       원인 태그·샷별 예측·시간 순서·누설 검사 (v1)
    predictions_CN7/RG3.csv       부품별 불량확률·위험점수·원인태그·보류사유 (전체 학습)
    predictions_forward_*.csv     시간 순서 검증 예측 (앞선 구간만 학습)
    extra_results.json            데이터 진단·모델 비교(F1)·상호작용·FN/FP·민감도·보류 규칙
    action_results.json, 대처_*.png  원인별 대처: PINN 지도 역설계, 레버 효율, 사출속도 창, 재가동 규칙
    request_results.json, 요청_*.png 데이터 요청 엔진 v2: 결정 질문·4단 판정·요청 카드 (A는 PINN 관측 함수)
    PINNSET_데이터요청서.html     요청 카드 14장 + 실행 계획 (자동 생성)
    verify_results.json           원인별·경보 기준별(E12 예측), 변수 빼기·민감도(고정 기준 참고), 다른 설비 적용성(E11)
    strict/                       엄격 검증(E12): 시간 순서·사건 블록 제외 예측, 운영표, 모델 비교, bootstrap, 물리·요청 검산
    PINNSET_검증리포트.html        위 결과를 표·그림으로 모은 리포트
소요: 일반 노트북 약 8~12분 (실행 시간은 run_info.json). 난수 시드 고정, 검증 환경에서 수치 재현 확인.
단계별 코드는 pinnset/ 폴더 (README.md '코드 구성').
"""
import argparse, os, sys, time, json, platform

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--out", default="pinnset_output")
    a = ap.parse_args()
    data = os.path.abspath(a.data); out = os.path.abspath(a.out)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(out, exist_ok=True); os.chdir(out)
    import numpy, pandas, scipy, sklearn
    from pinnset import analysis as an, action as ac, request as rq, evaluate as st, report as mr
    env = dict(python=platform.python_version(), numpy=numpy.__version__, pandas=pandas.__version__,
               scipy=scipy.__version__, sklearn=sklearn.__version__, data=data)
    T = {}; t0 = time.time()
    print("[1/8] v0 모델 재현·검증·유의성", flush=True); an.run_v0(data); T["v0"] = time.time() - t0
    print("[2/8] v1 원인 태그·샷별 예측·시간 순서 검증", flush=True); an.run_v1(data); T["v1"] = time.time() - t0
    print("[3/8] 추가 분석 (진단·비교·상호작용·FN/FP·민감도·보류)", flush=True); an.run_extra(data, "."); T["extra"] = time.time() - t0
    print("[4/8] 원인별 대처 (PINN 지도 역설계·속도 창·재가동 규칙)", flush=True); ac.run(data, "."); T["action"] = time.time() - t0
    print("[5/8] 데이터 요청 엔진 v2 (식별성 4단 판정·요청 카드)", flush=True); rq.run("."); T["request"] = time.time() - t0
    print("[6/8] 엄격 검증 (E12: 전처리 학습 내부화, 사건 블록 제외, 운영표) — 공식 성능", flush=True); st.run(data, "."); T["strict"] = time.time() - t0
    print("[7/8] 원인별·경보 기준별 (E12 예측 사용), 변수 빼기·민감도(고정 기준 참고), 다른 설비 적용성", flush=True); an.run_verify(data, "."); T["verify"] = time.time() - t0
    print("[8/8] 리포트 생성", flush=True)
    json.dump(dict(env=env, elapsed_s=T), open("run_info.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    mr.build("."); print(f"완료: {out}/PINNSET_검증리포트.html  (총 {time.time() - t0:.0f}초)")

if __name__ == "__main__":
    main()
