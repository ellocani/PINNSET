# PINNSET 재현 코드

제6회 K-인공지능 제조데이터 분석 경진대회 · 주제 ① 사출성형 품질불량 사전예측 및 검사 우선순위 결정

## 기본 결과 재현 (저장된 PINN 가중치 사용)
```
pip install -r requirements.txt
python run_all.py --data <데이터 폴더> --out <결과 폴더>
```
- 데이터 폴더: KAMP 사출성형기 AI 데이터셋 원본의 `labeled_data.csv`, `unlabeled_data.csv`
- 새 레시피·새 수지 적용성 검증(선택, 약 60분, `--fast` 약 30분): `python run_newrecipe.py --data <데이터 폴더> --out <결과 폴더>` → `<결과 폴더>/newrecipe/newrecipe_results.json`
- 현장 화면 시연(선택, 약 30초): `python run_dashboard.py --data <데이터 폴더> --results <결과 폴더>` → `<결과 폴더>/PINNSET_대시보드.html` (파일 하나, 인터넷 불필요)
- 결과 숫자는 requirements.txt 주석의 검증 버전(scikit-learn 1.8.0 등) 기준이다. 비교용 트리 부스팅(HistGB)은 scikit-learn 버전에 따라 소수점 아래 값이 달라질 수 있다(PINNSET 모델·운영표·예측값은 검증한 환경에서 수치 재현을 확인했다).
- 리포트 그림의 한글 글꼴: Windows는 맑은 고딕, macOS는 Apple SD 고딕을 자동으로 쓴다. Linux는 `/usr/share/fonts/`, `/usr/local/share/fonts/`, `~/.local/share/fonts/`, `~/.fonts/`에서 나눔고딕 또는 Noto Sans CJK를 찾는다. 없으면 `sudo apt install fonts-nanum`(또는 `fonts-noto-cjk`)로 설치한다.
- GPU 불필요. 일반 노트북 약 8~12분 (검증 환경 실행 시간은 run_info.json). 난수 시드를 고정했고, 검증한 환경(requirements.txt 주석의 버전)에서 같은 수치가 재현됨을 확인했다.

## 목적별 재현 절차

### 실행 위치와 데이터 준비

아래 명령은 `run_all.py`와 `requirements.txt`가 있는 **`PINNSET_code/` 폴더에서 실행**한다. 재현에 필요한 원본 CSV는 KAMP 사출성형기 AI 데이터셋에서 별도로 확보해 아래 예시처럼 배치한다. `--data`에는 `labeled_data.csv`와 `unlabeled_data.csv`가 직접 들어 있는 폴더를 지정한다. Dataset_Molding ZIP의 `dataset/` 폴더도 사용할 수 있다.

예시 배치:

```text
PINNSET/
    PINNSET_code/
        README.md
        requirements.txt
        run_all.py
        pinn_torch/
        weights/
    data/
        labeled_data.csv
        unlabeled_data.csv
```

이 배치에서는 기본 데이터 경로가 `../data`다. 실제 압축 해제 위치가 다르면 해당 경로로 바꾼다. 공백이 있는 경로는 큰따옴표로 감싼다. PINN 그림 생성에 쓰는 `--data`는 아래 설명대로 **절대 경로**를 사용한다.

### Windows에서 실행할 때 (필수)

Windows 기본 문자 인코딩(CP949)에서는 일부 기호(≥, −, → 등)를 저장·출력할 때 오류가 날 수 있다. 코드의 텍스트 입출력에는 `encoding="utf-8"`을 명시했지만, 아래처럼 UTF-8 모드를 켜고 실행한다. 아래 명령은 결과를 `results_final_check/`에 저장한다.

PowerShell (`PINNSET_code` 폴더에서):

```powershell
$env:PYTHONUTF8="1"
python -m pip install -r requirements.txt
python run_all.py --data "../data" --out "results_final_check"
python run_dashboard.py --data "../data" --results "results_final_check"
```

명령 프롬프트(cmd)를 쓰면 첫 줄을 `set PYTHONUTF8=1`로 바꾼다. 원본 CSV(`labeled_data.csv`, `unlabeled_data.csv`)를 읽는 방식은 바꾸지 않았다.

### 1. 제출된 기본 결과 재현

```bash
python -m pip install -r requirements.txt
python run_all.py --data "../data" --out "results_reproduced"
```

`run_all.py`는 전처리, 이탈 점수 계산, 제품별 보정층 학습, 예측, 추가 분석, 대처·요청 분석, 엄격 검증, 리포트 생성을 순서대로 실행한다. 위험점수 모델과 비교 모델은 제공 데이터로 다시 학습한다. 기본 위험점수 모델의 FLR은 해석식으로 계산하며, 대처·요청 분석의 PINN 계산에는 **코드 폴더의 `weights/`에 저장된 가중치**를 사용한다. 이 명령에서 PINN 신경망을 처음부터 재학습하지 않으므로 PyTorch는 필요하지 않다.

주요 산출물은 `results_reproduced/`에 생성된다.

| 산출물 | 내용 |
|---|---|
| `predictions_CN7.csv`, `predictions_RG3.csv` | 전체 라벨로 학습한 최종 모델(현장 투입용)의 부품별 출력. **검증 성능의 근거로 쓰지 않는다** |
| `predictions_forward_CN7.csv`, `predictions_forward_RG3.csv` | 고정 기준 시간 순서 예측, 참고·진단용 |
| `strict/predictions_forward_*.csv` | 전처리까지 학습 구간 안에서 계산한 엄격 시간 순서 검증 예측 (**공식 성능 근거**) |
| `strict/predictions_leave_block_out_*.csv` | 사건 블록 제외 검증 예측 (**공식 성능 근거**) |
| `strict/model_comparison.csv`, `strict/operating_comparison.csv` | 공식 모델 비교·운영 평가 |
| `action_results.json`, `request_results.json` | 대처·데이터 요청 분석 |
| `PINNSET_검증리포트.html` | 자동 생성 검증 리포트 |
| `PINNSET_데이터요청서.html` | 자동 생성 요청 카드·실행 계획 |
| `run_info.json`, `strict/run_receipt.json` | 실행 환경·시간, 입력 데이터·코드·가중치 해시 |

공식 성능은 `strict/`의 엄격 검증 결과를 기준으로 확인한다. 별도 공식 테스트셋은 제공되지 않아 자체 검증 분할(시간 순서, 사건 블록 제외)을 썼으며, 이를 독립된 공식 테스트셋이라고 부르지 않는다. 제출 당시의 `results/`를 별도로 보관하고 있다면 실행 시간·환경 경로 등 실행별 메타데이터는 구분하고, 같은 평가의 분할·대상 수·검출 수·성능 수치·예측값을 비교한다.

### 2. 확장 실험과 대시보드까지 재현

기본 실행이 완료된 후, 같은 데이터와 결과 폴더로 실행한다.

```bash
python run_newrecipe.py --data "../data" --out "results_reproduced"
python run_dashboard.py --data "../data" --results "results_reproduced"
```

- 확장 실험: `results_reproduced/newrecipe/newrecipe_results.json`, `onboarding_frozen.csv`, `onboarding_daily.csv` 및 변형별 예측·요약 파일 생성.
- 대시보드: `results_reproduced/PINNSET_대시보드.html`, `dashboard_checks.json` 생성. HTML을 브라우저에서 열어 사용한다.
- 확장 실험은 기본 결과와 별도 경로로 저장한다. 대시보드는 기본 실행이 만든 엄격 예측 CSV와 대처·요청 결과를 읽고, 다시 계산한 예측·운영 수치가 일치하는지 확인한 뒤 생성한다.
- 확장 실험과 대시보드는 서로 직접 의존하지 않는다. 대시보드만 필요하면 확장 실험을 생략할 수 있다.

빠른 확장 실험은 다음 명령으로 실행한다.

```bash
python run_newrecipe.py --data "../data" --out "results_reproduced_fast" --fast
```

`--fast`는 엄격 평가의 변형을 `official`, `pinn_auto` 두 가지로 줄인다. 네 가지 변형(`official`, `auto`, `pinn`, `pinn_auto`)의 전체 비교가 필요하면 `--fast`를 사용하지 않는다. 기본 분석, 확장 실험, 대시보드 생성은 모두 `run_all.py` 하나에 포함되어 있지 않으므로 목적에 맞게 위 명령을 순서대로 실행한다.

### 3. PINN을 처음부터 재학습

이 단계는 PINN 가중치 생성 과정까지 확인하려는 경우에 실행한다. 저장된 가중치로 제출 결과를 재현하는 목적이라면 1~2절의 절차를 사용한다.

```bash
python -m pip install -r pinn_torch/requirements_torch.txt
python pinn_torch/pinn_torch_train.py --exp 1 2 3 3b 4 --out "weights_torch" --data "<데이터 폴더의 절대 경로>" --figures
```

`<데이터 폴더의 절대 경로>`는 실제 경로로 교체한다. 예를 들어 Windows에서는 `"C:/PINNSET/data"`, Linux·macOS에서는 `"/home/user/PINNSET/data"`처럼 지정한다. 그림 스크립트는 출력 폴더를 작업 위치로 실행하므로, `--figures`와 함께 쓰는 데이터 경로는 절대 경로로 지정한다.

**실험 선택 주의:** 현재 코드에서 `--exp all`은 **실험 1·2·3·3b만** 선택한다. 실험 4를 포함한 모든 실험은 위와 같이 `--exp 1 2 3 3b 4`를 명시한다. `all`과 `4`를 함께 나열하는 대신 실험 번호 전체를 명시한다.

| 실험 | 가중치 파일 | 역할 |
|---|---|---|
| 1 | `pinn_exp1_const.npz` | 물성 일정 열전도·정답 대조 |
| 2 | `pinn_exp2_map.npz` | 온도 의존 물성·성형 윈도우 지도 |
| 3 | `pinn_exp3_bi.npz` | 접촉 열저항 Bi 역산 |
| 3b | `pinn_exp3b_bi.npz` | 보강 학습에 의한 Bi 역산 |
| 4 | `pinn_exp4_resin.npz` | 열확산율 비를 입력하는 수지 일반화 |

학습 출력 폴더에는 가중치, `train.log`, `train_log.json`, `verify_report.json`, `verify_report.md`와 그림용 `P1.pkl`, `P2.pkl`, `R3.pkl`, `R3b.pkl`이 생성된다. `--figures`를 사용하면 PINN 그림 3장도 생성된다. 제출 가중치를 만든 원 학습의 기록은 `weights/학습기록/`에 있다(실험 4는 `weights/학습기록/exp4/`). 새로 학습한 기록은 지정한 `--out` 폴더에서 확인한다.

실험 1~3b는 CPU로 실행 가능하며, 실험 4도 기본은 CPU다(제출 가중치는 코랩 T4 GPU로 학습했고, CPU에서는 시간이 더 걸린다). 실험 4를 GPU에서 별도로 실행하려면 CUDA를 사용할 수 있는 PyTorch 환경에서 다음 명령을 사용한다.

```bash
python pinn_torch/pinn_torch_train.py --exp 4 --device cuda --out "weights_torch"
```

별도 실행의 `train_log.json`에는 그 실행에서 선택한 실험의 기록이 저장된다. 모든 실험의 학습 기록을 한 파일에 남기려면 전체 실험을 한 명령에서 선택한다. CPU·GPU·스레드 수 등에 따라 학습 시간과 부동소수점 계산이 달라질 수 있으므로, 가중치의 비트 단위 일치 대신 검증 보고서의 물리량 오차와 통과 여부를 확인한다. 세부 학습 설정·검증 기준은 `pinn_torch/README.md`를 참고하되, 실험 4 포함 명령과 경로 안내는 이 절차를 따른다.

### 4. 재학습 가중치 검증과 본 실행 연결

`--out weights_torch`로 학습한 가중치는 **본 분석에 자동 적용되지 않는다.** `run_all.py`의 대처·요청 분석과 확장 실험은 코드 폴더의 `weights/`를 읽는다. `run_all.py --out`은 분석 결과 저장 위치를 지정하는 인수이며 PINN 가중치 경로를 바꾸지 않는다.

먼저 새 가중치를 검증한다.

```bash
python pinn_torch/pinn_torch_train.py --verify-only --out "weights_torch" --ref "weights"
```

`weights_torch/verify_report.json`과 `verify_report.md`에서 최종 통과 여부와 기준점·FDM·제출 가중치 대비 오차를 확인한다. 모든 실험을 학습했다면 위 표의 가중치 5개가 모두 생성되었는지도 확인한다. `--verify-only`는 PyTorch 없이 실행할 수 있다. 검증할 실험의 가중치 파일이 없거나 검증 항목이 하나도 실행되지 않으면 실패로 처리하며, 검증에 실패하면 종료 코드 1을 반환한다(`--quick`은 반복 1/20의 동작 확인이므로 수렴 검증 실패는 안내만 하고 종료 코드 0, 파일 누락은 1). 실험 3·3b의 역산 결과는 검증 보고서의 참고 항목으로 따로 확인한다.

검증을 통과한 새 가중치로 분석하려면 다음 순서로 적용한다.

1. 현재 `weights/` 폴더 전체를 `weights_original_backup/`처럼 별도 이름으로 백업한다. 이미 같은 이름의 백업이 있으면 새 이름을 사용한다.
2. `weights_torch/`의 가중치 5개(`pinn_exp1_const.npz`, `pinn_exp2_map.npz`, `pinn_exp3_bi.npz`, `pinn_exp3b_bi.npz`, `pinn_exp4_resin.npz`)를 `weights/`에 복사해 같은 이름의 가중치 파일을 교체한다. 가중치 로더는 `pinnset/pinn.py`다.
3. 학습·검증 기록은 `weights_torch/`에 보관하고, 기존 제출 결과와 다른 결과 폴더를 지정해 다시 실행한다.

```bash
python run_all.py --data "../data" --out "results_retrained"
python run_newrecipe.py --data "../data" --out "results_retrained"
python run_dashboard.py --data "../data" --results "results_retrained"
```

필요한 경우 확장 실험이나 대시보드는 생략할 수 있다. 기존 제출 가중치를 사용하는 재현 결과(`results_reproduced/`)와 새 가중치를 사용하는 결과(`results_retrained/`)를 구분한다. 재학습 후 물리 지도·처방·요청·PINN을 사용하는 확장 실험의 수치는 허용오차 안에서도 달라질 수 있다. 기본 위험점수 모델은 해석식 FLR을 사용한다.

`weights/pinn_checks.json`은 제출 가중치의 기존 검산 기록이다. 가중치 5개를 교체하는 위 절차는 이 기록을 갱신하지 않으며, 엄격 검증의 `reused_PINN_evidence`에도 기존 기록이 포함된다. **재학습 가중치의 물리 검증 근거는 새로 생성한 `weights_torch/verify_report.json/.md`로 확인한다.** 기본 재현은 저장 가중치 검증이고, 최초 재학습의 최종 수렴·일치는 해당 정규 학습과 검증을 완료한 후 판단한다.

제출 가중치로 되돌리려면 백업의 가중치 5개를 `weights/`에 다시 복사하고, 복원된 가중치로 필요한 분석을 실행한다.

동작 점검용 축약 학습은 별도 출력 폴더에서 실행한다.

```bash
python pinn_torch/pinn_torch_train.py --exp 1 2 3 3b 4 --quick --out "weights_smoke"
```

`--quick`은 반복 수를 줄이는 동작 확인용이다. 물리 검증 실패가 예상되며, 이 결과를 제출용 가중치로 적용하거나 정규 학습 재현 완료의 근거로 사용하지 않는다.


## 코드 구성 (실행 흐름 순서)

실행 스크립트 3개는 `PINNSET_code/`에, 단계별 코드는 `pinnset/`에 있다. 대괄호 번호는 `run_all.py`의 실행 단계다.

| 실행 | 하는 일 |
|---|---|
| `run_all.py` | 기본 분석 [1]~[8] 전체 |
| `run_dashboard.py` | (별도) 현장 화면 시연 — `pinnset/dashboard.py` |
| `run_newrecipe.py` | (별도) 새 레시피·새 수지 적용성 — `pinnset/newrecipe.py` |

| 파일 (`pinnset/`) | 단계 | 내용 |
|---|---|---|
| `config.py` | 공통 | 동결 파라미터 `CFG`, 물성 상수(반두께·열확산율·유리전이·무유동 온도), 라벨 정상 운전점, 검사 예산, 공정 변수 목록 |
| `data.py` | 공통 (설계도 S1·S2) | 원본 CSV 읽기(무라벨 225MB는 프로세스마다 한 번만), 중복 제거, 노즐측 용융온도·금형온도·FLR 계산, 샷 표, 연속 가동 구간, LH/RH 쌍, 원인 분류 |
| `model.py` | 공통 (S3~S7) | 기준값, 무라벨 1층 선별, 적용 범위 문, 이탈 점수 D(`Baseline`: 일괄 계산·샷별 스트림이 같은 규칙), 제품별 보정층(CN7 공정 이탈 / RG3 캐비티 축소 추정), 원인 태그, `RiskModel.predict_stream`(샷 순서대로 불량확률·위험점수·원인태그·보류사유) |
| `stats.py` | 공통 | 구간 내 순위(centered mid-rank), F1·혼동행렬, 원형 이동 귀무분포, JSON 정리 |
| `analysis.py` | [1][2][3][7] | 고정 기준(참고) 평가: v0 재현·구간 빼기 검증·유의성(S8·S9), v1 일괄=샷별·미래 누설·시간 순서·원인 태그와 부품별 예측 CSV, 추가 분석 E1~E9(진단, 일반 ML 비교·F1, 상호작용, FN/FP, 민감도, 보류 규칙, C 단서 재현, 보류 포함 확률, 위치 변수), 강건성 E10·적용성 E11 |
| `pinn.py` | [4][5][6] | PINN 가중치(`weights/*.npz`) 불러오기·FLR·지도 계산 (PyTorch 불필요). `python -m pinnset.pinn`으로 기준점 확인 |
| `action.py` | [4] | PINN 성형 윈도우 지도 위 역설계, 레버 효율, 사출속도 창, 재가동 규칙, 무라벨 재가동 지문 |
| `request.py` | [5] | 데이터 요청 엔진 v2: 사전 고정 결정 질문, 채널별 측정오차, 4단 판정(식별·엉킴·결정·비용), 가정치 격자, A 관측 함수 PINN 교체, 요청 카드·데이터 요청서 자동 생성 |
| `evaluate.py` + `docs/protocol_frozen.md` | [6] **공식 성능** | 엄격 검증(E12): 전처리까지 학습 구간 안에서 계산하는 시간 순서 평가, 사건 블록 제외, 운영표, 모델 비교, bootstrap, 물리·요청 검산 |
| `report.py` | [8] | `PINNSET_검증리포트.html` (0~9장). 데이터 요청서는 `request.py`가 따로 생성 |
| `dashboard.py` | 별도 | run_all 결과 폴더를 읽어 `PINNSET_대시보드.html` 생성. ① 재생 모드: 엄격 시간 순서 예측을 샷 순서대로 재생(판정·확률·위험점수·원인·근거 z·대처·검사 대기열·데이터 요청), 공식 예측 CSV·운영표와 대조해 다르면 만들지 않음. ② 실시간 모드: 정상 기준에 쓰지 않은 무라벨 실제 운전 기록을 샷마다 입력하면 화면 안의 판정 엔진(RiskModel.step과 같은 계산)이 즉시 판정, 같은 입력의 Python 결과와 자체 대조(2,400샷 불일치 0). 모의 사건 주입은 실제 사건의 변화 비율 사용. ③ CSV 불러오기: 사용자가 넣은 KAMP 형식 CSV를 같은 판정 엔진으로 처리 — 과거 파일 한 번에 처리(전체 무라벨 225MB·79만 행 약 7초), 또는 계속 늘어나는 파일을 1~5초마다 다시 확인해 새 행만 판정(Chrome·Edge). 파일은 브라우저 밖으로 보내지 않음 |
| `newrecipe.py` | 별도 (E13) | ① 라벨 없는 레시피 자동 등록: 무라벨 작업오더의 공정 설정 지문을 서로 비교해 레시피를 스스로 묶고(거리 3.0, 300행 이상) 레시피마다 적용 범위 기준·이탈 척도 계산, 샷은 가장 가까운 레시피로 판정 ② PINN FLR 직접 사용: 해석식 대신 학습된 PINN으로 샷별 FLR(이분법) ③ 같은 엄격 평가 틀(`evaluate.evaluate_product`)로 공식·자동·PINN·자동+PINN 비교 ④ 무라벨 전체 설비·품목에 매일 다시 등록하며 판정(정상·재가동 경보율, 판정률) ⑤ 다른 두께·수지에 무차원 변환만으로 PINN 사용 → 실단위 FDM 대조 |

| 그 밖의 폴더 | 내용 |
|---|---|
| `weights/` | PINN 가중치(npz)와 학습·검증 기록 |
| `pinn_torch/` | PINN 실험 1·2·3·3b·4를 PyTorch로 학습해 가중치와 학습·검증 기록을 생성. 검증 후 가중치를 `weights/`에 반영하는 절차는 위 3~4절 (`pinn_torch/README.md`) |
| `tools/csv_feed_sim.py` | 대시보드 'CSV 불러오기 → 실시간 감시' 시연용으로 계속 늘어나는 CSV를 만든다 |
| `tools/compare_results.py` | 두 결과 폴더의 산출물이 같은지 확인 (`python tools/compare_results.py <기준 결과> <새 결과>`) |

## 동결 파라미터 (2026-09-29, 결과 보고용으로 고정)
GATE_DIST 3.0 · ALARM 5.0 · REBASE_RUN 20 · SHRINK_TAU 0.7 · SEG_GAP_S 300초 · TAG_Z 2.0 · PCT_WINDOW 1,000
모두 `pinnset/config.py`의 `CFG`에서만 바꾼다.

## 정정 이력
- 2026-11-20: 코드 구조 정리 — 단계별 파일을 `pinnset/` 패키지로 옮기고(설정 `config.py`, 데이터 `data.py`, 모델 `model.py`, 고정 기준 분석 `analysis.py`, 엄격 검증 `evaluate.py` 등) 실행 스크립트를 `run_all.py`·`run_dashboard.py`·`run_newrecipe.py` 셋으로 정리. 이탈 점수 계산 3벌(일괄·샷별 스트림·다중 레시피)을 `model.Baseline` 하나로, 구간 내 순위·원형 이동·JSON 정리 등 중복 함수를 `stats.py`로 합침. newrecipe의 모듈 바꿔치기를 인자 전달로, CFG 임시 변경을 `config.override`로 바꿈. 쓰이지 않던 변수·상수 제거. 원본 코드와 같은 데이터로 `run_all`·대시보드·`run_newrecipe --fast`를 실행해 `tools/compare_results.py`로 대조: 모든 예측 CSV·수치 JSON 같음, 다른 곳은 결과 안내문 속 파일 이름(예: pinnset_strict.py → evaluate.py)과 실행 시간뿐. 계산 로직·예측·성능 수치 변경 없음.
- 2026-10-08 (최종 2): 검증 리포트의 구간 AUC 설명을 실제 계산(구간별 순위 백분위 → 전체 ROC-AUC)에 맞춤, 무라벨 경보율과 라벨 양품 오경보율 구분, 2·3장에 '고정 기준 참고 평가' 표시, 재가동 표 표시 이름을 '검사 대기 분리'로 변경. PINN 검증 스크립트가 가중치 누락·검증 실패를 종료 코드 1로 알리도록 수정. JSON의 NaN을 null로 저장. 그림 2 제목의 열확산율 표현 정정. 계산 로직·예측·성능 수치 변경 없음.
- 2026-10-08 (최종): 텍스트 입출력 encoding="utf-8" 명시, 콘솔 출력 UTF-8 설정, README에 Windows 실행 안내와 예측 파일 용도 구분 추가, 대시보드 대처 문구를 "폐기"에서 "검사 대기 분리"로 정정. 계산 로직·예측·성능 수치 변경 없음(전체 재실행으로 확인).
- 2026-10-08: README 재현 절차 보강 — 실행 위치·데이터 배치, 목적별 재현 절차(기본 결과 / 확장 실험·대시보드 / PINN 재학습 / 재학습 가중치 검증과 적용) 추가. 코드·결과 변경 없음.
- 2026-10-07 (3차): `pinnset_newrecipe.py`(E13) 추가 — 라벨 없이 레시피를 자동 등록해도 CN7 검출 시간 순서 10/11·사건 블록 제외 17/28, 구간 AUC 0.792, 보류 부품 수가 공식과 같고 경보는 20 → 18부품(잡은 불량 10개 그대로, 첫 구간 제외 검사 비율 4.6% → 4.5%), 사건 블록 제외 확률 오차 개선 35.4% → 37.0%. RG3 전 항목 같음. 보류된(어느 레시피와도 먼) 샷은 레시피를 바꾸지 않게 해 다른 레시피 샷이 끼어들어도 기준이 다시 쌓이지 않음. PINN FLR을 위험점수 모델에 직접 넣어도 경보·보류·검출 전부 같음(이탈 점수 최대 차이 0.015). 무라벨 12개 설비·51개 품목(매일 다시 등록): 정상 운전 경보율 샷 가중 0.18%, 재가동 직후 3샷 7.3%, 판정률 69%. 다른 두께·수지 5가지에서 무차원 변환 PINN vs FDM 최대 0.0012(해석식 0.0025). pinn_torch에 실험 4(수지 일반화: 열확산율 비 r 입력, 검증 ⑥) 추가 — 코랩 T4 학습 결과 weights/pinn_exp4_resin.npz, 45조건 FDM 대비 중앙 0.00047·최대 0.00369로 통과(해석식 중앙 0.00072·최대 0.00706), fdm.py에 열확산율 비 선택 인수, weights/pinn_map.py에 수지·두께 인수(Tg, Tnf, half_mm, a_melt, ratio) 추가 — 기존 호출 결과는 같음(기준점 0.1763/0.1722/0.1322/0.1012 그대로).
- 2026-10-07 (2차): CSV 불러오기 보강 — ① 감시 중지·재시작이 확인 도중에 겹치면 확인 예약이 둘로 늘던 문제 수정(감시 세대 확인), ② 새 파일은 끝까지 성공해야 화면을 바꿈(실패 시 이전 결과 유지)·불러오는 중 중복 실행 막기, ③ 숫자 열은 '4.5초'·'1,200'·빈 칸을 오류로 셈(앞 숫자만 읽지 않음), 없는 날짜·시각 거절, ④ 이미 읽은 파일 끝 4KB가 바뀌면 감지해 알림(이미 판정한 샷은 그대로, 그 뒤 시각 행만 반영), ⑤ 3분 넘게 새 샷이 없으면 '입력 지연' 표시, ⑥ 판정 결과 CSV 저장(엑셀 수식 주입 방지), ⑦ 파일 이름·Reason 문자열을 화면에 넣기 전 이스케이프. Python 대조·감시 시험 결과 변화 없음.
- 2026-10-07: 대시보드에 'CSV 불러오기' 모드 추가, 시연용 `tools/csv_feed_sim.py`(늘어나는 CSV 흉내) 추가. Python 원본(최종 모델 + RiskModel.predict_stream)과 대조: 전체 무라벨 CSV 58,377샷·라벨 CSV 2,624샷에서 판정·보류 사유·원인 태그 불일치 0, 이탈 점수 차이 최대 7e-14, 확률 차이 최대 7e-16. 위험점수(백분위)는 확률이 사실상 같은 부품끼리의 순서가 부동소수 마지막 자리로 갈리는 경우에만 다름(라벨 CN7에서 해당). 실시간 감시는 무작위 위치에서 끊어 1,007번 덧붙인 파일·통째 다시 쓴 파일 모두 한 번에 처리한 결과와 완전히 같음. 대시보드 표의 배지 글자색 오류(회색으로 덮임) 수정. 모델·기존 결과 숫자 변경 없음.
- 2026-10-06 (2차): PINN 가중치를 PyTorch 학습본으로 교체(검증 통과: 기준점, 푸리에 급수, FDM 중앙 0.0009·최대 0.0021). 지도·역설계 값이 소수점 셋째 자리에서 바뀜(예: 라벨 운전점 FLR 0.1759→0.1763, 첫 샷 필요 금형온도 52.1→52.3℃). 위험점수 모델·검출 성능은 PINN을 쓰지 않으므로 변화 없음.
- 2026-10-06: make_dashboard.py 추가 (현장 화면: 재생 모드 + 실시간 모드). 모델·기존 결과 숫자 변경 없음.
- 2026-09-30: 구간 AUC를 centered mid-rank로 정정 (모든 점수가 같으면 정확히 0.5). RG3 0.698 → 0.674, CN7 변화 0.001 이하.
- 2026-10-04 (3차): pinn_torch/ 추가 — PINN 가중치를 PyTorch로 처음부터 다시 만드는 경로와 PyTorch 없이 돌아가는 가중치 검증(FDM 15조건 중앙 0.0008·최대 0.0015 재계산).
- 2026-10-04 (2차): 원본 평가 경로에서 누설·겹침 제거. verify의 사건 빼기·운영점을 E12 예측 기반으로 교체, 구간 부트스트랩 삭제(E12 블록 bootstrap으로 대체), v1·extra의 고정 기준 평가에 '참고용' 표시. 실행 순서: strict → verify.
- 2026-10-04: 독립 재검증 핵심 항목(K01~K08) 반영. E12 엄격 검증 추가(공식 성능은 E12 기준), 재가동 규칙에서 자료 첫 구간 제외·보수적 정책 추가, C 압력 후보를 '유량 직접 관측(이상적 가정)'으로 정정하고 정량 카드에서 제외, 사건 빼기를 '원인별 후향 분석'으로 정정, RG3 위험비 → 오즈비 표기.
- 2026-10-02: E10·E11 추가. 보고서의 강건성 수치(이전에는 별도 검증 코드)를 제출 코드만으로 재현. run_all 7단계.
- 2026-09-30: 요청 엔진 단계 2·3: A 관측 함수를 PINN 지도로 교체(손 수식은 대조), 데이터 요청서 자동 생성.
- 2026-09-30: 요청 엔진 v2 추가. ② 엉킴 기준은 결과를 본 뒤 A에만 적용하도록 수정(사전 판정 병기).
- 2026-09-30: 위치 변수 분석(E9) 추가: 절환·형개 위치 미기록 확인, C 불량 샷 쿠션 정상(총량 정상 → 분배 문제).
- 2026-09-30: 보류 부품 포함 확률 평가(E8) 추가. 모델 계산은 동일, 평가 방식만 추가.
- 2026-09-30: C 단서(호퍼·배럴6) 재현 검정 추가 → RG3에서 기각. 무라벨 재가동 지문 분석 추가.
- v1 예측 CSV에 이탈 성분 z (z_speed, z_fill, z_inj, z_flr) 열 추가, 시간 순서 예측 CSV 저장 추가. 모델 계산은 동일.
