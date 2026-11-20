"""설비 기록 흉내: 계속 늘어나는 CSV 만들기 (대시보드 'CSV 불러오기 → 실시간 감시' 시연·점검용)

    python tools/csv_feed_sim.py --src <KAMP 데이터 폴더>/unlabeled_data.csv --out live.csv --rate 2

· src에서 학습 설비(650톤-우진2호기)의 CN7·RG3 행을 시간순으로 골라, 같은 시각 행(LH·RH)을 한 샷으로 묶어
  초당 --rate 샷 속도로 out 파일 끝에 덧붙인다. 대시보드에서 out 파일을 '실시간 감시'로 열어 두면 샷이 붙을 때마다 판정이 갱신된다.
· --start로 시작 시각, --shots로 샷 수를 정한다. 모델·결과 파일은 건드리지 않는다.
"""
import argparse
import csv
import time
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # Windows(CP949) 콘솔에서도 기호 출력 오류가 나지 않게
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass

import pandas as pd

EQUIP = "650톤-우진2호기"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="live.csv")
    ap.add_argument("--rate", type=float, default=2.0, help="초당 샷 수")
    ap.add_argument("--start", default=None, help="이 시각 이후 행부터 (예: 2020-09-07 03:10)")
    ap.add_argument("--shots", type=int, default=2000)
    ap.add_argument("--product", default="CN7", choices=["CN7", "RG3", "all"])
    a = ap.parse_args()
    d = pd.read_csv(a.src, dtype=str)
    d = d.loc[:, ~d.columns.str.startswith("Unnamed")]
    keep = d.PART_NAME.str.startswith(("CN7", "RG3") if a.product == "all" else a.product)
    if "EQUIP_NAME" in d:
        keep &= d.EQUIP_NAME.eq(EQUIP)
    d = d[keep].sort_values("TimeStamp", kind="stable")
    if a.start:
        d = d[d.TimeStamp >= a.start]
    groups = [g for _, g in d.groupby("TimeStamp", sort=True)][: a.shots]
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(d.columns)
    print(f"{a.out}: {len(groups)}샷을 초당 {a.rate}샷으로 덧붙임 (Ctrl+C로 중지)")
    for k, g in enumerate(groups, 1):
        with open(a.out, "a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerows(g.values.tolist())
        if k % 50 == 0:
            print(f"  {k}샷 · {g.TimeStamp.iloc[0]}")
        time.sleep(1 / a.rate)


if __name__ == "__main__":
    main()
