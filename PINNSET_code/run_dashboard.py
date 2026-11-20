"""PINNSET 현장 대시보드 생성 (run_all.py 실행 후)

    python run_dashboard.py --data <KAMP 데이터 폴더> --results <결과 폴더>

결과: <결과 폴더>/PINNSET_대시보드.html, dashboard_checks.json. 자세한 내용은 pinnset/dashboard.py 맨 위 설명.
"""
import argparse, os, sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pinnset import dashboard
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--results", default="results"); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dashboard.build(a.data, a.results, a.out)
