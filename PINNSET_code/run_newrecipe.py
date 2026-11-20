"""PINNSET 새 레시피·새 수지 적용성 검증 (E13, 별도 실행)

    python run_newrecipe.py --data <KAMP 데이터 폴더> --out <결과 폴더>           # 전체 (약 60분)
    python run_newrecipe.py --data <KAMP 데이터 폴더> --out <결과 폴더> --fast    # 엄격 평가 2가지만 (약 30분)

결과: <결과 폴더>/newrecipe/. 자세한 내용은 pinnset/newrecipe.py 맨 위 설명.
"""
import os, sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pinnset import newrecipe
    newrecipe.main()
