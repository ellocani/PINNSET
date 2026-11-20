"""PINNSET 분석 패키지. 실행 순서와 파일 역할은 PINNSET_code/README.md의 '코드 구성'을 본다."""
import sys as _sys
for _s in (_sys.stdout, _sys.stderr):   # Windows(CP949) 콘솔에서도 기호 출력 오류가 나지 않게
    try: _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception: pass
