"""두 결과 폴더(리팩토링 전·후 등)의 산출물이 같은지 확인한다.

    python tools/compare_results.py <기준 결과 폴더> <새 결과 폴더>

· CSV·JSON은 값을 비교한다(실행 시간·코드 해시처럼 실행마다 달라지는 항목은 제외). 그 밖의 파일은 바이트 단위로 비교한다.
· 다른 파일이 하나라도 있으면 목록을 출력하고 종료 코드 1을 반환한다.
"""
import json, sys
from pathlib import Path
import pandas as pd

SKIP_FILES = {"run_info.json"}                                     # 실행 환경·소요 시간
SKIP_KEYS = {"elapsed_seconds", "original_sha256", "소요초"}        # 실행마다 달라지는 값


def diff_json(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b), key=str):
            if k in SKIP_KEYS: continue
            if k not in a or k not in b: out.append(f"{path}/{k}: 한쪽에만 있음")
            else: out += diff_json(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): return [f"{path}: 길이 {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff_json(x, y, f"{path}[{i}]")]
    return [] if a == b or (a != a and b != b) else [f"{path}: {a!r} != {b!r}"]


def compare(base, new):
    base, new = Path(base), Path(new)
    fa = {p.relative_to(base) for p in base.rglob("*") if p.is_file()}
    fb = {p.relative_to(new) for p in new.rglob("*") if p.is_file()}
    problems = [f"기준에만 있음: {p}" for p in sorted(fa - fb)] + [f"새 결과에만 있음: {p}" for p in sorted(fb - fa)]
    same = 0
    for rel in sorted(fa & fb):
        if rel.name in SKIP_FILES: continue
        pa, pb = base / rel, new / rel
        if pa.read_bytes() == pb.read_bytes(): same += 1; continue
        if rel.suffix == ".json":
            d = diff_json(json.loads(pa.read_text(encoding="utf-8")), json.loads(pb.read_text(encoding="utf-8")))
            if not d: same += 1; continue
            problems.append(f"{rel}: 값 {len(d)}곳 다름, 예) " + " | ".join(d[:3]))
        elif rel.suffix == ".csv":
            A, B = pd.read_csv(pa), pd.read_csv(pb)
            if A.equals(B): same += 1; continue
            problems.append(f"{rel}: 표 다름 (모양 {A.shape} / {B.shape})")
        else:
            problems.append(f"{rel}: 바이트 다름")
    print(f"같음 {same}개 · 다름 {len(problems)}개")
    for p in problems: print("  " + p)
    return not problems


if __name__ == "__main__":
    sys.exit(0 if compare(sys.argv[1], sys.argv[2]) else 1)
