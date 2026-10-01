"""ReAct vs 보조-뇌 비교 실행기. 결과는 한 줄에 한 실행씩 jsonl 로 쌓인다(끊겨도 이어 돈다).

    python3 -m bench.run --model haiku --n 12 --workers 4
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import react, sbagent
from .llm import ClaudeP
from .tasks import suite

METHODS = {"react": react.run, "subbrain": sbagent.run}
lock = threading.Lock()


def overhead(model: str, thinking=None) -> dict:
    """빈 호출 하나의 입력 토큰 -- 호출마다 CLI 가 붙이는 고정분을 잰다."""
    llm = ClaudeP(model, thinking=thinking)
    llm("Reply with the single word OK.", "x")
    return llm.calls[-1]


def one(task, method, model, rep, max_calls, out, thinking=None):
    llm = ClaudeP(model, thinking=thinking)
    t0 = time.time()
    try:
        r = METHODS[method](task, llm, max_calls=max_calls)
    except Exception as e:
        r = {"answer": None, "crash": f"{type(e).__name__}: {e}"}
    rec = {"task": task.id, "family": task.family, "method": method, "model": model, "rep": rep,
           "thinking": thinking,
           "expected": task.answer, "answer": r.get("answer"), "correct": task.check(r.get("answer")),
           "said_correct": task.check(r.get("said") or r.get("answer")),
           "meta": task.meta, "tok": llm.totals(), "wall": round(time.time() - t0, 1), "detail": r,
           "transcript": [{k: c.get(k) for k in ("in", "out", "thinking", "user", "reply", "error")} for c in llm.calls]}
    with lock:
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"{task.id:5} {method:9} rep{rep} {'O' if rec['correct'] else 'X'} calls={rec['tok']['calls']} "
          f"in={rec['tok']['in']} out={rec['tok']['out']} ans={str(r.get('answer'))[:40]!r}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-calls", type=int, default=10)
    ap.add_argument("--methods", default="react,subbrain")
    ap.add_argument("--out", default=None)
    ap.add_argument("--thinking", type=int, default=None, help="0 이면 생각을 끈다")
    a = ap.parse_args()
    tag = a.model + ("" if a.thinking is None else f"-think{a.thinking}")
    out = Path(a.out or f"bench/results/{tag}.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    head = out.with_suffix(".overhead.json")
    if not head.exists():
        head.write_text(json.dumps(overhead(a.model, a.thinking)))
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            done.add((d["task"], d["method"], d["rep"]))
    jobs = [(t, m, r) for r in range(a.reps) for t in suite(a.n, a.start) for m in a.methods.split(",")
            if (t.id, m, r) not in done]
    print(f"{len(jobs)} runs ({len(done)} already done) -> {out}", flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        for t, m, r in jobs:
            ex.submit(one, t, m, a.model, r, a.max_calls, out, a.thinking)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
