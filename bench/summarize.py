"""결과 집계. 정답률만이 아니라 '사소한 설명' 을 가를 수 있게 같이 낸다.

  - 답을 낸 비율과 낸 답의 정답률을 가른다(답을 아껴서 정답률이 오른 것인지)
  - 토큰은 전체 평균과 '둘 다 맞힌 과제끼리' 짝지은 차이를 같이 낸다(일찍 포기해서 싸진 것인지)
  - claude -p 가 호출마다 붙이는 고정분(빈 호출로 잰 값)을 뺀 입력 토큰도 낸다
  - 짝지은 차이에는 부트스트랩 95% 구간을 붙인다
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path


def boot(xs, n=4000, seed=0):
    if not xs:
        return (float("nan"),) * 3
    r = random.Random(seed)
    ms = sorted(sum(r.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return sum(xs) / len(xs), ms[int(.025 * n)], ms[int(.975 * n)]


def load(path):
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]
    from .tasks import by_id                 # 채점기를 고쳐도 다시 돌릴 필요가 없게 저장된 답을 다시 채점한다
    for r in rows:
        r["correct_at_run"] = r["correct"]
        r["correct"] = by_id(r["task"]).check(r["answer"])
    head = Path(path).with_suffix(".overhead.json")
    over = json.loads(head.read_text())["in"] - 15 if head.exists() else 0     # 15 ~ 우리 쪽 빈 프롬프트
    return rows, over


def table(rows, over):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["family"], r["method"])].append(r)
        groups[("ALL", r["method"])].append(r)
    out.append(f"{'family':9} {'method':9} {'n':>3} {'acc':>5} {'ans%':>5} {'acc|ans':>7} {'calls':>5} "
               f"{'in':>7} {'in-ovh':>7} {'out':>6} {'think':>6} {'cost$':>7} {'sec':>5} {'err':>3}")
    for (fam, m), rs in sorted(groups.items()):
        n = len(rs)
        acc = sum(r["correct"] for r in rs) / n
        ans = [r for r in rs if r["answer"]]
        acca = sum(r["correct"] for r in ans) / len(ans) if ans else float("nan")
        mean = lambda k: sum(r["tok"][k] for r in rs) / n
        calls = mean("calls")
        out.append(f"{fam:9} {m:9} {n:3d} {acc:5.2f} {len(ans) / n:5.2f} {acca:7.2f} {calls:5.1f} "
                   f"{mean('in'):7.0f} {mean('in') - over * calls:7.0f} {mean('out'):6.0f} {mean('thinking'):6.0f} "
                   f"{mean('cost'):7.4f} {mean('sec'):5.0f} {sum(r['tok']['errors'] for r in rs):3d}")
    return out


def paired(rows, over, a="react", b="subbrain"):
    by = defaultdict(dict)
    for r in rows:
        by[(r["task"], r["rep"])][r["method"]] = r
    pairs = [(v[a], v[b]) for v in by.values() if a in v and b in v]
    out = [f"\n짝지은 비교 ({len(pairs)} 쌍)  -- 차이는 {b} - {a}"]
    for fam in ("multihop", "diagnose", None):
        ps = [p for p in pairs if fam is None or p[0]["family"] == fam]
        if not ps:
            continue
        both = sum(x["correct"] and y["correct"] for x, y in ps)
        only_a = sum(x["correct"] and not y["correct"] for x, y in ps)
        only_b = sum(y["correct"] and not x["correct"] for x, y in ps)
        none = len(ps) - both - only_a - only_b
        out.append(f"[{fam or 'ALL'}] 둘다맞음 {both} · {a}만 {only_a} · {b}만 {only_b} · 둘다틀림 {none}")
        for key, lab in (("in", "입력토큰"), ("out", "출력토큰"), ("cost", "비용$")):
            d_all = [y["tok"][key] - x["tok"][key] for x, y in ps]
            d_ok = [y["tok"][key] - x["tok"][key] for x, y in ps if x["correct"] and y["correct"]]
            m, lo, hi = boot(d_all)
            m2, lo2, hi2 = boot(d_ok)
            out.append(f"   {lab:6} 전체 {m:+9.1f} [{lo:+.1f},{hi:+.1f}]   둘다맞은것 {m2:+9.1f} [{lo2:+.1f},{hi2:+.1f}]")
        dc = [y["tok"]["calls"] - x["tok"]["calls"] for x, y in ps]
        m, lo, hi = boot(dc)
        out.append(f"   LLM호출 전체 {m:+.2f} [{lo:+.2f},{hi:+.2f}]")
    return out


if __name__ == "__main__":
    for path in sys.argv[1:]:
        rows, over = load(path)
        print(f"== {path}  (호출당 고정분 추정 {over} 토큰)")
        print("\n".join(table(rows, over)))
        print("\n".join(paired(rows, over)))
