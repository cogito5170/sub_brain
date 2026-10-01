"""장기 기억 -- 세션을 넘어 남는다. 벡터 DB 가 아니라 BM25 (의존성 0).

    episodic    지난 문제를 어떻게 풀었나 (목표 · 결과 · 걸음 수)
    semantic    사실 · 관계
    procedural  문제 해결 방법
    failure     실패한 접근 -- nogood 과 실패한 행동이 자동으로 쌓인다

한국어는 띄어쓰기 단위 토큰만으로는 조사 때문에 안 맞는다("센서가" vs "센서").
그래서 단어와 함께 글자 2-gram 을 같이 색인한다.
"""
from __future__ import annotations

import json
import math
import re
import time
from pathlib import Path

KINDS = ("episodic", "semantic", "procedural", "failure")


def tokens(text: str) -> list[str]:
    out: list[str] = []
    for w in re.findall(r"\w+", text.lower()):
        out.append(w)
        if re.search(r"[^\x00-\x7f]", w) and len(w) > 1:
            out.extend(w[i:i + 2] for i in range(len(w) - 1))
    return out


class Memory:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else None
        self.items: list[dict] = []
        if self.path and self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.items.append(json.loads(line))

    def add(self, kind: str, text: str, tags: list[str] | None = None, **meta) -> dict:
        if kind not in KINDS:
            raise ValueError(f"memory kind must be one of {KINDS}")
        for it in self.items:                      # 같은 기억을 두 번 쌓지 않는다
            if it["kind"] == kind and it["text"] == text:
                it["hits"] = it.get("hits", 0) + 1
                self._flush()
                return it
        it = {"id": f"m{len(self.items) + 1}", "kind": kind, "text": text, "tags": tags or [],
              "t": round(time.time()), **meta}
        self.items.append(it)
        self._flush()
        return it

    def _flush(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in self.items),
                       encoding="utf-8")
        tmp.replace(self.path)

    def recall(self, query: str, kinds: list[str] | None = None, k: int = 3,
               min_score: float = 0.5) -> list[dict]:
        pool = [i for i in self.items if not kinds or i["kind"] in kinds]
        if not pool:
            return []
        docs = [tokens(i["text"] + " " + " ".join(i.get("tags", []))) for i in pool]
        q = set(tokens(query))
        N = len(docs)
        avg = sum(map(len, docs)) / N or 1.0
        df: dict[str, int] = {}
        for d in docs:
            for t in set(d):
                df[t] = df.get(t, 0) + 1
        scored = []
        for it, d in zip(pool, docs):
            s = 0.0
            for t in q:
                f = d.count(t)
                if not f:
                    continue
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * len(d) / avg))
            if s >= min_score:
                scored.append((s, it))
        scored.sort(key=lambda x: -x[0])
        return [dict(it, score=round(s, 2)) for s, it in scored[:k]]
