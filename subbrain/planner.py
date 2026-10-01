"""STRIPS 계획기 (Fikes & Nilsson 1971).

연산자 = 이름 · 전제(pre) · 추가(add) · 삭제(delete) · 비용. 상태는 IN 인 노드 id 의 집합이다.
연산자는 두 종류다:

    tool       바깥 세계에서 실제로 하는 일(검색 · 계산 · 측정). 실행은 호출자가 한다.
    cognitive  LLM 에게 시킬 사고 작업.

계획기는 LLM 에게 "전체 계획을 세워라" 를 요구하지 않는다. 막힌 노드 하나를 세우는
가장 싼 연산자 사슬을 A* 로 찾아 '다음 한 걸음' 만 내놓는다.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field, asdict


@dataclass
class Operator:
    name: str
    pre: list[str] = field(default_factory=list)
    add: list[str] = field(default_factory=list)
    delete: list[str] = field(default_factory=list)
    cost: float = 1.0
    kind: str = "tool"
    doc: str = ""


class Planner:
    def __init__(self) -> None:
        self.ops: dict[str, Operator] = {}

    def register(self, op: Operator) -> None:
        self.ops[op.name] = op

    def achievers(self, literal: str) -> list[Operator]:
        return sorted((o for o in self.ops.values() if literal in o.add), key=lambda o: o.cost)

    def plan(self, goal: list[str], state: set[str], max_expand: int = 5000) -> list[Operator] | None:
        goal_s = set(goal)
        if goal_s <= state:
            return []
        start = frozenset(state)
        tie = 0
        frontier = [(len(goal_s - start), 0.0, tie, start, [])]
        best: dict[frozenset, float] = {start: 0.0}
        expanded = 0
        while frontier and expanded < max_expand:
            _f, g, _t, s, path = heapq.heappop(frontier)
            if goal_s <= s:
                return [self.ops[n] for n in path]
            expanded += 1
            for o in self.ops.values():
                if not set(o.pre) <= s:
                    continue
                ns = frozenset((s - set(o.delete)) | set(o.add))
                if ns == s:
                    continue
                ng = g + o.cost
                if ng < best.get(ns, float("inf")):
                    best[ns] = ng
                    tie += 1
                    heapq.heappush(frontier, (ng + len(goal_s - ns), ng, tie, ns, path + [o.name]))
        return None

    def to_dict(self) -> list[dict]:
        return [asdict(o) for o in self.ops.values()]

    @classmethod
    def from_dict(cls, d: list[dict]) -> "Planner":
        p = cls()
        for o in d or []:
            p.register(Operator(**o))
        return p
