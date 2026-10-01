"""Justification-based Truth Maintenance System (Doyle 1979 계열).

모든 믿음은 노드이고, 노드는 정당화(justification)를 통해서만 IN 이 된다.

    premise     관측된 사실. 켜져 있으면 IN.
    assumption  가정. 켜져 있으면 IN. 모순이 나면 되돌릴 후보가 된다.
    derived     정당화로만 IN 이 된다. 정당화가 없으면 영원히 OUT.
    contradiction  IN 이 되면 모순이다. 그 밑에 깔린 가정 집합이 nogood 이 된다.

정당화는  (in-list 가 전부 IN) 이고 (out-list 가 전부 OUT) 일 때 결론을 IN 으로 만든다.
out-list 가 있으면 비단조라서 라벨이 하나로 안 정해질 수 있다. 그래서 라벨은
well-founded semantics 의 교대 고정점(alternating fixpoint)으로 매긴다:

    L0 = {} ,  U_i = G(L_i) ,  L_{i+1} = G(U_i)        (G 는 Gelfond-Lifschitz 연산)

L 에 든 노드가 IN, U 에도 안 든 노드가 OUT, 그 사이(U - L)가 UNDETERMINED 다.
UNDETERMINED 는 홀수 고리(A unless A 같은 것)나 짝수 고리에서 나오고, 위층에서는
그것을 impasse(LOOP)로 올린다. 조용히 OUT 으로 접지 않는다.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

IN, OUT, UNDET = "IN", "OUT", "UNDETERMINED"
PREMISE, ASSUMPTION, DERIVED, CONTRADICTION = "premise", "assumption", "derived", "contradiction"
KINDS = (PREMISE, ASSUMPTION, DERIVED, CONTRADICTION)


@dataclass
class Node:
    id: str
    text: str
    kind: str = DERIVED
    enabled: bool = False          # premise / assumption 만 의미가 있다
    label: str = OUT
    support: str | None = None     # IN 일 때 그것을 세운 정당화 id ("premise" / "assumption" 포함)
    created: int = 0
    meta: dict = field(default_factory=dict)


@dataclass
class Justification:
    id: str
    consequent: str
    ins: list[str]
    outs: list[str] = field(default_factory=list)
    informant: str = ""            # 누가 세웠나: llm / tool / user / rule 이름
    active: bool = True


class TMS:
    def __init__(self) -> None:
        self.nodes: dict[str, Node] = {}
        self.justs: dict[str, Justification] = {}
        self.nogoods: list[dict] = []        # {"assumptions": [...], "contradiction": id}
        self.clock = 0                       # 노드 생성 순서
        self.version = 0                     # 라벨이 바뀐 횟수(진전의 척도)
        self._jseq = 0

    # ------------------------------------------------------------------ build
    def node(self, id: str, text: str | None = None, kind: str | None = None, **meta) -> Node:
        """노드를 만들거나 가져온다. kind 를 주면 승격한다(derived -> premise 등)."""
        n = self.nodes.get(id)
        if n is None:
            self.clock += 1
            n = Node(id=id, text=text or id, kind=kind or DERIVED, created=self.clock)
            self.nodes[id] = n
        else:
            if text and n.text == n.id:
                n.text = text
            if kind and kind != n.kind:
                if kind not in KINDS:
                    raise ValueError(f"unknown kind {kind!r}")
                n.kind = kind
        n.meta.update({k: v for k, v in meta.items() if v is not None})
        return n

    def justify(self, consequent: str, ins: list[str], outs: list[str] | None = None,
                informant: str = "") -> Justification:
        outs = list(outs or [])
        for x in [consequent, *ins, *outs]:
            if x not in self.nodes:
                self.node(x)
        for j in self.justs.values():           # 같은 정당화를 두 번 넣지 않는다
            if j.consequent == consequent and sorted(j.ins) == sorted(ins) and sorted(j.outs) == sorted(outs):
                j.active = True
                return j
        self._jseq += 1
        j = Justification(id=f"j{self._jseq}", consequent=consequent, ins=list(ins), outs=outs,
                          informant=informant)
        self.justs[j.id] = j
        return j

    def justifications_for(self, id: str, active_only: bool = True) -> list[Justification]:
        return [j for j in self.justs.values() if j.consequent == id and (j.active or not active_only)]

    def enable(self, id: str) -> None:
        self.nodes[id].enabled = True

    def disable(self, id: str) -> None:
        self.nodes[id].enabled = False

    # --------------------------------------------------------------- labeling
    def _gamma(self, guess: set[str], record: bool = False) -> tuple[set[str], dict[str, str]]:
        """guess 에 대한 reduct 의 최소 모형. record 면 각 노드를 세운 정당화를 남긴다."""
        model: set[str] = set()
        support: dict[str, str] = {}
        for n in self.nodes.values():
            if n.kind in (PREMISE, ASSUMPTION) and n.enabled:
                model.add(n.id)
                support[n.id] = n.kind
        live = [j for j in self.justs.values() if j.active and not (set(j.outs) & guess)]
        changed = True
        while changed:
            changed = False
            for j in live:
                if j.consequent in model:
                    continue
                if all(a in model for a in j.ins):
                    model.add(j.consequent)
                    support[j.consequent] = j.id
                    changed = True
        return model, support

    def relabel(self) -> dict[str, tuple[str, str]]:
        """라벨을 다시 매기고 바뀐 것만 돌려준다: {id: (old, new)}."""
        lower: set[str] = set()
        for _ in range(4 * len(self.nodes) + 4):
            upper, _s = self._gamma(lower)
            new_lower, support = self._gamma(upper, record=True)
            if new_lower == lower:
                break
            lower = new_lower
        upper, _s = self._gamma(lower)
        _l, support = self._gamma(upper)
        changes: dict[str, tuple[str, str]] = {}
        for n in self.nodes.values():
            new = IN if n.id in lower else (UNDET if n.id in upper else OUT)
            if new != n.label:
                changes[n.id] = (n.label, new)
                n.label = new
            n.support = support.get(n.id) if new == IN else None
        if changes:
            self.version += 1
        return changes

    # ------------------------------------------------------------ explaining
    def is_in(self, id: str) -> bool:
        n = self.nodes.get(id)
        return bool(n and n.label == IN)

    def _walk(self, id: str, seen: set[str]):
        """IN 노드의 지지 사슬을 따라간다(well-founded 라 고리가 없다)."""
        if id in seen:
            return
        seen.add(id)
        n = self.nodes[id]
        yield n
        if n.label != IN or n.support in (None, PREMISE, ASSUMPTION):
            return
        for a in self.justs[n.support].ins:
            yield from self._walk(a, seen)

    def assumptions_of(self, id: str) -> list[str]:
        return [n.id for n in self._walk(id, set()) if n.kind == ASSUMPTION and n.support == ASSUMPTION]

    def premises_of(self, id: str) -> list[str]:
        return [n.id for n in self._walk(id, set()) if n.kind == PREMISE and n.support == PREMISE]

    def touches(self, id: str, pred, seen: set[str] | None = None) -> bool:
        """IN 인 정당화 어느 하나라도 따라가서 pred 를 만족하는 premise 에 닿는가."""
        seen = seen if seen is not None else set()
        if id in seen or not self.is_in(id):
            return False
        seen.add(id)
        n = self.nodes[id]
        if n.kind == PREMISE and n.enabled and pred(n):
            return True
        return any(all(self.is_in(a) for a in j.ins) and all(not self.is_in(b) for b in j.outs)
                   and any(self.touches(a, pred, seen) for a in j.ins)
                   for j in self.justifications_for(id))

    def why(self, id: str, depth: int = 0, max_depth: int = 8) -> dict:
        """IN 이면 지지 사슬을, OUT 이면 왜 안 서는지를 나무로 돌려준다."""
        n = self.nodes[id]
        out: dict = {"id": n.id, "text": n.text, "kind": n.kind, "label": n.label}
        if depth >= max_depth:
            out["truncated"] = True
            return out
        if n.label == IN:
            if n.support in (PREMISE, ASSUMPTION):
                out["because"] = n.support
            else:
                j = self.justs[n.support]
                out["because"] = j.id
                out["informant"] = j.informant
                out["ins"] = [self.why(a, depth + 1, max_depth) for a in j.ins]
                if j.outs:
                    out["unless"] = j.outs
            return out
        if n.kind in (PREMISE, ASSUMPTION) and not n.enabled:
            out["because"] = "retracted" if n.meta.get("retracted") else "disabled"
        js = self.justifications_for(id)
        if not js and n.kind == DERIVED:
            out["because"] = "no-justification"
        if js:
            out["failing"] = []
            for j in js:
                out["failing"].append({
                    "just": j.id,
                    "missing": [a for a in j.ins if not self.is_in(a)],
                    "defeated_by": [b for b in j.outs if self.nodes[b].label != OUT],
                })
        return out

    def missing_leaves(self, id: str, seen: set[str] | None = None, depth: int = 0) -> list[str]:
        """OUT 노드를 세우려면 결국 무엇이 필요한가 -- 정당화를 아래로 따라가 맨 끝을 모은다.

        정당화가 여럿이면 빠진 것이 가장 적은 것 하나만 따라간다(가장 싼 길)."""
        seen = seen if seen is not None else set()
        n = self.nodes[id]
        if n.label == IN or id in seen or depth > 12:
            return []
        seen.add(id)
        js = self.justifications_for(id)
        if not js:
            return [id]
        best: list[str] | None = None
        for j in js:
            miss = [a for a in j.ins if not self.is_in(a)]
            leaves: list[str] = []
            for a in miss:
                for x in self.missing_leaves(a, seen, depth + 1):
                    if x not in leaves:
                        leaves.append(x)
            if best is None or len(leaves) < len(best):
                best = leaves
        return best or []

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"nodes": [asdict(n) for n in self.nodes.values()],
                "justs": [asdict(j) for j in self.justs.values()],
                "nogoods": self.nogoods, "clock": self.clock, "version": self.version,
                "jseq": self._jseq}

    @classmethod
    def from_dict(cls, d: dict) -> "TMS":
        t = cls()
        for n in d.get("nodes", []):
            t.nodes[n["id"]] = Node(**n)
        for j in d.get("justs", []):
            t.justs[j["id"]] = Justification(**j)
        t.nogoods = d.get("nogoods", [])
        t.clock, t.version, t._jseq = d.get("clock", 0), d.get("version", 0), d.get("jseq", 0)
        return t
