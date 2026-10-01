"""Blackboard (Hearsay-II 계열) -- 지금의 사고 상태 전부가 여기 한 곳에 있다.

    GOAL / SUBGOAL     무엇을 세우려 하는가 (requires: 서야 하는 노드들)
    FACT               관측 -> TMS premise
    ASSUMPTION         가정 -> TMS assumption (모순 시 되돌릴 후보)
    CLAIM              추론 -> TMS derived (because 없이 말하면 OUT 으로 남는다)
    QUESTION           열린 물음
    ACTION             한 행동과 결과 (반복 감지용)
    RECHECK            지지를 잃은 주장 -- "다시 보라" 는 표시와 그 원인

LLM 이 무엇을 말하든 노드는 **근거가 있을 때만 IN** 이다. 말한 것과 믿는 것을 가른다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from .tms import TMS, IN, OUT, UNDET, PREMISE, ASSUMPTION, DERIVED, CONTRADICTION

NEG = "not:"


_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,40}$")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower())


@dataclass
class Goal:
    id: str
    text: str
    requires: list[str]
    parent: str | None = None
    origin: str | None = None          # 이 subgoal 을 낳은 impasse key
    status: str = "open"               # open / achieved / conditional / abandoned
    created: int = 0


@dataclass
class Action:
    seq: int
    name: str
    args: dict
    ok: bool | None = None
    result: str | None = None
    progress: int = 0                  # 이 행동 시점의 TMS version


@dataclass
class Event:
    seq: int
    kind: str
    detail: dict


class Blackboard:
    def __init__(self) -> None:
        self.tms = TMS()
        self.goals: dict[str, Goal] = {}
        self.questions: dict[str, dict] = {}
        self.actions: list[Action] = []
        self.recheck: dict[str, dict] = {}     # node id -> {"cause": [...], "since": seq}
        self.events: list[Event] = []
        self.seq = 0
        self._ids = {"n": 0, "g": 0, "q": 0}
        self.auto_resolve = True               # 모순을 만나면 가장 약한 가정을 스스로 내린다
        self.requests: list[dict] = []         # LLM 이 실행을 청한 도구 {name, args, why}
        self.require_external = False          # 도구가 있으면 True -- 답은 세계를 한 번은 만져야 한다

    # ------------------------------------------------------------- utilities
    def _next_id(self, p: str) -> str:
        self._ids[p] += 1
        return f"{p}{self._ids[p]}"

    def log(self, kind: str, **detail) -> None:
        self.seq += 1
        self.events.append(Event(self.seq, kind, detail))
        if len(self.events) > 500:
            self.events = self.events[-500:]

    def resolve(self, ref: str, create: bool = True) -> str | None:
        """id 나 글귀로 노드를 찾는다. 없으면 (create 면) 근거 없는 노드로 만든다."""
        if ref is None:
            raise ValueError("node reference is None")
        ref = str(ref)
        if ref in self.tms.nodes:
            return ref
        key = _norm(ref)
        for n in self.tms.nodes.values():
            if _norm(n.text) == key:
                return n.id
        if not create:
            return None
        if _IDENT.match(ref):              # "X", "temp_ok" 같은 이름은 그대로 id 로 쓴다
            self.tms.node(ref)
            return ref
        nid = self._next_id("n")
        while nid in self.tms.nodes:
            nid = self._next_id("n")
        self.tms.node(nid, text=ref)
        return nid

    def _named(self, text: str, id: str | None) -> str:
        if id:
            n = self.tms.node(id, text=text)
            return n.id
        return self.resolve(text)

    def text(self, nid: str) -> str:
        n = self.tms.nodes.get(nid)
        return n.text if n else nid

    def neg_id(self, nid: str) -> str:
        return nid[len(NEG):] if nid.startswith(NEG) else NEG + nid

    # ------------------------------------------------------------ writing
    def fact(self, text: str, id: str | None = None, source: str = "user", **meta) -> str:
        nid = self._named(text, id)
        n = self.tms.node(nid, kind=PREMISE, source=source, **meta)
        n.enabled = True
        n.meta.pop("retracted", None)
        neg = self.neg_id(nid)
        if neg in self.tms.nodes:          # 반대를 관측했던 적이 있으면 짝을 맺어 둔다
            self.conflict(nid, neg, informant="negation")
        self.log("fact", id=nid, source=source)
        return nid

    def assume(self, text: str, id: str | None = None, source: str = "llm",
               confidence: float = 0.5, force: bool = False, **meta) -> str:
        nid = self._named(text, id)
        n = self.tms.node(nid, kind=ASSUMPTION, source=source, confidence=confidence, **meta)
        if not force:
            blocked = self._nogood_hit(nid)
            if blocked:
                n.meta["blocked_by"] = blocked["assumptions"]
                self.log("assume-refused", id=nid, nogood=blocked["assumptions"])
                return nid
        n.enabled = True
        n.meta.pop("retracted", None)
        n.meta.pop("blocked_by", None)
        self.log("assume", id=nid, source=source)
        return nid

    def claim(self, text: str, because: list[str] | None = None, unless: list[str] | None = None,
              id: str | None = None, source: str = "llm", confidence: float | None = None, **meta) -> str:
        nid = self._named(text, id)
        self.tms.node(nid, source=source, confidence=confidence, **meta)
        if because or unless:
            ins = [self.resolve(b) for b in (because or [])]
            outs = [self.resolve(u) for u in (unless or [])]
            self.tms.justify(nid, ins, outs, informant=source)
        self.recheck.pop(nid, None)
        self.log("claim", id=nid, source=source, because=because or [], unless=unless or [])
        return nid

    def rule(self, then: str, when: list[str], unless: list[str] | None = None,
             informant: str = "rule") -> str:
        """claim 과 같지만 '말' 이 아니라 '규칙' 이다 -- 아직 아무도 그 결론을 주장하지 않았다."""
        nid = self.resolve(then)
        self.tms.justify(nid, [self.resolve(w) for w in when],
                         [self.resolve(u) for u in (unless or [])], informant=informant)
        self.log("rule", id=nid)
        return nid

    def observe(self, ref: str, value: bool = True, source: str = "tool") -> str:
        """관측. False 면 그 노드를 내리고 반대 노드(not:X)를 사실로 세운다."""
        nid = self.resolve(ref)
        neg = self.neg_id(nid)
        if value:
            if neg in self.tms.nodes and self.tms.nodes[neg].kind == PREMISE:
                self.tms.nodes[neg].enabled = False
            return self.fact(self.text(nid), id=nid, source=source)
        n = self.tms.nodes[nid]
        if n.kind in (PREMISE, ASSUMPTION):
            n.enabled = False
            n.meta["retracted"] = f"observed false ({source})"
        self.tms.node(neg, text="NOT " + n.text)
        self.fact("NOT " + n.text, id=neg, source=source)
        self.conflict(nid, neg, informant="negation")
        self.log("observe-false", id=nid, source=source)
        return neg

    def retract(self, ref: str, reason: str = "") -> str:
        nid = self.resolve(ref, create=False)
        if nid is None:
            raise KeyError(ref)
        n = self.tms.nodes[nid]
        n.enabled = False
        n.meta["retracted"] = reason or "retracted"
        if n.kind == DERIVED:                 # 주장을 거두면 그 주장의 정당화를 끈다
            for j in self.tms.justifications_for(nid):
                j.active = False
        self.log("retract", id=nid, reason=reason)
        return nid

    def conflict(self, a: str, b: str, informant: str = "user") -> str:
        """a 와 b 는 함께 설 수 없다."""
        a, b = self.resolve(a), self.resolve(b)
        x, y = sorted([a, b])
        cid = f"⊥({x},{y})"
        self.tms.node(cid, text=f"{self.text(x)}  ⊥  {self.text(y)}", kind=CONTRADICTION)
        self.tms.justify(cid, [x, y], informant=informant)
        return cid

    def goal(self, text: str, requires: list[str] | None = None, parent: str | None = None,
             origin: str | None = None, id: str | None = None) -> str:
        reqs = [self.resolve(r) for r in (requires or [text])]
        for g in self.goals.values():      # 같은 subgoal 을 두 번 만들지 않는다
            if origin and g.origin == origin and g.status == "open":
                return g.id
        gid = id
        while not gid or (gid in self.goals and gid != id):
            gid = self._next_id("g")
        self.goals[gid] = Goal(gid, text, reqs, parent=parent, origin=origin, created=self.seq)
        self.log("goal", id=gid, requires=reqs, parent=parent)
        return gid

    def abandon(self, gid: str, reason: str = "") -> None:
        self.goals[gid].status = "abandoned"
        self.log("abandon", id=gid, reason=reason)

    def question(self, text: str, about: list[str] | None = None) -> str:
        qid = self._next_id("q")
        self.questions[qid] = {"text": text, "about": [self.resolve(a) for a in (about or [])],
                               "open": True}
        return qid

    def act(self, name: str, args: dict | None = None, ok: bool | None = None,
            result: str | None = None) -> Action:
        a = Action(len(self.actions) + 1, name, dict(args or {}), ok, result, self.tms.version)
        self.actions.append(a)
        self.log("act", name=name, ok=ok)
        return a

    # ------------------------------------------------------ maintenance loop
    def _nogood_hit(self, nid: str) -> dict | None:
        on = {n.id for n in self.tms.nodes.values() if n.kind == ASSUMPTION and n.enabled} | {nid}
        for ng in self.tms.nogoods:
            s = set(ng["assumptions"])
            if nid in s and s <= on:
                return ng
        return None

    def maintain(self) -> dict:
        """라벨을 다시 매기고, 모순을 되돌리고(DDB), 지지를 잃은 주장을 RECHECK 로 표시한다."""
        before = {i: n.label for i, n in self.tms.nodes.items()}
        report = {"changed": {}, "nogoods": [], "retracted": [], "unresolved": []}
        for _ in range(20):
            self.tms.relabel()
            bad = [n for n in self.tms.nodes.values() if n.kind == CONTRADICTION and n.label == IN]
            if not bad:
                break
            progressed = False
            for c in bad:
                culprits = self.tms.assumptions_of(c.id)
                key = sorted(culprits)
                if key and not any(sorted(ng["assumptions"]) == key for ng in self.tms.nogoods):
                    self.tms.nogoods.append({"assumptions": key, "contradiction": c.id})
                    report["nogoods"].append({"assumptions": key, "contradiction": c.id})
                if not culprits:
                    report["unresolved"].append({"contradiction": c.id,
                                                 "premises": self.tms.premises_of(c.id)})
                    continue
                if not self.auto_resolve:
                    report["unresolved"].append({"contradiction": c.id, "assumptions": key})
                    continue
                victim = min(culprits, key=lambda a: (
                    float(self.tms.nodes[a].meta.get("confidence", 0.5)), -self.tms.nodes[a].created))
                self.tms.nodes[victim].enabled = False
                self.tms.nodes[victim].meta["retracted"] = f"nogood {key} via {c.id}"
                report["retracted"].append({"assumption": victim, "contradiction": c.id, "nogood": key})
                self.log("ddb-retract", id=victim, contradiction=c.id)
                progressed = True
            if not progressed:
                break
        after = {i: n.label for i, n in self.tms.nodes.items()}
        changed = {i: (before.get(i, OUT), after[i]) for i in after if before.get(i, OUT) != after[i]}
        report["changed"] = changed
        lost = [i for i, (o, nw) in changed.items() if o == IN and nw != IN]
        for i in lost:
            n = self.tms.nodes[i]
            if n.kind == DERIVED:
                cause = [i2 for i2, (o, nw) in changed.items() if o == IN and nw != IN
                         and self.tms.nodes[i2].kind in (PREMISE, ASSUMPTION)]
                self.recheck[i] = {"cause": cause, "since": self.seq}
        for i in list(self.recheck):
            if self.tms.is_in(i):
                self.recheck.pop(i)
        self._update_goals()
        return report

    def _update_goals(self) -> None:
        for g in self.goals.values():
            if g.status == "abandoned":
                continue
            if all(self.tms.is_in(r) for r in g.requires):
                rests = {a for r in g.requires for a in self.tms.assumptions_of(r)}
                untouched = self.require_external and not all(
                    self.tms.touches(r, lambda n: n.meta.get("source", "llm") not in ("llm", "guess"))
                    for r in g.requires)
                g.status = "conditional" if rests or untouched else "achieved"
            else:
                g.status = "open"

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {"tms": self.tms.to_dict(), "goals": [asdict(g) for g in self.goals.values()],
                "questions": self.questions, "actions": [asdict(a) for a in self.actions],
                "recheck": self.recheck, "events": [asdict(e) for e in self.events[-200:]],
                "seq": self.seq, "ids": self._ids, "auto_resolve": self.auto_resolve,
                "requests": self.requests, "require_external": self.require_external}

    @classmethod
    def from_dict(cls, d: dict) -> "Blackboard":
        b = cls()
        b.tms = TMS.from_dict(d.get("tms", {}))
        b.goals = {g["id"]: Goal(**g) for g in d.get("goals", [])}
        b.questions = d.get("questions", {})
        b.actions = [Action(**a) for a in d.get("actions", [])]
        b.recheck = d.get("recheck", {})
        b.events = [Event(**e) for e in d.get("events", [])]
        b.seq, b._ids = d.get("seq", 0), d.get("ids", b._ids)
        b.auto_resolve = d.get("auto_resolve", True)
        b.requests = d.get("requests", [])
        b.require_external = d.get("require_external", False)
        return b
