"""Impasse 감지 (SOAR 계열) -- 막힌 자리를 이름 붙여 subgoal 로 바꾼다.

LLM 이 "생각 -> 생각 -> 생각" 으로 맴도는 대신, 바깥에서 막힌 꼴을 알아본다.

    CONFLICT           모순이 IN 인데 스스로 못 풀었다 (관측끼리 부딪혔거나 auto_resolve 꺼짐)
    LOOP               비단조 고리 때문에 참 거짓이 안 정해진다
    RECHECK            주장이 지지를 잃었다 -- 무엇이 깨져서인지 같이 준다
    REPEATED_ACTION    같은 행동을 진전 없이 되풀이한다
    STALL              control 이 여러 번 돌았는데 믿음이 하나도 안 바뀌었다
    UNKNOWN            목표가 X 를 요구하는데 X 에 대해 아무것도 모른다
    MISSING_PREMISE    X 를 세울 정당화는 있는데 그 전제가 빠졌다
    UNSUPPORTED_CLAIM  근거 없이 말해진 주장
    WEAK_SUPPORT       목표는 섰는데 약한 가정에 기대고 있다
    UNVERIFIED         목표는 섰는데 근거가 전부 LLM 이 적은 사실이다(도구가 있는데 안 썼다)
    NO_PLAN            연산자 목록으로 빠진 것을 세울 길이 없다
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .tms import IN, UNDET, DERIVED, CONTRADICTION
from .verifier import verify

PRIORITY = {"CONFLICT": 100, "LOOP": 90, "RECHECK": 80, "REPEATED_ACTION": 75, "STALL": 85,
            "MISSING_PREMISE": 60, "UNKNOWN": 60, "NO_PLAN": 50, "WEAK_SUPPORT": 45, "UNVERIFIED": 45,
            "UNSUPPORTED_CLAIM": 40}


@dataclass
class Impasse:
    kind: str
    key: str
    nodes: list[str]
    goal: str | None = None
    detail: dict = field(default_factory=dict)
    priority: float = 0.0

    def __post_init__(self):
        if not self.priority:
            self.priority = PRIORITY[self.kind]


def _depth(bb, gid: str | None) -> int:
    d = 0
    while gid and bb.goals.get(gid) and bb.goals[gid].parent:
        gid = bb.goals[gid].parent
        d += 1
    return d


def relevant(bb) -> set[str] | None:
    """살아 있는 목표가 기대는(기댈 수 있는) 노드 전부. 목표가 없으면 None(=전부 관련)."""
    live = [g for g in bb.goals.values() if g.status != "abandoned"]
    if not live:
        return None
    seen: set[str] = set()
    stack = [r for g in live for r in g.requires]
    while stack:
        x = stack.pop()
        if x in seen:
            continue
        seen.add(x)
        for j in bb.tms.justifications_for(x):
            stack += j.ins + j.outs
    return seen


def detect(bb, planner=None, repeat_limit: int = 2, stall: int = 0, stall_limit: int = 3) -> list[Impasse]:
    t = bb.tms
    found: list[Impasse] = []

    for n in t.nodes.values():
        if n.kind == CONTRADICTION and n.label == IN:
            asm = t.assumptions_of(n.id)
            found.append(Impasse("CONFLICT", f"CONFLICT:{n.id}", [n.id], detail={
                "assumptions": asm, "premises": t.premises_of(n.id),
                "between": t.justs[n.support].ins if n.support in t.justs else []}))
        elif n.label == UNDET and n.kind != CONTRADICTION:
            found.append(Impasse("LOOP", f"LOOP:{n.id}", [n.id]))

    rel = relevant(bb)
    for nid, info in bb.recheck.items():
        if rel is not None and nid not in rel:      # 버린 목표 쪽의 무너짐은 막힘이 아니다(brief 에만 적는다)
            continue
        found.append(Impasse("RECHECK", f"RECHECK:{nid}", [nid], detail=info))

    # 같은 행동을 진전 없이 되풀이 -- 끝에서부터 같은 서명이 몇 번 이어졌나
    if bb.actions:
        last = bb.actions[-1]
        sig = (last.name, sorted(last.args.items()))
        run = [a for a in bb.actions if (a.name, sorted(a.args.items())) == sig and a.progress == last.progress]
        if len(run) >= repeat_limit:
            found.append(Impasse("REPEATED_ACTION", f"REPEAT:{last.name}:{last.progress}", [], detail={
                "action": last.name, "args": last.args, "times": len(run)}))
    if stall >= stall_limit:
        found.append(Impasse("STALL", f"STALL:{t.version}", [], detail={"turns": stall}))

    state = {i for i, n in t.nodes.items() if n.label == IN}
    for g in bb.goals.values():
        if g.status == "abandoned":
            continue
        depth = _depth(bb, g.id)
        if g.status == "open":
            for r in g.requires:
                if t.is_in(r) or t.nodes[r].label == UNDET:
                    continue
                v = verify(bb, r)
                if v["verdict"] in ("CONTRADICTED", "RETRACTED"):
                    found.append(Impasse("CONFLICT", f"BLOCKED:{g.id}:{r}", [r], g.id,
                                         detail={"verdict": v["verdict"], "goal_blocked": True,
                                                 **{k: v[k] for k in ("by", "reason") if k in v}},
                                         priority=PRIORITY["CONFLICT"] - 5))
                    continue
                leaves = t.missing_leaves(r)
                kind = "UNKNOWN" if leaves == [r] else "MISSING_PREMISE"
                for leaf in list(leaves):
                    lv = verify(bb, leaf)
                    if lv["verdict"] in ("CONTRADICTED", "RETRACTED"):
                        # 반대가 관측됐거나 내려진 전제다 -- 근거를 더 찾을 자리가 아니라 길이 막힌 자리다
                        found.append(Impasse("CONFLICT", f"BLOCKED:{g.id}:{leaf}", [leaf], g.id,
                                             detail={"verdict": lv["verdict"], "goal_blocked": True,
                                                     "for": r, **{k: lv[k] for k in ("by", "reason") if k in lv}},
                                             priority=PRIORITY["CONFLICT"] - 5))
                        leaves.remove(leaf)
                for leaf in leaves:
                    imp = Impasse(kind, f"{kind}:{leaf}", [leaf], g.id,
                                  detail={"for": r, "missing": leaves},
                                  priority=PRIORITY[kind] + depth)
                    if planner is not None and planner.ops and not planner.achievers(leaf):
                        imp.detail["no_operator"] = True
                    found.append(imp)
                if planner is not None and planner.ops and leaves and \
                        planner.plan(leaves, state) is None and \
                        not any(planner.achievers(x) for x in leaves):
                    found.append(Impasse("NO_PLAN", f"NO_PLAN:{g.id}:{r}", leaves, g.id))
        elif g.status == "conditional":
            weak = []
            for r in g.requires:
                v = verify(bb, r)
                weak += [a for a in v.get("weak", []) if a not in weak]
                if v["verdict"] == "SELF_REPORTED":
                    found.append(Impasse("UNVERIFIED", f"UNVERIFIED:{g.id}:{r}", [r], g.id,
                                         detail={"self_reported": v["self_reported"]}))
            if weak:
                found.append(Impasse("WEAK_SUPPORT", f"WEAK:{g.id}", weak, g.id, detail={
                    "assumptions": [{"id": a, "text": t.nodes[a].text,
                                     "confidence": t.nodes[a].meta.get("confidence")} for a in weak]}))

    for n in t.nodes.values():
        if n.kind == DERIVED and n.meta.get("source") and n.label != IN \
                and not t.justifications_for(n.id) and n.id not in bb.recheck \
                and not n.meta.get("retracted") and (rel is None or n.id in rel):
            found.append(Impasse("UNSUPPORTED_CLAIM", f"UNSUPPORTED:{n.id}", [n.id]))

    uniq: dict[str, Impasse] = {}
    for i in found:
        if i.key not in uniq or uniq[i.key].priority < i.priority:
            uniq[i.key] = i
    return sorted(uniq.values(), key=lambda i: -i.priority)
