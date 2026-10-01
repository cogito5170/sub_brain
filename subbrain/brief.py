"""LLM 에게 보내는 한 덩어리. 긴 설명이 아니라 막힌 자리와 다음 연산만 준다."""
from __future__ import annotations

from .tms import IN, OUT, UNDET, PREMISE, ASSUMPTION, CONTRADICTION

MARK = {IN: "+", OUT: "-", UNDET: "?"}


def _n(brain, nid: str) -> str:
    n = brain.bb.tms.nodes.get(nid)
    if not n:
        return nid
    tag = {PREMISE: "F", ASSUMPTION: "A"}.get(n.kind, "C")
    extra = f" c={n.meta['confidence']}" if n.kind == ASSUMPTION and "confidence" in n.meta else ""
    return f"{nid}[{tag}{MARK[n.label]}{extra}] {n.text}"


def _goal_tree(brain) -> list[str]:
    goals = brain.bb.goals
    out: list[str] = []

    def walk(gid: str, depth: int):
        g = goals[gid]
        if g.status == "abandoned":
            return
        out.append("  " * depth + f"{gid} [{g.status}] {g.text}  needs: {', '.join(g.requires)}")
        for c in goals.values():
            if c.parent == gid:
                walk(c.id, depth + 1)

    for g in goals.values():
        if g.parent is None:
            walk(g.id, 0)
    return out


def render(brain, d: dict, budget: int = 1600) -> str:
    t = brain.bb.tms
    lines: list[str] = []
    if brain.bb.goals:
        lines.append("GOALS")
        lines += ["  " + s for s in _goal_tree(brain)]
    imp = d.get("impasse")
    if d["status"] == "DONE":
        lines.append("STATUS DONE -- 모든 목표가 섰다. LLM 호출 필요 없음.")
    elif imp:
        lines.append(f"IMPASSE {imp['kind']}  " + ", ".join(_n(brain, x) for x in imp["nodes"]))
        det = imp["detail"]
        if imp["kind"] in ("UNKNOWN", "MISSING_PREMISE"):
            for_ = det.get("for")
            if for_ and for_ not in imp["nodes"]:
                lines.append(f"  for: {_n(brain, for_)}")
                have = []
                for j in t.justifications_for(for_):
                    have += [a for a in j.ins if t.is_in(a) and a not in have]
                if have:
                    lines.append("  have: " + "; ".join(_n(brain, a) for a in have))
                lines.append("  missing: " + "; ".join(_n(brain, a) for a in det.get("missing", [])))
        elif imp["kind"] == "CONFLICT":
            if det.get("between"):
                lines.append("  between: " + "; ".join(_n(brain, a) for a in det["between"]))
            if det.get("assumptions"):
                lines.append("  rests on: " + "; ".join(_n(brain, a) for a in det["assumptions"]))
        elif imp["kind"] == "RECHECK":
            cause = det.get("cause") or []
            if cause:
                lines.append("  broken: " + "; ".join(_n(brain, a) for a in cause))
        elif imp["kind"] == "UNVERIFIED":
            lines.append("  rests only on your own notes: " + "; ".join(_n(brain, a) for a in det.get("self_reported", [])))
        mem = brain.memory.recall(" ".join(t.nodes[x].text for x in imp["nodes"] if x in t.nodes)
                                  or imp["kind"], ["failure", "procedural", "episodic"], k=2)
        for m in mem:
            lines.append(f"  memory[{m['kind']}]: {m['text']}")
    if d.get("retracted"):
        for r in d["retracted"]:
            lines.append(f"ROLLBACK {_n(brain, r['assumption'])}  (nogood {r['nogood']})")
    rc = [x for x in brain.bb.recheck if not (imp and x in imp["nodes"])]
    if rc:
        lines.append("RECHECK " + "; ".join(_n(brain, x) for x in rc))
    op = d.get("op")
    if op:
        tgt = op.get("target") or op.get("candidates") or op.get("avoid") or ""
        lines.append(f"NEXT {op['op']}({tgt if isinstance(tgt, str) else ','.join(map(str, tgt))}) by {op['by']} -- {op['why']}")
        for a in d.get("alternatives", []):
            tgt = a.get("target") or a.get("candidates") or ""
            lines.append(f"  or {a['op']}({tgt if isinstance(tgt, str) else ','.join(map(str, tgt))}) by {a['by']}")
    tools = [o for o in brain.planner.ops.values() if o.kind == "tool"]
    if tools and (d.get("call_llm") or d["status"] == "ACT"):
        lines.append("TOOLS " + "; ".join(
            f"{o.name}({','.join(o.pre)})->{','.join(o.add)}" + (f" {o.doc}" if o.doc else "") for o in tools))
    # 남은 자리에 지금 믿는 것(IN)을 채운다 -- 최근 것부터
    head = "\n".join(lines)
    beliefs = sorted((n for n in t.nodes.values() if n.label == IN and n.kind != CONTRADICTION),
                     key=lambda n: -n.created)
    tail: list[str] = []
    room = budget - len(head) - 10
    for n in beliefs:
        s = "  " + _n(brain, n.id)
        if room - len(s) - 1 < 0:
            tail.append(f"  ... +{len(beliefs) - len(tail)} more")
            break
        tail.append(s)
        room -= len(s) + 1
    if tail:
        head += "\nBELIEFS\n" + "\n".join(tail)
    return head[:budget]
