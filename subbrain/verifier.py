"""Verifier -- LLM 에게 "맞아?" 라고 묻지 않는다. 바깥에서 지지 구조를 잰다.

    GROUNDED      IN 이고, 관측(premise)에만 기대고 있다. 도구가 있는 과제라면 그 관측 가운데
                  적어도 하나는 바깥(도구·사람)에서 온 것이어야 한다
    SELF_REPORTED IN 이지만 근거가 전부 LLM 이 스스로 적은 사실이다 -- 세계를 한 번도 안 만졌다
    CONDITIONAL   IN 이지만 가정에 기대고 있다 -- 어느 가정인지, 그 확신도가 얼마인지 같이 준다
    UNPROVEN      정당화는 있는데 전제가 빠졌다 -- 무엇이 빠졌는지 준다
    UNSUPPORTED   근거 없이 말해졌다 (정당화 0)
    CONTRADICTED  그 반대가 관측됐다
    RETRACTED     가정이었는데 모순 때문에(또는 손으로) 내려졌다
    UNDETERMINED  비단조 고리 때문에 참 거짓이 안 정해진다
"""
from __future__ import annotations

from .tms import IN, UNDET, PREMISE, ASSUMPTION, DERIVED

WEAK = 0.6           # 이보다 확신도가 낮은 가정에 기대면 약한 지지다
UNTRUSTED = ("llm", "guess")


def verify(bb, ref: str) -> dict:
    nid = bb.resolve(ref, create=False)
    if nid is None:
        return {"node": ref, "verdict": "UNKNOWN_NODE"}
    t = bb.tms
    n = t.nodes[nid]
    neg = bb.neg_id(nid)
    rep: dict = {"node": nid, "text": n.text, "label": n.label, "kind": n.kind}
    if neg in t.nodes and t.is_in(neg):
        rep["verdict"] = "CONTRADICTED"
        rep["by"] = t.premises_of(neg) or [neg]
    elif n.label == UNDET:
        rep["verdict"] = "UNDETERMINED"
    elif n.label == IN:
        asm = t.assumptions_of(nid)
        rep["premises"] = t.premises_of(nid)
        rep["self_reported"] = [p for p in rep["premises"]
                                if t.nodes[p].meta.get("source", "llm") in UNTRUSTED]
        external = t.touches(nid, lambda n: n.meta.get("source", "llm") not in UNTRUSTED)
        if asm:
            rep["verdict"] = "CONDITIONAL"
            rep["assumptions"] = [{"id": a, "text": t.nodes[a].text,
                                   "confidence": t.nodes[a].meta.get("confidence", 0.5),
                                   "source": t.nodes[a].meta.get("source")} for a in asm]
            rep["weak"] = [a["id"] for a in rep["assumptions"]
                           if float(a["confidence"]) < WEAK or a["source"] in UNTRUSTED]
            rep["risky_nogoods"] = [ng for ng in t.nogoods if set(ng["assumptions"]) & set(asm)]
        elif external or not getattr(bb, "require_external", False):
            rep["verdict"] = "GROUNDED"
        else:
            rep["verdict"] = "SELF_REPORTED"
    elif n.kind in (PREMISE, ASSUMPTION) and n.meta.get("retracted"):
        rep["verdict"] = "RETRACTED"
        rep["reason"] = n.meta["retracted"]
    elif n.kind == ASSUMPTION and n.meta.get("blocked_by"):
        rep["verdict"] = "RETRACTED"
        rep["reason"] = f"nogood {n.meta['blocked_by']}"
    elif not t.justifications_for(nid):
        rep["verdict"] = "UNSUPPORTED"
    else:
        rep["verdict"] = "UNPROVEN"
        rep["missing"] = t.missing_leaves(nid)
    if nid in bb.recheck:
        rep["recheck"] = bb.recheck[nid]
    return rep


def accept(rep: dict) -> bool:
    """받아들일 만한가: 근거가 관측에 닿거나, 기대는 가정이 전부 약하지 않을 때."""
    return rep["verdict"] == "GROUNDED" or (rep["verdict"] == "CONDITIONAL" and not rep.get("weak"))
