"""LLM <-> 보조-뇌 사이의 말: JSON 연산 목록.

LLM 은 산문 대신(또는 산문과 함께) 이런 블록을 낸다.

    ```subbrain
    {"ops": [
      {"op": "fact",   "text": "센서가 정상이다", "id": "sensor_ok"},
      {"op": "assume", "text": "온도가 정상이다", "id": "temp_ok", "confidence": 0.7},
      {"op": "claim",  "text": "A는 안전하다", "id": "safe", "because": ["sensor_ok", "temp_ok"]},
      {"op": "goal",   "text": "A가 안전한지 판정", "requires": ["safe"]}
    ]}
    ```

노드는 id 로도, 글귀 그대로로도 가리킬 수 있다. 처음 보는 글귀는 근거 없는 노드가 된다.
"""
from __future__ import annotations

import json
import re

OPS = {
    "fact": "관측된 사실. {text, id?, source?}",
    "assume": "가정. {text, id?, confidence?=0.5, source?}",
    "claim": "추론한 주장. because 가 없으면 믿지 않는다. {text, because?:[..], unless?:[..], id?}",
    "rule": "규칙(아직 주장 아님). {then, when:[..], unless?:[..]}",
    "observe": "관측 결과. {node, value: true|false}  false 면 반대가 사실이 된다",
    "retract": "거둔다. {node, reason?}",
    "conflict": "둘은 함께 설 수 없다. {a, b}",
    "goal": "목표. {text, requires?:[..], parent?}",
    "abandon": "목표를 버린다. {goal, reason?}",
    "question": "열린 물음. {text, about?:[..]}",
    "act": "한 행동의 기록. {name, args?, ok?, result?}",
    "remember": "장기 기억. {kind: episodic|semantic|procedural|failure, text, tags?}",
    "use": "도구 실행을 청한다(TOOLS 에 있는 것만). {name, args?, why?} 결과는 다음 턴에 사실로 들어온다",
    "operator": "STRIPS 연산자. {name, pre?, add?, delete?, cost?, kind?: tool|cognitive, doc?}",
}

_ALIASES = {"node": ("node", "ref", "id", "text")}


def _get(op: dict, *names, default=None):
    for n in names:
        if n in op and op[n] is not None:
            return op[n]
    return default


def apply_one(brain, op: dict) -> dict:
    kind = op.get("op")
    bb = brain.bb
    if kind == "fact":
        return {"id": bb.fact(op["text"], id=op.get("id"), source=op.get("source", "llm"))}
    if kind == "assume":
        nid = bb.assume(op["text"], id=op.get("id"), source=op.get("source", "llm"),
                        confidence=float(op.get("confidence", 0.5)), force=bool(op.get("force")))
        n = bb.tms.nodes[nid]
        r = {"id": nid}
        if n.meta.get("blocked_by"):
            r["refused"] = f"nogood {n.meta['blocked_by']}"
        return r
    if kind == "claim":
        return {"id": bb.claim(op["text"], because=op.get("because"), unless=op.get("unless"),
                               id=op.get("id"), source=op.get("source", "llm"),
                               confidence=op.get("confidence"))}
    if kind == "rule":
        return {"id": bb.rule(op["then"], op.get("when", []), op.get("unless"),
                              informant=op.get("source", "rule"))}
    if kind == "observe":
        return {"id": bb.observe(_get(op, "node", "ref", "id", "text"), bool(op.get("value", True)),
                                 source=op.get("source", "llm"))}
    if kind == "retract":
        return {"id": bb.retract(_get(op, "node", "ref", "id", "text"), op.get("reason", ""))}
    if kind == "conflict":
        return {"id": bb.conflict(op["a"], op["b"], informant=op.get("source", "llm"))}
    if kind == "goal":
        return {"id": bb.goal(op["text"], requires=op.get("requires"), parent=op.get("parent"),
                              id=op.get("id"))}
    if kind == "abandon":
        bb.abandon(op["goal"], op.get("reason", ""))
        return {"id": op["goal"]}
    if kind == "question":
        return {"id": bb.question(op["text"], op.get("about"))}
    if kind == "act":
        a = brain.act(op["name"], op.get("args"), op.get("ok"), op.get("result"))
        return {"seq": a.seq}
    if kind == "remember":
        return {"id": brain.remember(op.get("kind", "semantic"), op["text"], op.get("tags"))["id"]}
    if kind == "use":
        if op["name"] not in brain.planner.ops:
            raise KeyError(f"no tool {op['name']!r}; TOOLS: {', '.join(brain.planner.ops) or '(none)'}")
        brain.bb.requests.append({"name": op["name"], "args": op.get("args") or {}, "why": op.get("why", "")})
        return {"id": op["name"], "queued": len(brain.bb.requests)}
    if kind == "operator":
        return {"id": brain.operator(op["name"], op.get("pre"), op.get("add"), op.get("delete"),
                                     float(op.get("cost", 1.0)), op.get("kind", "tool"), op.get("doc", ""))}
    raise ValueError(f"unknown op {kind!r}; known: {', '.join(OPS)}")


def apply(brain, ops: list[dict]) -> dict:
    results = []
    for op in ops:
        try:
            results.append({"op": op.get("op"), **apply_one(brain, op)})
        except Exception as e:                     # 한 연산이 틀려도 나머지는 들어간다
            results.append({"op": op.get("op"), "error": f"{type(e).__name__}: {e}"})
    rep = brain.maintain()
    brain.save()
    return {"results": results,
            "changed": {k: list(v) for k, v in rep["changed"].items() if not k.startswith("⊥")},
            "retracted": rep["retracted"], "unresolved": rep["unresolved"]}


_BLOCK = re.compile(r"```(?:subbrain|json)?\s*\n(.*?)```", re.S)


def parse(text: str) -> list[dict]:
    """LLM 출력에서 연산 목록을 꺼낸다. ```subbrain 블록 > ```json 블록 > 맨 JSON 순."""
    cands = _BLOCK.findall(text) or [text]
    ops: list[dict] = []
    for c in cands:
        c = c.strip()
        try:
            obj = json.loads(c)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "ops" in obj:
            obj = obj["ops"]
        if isinstance(obj, dict) and "op" in obj:
            obj = [obj]
        if isinstance(obj, list):
            ops += [o for o in obj if isinstance(o, dict) and "op" in o]
    return ops


SYSTEM = """너는 보조-뇌(subbrain)와 함께 생각한다. 보조-뇌는 네 믿음의 근거·모순·막힌 자리를 관리한다.
매 턴 보조-뇌가 지금 상태(GOALS/IMPASSE/NEXT/BELIEFS)를 준다. 노드 표기: id[F+]=사실·IN, [A+]=가정·IN, [C-]=주장·OUT.
너는 NEXT 에 적힌 작업을 하고, 결과를 반드시 아래 블록 하나로 낸다(산문은 블록 밖에 짧게만).

```subbrain
{"ops": [ ... ]}
```

연산:
""" + "\n".join(f"  {k}: {v}" for k, v in OPS.items()) + """

규칙: 주장은 because 로 근거 노드를 대야 믿어진다. 근거 없는 것은 assume 으로 확신도를 낮게 적어라.
모르면 지어내지 말고 question 을 남겨라."""
