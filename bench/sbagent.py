"""보조-뇌 쪽 -- 같은 과제·같은 도구·같은 LLM 호출 상한. 기억은 과제마다 비운다(과제 사이 누설 없음)."""
from __future__ import annotations

from subbrain import AuxBrain, apply
from subbrain.loop import run as sb_run, fact_tool


def run(task, llm, max_calls: int = 10) -> dict:
    b = AuxBrain(memory=False)
    ops = [{"op": "operator", "name": name, "kind": "tool",
            "doc": f"{doc}. use 로 부른다: {{\"op\":\"use\",\"name\":\"{name}\",\"args\":{{\"arg\":\"...\"}}}}"}
           for name, (doc, _f) in task.tools.items()]
    ops.append({"op": "goal", "id": "G", "text": "과제의 최종 답을 id 가 answer 인 claim 으로, 관측 근거(because)와 함께 세운다",
                "requires": ["answer"]})
    apply(b, ops)
    tools = {name: fact_tool(name, fn) for name, (_d, fn) in task.tools.items()}
    res = sb_run(b, llm, tools, max_steps=40, task=task.question, max_llm_calls=max_calls)
    n = b.bb.tms.nodes.get("answer")
    # 보조-뇌가 받아들인 답(목표 G 가 achieved)만 답으로 친다. 받아들이지 않은 말은 said 로 따로 센다
    answer = n.text if n and b.bb.goals["G"].status == "achieved" else None
    said = n.text if n and n.text != "answer" else None        # 근거 없이라도 말은 했나
    v = b.verify("answer") if n else {}
    return {"answer": answer, "said": said, "verdict": v.get("verdict"), "final": res["final"],
            "handoff": (res["trace"][-1].get("handoff") or {}).get("op") if res["trace"] else None,
            "tool_steps": res["stats"]["tool"], "impasses": [t.get("impasse") for t in res["trace"]]}
