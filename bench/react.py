"""ReAct 기준선 (Yao et al. 2022 꼴). Thought / Action / Observation 을 한 글뭉치로 쌓아 매 걸음 다시 보낸다.

claude -p 에는 stop sequence 가 없다. 그래서 모델이 Observation 을 지어내면 첫 Action 뒤를 잘라 버린다
(원 논문의 stop=["\\nObservation"] 과 같은 효과).
"""
from __future__ import annotations

import re

SYSTEM = """You solve a task by interleaving Thought, Action and Observation steps.
Available actions:
{tools}
  finish[answer] -- give the final answer (only the answer itself, short)
Reply with exactly one Thought line and one Action line, then stop:
Thought: <your reasoning>
Action: <action>[<argument>]
The Observation will be returned to you. Example:
Thought: I need to know where X is located.
Action: lookup[X]"""

ACT = re.compile(r"Action\s*:\s*(\w+)\s*\[(.*?)\]", re.S)


def run(task, llm, max_calls: int = 10) -> dict:
    tools = "\n".join(f"  {name}[...] -- {doc}" for name, (doc, _f) in task.tools.items())
    system = SYSTEM.format(tools=tools)
    transcript = ""
    answer, steps, bad = None, [], 0
    for _ in range(max_calls):
        out = llm(system, f"Task: {task.question}\n{transcript}")
        cut = out.split("Observation")[0]
        m = ACT.search(cut)
        thought = re.search(r"Thought\s*:\s*(.*)", cut)
        line = (f"Thought: {thought.group(1).strip()}\n" if thought else "")
        if not m:
            bad += 1
            transcript += line + "Observation: Invalid format. Use: Action: tool[argument]\n"
            steps.append({"bad": out[:200]})
            continue
        name, arg = m.group(1), m.group(2).strip()
        line += f"Action: {name}[{arg}]\n"
        if name == "finish":
            answer = arg
            steps.append({"finish": arg})
            break
        if name in task.tools:
            obs = task.tools[name][1](arg=arg)
        else:
            obs = f"Unknown action {name}. Available: {', '.join(task.tools)}, finish"
        transcript += line + f"Observation: {obs}\n"
        steps.append({"act": name, "arg": arg, "obs": obs[:200]})
    return {"answer": answer, "steps": steps, "bad_format": bad}
