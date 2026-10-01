"""LLM 과 보조-뇌를 한 고리로 돌린다.

    while not DONE:
        d = brain.next()                 # 막힌 자리와 다음 연산
        if d.call_llm:  ops = parse(llm(SYSTEM, d.brief))
        elif tool:      ops = tools[op](brain, d.op)
        else:           사람에게 넘긴다(ASK_USER) -- 멈춘다
        apply(brain, ops)

LLM 은 control 이 부를 가치가 있다고 할 때만 불린다. 그 횟수를 그대로 센다.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Callable

from .protocol import SYSTEM, apply, parse

LLM = Callable[[str, str], str]          # (system, user) -> text


def claude_cli(model: str | None = None, timeout: int = 300) -> LLM:
    """`claude -p` 를 LLM 으로 쓴다(Claude Code 가 깔려 있어야 한다)."""
    exe = shutil.which("claude")
    if not exe:
        raise RuntimeError("claude CLI 가 없다")

    def call(system: str, user: str) -> str:
        # 도구를 끄고 시스템 프롬프트를 갈아 끼운다 -- 이 고리에서 LLM 은 생각만 하고, 행동은 보조-뇌가 고른다
        cmd = [exe, "-p", "--system-prompt", system, "--tools", ""]
        if model:
            cmd += ["--model", model]
        r = subprocess.run(cmd, input=user, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            raise RuntimeError(f"claude -p rc={r.returncode}: {(r.stderr or r.stdout).strip()[:500]}")
        return r.stdout
    return call


def anthropic_api(model: str, max_tokens: int = 2000) -> LLM:
    """Anthropic SDK 로 부른다(ANTHROPIC_API_KEY 필요)."""
    import anthropic
    client = anthropic.Anthropic()

    def call(system: str, user: str) -> str:
        m = client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                   messages=[{"role": "user", "content": user}])
        return "".join(b.text for b in m.content if getattr(b, "type", "") == "text")
    return call


def run(brain, llm: LLM, tools: dict[str, Callable] | None = None, max_steps: int = 20,
        task: str | None = None, on_step: Callable[[dict], None] | None = None) -> dict:
    tools = tools or {}
    trace = []
    first = True
    for step in range(max_steps):
        d = brain.next()
        rec = {"step": step, "status": d["status"], "op": d.get("op"),
               "impasse": (d.get("impasse") or {}).get("kind")}
        if d["status"] == "DONE":
            trace.append(rec)
            break
        op = d["op"]
        if op and op.get("requested"):
            brain.bb.requests.pop(0)
        if op and op["by"] == "tool" and op["op"] not in tools:
            # 실행할 수 없는 도구 단계면 실행할 수 있는 다음 대안으로 내려간다
            alt = [a for a in d.get("alternatives", []) if a["by"] == "llm" or a["op"] in tools]
            if alt:
                op = alt[0]
                d["call_llm"] = op["by"] == "llm"
                rec["fallback"] = op["op"]
        if d["call_llm"]:
            msg = d["brief"]
            if first and task:
                msg = f"TASK: {task}\n\n{msg}"
            first = False
            text = llm(SYSTEM, msg)
            ops = parse(text)
            rec["llm_ops"] = len(ops)
            if not ops:
                brain.act("llm_no_ops", {"op": op["op"]}, ok=False, result=text[:200])
        elif op["by"] == "tool" and op["op"] in tools:
            ops = tools[op["op"]](brain, op) or []
            brain.act(op["op"], {k: op[k] for k in ("target", "candidates") if k in op},
                      ok=bool(ops), result=f"{len(ops)} ops")
        else:
            rec["handoff"] = op
            trace.append(rec)
            if on_step:
                on_step(rec)
            break
        rec["applied"] = apply(brain, ops)["results"]
        trace.append(rec)
        if on_step:
            on_step(rec)
    return {"trace": trace, "stats": dict(brain.stats), "final": trace[-1]["status"] if trace else None}
