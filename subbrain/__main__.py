"""python3 -m subbrain <명령>

    mcp      [--session S]            MCP 서버(stdio)로 띄운다
    ops      [--session S] JSON|-     연산을 적는다 ({"ops":[...]} 또는 [...] 또는 LLM 출력 그대로)
    next     [--session S]            다음 사고 작업과 brief
    why      NODE / verify NODE / plan NODE.. / recall QUERY / state
    close    OUTCOME [--lesson L] [--reset]
    run      TASK [--model M]         claude -p 를 LLM 으로 고리를 돌린다
    demo                              센서-온도 예제(LLM 없이)
"""
from __future__ import annotations

import argparse
import json
import sys

from .brain import AuxBrain, default_session
from .protocol import apply, parse


def _brain(session: str) -> AuxBrain:
    return AuxBrain(session if session.endswith(".json") else default_session(session))


def _p(x) -> None:
    print(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=1))


def demo() -> None:
    b = AuxBrain(memory=False)

    def step(title, ops=None):
        print(f"\n=== {title}")
        if ops:
            r = apply(b, ops)
            if r["changed"]:
                print("changed:", ", ".join(f"{k} {o}->{n}" for k, (o, n) in r["changed"].items()))
        d = b.next()
        print(f"call_llm={str(d['call_llm']).lower()} status={d['status']}")
        print(d["brief"])

    step("LLM: 'A는 안전하다 -- 센서가 정상이고, (아마) 온도도 정상이니까'", [
        {"op": "fact", "id": "E1", "text": "센서가 정상이다"},
        {"op": "assume", "id": "E2", "text": "온도가 정상이다", "confidence": 0.7},
        {"op": "claim", "id": "B1", "text": "A는 안전하다", "because": ["E1", "E2"]},
        {"op": "claim", "id": "C", "text": "A를 계속 가동해도 된다", "because": ["B1"]},
        {"op": "goal", "text": "A를 계속 가동할지 판정", "requires": ["C"]},
    ])
    step("도구 관측: 온도 정상(E2) = FALSE  -> E2 를 딛고 선 B1, C 가 무너진다", [
        {"op": "observe", "node": "E2", "value": False}])
    print("\nverify(C) =", json.dumps(b.verify("C"), ensure_ascii=False))
    step("LLM: 근거 없이 '냉각수 펌프가 고장났다' 고 말한다", [
        {"op": "claim", "id": "P", "text": "냉각수 펌프가 고장났다"},
        {"op": "abandon", "goal": "g1", "reason": "온도 이상 관측 -- 가동 판정을 정지 판정으로 바꾼다"},
        {"op": "goal", "text": "A를 멈춰야 하는가", "requires": ["P"]}])
    step("LLM: 두 가정을 세운다 -- 그런데 둘은 함께 설 수 없다 (DDB)", [
        {"op": "assume", "id": "H1", "text": "펌프 전원이 끊겼다", "confidence": 0.4},
        {"op": "assume", "id": "H2", "text": "펌프 전원 계통이 정상이다", "confidence": 0.8},
        {"op": "claim", "id": "P", "text": "냉각수 펌프가 고장났다", "because": ["H1", "not:E2"]},
        {"op": "conflict", "a": "H1", "b": "H2"}])
    print("\nnogoods =", b.bb.tms.nogoods)
    step("도구: 전원 로그 확인 -> 펌프 전원이 끊겼다는 관측", [
        {"op": "observe", "node": "H1", "value": True}])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="subbrain")
    ap.add_argument("cmd")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--session", default="default")
    ap.add_argument("--lesson")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--budget", type=int, default=1600)
    a = ap.parse_args(argv)
    if a.cmd == "mcp":
        from .mcp_server import serve
        serve(a.session)
        return 0
    if a.cmd == "demo":
        demo()
        return 0
    b = _brain(a.session)
    if a.cmd == "ops":
        raw = sys.stdin.read() if not a.args or a.args == ["-"] else " ".join(a.args)
        ops = parse(raw)
        if not ops:
            print("연산을 못 찾았다", file=sys.stderr)
            return 2
        _p(apply(b, ops))
    elif a.cmd == "next":
        d = b.next(a.budget)
        print(f"call_llm={str(d['call_llm']).lower()} status={d['status']}")
        print(d["brief"])
    elif a.cmd == "why":
        _p(b.why(a.args[0]))
    elif a.cmd == "verify":
        _p(b.verify(a.args[0]))
    elif a.cmd == "plan":
        _p(b.plan(a.args))
    elif a.cmd == "recall":
        _p(b.recall(" ".join(a.args)))
    elif a.cmd == "state":
        from .mcp_server import state
        _p(state(b))
    elif a.cmd == "close":
        from .mcp_server import call
        _p(call(b, "subbrain_close", {"outcome": " ".join(a.args), "lesson": a.lesson, "reset": a.reset}))
    elif a.cmd == "run":
        from .loop import run, claude_cli
        task = " ".join(a.args)
        if task and not b.bb.goals:
            b.goal(task)
        res = run(b, claude_cli(a.model), max_steps=a.steps, task=task,
                  on_step=lambda r: print(json.dumps(r, ensure_ascii=False)[:400]))
        _p({"stats": res["stats"], "final": res["final"]})
    else:
        ap.print_help()
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
