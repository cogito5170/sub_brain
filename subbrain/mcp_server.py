"""MCP 서버(stdio) -- 어떤 LLM 클라이언트에든 보조-뇌를 도구로 붙인다. 의존성 0.

    claude mcp add subbrain -- python3 -m subbrain mcp --session myproblem

메시지는 줄 단위 JSON-RPC 2.0 이다(MCP stdio transport).
"""
from __future__ import annotations

import json
import sys
import traceback

from . import __version__
from .brain import AuxBrain, default_session
from .protocol import OPS, apply

NODE = {"type": "string", "description": "노드 id 또는 글귀"}
TOOLS = [
    {"name": "subbrain_ops",
     "description": "보조-뇌 칠판에 연산을 한꺼번에 적는다. 주장은 because 로 근거를 대야 믿어진다. 연산:\n"
                    + "\n".join(f"- {k}: {v}" for k, v in OPS.items())
                    + "\n결과로 바뀐 믿음(changed)과 모순 때문에 되돌린 가정(retracted)을 준다.",
     "inputSchema": {"type": "object", "required": ["ops"], "properties": {
         "ops": {"type": "array", "items": {"type": "object", "required": ["op"],
                                            "properties": {"op": {"type": "string", "enum": list(OPS)}},
                                            "additionalProperties": True}}}}},
    {"name": "subbrain_next",
     "description": "지금 가장 가치 있는 다음 사고 작업 하나. 막힌 자리(IMPASSE)·되돌린 가정(ROLLBACK)·"
                    "다시 볼 주장(RECHECK)·다음 연산(NEXT)을 짧은 brief 로 준다. call_llm=false 면 "
                    "생각할 필요 없이 도구를 쓰거나 끝내면 된다.",
     "inputSchema": {"type": "object", "properties": {
         "budget": {"type": "integer", "description": "brief 글자 수 상한", "default": 1600},
         "full": {"type": "boolean", "description": "brief 말고 결정 전체(JSON)를 준다"}}}},
    {"name": "subbrain_why",
     "description": "이 노드를 왜 믿는가(IN) 또는 왜 못 믿는가(OUT) -- 근거 사슬",
     "inputSchema": {"type": "object", "required": ["node"], "properties": {"node": NODE}}},
    {"name": "subbrain_verify",
     "description": "주장을 바깥에서 검사: GROUNDED / CONDITIONAL(어느 가정에 기대나) / UNPROVEN(무엇이 빠졌나) / "
                    "UNSUPPORTED / CONTRADICTED / RETRACTED / UNDETERMINED",
     "inputSchema": {"type": "object", "required": ["node"], "properties": {"node": NODE}}},
    {"name": "subbrain_plan",
     "description": "등록된 STRIPS 연산자로 목표 노드들을 세우는 가장 싼 계획",
     "inputSchema": {"type": "object", "required": ["targets"],
                     "properties": {"targets": {"type": "array", "items": NODE}}}},
    {"name": "subbrain_recall",
     "description": "장기 기억 검색(세션을 넘어 남는다). kinds: episodic/semantic/procedural/failure",
     "inputSchema": {"type": "object", "required": ["query"], "properties": {
         "query": {"type": "string"}, "kinds": {"type": "array", "items": {"type": "string"}},
         "k": {"type": "integer", "default": 3}}}},
    {"name": "subbrain_state",
     "description": "칠판 전체 -- 목표·노드(라벨)·nogood·RECHECK",
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "subbrain_close",
     "description": "이 문제를 닫고 에피소드·교훈을 장기 기억에 남긴다. reset=true 면 칠판을 비운다",
     "inputSchema": {"type": "object", "required": ["outcome"], "properties": {
         "outcome": {"type": "string"}, "lesson": {"type": "string"}, "reset": {"type": "boolean"}}}},
]


def state(brain) -> dict:
    brain.maintain()
    t = brain.bb.tms
    return {"goals": [vars(g) for g in brain.bb.goals.values()],
            "nodes": [{"id": n.id, "text": n.text, "kind": n.kind, "label": n.label,
                       **({"confidence": n.meta["confidence"]} if "confidence" in n.meta else {}),
                       **({"retracted": n.meta["retracted"]} if n.meta.get("retracted") else {})}
                      for n in t.nodes.values() if not n.id.startswith("⊥")],
            "nogoods": t.nogoods, "recheck": brain.bb.recheck, "stats": brain.stats}


def call(brain, name: str, a: dict):
    if name == "subbrain_ops":
        return apply(brain, a.get("ops", []))
    if name == "subbrain_next":
        d = brain.next(int(a.get("budget", 1600)))
        if a.get("full"):
            return d
        return f"call_llm={str(d['call_llm']).lower()} status={d['status']}\n{d['brief']}"
    if name == "subbrain_why":
        return brain.why(a["node"])
    if name == "subbrain_verify":
        return brain.verify(a["node"])
    if name == "subbrain_plan":
        return brain.plan(a["targets"])
    if name == "subbrain_recall":
        return brain.recall(a["query"], a.get("kinds"), int(a.get("k", 3)))
    if name == "subbrain_state":
        return state(brain)
    if name == "subbrain_close":
        it = brain.close(a["outcome"], a.get("lesson"))
        if a.get("reset"):
            path, mem = brain.path, brain.memory
            brain.__init__(None, False)
            brain.path, brain.memory = path, mem
            brain.save()
        return {"remembered": it["id"], "reset": bool(a.get("reset"))}
    raise KeyError(f"unknown tool {name}")


def serve(session: str = "default", stdin=None, stdout=None) -> None:
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout
    path = session if session.endswith(".json") else default_session(session)
    brain = AuxBrain(path)

    def send(msg):
        stdout.write(json.dumps(msg, ensure_ascii=False) + "\n")
        stdout.flush()

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            send({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        mid, method, params = req.get("id"), req.get("method"), req.get("params") or {}
        if mid is None:                      # notification (initialized 등)
            continue
        try:
            if method == "initialize":
                res = {"protocolVersion": params.get("protocolVersion", "2025-06-18"),
                       "capabilities": {"tools": {}},
                       "serverInfo": {"name": "subbrain", "version": __version__}}
            elif method == "ping":
                res = {}
            elif method == "tools/list":
                res = {"tools": TOOLS}
            elif method == "tools/call":
                try:
                    out = call(brain, params["name"], params.get("arguments") or {})
                    text = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False, indent=1)
                    res = {"content": [{"type": "text", "text": text}], "isError": False}
                except Exception as e:
                    res = {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}], "isError": True}
            else:
                send({"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}})
                continue
            send({"jsonrpc": "2.0", "id": mid, "result": res})
        except Exception:
            send({"jsonrpc": "2.0", "id": mid,
                  "error": {"code": -32603, "message": traceback.format_exc(limit=3)}})
