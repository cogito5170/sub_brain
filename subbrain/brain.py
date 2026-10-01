"""AuxBrain -- 여섯 기관을 묶고, Hearsay-II 식 제어(control)로 '다음 한 걸음' 을 고른다.

    Blackboard  Memory  TMS  Impasse  Planner  Verifier
                         \\     |     /
                          CONTROL (next)
                              |
             {call_llm, op, focus, brief}   <- LLM 이 매 걸음 받는 것은 이 한 덩어리뿐

LLM 은 여러 지식원(knowledge source) 가운데 하나다. control 은 막힌 자리를 보고
**LLM 을 부를 가치가 있을 때만** call_llm=True 를 낸다. 도구 한 번으로 풀리는 것은
도구로, 다 섰으면 DONE 으로, 같은 자리를 진전 없이 다시 물으려 하면 STALL 로 끊는다.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from .blackboard import Blackboard
from .impasse import detect, Impasse
from .memory import Memory
from .planner import Planner, Operator
from .tms import IN, ASSUMPTION
from .verifier import verify, accept
from . import brief as _brief


def home() -> Path:
    return Path(os.environ.get("SUBBRAIN_HOME", Path.home() / ".subbrain"))


class AuxBrain:
    def __init__(self, session: str | Path | None = None, memory: str | Path | None | bool = None) -> None:
        self.bb = Blackboard()
        self.planner = Planner()
        self.path = Path(session) if session else None
        if memory is False:
            self.memory = Memory(None)
        else:
            self.memory = Memory(memory if memory else (home() / "memory.jsonl" if self.path else None))
        self.stall = 0
        self.last: dict | None = None
        self.stats = {"next": 0, "llm": 0, "tool": 0, "done": 0}
        self.rollbacks: list[dict] = []      # 지난 next 이후 DDB 가 내린 가정들
        if self.path and self.path.exists():
            self._load(json.loads(self.path.read_text(encoding="utf-8")))

    # --------------------------------------------------------------- persistence
    def to_dict(self) -> dict:
        return {"bb": self.bb.to_dict(), "planner": self.planner.to_dict(), "stall": self.stall,
                "last": self.last, "stats": self.stats,
                "rollbacks": self.rollbacks}

    def _load(self, d: dict) -> None:
        self.bb = Blackboard.from_dict(d.get("bb", {}))
        self.planner = Planner.from_dict(d.get("planner", []))
        self.stall, self.last = d.get("stall", 0), d.get("last")
        self.stats = d.get("stats", self.stats)
        self.rollbacks = d.get("rollbacks", [])

    def save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    # ------------------------------------------------------------- thin facade
    def __getattr__(self, name):          # fact / assume / claim / goal / observe / ... 는 칠판으로
        if name in ("fact", "assume", "claim", "rule", "observe", "retract", "conflict", "goal",
                    "abandon", "question", "act"):
            return getattr(self.bb, name)
        raise AttributeError(name)

    def operator(self, name: str, pre=None, add=None, delete=None, cost: float = 1.0,
                 kind: str = "tool", doc: str = "") -> str:
        r = self.bb.resolve
        if kind == "tool":
            self.bb.require_external = True
        self.planner.register(Operator(name, [r(x) for x in pre or []], [r(x) for x in add or []],
                                       [r(x) for x in delete or []], cost, kind, doc))
        return name

    def remember(self, kind: str, text: str, tags=None) -> dict:
        return self.memory.add(kind, text, tags)

    def recall(self, query: str, kinds=None, k: int = 3) -> list[dict]:
        return self.memory.recall(query, kinds, k)

    def why(self, ref: str) -> dict:
        nid = self.bb.resolve(ref, create=False)
        if nid is None:
            return {"node": ref, "error": "unknown node"}
        self.bb.maintain()
        return self.bb.tms.why(nid)

    def verify(self, ref: str) -> dict:
        self.bb.maintain()
        rep = verify(self.bb, ref)
        if "verdict" in rep and rep["verdict"] != "UNKNOWN_NODE":
            rep["accept"] = accept(rep)
        return rep

    def plan(self, targets: list[str]) -> dict:
        self.bb.maintain()
        goal = [self.bb.resolve(x) for x in targets]
        state = {i for i, n in self.bb.tms.nodes.items() if n.label == IN}
        p = self.planner.plan(goal, state)
        if p is None:
            return {"goal": goal, "plan": None}
        return {"goal": goal, "plan": [{"op": o.name, "kind": o.kind, "pre": o.pre, "add": o.add}
                                       for o in p]}

    def maintain(self) -> dict:
        rep = self.bb.maintain()
        self.rollbacks += rep["retracted"]
        for ng in rep["nogoods"]:
            texts = [self.bb.text(a) for a in ng["assumptions"]]
            self.memory.add("failure", "함께 가정하면 모순: " + " / ".join(texts),
                            tags=["nogood"], contradiction=self.bb.text(ng["contradiction"]))
        return rep

    def act(self, name: str, args=None, ok=None, result=None):
        a = self.bb.act(name, args, ok, result)
        if ok is False:
            self.memory.add("failure", f"{name}({json.dumps(args or {}, ensure_ascii=False)}) 실패: {result or ''}".strip(),
                            tags=["action", name])
        return a

    def close(self, outcome: str, lesson: str | None = None) -> dict:
        """에피소드를 닫고 기억에 남긴다."""
        goals = [g for g in self.bb.goals.values() if g.parent is None]
        txt = "; ".join(f"{g.text} -> {g.status}" for g in goals) or "(목표 없음)"
        it = self.memory.add("episodic", f"{txt} | 결과: {outcome} | 걸음 {self.stats['next']}, "
                                         f"LLM {self.stats['llm']}회")
        if lesson:
            self.memory.add("procedural", lesson)
        return it

    # ----------------------------------------------------------------- control
    def next(self, budget: int = 1600) -> dict:
        """지금 가장 가치 있는 사고 작업 하나를 고른다."""
        self.stats["next"] += 1
        rep = self.maintain()
        imps = detect(self.bb, self.planner, stall=self.stall)
        for imp in imps:                    # 막힌 자리 -> subgoal (SOAR)
            if imp.kind in ("UNKNOWN", "MISSING_PREMISE") and imp.goal:
                g = self.bb.goals[imp.goal]
                if g.requires == imp.nodes:     # 이미 이 노드를 세우려는 subgoal 이면 또 만들지 않는다
                    imp.detail["subgoal"] = g.id
                    continue
                imp.detail["subgoal"] = self.bb.goal(
                    f"근거 찾기: {self.bb.text(imp.nodes[0])}", requires=imp.nodes,
                    parent=imp.goal, origin=imp.key)
        top = imps[0] if imps else None
        req = self.bb.requests[0] if self.bb.requests else None

        if req and (top is None or top.priority < 80):   # LLM 이 청한 도구 -- 모순·되돌림보다는 뒤에
            decision = {"status": "ACT", "call_llm": False, "impasse": None,
                        "op": {"op": req["name"], "by": "tool", "args": req["args"], "requested": True,
                               "why": req.get("why") or "LLM 이 청한 도구"}}
        elif top is None:
            open_goals = [g for g in self.bb.goals.values() if g.status == "open"]
            if self.bb.goals and not open_goals:
                decision = {"status": "DONE", "call_llm": False, "op": None, "impasse": None}
                self.stats["done"] += 1
            elif not self.bb.goals:
                decision = {"status": "IDLE", "call_llm": True, "impasse": None,
                            "op": {"op": "GOAL", "by": "llm", "why": "목표가 없다 -- 무엇을 세울지 goal 로 적어라"}}
            else:
                decision = {"status": "THINK", "call_llm": True, "impasse": None,
                            "op": {"op": "DERIVE", "by": "llm", "why": "막힌 곳은 없다 -- 다음 추론을 적어라"}}
        else:
            ops = self._suggest(top)
            decision = {"status": "IMPASSE", "impasse": {"kind": top.kind, "key": top.key,
                                                         "nodes": top.nodes, "goal": top.goal,
                                                         "detail": top.detail},
                        "op": ops[0], "alternatives": ops[1:4], "call_llm": ops[0]["by"] == "llm",
                        "others": [i.key for i in imps[1:6]]}

        # 지난번 LLM 을 불렀는데 믿음이 하나도 안 바뀌었다 -- 또 부르는 것은 진전이 없다는 뜻이다
        sig = {"version": self.bb.tms.version, "key": (decision.get("impasse") or {}).get("key"),
               "llm": decision["call_llm"]}
        if self.last and self.last["llm"] and sig["version"] == self.last["version"] \
                and decision["status"] != "DONE":
            self.stall += 1
        elif sig["version"] != (self.last or {}).get("version"):
            self.stall = 0
        self.last = sig
        if decision["call_llm"]:
            self.stats["llm"] += 1
        elif decision["op"] and decision["op"].get("by") == "tool":
            self.stats["tool"] += 1
        decision["changed"] = {k: list(v) for k, v in rep["changed"].items()
                               if not k.startswith("⊥")}
        decision["retracted"], self.rollbacks = self.rollbacks, []
        decision["brief"] = _brief.render(self, decision, budget)
        self.save()
        return decision

    def _suggest(self, imp: Impasse) -> list[dict]:
        t, bb = self.bb.tms, self.bb
        ops: list[dict] = []
        k = imp.kind
        if k in ("UNKNOWN", "MISSING_PREMISE"):
            x = imp.nodes[0]
            state = {i for i, n in t.nodes.items() if n.label == IN}
            p = self.planner.plan([x], state) if self.planner.ops else None
            if p:
                o = p[0]
                ops.append({"op": o.name, "by": "llm" if o.kind == "cognitive" else "tool",
                            "target": x, "plan": [s.name for s in p],
                            "why": o.doc or f"{o.name} 가 {bb.text(x)} 쪽으로 간다"})
            n = t.nodes[x]
            if n.meta.get("retracted") or n.meta.get("blocked_by"):
                ops.append({"op": "FIND_INDEPENDENT_EVIDENCE", "by": "llm", "target": x,
                            "why": f"이 노드는 내려졌다({n.meta.get('retracted') or n.meta.get('blocked_by')}) -- 같은 가정을 다시 세우지 말고 다른 근거를 찾아라"})
            ops += [{"op": "RETRIEVE", "by": "llm", "target": x, "why": "알려진 사실·도구로 근거를 찾는다"},
                    {"op": "DERIVE", "by": "llm", "target": x, "why": "IN 인 노드에서 because 로 이끌어 낸다"},
                    {"op": "ASSUME", "by": "llm", "target": x, "why": "마지막 수단 -- 확신도를 낮게 적어라"}]
        elif k == "CONFLICT":
            d = imp.detail
            if d.get("goal_blocked"):
                ops.append({"op": "REPLAN", "by": "llm", "target": imp.nodes[0],
                            "why": f"목표가 요구하는 노드가 {d.get('verdict')} -- 다른 길로 목표를 세우거나 목표를 고쳐라"})
                if d.get("verdict") == "RETRACTED":
                    ops.append({"op": "TEST", "by": "llm", "target": imp.nodes[0],
                                "why": "모순 때문에 내려진 가정이다 -- 관측으로 가리면 nogood 의 다른 쪽이 내려간다"})
            elif d.get("assumptions"):
                cands = sorted(d["assumptions"], key=lambda a: float(t.nodes[a].meta.get("confidence", .5)))
                ops.append({"op": "RETRACT", "by": "llm", "candidates": cands,
                            "why": "이 가정들은 함께 설 수 없다 -- 하나를 내려라"})
            else:
                ops.append({"op": "REOBSERVE", "by": "llm", "candidates": d.get("premises", []),
                            "why": "관측끼리 부딪혔다 -- 다시 재거나 어느 관측이 틀렸는지 확인하라"})
                ops.append({"op": "ASK_USER", "by": "user", "candidates": d.get("premises", []),
                            "why": "관측끼리 부딪혔다"})
        elif k == "LOOP":
            ops.append({"op": "BREAK_LOOP", "by": "llm", "target": imp.nodes[0],
                        "why": "unless 고리 때문에 참 거짓이 안 정해진다 -- 한쪽을 가정이나 관측으로 정하라"})
        elif k == "RECHECK":
            x = imp.nodes[0]
            ops.append({"op": "REDERIVE", "by": "llm", "target": x,
                        "why": "지지를 잃었다 -- 남은 IN 노드로 다시 세우거나 retract 하라"})
        elif k == "REPEATED_ACTION":
            ops.append({"op": "CHANGE_APPROACH", "by": "llm", "avoid": imp.detail.get("action"),
                        "why": f"{imp.detail.get('action')} 를 진전 없이 {imp.detail.get('times')}번 했다"})
        elif k == "STALL":
            ops.append({"op": "ASK_USER", "by": "user",
                        "why": f"LLM 을 {imp.detail.get('turns')}번 더 불렀는데 믿음이 하나도 안 바뀌었다"})
            ops.append({"op": "ABANDON_SUBGOAL", "by": "llm", "why": "이 길을 버리고 부모 목표로 돌아간다"})
        elif k == "NO_PLAN":
            ops.append({"op": "PROPOSE_OPERATOR", "by": "llm", "target": imp.nodes,
                        "why": "등록된 연산자로는 여기에 닿을 길이 없다"})
        elif k == "WEAK_SUPPORT":
            for a in imp.nodes:
                ach = self.planner.achievers(a)
                if ach:
                    ops.append({"op": ach[0].name, "by": "llm" if ach[0].kind == "cognitive" else "tool",
                                "target": a, "why": f"약한 가정({t.nodes[a].meta.get('confidence')})을 {ach[0].name} 로 가린다"})
                else:
                    ops.append({"op": "TEST", "by": "llm", "target": a,
                                "why": f"목표가 약한 가정({t.nodes[a].meta.get('confidence')})에 기대고 있다 -- "
                                       "TOOLS 로 시험(use)하거나 observe 로 바꿔라"})
            ops.append({"op": "ACCEPT_CONDITIONAL", "by": "llm", "why": "시험할 길이 없으면 조건부로 받아들이고 그 조건을 말하라"})
        elif k == "UNVERIFIED":
            ops.append({"op": "TEST", "by": "llm", "target": imp.detail.get("self_reported") or imp.nodes,
                        "why": "답이 네가 적은 사실에만 기대고 도구 관측이 하나도 없다 -- 그 사실들 가운데 지금 상태에 "
                               "관한 것을 TOOLS 로 확인(use)하고, 그 관측을 because 에 넣어 다시 세워라"})
        elif k == "UNSUPPORTED_CLAIM":
            ops.append({"op": "SUPPORT", "by": "llm", "target": imp.nodes[0],
                        "why": "근거(because) 없이 말해졌다 -- 근거를 대거나 retract 하라"})
        return ops


def default_session(name: str = "default") -> Path:
    return home() / "sessions" / f"{name}.json"
