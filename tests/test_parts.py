import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from subbrain import AuxBrain, apply
from subbrain.loop import run
from subbrain.memory import Memory
from subbrain.planner import Planner, Operator

ROOT = Path(__file__).resolve().parent.parent


class MemoryTest(unittest.TestCase):
    def test_korean_particles_still_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = Memory(Path(tmp) / "m.jsonl")
            m.add("failure", "온도센서를 믿고 가동 판정했다가 틀렸다")
            m.add("semantic", "고양이는 포유류다")
            hit = m.recall("온도센서가 이상하다")
            self.assertEqual(hit[0]["text"], "온도센서를 믿고 가동 판정했다가 틀렸다")
            self.assertEqual(len(Memory(Path(tmp) / "m.jsonl").items), 2)   # 디스크에 남았다
            self.assertEqual(m.recall("온도센서", kinds=["semantic"]), [])


class PlannerTest(unittest.TestCase):
    def test_astar_cheapest(self):
        p = Planner()
        p.register(Operator("expensive", [], ["goal"], cost=10))
        p.register(Operator("get_key", [], ["key"], cost=1))
        p.register(Operator("open", ["key"], ["goal"], cost=1))
        self.assertEqual([o.name for o in p.plan(["goal"], set())], ["get_key", "open"])
        self.assertIsNone(p.plan(["nowhere"], set()))
        self.assertEqual(p.plan(["key"], {"key"}), [])


class LoopTest(unittest.TestCase):
    """가짜 LLM 으로 고리를 돌린다 -- 실제 모델 호출 없음."""

    def test_scripted_llm_and_tool(self):
        b = AuxBrain(memory=False)
        apply(b, [{"op": "fact", "id": "ab", "text": "A이면 B"},
                  {"op": "fact", "id": "bc", "text": "B이면 C"},
                  {"op": "rule", "then": "B", "when": ["A", "ab"]},
                  {"op": "rule", "then": "C", "when": ["B", "bc"]},
                  {"op": "operator", "name": "lookup_A", "add": ["A"], "kind": "tool"},
                  {"op": "goal", "text": "C 를 보여라", "requires": ["C"]}])
        prompts = []

        def llm(system, user):
            prompts.append(user)
            return '```subbrain\n{"ops":[{"op":"fact","text":"A"}]}\n```'

        tools = {"lookup_A": lambda brain, op: [{"op": "observe", "node": "A", "value": True, "source": "tool"}]}
        res = run(b, llm, tools)
        self.assertEqual(res["final"], "DONE")
        self.assertEqual(prompts, [])                 # 도구로 풀리니 LLM 을 안 불렀다
        self.assertEqual(b.verify("C")["verdict"], "GROUNDED")

    def test_llm_called_when_no_tool(self):
        b = AuxBrain(memory=False)
        apply(b, [{"op": "goal", "text": "X?", "requires": ["X"]}])
        seen = []

        def llm(system, user):
            seen.append(user)
            self.assertIn("```subbrain", system)
            return '{"ops":[{"op":"fact","id":"X","text":"X"}]}'

        res = run(b, llm, task="X 를 보여라")
        self.assertEqual(res["final"], "DONE")
        self.assertEqual(len(seen), 1)
        self.assertIn("IMPASSE UNKNOWN", seen[0])

    def test_useless_llm_is_cut_off(self):
        b = AuxBrain(memory=False)
        apply(b, [{"op": "goal", "text": "X?", "requires": ["X"]}])
        n = []
        res = run(b, lambda s, u: n.append(1) or "모르겠습니다", max_steps=30)
        self.assertLess(len(n), 30)
        self.assertNotEqual(res["final"], "DONE")


class MCPTest(unittest.TestCase):
    def test_stdio_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = dict(os.environ, SUBBRAIN_HOME=tmp, PYTHONPATH=str(ROOT))
            reqs = [
                {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                 "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                            "clientInfo": {"name": "t", "version": "0"}}},
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {
                    "name": "subbrain_ops", "arguments": {"ops": [
                        {"op": "fact", "id": "E1", "text": "센서 정상"},
                        {"op": "assume", "id": "E2", "text": "온도 정상", "confidence": 0.9},
                        {"op": "claim", "id": "S", "text": "안전", "because": ["E1", "E2"]},
                        {"op": "goal", "text": "안전?", "requires": ["S"]}]}}},
                {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {
                    "name": "subbrain_ops", "arguments": {"ops": [{"op": "observe", "node": "E2", "value": False}]}}},
                {"jsonrpc": "2.0", "id": 5, "method": "tools/call", "params": {"name": "subbrain_next", "arguments": {}}},
                {"jsonrpc": "2.0", "id": 6, "method": "tools/call", "params": {"name": "subbrain_verify", "arguments": {"node": "S"}}},
                {"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "nope", "arguments": {}}},
            ]
            p = subprocess.run([sys.executable, "-m", "subbrain", "mcp", "--session", "t"], cwd=tmp, env=env,
                               input="".join(json.dumps(r, ensure_ascii=False) + "\n" for r in reqs),
                               capture_output=True, text=True, timeout=30)
            self.assertEqual(p.returncode, 0, p.stderr)
            out = {m["id"]: m for m in map(json.loads, p.stdout.splitlines())}
            self.assertEqual(set(out), {1, 2, 3, 4, 5, 6, 7})          # 알림에는 답하지 않는다
            self.assertEqual(out[1]["result"]["serverInfo"]["name"], "subbrain")
            self.assertEqual(len(out[2]["result"]["tools"]), 8)
            r4 = json.loads(out[4]["result"]["content"][0]["text"])
            self.assertEqual(r4["changed"]["S"], ["IN", "OUT"])
            self.assertIn("RECHECK", out[5]["result"]["content"][0]["text"])
            self.assertEqual(json.loads(out[6]["result"]["content"][0]["text"])["verdict"], "UNPROVEN")
            self.assertTrue(out[7]["result"]["isError"])
            self.assertTrue((Path(tmp) / "sessions" / "t.json").exists())   # 세션이 남는다


if __name__ == "__main__":
    unittest.main()
