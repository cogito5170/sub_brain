import json
import tempfile
import unittest
from pathlib import Path

from subbrain import AuxBrain, apply, parse


def brain():
    return AuxBrain(memory=False)


class BlackboardTest(unittest.TestCase):
    def test_claim_without_because_is_not_believed(self):
        b = brain()
        apply(b, [{"op": "claim", "id": "c", "text": "이 방법이 가장 적합하다"}])
        self.assertEqual(b.verify("c")["verdict"], "UNSUPPORTED")
        d = b.next()
        self.assertEqual(d["impasse"]["kind"], "UNSUPPORTED_CLAIM")

    def test_observe_false_marks_recheck_with_cause(self):
        b = brain()
        apply(b, [{"op": "fact", "id": "s", "text": "센서 정상"},
                  {"op": "fact", "id": "t", "text": "온도 정상"},
                  {"op": "claim", "id": "safe", "text": "안전", "because": ["s", "t"]},
                  {"op": "goal", "text": "안전 판정", "requires": ["safe"]}])
        self.assertEqual(b.verify("safe")["verdict"], "GROUNDED")
        r = apply(b, [{"op": "observe", "node": "t", "value": False}])
        self.assertEqual(r["changed"]["safe"], ["IN", "OUT"])
        self.assertEqual(b.bb.recheck["safe"]["cause"], ["t"])
        self.assertEqual(b.verify("t")["verdict"], "CONTRADICTED")

    def test_ddb_retracts_weakest_and_records_nogood(self):
        b = brain()
        r = apply(b, [{"op": "assume", "id": "h1", "text": "전원 끊김", "confidence": 0.3},
                      {"op": "assume", "id": "h2", "text": "전원 정상", "confidence": 0.9},
                      {"op": "conflict", "a": "h1", "b": "h2"}])
        self.assertEqual(r["retracted"][0]["assumption"], "h1")
        self.assertEqual(b.bb.tms.nogoods[0]["assumptions"], ["h1", "h2"])
        # 같은 조합을 다시 가정하면 거절된다
        r = apply(b, [{"op": "assume", "id": "h1", "text": "전원 끊김"}])
        self.assertIn("refused", r["results"][0])
        self.assertFalse(b.bb.tms.is_in("h1"))

    def test_premise_conflict_is_unresolved_not_silently_fixed(self):
        b = brain()
        r = apply(b, [{"op": "fact", "id": "a", "text": "a"}, {"op": "fact", "id": "b", "text": "b"},
                      {"op": "conflict", "a": "a", "b": "b"}])
        self.assertEqual(r["retracted"], [])
        self.assertTrue(r["unresolved"])
        d = b.next()
        self.assertEqual(d["impasse"]["kind"], "CONFLICT")
        self.assertEqual(d["op"]["op"], "REOBSERVE")

    def test_reference_by_text(self):
        b = brain()
        apply(b, [{"op": "fact", "text": "물은 100도에서 끓는다"},
                  {"op": "claim", "id": "k", "text": "끓었다", "because": ["물은 100도에서  끓는다 "]}])
        self.assertEqual(b.verify("k")["verdict"], "GROUNDED")


class ControlTest(unittest.TestCase):
    def test_unknown_creates_subgoal_soar_style(self):
        b = brain()
        apply(b, [{"op": "fact", "id": "ab", "text": "A -> B"},
                  {"op": "rule", "then": "X", "when": ["A", "ab"]},
                  {"op": "goal", "text": "X 를 증명하라", "requires": ["X"]}])
        d = b.next()
        self.assertEqual(d["impasse"]["kind"], "MISSING_PREMISE")
        self.assertEqual(d["impasse"]["nodes"], [b.bb.resolve("A")])
        sub = d["impasse"]["detail"]["subgoal"]
        self.assertEqual(b.bb.goals[sub].parent, "g1")
        self.assertTrue(d["call_llm"])
        self.assertIn("missing:", d["brief"])
        # 같은 막힘에 subgoal 을 또 만들지 않는다
        b.next()
        self.assertEqual(len(b.bb.goals), 2)
        apply(b, [{"op": "fact", "text": "A"}])
        d = b.next()
        self.assertEqual(d["status"], "DONE")
        self.assertFalse(d["call_llm"])

    def test_planner_tool_step_does_not_call_llm(self):
        b = brain()
        apply(b, [{"op": "operator", "name": "measure_temp", "add": ["temp_ok"], "kind": "tool"},
                  {"op": "goal", "text": "온도 확인", "requires": ["temp_ok"]}])
        d = b.next()
        self.assertFalse(d["call_llm"])
        self.assertEqual(d["op"]["op"], "measure_temp")

    def test_weak_support_asks_for_test(self):
        b = brain()
        apply(b, [{"op": "assume", "id": "a", "text": "a", "confidence": 0.3},
                  {"op": "claim", "id": "c", "text": "c", "because": ["a"]},
                  {"op": "goal", "text": "c?", "requires": ["c"]}])
        d = b.next()
        self.assertEqual(d["impasse"]["kind"], "WEAK_SUPPORT")
        self.assertEqual(d["op"], {**d["op"], "op": "TEST", "target": "a", "by": "llm"})
        # 그 가정을 가릴 도구가 등록돼 있으면 LLM 을 안 부르고 도구로 간다
        apply(b, [{"op": "operator", "name": "probe_a", "add": ["a"]}])
        d = b.next()
        self.assertEqual((d["op"]["op"], d["op"]["by"], d["call_llm"]), ("probe_a", "tool", False))

    def test_llm_requests_tool_with_use(self):
        b = brain()
        apply(b, [{"op": "operator", "name": "thermo", "add": ["t"], "doc": "온도계"},
                  {"op": "goal", "text": "판정", "requires": ["v"]}])
        d = b.next()
        self.assertIn("TOOLS thermo()->t 온도계", d["brief"])
        r = apply(b, [{"op": "use", "name": "thermo", "why": "실측"}, {"op": "use", "name": "nope"}])
        self.assertIn("error", r["results"][1])
        d = b.next()
        self.assertEqual((d["status"], d["op"]["op"], d["call_llm"]), ("ACT", "thermo", False))

    def test_repeated_action(self):
        b = brain()
        apply(b, [{"op": "goal", "text": "x", "requires": ["x"]}])
        for _ in range(2):
            apply(b, [{"op": "act", "name": "search", "args": {"q": "x"}, "ok": True}])
        d = b.next()
        self.assertEqual(d["impasse"]["kind"], "REPEATED_ACTION")

    def test_stall_stops_calling_llm(self):
        b = brain()
        apply(b, [{"op": "goal", "text": "x", "requires": ["x"]}])
        calls = [b.next()["call_llm"] for _ in range(6)]
        self.assertTrue(calls[0])
        self.assertIn(False, calls)          # 진전 없이 묻기를 계속하지 않는다
        d = b.next()
        self.assertEqual(d["impasse"]["kind"], "STALL")
        self.assertEqual(d["op"]["by"], "user")

    def test_abandoned_goal_recheck_does_not_block_done(self):
        b = brain()
        apply(b, [{"op": "assume", "id": "a", "text": "a"},
                  {"op": "claim", "id": "c", "text": "c", "because": ["a"]},
                  {"op": "goal", "id": "g1", "text": "c?", "requires": ["c"]}])
        apply(b, [{"op": "observe", "node": "a", "value": False},
                  {"op": "abandon", "goal": "g1"},
                  {"op": "fact", "id": "z", "text": "z"},
                  {"op": "goal", "text": "z?", "requires": ["z"]}])
        self.assertEqual(len(b.bb.goals), 2)
        d = b.next()
        self.assertEqual(d["status"], "DONE")
        self.assertIn("RECHECK", d["brief"])

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "s.json"
            b = AuxBrain(p, memory=Path(tmp) / "m.jsonl")
            apply(b, [{"op": "assume", "id": "h1", "text": "h1", "confidence": 0.2},
                      {"op": "assume", "id": "h2", "text": "h2", "confidence": 0.9},
                      {"op": "conflict", "a": "h1", "b": "h2"},
                      {"op": "goal", "text": "q", "requires": ["q"]}])
            d1 = b.next()
            b2 = AuxBrain(p, memory=Path(tmp) / "m.jsonl")
            self.assertEqual(b2.bb.tms.nogoods, b.bb.tms.nogoods)
            self.assertEqual(b2.verify("h1")["verdict"], "RETRACTED")
            self.assertEqual(b2.recall("h1 h2", ["failure"])[0]["kind"], "failure")
            self.assertEqual(d1["impasse"]["kind"], "UNKNOWN")


class ProtocolTest(unittest.TestCase):
    def test_parse_variants(self):
        t = 'blah\n```subbrain\n{"ops":[{"op":"fact","text":"a"}]}\n```\nmore'
        self.assertEqual(parse(t), [{"op": "fact", "text": "a"}])
        self.assertEqual(parse('[{"op":"fact","text":"a"}]'), [{"op": "fact", "text": "a"}])
        self.assertEqual(parse("no json here"), [])

    def test_bad_op_does_not_poison_batch(self):
        b = brain()
        r = apply(b, [{"op": "nope"}, {"op": "fact", "id": "a", "text": "a"}])
        self.assertIn("error", r["results"][0])
        self.assertTrue(b.bb.tms.is_in("a"))


if __name__ == "__main__":
    unittest.main()


class ExternalGroundingTest(unittest.TestCase):
    """예비 실행에서 난 구멍: Haiku 가 과제 문장을 fact 로 옮겨 적고 그것만으로 답을 세웠는데 GROUNDED 가 났다."""

    def test_self_reported_answer_is_not_accepted_when_tools_exist(self):
        b = brain()
        apply(b, [{"op": "operator", "name": "measure", "kind": "tool"},
                  {"op": "goal", "id": "G", "text": "답", "requires": ["answer"]},
                  {"op": "fact", "id": "F3", "text": "어제 보고서: D5 정상"},
                  {"op": "claim", "id": "answer", "text": "H6", "because": ["F3"]}])
        self.assertEqual(b.verify("answer")["verdict"], "SELF_REPORTED")
        d = b.next()
        self.assertEqual(d["impasse"]["kind"], "UNVERIFIED")
        self.assertNotEqual(d["status"], "DONE")
        apply(b, [{"op": "fact", "id": "m1", "text": "measure(D5) -> 낮음", "source": "tool"},
                  {"op": "claim", "id": "answer", "text": "D5", "because": ["m1", "F3"]}])
        self.assertEqual(b.verify("answer")["verdict"], "GROUNDED")
        self.assertEqual(b.next()["status"], "DONE")

    def test_without_tools_llm_facts_are_given(self):
        b = brain()
        apply(b, [{"op": "goal", "text": "답", "requires": ["answer"]},
                  {"op": "fact", "id": "F", "text": "전제"},
                  {"op": "claim", "id": "answer", "text": "결론", "because": ["F"]}])
        self.assertEqual(b.next()["status"], "DONE")

    def test_loop_strips_spoofed_tool_source(self):
        from subbrain.loop import run
        b = brain()
        apply(b, [{"op": "operator", "name": "measure", "kind": "tool"},
                  {"op": "goal", "text": "답", "requires": ["answer"]}])
        spoof = '{"ops":[{"op":"fact","id":"m","text":"측정 결과 낮음","source":"tool"},' \
                '{"op":"claim","id":"answer","text":"x","because":["m"]}]}'
        res = run(b, lambda s, u: spoof, max_steps=4, max_llm_calls=2)
        self.assertNotEqual(res["final"], "DONE")
        self.assertEqual(b.bb.tms.nodes["m"].meta["source"], "llm")
