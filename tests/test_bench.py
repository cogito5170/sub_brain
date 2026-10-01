"""비교 장치의 배선 점검 -- 가짜 LLM(정답을 아는 신탁)으로 두 길이 끝까지 돌고 채점되는지 본다.

이것은 성능 측정이 아니다. 신탁이 맞히는데 0점이 나오면 장치가 틀린 것이다."""
import json
import re
import unittest

from bench import react, sbagent
from bench.tasks import suite


class Oracle:
    def __init__(self, task):
        self.t, self.calls = task, 0

    def tool_arg(self):
        return self.t.answer if self.t.family == "diagnose" else self.t.question.split("를")[0].split("의")[0]


class ReactOracle(Oracle):
    def __call__(self, system, user):
        self.calls += 1
        if "Observation" not in user:
            return f"Thought: look\nAction: {list(self.t.tools)[0]}[{self.tool_arg()}]\nObservation: (hallucinated)"
        return f"Thought: done\nAction: finish[{self.t.answer}]"


class SbOracle(Oracle):
    def __call__(self, system, user):
        self.calls += 1
        obs = re.findall(r"(obs_\w+)\[F\+\]", user)
        if not obs:
            ops = [{"op": "use", "name": list(self.t.tools)[0], "args": {"arg": self.tool_arg()}}]
        else:
            ops = [{"op": "claim", "id": "answer", "text": self.t.answer, "because": obs[:1]}]
        return "```subbrain\n" + json.dumps({"ops": ops}, ensure_ascii=False) + "\n```"


class BenchWiringTest(unittest.TestCase):
    def test_both_paths_score_an_oracle(self):
        for t in suite(2):
            r = react.run(t, ReactOracle(t))
            self.assertTrue(t.check(r["answer"]), (t.id, r))
            llm = SbOracle(t)
            s = sbagent.run(t, llm)
            self.assertTrue(t.check(s["answer"]), (t.id, s))
            self.assertEqual(s["verdict"], "GROUNDED")
            self.assertEqual(llm.calls, 2)            # use 1회 + claim 1회, 도구 걸음은 LLM 을 안 부른다

    def test_checker_rejects_hedging_and_trap(self):
        t = suite(1)[0]
        self.assertFalse(t.check(None))
        self.assertFalse(t.check(f"{t.answer} 또는 {[d for d in t.distractors if d != t.answer][0]}"))
        self.assertTrue(t.check(f"답: {t.answer}"))

    def test_ungrounded_answer_does_not_count(self):
        t = suite(1)[1]

        def llm(system, user):
            return json.dumps({"ops": [{"op": "claim", "id": "answer", "text": t.answer}]})
        s = sbagent.run(t, llm, max_calls=3)
        self.assertIsNone(s["answer"])
        self.assertEqual(s["said"], t.answer)


if __name__ == "__main__":
    unittest.main()


class CheckerTest(unittest.TestCase):
    def test_short_code_counts_and_other_codes_do_not(self):
        from bench.tasks import diagnose
        t = diagnose(107)
        code = t.answer[-2:]
        self.assertTrue(t.check(code))
        self.assertTrue(t.check(t.answer + "가 고장"))
        other = [d for d in t.distractors if d != t.answer][0]
        self.assertFalse(t.check(f"{code} 또는 {other[-2:]}"))
        self.assertFalse(t.check(other))
