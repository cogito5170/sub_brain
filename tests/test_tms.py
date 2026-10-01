import unittest

from subbrain.tms import TMS, IN, OUT, UNDET, PREMISE, ASSUMPTION


class TMSTest(unittest.TestCase):
    def setUp(self):
        self.t = TMS()

    def prem(self, i):
        self.t.node(i, kind=PREMISE).enabled = True

    def test_chain_and_retraction(self):
        self.prem("a")
        self.t.node("b", kind=ASSUMPTION).enabled = True
        self.t.justify("c", ["a", "b"])
        self.t.justify("d", ["c"])
        self.t.relabel()
        self.assertEqual(self.t.nodes["d"].label, IN)
        self.assertEqual(self.t.assumptions_of("d"), ["b"])
        self.assertEqual(self.t.premises_of("d"), ["a"])
        self.t.disable("b")
        ch = self.t.relabel()
        self.assertEqual(ch["d"], (IN, OUT))
        self.assertEqual(self.t.missing_leaves("d"), ["b"])

    def test_no_justification_stays_out(self):
        self.t.node("x")
        self.t.relabel()
        self.assertEqual(self.t.nodes["x"].label, OUT)
        self.assertEqual(self.t.why("x")["because"], "no-justification")

    def test_outlist_default(self):
        # 새는 난다, 펭귄이 아니라면
        self.prem("bird")
        self.t.justify("flies", ["bird"], ["penguin"])
        self.t.relabel()
        self.assertEqual(self.t.nodes["flies"].label, IN)
        self.prem("penguin")
        self.t.relabel()
        self.assertEqual(self.t.nodes["flies"].label, OUT)

    def test_odd_loop_is_undetermined_not_silently_out(self):
        self.t.justify("p", [], ["p"])          # p unless p
        self.t.relabel()
        self.assertEqual(self.t.nodes["p"].label, UNDET)

    def test_even_loop_undetermined(self):
        self.t.justify("p", [], ["q"])
        self.t.justify("q", [], ["p"])
        self.t.relabel()
        self.assertEqual({self.t.nodes["p"].label, self.t.nodes["q"].label}, {UNDET})

    def test_missing_leaves_picks_cheapest_justification(self):
        self.prem("a")
        self.t.justify("g", ["x", "y", "z"])
        self.t.justify("g", ["a", "w"])
        self.t.relabel()
        self.assertEqual(self.t.missing_leaves("g"), ["w"])

    def test_roundtrip(self):
        self.prem("a")
        self.t.justify("b", ["a"])
        self.t.relabel()
        t2 = TMS.from_dict(self.t.to_dict())
        self.assertEqual(t2.nodes["b"].label, IN)
        t2.justify("c", ["b"])
        self.assertNotIn(t2.justify("c", ["b"]).id, [j for j in self.t.justs])


if __name__ == "__main__":
    unittest.main()
