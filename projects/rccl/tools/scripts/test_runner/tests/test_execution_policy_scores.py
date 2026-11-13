import unittest
from importlib.machinery import SourceFileLoader
from importlib.util import module_from_spec, spec_from_loader
from pathlib import Path


RCCL_ROOT = Path(__file__).resolve().parents[4]
SCRIPT_PATH = RCCL_ROOT / "tools" / "scripts" / "execution_policy_scores.py"

# main() is guarded by if __name__ == '__main__', so loading by path runs nothing.
_loader = SourceFileLoader("execution_policy_scores", str(SCRIPT_PATH))
_spec = spec_from_loader(_loader.name, _loader)
scores = module_from_spec(_spec)
_loader.exec_module(scores)


def rule(lo, hi, in_place="ANY_BOOL", coll="ALL_REDUCE", transport="SHM", arch="gfx1201",
         ranks="EQ(8)", profile="{8, RING, SIMPLE, AUTO_PATH, KEEP}"):
    return scores.Rule(f'{{{{COLL, "{arch}", {coll}, {{{lo}, {hi}}}, EQ(1), {ranks}, kAnyInt, '
                       f'{transport}, {in_place}}}, {profile}, UNMEASURED}}')


def ranges(pieces):
    return [(r.lo, r.hi, r.in_place) for _, r in pieces]


class EvalBytesTest(unittest.TestCase):
    def test_shift_and_size_max_expressions(self):
        self.assertEqual(16 << 20, scores.eval_bytes("16ULL << 20"))
        self.assertEqual((64 << 10) - 1, scores.eval_bytes("(64ULL << 10) - 1"))
        self.assertEqual(scores.SIZE_MAX, scores.eval_bytes("SIZE_MAX"))

    def test_builtins_are_unreachable(self):
        with self.assertRaises(NameError):
            scores.eval_bytes("__import__('os').getpid()")


class CanMatchSameInputTest(unittest.TestCase):
    def test_overlapping_bytes_and_compatible_placement_co_match(self):
        self.assertTrue(scores.can_match_same_input(rule(0, 99), rule(50, 199, "BOOL_TRUE")))

    def test_transport_does_not_separate_rules(self):
        self.assertTrue(scores.can_match_same_input(rule(0, 99), rule(0, 99, transport="IPC")))

    def test_arch_prefix_co_matches(self):
        self.assertTrue(scores.can_match_same_input(rule(0, 99, arch="gfx12"), rule(0, 99)))

    def test_disjoint_shapes_do_not_co_match(self):
        base = rule(0, 99, "BOOL_FALSE")
        for other in (rule(100, 199, "BOOL_FALSE"), rule(0, 99, "BOOL_TRUE"), rule(0, 99, coll="REDUCE"),
                      rule(0, 99, "BOOL_FALSE", ranks="EQ(4)"), rule(0, 99, "BOOL_FALSE", arch="gfx1100")):
            self.assertFalse(scores.can_match_same_input(base, other))


class MergeAdjacentTest(unittest.TestCase):
    def test_adjacent_identical_entries_merge(self):
        merged = scores.merge_adjacent([("rule", rule(0, 99)), ("rule", rule(100, 199))])
        self.assertEqual([(0, 199)], [(r.lo, r.hi) for _, r in merged])

    def test_gap_profile_or_text_keeps_entries_apart(self):
        other_profile = "{4, RING, SIMPLE, AUTO_PATH, KEEP}"
        for chunks in ([("rule", rule(0, 99)), ("rule", rule(101, 199))],
                       [("rule", rule(0, 99)), ("rule", rule(100, 199, profile=other_profile))],
                       [("rule", rule(0, 99)), ("text", "  // note"), ("rule", rule(100, 199))]):
            self.assertEqual(len(chunks), len(scores.merge_adjacent(chunks)))

    def test_size_max_upper_bound_never_merges(self):
        chunks = [("rule", rule(0, "SIZE_MAX")), ("rule", rule(0, 99))]
        self.assertEqual(2, len(scores.merge_adjacent(chunks)))


class AlignTest(unittest.TestCase):
    def test_overlapping_ranges_split_at_each_others_boundaries(self):
        pieces = scores.align([(0, rule(0, 199)), (1, rule(100, 299, transport="IPC"))])
        self.assertEqual([(0, 99, "ANY_BOOL"), (100, 199, "ANY_BOOL"), (100, 199, "ANY_BOOL"),
                          (200, 299, "ANY_BOOL")], ranges(pieces))
        self.assertEqual([0, 0, 1, 1], [slot for slot, _ in pieces])

    def test_any_bool_splits_against_a_placement_specific_rule(self):
        pieces = scores.align([(0, rule(0, 99, "BOOL_TRUE")), (1, rule(0, 99))])
        self.assertEqual([(0, 99, "BOOL_TRUE"), (0, 99, "BOOL_FALSE"), (0, 99, "BOOL_TRUE")], ranges(pieces))

    def test_aligned_result_has_no_partial_overlap(self):
        pieces = scores.align([(0, rule(0, 999)), (1, rule(100, 199, "BOOL_FALSE")),
                               (2, rule(150, "SIZE_MAX", transport="IPC"))])
        for _, a in pieces:
            for _, b in pieces:
                if a is not b and scores.can_match_same_input(a, b):
                    self.assertEqual((a.lo, a.hi), (b.lo, b.hi))
                    self.assertFalse((a.in_place == "ANY_BOOL") != (b.in_place == "ANY_BOOL"))


if __name__ == "__main__":
    unittest.main()
