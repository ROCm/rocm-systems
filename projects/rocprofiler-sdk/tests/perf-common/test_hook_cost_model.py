# MIT License
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

"""Tests for the queue-hook cost and semantics models.

These run everywhere: no GPU, no HSA, no rocprofiler runtime. They guard the decisions
that the GPU-only tests cannot reach on a machine without hardware, and they fail loudly
if someone changes a documented semantic (empty agent set means "all agents", the replay
override is AND-ed rather than OR-ed, PC sampling is override-blind) without intending to.
"""

import unittest

import hook_cost_model as m


# ---------------------------------------------------------------------------
# Shared: the per-dispatch context gate (all four PRs touch this path)
# ---------------------------------------------------------------------------
class ContextGateCost(unittest.TestCase):
    def test_idle_gate_is_bounded(self):
        # No active contexts: get_active_contexts() early-outs on one atomic load.
        self.assertLessEqual(m.gate_ceiling_ns(0, walks=1), m.GATE_IDLE_CEILING_NS)

    def test_gate_cost_grows_linearly_in_walks(self):
        one = m.gate_ceiling_ns(4, walks=1)
        three = m.gate_ceiling_ns(4, walks=3)
        self.assertAlmostEqual(three, 3.0 * one)

    def test_gate_cost_grows_linearly_in_active_contexts(self):
        base = m.gate_ceiling_ns(0)
        step = m.gate_ceiling_ns(1) - base
        # Strictly positive, or the ceiling would stop tracking the walk it bounds.
        self.assertGreater(step, 0.0)
        self.assertAlmostEqual(step, m.GATE_PER_CONTEXT_CEILING_NS)
        self.assertAlmostEqual(m.gate_ceiling_ns(5) - m.gate_ceiling_ns(4), step)
        self.assertAlmostEqual(m.gate_ceiling_ns(4), base + 4 * step)

    def test_no_heap_allocation_at_or_below_inline_capacity(self):
        # This is what keeps the interceptor gate allocation-free for real tools.
        for n in range(0, m.CONTEXT_ARRAY_INLINE_CAPACITY + 1):
            self.assertFalse(m.gate_allocates(n), f"{n} contexts should stay inline")

    def test_heap_allocation_cliff_above_inline_capacity(self):
        # Crossing 6 active contexts makes every walk allocate, on every dispatch.
        self.assertTrue(m.gate_allocates(m.CONTEXT_ARRAY_INLINE_CAPACITY + 1))

    def test_inline_capacity_matches_small_vector_derivation(self):
        # (64-byte preferred object size - 16-byte header) / 8-byte pointer.
        self.assertEqual(m.CONTEXT_ARRAY_INLINE_CAPACITY, (64 - 16) // 8)

    def test_rejects_nonsense_inputs(self):
        with self.assertRaises(ValueError):
            m.gate_ceiling_ns(0, walks=0)
        with self.assertRaises(ValueError):
            m.gate_ceiling_ns(-1)


# ---------------------------------------------------------------------------
# Kernel replay window cost
# ---------------------------------------------------------------------------
class ReplayWindowCost(unittest.TestCase):
    def test_single_pass_pays_one_snapshot_and_no_restore(self):
        self.assertAlmostEqual(m.replay_window_seconds(9.0, passes=1, bw_gbps=9.0), 1.0)

    def test_final_pass_skips_restore(self):
        # N passes => 1 snap + (N-1) restores, not N restores.
        two = m.replay_window_seconds(9.0, passes=2, bw_gbps=9.0)
        self.assertAlmostEqual(two, 2.0)

    def test_cost_is_linear_in_passes(self):
        a = m.replay_window_seconds(4.0, passes=2)
        b = m.replay_window_seconds(4.0, passes=3)
        c = m.replay_window_seconds(4.0, passes=4)
        self.assertAlmostEqual(b - a, c - b)

    def test_cost_is_linear_in_resident_memory(self):
        small = m.replay_window_seconds(1.0, passes=4)
        big = m.replay_window_seconds(10.0, passes=4)
        self.assertAlmostEqual(big, 10.0 * small)

    def test_window_is_bandwidth_bound_not_kernel_bound(self):
        # A 50 us kernel against a 16 GB working set: the copies dominate by orders of
        # magnitude, which is why replay does not scale to large device footprints.
        without = m.replay_window_seconds(16.0, passes=4)
        with_kernel = m.replay_window_seconds(16.0, passes=4, kernel_seconds=50e-6)
        self.assertLess(with_kernel - without, 0.001 * without)

    def test_snapshot_is_a_full_copy_not_a_diff(self):
        # Documents the current limitation: host RAM scales 1:1 with tracked device
        # memory. If dirty-page diffing ever lands, this test should change with it.
        self.assertAlmostEqual(m.replay_host_ram_gb(80.0), 80.0)

    def test_rejects_nonsense_inputs(self):
        with self.assertRaises(ValueError):
            m.replay_window_seconds(1.0, passes=0)
        with self.assertRaises(ValueError):
            m.replay_window_seconds(1.0, passes=1, bw_gbps=0.0)


# ---------------------------------------------------------------------------
# #11970 counter collection
# ---------------------------------------------------------------------------
class CounterCollectionSemantics(unittest.TestCase):
    def test_empty_agent_set_collects_on_every_agent(self):
        # Unrestricted contexts must behave exactly as before per-agent scoping existed.
        self.assertTrue(m.collects_on(None, "gpu0"))
        self.assertTrue(m.collects_on(set(), "gpu1"))

    def test_scoped_context_ignores_other_agents(self):
        self.assertTrue(m.collects_on({"gpu1"}, "gpu1"))
        self.assertFalse(m.collects_on({"gpu1"}, "gpu0"))

    def test_disjoint_contexts_do_not_conflict(self):
        # This is the relaxation #11970 makes to start_context().
        self.assertFalse(m.contexts_conflict({"gpu0"}, {"gpu1"}))

    def test_overlapping_contexts_conflict(self):
        self.assertTrue(m.contexts_conflict({"gpu0", "gpu1"}, {"gpu1"}))

    def test_unrestricted_context_conflicts_with_everything(self):
        # An empty set claims every agent, so it must still be rejected.
        self.assertTrue(m.contexts_conflict(set(), {"gpu0"}))
        self.assertTrue(m.contexts_conflict({"gpu0"}, set()))
        self.assertTrue(m.contexts_conflict(set(), set()))

    def test_per_agent_scoping_preserves_machine_concurrency(self):
        # The headline multi-GPU benefit: 8 GPUs, a context scoped to one of them.
        self.assertAlmostEqual(m.serialization_concurrency_fraction(8, 1), 7 / 8)

    def test_unrestricted_context_serializes_whole_machine(self):
        self.assertAlmostEqual(m.serialization_concurrency_fraction(8, 0), 0.0)

    def test_replay_override_can_disable_for_one_pass(self):
        self.assertFalse(m.effective_collection("counters", True, False))

    def test_replay_override_cannot_promote_a_stopped_context(self):
        # Local start only undoes a local stop; its callback thread may already be gone.
        self.assertFalse(m.effective_collection("counters", False, True))

    def test_no_override_leaves_global_state_untouched(self):
        self.assertTrue(m.effective_collection("counters", True, None))
        self.assertFalse(m.effective_collection("counters", False, None))


# ---------------------------------------------------------------------------
# #11968 SPM
# ---------------------------------------------------------------------------
class SpmSemantics(unittest.TestCase):
    def test_override_semantics_mirror_counters(self):
        for g in (True, False):
            for ov in (True, False, None):
                self.assertEqual(
                    m.effective_collection("spm", g, ov),
                    m.effective_collection("counters", g, ov),
                    f"spm diverged from counters at globally_enabled={g} override={ov}",
                )

    def test_disjoint_agent_contexts_may_run_concurrently(self):
        self.assertFalse(m.contexts_conflict({"gpu2"}, {"gpu3"}))

    def test_unrestricted_spm_context_claims_every_agent(self):
        self.assertTrue(m.collects_on(set(), "gpu7"))

    def test_scoped_spm_leaves_other_agents_concurrent(self):
        self.assertAlmostEqual(m.serialization_concurrency_fraction(4, 1), 0.75)

    def test_replay_override_disables_spm_for_one_pass(self):
        self.assertFalse(m.effective_collection("spm", True, False))


# ---------------------------------------------------------------------------
# #11967 thread trace
# ---------------------------------------------------------------------------
class ThreadTraceSemantics(unittest.TestCase):
    def test_override_semantics_mirror_counters(self):
        for g in (True, False):
            for ov in (True, False, None):
                self.assertEqual(
                    m.effective_collection("thread_trace", g, ov),
                    m.effective_collection("counters", g, ov),
                )

    def test_forced_off_pass_does_not_trace(self):
        self.assertFalse(m.effective_collection("thread_trace", True, False))

    def test_disjoint_agent_contexts_do_not_conflict(self):
        self.assertFalse(m.contexts_conflict({"gpu0"}, {"gpu3"}))


# ---------------------------------------------------------------------------
# #11969 PC sampling -- documents a KNOWN GAP, not desired behaviour
# ---------------------------------------------------------------------------
class PcSamplingSemantics(unittest.TestCase):
    def test_pc_sampling_is_override_blind(self):
        # PC sampling is agent-wide and does not consult local_context_override().
        # If a consumer is ever wired up, this test should be updated deliberately.
        self.assertTrue(m.effective_collection("pc_sampling", True, False))

    def test_pc_sampling_still_respects_global_stop(self):
        self.assertFalse(m.effective_collection("pc_sampling", False, False))

    def test_pc_sampling_is_not_listed_as_override_aware(self):
        self.assertNotIn("pc_sampling", m.OVERRIDE_AWARE_SERVICES)
        self.assertIn("pc_sampling", m.OVERRIDE_BLIND_SERVICES)


# ---------------------------------------------------------------------------
# The stated goal: one service per replay pass
# ---------------------------------------------------------------------------
class PerPassServiceIsolation(unittest.TestCase):
    def _schedule(self, wanted, services):
        """Overrides a correct tool records at PASS PHASE_ENTER: name every service."""
        return {s: (s == wanted) for s in services}

    def test_exactly_one_override_aware_service_collects_per_pass(self):
        services = m.OVERRIDE_AWARE_SERVICES
        for wanted in services:
            ov = self._schedule(wanted, services)
            live = [s for s in services if m.effective_collection(s, True, ov[s])]
            self.assertEqual(live, [wanted])

    def test_isolation_holds_when_some_services_are_globally_stopped(self):
        services = m.OVERRIDE_AWARE_SERVICES
        ov = self._schedule("counters", services)
        live = [
            s
            for s in services
            if m.effective_collection(s, s != "spm", ov[s])  # spm never started
        ]
        self.assertEqual(live, ["counters"])

    def test_pc_sampling_breaks_strong_isolation(self):
        # With PC sampling started, more than one service collects in every pass. This
        # is the formal gap: replay reserves ONE dispatch id for all passes, so an
        # N-pass replay attributes N times the PC samples to a single dispatch.
        services = list(m.OVERRIDE_AWARE_SERVICES) + ["pc_sampling"]
        ov = self._schedule("counters", services)
        live = [s for s in services if m.effective_collection(s, True, ov[s])]
        self.assertEqual(sorted(live), ["counters", "pc_sampling"])

    def test_strong_isolation_holds_when_pc_sampling_is_not_started(self):
        services = list(m.OVERRIDE_AWARE_SERVICES) + ["pc_sampling"]
        ov = self._schedule("counters", services)
        live = [
            s
            for s in services
            if m.effective_collection(s, s != "pc_sampling", ov[s])
        ]
        self.assertEqual(live, ["counters"])


# ---------------------------------------------------------------------------
# Serialization refcount  (shared by all four services)
# ---------------------------------------------------------------------------
class SerializationRefcount(unittest.TestCase):
    def test_zero_counts_means_disabled(self):
        self.assertFalse(m.serialization_enabled(0, 0))

    def test_global_claim_enables_every_agent(self):
        # A service calling the no-argument overload bumps `all`, which covers agents
        # that no per-agent claim names.
        self.assertTrue(m.serialization_enabled(1, 0))

    def test_per_agent_claim_enables_only_that_agent(self):
        self.assertTrue(m.serialization_enabled(0, 1))
        self.assertFalse(m.serialization_enabled(0, 0))

    def test_global_and_per_agent_claims_are_additive(self):
        # Releasing one must not disable an agent the other still holds.
        self.assertTrue(m.serialization_enabled(1, 1))
        self.assertTrue(m.serialization_enabled(0, 1))
        self.assertTrue(m.serialization_enabled(1, 0))

    def test_concurrency_fraction_rejects_out_of_range_scopes(self):
        with self.assertRaises(ValueError):
            m.serialization_concurrency_fraction(4, 5)
        with self.assertRaises(ValueError):
            m.serialization_concurrency_fraction(0, 0)


if __name__ == "__main__":
    unittest.main()
