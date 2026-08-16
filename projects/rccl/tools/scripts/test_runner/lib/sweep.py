#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE.txt for license information
"""
Pure sweep enumeration and fork eligibility for the init-pipeline (plan 4.1/4.1a).

``enumerate_sweep`` is a side-effect-free mirror of the C++ ``RunSimpleSweep``
outer loops (``TestBed.cpp``): it turns a fork test's sweep selectors into the
exact list of ``(num_gpus, multi_process, ranks_per_gpu)`` points, in the same
nested-loop order. No HIP, no fork, no communicator, no dependence on runtime
failure -- so the runner can use ONE enumerator for both ``--dry-run`` manifest
output and real sub-entry expansion, and the Python/C++ sweep shapes cannot drift.

Option B pins each fork entry to a single child generation by expanding the sweep
into per-point sub-entries and pinning the three outer dimensions
(``UT_MIN_GPUS``/``UT_MAX_GPUS``/``UT_POW2_GPUS``, ``UT_PROCESS_MASK``,
``UT_RANKS_PER_GPU``). Each resulting sub-entry runs exactly one ``InitComms`` =
one generation = one READY/GO.

The eligibility inventory declares, per suite/test, a ``warmup_profile`` and (for
``fork_coll``) the expansion selectors. Unknown/unlisted families default to the
serial path -- routing is decided here, before launch, never from a runtime
signal (the C++ generation==1 assertion is only defense-in-depth).
"""

from dataclasses import dataclass, field

# warmup_profile values (must match the C++/config vocabulary).
PROFILE_FORK = "fork_coll"
PROFILE_MPI = "mpi_coll"
PROFILE_NETIB = "netib_plugin"
PROFILE_NONE = "none"
PIPELINE_PROFILES = frozenset({PROFILE_FORK, PROFILE_MPI, PROFILE_NETIB})

# UT_PROCESS_MASK bits (match EnvVars: UT_SINGLE_PROCESS=1, UT_MULTI_PROCESS=2).
PROCESS_SINGLE = 1
PROCESS_MULTI = 2


def is_pipeline_profile(profile):
    """True if a warmup_profile routes to the init-pipeline (vs the serial path)."""
    return profile in PIPELINE_PROFILES


@dataclass(frozen=True)
class SweepConfig:
    """One pinned sweep point == one child generation."""
    num_gpus: int
    multi_process: bool
    ranks_per_gpu: int

    def env(self):
        """Environment overrides that pin RunSimpleSweep to exactly this point."""
        return {
            "UT_MIN_GPUS": str(self.num_gpus),
            "UT_MAX_GPUS": str(self.num_gpus),
            "UT_POW2_GPUS": "0",
            "UT_PROCESS_MASK": str(PROCESS_MULTI if self.multi_process else PROCESS_SINGLE),
            "UT_RANKS_PER_GPU": str(self.ranks_per_gpu),
        }

    def suffix(self):
        """Stable, filesystem-safe config suffix for sub-entry names/logs."""
        return f"g{self.num_gpus}_{'mp' if self.multi_process else 'sp'}_r{self.ranks_per_gpu}"


def enumerate_sweep(num_gpus_list, *, process_mask=PROCESS_SINGLE | PROCESS_MULTI,
                    max_ranks_per_gpu=1, ranks_per_gpu=0, only_pow2=False):
    """Expand a fork sweep's outer loops into individual single-generation points.

    Order matches C++ RunSimpleSweep exactly: num_gpus (outer) -> multi_process
    (single before multi) -> ranks_per_gpu (inner). This is what makes
    ``--fork-sweep-policy=legacy`` able to reproduce the original order.

    Args:
        num_gpus_list: GPU counts to sweep (already resolved, e.g. [8] or [1..8]).
        process_mask: UT_PROCESS_MASK bits (1=single, 2=multi, 3=both).
        max_ranks_per_gpu: upper bound when ranks_per_gpu is unset.
        ranks_per_gpu: exact selector; 0 = unset (sweep 1..max). Must be >= 0 and,
            when set, <= max_ranks_per_gpu (mirrors the C++ UT_RANKS_PER_GPU
            validation -- the exact selector pins the loop, it does not raise max).
        only_pow2: drop non-power-of-2 GPU counts (UT_POW2_GPUS).

    Returns a list of SweepConfig (each == one generation).
    """
    if ranks_per_gpu < 0:
        raise ValueError(f"ranks_per_gpu must be >= 0 (0 = unset); got {ranks_per_gpu}")
    if ranks_per_gpu > 0 and ranks_per_gpu > max_ranks_per_gpu:
        raise ValueError(
            f"ranks_per_gpu ({ranks_per_gpu}) exceeds max_ranks_per_gpu "
            f"({max_ranks_per_gpu}); the exact selector pins the loop, it does not raise the max")

    mp_list = []
    if process_mask & PROCESS_SINGLE:
        mp_list.append(False)
    if process_mask & PROCESS_MULTI:
        mp_list.append(True)

    rpg_lo = ranks_per_gpu if ranks_per_gpu > 0 else 1
    rpg_hi = ranks_per_gpu if ranks_per_gpu > 0 else max_ranks_per_gpu

    configs = []
    for g in num_gpus_list:
        if only_pow2 and (g & (g - 1)) != 0:
            continue
        for mp in mp_list:
            for r in range(rpg_lo, rpg_hi + 1):
                configs.append(SweepConfig(g, mp, r))
    return configs


@dataclass
class Eligibility:
    """Per suite/test init-pipeline eligibility (from config; §4.1a)."""
    warmup_profile: str = PROFILE_NONE
    # fork_coll expansion selectors (ignored for other profiles):
    num_gpus: list = field(default_factory=list)
    process_mask: int = PROCESS_SINGLE | PROCESS_MULTI
    max_ranks_per_gpu: int = 1
    ranks_per_gpu: int = 0
    only_pow2: bool = False

    @property
    def is_pipeline(self):
        return is_pipeline_profile(self.warmup_profile)


def expand_fork_entry(eligibility):
    """Expand a fork_coll entry into single-generation sub-entries.

    Returns a list of dicts: {suffix, env, expected_generations}. Non-fork
    profiles return [] (they are not sweep-expanded: MPI/NetIb run one generation
    per process already). Raises ValueError via enumerate_sweep on a bad selector.
    """
    if eligibility.warmup_profile != PROFILE_FORK:
        return []
    configs = enumerate_sweep(
        eligibility.num_gpus,
        process_mask=eligibility.process_mask,
        max_ranks_per_gpu=eligibility.max_ranks_per_gpu,
        ranks_per_gpu=eligibility.ranks_per_gpu,
        only_pow2=eligibility.only_pow2,
    )
    return [{"suffix": c.suffix(), "env": c.env(), "expected_generations": 1} for c in configs]


def build_manifest(parent_test, eligibility):
    """Authoritative, side-effect-free manifest for --dry-run/--emit-manifest.

    One row per sub-entry for fork_coll; a single row for other pipeline profiles;
    empty for the serial path.
    """
    if not eligibility.is_pipeline:
        return []
    if eligibility.warmup_profile == PROFILE_FORK:
        rows = []
        for c in enumerate_sweep(
                eligibility.num_gpus,
                process_mask=eligibility.process_mask,
                max_ranks_per_gpu=eligibility.max_ranks_per_gpu,
                ranks_per_gpu=eligibility.ranks_per_gpu,
                only_pow2=eligibility.only_pow2):
            rows.append({
                "parent_test": parent_test,
                "num_gpus": c.num_gpus,
                "multi_process": c.multi_process,
                "ranks_per_gpu": c.ranks_per_gpu,
                "expected_generations": 1,
            })
        return rows
    # mpi_coll / netib_plugin: one entry, one generation per process.
    return [{
        "parent_test": parent_test,
        "warmup_profile": eligibility.warmup_profile,
        "expected_generations": 1,
    }]
