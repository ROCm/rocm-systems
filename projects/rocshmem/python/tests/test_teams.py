"""
Multi-PE tests; require at least 2 PEs.
"""

import pytest

import rocshmem4py
from rocshmem4py import (
    ROCSHMEM_SUCCESS,
    ROCSHMEM_TEAM_INVALID,
    ROCSHMEM_TEAM_WORLD,
    TeamConfig,
    rocshmem_my_pe,
    rocshmem_n_pes,
    rocshmem_team_my_pe,
    rocshmem_team_n_pes,
    rocshmem_team_split_strided,
    rocshmem_team_destroy,
    rocshmem_team_translate_pe,
    rocshmem_barrier_all,
)
from conftest import requires_multi_pe


def test_team_config_struct():
    """TeamConfig is constructible and num_contexts is read/write."""
    cfg = TeamConfig()
    cfg.num_contexts = 4
    assert cfg.num_contexts == 4


@requires_multi_pe
def test_split_strided_world():
    """split_strided(WORLD, 0, 1, n_pes) -> handle equivalent to WORLD."""
    n = rocshmem_n_pes()
    cfg = TeamConfig()
    cfg.num_contexts = 1
    status, team = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 1, n, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS
    assert team != ROCSHMEM_TEAM_INVALID

    assert rocshmem_team_n_pes(team) == n
    assert rocshmem_team_my_pe(team) == rocshmem_my_pe()

    rocshmem_barrier_all()
    rocshmem_team_destroy(team)
    rocshmem_barrier_all()


@requires_multi_pe
def test_split_strided_even_subteam():
    """split_strided over even PEs only; odd PEs see team_my_pe == -1."""
    n = rocshmem_n_pes()
    if n < 4:
        pytest.skip("requires >= 4 PEs so the even subteam has multiple members")

    cfg = TeamConfig()
    cfg.num_contexts = 1
    even_size = (n + 1) // 2  # ceil(n/2)
    status, team = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 2, even_size, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS

    me = rocshmem_my_pe()
    if me % 2 == 0:
        assert team != ROCSHMEM_TEAM_INVALID
        assert rocshmem_team_my_pe(team) == me // 2
        assert rocshmem_team_n_pes(team) == even_size
    else:
        # Odd PEs are non-members of the even-subteam; the binding's
        # tracked split helper translates the C nullptr to the Python
        # ROCSHMEM_TEAM_INVALID sentinel.  Asserting `==` (not just
        # `!= some_real_team`) is the regression guard that the
        # WORLD/INVALID sentinel collapse stays fixed.
        assert team == ROCSHMEM_TEAM_INVALID

    rocshmem_barrier_all()
    rocshmem_team_destroy(team)  # no-op for INVALID per spec
    rocshmem_barrier_all()


@requires_multi_pe
def test_split_strided_subset_parent_then_split():
    """Port of upstream teamctxsubsetparentinfra (PR ROCm/rocm-systems#5960).

    Two-level split exercising the non-WORLD-parent path that PR #5960
    enabled in the IPC backend's create_new_team via groupAllGather over
    the new team's world ranks (instead of an all-PEs allGather).

    Step 1: parity-based subset parent.
      - even PEs build team {0,2,4,...}
      - odd  PEs build team {1,3,5,...}
      With odd n_pes the even subset is one PE larger than the odd subset.

    Step 2: split that subset parent (which is NOT WORLD) into two halves.
      Lower half PEs of the subset parent get team {0..mid-1}; upper half
      PEs get team {mid..subset_n-1}.

    Designed for 4 and 5 PEs to mirror the upstream functional-tests
    driver lines `ExecTest "teamctxsubsetparentinfra" 4 1 1` and `5 1 1`.
    """
    n = rocshmem_n_pes()
    if n < 4:
        pytest.skip("requires >= 4 PEs to form non-trivial subset parents")

    me = rocshmem_my_pe()
    cfg = TeamConfig()
    cfg.num_contexts = 1

    # ---------- Step 1: subset parent via parity split ----------
    parity = me % 2
    start_pe = parity  # 0 for even, 1 for odd
    subset_size = n // 2
    if (n % 2) != 0 and parity == 0:
        subset_size += 1  # even subset absorbs the leftover when n is odd

    status, subset_parent = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, start_pe, 2, subset_size, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS
    assert subset_parent != ROCSHMEM_TEAM_INVALID, (
        f"PE {me}: subset parent (parity={parity}, size={subset_size}) "
        "must be valid; every PE is a member of exactly one parity subset"
    )
    assert rocshmem_team_n_pes(subset_parent) == subset_size
    assert rocshmem_team_my_pe(subset_parent) == me // 2

    # ---------- Step 2: split the subset parent (non-WORLD) into halves ----------
    subset_n = rocshmem_team_n_pes(subset_parent)
    subset_me = rocshmem_team_my_pe(subset_parent)
    mid = subset_n // 2

    if subset_me < mid:
        child_start = 0
        child_size = mid
        expected_child_pe = subset_me
    else:
        child_start = mid
        child_size = subset_n - mid
        expected_child_pe = subset_me - mid

    status, child = rocshmem_team_split_strided(
        subset_parent, child_start, 1, child_size, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS
    assert child != ROCSHMEM_TEAM_INVALID, (
        f"PE {me}: child team (subset_parent parity={parity}, "
        f"start={child_start}, size={child_size}) must be valid; every PE in "
        "the subset parent participates in exactly one half"
    )
    assert rocshmem_team_n_pes(child) == child_size
    assert rocshmem_team_my_pe(child) == expected_child_pe

    # ---------- Cleanup: destroy in reverse creation order ----------
    rocshmem_barrier_all()
    rocshmem_team_destroy(child)
    rocshmem_team_destroy(subset_parent)
    rocshmem_barrier_all()


@requires_multi_pe
def test_split_strided_recursive_split_world():
    """Recursive split of WORLD: WORLD -> halves -> split one half.

    Complementary to the parity-based test above: this exercises the
    non-WORLD-parent path with a contiguous (stride=1) child, and also
    confirms ROCSHMEM_TEAM_INVALID for non-members of the second-level
    split is observed correctly across PEs.
    """
    n = rocshmem_n_pes()
    if n < 4:
        pytest.skip("requires >= 4 PEs so each half can be split again")

    me = rocshmem_my_pe()
    cfg = TeamConfig()
    cfg.num_contexts = 1

    # First-level split: lower half {0..mid-1}, upper half {mid..n-1}.
    mid = n // 2
    if me < mid:
        half_start, half_size = 0, mid
        my_half_pe = me
    else:
        half_start, half_size = mid, n - mid
        my_half_pe = me - mid

    status, half = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, half_start, 1, half_size, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS
    assert half != ROCSHMEM_TEAM_INVALID
    assert rocshmem_team_n_pes(half) == half_size
    assert rocshmem_team_my_pe(half) == my_half_pe

    # Second-level split: split each half into a single-PE team containing
    # only its first PE.  Non-members (everyone except the half's PE 0)
    # must observe ROCSHMEM_TEAM_INVALID per OpenSHMEM spec.
    status, leader = rocshmem_team_split_strided(half, 0, 1, 1, cfg, 0)
    assert status == ROCSHMEM_SUCCESS

    if my_half_pe == 0:
        assert leader != ROCSHMEM_TEAM_INVALID
        assert rocshmem_team_n_pes(leader) == 1
        assert rocshmem_team_my_pe(leader) == 0
    else:
        # Every non-leader gets the Python ROCSHMEM_TEAM_INVALID sentinel.
        # Asserting `==` makes this a positive check for the sentinel
        # split (vs the previous WORLD==INVALID==0 collapse where this
        # would tautologically pass against any non-WORLD return).
        assert leader == ROCSHMEM_TEAM_INVALID

    rocshmem_barrier_all()
    rocshmem_team_destroy(leader)
    rocshmem_team_destroy(half)
    rocshmem_barrier_all()


def test_team_invalid_sentinel_distinct_from_world():
    """Regression guard: the two special team handles are distinct.

    Previously both ``ROCSHMEM_TEAM_WORLD`` and ``ROCSHMEM_TEAM_INVALID``
    were Python ``0``, which made non-member returns from
    ``rocshmem_team_split_strided`` indistinguishable from WORLD.  This
    test prevents that collapse from being re-introduced.
    """
    assert ROCSHMEM_TEAM_WORLD == 0
    assert ROCSHMEM_TEAM_INVALID == -1
    assert ROCSHMEM_TEAM_WORLD != ROCSHMEM_TEAM_INVALID
    # Both sentinels must be no-op safe to pass to destroy.
    rocshmem_team_destroy(ROCSHMEM_TEAM_WORLD)
    rocshmem_team_destroy(ROCSHMEM_TEAM_INVALID)


@requires_multi_pe
def test_team_translate_pe_round_trip():
    """team_translate_pe(child, child_pe, WORLD) returns the world rank."""
    n = rocshmem_n_pes()
    cfg = TeamConfig()
    cfg.num_contexts = 1
    status, team = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 1, n, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS

    # In a contiguous-strided child the translation is identity.
    for child_pe in range(n):
        world_pe = rocshmem_team_translate_pe(team, child_pe, ROCSHMEM_TEAM_WORLD)
        assert world_pe == child_pe

    rocshmem_barrier_all()
    rocshmem_team_destroy(team)
    rocshmem_barrier_all()


@requires_multi_pe
def test_team_destroy_idempotent_on_special_handles():
    """Destroying ROCSHMEM_TEAM_{INVALID,WORLD} is a documented no-op."""
    rocshmem_team_destroy(ROCSHMEM_TEAM_INVALID)
    rocshmem_team_destroy(ROCSHMEM_TEAM_WORLD)
    # Just survives without segfault.


@requires_multi_pe
def test_live_teams_registry_drain():
    """Tracked split adds to _live_teams; tracked destroy removes it."""
    from rocshmem4py import _live_teams

    n = rocshmem_n_pes()
    cfg = TeamConfig()
    cfg.num_contexts = 1
    initial = set(_live_teams)

    status, t1 = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 1, n, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS
    assert t1 in _live_teams
    assert t1 not in initial

    rocshmem_barrier_all()
    rocshmem_team_destroy(t1)
    rocshmem_barrier_all()
    assert t1 not in _live_teams
