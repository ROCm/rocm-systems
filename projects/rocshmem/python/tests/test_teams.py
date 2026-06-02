"""
Multi-PE tests; require at least 2 PEs.
"""

import time

import pytest

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
    rocshmem_barrier,
    rocshmem_team_sync,
    rocshmem_barrier_on_stream,
    rocshmem_team_sync_on_stream,
    hip_device_synchronize,
    rocshmem_malloc,
    rocshmem_free,
    rocshmem_putmem_on_stream,
    rocshmem_getmem_on_stream,
)
from conftest import requires_multi_pe
from hip_test_utils import HipStream, load_u64, store_u64


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

    my_pe = rocshmem_my_pe()
    if my_pe % 2 == 0:
        assert team != ROCSHMEM_TEAM_INVALID
        assert rocshmem_team_my_pe(team) == my_pe // 2
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

    my_pe = rocshmem_my_pe()
    cfg = TeamConfig()
    cfg.num_contexts = 1

    # ---------- Step 1: subset parent via parity split ----------
    parity = my_pe % 2
    start_pe = parity  # 0 for even, 1 for odd
    subset_size = n // 2
    if (n % 2) != 0 and parity == 0:
        subset_size += 1  # even subset absorbs the leftover when n is odd

    status, subset_parent = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, start_pe, 2, subset_size, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS
    assert subset_parent != ROCSHMEM_TEAM_INVALID, (
        f"PE {my_pe}: subset parent (parity={parity}, size={subset_size}) "
        "must be valid; every PE is a member of exactly one parity subset"
    )
    assert rocshmem_team_n_pes(subset_parent) == subset_size
    assert rocshmem_team_my_pe(subset_parent) == my_pe // 2

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
        f"PE {my_pe}: child team (subset_parent parity={parity}, "
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

    my_pe = rocshmem_my_pe()
    cfg = TeamConfig()
    cfg.num_contexts = 1

    # First-level split: lower half {0..mid-1}, upper half {mid..n-1}.
    mid = n // 2
    if my_pe < mid:
        half_start, half_size = 0, mid
        my_half_pe = my_pe
    else:
        half_start, half_size = mid, n - mid
        my_half_pe = my_pe - mid

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
def test_team_sync_barrier_member_scoped():
    """Non-members should not block on team collectives.

    Construct an even-rank subteam. Delay one team member before entering
    team sync / barrier, then assert:
      - other members block waiting for the delayed member
      - non-members (TEAM_INVALID path) return quickly
    """
    n = rocshmem_n_pes()
    if n < 4:
        pytest.skip("requires >= 4 PEs to exercise member/non-member timing")

    cfg = TeamConfig()
    cfg.num_contexts = 1
    even_size = (n + 1) // 2
    status, even_team = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 2, even_size, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS

    my_pe = rocshmem_my_pe()
    is_member = (my_pe % 2) == 0
    if is_member:
        assert even_team != ROCSHMEM_TEAM_INVALID
        team_rank = rocshmem_team_my_pe(even_team)
    else:
        assert even_team == ROCSHMEM_TEAM_INVALID
        team_rank = -1

    delay_s = 0.8

    # Phase 1: sync timing
    rocshmem_barrier_all()
    if is_member and team_rank == 0:
        time.sleep(delay_s)
    t0 = time.perf_counter()
    rocshmem_team_sync(even_team)
    sync_dt = time.perf_counter() - t0

    # Phase 2: barrier timing
    rocshmem_barrier_all()
    if is_member and team_rank == 0:
        time.sleep(delay_s)
    t0 = time.perf_counter()
    rocshmem_barrier(even_team)
    barrier_dt = time.perf_counter() - t0

    rocshmem_barrier_all()
    rocshmem_team_destroy(even_team)
    rocshmem_barrier_all()

    if is_member and team_rank != 0:
        # Other team members must wait for delayed member.
        # higher time on the non-delayed member is the expected 
        # behavior for a collective barrier/sync.
        assert sync_dt > delay_s * 0.6
        assert barrier_dt > delay_s * 0.6
    if not is_member:
        # Non-members pass TEAM_INVALID and should not wait for team progress.
        assert sync_dt < delay_s * 0.5
        assert barrier_dt < delay_s * 0.5


@requires_multi_pe
def test_team_sync_store_visibility():
    """`team_sync` gates team-local store visibility for member communication."""
    n = rocshmem_n_pes()
    if n < 4:
        pytest.skip("requires >= 4 PEs to exercise member/non-member behavior")

    cfg = TeamConfig()
    cfg.num_contexts = 1
    even_size = (n + 1) // 2
    status, even_team = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 2, even_size, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS

    my_pe = rocshmem_my_pe()
    is_member = (my_pe % 2) == 0
    if is_member:
        assert even_team != ROCSHMEM_TEAM_INVALID
        team_rank = rocshmem_team_my_pe(even_team)
        team_n = rocshmem_team_n_pes(even_team)
    else:
        assert even_team == ROCSHMEM_TEAM_INVALID
        team_rank = -1
        team_n = 0

    local_val = rocshmem_malloc(8)
    pull_buf = rocshmem_malloc(8)
    try:
        store_u64(local_val, my_pe + 5000)
        store_u64(pull_buf, 0xFFFFFFFFFFFFFFFF)

        # team_sync is the ONLY synchronization gating the cross-member read
        # below.  Per rocSHMEM semantics it must guarantee every member's
        # local store (above) is complete and visible to the other members
        # before any of them returns. 
        rocshmem_team_sync(even_team)

        if is_member and team_n > 1:
            prev_rank = (team_rank - 1 + team_n) % team_n
            prev_world = rocshmem_team_translate_pe(
                even_team, prev_rank, ROCSHMEM_TEAM_WORLD
            )
            rocshmem_getmem_on_stream(pull_buf, local_val, 8, prev_world, 0)
            hip_device_synchronize()
            assert load_u64(pull_buf) == prev_world + 5000

        rocshmem_barrier_all()
    finally:
        rocshmem_free(local_val)
        rocshmem_free(pull_buf)
        rocshmem_barrier_all()
        rocshmem_team_destroy(even_team)
        rocshmem_barrier_all()


@requires_multi_pe
def test_team_barrier_rma_ordering():
    """Team `barrier` synchronizes members after remote RMA so each member
    can safely read data a peer wrote into its inbox.

    Production gotcha exercised here: the put is issued via the stream-ordered
    API and flushed with ``hip_device_synchronize()`` *before* the blocking
    host barrier.  A blocking host barrier does NOT flush stream-enqueued RMA,
    so omitting that device sync would race.  The team barrier's load-bearing
    role is the cross-PE arrival ordering -- a member cannot read its inbox
    until the writing peer has reached the barrier, i.e. issued and locally
    completed its put.
    """
    n = rocshmem_n_pes()
    if n < 4:
        pytest.skip("requires >= 4 PEs to exercise member/non-member behavior")

    cfg = TeamConfig()
    cfg.num_contexts = 1
    even_size = (n + 1) // 2
    status, even_team = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 2, even_size, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS

    my_pe = rocshmem_my_pe()
    is_member = (my_pe % 2) == 0
    if is_member:
        assert even_team != ROCSHMEM_TEAM_INVALID
        team_rank = rocshmem_team_my_pe(even_team)
        team_n = rocshmem_team_n_pes(even_team)
    else:
        assert even_team == ROCSHMEM_TEAM_INVALID
        team_rank = -1
        team_n = 0

    inbox = rocshmem_malloc(8)
    send_val_ptr = rocshmem_malloc(8)
    try:
        store_u64(inbox, 0xABCDEF00 + my_pe)
        store_u64(send_val_ptr, my_pe + 7000)

        rocshmem_barrier_all()

        if is_member and team_n > 1:
            next_rank = (team_rank + 1) % team_n
            next_world = rocshmem_team_translate_pe(
                even_team, next_rank, ROCSHMEM_TEAM_WORLD
            )
            rocshmem_putmem_on_stream(inbox, send_val_ptr, 8, next_world, 0)
            # Required: flush the stream-enqueued put before the blocking host
            # barrier, which has no visibility into stream-ordered RMA.
            hip_device_synchronize()

        # Cross-PE arrival ordering: no member proceeds past here until every
        # member has reached the barrier, i.e. completed its put above.
        rocshmem_barrier(even_team)

        if is_member and team_n > 1:
            prev_rank = (team_rank - 1 + team_n) % team_n
            prev_world = rocshmem_team_translate_pe(
                even_team, prev_rank, ROCSHMEM_TEAM_WORLD
            )
            assert load_u64(inbox) == prev_world + 7000

        rocshmem_barrier_all()
    finally:
        rocshmem_free(inbox)
        rocshmem_free(send_val_ptr)
        rocshmem_barrier_all()
        rocshmem_team_destroy(even_team)
        rocshmem_barrier_all()


@requires_multi_pe
def test_team_barrier_on_stream_rma_ordering():
    """Stream-ordered team barrier orders an enqueued put against a later read.

    This is the realistic overlap pattern: the put, the team barrier, and the
    dependent consumption are all driven from a single dedicated ``hipStream_t``
    with NO host-side device sync in between.  Correctness relies on two
    properties together:
      1. in-order stream execution -- the put copy completes before the
         barrier kernel on the same stream begins; and
      2. the collective team barrier -- a member's barrier completes only
         after every member (including the peer writing into its inbox) has
         reached the barrier, so the peer's put has landed and is visible.

    Non-members pass ROCSHMEM_TEAM_INVALID and must no-op while members make
    real RMA progress -- exercising the INVALID path interleaved with live
    team traffic, not in isolation.
    """
    n = rocshmem_n_pes()
    if n < 4:
        pytest.skip("requires >= 4 PEs to exercise member/non-member behavior")

    cfg = TeamConfig()
    cfg.num_contexts = 1
    even_size = (n + 1) // 2
    status, even_team = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 2, even_size, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS

    my_pe = rocshmem_my_pe()
    is_member = (my_pe % 2) == 0
    if is_member:
        assert even_team != ROCSHMEM_TEAM_INVALID
        team_rank = rocshmem_team_my_pe(even_team)
        team_n = rocshmem_team_n_pes(even_team)
    else:
        assert even_team == ROCSHMEM_TEAM_INVALID
        team_rank = -1
        team_n = 0

    inbox = rocshmem_malloc(8)
    send_val_ptr = rocshmem_malloc(8)
    try:
        store_u64(inbox, 0xABCDEF00 + my_pe)
        store_u64(send_val_ptr, my_pe + 9000)

        # Every inbox must be initialized before any peer writes into it,
        # otherwise a late init store could clobber a received value.
        rocshmem_barrier_all()

        with HipStream() as s:
            if is_member and team_n > 1:
                next_rank = (team_rank + 1) % team_n
                next_world = rocshmem_team_translate_pe(
                    even_team, next_rank, ROCSHMEM_TEAM_WORLD
                )
                rocshmem_putmem_on_stream(
                    inbox, send_val_ptr, 8, next_world, s.handle
                )
            # Stream-ordered team barrier: orders the put enqueued above and
            # synchronizes members.  Non-members enqueue the INVALID no-op.
            rocshmem_barrier_on_stream(even_team, s.handle)
            s.synchronize()

        if is_member and team_n > 1:
            prev_rank = (team_rank - 1 + team_n) % team_n
            prev_world = rocshmem_team_translate_pe(
                even_team, prev_rank, ROCSHMEM_TEAM_WORLD
            )
            assert load_u64(inbox) == prev_world + 9000

        # Explicit INVALID sentinel on a dedicated stream must no-op for every
        # rank (members and non-members alike) without hanging.
        with HipStream() as s2:
            rocshmem_barrier_on_stream(ROCSHMEM_TEAM_INVALID, s2.handle)
            rocshmem_team_sync_on_stream(ROCSHMEM_TEAM_INVALID, s2.handle)
            s2.synchronize()

        rocshmem_barrier_all()
    finally:
        rocshmem_free(inbox)
        rocshmem_free(send_val_ptr)
        rocshmem_barrier_all()
        rocshmem_team_destroy(even_team)
        rocshmem_barrier_all()


@requires_multi_pe
def test_team_sync_barrier_on_stream_member_scoped():
    """Stream-ordered team sync/barrier are team-scoped: non-members no-op.

    Stream analogue of ``test_team_sync_barrier_member_scoped``.  One team
    member delays *before* it enqueues + drives its stream-ordered collective,
    so its barrier kernel rendezvouses late.  Then:
      - other members block in ``stream.synchronize()`` waiting for the
        delayed member's barrier kernel to arrive on the device, and
      - non-members (TEAM_INVALID) enqueue nothing, so their
        ``stream.synchronize()`` on an empty stream returns promptly.

    This is the property the explicit-INVALID block in the data test cannot
    show: a non-member must not stall on team progress driven by the stream.
    """
    n = rocshmem_n_pes()
    if n < 4:
        pytest.skip("requires >= 4 PEs to exercise member/non-member timing")

    cfg = TeamConfig()
    cfg.num_contexts = 1
    even_size = (n + 1) // 2
    status, even_team = rocshmem_team_split_strided(
        ROCSHMEM_TEAM_WORLD, 0, 2, even_size, cfg, 0,
    )
    assert status == ROCSHMEM_SUCCESS

    my_pe = rocshmem_my_pe()
    is_member = (my_pe % 2) == 0
    if is_member:
        assert even_team != ROCSHMEM_TEAM_INVALID
        team_rank = rocshmem_team_my_pe(even_team)
    else:
        assert even_team == ROCSHMEM_TEAM_INVALID
        team_rank = -1

    delay_s = 0.8

    # Phase 1: sync_on_stream timing
    rocshmem_barrier_all()
    if is_member and team_rank == 0:
        time.sleep(delay_s)
    t0 = time.perf_counter()
    with HipStream() as s:
        rocshmem_team_sync_on_stream(even_team, s.handle)
        s.synchronize()
    sync_dt = time.perf_counter() - t0

    # Phase 2: barrier_on_stream timing
    rocshmem_barrier_all()
    if is_member and team_rank == 0:
        time.sleep(delay_s)
    t0 = time.perf_counter()
    with HipStream() as s:
        rocshmem_barrier_on_stream(even_team, s.handle)
        s.synchronize()
    barrier_dt = time.perf_counter() - t0

    rocshmem_barrier_all()
    rocshmem_team_destroy(even_team)
    rocshmem_barrier_all()

    if is_member and team_rank != 0:
        # Members must wait on the device rendezvous for the delayed member.
        assert sync_dt > delay_s * 0.6
        assert barrier_dt > delay_s * 0.6
    if not is_member:
        # Non-members enqueue nothing for INVALID; the empty stream drains
        # immediately and must not wait for the team's progress.
        assert sync_dt < delay_s * 0.5
        assert barrier_dt < delay_s * 0.5


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
