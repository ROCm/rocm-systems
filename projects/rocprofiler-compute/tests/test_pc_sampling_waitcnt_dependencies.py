# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""Unit tests for ``utils.parser.parse_waitcnt_dependencies``.

These exercise the s_waitcnt-family dependency parser with small synthetic
single-code-object instruction lists (each dict carries ``code_obj_offset``,
``size``, ``name``, ``comment`` as produced by ``load_code_obj_info``). The
expected values mirror the real implementation's FIFO front-drain semantics.
"""

from typing import Any, Optional

import common  # noqa: F401  (adds src/ to sys.path)

from utils.parser import parse_waitcnt_dependencies


def _inst(
    offset: int,
    name: Optional[str],
    comment: Optional[str] = "",
    size: int = 4,
) -> dict[str, Any]:
    """Build one synthetic instruction dict in the load_code_obj_info shape."""
    return {
        "code_obj_offset": offset,
        "size": size,
        "name": name,
        "comment": comment,
    }


def test_vmcnt_producer_then_wait_records_dependency() -> None:
    """A vmcnt producer followed by s_waitcnt vmcnt(0) -> wait depends on it."""
    instructions = [
        _inst(0, "global_load_dwordx4 v[0:3], v[4:5]"),
        _inst(8, "s_waitcnt vmcnt(0)"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {8: [0]}


def test_lgkmcnt_producer_then_wait_records_dependency() -> None:
    """A ds_read producer followed by s_waitcnt lgkmcnt(0) -> dependency recorded."""
    instructions = [
        _inst(0, "ds_read_b32 v0, v1"),
        _inst(4, "s_waitcnt lgkmcnt(0)"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {4: [0]}


def test_scalar_load_is_lgkmcnt_producer() -> None:
    """s_load_* increments lgkmcnt, so a later lgkmcnt wait depends on it."""
    instructions = [
        _inst(0, "s_load_b64 s[0:1], s[2:3]"),
        _inst(8, "s_waitcnt lgkmcnt(0)"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {8: [0]}


def test_export_is_expcnt_producer() -> None:
    """exp instructions bump expcnt; s_waitcnt expcnt(0) depends on them."""
    instructions = [
        _inst(0, "exp mrt0 v0, v1, v2, v3"),
        _inst(8, "s_waitcnt expcnt(0)"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {8: [0]}


def test_combined_waitcnt_depends_on_both_classes() -> None:
    """A combined vmcnt(0) lgkmcnt(0) wait depends on both producer classes."""
    instructions = [
        _inst(0, "global_load_dwordx4 v[0:3], v[4:5]"),  # vmcnt producer
        _inst(4, "ds_read_b32 v6, v7"),  # lgkmcnt producer
        _inst(8, "s_waitcnt vmcnt(0) lgkmcnt(0)"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {8: [0, 4]}


def test_vmcnt_one_drains_oldest_keeps_most_recent_outstanding() -> None:
    """vmcnt(N) front-drains the queue: vmcnt(1) clears all but the newest.

    With two outstanding producers [P0, P1] and target N=1, keep_from =
    len-N = 1, so the front slice [P0] is satisfied and P1 stays outstanding.
    A following vmcnt(0) then picks up the still-outstanding P1.
    """
    instructions = [
        _inst(0, "global_load_dwordx4 v[0:3], v[4:5]"),  # P0
        _inst(4, "global_load_dword v8, v[9:10]"),  # P1 (most recent)
        _inst(8, "s_waitcnt vmcnt(1)"),  # drains oldest P0 only
        _inst(12, "s_waitcnt vmcnt(0)"),  # drains the remaining P1
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {8: [0], 12: [4]}


def test_bare_waitcnt_drains_all_classes() -> None:
    """A bare s_waitcnt with no operands drains every counter class fully."""
    instructions = [
        _inst(0, "global_load_dwordx4 v[0:3], v[4:5]"),  # vmcnt
        _inst(4, "ds_read_b32 v6, v7"),  # lgkmcnt
        _inst(8, "exp mrt0 v0, v1, v2, v3"),  # expcnt
        _inst(12, "s_waitcnt"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {12: [0, 4, 8]}


def test_vscnt_shares_vmcnt_queue() -> None:
    """vscnt(0) drains the vmcnt outstanding queue (shared counter family)."""
    instructions = [
        _inst(0, "global_store_dword v[0:1], v2"),  # vmcnt producer (global_)
        _inst(8, "s_waitcnt vscnt(0)"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {8: [0]}


def test_class_specific_mnemonic_waitcnt() -> None:
    """A class-specific s_waitcnt_lgkmcnt mnemonic drains the lgkmcnt queue."""
    instructions = [
        _inst(0, "ds_read_b32 v0, v1"),
        _inst(8, "s_waitcnt_lgkmcnt 0"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {8: [0]}


def test_waitcnt_with_no_preceding_producer_has_no_entry() -> None:
    """A waitcnt with nothing outstanding records no dependency entry."""
    instructions = [
        _inst(0, "v_mov_b32 v0, v1"),  # not a producer
        _inst(4, "s_waitcnt vmcnt(0)"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {}


def test_producer_without_waitcnt_is_not_a_key() -> None:
    """A producer with no governing waitcnt never appears as a dict key."""
    instructions = [
        _inst(0, "global_load_dwordx4 v[0:3], v[4:5]"),
        _inst(4, "v_add_u32 v0, v1, v2"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {}


def test_empty_instruction_list_returns_empty_dict() -> None:
    """An empty instruction list yields an empty dependency map."""
    assert parse_waitcnt_dependencies([]) == {}


def test_missing_or_none_name_does_not_raise() -> None:
    """Instructions with a missing or None name are skipped, no exception."""
    instructions = [
        {"code_obj_offset": 0, "size": 4},  # no "name" key at all
        _inst(4, None, comment="/x.cpp:1"),  # explicit None name
        _inst(8, "global_load_dwordx4 v[0:3], v[4:5]"),  # producer survives
        _inst(12, "s_waitcnt vmcnt(0)"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {12: [8]}


def test_missing_comment_key_does_not_raise() -> None:
    """A producer/waitcnt pair with no comment key still parses normally."""
    instructions = [
        {"code_obj_offset": 0, "size": 4, "name": "ds_read_b32 v0, v1"},
        {"code_obj_offset": 4, "size": 4, "name": "s_waitcnt lgkmcnt(0)"},
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {4: [0]}


def test_producer_with_none_offset_is_skipped() -> None:
    """A producer whose offset is None cannot be depended on; no entry results."""
    instructions = [
        {"code_obj_offset": None, "size": 4, "name": "global_load_dwordx4 v0, v1"},
        _inst(8, "s_waitcnt vmcnt(0)"),
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {}


def test_only_nonempty_deps_recorded_across_multiple_waits() -> None:
    """Multiple waits: only those with outstanding producers get an entry."""
    instructions = [
        _inst(0, "global_load_dwordx4 v[0:3], v[4:5]"),  # vmcnt P0
        _inst(4, "s_waitcnt lgkmcnt(0)"),  # nothing in lgkmcnt -> no entry
        _inst(8, "s_waitcnt vmcnt(0)"),  # drains P0 -> entry
    ]
    deps = parse_waitcnt_dependencies(instructions)
    assert deps == {8: [0]}


def test_combined_vmcnt_vscnt_drains_fully_regardless_of_operand_order() -> None:
    """vscnt shares the vmcnt outstanding queue. A single s_waitcnt carrying
    both vmcnt(N) and vscnt(0) must drain to the stricter target (0) no matter
    which operand appears first, so the producer is always a dependency."""
    for wait in ("s_waitcnt vscnt(0) vmcnt(2)", "s_waitcnt vmcnt(2) vscnt(0)"):
        instructions = [
            _inst(0, "global_load_dwordx4 v[0:3], v[4:5]"),
            _inst(8, wait),
        ]
        deps = parse_waitcnt_dependencies(instructions)
        assert deps.get(8) == [0], f"operand order {wait!r} under-drained: {deps}"
