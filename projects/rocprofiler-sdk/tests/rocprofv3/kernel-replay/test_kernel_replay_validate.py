#!/usr/bin/env python3
#
# Unit tests for validate.py, the kernel-replay JSON validator.
#
# These feed synthetic rocprofv3 JSON to the validator's test functions and check which
# ones pass, skip or fail. They need no GPU, so they guard the rule for skipping a common
# sanity counter on every CI target: a counter is skipped only when a replayed agent does
# not report it, or when every replayed agent is gfx1200 or gfx1201 and the counter is
# explicitly zero in every pass of every dispatch. The same all-zero shape on gfx1250,
# gfx942, gfx11, or a mix with those agents must still fail.

import importlib.util
import itertools
import os
import sys

import pytest

COMMON = ["SQ_WAVES", "SQ_INSTS_VALU"]
TARGET = "SQ_INSTS_VALU"
UNIQUE = [
    "GRBM_COUNT",
    "GRBM_GUI_ACTIVE",
    "SQ_INSTS_SALU",
    "SQ_INSTS_SMEM",
    "SQ_INSTS_LDS",
]
KERNELS = [
    ("vecAdd", 1024 * 1024, 1024, 16384.0),
    ("saxpy", 512 * 512, 512, 4096.0),
    ("vecScale", 256 * 256, 256, 1024.0),
]

# Values a common counter can take in one batch (one pass of one dispatch).
ABSENT = None
ZERO = 0.0
NONZERO = 500.0

AGENT_SETS = [
    ("gfx1201",),
    ("gfx1201", "gfx1201"),
    ("gfx1250",),
    ("gfx942",),
    ("gfx1200",),
    ("gfx1200", "gfx1201"),
    ("gfx1200", "gfx1250"),
    ("gfx1201", "gfx1250"),
    ("gfx1201", "gfx942"),
    (None,),
]

_MODULE = None


def validate():
    global _MODULE
    if _MODULE is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate.py")
        spec = importlib.util.spec_from_file_location("kernel_replay_validate", path)
        _MODULE = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_MODULE)
    return _MODULE


def make_json(archs, values, waves=None, unreported=(), passes=None):
    """Build rocprofv3 kernel-replay JSON.

    archs: one name per GPU agent; None leaves that agent out of the agents section.
    values: values[dispatch][pass] for TARGET (ABSENT omits it from that batch).
    waves: optional values[dispatch][pass] for SQ_WAVES; defaults to a per-kernel constant.
    unreported: counter names left out of every agent's counters section.
    Dispatch d runs kernel d % 3 on agent d % len(archs).
    """
    passes = passes if passes is not None else len(values[0])
    agents, counters, cid = [], [], {}
    for index, arch in enumerate(archs):
        handle = 100 + index
        if arch is not None:
            agents.append({"id": {"handle": handle}, "name": arch, "type": 2})
        for name in COMMON + UNIQUE:
            cid[(handle, name)] = 1000 * (index + 1) + len(cid)
            if name not in unreported:
                counters.append(
                    {
                        "id": {"handle": cid[(handle, name)]},
                        "name": name,
                        "agent_id": {"handle": handle},
                    }
                )
    symbols = [
        {"kernel_id": k + 1, "formatted_kernel_name": f"{name}(float*)"}
        for k, (name, _, _, _) in enumerate(KERNELS)
    ]
    records = []
    for dispatch, row in enumerate(values):
        handle = 100 + dispatch % len(archs)
        kernel = dispatch % len(KERNELS)
        _, grid, workgroup, wave_count = KERNELS[kernel]
        for pass_index in range(passes):
            batch = {}
            wave = wave_count if waves is None else waves[dispatch][pass_index]
            if wave is not ABSENT:
                batch["SQ_WAVES"] = wave
            if row[pass_index] is not ABSENT:
                batch[TARGET] = row[pass_index]
            batch[UNIQUE[pass_index % len(UNIQUE)]] = 7.0 + pass_index
            records.append(
                {
                    "replay_pass": pass_index,
                    "dispatch_data": {
                        "dispatch_info": {
                            "dispatch_id": dispatch + 1,
                            "kernel_id": kernel + 1,
                            "agent_id": {"handle": handle},
                            "grid_size": {"x": grid},
                            "workgroup_size": {"x": workgroup},
                        }
                    },
                    "records": [
                        {"counter_id": {"handle": cid[(handle, name)]}, "value": value}
                        for name, value in batch.items()
                    ],
                }
            )
    tool = {
        "agents": agents,
        "counters": counters,
        "kernel_symbols": symbols,
        "callback_records": {"counter_collection": records},
    }
    return {"rocprofiler-sdk-tool": [tool]}


def outcome(func, *args):
    try:
        func(*args)
    except pytest.skip.Exception:
        return "skip"
    except AssertionError:
        return "fail"
    return "pass"


def constant(data, counter=TARGET):
    return outcome(validate().test_common_counters_constant_across_passes, data, counter)


def grid(fill, dispatches=3, passes=5):
    return [[fill] * passes for _ in range(dispatches)]


def all_navi4(archs):
    return bool(archs) and all(arch in ("gfx1200", "gfx1201") for arch in archs)


def expected_target_outcome(archs, values, unreported=()):
    """The rule under test, stated independently of validate.py."""
    batches = [v for row in values for v in row]
    if TARGET in unreported:
        return "skip"
    if all_navi4(archs) and all(v is not ABSENT and v == 0 for v in batches):
        return "skip"
    if all(v is not ABSENT and v > 0 for v in batches):
        return "pass"
    return "fail"


def test_exhaustive_two_by_two_batches():
    # Every assignment of {absent, zero, nonzero} to 2 dispatches x 2 passes, on every
    # agent set: the validator must agree with the stated rule in all 81 x 10 cases.
    states = (ABSENT, ZERO, NONZERO)
    mismatches = []
    for archs in AGENT_SETS:
        for cells in itertools.product(states, repeat=4):
            values = [list(cells[:2]), list(cells[2:])]
            got = constant(make_json(archs, values))
            want = expected_target_outcome(archs, values)
            if got != want:
                mismatches.append((archs, values, got, want))
    assert not mismatches, f"{len(mismatches)} mismatches, first: {mismatches[:3]}"


def test_zero_everywhere_skips_only_on_navi4():
    for archs in AGENT_SETS:
        want = "skip" if all_navi4(archs) else "fail"
        got = constant(make_json(archs, grid(ZERO)))
        assert got == want, f"{archs}: all-zero {TARGET} gave {got}, want {want}"


def test_gfx1250_all_zero_fails():
    # ROCM-32192: an all-zero SQ_INSTS_VALU on gfx1250 is an open incident, not a skip.
    data = make_json(("gfx1250",), grid(ZERO))
    assert constant(data) == "fail"
    assert outcome(validate().test_some_common_counter_usable, data, COMMON) == "pass"


def test_zero_in_one_pass_fails_on_gfx1201():
    values = grid(NONZERO)
    values[1][3] = ZERO
    assert constant(make_json(("gfx1201",), values)) == "fail"


def test_zero_for_one_dispatch_fails_on_gfx1201():
    values = grid(NONZERO)
    values[2] = [ZERO] * 5
    assert constant(make_json(("gfx1201",), values)) == "fail"


def test_zero_with_missing_batch_fails_on_gfx1201():
    # A missing batch is not a zero reading; it must reach the per-pass presence check.
    values = grid(ZERO)
    values[0][2] = ABSENT
    assert constant(make_json(("gfx1201",), values)) == "fail"


def test_absent_everywhere_fails_when_reported():
    for archs in AGENT_SETS:
        assert constant(make_json(archs, grid(ABSENT))) == "fail", archs


def test_unreported_candidate_is_skipped_everywhere():
    for archs in AGENT_SETS:
        data = make_json(archs, grid(ABSENT), unreported=(TARGET,))
        assert constant(data) == "skip", archs
        assert outcome(validate().test_expected_counters_present, data, COMMON) == "pass"
        assert outcome(validate().test_some_common_counter_usable, data, COMMON) == "pass"


def test_unreported_non_common_counter_still_required():
    data = make_json(("gfx1201",), grid(NONZERO), unreported=("GRBM_COUNT",))
    assert outcome(validate().test_expected_counters_present, data, COMMON) == "fail"


def test_reported_common_counter_still_required():
    data = make_json(("gfx1201",), grid(ABSENT))
    assert outcome(validate().test_expected_counters_present, data, COMMON) == "fail"


def test_wave_check_still_runs_when_target_skipped():
    waves = [[16384.0] * 5, [4096.0] * 5, [1024.0] * 5]
    waves[0][3] = 16352.0
    data = make_json(("gfx1201",), grid(ZERO), waves=waves)
    assert constant(data) == "skip"
    assert constant(data, "SQ_WAVES") == "fail"


def test_no_usable_common_counter_fails():
    data = make_json(("gfx1201",), grid(ZERO), waves=grid(ZERO))
    assert constant(data) == "skip"
    assert constant(data, "SQ_WAVES") == "skip"
    assert outcome(validate().test_some_common_counter_usable, data, COMMON) == "fail"
    assert (
        outcome(validate().test_counters_differ_between_kernels, data, COMMON) == "skip"
    )


def test_kernels_compared_over_usable_counters():
    data = make_json(("gfx1201",), grid(ZERO))
    assert (
        outcome(validate().test_counters_differ_between_kernels, data, COMMON) == "pass"
    )


def test_baseline_passes_every_validator_check():
    v = validate()
    for archs in (("gfx942",), ("gfx1201",), ("gfx1250",)):
        data = make_json(archs, grid(NONZERO))
        checks = [
            (v.test_dispatch_id_constant_across_replay_passes, data, 5),
            (v.test_dispatch_ids_increment_sequentially, data),
            (v.test_dispatch_id_nonzero_every_pass, data),
            (v.test_some_common_counter_usable, data, COMMON),
            (v.test_each_pass_collects_distinct_batch, data, 5, COMMON),
            (v.test_counters_differ_between_kernels, data, COMMON),
            (v.test_replayed_kernels_present, data),
            (v.test_expected_counters_present, data, COMMON),
            (v.test_launch_dimensions, data),
        ]
        for check in checks:
            assert outcome(*check) == "pass", f"{archs}: {check[0].__name__}"
        for counter in COMMON:
            assert constant(data, counter) == "pass", f"{archs}: {counter}"


def test_zero_everywhere_fails_on_gfx11_and_other_gfx12_names():
    # Only gfx1200 and gfx1201 may skip an all-zero counter. gfx11 (where the old AUTO
    # perf-level skip applied) and any other gfx12 name must still fail, alone or mixed.
    for archs in (
        ("gfx1100",),
        ("gfx1102",),
        ("gfx1151",),
        ("gfx1202",),
        ("gfx1201", "gfx1100"),
        ("gfx1200", "gfx1151"),
    ):
        assert constant(make_json(archs, grid(ZERO))) == "fail", archs


def test_instruction_counter_is_exact_across_passes():
    # SQ_INSTS_VALU replays bit-for-bit. One pass off by a single instruction, or one pass
    # missing 1 of 32 instance records (~3% low), is a failure on every agent.
    for archs in (("gfx942",), ("gfx1201",), ("gfx1250",)):
        for bad in (NONZERO + 1, NONZERO * 31 / 32):
            values = grid(NONZERO)
            values[1][2] = bad
            assert constant(make_json(archs, values)) == "fail", (archs, bad)


def test_wave_spread_boundary_is_eight_waves():
    # SQ_WAVES jitters by a few waves on gfx94x replay: a spread of exactly 8 waves passes,
    # 9 fails. This pins the tolerance from both sides.
    for spread, want in ((8.0, "pass"), (9.0, "fail")):
        waves = [[16384.0] * 5, [4096.0] * 5, [1024.0] * 5]
        waves[2][4] = 1024.0 + spread
        data = make_json(("gfx942",), grid(NONZERO), waves=waves)
        assert constant(data, "SQ_WAVES") == want, spread


def test_identical_kernel_signatures_fail():
    # Metric 4 must be able to fail: kernels with the same usable counters are not
    # distinguishable. On gfx1201 the skipped SQ_INSTS_VALU is left out of the signature,
    # so identical SQ_WAVES alone must fail there too.
    waves = [[4096.0] * 5 for _ in range(3)]
    v = validate()
    for archs, values in ((("gfx942",), grid(NONZERO)), (("gfx1201",), grid(ZERO))):
        data = make_json(archs, values, waves=waves)
        assert (
            outcome(v.test_counters_differ_between_kernels, data, COMMON) == "fail"
        ), archs


def test_negative_reading_is_not_an_unimplemented_zero():
    # Only explicit zeros mark a counter as unimplemented on gfx1200/gfx1201. A negative
    # reading (for example a wrapped accumulator) is a failure, not a skip.
    values = grid(ZERO)
    values[1][1] = -1.0
    assert constant(make_json(("gfx1201",), values)) == "fail"
    assert constant(make_json(("gfx1200",), grid(-1.0))) == "fail"


def test_duplicate_pass_record_fails_dispatch_id_check():
    # _records_by_dispatch keeps the last record of a (dispatch, pass), so a duplicated pass
    # carrying a nonzero SQ_INSTS_VALU in front of a zero one is invisible to the zero check.
    # The dispatch-id check is what rejects the duplicate.
    data = make_json(("gfx1201",), grid(ZERO))
    records = data["rocprofiler-sdk-tool"][0]["callback_records"]["counter_collection"]
    first = records[0]
    records.insert(
        0, dict(first, records=[dict(s, value=NONZERO) for s in first["records"]])
    )
    v = validate()
    assert outcome(v.test_dispatch_id_constant_across_replay_passes, data, 5) == "fail"


def test_n_pass_key_is_accepted():
    # The per-pass index may be named "n" instead of "replay_pass".
    data = make_json(("gfx942",), grid(NONZERO))
    for rec in data["rocprofiler-sdk-tool"][0]["callback_records"]["counter_collection"]:
        rec["n"] = rec.pop("replay_pass")
    assert constant(data) == "pass"


def test_nan_in_a_later_pass_fails():
    # rocprofv3 JSON can carry NaN (cereal writes NaN tokens and json.load accepts them).
    # min() and max() skip a NaN that is not the first value, so every value must be checked.
    for archs in (("gfx942",), ("gfx1201",)):
        values = grid(NONZERO)
        values[0][3] = float("nan")
        assert constant(make_json(archs, values)) == "fail", archs
        waves = [[16384.0] * 5, [4096.0] * 5, [1024.0] * 5]
        waves[1][2] = float("nan")
        data = make_json(archs, grid(NONZERO), waves=waves)
        assert constant(data, "SQ_WAVES") == "fail", archs


def main():
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failures = []
    for name, func in tests:
        try:
            func()
        except Exception as exc:  # noqa: BLE001 - report every failure
            failures.append((name, exc))
            print(f"FAIL {name}: {exc}")
        else:
            print(f"ok   {name}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
