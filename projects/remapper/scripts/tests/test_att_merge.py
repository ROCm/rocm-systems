import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "att_merge"))
import align
import barrier_sync
import phase
from run import assemble_summary


def test_anchor_interpolation_and_offsets():
    a = {"a": 0, "b": 10, "c": 20}
    b = {"a": 4, "b": 16, "c": 28}
    keys, x, d = align.offsets(a, a, b)
    assert keys == ["a", "b", "c"]
    np.testing.assert_allclose(d, [4, 6, 8])
    np.testing.assert_allclose(
        align.warp(a, a, b)(np.array([-1, 5, 15, 25])), [3, 10, 22, 33]
    )


def test_same_role_barrier_spread(tmp_path):
    code = {"code": [["s_barrier_wait 0xfffd", 0, 1], ["s_barrier_wait -1", 0, 2]]}
    (tmp_path / "code.json").write_text(json.dumps(code))
    uids = {}
    for uid in range(8):
        fname = f"se0_sm{uid % 4}_sl{uid}_wv0.json"
        uids[fname] = uid
        # Roles intentionally differ by 1000 clocks; they must never be pooled together.
        base = (uid // 4) * 1000 + uid % 4
        wave = dict(
            simd=uid % 4, instructions=[[base, 0, 0, 2, 1], [base + 100, 0, 0, 2, 2]]
        )
        (tmp_path / fname).write_text(json.dumps({"wave": wave}))
    (tmp_path / "merge_provenance.json").write_text(json.dumps({"wave_uids": uids}))
    result = barrier_sync.summarize([tmp_path], 8)
    assert result["incomplete_groups"] == 0
    assert len(result["groups"]) == 4
    assert all(v["median"] == 3 and v["p90"] == 3 for v in result["groups"].values())


def test_phase_compares_against_both_baselines():
    def mode(k_loop):
        m = {
            k: {"median": 100}
            for k in ("remap_prologue", "body_prologue", "epilogue", "wave")
        }
        m["k_loop"] = {"median": k_loop}
        return m

    res = phase.add_comparisons(
        {"original": mode(200), "natural": mode(210), "table": mode(220)}
    )
    assert res["table"]["percent_vs_original"]["k_loop"] == 10.0
    assert res["table"]["percent_vs_natural"]["k_loop"] == round(
        100 * (220 / 210 - 1), 3
    )
    assert res["natural"]["percent_vs_original"]["k_loop"] == 5.0
    assert "percent_vs_original" not in res["original"]


def test_summary_fails_on_missing_identity_or_sequence():
    same = dict(
        se_simd_pairs=4,
        waves=8,
        unidentified_waves=0,
        same_wave_identities_every_sweep=4,
        identical_instruction_sequence_every_sweep=8,
    )
    audit = dict(
        same_work=same,
        merged=[dict(unidentified_waves=0, waves_with_timestamp_problems=0)],
    )
    alignment = {
        "methods": {"occupancy + barriers": {"barrier": {"median": 0, "p90": 13}}}
    }
    partner = dict(
        checks={},
        methods={"same sweep | occupancy + barriers": {"category 5": {"abs_p90": 10}}},
    )
    assert assemble_summary(alignment, partner, audit, {}, {}, ["ui"])["valid"]
    same["identical_instruction_sequence_every_sweep"] = 7
    failed = assemble_summary(alignment, partner, audit, {}, {}, ["ui"])
    assert not failed["valid"] and "--no-sequence-check" in failed["reason"]
    assert assemble_summary(
        alignment, partner, audit, {}, {}, ["ui"], sequence_check=False
    )["valid"]
    same["same_wave_identities_every_sweep"] = 3
    assert not assemble_summary(
        alignment, partner, audit, {}, {}, ["ui"], sequence_check=False
    )["valid"]


def test_a2_marker_flags_fake_residency_without_problems():
    import extract

    waves = [dict(cu=0, simd=s, slot=0, begin=0, end=100) for s in range(3)]
    markers = [(50, 0xA1000005, 0, 0, 0, 0), (50, 0xA2000001, 0, 1, 0, 0)]
    problems = extract.identify(waves, markers)
    assert waves[0]["uid"] == 5 and "fake" not in waves[0]
    assert waves[1]["uid"] is None and waves[1]["fake"]
    # Only the residency with no marker at all is a problem.
    assert dict(problems) == {"unidentified_residency": 1}


def test_late_a2_marker_belongs_to_the_wave_that_just_ended():
    import extract

    waves = [
        dict(cu=0, simd=0, slot=0, begin=0, end=100),
        dict(cu=0, simd=0, slot=0, begin=1000, end=1100),
        dict(cu=0, simd=0, slot=1, begin=0, end=100),
        dict(cu=0, simd=0, slot=1, begin=150, end=900),
    ]
    markers = [
        (160, 0xA2000000, 0, 0, 0, 0),  # 60 clocks after slot 0's first wave ended
        (160, 0xA2000000, 0, 0, 1, 0),
    ]  # slot 1's next wave already began: ambiguous
    problems = extract.identify(waves, markers)
    assert waves[0].get("fake") and not waves[1].get("fake")
    assert not waves[2].get("fake") and not waves[3].get("fake")
    assert problems["ambiguous_or_boundary_marker"] == 1
