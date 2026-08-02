"""Isolation tests for Trace-Diff Specialist (Layer 2).

The diff verdict and per-kernel deltas are measurements, not opinions:
``trace_diff.diff_runs`` owns them and the model may only phrase them.
Most of these tests script a model response that contradicts the
measurement and assert the measurement wins.
"""

import pytest

from perfxpert.agents import diff_specialist as ds_module
from perfxpert.agents import schemas
from perfxpert.agents.framework import (
    FakeProviderResponse,
    HandoffPolicyViolation,
    dispatch_handoff,
)


# wall_delta_pct > +0.5 and a non-empty primary_regressions list both
# independently force verdict == "regressed".
_REGRESSED_DIFF = {
    "wall_delta_pct": 8.0,
    "primary_regressions": [
        {
            "name": "matmul",
            "baseline_ns": 100,
            "new_ns": 134,
            "delta_ns": 34,
            "delta_pct": 34.0,
            "regressed": True,
            "was_hot": True,
        }
    ],
    "primary_improvements": [],
}


@pytest.fixture
def stub_diff(monkeypatch):
    """Serve a fixed diff dict so no real .db file is needed."""

    def _install(diff_result=_REGRESSED_DIFF):
        monkeypatch.setattr(
            ds_module.trace_diff, "diff_runs", lambda **kw: diff_result
        )
        return diff_result

    return _install


def _payload() -> schemas.DiffSpecialistInput:
    return schemas.DiffSpecialistInput(baseline_db="base.db", new_db="new.db")


def test_diff_specialist_builds():
    agent = ds_module.build_diff_specialist()
    assert agent.name == "TraceDiffSpecialist"
    assert agent.layer == 2


def test_diff_specialist_tool_count_within_cap():
    agent = ds_module.build_diff_specialist()
    assert len(agent.tools) <= 5


def test_diff_specialist_no_execution_tools():
    agent = ds_module.build_diff_specialist()
    forbidden = {"patch.apply", "compile.build", "profile.run", "anchors.check"}
    declared = {t.name for t in agent.tools}
    assert not (declared & forbidden)


def test_diff_specialist_cannot_handoff_laterally():
    agent = ds_module.build_diff_specialist()
    with pytest.raises(HandoffPolicyViolation):
        dispatch_handoff(agent, "compute_specialist")
    with pytest.raises(HandoffPolicyViolation):
        dispatch_handoff(agent, "memory_specialist")


def test_model_cannot_override_verdict(stub_diff, fake_provider):
    """A model calling a measured regression an improvement is ignored."""
    stub_diff()
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"verdict": "improved", "narrative": "Looks great!"},
    )
    result = ds_module.run_diff_specialist(_payload(), provider="anthropic")
    assert result.verdict == "regressed"


def test_model_cannot_override_kernel_deltas(stub_diff, fake_provider):
    """Fabricated kernel deltas never displace the measured ones."""
    stub_diff()
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "regressions": [],
            "improvements": [{"name": "invented_kernel", "delta_pct": -99.0}],
            "narrative": "Everything improved.",
        },
    )
    result = ds_module.run_diff_specialist(_payload(), provider="anthropic")
    assert [k["name"] for k in result.kernel_deltas["regressions"]] == ["matmul"]
    assert result.kernel_deltas["improvements"] == []


def test_model_cannot_override_wall_delta(stub_diff, fake_provider):
    stub_diff()
    fake_provider.return_value = FakeProviderResponse(
        structured_output={"wall_delta_pct": -50.0, "narrative": "Much faster."},
    )
    result = ds_module.run_diff_specialist(_payload(), provider="anthropic")
    assert result.wall_delta_pct == 8.0


def test_model_still_supplies_narrative_and_confidence(stub_diff, fake_provider):
    """Freezing the arithmetic must not mute the model's prose."""
    stub_diff()
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "narrative": "matmul regressed 34%, now memory-bound.",
            "confidence": 0.9,
        },
    )
    result = ds_module.run_diff_specialist(_payload(), provider="anthropic")
    assert result.narrative == "matmul regressed 34%, now memory-bound."
    assert result.confidence == 0.9


def test_missing_narrative_falls_back_to_template(stub_diff, fake_provider):
    stub_diff()
    fake_provider.return_value = FakeProviderResponse(structured_output={})
    result = ds_module.run_diff_specialist(_payload(), provider="anthropic")
    assert result.narrative == ds_module._airgap_narrative(_REGRESSED_DIFF)


def test_live_and_airgap_agree_on_measured_fields(stub_diff, fake_provider):
    """The air-gap parity invariant, stated as a test."""
    stub_diff()
    fake_provider.return_value = FakeProviderResponse(
        structured_output={
            "verdict": "improved",
            "regressions": [],
            "improvements": [{"name": "invented_kernel"}],
            "narrative": "Contradicts the measurement.",
        },
    )
    live = ds_module.run_diff_specialist(_payload(), provider="anthropic")
    airgapped = ds_module.run_diff_specialist(_payload(), airgap=True)

    assert live.verdict == airgapped.verdict
    assert live.wall_delta_pct == airgapped.wall_delta_pct
    assert live.kernel_deltas == airgapped.kernel_deltas


@pytest.mark.parametrize(
    "wall_delta_pct, expected",
    [(8.0, "regressed"), (-8.0, "improved"), (0.2, "neutral")],
)
def test_verdict_thresholds(stub_diff, fake_provider, wall_delta_pct, expected):
    stub_diff({
        "wall_delta_pct": wall_delta_pct,
        "primary_regressions": [],
        "primary_improvements": [],
    })
    fake_provider.return_value = FakeProviderResponse(structured_output={})
    result = ds_module.run_diff_specialist(_payload(), provider="anthropic")
    assert result.verdict == expected
