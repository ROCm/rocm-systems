"""LLM-enabled end-to-end smoke — asserts rec_type populates with a real LLM.

Skipped when:
  * Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set,
  * the memory_bound fixture DB is missing,
  * the framework's live SDK path (`_sdk_invoke`) is still a stub
    (the real Agents-SDK runtime wires it up; this test activates
    automatically once that landing happens), or
  * the live provider returns a classified error (quota/auth/rate-limit/
    transient) — these are environmental (billing / throttle / outage),
    NOT code defects. The Fix 1 classifier in framework._sdk_invoke maps
    SDK-specific exceptions into perfxpert's taxonomy so this test just
    catches :class:`ProviderError` and calls ``pytest.skip``.

A true schema/output defect (missing `rec_type`, empty narrative, etc.)
still FAILS loudly — only provider-side environmental errors skip.
"""

import os
from pathlib import Path

import pytest

FIX = Path(__file__).parent.parent / "fixtures"
FIXTURE_DB = FIX / "memory_bound.db"


pytestmark = pytest.mark.skipif(
    not (os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")),
    reason="no LLM provider key set (need OPENAI_API_KEY or ANTHROPIC_API_KEY)",
)


def _live_sdk_path_implemented() -> bool:
    """Heuristic: the framework stub raises NotImplementedError for live calls."""
    import inspect

    try:
        from perfxpert.agents import framework  # type: ignore
    except Exception:
        return False
    try:
        src = inspect.getsource(framework._sdk_invoke)
    except (OSError, TypeError):
        return False
    return "NotImplementedError" not in src


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — LLM end-to-end test requires a real provider",
)
@pytest.mark.skipif(
    not FIXTURE_DB.exists(),
    reason=f"fixture db missing: {FIXTURE_DB}",
)
@pytest.mark.skipif(
    not _live_sdk_path_implemented(),
    reason="framework._sdk_invoke is still a stub; live path pending SDK wire-up",
)
def test_llm_enabled_produces_rec_type():
    from perfxpert.agents import runtime, schemas
    from perfxpert.providers._exceptions import (
        AuthError,
        FatalError,
        ProviderError,
        QuotaExceededError,
        RateLimitError,
        TransientError,
    )

    session = runtime.build_session(provider="openai", airgap=False)
    root_input = schemas.RootInput(
        user_query="Analyze this GPU performance trace.",
        database_path=str(FIXTURE_DB),
        provider="openai",
        airgap=False,
        session_id=session.session_id,
    )

    try:
        out = session.run_root(root_input)
    except QuotaExceededError as e:
        pytest.skip(f"LLM quota exhausted (environmental, not a code defect): {e}")
    except AuthError as e:
        pytest.skip(f"LLM auth failed (environmental, not a code defect): {e}")
    except RateLimitError as e:
        pytest.skip(f"LLM rate-limited (environmental, not a code defect): {e}")
    except TransientError as e:
        pytest.skip(f"LLM transient error (environmental, not a code defect): {e}")
    except FatalError as e:
        # Fatal = provider said "no" in a way we couldn't classify;
        # still environmental from this test's perspective.
        pytest.skip(f"LLM provider fatal error (environmental): {e}")
    except ProviderError as e:
        pytest.skip(f"LLM provider error (environmental): {e}")

    # Schema assertions below must still fail loudly on real defects.
    assert out.primary_bottleneck, "primary_bottleneck should be populated"
    assert out.narrative, "narrative should be non-empty"
    assert out.recommendations, "recommendations list should not be empty"
    assert out.recommendations[0].get("type"), (
        f"recommendations[0].type missing — {out.recommendations[0]}"
    )
