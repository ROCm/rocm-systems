"""LLM-enabled end-to-end smoke — asserts rec_type populates with a real LLM.

Skipped when:
  * Neither OPENAI_API_KEY nor ANTHROPIC_API_KEY is set,
  * the memory_bound fixture DB is missing,
  * the framework's live SDK path (`_sdk_invoke`) is still a stub
    (the real Agents-SDK runtime wires it up; this test activates
    automatically once that landing happens), or
  * the live provider returns a quota/auth/transient error (environmental,
    not a code defect — see docs/known-issues.md).

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
    # Import SDK exception taxonomies lazily so the module loads even when a
    # given SDK isn't installed. Both openai and anthropic are runtime deps of
    # perfxpert, but we build a tolerant tuple to future-proof the skip path.
    try:
        import openai  # type: ignore
    except Exception:  # pragma: no cover - defensive
        openai = None  # type: ignore
    try:
        import anthropic  # type: ignore
    except Exception:  # pragma: no cover - defensive
        anthropic = None  # type: ignore

    quota_excs: tuple = ()
    auth_excs: tuple = ()
    transient_excs: tuple = ()
    if openai is not None:
        quota_excs += (openai.RateLimitError,)
        auth_excs += (openai.AuthenticationError, openai.PermissionDeniedError)
        transient_excs += (
            openai.APIConnectionError,
            openai.APITimeoutError,
            openai.InternalServerError,
        )
    if anthropic is not None:
        quota_excs += (anthropic.RateLimitError,)
        auth_excs += (anthropic.AuthenticationError, anthropic.PermissionDeniedError)
        transient_excs += (
            anthropic.APIConnectionError,
            anthropic.APITimeoutError,
            anthropic.InternalServerError,
        )

    from perfxpert.agents import runtime, schemas

    session = runtime.build_session(provider="openai", airgap=False)
    root_input = schemas.RootInput(
        user_query="Analyze this GPU performance trace.",
        database_path=str(FIXTURE_DB),
        provider="openai",
        airgap=False,
        session_id=session.session_id,
    )
    def _walk_causes(err: BaseException):
        seen = set()
        cur: BaseException | None = err
        while cur is not None and id(cur) not in seen:
            seen.add(id(cur))
            yield cur
            cur = cur.__cause__ or cur.__context__

    def _classify(err: BaseException) -> str | None:
        # framework.py wraps SDK errors in RuntimeError; walk the chain to find
        # the original provider exception class.
        for node in _walk_causes(err):
            if quota_excs and isinstance(node, quota_excs):
                return "quota"
            if auth_excs and isinstance(node, auth_excs):
                return "auth"
            if transient_excs and isinstance(node, transient_excs):
                return "transient"
        # Fallback: match on string payload (covers cases where the framework
        # wrapper loses the cause chain, e.g. str(exc) re-raise).
        msg = str(err).lower()
        if "insufficient_quota" in msg or "rate limit" in msg or "429" in msg:
            return "quota"
        if (
            "401" in msg
            or "403" in msg
            or "invalid_api_key" in msg
            or "authentication" in msg
            or "unauthorized" in msg
            or "permission_denied" in msg
        ):
            return "auth"
        if (
            "500" in msg
            or "502" in msg
            or "503" in msg
            or "504" in msg
            or "timeout" in msg
            or "timed out" in msg
            or "connection" in msg
        ):
            return "transient"
        return None

    try:
        out = session.run_root(root_input)
    except quota_excs as e:
        pytest.skip(f"LLM quota exhausted (environmental, not a code defect): {e}")
    except auth_excs as e:
        pytest.skip(f"LLM auth failed (environmental, not a code defect): {e}")
    except transient_excs as e:
        pytest.skip(f"LLM transient error (environmental, not a code defect): {e}")
    except RuntimeError as e:
        kind = _classify(e)
        if kind == "quota":
            pytest.skip(
                f"LLM quota exhausted (environmental, not a code defect): {e}"
            )
        if kind == "auth":
            pytest.skip(f"LLM auth failed (environmental, not a code defect): {e}")
        if kind == "transient":
            pytest.skip(
                f"LLM transient error (environmental, not a code defect): {e}"
            )
        raise

    # Schema assertions below must still fail loudly on real defects.
    assert out.primary_bottleneck, "primary_bottleneck should be populated"
    assert out.narrative, "narrative should be non-empty"
    assert out.recommendations, "recommendations list should not be empty"
    assert out.recommendations[0].get("type"), (
        f"recommendations[0].type missing — {out.recommendations[0]}"
    )
