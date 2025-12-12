"""Cycle-4 B1 — tool-priority gate / prompt-layer enforcement.

The cycle-4 blocker report observed a real `perfxpert-code run` session
emitting 17 tool calls, ZERO of which were `perfxpert_*` — the first
five were `bash, bash, read, todowrite, edit`. The 8-prompt stanza
patches from cycle-2 had no observable runtime effect.

This module tests the *weaker-variant* tool-gate ships in cycle-4:

  1. MCP tool descriptions lead with the ALL-CAPS bracketed imperative
     `[MUST BE CALLED FIRST FOR GPU-PERF QUERIES]` (strengthens the
     priority hint from cycle-2).
  2. The `0020-perfxpert-tool-gate.patch` appends a **TOOL GATE
     ENFORCEMENT** block to every primary opencode prompt, documenting
     the retry protocol and the `PERFXPERT_DISABLE_TOOL_GATE=1` escape.

These are structural / static-content checks — a real runtime tool-gate
hook is tracked as a follow-up in ``docs/known-issues.md``. The tests
here are the contract for the weaker variant; if any of the literal
strings below change, the gate's behaviour regresses silently.
"""

from __future__ import annotations

from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]
PATCH_DIR = REPO / ".patches"
PATCH_FILE = PATCH_DIR / "0020-perfxpert-tool-gate.patch"


# ------------------------------------------------------------------
# Schema-layer gate: MCP tool descriptions
# ------------------------------------------------------------------


def test_mcp_tool_description_leads_with_allcaps_gate_bracket() -> None:
    pytest.importorskip("mcp")
    from mcp_server._registry import discover_read_only_tools
    from mcp_server.server import _fn_to_tool_schema

    tools = discover_read_only_tools()
    assert tools, "no READ_ONLY MCP tools discovered"

    # Every single tool must lead with the bracketed gate marker.
    # Missing = regression: live-scenario D showed LLMs ignoring plain
    # sentence-form priority hints.
    for name, fn in tools.items():
        schema = _fn_to_tool_schema(name, fn)
        desc = schema.description  # type: ignore[attr-defined]
        assert desc.startswith(
            "[MUST BE CALLED FIRST FOR GPU-PERF QUERIES]"
        ), (
            f"{name!r} MCP tool description lost the cycle-4 B1 gate "
            f"bracket — got: {desc[:80]!r}"
        )


# ------------------------------------------------------------------
# Prompt-layer gate: 0020-perfxpert-tool-gate.patch
# ------------------------------------------------------------------


def test_tool_gate_patch_file_present() -> None:
    assert PATCH_FILE.exists(), (
        f"0020-perfxpert-tool-gate.patch missing at {PATCH_FILE}"
    )


def test_tool_gate_patch_covers_all_primary_prompts() -> None:
    """The 0020 patch must touch all 8 primary prompt files.

    Coverage parity with 0002 + 0010 + 0012-0017 (see .patches/README.md
    "Coverage note" section). If a future opencode bump renames a primary
    prompt, this test will fail, flagging that 0020 needs an update.
    """
    text = PATCH_FILE.read_text()
    expected_prompts = [
        "default.txt",
        "anthropic.txt",
        "gpt.txt",
        "gemini.txt",
        "kimi.txt",
        "codex.txt",
        "beast.txt",
        "trinity.txt",
    ]
    missing = [
        name
        for name in expected_prompts
        if f"src/session/prompt/{name}" not in text
    ]
    assert not missing, (
        f"0020 does not patch these primary prompt files: {missing}"
    )


def test_tool_gate_patch_documents_escape_hatch() -> None:
    text = PATCH_FILE.read_text()
    assert "PERFXPERT_DISABLE_TOOL_GATE" in text, (
        "cycle-4 B1 requires a PERFXPERT_DISABLE_TOOL_GATE escape hatch "
        "documented in the tool-gate prompt"
    )


def test_tool_gate_patch_mentions_forbidden_first_tools() -> None:
    """The gate must name the six first-turn-forbidden tools by name.

    This is the literal list observed in live-scenario D (cycle-4 report):
    bash, read, glob, grep, edit, todowrite. If an LLM emits any of
    these as the first tool call for a GPU-perf query, the gate fires.
    """
    text = PATCH_FILE.read_text()
    for tool in ("bash", "read", "glob", "grep", "edit", "todowrite"):
        assert f"`{tool}`" in text, (
            f"tool-gate patch does not mention forbidden first tool "
            f"{tool!r} — regression from cycle-4 B1 live validation"
        )


def test_tool_gate_patch_mentions_intent_classify() -> None:
    """The gate must redirect to intent_classify as the canonical first call."""
    text = PATCH_FILE.read_text()
    assert "intent_classify" in text, (
        "tool-gate patch must redirect non-perfxpert first calls to "
        "intent_classify (the workflow's entry point)"
    )


# ------------------------------------------------------------------
# Simulated tool-call sequence: gate rejects `bash` first
# ------------------------------------------------------------------


def _simulated_gate(tool_calls: list[str], *, gate_disabled: bool = False) -> list[dict]:
    """Reference implementation of the tool-gate's intended semantics.

    This is the contract the prompt-layer gate enforces *via LLM
    self-correction* (weaker variant) and the follow-up runtime hook
    will enforce *mechanically* (future).

    The implementation itself lives in the opencode session layer (and
    in the prompt instructions). This function exists so tests can
    assert the SEMANTIC contract without wiring up a live opencode
    process.
    """
    if gate_disabled:
        return [{"call": c, "gate": "bypassed"} for c in tool_calls]

    PERFXPERT_PREFIXES = (
        "perfxpert_",
        "intent_",
        "workflow_",
        "bottleneck_",
        "roofline_",
        "counters_",
        "att_",
        "regression_",
        "tasks_",
    )
    result: list[dict] = []
    gate_active = True
    for idx, call in enumerate(tool_calls):
        is_perfxpert = any(call.startswith(p) for p in PERFXPERT_PREFIXES)
        if gate_active and not is_perfxpert:
            result.append(
                {
                    "call": call,
                    "gate": "rejected",
                    "retry": (
                        "PERFXPERT_TOOL_PRIORITY: you must call "
                        "perfxpert_intent_classify first. "
                        "Please retry with perfxpert_* tools only."
                    ),
                }
            )
            continue
        # First perfxpert call lifts the gate.
        if is_perfxpert and gate_active:
            gate_active = False
        result.append({"call": call, "gate": "ok"})
    return result


def test_gate_simulation_rejects_bash_first() -> None:
    """Contract: LLM's first `bash` call gets a rejection + retry instruction."""
    proposed = ["bash", "read", "todowrite", "edit"]
    out = _simulated_gate(proposed)
    assert out[0]["gate"] == "rejected"
    assert "perfxpert_intent_classify first" in out[0]["retry"]


def test_gate_simulation_accepts_perfxpert_first_then_lifts() -> None:
    """After intent_classify, the gate lifts and later bash is allowed."""
    proposed = ["intent_classify", "workflow_next_step", "bash"]
    out = _simulated_gate(proposed)
    assert out[0]["gate"] == "ok"
    assert out[1]["gate"] == "ok"
    # bash AFTER a perfxpert call is permitted — the gate is for FIRST-turn.
    assert out[2]["gate"] == "ok"


def test_gate_simulation_env_var_disables_gate() -> None:
    """PERFXPERT_DISABLE_TOOL_GATE=1 short-circuits the gate entirely."""
    proposed = ["bash", "read", "todowrite"]
    out = _simulated_gate(proposed, gate_disabled=True)
    for entry in out:
        assert entry["gate"] == "bypassed"


def test_gate_simulation_all_perfxpert_path_never_rejects() -> None:
    """Happy path: perfxpert_* first call, then whatever the workflow wants."""
    proposed = [
        "perfxpert_intent_classify",
        "perfxpert_workflow_next_step",
        "perfxpert_bottleneck_classify",
    ]
    out = _simulated_gate(proposed)
    assert all(e["gate"] == "ok" for e in out)
