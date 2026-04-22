# Deprecation: `PERFXPERT_LEGACY=1`

**Status:** Deprecated as of v0.2.0.
**Removal target:** vX.Y+1 (next minor release).

## What it does

Setting `PERFXPERT_LEGACY=1` routes the perfxpert analyze pipeline through
the deprecated local-only code path. This exists as a one-version safety net
so users can roll back if the new agentic path has an unforeseen regression
in their specific workflow.

The variable also:
- Emits a `DeprecationWarning`
- Causes `perfxpert doctor` to report `Mode: legacy (DEPRECATED)`
- Disables LLM enhancement in legacy mode

## Why it's being removed

The legacy path comprises ~4600 lines of Python that duplicate functionality
already provided by the agentic runtime + bundled opencode. Maintaining both
paths indefinitely multiplies test surface, doubles CI cost, and delays bug
fixes. The one-minor-version safety net window is the project's commitment
to a soft landing; after that window, the legacy files are permanently gone
from the repo history (they will remain accessible via `git log` + any
release tag ≤ v0.2.x).

## Timeline

| Release | Status |
|---------|--------|
| v0.1.x | Legacy was default. `PERFXPERT_USE_AGENTS=1` was experimental opt-in. |
| v0.2.0 | Agentic is default. `PERFXPERT_LEGACY=1` is opt-in safety net. **You are here.** |
| vX.Y+1 (next minor) | `PERFXPERT_LEGACY=1` is a no-op. Deprecation warning → NotImplementedError. Legacy code paths in `api.py` deleted. |
| vX.Y+2 | `LLMAnalyzer` stub class deleted. |

## Action required

Migrate to the agentic default **before vX.Y+1**. See
[migration-to-agentic.md](../migration-to-agentic.md).

## If you cannot migrate

Open a GitHub issue with:
- The specific workflow or output that differs between legacy and agentic
- A reproducer DB (minimal fixture preferred)
- `perfxpert doctor` output from both modes
- Expected vs actual behavior

We treat legacy-vs-agentic differences as bugs in the agentic path and aim
to close them before removing the safety net.
