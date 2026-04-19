# BackendAdapter Protocol (multi-backend contract)

The `BackendAdapter` Protocol is the contract every `perfxpert-code
<backend>` adapter satisfies. Claude Code, Gemini CLI, and (PR 2)
Codex CLI each implement this Protocol; the dispatcher in
`perfxpert/cli/_backend_dispatch.py` routes subcommands into the
registered adapter by name.

This doc is aimed at contributors adding a fifth backend (or
updating an existing one). The user-facing install/uninstall surface
is documented in [../guides/backends.md](../guides/backends.md).

Source of truth: `perfxpert/cli/_backend/protocol.py`.

## Protocol interface

Every adapter declares six class attributes and implements six
methods. All signatures are locked from day one (plan cycle-2 I2) so
downstream consumers never need to mutate them.

```python
# SKIP-SAMPLE — Protocol summary; see perfxpert/cli/_backend/protocol.py for the real source
from typing import Literal, Protocol

class BackendAdapter(Protocol):
    # Class attributes
    name: str                         # routing key: "claude", "gemini", ...
    binary_name: str                  # shutil.which target
    install_hint: str                 # one-liner shown on check_available=False
    min_version: str | None           # SemVer floor; None disables the check
    known_schema_versions: tuple[str, ...]  # parseable config-schema versions
    tool_name_template: str           # e.g. "mcp__perfxpert__{tool}"
    spawn_strategy: Literal["execvpe", "subprocess"]

    # Lifecycle methods
    def check_available(self) -> tuple[bool, str]: ...
    def plan(self, cwd, scope="project", dry_run=True) -> Plan: ...
    def install(self, cwd, scope="project", allow_agents_md_append=False,
                dry_run=False, quiet=False) -> InstallReport: ...
    def verify_mcp_live(self, cwd, telemetry=False) -> LiveCheckReport: ...
    def uninstall(self, cwd, scope="project") -> UninstallReport: ...
    def spawn(self, argv, env, cwd) -> int: ...
```

The four report dataclasses (`Plan`, `InstallReport`, `UninstallReport`,
`LiveCheckReport`) are frozen — an adapter's return value cannot be
mutated by the dispatcher after it has been logged.

## Lifecycle

```
                check_available()
                        │
                        ▼
                     plan()  ◀──── (dry-run path exits here)
                        │
                        ▼
                    install()
                        │
                        ├─ 1. Consent (per backend × cwd × file-set hash)
                        ├─ 2. Gate-hook install   (BEFORE MCP — I-N1)
                        ├─ 3. MCP registration    (backend-native config)
                        ├─ 4. Stage AGENTS.md cache
                        └─ 5. verify_mcp_live()   (skippable via env)
                        │
                        ▼
                     spawn()   ◀──── execvpe (default) or subprocess
                        │
                        ▼
                  [backend TUI runs — control never returns on execvpe]

       uninstall()   ◀──── symmetric reverse of install()
```

The ordering constraint "gate-hook BEFORE MCP" (plan cycle-2 I-N1) is
load-bearing: if the gate-hook installer raises `GateHookUnsupported`
mid-install, there is no partial MCP registration to clean up.

## Per-backend state model

| Backend | MCP registration target | Prompt cache | Gate hook surface | Session state |
|---------|------------------------|--------------|-------------------|---------------|
| opencode (bundled) | `~/.cache/perfxpert/opencode/opencode.json` | Patched system prompt (AMD fork patches 0010+0020) | Fork patches 0010, 0012-0017 + STRICT-TOOL-DISCIPLINE stanza | N/A — state carried in opencode session |
| Claude Code | `<cwd>/.mcp.json` (project scope) | `<cwd>/.perfxpert/AGENTS.md` + `<cwd>/.claude/CLAUDE.md` pointer | Native `PreToolUse` hook in `<cwd>/.claude/settings.json` | `<cwd>/.claude/.perfxpert-gate-state.<session_id>.json` |
| Gemini CLI | `~/.gemini/settings.json` (user scope) | `<cwd>/.perfxpert/AGENTS.md` referenced via `context.fileName` list-append | `allowedTools` restriction in `~/.gemini/settings.json` | In-settings (session-ephemeral via `allowedTools`) |
| Codex CLI (PR 2) | `~/.codex/config.toml` (TBD) | TBD | TBD | TBD |

A NEW session (different `session_id`) always starts with the gate
engaged — even in the same cwd. The sidecar file is keyed on
`session_id` to enforce this naturally.

## Gate-hook contract

The gate-hook layer lives in `perfxpert/cli/_gate_hooks/` — one
module per backend. Every gate-hook installer satisfies three rules:

1. **Event-based lift.** The gate lifts once
   `perfxpert_intent_classify` has returned in the current session.
   The lift signal is a sidecar state file (Claude) or an
   `allowedTools` mutation driven by a companion `PostToolUse` path
   (Gemini). Static deny-lists alone are insufficient — they cannot
   lift mid-session.

2. **Raise `GateHookUnsupported` before MCP.** If the gate cannot be
   installed cleanly (pre-existing conflicting hook, invalid JSON,
   surface not available on this backend version, etc.), the
   installer MUST raise `GateHookUnsupported` BEFORE any MCP or
   prompt-cache write. The adapter's `verify_mcp_live()` then reports
   `gate_hook_installed=False` as a documented-known-limit rather
   than a failure.

3. **Reject non-perfxpert tool calls until lift.** Until
   `perfxpert_intent_classify` has been invoked in the current
   session, every non-`perfxpert_*` tool call is denied with a
   user-visible reason matching the opencode fork patch 0020
   `retryWith` message ("call `intent_classify` first"). This keeps
   the UX identical across backends.

The three `LiveCheckReport.gate_hook_installed` states encode these
outcomes: `None` = probe skipped, `False` = surface unsupported
(known limit), `True` = installed AND effective.

## Adding a new backend

Steps to add a fifth backend (example: `aider`):

1. **Write the adapter.** Create
   `perfxpert/cli/_backend/aider.py` with a class `AiderAdapter`
   satisfying the `BackendAdapter` Protocol. Declare all six class
   attributes + implement all six methods. Borrow the structure of
   `perfxpert/cli/_backend/claude.py` or `gemini.py` — both are
   self-contained and under 800 lines.

2. **Write the gate hook.** Create
   `perfxpert/cli/_gate_hooks/aider.py` with a class `AiderGateHook`
   that implements `install(cwd)` per the gate-hook contract above.
   If the backend exposes no suitable pre-tool-call surface, raise
   `GateHookUnsupported` from `install()` — the adapter treats this
   cleanly.

3. **Register with the dispatcher.** Edit
   `perfxpert/cli/_backend_dispatch.py` and add a `_aider_runner`
   function + entry in `BACKEND_REGISTRY`. Follow the existing
   pattern; the dispatcher's Task 6 install-then-spawn flow picks
   up the new entry without further changes.

4. **Wire up uninstall.** Edit
   `perfxpert/cli/opencode_launcher.py::_run_uninstall` to lazy-import
   `AiderAdapter` under the `aider` branch (mirrors the existing
   `claude` / `gemini` branches).

5. **Tests + docs.** Add `tests/test_cli/test_aider_adapter.py`
   mirroring the existing adapter tests. Update the backend
   comparison table in [../guides/backends.md](../guides/backends.md)
   and the per-backend state model table above in this file.

## References

- [../guides/backends.md](../guides/backends.md) — user-facing
  install/uninstall recipes
- [../integration/mcp-server.md](../integration/mcp-server.md) —
  the 34 READ_ONLY tools every adapter exposes to its backend
- [../../../../../docs/decisions/2026-04-19-claude-hook-surface.md](../../../../../docs/decisions/2026-04-19-claude-hook-surface.md)
  — the Claude PreToolUse decision record (why native hook, not
  `permissions.deny`)
- [../../../../../docs/superpowers/plans/2026-04-19-perfxpert-code-multi-backend-plan.md](../../../../../docs/superpowers/plans/2026-04-19-perfxpert-code-multi-backend-plan.md)
  — the 14-task implementation plan (PR 1 + PR 2 breakdown)
- Source: `perfxpert/cli/_backend/protocol.py`,
  `perfxpert/cli/_backend/{claude,gemini}.py`,
  `perfxpert/cli/_gate_hooks/{claude,gemini,opencode}.py`,
  `perfxpert/cli/_backend_dispatch.py`
