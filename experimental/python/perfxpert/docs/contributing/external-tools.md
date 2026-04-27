# Contributing: external-tool dependencies

PerfXpert talks to external binaries and shared libraries (opencode,
rocprofv3, amdclang++, the ROCprof Trace Decoder, …). Every one of
them goes through a single helper — `perfxpert.tools._tooldep` — so
that missing-tool errors surface consistently, install hints are a
one-liner to update, and air-gap / CI configs have one place to
override paths.

This guide explains how to use the helper from your own tool code and
how to register a new external dependency.

## What's already wired

The helper keeps a canonical registry (`_TOOL_REGISTRY`) of every
external the repo touches. Current entries:

| Name | Kind | Where it comes from |
|------|------|---------------------|
| `opencode` | binary | `perfxpert-code` bundled TUI |
| `rocprofv3` | binary | ROCm; used to collect traces |
| `amdclang++` | binary | ROCm; used by the Compile gate |
| `rocprof-trace-decoder` | shared_lib | ATT / SQTT decoder library |
| `mcp` | pylib | MCP SDK for `perfxpert-mcp` |
| `pexpect` | pylib | Optional pty driver for runners |
| `anthropic` | pylib | Provider adapter |
| `openai` | pylib | Provider adapter |

Entries live in `perfxpert/tools/_tooldep.py`.

## Calling `require_tool()` from a tool

Any tool that shells out or imports an optional dependency should call
`require_tool(name)` first. The helper raises `ExternalToolMissing`
with a human-readable install hint if the dep isn't available; your
tool never has to hand-craft a "not found" error message.

```python
from perfxpert.tools._tooldep import require_tool, ExternalToolMissing

def lookup_trace_decoder_version() -> str:
    try:
        require_tool("rocprof-trace-decoder")
    except ExternalToolMissing as e:
        # Fail soft — e.install_hint is the one-liner users can copy/paste.
        return f"rocprof-trace-decoder: unavailable ({e.install_hint})"
    # Safe to proceed; helper verified the library is on disk.
    return "rocprof-trace-decoder: found"

# Executable sanity check: invoking on an unregistered name raises
# immediately and doesn't talk to PATH.
import pytest
with pytest.raises(Exception):
    require_tool("not-a-registered-dep")
```

`require_tool` does three things in order:

1. Looks up the registry entry for `name`.
2. Dispatches to one of three detectors:
   - `binary` — `shutil.which(name)` + optional smoke-test. The
     smoke-test is a configurable `Optional[List[str]]` on the `_Dep`
     entry (see `_tooldep.py:52`); the helper runs `<binary> <args>`
     and treats a non-zero exit as a failed smoke-test. At present
     only `opencode` registers a smoke-test (`["--version"]`); other
     registered binaries rely on `shutil.which` alone.
   - `pylib` — `importlib.import_module(name)`
   - `shared_lib` — ROCm-aware search across `${ROCM_PATH}/lib*`,
     `${PERFXPERT_<NAME>_PATH}`, `~/.local/opt/<name>/opt/rocm/lib*`,
     and `LD_LIBRARY_PATH`.
3. Returns a detail string on success; raises `ExternalToolMissing`
   with `install_hint` + optional `install_cmd` otherwise.

Environment overrides (per-tool):

```
PERFXPERT_ROCPROF_TRACE_DECODER_PATH   → shared-lib search path
PERFXPERT_OPENCODE_PATH                → explicit binary path
PERFXPERT_ALLOW_INSTALL=1              → unlocks offer_install prompt
```

## Registering a new dependency

1. Add a `_Dep` entry to `_TOOL_REGISTRY`. Pick the right `kind`
   (`binary`, `pylib`, `shared_lib`) and provide an `install_hint`
   that a user can copy/paste.

2. If the dep has a one-shot installer that's safe to run without
   sudo, set `install_cmd` so `offer_install()` can drive it after a
   `PERFXPERT_ALLOW_INSTALL=1` prompt.

3. For `binary` deps, add a `smoke_test` (e.g. `["--version"]`) so a
   broken wrapper script doesn't look healthy.

4. Add a unit test under `tests/test_tools/test_tooldep_<name>.py`
   that asserts `require_tool("<name>")` raises when the binary is
   missing. Monkeypatch `shutil.which` / `importlib.import_module` to
   simulate absence.

5. Update the table above.

## Full worked example — the ATT decoder

The ROCprof Trace Decoder is a shared library bundled with
`rocprof-trace-decoder-manylinux-2.28-X.Y.Z-Linux.sh`. PerfXpert uses
it under Tier 3 (instruction-level analysis). Registering it looked
like this:

```python
# SKIP-SAMPLE — shown as an illustrative diff from _tooldep.py
from perfxpert.tools._tooldep import _Dep, _TOOL_REGISTRY

_TOOL_REGISTRY["rocprof-trace-decoder"] = _Dep(
    kind="shared_lib",
    shared_lib_filename="librocprof-trace-decoder.so",
    install_hint=(
        "ATT requires the ROCprof Trace Decoder library. Download "
        "rocprof-trace-decoder-manylinux-2.28-X.Y.Z-Linux.sh from the "
        "AMD ROCm releases and install under /opt/rocm."
    ),
)
```

Because `kind="shared_lib"`, the detector searches:

1. `${ROCM_PATH}/lib`, `${ROCM_PATH}/lib64`,
   `${ROCM_PATH}/lib/rocprofiler-sdk`
2. `${PERFXPERT_ROCPROF_TRACE_DECODER_PATH}` (file or dir)
3. `~/.local/opt/rocprof-trace-decoder/opt/rocm/lib*`
4. `$LD_LIBRARY_PATH`

If none of those contain `librocprof-trace-decoder.so`, the tool that
called `require_tool("rocprof-trace-decoder")` gets the exact install
hint above — no guessing about which ROCm subdirectory to look in.

Once registered, ATT-using tools never reimplement the lookup. This is also
the contract tests should follow: ATT availability is defined by whether the
runtime can resolve `librocprof-trace-decoder.so`, not by whether a Python
wrapper module imports in the current environment.

```python
from perfxpert.tools._tooldep import require_tool

def run_att_analysis(db_path: str) -> dict:
    # SKIP-SAMPLE — elides the actual ATT invocation
    require_tool("rocprof-trace-decoder")
    # ... call decoder, parse stall CSV, return classified stalls ...
    return {}
```

The user who's missing the decoder gets a pointed error on the very
first ATT call, not a generic `OSError` from deep inside the decoder
shim.

## When to bypass the helper

Almost never. If you think you need to, open an issue first — the
usual cause is a dep that's trivially optional (e.g. a test-only
import), in which case a plain `try: import foo except ImportError:`
is fine. The helper exists specifically so users and CI get a
*consistent* failure surface.
