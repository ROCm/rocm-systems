# amdsmi ABI checks

Two checks compare `include/amd_smi/amdsmi.h` across commits:

| | `abi_check.py` | `abi_gate.py` |
|---|---|---|
| Workflow | `ABI Compliance Check` | `AMDSMI ABI Gate` |
| Tool | abi-compliance-checker 2.3 | libclang layout snapshot |
| Blocks merge | no, labels `MAJOR/MINOR ABI BREAKAGE` | yes, once required in the ruleset |
| Override | none | an entry in `projects/amdsmi/abi/abi_exceptions.json` and `CHANGELOG.md` |

Callers allocate most amdsmi structs and resolve symbols by name (`dlopen`, ctypes,
cgo), so any change to a shipped layout corrupts memory in programs built against
the old header instead of failing to load.

## What the gate fails on

`abi_gate.py gate` parses amdsmi.h as C11 on x86_64 Linux with `ENABLE_ESMI_LIB`
defined, and compares the PR against its base and the highest `release/therock-*`
branch, both read from git. Nothing about the header is checked in. It fails on:

- a public type removed, or changed in size or alignment
- a field removed, or changed in offset, size or type
- an enumerator removed or renumbered, or a new one reusing a removed value or a
  value an acknowledged break retired
- a function removed, made `static`, or changed in return or parameter types
- an integer `#define` removed or changed, except `AMDSMI_LIB_VERSION_*`, and any
  change to a function-like macro
- a field or parameter whose doc comment gains a unit, sentinel value or bitmask
  encoding, or a direction that now includes `IN`
- a header the gate cannot fully see: a non-standard `#include`, an `#if` on an
  unknown macro, or a layout that differs in C++17, without `ENABLE_ESMI_LIB`, or
  on LLP64 Windows; in C++17 also a field type or function that differs, or a
  function without C linkage

Additions pass: new functions, types, enumerators and macros, and fields carved
from a `reserved*`/`rsvd*`/`pad*`/`unused*` array or bitfield when the struct
keeps its size. A `*_LAST`, `*_MAX`, `*_COUNT` or `*_END` enumerator may grow when
it holds the enum's highest value, ignoring markers such as `*_INVALID`, and is not
a `*_MAX` next to a `*_MIN`. Changed doc comments and `static inline`
bodies are listed in the job summary without failing.

A struct that starts with a `struct_size` the caller sets, and that the library
writes only up to that size, may be listed under `append_only` in
`projects/amdsmi/abi/abi_exceptions.json`. It may then grow at the end; its
existing fields still may not move. The gate reads this list from the PR's base,
so a PR cannot exempt a struct it grows.

gcc and g++ also compile static asserts of the base's sizes, offsets and values
against the PR's header, skipping the breaks the PR acknowledges.

## When a break is intended

```bash
python3 -m venv .venv && .venv/bin/pip install -r projects/amdsmi/tests/abi_check/requirements.txt
.venv/bin/python projects/amdsmi/tests/abi_check/abi_gate.py acknowledge \
    --base origin/develop --release auto --why "#<PR>: <what changed and why>"
```

This appends one line per break to `projects/amdsmi/abi/abi_exceptions.json`
with the gate's path and the exact old and new values, for example:

```json
{"path": "types/amdsmi_fabric_info_t/size", "old": 320, "new": 552, "why": "#11741: Ppod/Vpod config"}
```

Commit it together with a `CHANGELOG.md` line that names the type or function
and says `ABI` or `breaking`. Against the base, an entry acknowledges only that
change and only in the PR that adds it; a PR that makes the same change again
later needs its own entry. Entries stay as history: against the release branch the gate accepts any
chain of entries from the release's value to the new one. Entries the latest
release already contains may be deleted, except enumerator entries, which keep
their old values from being reused; the gate fails if one is dropped.

A meaning the doc comment cannot express goes under `notes`, keyed like the
semantic paths (`"field:amdsmi_x_t.value": "0 means unsupported since 10.3"`).
Adding or changing a note or an `append_only` entry also needs a `CHANGELOG.md`
line that names it.

Both commands compare against the merge base of `--base` and your branch, so a
branch behind develop does not see develop's newer changes as its own. Check a
change locally:

```bash
.venv/bin/python projects/amdsmi/tests/abi_check/abi_gate.py gate \
    --base origin/develop --new WORKTREE --release auto
```

`abi_gate_test.py` runs from pre-commit when these files change.
