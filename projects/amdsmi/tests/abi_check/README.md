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
branch on develop PRs. On release PRs it uses the PR base, not a newer
release branch. Both also check the [shipped ABI](#shipped-abi). It fails on:

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
  on LLP64 Windows (typedefs included); in C++17 also a field type, function or
  macro that differs, or a function without C linkage

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

## Shipped ABI

`projects/amdsmi/abi/shipped.json` records the last shipped x86-64 Linux
system ABI for this branch: the release source commit, layouts (including enum
values), function signatures, constants, SONAME and versioned exports. It is
independent of the moving release branch and PR exceptions. A reserved carve or
new getter still passes; `acknowledge` cannot waive a shipped binary break. The
base/release comparisons still cover doc semantics, which the snapshot omits.

The header job reads the snapshot from git. A PR replacing it under the same
SONAME must preserve the base snapshot's obligations as well. Each manylinux
build uploads its ESMI-on **system** ELF. A fresh runner checks it
with `readelf`; it never loads the ELF or runs PR build scripts. On a PR, both
`abi_gate.py` and `shipped.json` come from the event's base SHA. On a push to
develop they come from the pushed SHA; a manual dispatch uses the default branch.
If a PR base does not yet track the baseline, the job reports that no ELF
comparison ran; it does **not** use the PR's baseline or checker instead. A
push or dispatch without a baseline fails. The workflows
target `develop` and `release/**` PRs. Older release branches need a backport
with their own last-shipped baseline. This does not configure required checks.

From the repository root, with the pinned `requirements.txt` installed:

```bash
gate=projects/amdsmi/tests/abi_check/abi_gate.py
python3 "$gate" shipped --new WORKTREE
python3 "$gate" shipped --library build/src/libamd_smi.so
```

**Reproduce or refresh.** Use the system ELF from the published package and the
matching rocm-systems pin, not the tip of a release branch. For this snapshot:

```bash
set -e
pkg=amdrocm-amdsmi10.2_10.2.0-2_amd64.deb
curl -fLO "https://stable.repo.amd.com/rocm/core/packages/deb/pool/main/$pkg"
echo "a607b7b9f21c3a50f0689ffff5fb4633a47e93a158f2e00f4e1f24db00a43221  $pkg" | sha256sum -c -
dpkg-deb -x "$pkg" extracted
root=extracted/opt/rocm/core-10.2
pin=43fa692e568e65a1fb84f2d1c74415b042839f10  # TheRock therock-10.2 gitlink
git show "$pin:projects/amdsmi/include/amd_smi/amdsmi.h" | cmp - "$root/include/amd_smi/amdsmi.h"
python3 "$gate" capture --release "$pin" --library "$root/lib/libamd_smi.so.27" > candidate.json
cmp candidate.json projects/amdsmi/abi/shipped.json
```

For a new release, review the diff before replacing `shipped.json`. Do not
regenerate it from a failing PR to make the check pass. For an intentional
major bump, change the library and header major together, document migration,
and acknowledge any PR/base breaks as below. Capture the new major's baseline
from the pinned release candidate in a follow-up commit on that PR; verify it
again against the promoted package. The PR ELF check will still report the
SONAME mismatch against the old base. Release owners must review and coordinate
that transition; changing the snapshot in the same PR is not a binary-check
override. Keep the old baseline on its maintenance branch. The old and new
contracts may differ only across the SONAME boundary.

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

`abi_gate_test.py` runs from pre-commit when these files change. To run its
workflow tests directly, also install `PyYAML` in the test environment.
