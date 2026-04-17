# Salmon transpiler — lit tests

This directory holds FileCheck-style lit tests for the AMDGPU binary
transpiler (Salmon). Unlike the gtest suite under `../tests/`, these
tests exercise the raiser at the IR level: they run `raise_cli
--emit-ir` on a small HSA code object and check either the emitted LLVM
IR on stdout or the raiser's diagnostic output on stderr.

Lit is well-suited for the kinds of properties we want to assert here:

*   Structural invariants of the raised IR — e.g. "every
    side-effectful instruction under divergent EXEC is wrapped in an
    SPE `spe_do` / `spe_skip` diamond keyed on `%spe_lane_active`".
*   Negative / diagnostic assertions — e.g. "the pre-translation abort
    gate fires when the code object contains an EXEC-writer whose
    SemOp is not on the SPE allow-list", or "cross-wave translation of
    an EXEC-manipulating kernel emits a modulo-replication warning".
*   Per-SemOp audits — asserting that every allow-listed EXEC-writer
    actually routes its write through `storeExec`, visible as the new
    EXEC SSA value being used by the next `lshr i64 %exec, %spe_lane_mod`.

The existing gtest suite still owns the *runtime* checks (the raised
kernel is linked and dispatched on a real GCN device, and the output
buffer is compared against hardware ground truth). Lit tests
complement that by giving us principled, fast, non-GPU IR assertions
that catch regressions in the raiser *before* they manifest as
hardware mismatches.

## Layout

Each test lives in its own subdirectory so that fixtures, CHECK
scripts, and expected outputs stay together:

```
lit_tests/
├── abort_gate/          — SPE allow-list abort gate fires on unknown EXEC-writer
├── cross_wave_warn/     — cross-wave translation emits modulo-replication warning
├── divergent_vgpr_ir/   — raised IR has the expected SPE diamond shape
├── allow_list_audit/    — each allow-listed SemOp routes its EXEC write
│                         through storeExec
├── CMakeLists.txt       — wiring (written by the harness, not by the test author)
├── lit.cfg.py           — lit configuration (substitutions, tool discovery)
├── lit.site.cfg.py.in   — CMake-generated site configuration
└── README.md            — this file
```

A test directory typically contains:

*   `<name>.hip` — a small HIP source that, when compiled with `hipcc
    --genco`, produces the code object the test feeds to the raiser.
    The `.hip` source uses `asm volatile(...)` blocks to pin down the
    exact AMDGPU instructions we want the raiser to see; we do *not*
    rely on hipcc's codegen for anything the test asserts against.
*   `<name>.ll` — the lit test itself. `.ll` is the LLVM-idiomatic
    extension for tests that FileCheck-match raised IR. Each test
    starts with `RUN:` lines and `CHECK:` directives.

Where a test only reads IR (e.g. `divergent_vgpr_ir`), the `.ll`
file's RUN line invokes `%raise_cli --emit-ir=<kernel>` and pipes the
output into FileCheck. Where a test reads the raiser's diagnostics
(e.g. `abort_gate`, `cross_wave_warn`), the RUN line redirects stderr
and optionally asserts a non-zero exit code via `not`.

## Substitutions provided by the harness

Tests should only reference the substitutions below. Do not hard-code
absolute paths or tool names.

| Substitution     | Meaning                                                                       |
|------------------|-------------------------------------------------------------------------------|
| `%raise_cli`     | Path to the built `raise_cli` binary.                                         |
| `%FileCheck`     | Path to `FileCheck` (LLVM).                                                   |
| `%not`           | Path to `not` (LLVM; runs its command and inverts the exit code).             |
| `%S`             | Source directory of the current lit test (the per-test subdirectory).         |
| `%t`             | Per-test temp dir (used to hold the built `.co`).                             |
| `%hip_genco`     | Command fragment that invokes `hipcc --genco --offload-arch=<arch>` with      |
|                  | the appropriate offload arch; see below.                                      |
| `%<name>_co`     | Absolute path to the built `.co` file for `<name>.hip` in the current test    |
|                  | directory. The harness compiles each `.hip` to `.co` at CMake configure /     |
|                  | build time and exposes it as a path substitution, so the lit test itself      |
|                  | does not need to invoke hipcc.                                                |

The harness is responsible for:

*   Discovering `hipcc` and the raiser tools.
*   Compiling each `<name>.hip` in each test directory to
    `<name>_gfx<arch>_unbundled.co` via `hipcc --genco
    --offload-arch=<arch>` followed by `clang-offload-bundler` to
    extract the unbundled code object (same flow as `tests/test_common.hpp`).
    The target arch for each fixture is declared in the test's
    `<name>.hip` as a comment on the first line, `// ARCH: gfx942`.
    If no `ARCH:` comment is present, `gfx942` is assumed.
*   Setting the `hipcc` feature in `config.available_features` iff
    hipcc is on `$PATH`. Tests requiring hipcc-built fixtures must
    declare `REQUIRES: hipcc`.

## Writing style

*   Keep `.hip` fixtures small. One kernel per file. Inline-assembly
    blocks with explicit `"exec"`, `"memory"`, and scalar-register
    clobbers are the recommended way to force the raiser to see a
    specific instruction sequence.
*   CHECK lines should match *SSA patterns*, not exact SSA names —
    the SSA numbering is not stable across LLVM versions. Use
    `CHECK:`, `CHECK-NEXT:`, `CHECK-SAME:`, and variable bindings
    `[[VAR:%[^ ,]+]]` liberally.
*   When a test exercises a diagnostic, FileCheck against `stderr`
    with `--check-prefix=STDERR` and match the *stable* substring of
    the message — "`pre-translation abort`", "`modulo-replication`",
    etc. — rather than pinning to the exact sentence, so future
    rewordings don't break the test.
*   Prefer `not %raise_cli ... 2>&1 | FileCheck` over
    `%raise_cli ... 2>&1 | FileCheck` when asserting an abort; this
    catches regressions where the abort silently degrades into a
    warning.

## Adding a new test

1.  Create `lit_tests/<name>/` with a `<name>.hip` fixture and a
    `<name>.ll` lit test.
2.  In `<name>.hip`, put an `// ARCH: gfx<arch>` comment on line 1 if
    the fixture needs something other than `gfx942`.
3.  In `<name>.ll`, use only the substitutions listed above.
4.  Run the test locally with `ninja check-salmon-lit` (name TBD by
    the harness) to confirm it passes.
5.  Commit the `.hip`, the `.ll`, and (if applicable) an `AGENTS.md`
    update if you're introducing a new audit protocol.

Do not commit built `.co` files. They are derived artifacts.
