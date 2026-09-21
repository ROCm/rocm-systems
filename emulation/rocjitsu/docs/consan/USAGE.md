# ConSan usage

ConSan checks AMD GPU LDS/shared-memory accesses for races. It supports native
code objects for `gfx942`, `gfx950`, `gfx1100`, `gfx1201`, and `gfx1250`.
It does not translate between GPU architectures or provide general
global-memory race detection.

Start with the default mode. Use a [preset](#presets) to trade overhead for
coverage, and a [kernel allowlist](#generate-and-use-a-kernel-allowlist) to focus
on the kernels your workload runs.

## Build and load the hook

Use a RocJitsu CMake build directory, separate from the source tree:

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks
export CONSAN_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"

env HSA_TOOLS_LIB="$CONSAN_HOOK" HSA_TOOLS_DISABLE_REGISTER=1 \
  RJ_CONSAN_LOG=1 \
  ./application 2>consan.log
```

Loading the hook enables ConSan; no separate enable variable is needed.
Registers and report buffers are managed automatically. The hook also runs
waitcheck at code-object load time, reports missing waits, and continues with
ConSan instrumentation. No separate waitcheck tool or settings are required.

`HSA_TOOLS_DISABLE_REGISTER=1` selects ROCr's environment-driven tools path.
Without it, profiler registration in frameworks such as PyTorch can take
precedence and prevent the hook from loading at all.

A zero exit status alone does not mean ConSan checked the workload. After a run,
require an applicable, complete verdict before interpreting its diagnostics:

```sh
grep -q 'ConSan analysis verdict applicable=true analysis_complete=true ' consan.log
```

Run this check only after the application succeeds. A missing verdict means no
analysis was reported; an incomplete verdict is not a clean result. Sampling
still permits false negatives even with a complete verdict.
Rejected configurations print an explicit unchecked-run warning; under
`RJ_CONSAN_POLICY=strict` they terminate with exit code 92.

## Core controls

| Variable | Default | When to use it |
| --- | --- | --- |
| `RJ_CONSAN_PRESET=low\|default\|high\|higher\|max` | `default` | Adjust coverage versus overhead in the default mode. |
| `RJ_CONSAN_KERNEL_ALLOWLIST=name[,name...]` | All kernels | Check a few exact kernel entry names. |
| `RJ_CONSAN_KERNEL_ALLOWLIST_FILE=path` | All kernels | Check a generated list, one exact name per line. |
| `RJ_CONSAN_LOG=N` | Logging disabled | Set `1` to inspect instrumentation, coverage, and results; `3` adds per-site details. |
| `RJ_CONSAN_MODE=default\|supercollider` | `default` | Select the default race detector or the complementary [SuperCollider check](#supercollider). |

## Presets

`RJ_CONSAN_PRESET` currently tunes **only the default mode**, not SuperCollider.
It changes two sampling strides: which workgroups and which four-byte LDS cells
contribute evidence. Smaller strides retain more evidence and generally cost
more. The preset does not change the analysis mode.

| Preset | Workgroup stride | LDS-cell stride | Use it for |
| --- | ---: | ---: | --- |
| `low` | 1024 | 1024 | Large workloads where you want to try lower overhead and accept more misses. |
| `default` | 256 | 256 | Ordinary runs. Unset, empty, and `default` have the same behavior. |
| `high` | 16 | 16 | Increased sampling on large grids, between the default and focused-repro settings. |
| `higher` | 1 | 4 | Small or minimized repros: select every workgroup and one in four LDS cells. Workgroup stride 1 removes dispatch-identity selection misses; see [sampling controls](EXPERT_CONTROLS.md#consan-event-and-sampling-controls). |
| `max` | 1 | 1 | Investigating an issue missed by `higher`: remove workgroup and cell filtering. |

For a small repro:

```sh
env HSA_TOOLS_LIB="$CONSAN_HOOK" HSA_TOOLS_DISABLE_REGISTER=1 \
  RJ_CONSAN_PRESET=higher RJ_CONSAN_LOG=1 \
  ./repro 2>consan.log
```

`higher` is the first preset that selects every workgroup (stride 1). Use `high`
for large grids. If the small repro misses an expected race, try `max`. Even
`max` retains bounded evidence and has analysis limitations; a clean run does
not prove race freedom.

Preset names are case-insensitive. Other settings keep their standard defaults.

At shutdown, an applicable, complete analysis with no reported races prints a
short suggestion to try the next preset (`low` → `default` → `high` → `higher`
→ `max`). It distinguishes zero collected runtime evidence from evidence with
no reported races; neither proves race freedom. The hint is omitted at `max`,
for SuperCollider, for incomplete or non-applicable analysis, and whenever an
explicit sampling stride or offset is supplied (including legacy controls).
This is a host-side summary message, not additional GPU instrumentation.
Explicit sampling overrides take precedence, so clear overrides from earlier
experiments when comparing presets. Check the resolved settings with
`RJ_CONSAN_LOG=1`; see the [expert reference](EXPERT_CONTROLS.md#consan-event-and-sampling-controls)
for override rules.

## Generate and use a kernel allowlist

An allowlist can greatly reduce startup work on large library binaries by
skipping code objects and kernels unrelated to your workload. Generate it from
the same command, inputs, and execution stages you plan to check, including
warm-up or setup that selects kernels.

### 1. Profile without ConSan

Use `rocprofv3` from the same ROCm installation as your application:

```sh
profile_dir="$(mktemp -d "$PWD/consan-profile.XXXXXX")"
env -u HSA_TOOLS_LIB \
  rocprofv3 --kernel-trace --output-format csv \
  --output-directory "$profile_dir" -- ./application
```

### 2. Convert the trace

```sh
rocjitsu_consan_allowlist.py \
  --output "$PWD/consan-kernels.txt" "$profile_dir"
```

The converter accepts trace CSV files or directories and writes unique kernel
names, one per line. It rejects missing/malformed traces and empty dispatch
lists. Keep the trace and regenerate the list when the workload or stack changes.

### 3. Run with the list

```sh
env -u RJ_CONSAN_KERNEL_ALLOWLIST \
  HSA_TOOLS_LIB="$CONSAN_HOOK" HSA_TOOLS_DISABLE_REGISTER=1 \
  RJ_CONSAN_LOG=1 \
  RJ_CONSAN_KERNEL_ALLOWLIST_FILE="$PWD/consan-kernels.txt" \
  ./application 2>consan.log
```

The inline and file allowlists are mutually exclusive. For a short list, instead
unset `RJ_CONSAN_KERNEL_ALLOWLIST_FILE` and set
`RJ_CONSAN_KERNEL_ALLOWLIST=attention_fwd,attention_bwd.kd`.
Names match exactly, with an optional `.kd` suffix; use the file form for names
containing commas.

At normal unload, check each `ConSan kernel allowlist entry` record. A selected
kernel should be loaded, instrumented, and dispatched. If it is not, verify the
workload and inspect instrumentation diagnostics. Nonzero dispatches with zero
`visible_records` warrant a denser preset before drawing conclusions.

Unlisted kernels are not checked. A shared helper is instrumented only when
**all** kernel entries that can reach it are selected. Reuse the same list when
comparing modes; include all relevant workload paths when profiling.

## Coverage and diagnostics

With `RJ_CONSAN_LOG=1`, look for:

```text
ConSan patch end ... outcome=modified-valid ... patches=N modified=true
ConSan coverage ... access=... barrier=... atomic=... fence=...
ConSan analysis verdict ... static_complete=... dynamic_complete=...
ConSan auto report ... visible=N ... conflicts=N ...
ConSan conflict ... first_instruction=... second_instruction=...
```

- **A conflict diagnostic** identifies a race in retained evidence; inspect the
  reported instructions, workgroup, and byte ranges.
- **`modified=true`** confirms that transformed code was loaded, not that a race
  was found. Check coverage and that the relevant kernels actually ran.
- **Incomplete analysis** means some supported instrumentation or required
  evidence is missing. Inspect the accompanying reason before trusting a clean
  result.
- **No conflicts** is inconclusive: sampling, bounded retention, and unsupported
  access patterns can hide races. Try `high`, `higher`, or `max` on a focused repro.
- **A crash, timeout, wrong application result, or GPU reset** is not itself a
  ConSan detection. Keep checking the application's own results.

Normal launch-and-synchronize loops recycle reports automatically. You do not
need checkpoint calls or report-buffer tuning for ordinary repeated workloads.

## SuperCollider

SuperCollider is a complementary check: it repeats supported LDS accesses and
reports when observed values differ. It does not identify an exact racing pair
or establish a happens-before violation. Compare known-correct and suspect runs;
a mismatch alone is not a causal race diagnosis. See [MODES.md](MODES.md).

```sh
env -u RJ_CONSAN_PRESET HSA_TOOLS_LIB="$CONSAN_HOOK" HSA_TOOLS_DISABLE_REGISTER=1 \
  RJ_CONSAN_MODE=supercollider RJ_CONSAN_POLICY=strict RJ_CONSAN_LOG=1 \
  ./application 2>consan.log
```

Use strict policy when collecting validation evidence: it rejects ineffective
instrumentation and terminates with exit code 92 on a load rejection. The default
policy may continue with incomplete analysis; that is not a clean result.

Leave `RJ_CONSAN_PRESET` unset or empty: SuperCollider currently rejects every
explicit nonempty preset, including `default`. It automatically allocates a
non-trapping mismatch marker; no expert controls are needed to start.

## Troubleshooting and expert controls

| Situation | Next action |
| --- | --- |
| Startup is expensive or transformation uses too much memory | Generate an allowlist to avoid transforming unrelated code. |
| Recording overhead is too high | Try `low` in the default mode, accepting reduced coverage. |
| A small known-racy repro gives no diagnostic | Confirm instrumentation and dispatch, then try `higher` and `max`. Use `high` for larger grids. |
| Coverage is incomplete or report allocation fails | Read the reported reason; consult [capabilities](CAPABILITIES.md) or [report controls](EXPERT_CONTROLS.md#consan-report-buffers). |
| A focused validation run must reject ineffective instrumentation | Use `RJ_CONSAN_POLICY=strict`. It can reject helper code objects and terminate with exit code 92; it does not make race diagnostics fatal. |

Use [EXPERT_CONTROLS.md](EXPERT_CONTROLS.md) only when you need finer control:

- [Shared controls](EXPERT_CONTROLS.md#shared-controls): strict-policy details,
  memory limits, register overrides, and fault-injection validation.
- [Default mode](EXPERT_CONTROLS.md#default-mode): independent sampling strides,
  report limits, diagnostic policies, and selective epoch analysis.
- [SuperCollider](EXPERT_CONTROLS.md#supercollider-controls): delays, report modes,
  and synchronization perturbation.
