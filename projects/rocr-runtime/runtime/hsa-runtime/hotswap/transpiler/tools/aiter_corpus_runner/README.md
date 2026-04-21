# `aiter_corpus_runner`

Runs every AITER op-test under native, legacy, and salmon and records
whether each run was right, wrong, crashed, or hung.  A `FAIL` verdict
means a real numerical mismatch — the correctness check comes from
AITER's own `checkAllclose(transpiled, reference)` calls inside each
test, which this runner promotes to hard failures via monkey-patch.

## Quick start

```bash
cd projects/rocr-runtime/runtime/hsa-runtime/hotswap/transpiler/tools/aiter_corpus_runner

# Curated subset (~12 op-tests), all three modes, 1 GPU
python3 runner.py

# Same sweep fanned out across 8 GPUs (one job per GPU)
export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python3 runner.py --jobs 8

# Full op_tests corpus, 8-way parallel
python3 runner.py --all --jobs 8

# One script, one mode
python3 runner.py --script _aiter/op_tests/test_moeTopkSoftmax.py --modes salmon

# Triage a hang / crash / fail: live stderr + full per-run logs
python3 runner.py --modes legacy,salmon --script _aiter/op_tests/test_X.py \
                  --tee-stderr --log-dir _logs
```

First-time setup is one block at the end of this doc.

## Parallel execution (`--jobs N`)

On a multi-GPU box the runner can fan every `(script, mode)` job out
across the GPUs in `HIP_VISIBLE_DEVICES` (or an explicit `--gpus` pool)
and cut wall-clock time roughly linearly.  Defaults:

- `--jobs 1` — strictly serial, GPU 0.  Same behaviour as before.
- `--jobs N > 1` — `N` worker threads, one per GPU slot.  Each child
  runs on exactly one GPU (its assigned slot pinned via
  `HIP_VISIBLE_DEVICES`) so a wedged kernel on one device can't
  blast the others.
- Pool size is capped to `len(--gpus)`; `--jobs 100 --gpus 0-3` runs 4.
- **No runner-side warmup pass.**  AITER's own
  `aiter/jit/core.py::mp_lock` (backed by `FileBaton`) serialises
  concurrent `@compile_ops` builds of the same module at the
  filesystem level: the first worker to enter the critical section
  runs `hipcc`, the others spin-wait on the marker file and then
  `dlopen` the cached artefact.  Parallel runs against a cold
  `_jit_cache/` are therefore correct without any pre-pass.

GPU discovery is **literally** `HIP_VISIBLE_DEVICES` (or an explicit
`--gpus 0,1,2,...` / `--gpus 0-7`).  We don't scan the box with
`rocminfo` — on a shared machine that'd be the wrong answer — so the
user's shell is the single source of truth for "which GPUs may I
touch."  `--jobs > 1` with neither source set errors out before
spawning anything.

Example output with `--jobs 4`:

```
aiter corpus runner: 14 script(s), modes=['native','legacy','salmon'], ...
  jobs=4 × gpus=[0,1,2,3]

[run 1/42] test_layernorm2d.py :: native ... (gpu=2)
[run 2/42] test_moeTopkSoftmax.py :: native ... (gpu=0)
[run 3/42] test_rmsnorm2d.py :: native ... (gpu=1)
[run 4/42] test_layernorm2d.py :: legacy ... (gpu=3)
[verdict 2/42] test_moeTopkSoftmax.py :: native ... PASS (ok, 5.8s, gpu=0)
...
```

`[run k/N]` lines go out before the child starts; `[verdict k/N]`
after it exits.  Interleaving is expected — sort the final grid /
summary instead of the live progress.

### Orphan JIT-cache lock reaper

AITER's `FileBaton` creates `_jit_cache/build/lock_<module_md5>` on
`acquire` and unlinks it on `release`.  Release runs in a `finally`,
so any exit path that skips finally blocks — SIGKILL, OOM-killer,
Ctrl-C mid-syscall — leaves the marker on disk.  Any future
concurrent build of that same module then spins forever in
`baton.wait()` waiting for a release that will never come.

At startup (every invocation, serial or parallel) the runner scans
`_jit_cache/build/` for `lock_*` files and cross-references each
file's inode against the open-FD table of every same-UID Python
process on the host (via `/proc/<pid>/fd`).  A live baton owner still
holds the FD; a crashed owner does not.  Unlocked entries get
unlinked and logged as `[jit-cache] reaped orphan FileBaton lock:
<path>`.

If any same-UID Python process's `fd` directory is unreadable
(ptrace hardening, setuid wrapper) the reaper refuses to touch any
lock and prints the exact `/proc/<pid>/fd` paths that blocked
inspection — deleting a live builder's lock would allow a second
concurrent `hipcc` to corrupt its `.so` output.  Non-Python and
different-UID processes are excluded from the scan because neither
can have opened an AITER baton through its normal code path.

## What it does

For each `(op-test, mode)` pair the runner spawns a child Python,
lets the op-test build its random inputs and call its kernels, and
watches the exit status:

| verdict      | meaning                                                             |
|--------------|---------------------------------------------------------------------|
| `PASS`       | exit 0 — every patched `checkAllclose` was within tolerance         |
| `FAIL`       | exit non-zero — patched `checkAllclose` raised (numerical mismatch) |
| `CRASH`      | terminated by signal (SIGSEGV / SIGABRT / SIGILL / ...)             |
| `HANG`       | exceeded `--timeout`; child SIGKILLed                               |
| `SPAWN_FAIL` | couldn't even start the child                                       |

A `PASS native + non-PASS legacy/salmon` row is the actual
transpilation-gap signal.  Scripts that fail under native get called
out separately in the summary — those are local install / test-script
problems, not transpiler bugs.

Output is a per-script grid, a Failures section with the last 16 KB
of each non-passing run's stderr, and a summary.  `--json PATH`
writes a machine-readable record of every verdict.

## What kernels it hits

AITER ships two kinds of device code:

1. **AOT asm kernels** under `hsa/<gfx>/<module>/<kernel>.co` — hand-
   tuned `.s` → `.co` built once at release time, loaded at runtime by
   AITER's C++ `AiterAsmKernelFast` loader.
2. **JIT / CK kernels** compiled by `hipcc` on first use, targeting
   whatever `gcnArchName` the real device reports.

This runner is built around **(1)**.  On a gfx942 host the C++ loader
would normally pick `hsa/gfx942/*.co` — matching ISA, no transpilation.
We flip it to `hsa/gfx950/*.co` via a small `LD_PRELOAD` shim
(`libaiter_arch_spoof.so`, built by `make` in this directory) that
spoofs `hipGetDeviceProperties().gcnArchName` and reroutes AITER's
`__hipRegisterFatBinary` path through `hipModuleLoadData()` so
Salmon's hook actually sees the foreign-ISA bytes.  Full rationale is
commented in `arch_spoof.cpp`.

**The JIT / CK kernels stay gfx942** — Salmon's hook is conditional on
ISA mismatch (it reads each code-object's `e_flags` and compares to
the device ISA), so matching-ISA loads pass straight through
untouched.  The only code that actually traverses the transpiler is
AITER's asm corpus.

## How correctness is checked

Each AITER op-test already computes the same output two or three ways
— typically `(asm, ck, torch)` — and calls
`aiter.test_common.checkAllclose(...)` to compare them pairwise.
Those calls normally just `print("FAIL", ...)` and keep going.
`_bootstrap.py` patches `checkAllclose` so that whenever the
element-wise mismatch percent exceeds `--strict-tolerance` (default
`0.01`), it raises `AssertionError`.  The script exits non-zero and
the runner records `FAIL`.

At process exit the patched wrapper prints a per-label summary:

```
[aiter_corpus_runner] checkAllclose: 4 pass / 1 fail (threshold=0.01)
[aiter_corpus_runner]      1 pass /    1 fail  label='asm' FAIL
[aiter_corpus_runner]      0 pass /    1 fail  label='asm res' FAIL
[aiter_corpus_runner] first failure: label='asm' percent=12.34% atol=0.03 rtol=0.01
```

so after a `FAIL` verdict the Failures section tells you *which*
comparison broke — `asm` (torch vs transpiled asm) vs `asm res`
(residual) vs `res check` (torch vs CK), etc.  That's how an
asm-transpilation regression is distinguished from a CK-side hiccup
inside the same test.

Determinism is enforced by `--rng-seed` (default `0`), which seeds
`random`, `numpy`, `torch.manual_seed`, and `torch.cuda.manual_seed_all`
before the script runs, so the random input tensors are byte-identical
across `(native, legacy, salmon)` invocations of the same script.

### What counts as the reference

AITER's op-tests use PyTorch's own HIP kernels (`F.layer_norm`,
`torch.matmul`, ...) and AITER's `hipcc`-compiled CK path as gold.
Both are gfx942 and run natively.  The `native` mode column is the
sanity check: if torch-vs-CK breaks between `native` and `salmon` for
the same script, the per-label summary surfaces it under a non-`asm`
label, separately from the `asm`-vs-torch failures the runner is
actually hunting.

## Modes

All three modes feed AITER the same `GPU_ARCHS=gfx942;gfx950` so the
on-disk JIT cache under `--jit-cache` is byte-identical across modes.

| mode   | gcnArchName spoof | LD_PRELOAD                                        | hotswap target                                              | effect                                                                   |
|--------|:-----------------:|---------------------------------------------------|-------------------------------------------------------------|--------------------------------------------------------------------------|
| native | no                | (none)                                            | n/a                                                         | C++ loader picks `hsa/gfx942/*.co` → matching-ISA load → baseline.       |
| legacy | → gfx950          | `libaiter_arch_spoof` + Salmon libhsa + libsalmon | `HSA_HOTSWAP_ISA_OVERRIDE=gfx942`                           | Loader picks `hsa/gfx950/*.co`; ROCR's byte translator rewrites `e_flags`. |
| salmon | → gfx950          | same as legacy                                    | `+HSA_HOTSWAP_IR_RAISER=1`, `+HSA_SALMON_STRICT=1`          | Same reroute; hotswap routes through the Salmon IR raiser instead.       |

## Triaging a HANG / CRASH / FAIL

The Failures section only keeps the last 16 KB of stderr.  Three flags
close the gap:

- **`--tee-stderr`** — streams every child's stderr live, each line
  prefixed with `[<script>::<mode>]`.  Use when a run hangs and you
  want to watch the last kernel load in real time.
- **`--log-dir DIR`** — writes every child's full unbuffered
  stdout+stderr to `DIR/<script>__<mode>__<ts>__pid<pid>.log` with a
  header (exact cmd + cwd) and footer (returncode + elapsed +
  timed-out).  Then:
  ```bash
  diff -u _logs/test_X.py__legacy__*.log _logs/test_X.py__salmon__*.log | less
  ```
  The first divergent line is usually the `LoadCodeObject` that
  tripped the bug.
- **`--print-command`** — no spawning; emits a paste-ready bash
  block for every `(script, mode)` pair with the exact env + argv
  (and only the env vars that differ from the parent shell).
  Paste under `rocgdb` / `strace` / `AMD_LOG_LEVEL=5`.

## Limitations

- Coverage is bounded by the op-tests AITER ships.  A `.co` in
  `hsa/gfx950/` that no test exercises is silently outside the signal.
- The per-script verdict is whole-script.  When a test fails you need
  the per-label `checkAllclose` summary in stderr to tell asm vs CK
  apart.
- Some tests in the curated subset don't exercise the asm path for
  every input shape; those pass trivially in legacy/salmon because no
  gfx950 `.co` ever loads.
- The salmon-mode column is sensitive to the current
  `libhsa-runtime64.so.1` build.  A coverage-WIP revision of the
  Salmon IR raiser can introduce a hang in a later non-asm kernel
  load even when the target asm kernel translates correctly; a clean
  `native PASS + legacy PASS + salmon HANG` row is usually a libhsa
  integration issue, not an asm-transpilation regression.  `legacy` is
  the stable baseline for the transpiler itself.

## Setup — one-time

```bash
# Triton runner's shared rocm-7 venv (torch + numpy + pytest)
cd ../triton_corpus_runner
python3.12 -m venv .venv-rocm7
.venv-rocm7/bin/pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.0
.venv-rocm7/bin/pip install numpy tabulate pytest

# libsalmon_intercept.so (built by compare_correctness)
cd ../compare_correctness && make libsalmon_intercept.so

# libaiter_arch_spoof.so (built here)
cd ../aiter_corpus_runner && make

# Clone AITER and install its extra Python deps into the shared venv
python3 runner.py --setup
```

`--setup` clones [ROCm/aiter](https://github.com/ROCm/aiter) into
`./_aiter` and `pip install -r _aiter/requirements.txt` into
`--triton-venv`.  We deliberately do **not** run AITER's
`setup.py develop` — we just prepend `--aiter-root` to `PYTHONPATH`
and let AITER's `@compile_ops` lazily build its C++ extensions into
`--jit-cache/` (default `./_jit_cache/`) on first use.

Everything lives under `$HOME`.  No `/etc/ld.so.preload`, no
`/opt/rocm/*` writes.  `LD_PRELOAD` is per-process and only affects
children we spawn.  Each worker pins its child to exactly one GPU
via `HIP_VISIBLE_DEVICES=<slot>`, so in `--jobs N` mode a wedged
kernel on one device can't take down the sibling workers — blast
radius scales with the size of the `--gpus` pool, not beyond it.

Point at an existing AITER checkout instead of cloning:

```bash
python3 runner.py --aiter-root /path/to/your/aiter \
                  --triton-venv /path/to/your/torch+rocm7/venv
```

## Flag reference

| flag                 | default                                                                    | what                                                                     |
|----------------------|----------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `--aiter-root`       | `<HERE>/_aiter`                                                            | AITER checkout                                                           |
| `--aiter-repo`       | `https://github.com/ROCm/aiter.git`                                        | repo URL for `--setup`                                                   |
| `--setup`            | off                                                                        | clone AITER + install its Python deps; exits after                       |
| `--script PATH`      | —                                                                          | run one script instead of the sweep (repeatable)                         |
| `--scripts-dir DIR`  | `<aiter_root>/op_tests`                                                    | where to discover `test_*.py` from                                       |
| `--all`              | off                                                                        | full `op_tests/test_*.py` corpus instead of the curated subset           |
| `--modes`            | `native,legacy,salmon`                                                     | comma-separated subset of `{native,legacy,salmon}`                       |
| `--source-gfx`       | `gfx950`                                                                   | AITER ISA forced in legacy/salmon modes                                  |
| `--native-gfx`       | auto-detected via `rocminfo`                                               | real device ISA; override for cross-host testing                         |
| `--strict-tolerance` | `0.01`                                                                     | `checkAllclose` mismatch percent above which the patched wrapper raises  |
| `--perftest-iters`   | `1`                                                                        | informational; the patched perftest always calls the kernel exactly once |
| `--rng-seed`         | `0`                                                                        | seed for `random` / `numpy` / `torch` before every script                |
| `--timeout`          | `600` s                                                                    | per-child wall clock; SIGKILL on overshoot                               |
| `--jit-cache`        | `<HERE>/_jit_cache`                                                        | persistent cache for AITER's `@compile_ops`                              |
| `--gpu`              | `0`                                                                        | `HIP_VISIBLE_DEVICES` in serial (`--jobs 1`) mode; ignored with `--jobs > 1` |
| `--jobs N`           | `1`                                                                        | run `N` `(script, mode)` jobs in parallel, one per GPU slot              |
| `--gpus LIST`        | — (falls back to `HIP_VISIBLE_DEVICES`)                                    | comma list / range for the parallel pool (`0,1,2`, `0-7`, `0,2-5,7`)     |
| `--script-arg A`     | —                                                                          | forward one argv token to the user script (repeatable)                   |
| `--tee-stderr`       | off                                                                        | stream each child's stderr live with `[<script>::<mode>]` prefix          |
| `--log-dir DIR`      | —                                                                          | full per-run transcripts under `DIR/<script>__<mode>__<ts>__pid<pid>.log` |
| `--print-command`    | off                                                                        | dry-run: emit paste-ready bash block per `(script, mode)` and exit 0     |
| `--json PATH`        | —                                                                          | machine-readable record of every verdict                                 |
| `--triton-venv`      | `../triton_corpus_runner/.venv-rocm7`                                      | venv with torch + rocm7                                                  |
| `--libsalmon`        | `../compare_correctness/libsalmon_intercept.so`                            | intercept shim                                                           |
| `--libhsa`           | `~/rocm-systems/projects/rocr-runtime/build/rocr/lib/libhsa-runtime64.so.1`| Salmon-enabled libhsa                                                    |
| `--libamdhip`        | —                                                                          | only set when the venv's torch ships HIP older than the Salmon ROCR      |
| `--libarch-spoof`    | `<HERE>/libaiter_arch_spoof.so`                                            | arch-spoof shim built by `make` here                                     |
| `--spoof-arch`       | `--source-gfx`                                                             | value written into `gcnArchName` by the shim                             |

`python3 runner.py --help` has the complete list.

## Files

- `runner.py` — entry point: discovery, mode dispatch, reporting.
- `_bootstrap.py` — child-side: env setup, `checkAllclose` patch,
  `perftest` patch, RNG seeding, `runpy` of the user script.
- `arch_spoof.cpp` + `arch_spoof.ver` — source for
  `libaiter_arch_spoof.so`.  Full rationale for every intercepted
  symbol is in the file comments.
- `Makefile` — builds `libaiter_arch_spoof.so`.  `make clean && make`
  after editing either of the two sources.
- `_aiter/`, `_jit_cache/`, `_logs*/`, `libaiter_arch_spoof.so` —
  local build/run artifacts; gitignored.
