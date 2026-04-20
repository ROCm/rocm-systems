# `aiter_corpus_runner` — breadth-only Salmon coverage sweep over AITER

A prototype of the **"find crashes only"** path for AITER's AOT
gfx950 kernel ship — the AITER counterpart to
[`triton_corpus_runner`](../triton_corpus_runner/README.md).
It runs each AITER op-test three times — once natively (no
intercept), once through the legacy hotswap byte translator, once
through Salmon — and records what happened to each run (`PASS` /
`FAIL` / `CRASH` / `HANG`).  No per-launch numerical verdict written
by us, no per-kernel shim, no AOT recipe: each AITER test already
calls `checkAllclose(asm_output, torch_reference)` itself, and the
runner monkey-patches that to actually raise on mismatch so the
script's exit code becomes the verdict.

The intended workflow is the dual of `compare_correctness`:

| tool                 | depth                     | breadth                                | new entry cost                   |
|----------------------|---------------------------|----------------------------------------|----------------------------------|
| `compare_correctness`| per-tensor numerical match| ~5 hand-curated kernels                | author a Python recipe + sidecar |
| `triton_corpus_runner`| exit-status + assertions  | every `.py` in a directory             | drop a file in                   |
| `aiter_corpus_runner` | exit-status + `checkAllclose`| every `op_tests/test_*.py` AITER ships| `--setup` once, drop a `.py` after that |

You'd use this to mass-screen AITER's gfx950 asm corpus against the
salmon transpiler and get a quick "salmon broke N of M" number;
then promote the most interesting failures to per-kernel
`compare_correctness` recipes for a real numerical post-mortem.

## What it does

`runner.py` discovers `op_tests/test_*.py` from a local AITER
checkout (defaulting to a curated single-GPU subset; `--all` for
the full corpus) and, for each `(script, mode)` pair, spawns a
child Python process that:

  1. Imports `_bootstrap` (this module), which:
     - sets `GPU_ARCHS` and `AITER_GPU_ARCHS` (both to
       `"gfx942;gfx950"` by default) so AITER's codegen emits
       config rows for both arches and the JIT cache is
       mode-invariant;
     - seeds `torch` / `numpy` / `random` with `--rng-seed`
       (default `0`) so the random input tensors the op-test
       builds are byte-identical across (native, legacy, salmon)
       invocations of the same script + args — that's the
       invariant the cross-mode comparison relies on;
     - monkey-patches `aiter.test_common.checkAllclose` to raise
       `AssertionError` when the per-element mismatch percent
       exceeds `--strict-tolerance` (default `0.01`, matching
       AITER's own warning-vs-failed cutoff);
     - monkey-patches `aiter.test_common.perftest` /
       `run_perftest` to call the kernel exactly once and return
       `avg=0.0` (kills the 101-iteration benchmark sweeps that
       would otherwise dominate wall clock).

     The `hsa/gfx950/` vs `hsa/gfx942/` choice at the C++ asm-loader
     layer is made separately by `libaiter_arch_spoof.so` (see
     *"The spoof shim"* below), driven by the
     `AITER_CORPUS_SPOOF_ARCH` env var the runner sets in the
     transpile modes and leaves unset in `native`.
  2. Runs the script via `runpy.run_path(..., run_name='__main__')`
     so its `if __name__ == "__main__":` block fires and any
     top-level test loops execute too.

The parent waits up to `--timeout` seconds, SIGKILLs hung children,
classifies the exit status, and prints:

  - a per-script grid (one row per script, one column per mode);
  - a Failures section with the last ~12 lines of stderr per
    non-passing run — including the patched-`checkAllclose`
    per-label summary the bootstrap dumps via `atexit`;
  - a Summary section that highlights "scripts that PASS native but
    not legacy/salmon" — the actual transpilation-coverage signal.

## Modes

All three modes feed AITER the **same** `GPU_ARCHS=gfx942;gfx950`
(and the matching `AITER_GPU_ARCHS`).  That keeps the on-disk JIT
cache under `--jit-cache` byte-identical across modes so running
`native` first doesn't poison `legacy`/`salmon` with a single-arch
build of the asm-config map (see *"The spoof shim"* below for the
heuristic-kernel-lookup abort that would otherwise fire).

| mode    | spoofs `gcnArchName`? | LD_PRELOAD                                         | ROCR hotswap target         | asm path                                                   |
|---------|:---------------------:|----------------------------------------------------|-----------------------------|------------------------------------------------------------|
| native  | no                    | (none)                                             | n/a                         | AITER's C++ asm loader sees the real `gfx942`, picks `hsa/gfx942/*.co`; matching-ISA load → plain `hipModuleLoad`; no translation. Baseline. |
| legacy  | **yes (→ gfx950)**    | `libaiter_arch_spoof` + Salmon libhsa + libsalmon  | `HSA_HOTSWAP_ISA_OVERRIDE=gfx942`, `HSA_HOTSWAP_IR_RAISER` unset | Spoofed loader picks `hsa/gfx950/*.co`; shim reroutes `__hipRegisterFatBinary` → `hipModuleLoadData`; ROCR's byte translator rewrites `e_flags` to gfx942.  |
| salmon  | **yes (→ gfx950)**    | same as legacy                                     | `HSA_HOTSWAP_ISA_OVERRIDE=gfx942`, `HSA_HOTSWAP_IR_RAISER=1`, `HSA_SALMON_STRICT=1` | Same reroute as legacy; ROCR's hotswap hook routes through the Salmon IR raiser instead of the byte translator.                                          |

## The spoof shim (`libaiter_arch_spoof.so`)

AITER's own C++ asm-kernel loader (`AiterAsmKernelFast::init` in
`aiter_hip_common.h`) makes two decisions that bypass Salmon's
normal interception point:

1. **Which `.co` to read** — it queries
   `hipGetDeviceProperties(..., device).gcnArchName` and reads
   `hsa/<gcnArchName>/<module>/<kernel>.co`.  On a gfx942 host this
   naturally picks `hsa/gfx942/*.co`, and Salmon never sees a
   foreign-ISA code object.
2. **How to register it** — the raw HSACO ELF bytes are wrapped in a
   private `FatBinaryWrapper{magic=0x48495046}` and handed to
   `__hipRegisterFatBinary` / `__hipRegisterFunction` /
   `hipGetFuncBySymbol`.  That **is not** the `hipModuleLoadData`
   path Salmon's intercept hooks.  Even if we force the gfx950 file
   via step 1, the real HIP impl sees an ISA mismatch, returns
   early, and launches later fail with `invalid resource handle`.

`libaiter_arch_spoof.so` is a small LD_PRELOAD shim that closes both
gaps without touching AITER or Salmon:

  - **`hipGetDevicePropertiesR0600` / `hipGetDevicePropertiesR0000`** —
    delegates to the real call, then rewrites `gcnArchName` to
    whatever `AITER_CORPUS_SPOOF_ARCH` is set to (typically
    `gfx950`).  The symbols are exported with the correct version
    tags via `arch_spoof.ver` so the dynamic linker actually picks
    our copy for AITER's `hipGetDevicePropertiesR0600@hip_6.0`
    reference — an unversioned export would silently lose the
    linker's version-matching tie-break.
  - **`__hipRegisterFatBinary`** — requires the wrapper's
    `.magic` to equal the HIPF sentinel (`0x48495046`) *and* the
    first 20 bytes of `.binary` to form a valid AMDGPU ELF64
    header (`\x7fELF`, `ELFCLASS64`, `ELFDATA2LSB`, `e_machine
    == EM_AMDGPU`).  Both checks must pass before we reroute
    through `hipModuleLoadData()` so **Salmon's hook actually
    sees the foreign-ISA bytes** and can transpile them.  Every
    other shape — non-HIPF wrappers from other toolchains,
    hipcc-generated `__CLANG_OFFLOAD_BUNDLE__` bundles,
    non-AMDGPU ELFs, malformed headers — falls through to the
    real HIP impl unchanged.
  - **`__hipRegisterFunction`** — for our rerouted modules, resolves
    the kernel via `hipModuleGetFunction` and caches the
    `hostFn → hipFunction_t` mapping.
  - **`hipGetFuncBySymbol`** — returns the cached function handle
    for rerouted modules; falls through to the real impl otherwise.
  - **`__hipUnregisterFatBinary`** — `hipModuleUnload`s rerouted
    modules, frees our boxed handle.

All of the above is gated on the HIPF-magic + AMDGPU-ELF sniff
(for the fat-binary path) or a lookup in our own handle table (for
the follow-up register/lookup/unregister calls), so real hipcc fat
binaries and payloads from foreign toolchains are never rerouted.
Build with `make`; the Makefile links against `arch_spoof.ver` so
the exports get the right version tags, compiles with
`-fvisibility=hidden` + explicit `__attribute__((visibility
("default")))` on the HIP hooks so no incidental symbols leak out,
and registers a `__attribute__((destructor))` drain that
`hipModuleUnload`s any modules still in our handle table on process
exit.

**Self-check.**  The shim registers a `__attribute__((constructor))`
that, for every exported hook, asks the dynamic linker to resolve
the versioned name via `dlsym(RTLD_DEFAULT, ...)` and compares the
address to our own.  A mismatch means a later `LD_PRELOAD` entry or
a broken `arch_spoof.ver` DAG has shadowed our export — the
runner's results would then silently look like native across all
three modes, so we `abort()` with a loud message instead.  Success
prints `startup self-check: all 5 exported hooks win dlsym(...) —
OK` to stderr in every transpile run.

**Verifying the shim engages.**  A Salmon-mode run should log, on
first invocation:

```
[aiter_arch_spoof] startup self-check: all 5 exported hooks win dlsym(RTLD_DEFAULT, ...) — OK
[aiter_arch_spoof] active: every hipGetDeviceProperties().gcnArchName will be rewritten to gfx950 ...
[aiter_arch_spoof] first rewrite fired: gcnArchName -> gfx950 (subsequent rewrites silent)
[aiter_arch_spoof] first __hipRegisterFatBinary reroute fired: raw HSACO ELF detected in FatBinaryWrapper -> hipModuleLoadData (Salmon will see it); subsequent reroutes silent
salmon_intercept: redirected dlsym("hipGetProcAddress")
salmon_intercept: active, target=gfx942
salmon_intercept: patched e_flags for gfx942 (NNNN bytes)
```

The first line proves the dynamic linker resolved our exports
ahead of the real libamdhip64 for every hook the runner depends
on; the last line is the proof that Salmon actually picked up and
rewrote a code object (size `NNNN` matches the on-disk
`hsa/gfx950/<module>/<kernel>.co` file).

## Verified asm-path test

The canonical smoke test for the full pipeline is
`test_moeTopkSoftmax.py`'s `topk_softmax_asm` invocation:

```bash
rm -rf _jit_cache && mkdir -p _jit_cache
python3 runner.py \
  --modes native,legacy,salmon \
  --script _aiter/op_tests/test_moeTopkSoftmax.py \
  --script-arg=-t --script-arg=8 \
  --script-arg=-e --script-arg=128 \
  --script-arg=-d --script-arg=bf16 \
  --script-arg=-k --script-arg=8 \
  --timeout 300
```

With `--rng-seed 0` (the default) this deterministically produces
`native PASS` + `legacy PASS` on a Salmon-enabled ROCR build — the
asm path (`topksoftmax_4x128x8_bf16`) gets translated by the
`HSA_HOTSWAP_*` byte-rewriter and every `checkAllclose` label
returns within tolerance.  The `salmon` column is sensitive to the
specific `libhsa-runtime64.so.1` build being used — a coverage-WIP
revision of the Salmon IR raiser can introduce infinite loops in a
later non-asm kernel load; when that happens you'll see a clean
`native PASS + legacy PASS + salmon HANG` row and the salmon child
will stop logging right after `salmon: OK (gfx950 -> gfx942, ...)`.
That is **not** an asm-path transpilation regression (legacy
confirms the asm kernel translated + ran + compared correctly); it
is a libhsa integration issue to be debugged against the current
`libhsa-runtime64.so.1` using the normal salmon triage tooling.

Per-script `checkAllclose` labels to look for:

  - `asm topk_weights` / `asm topk_ids` — the transpiled
    `topksoftmax_4x128x8_bf16.co` output vs AITER's own HIP
    reference.  **These pass in every mode that doesn't crash
    before reaching them**, including `salmon` — that's the
    transpile-correctness signal for the asm path.
  - `hip topk_weights` / `hip topk_ids` — the HIP kernel vs the
    same reference.  Mode-independent; always passes on gfx942
    hardware.
  - `topk_weights [golden vs aiter]` from `test_grouped_topk` —
    numerically marginal at the upstream default `atol=0.01
    rtol=0.01` on bf16 inputs.  **Not** an asm-path kernel,
    **not** something the shim or Salmon touches.  With
    `--rng-seed 0` (the default) its PASS/FAIL verdict is now
    stable across runs *and identical across modes*, so when
    this label shows up in the Failures section it appears in
    every mode simultaneously and never masquerades as a
    transpilation regression.  Treat it as a known AITER-side
    numerical-stability issue; the full test is still useful
    because it's what exercises `topk_softmax_asm` on our end.

## Verdicts

| verdict      | meaning                                                                 |
|--------------|-------------------------------------------------------------------------|
| `PASS`       | exit 0 (= every patched `checkAllclose` was within tolerance)           |
| `FAIL`       | exit non-zero, normal termination (e.g. patched `checkAllclose` raised) |
| `CRASH`      | terminated by signal (SIGSEGV / SIGABRT / SIGILL / ...)                 |
| `HANG`       | exceeded `--timeout`; child SIGKILLed                                   |
| `SPAWN_FAIL` | couldn't even start the child (missing python, missing libsalmon, ...) |

A `PASS` under native + non-`PASS` under legacy/salmon is a real
transpilation gap.  A non-`PASS` under native is a problem with the
script itself (or the local AITER install) and the summary calls
those out separately so they don't pollute the salmon-coverage
signal.

## Triaging a HANG / CRASH / FAIL

When a run ends in a verdict other than `PASS` the stderr tail in
the Failures section only shows the last 16KB — enough to see
where the run was when it died, usually not enough to diff two
full transcripts or to single-step the actual kernel.  Three
flags are there to close that gap.

### `--tee-stderr` — live child output

```bash
python3 runner.py --modes salmon ... --tee-stderr
```

Streams every child's stderr to the runner's stderr **as it's
emitted**, prefixed with `[<script>::<mode>]` on each line so
multi-script runs stay legible.  No more "blinking cursor for
180s while salmon hangs" — you see the last `hotswap:
LoadCodeObject ...` line land in your terminal and can Ctrl-C
the runner the moment you have enough.  Adds zero overhead to a
clean-pass run; the pump threads only ever read what the child
already wrote.

### `--log-dir DIR` — full per-run transcripts

```bash
python3 runner.py --modes native,legacy,salmon ... --log-dir _logs
```

Writes every child's **complete unbuffered stdout+stderr** to
`<DIR>/<script>__<mode>__<YYYYMMDD-HHMMSS>.log`.  Each file has
a short header with the exact `python -m _bootstrap` command
used, the child cwd, and the start timestamp, and a footer with
the return code + elapsed time + timed-out flag.  Ideal for
diffing a hung salmon run against a clean legacy one:

```bash
diff -u _logs/test_moeTopkSoftmax.py__legacy__*.log \
        _logs/test_moeTopkSoftmax.py__salmon__*.log \
  | less
```

The first divergent line is almost always the `LoadCodeObject`
call that tripped the bug.

### `--print-command` — paste-ready reproduction

```bash
python3 runner.py --modes salmon ... --print-command > repro.sh
```

For every `(script, mode)` pair in the matrix, emits the exact
env + argv the runner **would have** used, as a paste-ready bash
block with every runner-specific env var `shlex.quote`d.
Nothing is spawned; the runner exits 0.  Then you can wrap the
child manually:

```bash
# Run under rocgdb
bash repro.sh                                     # normal
AMD_LOG_LEVEL=5 bash repro.sh                     # with HIP-side logs
gdb --args $(tail -1 repro.sh | tr -d '()\\')     # under gdb
strace -f -e trace=futex bash repro.sh            # to chase a hang
```

The printed block diffs the child env against the runner's own
parent shell, so you only see the vars the runner actually
added/changed; everything else (HOME, DISPLAY, PATH, ...) is
inherited verbatim from your interactive environment, which
matches what the runner itself does.

## Reference & isolation — what's actually being compared

This section is non-optional reading.  AITER's op-tests use
**PyTorch's own HIP backend** (`F.layer_norm`, `torch.matmul`, ...)
as the gold, plus often a CK reference (AITER's `hipcc`-compiled C++
side).  Both run natively on the actual gfx942 device.  Only AITER's
asm path — the one that does `hipModuleLoad(hsa/gfx950/*.co)` — is
forced through Salmon.

How the selectivity works:

  - The hotswap hook in our Salmon-enabled libhsa is **conditional
    on ISA mismatch**: it reads each code-object's `.note` /
    `e_flags` and compares to the actual device ISA.  Matching ISA
    → passthrough; mismatching ISA → transpile.
  - PyTorch's pre-baked HIP kernels are gfx942 (matching the
    device) → passthrough.
  - AITER's CK kernels (`hipcc`-compiled at install time, target =
    actual device) are gfx942 → passthrough.
  - AITER's asm kernels ship as foreign-ISA `.co` files.
    `AITER_CORPUS_SPOOF_ARCH=gfx950` + `libaiter_arch_spoof.so`
    flips AITER's loader toward `hsa/gfx950/*.co`, and the shim's
    `__hipRegisterFatBinary` reroute funnels the raw HSACO ELF
    through `hipModuleLoadData` so ROCR's ISA-mismatch check fires
    and **Salmon transpiles** the kernel.

The `GPU_ARCHS=gfx942;gfx950` env setting (always applied) only
steers *AITER's Python-side* paths — JIT-build `--offload-arch`
flags, `codegen.py`'s config-map filter, asm-availability guards
in tests.  It does NOT rebuild PyTorch or AITER's CK side; those
were committed to gfx942 at install time and stay gfx942, and
AITER's C++ asm loader still reads `hsa/<gcnArchName>/`.  Which is
why the shim is required on top of the env setting: the env alone
tells AITER *what arches exist*, the shim tells AITER's C++ side
which one the device is *(lying that it's gfx950)* so the loader
reaches for the foreign-ISA `.co` files in the first place.

The one trust assumption to be honest about: in legacy/salmon mode,
**every** `hipModuleLoad` in the process transits our Salmon-enabled
libhsa, even though it just passes through unchanged for matching-ISA
loads.  Same trust the Triton runner already operates on.  The
`native` mode column (no `LD_PRELOAD`, no shim) is the sanity check —
if `run_torch` numerics differ between `native` and `salmon` for a
script, the patched-`checkAllclose` summary will surface it as a
`torch-vs-ck` (or `res check`) failure with a distinct `msg=` label,
separate from the `torch-vs-asm` failures the runner is actually
hunting.

## Per-label `checkAllclose` summary

After every script the bootstrap prints something like:

```
[aiter_corpus_runner] checkAllclose: 4 pass / 1 fail (threshold=0.01)
[aiter_corpus_runner]      2 pass /    0 fail  label='[perf] dim: (128, 8192) ...'
[aiter_corpus_runner]      1 pass /    0 fail  label='res check'
[aiter_corpus_runner]      1 pass /    1 fail  label='asm' FAIL
[aiter_corpus_runner]      0 pass /    1 fail  label='asm res' FAIL
[aiter_corpus_runner] first failure: label='asm' percent=12.3400% atol=0.03 rtol=0.01
```

`asm` and `asm res` are the AITER-test labels for "torch vs asm
output" and "torch vs asm residual"; `res check` is "torch vs CK
residual"; the `[perf] dim:` rows are the torch-vs-CK comparisons.
This breakdown lives in the child's stderr, which the parent
captures and surfaces in the Failures section.

## Setup — one-time

The runner shares `triton_corpus_runner`'s `.venv-rocm7` (torch +
rocm7 + numpy + pytest) for the heavy Python deps, plus AITER's
own `pandas` / `psutil` / `einops` / `matplotlib` / `pyyaml` /
`pybind11>=3.0.1` / `ninja` / `flydsl` from the checkout's
`requirements.txt`.

```bash
# 1) Make sure triton_corpus_runner's rocm-7 venv exists.  See its
# README; one-line summary:
cd ../triton_corpus_runner
python3.12 -m venv .venv-rocm7
.venv-rocm7/bin/pip install --pre torch \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.0
.venv-rocm7/bin/pip install numpy tabulate pytest

# 2) Build libsalmon_intercept.so via compare_correctness if you
# haven't already:
cd ../compare_correctness
make libsalmon_intercept.so

# 3) Build the arch-spoof shim:
cd ../aiter_corpus_runner
make   # produces libaiter_arch_spoof.so in this directory

# 4) Clone AITER and pip-install its extra Python deps into the
# rocm-7 venv (idempotent):
python3 runner.py --setup
```

`--setup` clones [ROCm/aiter](https://github.com/ROCm/aiter) into
`./_aiter` and runs `pip install -r _aiter/requirements.txt`
against `--triton-venv`.  We deliberately do **not** run AITER's
`setup.py develop` — the runner just prepends `--aiter-root` to
`PYTHONPATH`, and AITER's `@compile_ops` decorator builds its C++
extensions lazily on first use.  Those builds get cached in
`--jit-cache` (default `./_jit_cache/`) so subsequent sweeps don't
pay the hipcc tax again.

If you'd rather point at a checkout you already have:

```bash
python3 runner.py --aiter-root /path/to/your/aiter \
                  --triton-venv /path/to/your/torch+rocm7/venv
```

The runner refuses to start if `--aiter-root` doesn't look like an
AITER checkout — no fallbacks.

## Usage

```bash
python3 runner.py                                  # curated subset, all modes
python3 runner.py --modes native                   # quick baseline
python3 runner.py --all                            # full op_tests corpus
python3 runner.py --script op_tests/test_pa.py --modes salmon
python3 runner.py --json out.json                  # machine-readable record
```

Knobs of interest:

| flag                  | default                                                                 | what                                                                       |
|-----------------------|-------------------------------------------------------------------------|----------------------------------------------------------------------------|
| `--aiter-root`        | `<HERE>/_aiter`                                                         | AITER checkout (cloned by `--setup`)                                       |
| `--setup`             | off                                                                     | clone AITER + pip install its Python deps; idempotent; exits after         |
| `--modes`             | `all`                                                                   | comma-separated subset of `{native,legacy,salmon}`                         |
| `--all`               | off                                                                     | use the full op_tests corpus instead of the curated subset                 |
| `--source-gfx`        | `gfx950`                                                                | AITER ISA forced in legacy/salmon modes                                    |
| `--strict-tolerance`  | `0.01`                                                                  | `checkAllclose` mismatch percent above which the patched wrapper raises    |
| `--perftest-iters`    | `1`                                                                     | informational; the patched perftest always calls the kernel exactly once   |
| `--timeout`           | `600` s                                                                 | per-child wall-clock; SIGKILL on overshoot.  First sweep includes hipcc compile of AITER's CK side; subsequent runs reuse `--jit-cache` and are fast |
| `--triton-venv`       | `../triton_corpus_runner/.venv-rocm7`                                   | venv root with torch+rocm7                                                 |
| `--libsalmon`         | `../compare_correctness/libsalmon_intercept.so`                         | the intercept shim                                                         |
| `--libhsa`            | `~/rocm-systems/projects/rocr-runtime/build/rocr/lib/libhsa-runtime64.so.1` | Salmon-enabled libhsa                                                  |
| `--libamdhip`         | (empty)                                                                 | only set when the venv's torch ships HIP older than the Salmon ROCR        |
| `--jit-cache`         | `<HERE>/_jit_cache`                                                     | persistent JIT cache for AITER's `@compile_ops`                            |
| `--gpu`               | `0`                                                                     | `HIP_VISIBLE_DEVICES` for every child.  Pass `''` to clear it (only on a machine you own) |

## Curated default subset

Without `--all`, the runner picks a curated set of single-GPU,
asm-exercising tests:

```
test_layernorm2d.py            test_layernorm2dFusedAddQuant.py
test_rmsnorm2d.py              test_rmsnorm2dFusedAddQuant.py
test_pa.py                     test_pa_v1.py
test_mla.py
test_mha.py                    test_mha_varlen.py
test_gemm_a16w16.py            test_gemm_a8w8.py             test_gemm_a4w4.py
test_quant.py
```

The full op_tests corpus includes multi-GPU communication tests
(`test_communication.py`, `test_custom_allreduce*.py`) and tests
whose default config requires hardware we don't have on this box,
which is why those aren't in the default sweep.

## What the runner cannot do (limitations)

- **Not 100% asm-coverage strict.**  It's the AITER tests' own
  coverage.  A kernel that's in `hsa/gfx950/` but not exercised by
  any `op_tests/test_*.py` is silently outside the breadth signal.
  A future "tests touched X of Y kernels" view could compare the
  per-mode `salmon: OK (...)` log lines from the hotswap hook
  against `ls hsa/gfx950/**/*.co`.
- **No isolation between asm and CK paths in the per-script
  verdict.**  When `test_layernorm2d.py` fails, you don't know
  from the exit code alone whether the asm or CK path is broken.
  The patched-`checkAllclose` per-label summary in stderr gives
  you the answer cheaply (look for the `asm` / `asm res` rows
  vs the `[perf]` torch-vs-CK rows), but the runner verdict
  itself is whole-script.
- **`test_pa.py` / `test_mla.py` use `argparse`.**  They have
  sensible top-level defaults so `runpy` works without args.
  Non-default configs would need a per-script overrides table we
  haven't built.
- **Some tests in the curated subset don't actually exercise the
  asm path** for every input shape (e.g. a CK-only function might
  be the only thing exercised by some shapes in `test_layernorm2d`).
  The native column will still PASS in those cases; legacy/salmon
  will too because no gfx950 .co was loaded.  That's a clean
  passthrough verdict, not a coverage win.

## Safety on a shared box

This tool **does not modify any system state** — no
`/etc/ld.so.preload`, no `/opt/rocm/*` writes, no system-Python
installs.  Everything we add lives under `$HOME` (the AITER clone,
the JIT cache, the rebuilt ROCR, the intercept shim).  `LD_PRELOAD`
is per-process and only affects children we spawn.  Other users'
shells and processes never see any of it.

The one shared resource on a multi-tenant box is the GPU itself.
A buggy transpiled kernel that wedges an SQ/CU would force the
kernel-mode `amdgpu` driver to do a queue reset on the device it
ran on — affecting only users actively using that *same* device
index at that instant.  This risk class is the same as for any
GPU compute workload, not specific to the intercept.

We default `--gpu 0` so any blast radius is contained to one of
the box's GPUs.  Pass `--gpu N` to use a different device, or
`--gpu ''` to expose all of them (only do that on a machine you
own outright).

## Files

  * `runner.py` — main entry, discovery, mode dispatch, reporting.
  * `_bootstrap.py` — child-side: sets `GPU_ARCHS` and
    `AITER_GPU_ARCHS` before importing aiter, patches
    `checkAllclose` to raise on mismatch with per-label `atexit`
    summary, replaces `perftest` with a single-call no-op shim,
    runpy's the user script.
  * `arch_spoof.cpp` — source for `libaiter_arch_spoof.so`: the
    `gcnArchName` rewrite hooks, the `__hipRegisterFatBinary`
    reroute, and the handle-table plumbing for
    `__hipRegisterFunction` / `__hipUnregisterFatBinary` /
    `hipGetFuncBySymbol`.  Reserved for this tool; not consumed by
    anything else in the tree.
  * `arch_spoof.ver` — linker version script pinning the shim's
    exports to the `hip_4.2` / `hip_6.0` / `hip_6.2` version nodes
    that match the symbols AITER-built modules reference.  Without
    this the dynamic linker silently falls through to the real
    `libamdhip64.so.7` for versioned references.
  * `Makefile` — builds `libaiter_arch_spoof.so` from the two
    files above.  `make clean && make` after editing either.
  * `libaiter_arch_spoof.so` — build product; not committed.
  * `_empty_rules.json` — minimal rules file for `HSA_HOTSWAP_RULES`
    (`{"version":1,"target":"amdgcn-amd-amdhsa--gfx942","rules":[]}`);
    silences the JSON-parse warning the hook emits when fed
    `/dev/null`.
  * `_aiter/` — local-only AITER checkout, populated by `--setup`.
    Not committed.
  * `_jit_cache/` — local-only JIT-build cache for AITER's
    `@compile_ops`.  Not committed.
  * `README.md` — this file.

## Relationship to `triton_corpus_runner` and `compare_correctness`

Same `libsalmon_intercept.so`, same Salmon-enabled libhsa, same
`HSA_HOTSWAP_ISA_OVERRIDE=gfx942` plumbing as both other tools.
We share `triton_corpus_runner`'s `.venv-rocm7` so neither runner
maintains its own copy of a torch+rocm7 wheel.

We do **not** share AITER source, the JIT cache, the per-script
glue, or the per-output comparator infrastructure — those are
exactly the surface area this tool trades away in exchange for
"`runner.py --setup` once, then `runner.py` is the whole loop".
