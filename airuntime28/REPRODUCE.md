# Reproducing

Everything in [REPORT.md](REPORT.md) comes from one command. The rest of this file is for
running pieces of it, or for the parts that need a patched CLR build.

## Layout

```
src/common/         the measurement core - one definition of each concept
  config.h            cache facts, flush size, dispatch geometry, statistics settings
  check.h             HIP error checking; assertions that fail a run loudly
  stats.h             median, percentile, paired bootstrap, the significance rule
  harness.h           shuffled slots, duplicate arms, event timing, provenance banner
  kernels.h           read sweep, dependent-load chase, flush scratch
  geometry.h          production dispatch geometry, byte-exactness check
  variants.h          the nine blit variants and the ISA each must emit
src/experiments/    one thin main per question
remote/             build, run and inspect, on the target
results/<stamp>/    one self-describing result set per run
```

## The one command

On the target, in `~/airuntime28`:

```bash
./build.sh          # all experiments
./run_all.sh        # everything, into results/<UTC timestamp>/
./run_all.sh quick  # same coverage, fewer iterations, for a smoke test
```

`run_all.sh` writes `provenance.txt` (machine, ROCm, revision, clocks, both cache sizes,
configured constants), `isa_check.txt`, one `.txt` per experiment, `rows.tsv` with every
machine-readable row, and `summary.txt` with the numbers the report quotes. It exits non-zero
if the ISA check fails or any experiment trips an assertion.

Report tables are regenerated from `rows.tsv` rather than hand-copied. Two tables in an earlier
revision had already gone stale by hand-copying, which is the reason the format exists.

To run part of it:

```bash
./run_all.sh only concurrency isolated_copy
```

## From this machine

```powershell
$env:AIRUNTIME28_HOST = "user@host"   # set once; rsh.ps1 and sync.ps1 both read it
.\sync.ps1                            # normalise to UTF-8/LF and push
.\rsh.ps1 -Command "cd ~/airuntime28 && ./build.sh && ./run_all.sh"
```

`sync.ps1` exists because the editor on this machine writes UTF-16LE without a BOM and
PowerShell rewrites LF to CRLF when piping to a native process; bash and hipcc both reject the
result. Normalising in one place means no future edit has to remember. `rsh.ps1` ships the
command base64-encoded for the same reason, and filters the target's login banner.

## Individual experiments

Each takes `--iters` and `--warmup`. Iteration counts differ by experiment because a case
timing 12 us needs far more repeats to resolve a percent than one timing 5 ms; the defaults in
`run_all.sh` reflect that.

```bash
# The headline table: nine variants, cold 1 GiB copy, plus the controlled comparisons.
# Verifies byte-exactness of all nine variants before timing anything.
./build/isolated_copy --iters 25 --warmup 8
./build/isolated_copy --size-mib 128          # a different size
./build/isolated_copy --warm                  # no flush before each sample

# Where in the size range the hint does anything. --sizes-mib probes a feature of the
# curve at finer spacing than a full sweep is worth running.
./build/size_curve --iters 25 --warmup 8
./build/size_curve --sizes-mib 8,12,16,20,24,32,48

# How big the cache actually is, and whether anything survives a dispatch.
./build/cache_capacity --iters 15 --warmup 4
./build/cache_capacity --capacity-only
./build/cache_capacity --residency-only --flush-mib 256

# Is "cold" actually cold? Sweeps the flush from 128 MiB to 2 GiB.
./build/flush_sensitivity --iters 20 --warmup 5

# A copy against a co-running cache-sensitive kernel, sweeping its working set.
# This is the only scenario where the change pays.
./build/concurrency --iters 25 --warmup 6
./build/concurrency --copy-mib 256

# Nine mechanisms by which the hint could lose, at three footprints relative to GL2.
# Use a high iteration count: the short cases have no resolving power without it.
./build/adversarial --iters 120 --warmup 15

# Dispatch-overhead-dominated sizes, with and without a second kernel object.
./build/small_copy --iters 120 --warmup 10

# Cross-dispatch retention by allocation kind.
./build/residency --iters 25 --warmup 5
```

## Checking the variants compile to what they claim

```bash
./isa_check.sh
```

Expectations are declared in `src/common/variants.h` and read out of the binary with
`./build/isolated_copy --print-isa-expectations`, so the script has no table of its own and
adding a variant needs no change to it. The script also asserts that the support kernels carry
no temporal hints, since a probe with its own cache policy is not a neutral probe. Exits
non-zero on any mismatch; `run_all.sh` runs it first and says so loudly if it fails.

To see the shipped kernel rather than the transcription — extracts the real
`BlitLinearSourceCode` blob out of `blitcl.cpp` and compiles it for gfx1250, so the check
cannot drift from what ships:

```bash
./validate_kernel.sh
```

## Machine and topology

```bash
./machine_state.sh          # host, ROCm, agents, clocks, other tenants
./arch_check.sh             # partition mode, XCC/AID layout
./kfd_cache.sh              # the KFD cache descriptor, including its zeroed fields
./clock_watch.sh 120        # clock distribution over 120s; this part cannot be clock-pinned
```

## The parts that need a patched CLR

The change is commit `81e65d6bbb` on `users/victzhan/AIRUNTIME-28-nt-blit`.
`clr_setup.sh` stands up a worktree on that branch, falling back to applying
`airuntime28-nt-blit.patch` at the base commit if the branch is not present; `clr_build.sh`
builds into `~/airuntime28-clr-install`.

The patch is generated from the commit, never hand-edited. After amending the commit, run
`./clr_patch.sh`, which rewrites it and then verifies that applying it to the base reproduces
the commit exactly.

```bash
./clr_setup.sh && ./clr_build.sh
```

Then:

```bash
./e2e_run.sh 30
```

This runs `hipMemcpyAsync` with `DEBUG_CLR_BLIT_NONTEMPORAL` off, off again, and on. The second
off run is the null control: the flag is read once at runtime init and cannot be changed inside
a process, so this is a between-process comparison and much noisier than everything else here.
Any off-versus-on gap smaller than the off-versus-off gap means nothing. The script also
confirms, from `AMD_LOG_LEVEL=4` output, that the flag actually switches which kernel is
dispatched — if `copyBufferNT` appears at both settings or neither, the comparison is void.

```bash
./fence_check.sh            # is the dispatch fence scope what removes cache residency
./hiptests_build.sh         # build the hip-tests memory suites against the patched runtime
./hiptests_run.sh           # run them with the flag off and on
```

## Reading the output

- **Negative is faster.** Every effect is expressed as a paired median difference in percent.
- **`(ns)` = "not separable". Do not quote the number.** Two bars have to be cleared for a
  result to count: its 95% interval must exclude zero, *and* its size must exceed the run's
  resolution limit. `(ns)` means it failed at least one. The resolution limit is the widest
  difference the rig reports between two slots running the *same* arm, where the true answer is
  zero — so an effect no bigger than that is indistinguishable from the rig measuring nothing,
  however tight its interval looks. It is a magnitude threshold, not a plus-or-minus band.
- **`res_lim` bounds the negative results too.** A case reporting "no difference" at a 8 pp
  resolution limit has not ruled out a 5% regression.
- **Absolute ms and GB/s in the 16-48 MiB band describe a run, not the hardware** — see the
  limits section of [REPORT.md](REPORT.md). Paired columns are unaffected.
