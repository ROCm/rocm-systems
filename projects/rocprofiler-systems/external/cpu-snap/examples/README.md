# cpu_snap examples

Worked demos that drive the cpu_snap component suite end to end. Each
demo is the smallest viable use of one component combination so that
new consumers can copy a single file and adapt. Every demo captures
through a `core::memory_sink` (optionally paired with a file sink for
on-disk persistence) and decodes with `core::sample_reconstructor` -
there is no longer a mechanism split between demos, only what each one
teaches.

## Per-demo summary

| Demo                            | What it shows                                                                 |
|----------------------------------|-------------------------------------------------------------------------------|
| `single_thread_callstack_demo`  | Smallest sampler usage: start, do work, stop, capture via `core::memory_sink` + decode via `core::sample_reconstructor`, write Perfetto. |
| `multi_thread_sample_demo`      | 4-thread workload, `core::memory_sink` capture decoded into per-tid sample vectors, Perfetto post-process. |
| `multi_thread_callstack_demo`   | 4-thread workload streamed through the sampler, replayed via `core::sample_reconstructor` into one pftrace of callstack slices. |
| `backend_comparison_demo`       | Runs the same workload on the perf and signal backends back-to-back and prints a samples/lost/duration comparison table. |
| `inline_frames_demo`            | `symbolizer_config.inline_frames=true` + `resolve_inline()` + `slice_emitter_config.inline_resolver`, expanding always-inlined helpers back into a per-level call chain in the trace. |
| `deep_chain_callstack_demo`     | 8-frame non-inlined call chain, each frame doing a distinctly different amount of its own work, so self/inclusive time visibly diverges by depth. Prints a resolved callstack tree in addition to the Perfetto trace. |
| `autoload/autoload_sampler.cpp` (+ `sample_target.cpp`) | `LD_PRELOAD` flavour: library ctor starts sampler, dtor stops + writes trace. |
| `tools/cpu_snap_run` (`cpu-snap-run`) | Not a demo: runs any program under the sampler (`cpu-snap-run -- ./app args`), one trace per process. See the README. |
| `hip_matmul_demo`               | HIP/ROCm tiled SGEMM; host stacks through hipMemcpy/hipLaunchKernelGGL into libamdhip64/libhsa-runtime64. Opt-in, needs ROCm. |

## HIP matmul demo (optional, needs ROCm/HIP)

`examples/hip_matmul_demo.cpp` runs a tiled SGEMM (N=1024, ~4s of
repeated upload/launch/sync/download passes) so the sampler catches
host-side CPU time spent inside the HIP runtime and HSA runtime, not just
in application code. It is gated behind `CPU_SNAP_BUILD_HIP_EXAMPLES`
(default `OFF`) because most build machines have no ROCm/HIP toolchain;
when `ON` but `hipcc` isn't found, the build emits a warning and skips the
target instead of failing.

Build:
```
cmake -S . -B build_hip -DCPU_SNAP_BUILD_HIP_EXAMPLES=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build_hip --target hip_matmul_demo -j$(nproc)
```

Run (writes a `.pftrace` and, since this demo persists via a dual
file+memory sink, also the raw wire-format bytes to `/tmp/hip_matmul.csnp`):
```
./build_hip/examples/hip_matmul_demo /tmp/hip_matmul_stream_perf.pftrace   --backend perf
./build_hip/examples/hip_matmul_demo /tmp/hip_matmul_stream_signal.pftrace --backend signal
./build_hip/examples/hip_matmul_demo /tmp/hip_matmul_stream_inline.pftrace --backend signal --inline-frames
```
Add `--blas` to route the matmul through `hipblasSgemm` instead of the
custom kernel when hipBLAS was found at configure time.

Other flags: `--freq=HZ` (default 1000), `--unwind=dwarf|fp` (default: dwarf,
the same as auto_best), `--stack-bytes=N` and `--pages=N`
(DWARF snapshot per sample and perf ring size; leave `--pages` unset to size it
from the snapshot), `--filter-stdlib` (drop C++ standard library frames),
`--no-merge` (one full stack per sample instead of merging consecutive samples;
the output is large), `--no-sampler` (run the workload alone and print the
number of passes, to measure sampling overhead).

**Backend note:** both backends can unwind with DWARF while streaming
(`--unwind=dwarf`). Frame-pointer unwinding on perf breaks as soon as a stack
enters ROCm libraries built without frame pointers, so use `--unwind=dwarf`
(or `--backend signal`, which always unwinds with DWARF) to reach all the way
back through libhsa-runtime64 / libamdhip64 to `run_matmul` / `main`. HIP
runtime and LLVM initialisation have frames larger than 8 KiB, so the default
32 KiB snapshot matters: `--stack-bytes=8192` splits `main` in the timeline.

View a trace: open https://ui.perfetto.dev/, "Open trace file", pick one
of the `/tmp/hip_matmul_stream_*.pftrace` files.

## Intentional gaps

- **autoload persistence.** `LD_PRELOAD` autoload keeps a single
  in-process `core::memory_sink`: the library ctor opens it, the dtor
  decodes and serialises a trace on exit. A variant that also persists
  the raw stream to disk while the host process runs would need a
  long-lived daemon to drain the sink concurrently, which is out of
  scope for the bundled demos.
