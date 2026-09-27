# Kernel scheduling hint

A small CPU-side estimate of how long an LLVM-lowered AMDGPU kernel runs, and whether it is ALU-bound, memory-bound, or latency-bound. The enqueue path only does arithmetic. Parsing happens once, when the code object is loaded.

```
kernel: saxpy
bound: memory (hbm), overlap with: alu
duration: 1.742 ms (roofline 612.778 us, range 612.778 us–6.083 ms)
arithmetic intensity: 0.167 flop/byte
waves: 4194304  trip: 1
confidence: class high, time low
```

That duration is a catalog roofline scaled by `1/0.35` until a measured dispatch replaces the scale. The bound is the part a scheduler can use before any timing exists.

## What is already available

Nothing in this tree, and nothing public that is small enough to run at enqueue, takes LLVM's AMDGPU ISA and returns both a duration and a bound class.

| Approach | What it consumes | Why it is not the enqueue hint |
| --- | --- | --- |
| rocprofiler-compute roofline, PerfXpert `roofline.classify` | Performance counters after the kernel has run | The label source for training. The kernel has already finished. |
| Roofline (Williams et al.) | Arithmetic intensity and device peaks | This model is that formula, fed by a static instruction mix. |
| llvm-mca | CPU machine model | No AMDGPU HBM-versus-VALU answer. |
| TVM Ansor / MetaSchedule | Schedule features, XGBoost on CPU | Wrong IR. Relative cost, no bound class. |
| MLGO, IR2Vec, ProGraML | LLVM IR embeddings | Built for inlining and regalloc, or as features to train. No shipped AMDGPU duration model. |
| Habitat, nn-Meter | DNN graphs | Not an arbitrary kernel. |
| PipeWeave (arXiv:2601.14910) | Pipeline demand plus a small MLP | Right shape (analytical features, tiny net, CPU inference). NVIDIA SM oriented, not a library we can call. |
| Computing Frontiers 2026, LLVM IR basic-block counts | Instrumented IR, then a per-GPU regressor | Closest published result (~8% MAPE on ROCm). Predicts time, not the bound. Needs a profiled corpus per kernel family. |
| GPUMech, PPT-GPU, Accel-Sim | Cycle simulation | Far too slow to consult while building an AQL packet. |

An LLM classifier of source text can guess compute-versus-bandwidth and cannot sit on the dispatch path.

## Metadata the runtime already parsed

COMGR metadata is walked in `device::Kernel::GetAttrCodePropMetadata` and `InitParameters` (`projects/clr/rocclr/device/devkernel.cpp`). `roc::Kernel::init` calls that. The keys are the code-object v3 names in `platform/kernel_init.hpp`:

- `.vgpr_count`, `.sgpr_count`, `.vgpr_spill_count`, `.sgpr_spill_count`
- `.wavefront_size`, `.group_segment_fixed_size`, `.private_segment_fixed_size`
- `.kernarg_segment_size`, `.max_flat_workgroup_size`, `.kind`
- `.args`: `.value_kind`, `.size`, `.address_space`, `.access`

`VirtualGPU::submitKernelInternal` in `rocvirtual.cpp` is where a scheduler would read them. At that point the kernel object already holds the signature (argument count and kinds), `workGroupInfo()` (VGPRs, SGPRs, private and group segments), and the call itself has the grid and `sharedMemBytes`. `processMemObjects` in the same file has each global buffer's allocation size. The hint does not parse those fields again.

`amd_comgr_lookup_code_object` reports `amd_comgr_code_object_info_t::size` for the whole ISA blob in a fat binary (`code_object_size`). That is not one kernel's machine-code length. Per-kernel ISA size is the ELF symbol size; HSA's `executable_symbol_get_info` does not return it for a kernel symbol. Pass it in as `isa_size` when the loader has it. `roc::Kernel::postLoad` already resolves that symbol.

`submitKernelInternal` also copies the kernel descriptor (`compute_pgm_rsrc1` / `compute_pgm_rsrc2`) into the metadata packet for the command processor. Those registers encode granulated VGPR and SGPR counts. The runtime does not decode them there; the decoded counts are the COMGR fields already stored on `device::Kernel`.

## What the model does

Two stages.

**Load time.** Walk the lowered ISA once (`llc` output or the LLVM IR still in SSA). Count, per lane: VALU flops by dtype, global / scratch / LDS bytes, atomics, waits, branches, special functions. Count MFMA and WMMA per wave (`2 * M * N * K`). Cache that histogram next to the COMGR record. Argument kinds do not override a real instruction mix: a kernel that takes two global pointers and then only does FMAs is ALU-bound.

**Enqueue time.** `cxx/kernel_sched_hint.hpp` is this step. Scale the histogram by the grid, cap global bytes by the buffer sizes from `processMemObjects`, and limit occupancy the way the hardware does:

- VGPRs, using the same waves-per-EU bins as the PerfXpert tables
- SGPRs, 800 per SIMD on gfx9 (`rocdevice.cpp` `sgprsPerSimd_`), no cap on gfx10+ where SGPRs are not shared
- LDS, `group_segment_fixed_size + sharedMemBytes` against 64 KiB per CU

```
t_valu   = waves * trips * sum(dtype_flops / vector_peak)
t_matrix = waves * trips * sum(mfma_flops / matrix_peak)
t_hbm    = min(requested_global, buffer_cap) + scratch  over  HBM bandwidth
t_lds    = lds_bytes / (128 * CUs * clock)
t_issue  = valu_issues / (CUs * SIMDs * clock)
t_throughput = max(t_valu, t_matrix, t_hbm, t_lds, t_issue)

hide = min(1, waves / resident_waves)
t_latency = dependency_chain * trips * (1 - hide)
roofline = t_throughput + t_latency + 5 µs
duration = t_throughput * scale + t_latency + 5 µs
```

VALU, matrix, and memory are overlapped. That is the optimistic roofline. `scale` is `1/0.35` until `Calibrator` sees a measured dispatch, or until `fit_weights` has a corpus. Peaks are the published MI250X (per GCD), MI300X, MI355X, and RX 7900 XTX numbers; pass a `Device` with measured clocks when you have them.

The bound is whichever term wins. A runner-up from another class within 25% is `mixed`. Latency is its own class: a short grid with a wait chain should not be paired as if it filled HBM.

If the ISA was never walked, the same function still answers from the COMGR record alone: one dword per thread per global buffer, the private segment as scratch, and `isa_size / 4` as an issue count. Class confidence stays low. That path exists so enqueue can say something before a histogram is cached, not because argument count determines arithmetic intensity.

## How a scheduler would use it

`overlap_with` is the scheduling output:

- HBM-bound overlaps with ALU or matrix work, and contends with another HBM-bound kernel
- LDS-bound overlaps with HBM-bound work
- ALU-bound overlaps with memory-bound work
- Latency-bound is short; it should not block a long HBM kernel, and it does not fill a pipe
- Mixed uses both, so it does not pair cleanly

Prefer the bound, the byte and flop counts, and the scale over the absolute microseconds. Clocks move. The first timed dispatch on that device and bound class (`Calibrator.observe`) replaces the 0.35 efficiency prior; absolute time gets honest after that, and the bound was already usable.

Trip count cancels in the ALU-versus-memory comparison. An unknown loop still has a usable label. Only the duration needs the trip count or the one-shot measurement.

## Other directions

- Store the histogram in the code object (a note, or a field next to the COMGR metadata) so the runtime never parses ISA. `submitKernelInternal` would pass the cached `Work` plus the launch into `predict()`.
- Train the residual offline on rocprofiler-compute labels (duration, plus VALU versus memory from the roofline counters). Ship the weights. Do not put a GNN or an LLM on the enqueue path.
- If the residual stays large, the Computing Frontiers basic-block model is the upgrade for duration, and PipeWeave's per-pipeline demand is the upgrade for the bound. Both still infer on the CPU from a small feature vector.
- Requested bytes are not DRAM bytes. When the working set fits in the last-level cache the hint says so and leaves the HBM time as an overestimate.
- Divergence, bank conflicts, and cache reuse inside a loop are invisible to a histogram. Confidence stays limited there; the measured scale is the correction.

## Run

```
cd experimental/kernel-sched-hint
PYTHONPATH=. python3 -m kernel_sched_hint tests/fixtures/saxpy.s \
  --gfx gfx942 --grid 1048576 1 1 --block 256 1 1
PYTHONPATH=. python3 -m unittest tests.test_hint -v
```

`gfx` accepts `gfx942`, `gfx90a`, `gfx950`, `gfx1100`, and `mi300x`, `mi250x`, `mi355x`, `7900xtx`.
