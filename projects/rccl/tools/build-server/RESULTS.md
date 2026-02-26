# RCCL Build Server: Performance Results

## Machine

- **CPU**: 2x AMD EPYC 9655 96-Core (Turin), 192 physical cores, 384 logical (SMT-2)
- **Memory**: 3 TB DDR5 across 2 NUMA nodes (~1.5 TB each)
- **NUMA distance**: 10 local / 32 remote (3.2x penalty)
- **OS**: Linux 5.15, transparent huge pages enabled

## What the Build Server Does

The RCCL build server replaces the CMake/Ninja build pipeline for `librccl.so`
with a single long-lived process that compiles all translation units in-process
using the Clang and LLVM C++ APIs.  Intermediate artifacts (LLVM IR, assembly
text, ELF objects) are passed between stages as in-memory data structures or
anonymous file descriptors (`memfd_create`), eliminating hundreds of
process-spawn and disk-I/O overheads.

The full pipeline covers:

1. **Device compilation** — Clang frontend (source → LLVM IR), AMDGPU backend
   (IR → assembly), in-memory assembly (assembly → ELF object) for 122 callee
   TUs and 1 kernel TU.
2. **Relocatable link** — in-process LLD (`lld -r`) combining all device objects.
3. **Fat binary generation** — `lld -shared` + `clang-offload-bundler` to
   produce a HIP fat binary (`.hipfb`).
4. **Host compilation** — 103 host C++ TUs compiled in-process via Clang/x86
   backend, plus 1 HIP TU (`onerank.cu.cpp`) as a subprocess.
5. **Host stub** — in-process Clang compile embedding the `.hipfb` into a host
   object.
6. **Final link** — `amdclang++` driver subprocess producing `librccl.so`.

## Task Graph

All work is expressed as a TBB `flow_graph` DAG of `continue_node` tasks:

| Task type         | Count | Description                                  |
|-------------------|------:|----------------------------------------------|
| `callee_fe`       |   122 | Clang frontend for callee device TUs         |
| `callee_be`       |   122 | AMDGPU backend for callee device TUs         |
| `callee_asm`      |   122 | In-memory assembler for callee device TUs    |
| `kernel_fe`       |     1 | Clang frontend for kernel TU                 |
| `kernel_be`       |     1 | AMDGPU backend for kernel TU                 |
| `kernel_patch`    |     1 | Patch kernel assembly with callee metadata   |
| `kernel_asm`      |     1 | In-memory assembler for kernel TU            |
| `lld_r`           |     1 | Relocatable link (all device objects)        |
| `split_cobj`      |     1 | `lld -shared` (device .o → .so)              |
| `split_hipfb`     |     1 | `clang-offload-bundler` (.so → .hipfb)       |
| `split_host`      |     1 | Host stub compile (embeds .hipfb)            |
| `host_compile`    |   103 | Host C++ TUs (in-process Clang/x86)          |
| `host_subprocess` |     1 | `onerank.cu.cpp` (subprocess, full HIP)      |
| `final_link`      |     1 | `amdclang++` link → `librccl.so`             |
| **Total**         | **480** |                                            |

## Memory Optimizations

Running 128 Clang instances in a single address space exposed two
memory-system bottlenecks that do not exist in Ninja's subprocess model:

1. **Allocator contention** — glibc malloc's arena locking became a bottleneck
   under 128 concurrent threads performing heavy allocation (LLVM IR, AST,
   `BumpPtrAllocator` slabs).  Replaced at runtime with TBB's scalable
   allocator via `LD_PRELOAD=libtbbmalloc_proxy.so`, which uses lock-free
   per-thread memory pools.

2. **TLB pressure** — the combined working set of 128 concurrent compilations
   far exceeds TLB capacity at 4 KB page granularity.  Enabled TBB's huge-page
   mode (`TBB_MALLOC_USE_HUGE_PAGES=1`) which uses 2 MB transparent huge pages,
   reducing TLB misses.

NUMA cross-node effects were investigated (3.2x remote-access latency) but
experimentally ruled out: pinning all 128 threads and memory to a single NUMA
node (`numactl -N 1 -l`) produced no improvement (71.9s vs 71.0s).

## Results: Build Server vs. Ninja

**Ninja baseline**: 79.6s (`ninja -j128`, clean build of `librccl.so`)

| Build server configuration            | Threads | Wall time | vs. Ninja |
|----------------------------------------|--------:|----------:|----------:|
| glibc malloc (no tuning)              |     128 |    79.7s  |   −0.1%   |
| tbbmalloc_proxy                       |     128 |    74.2s  |   −6.8%   |
| tbbmalloc_proxy + huge pages          |     128 |    71.0s  |  −10.8%   |
| tbbmalloc_proxy + huge pages          |      96 |    73.1s  |   −8.2%   |
| tbbmalloc_proxy + huge pages          |      64 |    73.1s  |   −8.2%   |
| tbbmalloc_proxy + huge pages          |      48 |    88.1s  |  +10.7%   |

**Best result: 73.1s at 64 threads — 8.2% faster than Ninja using one-third
of the machine's cores.**

Performance scales up to 64 threads and then plateaus completely, indicating
the build is critical-path-limited rather than throughput-limited.

## Critical Path Analysis

The trace was analyzed with a custom DAG reconstruction and scheduling
simulator.  The critical path accounts for 96.5% of actual wall time; TBB's
scheduler achieves only 3.5% slack versus the simulated optimal schedule.

| Metric                      | Value   |
|-----------------------------|--------:|
| Total task work             | 3,207s  |
| Actual wall time            | 73.4s   |
| Critical path               | 70.8s   |
| Simulated optimal (P=64)    | 70.8s   |
| Scheduling slack            | 2.6s (3.5%) |

### Critical path breakdown

| Stage            | TU / description                | Time   |
|------------------|---------------------------------|-------:|
| `callee_fe`      | `all_reduce_minmax_f8e5m2`      | 37.47s |
| `callee_be`      | `all_reduce_minmax_f8e5m2`      | 30.49s |
| `callee_asm`     | `all_reduce_minmax_f8e5m2`      |  1.44s |
| `lld_r`          | Relocatable link                |  0.10s |
| `split_cobj`     | `lld -shared`                   |  0.08s |
| `split_hipfb`    | Offload bundler                 |  0.26s |
| `split_host`     | Host stub compile               |  0.67s |
| `final_link`     | Link `librccl.so`               |  0.28s |
| **Total**        |                                 | **70.8s** |

A single translation unit — `all_reduce_minmax_f8e5m2` — dominates the
critical path at 69.4s (98% of the critical path, 95% of wall time).  Its
frontend alone takes 37.5s.  No amount of parallelism can reduce the build
time below this single TU's compile time.  Further improvement requires either
splitting this TU or optimizing the Clang frontend for it.
