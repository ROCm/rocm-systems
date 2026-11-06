# Out-of-process tracing and counters: proof of concept on gfx1250

**Status: local proof of concept, not for merge.** This branch is rocm-systems `4fbbb315`
(the source of TheRock ROCm 10.2.0a20260922) plus POC changes, so the runtime changes apply
exactly as they were built and measured. It exists to share the code and the findings behind the
API tracing and runtime-contract design discussions (`sdk-future-design`:
`proposals/api-tracing-hld.md`, `proposals/runtime-contract.md`, `api-tracing-plan.md`).

It shows three things working together, with nothing from the profiler running in the application:

1. **The firmware** writes a dispatch log from the CP (KFD `KFD_IOC_PROFILER_DLOG`, stream ABI v4).
2. **The runtime** emits tracepoints (TraceLogging on LTTng): one event per HIP API call and a
   `kernel_dispatch` event.
3. **A separate collector** drains the dispatch log, LTTng records the tracepoints, and a correlator
   joins both into a Perfetto trace.

It also implements per-dispatch counters through runtime-contract section 11 (current text).

## Layout

| Path | What |
|---|---|
| `projects/clr/hipamd/src/tltrace/hip_tltrace.cpp` | `rocm_hip` TraceLogging provider: one event per call for 14 HIP APIs (dispatch-table wrappers installed at the end of `UpdateDispatchTable`) and `kernel_dispatch` |
| `projects/clr/hipamd/src/tltrace/tl/` | microsoft/tracelogging LTTng helper (MIT), as in `users/bewelton/tracelogging-experiments` |
| `projects/clr/hipamd/src/tltrace/rocm_runtime_contract.h`, `hip_rtcontract.*` | Section 11 structs; hook and services tables, registration, allocator, completion signals and cookies, serialization |
| `projects/clr/rocclr/device/rocm/rocvirtual.cpp` | `kernel_dispatch` emission; section 11 hook call; one-reservation write of [serialization barrier] + pre + kernel + post; done-call drains |
| `projects/clr/rocclr/device/rocm/rocdevice.cpp` | done-call drain after event waits |
| `projects/rocprofiler-register/.../rocprofiler_register.cpp` | `hip_runtime_contract` added to the known libraries (RC-PKT-0a) |
| `experimental/oop-tracing-poc/collector/dlogc.cpp` | Out-of-process dispatch-log collector |
| `experimental/oop-tracing-poc/correlator/join.py` | Joins LTTng events and dispatch-log records, writes Perfetto JSON. Pairs `<api>_enter`/`<api>_exit` per thread (the shape decided on Nov 4 and emitted by ROCm/rocm-systems#15122), with correlation and parent ids; also reads this branch's one-event probes |
| `experimental/oop-tracing-poc/counters/poc_selector.cpp` | Counter selector loaded by rocprofiler-register (aqlprofile packets, slot pool, done handling) |
| `experimental/oop-tracing-poc/apps/` | `app.hip` (2 streams, copies), `bench.hip` (overhead benchmarks) |
| `experimental/oop-tracing-poc/scripts/` | Run and benchmark scripts (run inside the container) |
| `experimental/oop-tracing-poc/container/` | Ubuntu 24.04 + LTTng 2.13 image; `c.sh` (user), `croot.sh` (root + `CAP_PERFMON`, for counters) |
| `experimental/oop-tracing-poc/results/` | Benchmark CSVs and a sample trace (`trace-20iter.json`, open in https://ui.perfetto.dev) |

## Setup used

- 4x gfx1250 (MI450-class engineering sample), 256 CPUs, Ubuntu 24.04, kernel 6.14.0-37.
- amdgpu-dkms 7.1.0 (`7.1.0.31300009`), KFD ioctl 1.23, dispatch-log stream ABI v4, MEC firmware 2480.
- Installed ROCm 10.2.0a20260922+bkc.20260929 (rocm-systems `4fbbb315`); rocprofv3 1.4.1 from the
  same commit.
- Everything runs in the container (`container/Dockerfile`): host ROCm mounted read-only, GPU devices
  passed in, LTTng 2.13.11 / lttng-ust 2.13.7. `ulimit -n` must exceed 1024 (lttng-consumerd on 256
  CPUs).

## Build and run (inside the container)

```sh
# runtime with the POC changes (HIP only)
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/rocm -DCLR_BUILD_HIP=ON \
      -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd -DHIP_COMMON_DIR=<repo>/projects/hip \
      -S <repo>/projects/clr -B clr-build && cmake --build clr-build --target amdhip64
# rocprofiler-register with hip_runtime_contract
cmake -G Ninja -DROCPROFILER_REGISTER_BUILD_GLOG=ON -DROCPROFILER_REGISTER_BUILD_FMT=ON \
      -S <repo>/projects/rocprofiler-register -B rpreg-build && cmake --build rpreg-build
# collector and selector need the dispatch-log uAPI from the installed amdgpu-dkms
KFD=/usr/src/amdgpu-<version>/include/uapi/linux
g++ -O2 -std=c++17 -I$KFD collector/dlogc.cpp -o dlogc
g++ -O2 -std=c++17 -fPIC -shared -I$KFD -I<repo>/projects/clr/hipamd/src/tltrace -I/opt/rocm/include \
    -D__HIP_PLATFORM_AMD__ counters/poc_selector.cpp -o libpoc_selector.so \
    -L/opt/rocm/lib -lhsa-amd-aqlprofile64 -lhsa-runtime64 -pthread
```

- Trace + dispatch log + correlation: `scripts/trace_run.sh <out> 2000`, then
  `correlator/join.py <out>`.
- Counters: `LD_LIBRARY_PATH=rpreg-build/lib:clr-build/hipamd/lib
  ROCPROFILER_REGISTER_LIBRARY=libpoc_selector.so POC_SVC_EXECUTABLE=1 [POC_SERIALIZE=1] <app>`,
  as root with `CAP_PERFMON`.

## Results

### API tracing (CPU), `apps/bench.hip`, median of 50 segments

| | Per API call (`hipGetDevice`) | Per kernel launch (enqueue only) |
|---|---|---|
| Stock HIP | 23.1 ns | 1371 ns |
| Probes installed, no session | 32.8 ns (+9.8) | 1414 ns (+43) |
| Probes installed, LTTng recording | 146.9 ns (+124) | 1710 ns (+338; two events per launch) |

No GPU cost (20,716 ns per kernel with and without probes). LTTng discarded 0 events.

### Kernel tracing: tracepoints joined to the dispatch log

- 6,000 of 6,000 dispatches matched and attributed to their API call (2,000-iteration run).
- Join key without KFD changes: `(doorbell_off & 1023, dispatch_id)` from the log equals
  `(doorbell slot, packet_index & 0xffffffff)` from the runtime (as rocprofiler-sdk's in-process
  reader does).
- GPU clock fit: 99.85 MHz, max residual 75-378 ns. Probe to GPU start: median 2.7 us; one dispatch
  in 6,000 starts before its probe because the probe is emitted after the doorbell.
- Dispatch log, CPU: no measurable cost; the collector kept up at ~250k dispatches/s with 0 loss.

### Dispatch log, GPU cost

A fixed ~1.1 us per kernel on every queue: kernel run time unchanged, END-to-next-START gap
1.24-1.44 us with the log vs ~0.33 us without. Same with 1, 2 or 4 streams on 4 CP pipes.

| Kernel | Added per kernel |
|---|---|
| ~6 us | +1.0 us (+17%) |
| ~21 us | +1.1 us (+5%) |
| ~99 us | +1.1 us (+1%) |

### Per-dispatch counters (SQ_WAVES + GRBM_COUNT on every dispatch, one stream)

| Kernel | Stock | rocprofv3 --pmc | Section 11 POC | Section 11 POC + SERIALIZE |
|---|---|---|---|---|
| ~6 us | 6.0 us | 236.3 us (+230) | 114.4 us (+108) | 121.9 us (+116) |
| ~21 us | 20.7 us | 240.8 us (+220) | 131.4 us (+111) | 133.8 us (+113) |
| ~99 us | 99.3 us | 364.7 us (+265) | 209.1 us (+110) | 211.3 us (+112) |

Values agree with rocprofv3 (SQ_WAVES 8 per one-block kernel). Every cookie got exactly one done
call (30 runs x 20,002). With two concurrent streams, SERIALIZE makes every count exact (131,072);
without it some mix.

## Findings for the design

1. **RC-SVC-1 needs an executable flag.** aqlprofile's command buffers are executed by the CP; with
   the allocator as specified (no executable mapping) the GPU faults. With executable allocations
   it works.
2. **~110 us per counted dispatch** remains with full section 11 conformance, about half of
   rocprofv3's. It comes from the counter packets themselves, so per-dispatch counters fit selected
   dispatches, not every one.
3. **The dispatch log costs ~1.1 us of CP idle per dispatch** on gfx1250 (concern #14, "does write
   and confirm block CP?").
4. **One event per API call** costs ~124 ns recording; probes cost ~10 ns per call with no session,
   within RC-GEN-2's 20 ns.
5. One contiguous reservation (RC-PKT-3) also avoided a crash seen when inserting packets around
   the runtime's internal copy kernel with separate reservations.

## Limits

- Hand-written probes for 14 HIP APIs (no generator, no HSA probes, no graph launches).
- Dispatch join by doorbell slot, not the design's KFD queue map; queue reuse not handled; no late
  attach (the collector attaches before the first queue).
- Section 11 not covered: graph replay (RC-PKT-11/12), GRID_UNKNOWN, RC-SER-2's wait for earlier
  work, the serialize_request control event, EXPORTABLE memory and export. One GPU per selector;
  results go to a CSV, not a collector socket.
- One machine, single-threaded benchmarks.
