# Future Work

Deferred follow-ups. Not blocking anything; captured here so they don't get
lost.

## IHSARuntime abstraction + HSA-free unit tests

Source: Phase 4 of the modularity refactor (plan line 103). The eight
component extractions (`HSAPoolManager`, `KernargPool`,
`TraceBufferAllocator`, `PersistentBufferCache`, `DispatchRegistry`,
`SignalMonitor`, `LoadedKernelCache`, `ProfilingResultsSink`) all landed,
and `TracingEngine.cpp` went from 1171 → 622 lines. One plan item remains.

### What's missing

Every extracted component still talks to `libhsa-runtime64` directly via
`hsa_agent_iterate_agents`, `hsa_amd_memory_pool_*`, `hsa_signal_create`,
`hsa_signal_load_relaxed`, `hsa_signal_destroy`, etc. This means their
tests must either run on a GPU host (`*_gpu_test` targets) or skip the
HSA-touching paths entirely. Several Phase 4 classes have no unit test at
all for exactly this reason.

### Proposed work

1. New header `include/aegisbit/HSARuntime.h` defining a small abstract
   interface covering only the HSA entry points we actually call today:

   - `iterateAgents(callback) -> Status`
   - `agentGetInfo(agent, attr, out) -> Status`
   - `amdAgentIterateMemoryPools(agent, callback) -> Status`
   - `amdMemoryPoolGetInfo(pool, attr, out) -> Status`
   - `amdMemoryPoolAllocate(pool, size, flags, out) -> Status`
   - `amdMemoryPoolFree(ptr) -> Status`
   - `amdAgentsAllowAccess(numAgents, agents, flags, ptr) -> Status`
   - `signalCreate(value, numAgents, agents, out) -> Status`
   - `signalDestroy(signal) -> Status`
   - `signalLoadRelaxed(signal) -> value`
   - `signalStoreRelaxed(signal, value)`

2. `RealHSARuntime` — thin pass-through to the real C API; the default
   instance, used by `TracingEngine::getInstance()`.

3. `MockHSARuntime` under `test/unit/mocks/`:
   - In-memory fake agents + pools keyed by synthetic `uint64_t` handles.
   - Allocator backs buffers with `malloc`/`free` so tests can inspect
     writes.
   - Signals tracked in a map with settable `load` values so tests can
     drive `SignalMonitor::loop` deterministically.

4. Plumb the interface through existing extracted classes. Every one
   already takes `HSAPoolManager&` or owns its HSA calls in one place;
   parameterizing on `IHSARuntime&` is a one-constructor change each.

5. `TracingEngine::createForTest(Deps)` — a back-door constructor accepting
   the component set (patcher, launcher, HSA runtime, sinks…), so the
   engine can be exercised in gtests that drive `onDispatch` +
   `onDispatchComplete` against mock HSA.

6. New unit tests:
   - `HSAPoolManagerGTest` — discovery against `MockHSARuntime`.
   - `TraceBufferAllocatorGTest` — fallback path (fine-grained → kernarg),
     explicit CPU-access grant observed on VRAM pools.
   - `KernargPoolGTest` — async init + acquire/release thread safety.
   - `SignalMonitorGTest` — completion and timeout callbacks driven by
     mock signal values.
   - `DispatchRegistryGTest` — already unit-testable today, no mock needed
     (no HSA dep), could land standalone.
   - `TracingEngineGTest` extensions — end-to-end dispatch happy path
     with mock runtime + fixture patched kernel.

### Why defer

- Today's HSA calls all live behind `HSAPoolManager`, `TraceBufferAllocator`,
  `KernargPool`, and `SignalMonitor` — swapping them in is mechanical but
  touches every Phase 4 component at once. Safer as its own commit after
  the current refactor settles on `main`.
- `TracingEngine.cpp` is already at 622 lines and clear enough to read; the
  "coordinator <300 lines" target in the plan was aspirational and not a
  correctness requirement. This work is about testability, not size.
- Pre-existing GPU-dependent test failures (`HSAPoolGTest`,
  `HSAPoolGPUGTest`, `ZeroSGPRSpillGPUTest`) are environment issues, not
  fixable by this abstraction — they need a working GPU.

### Size estimate

- `HSARuntime.h` + `RealHSARuntime.cpp`: ~150 lines.
- `MockHSARuntime`: ~200 lines.
- Plumbing changes to the 4 extracted classes: ~30 lines total.
- `TracingEngine::createForTest` + friend declarations: ~20 lines.
- New gtests: ~400 lines across 5 files.

Estimated effort: one focused session.
