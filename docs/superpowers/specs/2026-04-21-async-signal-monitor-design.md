# Async Signal Monitor Design (Queue-Only Phase 1)

Date: 2026-04-21  
Status: Approved for spec write-up  
Owner: rocprofiler-sdk

## 1. Context

`rocprofiler-sdk` currently uses `hsa_amd_signal_async_handler_fn` in multiple components. The first migration target is queue completion in:

- `source/lib/rocprofiler-sdk/hsa/queue.cpp`

The goal is to replace HSA-managed async handler registration with a native rocprofiler signal monitor that can:

- support multithreaded callback execution
- detect signal changes without relying on HSA async handler registration
- remain modular and testable independent of GPU hardware

The design explicitly studies how ROCR `AsyncEventHandler` handles readiness (IOCTL/event wait path vs polling) and keeps both backends available until benchmark data selects a default.

## 2. Goals and Non-Goals

### Goals

- Preserve queue completion behavior during migration.
- Support queue-relevant condition semantics used today: `EQ` and `LT`.
- Optimize for low latency and high throughput; ordering may be out-of-order where callbacks are independently ready.
- Build a benchmark framework that compares IOCTL/event wait and polling backends under the same API and workload.
- Enable hardware-independent functional validation via mocks.

### Non-Goals (Phase 1)

- Migrating all current async-handler call sites (`async_copy`, `profile_serializer`, `device_counting`) in the same patch.
- Expanding condition semantics beyond current usage patterns.
- Making CPU efficiency the primary selection criterion for initial demo evaluation (unlimited CPU assumption for demo).

## 3. Requirements

### Functional

- Register callback for `(signal, condition, threshold, handler)` and trigger once condition is met.
- Support concurrent registration/unregistration and concurrent callback dispatch.
- Guarantee no callback execution after successful cancel/deregister.
- Deterministic shutdown semantics: stop detector, drain callbacks, release resources.

### Performance

- Target low-latency detection with sub-50us typical response for queue-style completion events.
- Sustain high callback throughput under many active signals.

### Reliability

- No dropped callbacks for valid registrations.
- No duplicate callbacks for one-shot registrations.
- No missed wakes in IOCTL wait path (re-check condition around wait boundaries).

## 4. Approaches Considered

1. Hybrid detector stack (IOCTL/event wait with polling fallback) behind one interface.
2. Polling-only detector.
3. IOCTL-only detector.

### Recommendation

Implement both 1 and 2 as selectable backends behind one interface, then decide default via benchmark data.  
Rationale:

- IOCTL/event wait can reduce detection overhead and tail jitter depending on kernel/runtime behavior.
- Polling can be simpler and may outperform under some high-frequency signal-change patterns.
- Shared interface avoids lock-in and enables objective measurement.

## 5. Architecture

### 5.1 Core Interface

Introduce a monitor abstraction used by queue code:

```cpp
class SignalMonitor {
public:
    using SubscriptionId = uint64_t;
    using Callback = std::function<void()>;

    virtual SubscriptionId subscribe(hsa_signal_t signal,
                                     hsa_signal_condition_t condition,
                                     hsa_signal_value_t value,
                                     Callback cb) = 0;
    virtual bool unsubscribe(SubscriptionId id) = 0;
    virtual void start() = 0;
    virtual void stop() = 0;
    virtual ~SignalMonitor() = default;
};
```

Queue code consumes only this API through an adapter layer.

### 5.2 Shared Components

- `SignalRegistry`: thread-safe storage for active subscriptions and state.
- `ConditionEvaluator`: shared logic for `EQ`/`LT` checks.
- `CallbackExecutor`: multithreaded callback dispatch pool.
- `MonitorMetrics`: counters/histograms for latency, throughput, miss/dup/error paths.

These are backend-agnostic and reused by both detectors.

### 5.3 Detector Backends

- `PollingDetector`
  - periodic or adaptive scans over active subscriptions
  - condition check via signal-load path
  - low complexity, predictable behavior

- `IoctlDetector`
  - event-driven wait path modeled after ROCR async event behavior
  - uses event registration/wait primitives where available
  - on wake: re-check condition and dispatch callback
  - includes fallback handling when event path is unavailable

### 5.4 Queue Adapter

`QueueSignalSubscription` binds queue completion use sites to `SignalMonitor`:

- no direct usage of `hsa_amd_signal_async_handler_fn` in queue path
- no backend-specific logic in queue code
- preserves existing completion callback behavior

## 6. Data Flow

1. Queue path registers completion callback via adapter.
2. Registry stores subscription and backend watch state.
3. Detector observes changes (poll scan or IOCTL wake).
4. Condition evaluator validates `EQ`/`LT`.
5. Subscription is atomically marked fired/canceled as appropriate.
6. Callback executor runs handler.
7. Metrics recorded for latency and outcome.

Shutdown flow:

1. Stop accepting new subscriptions.
2. Stop detector loop/waiters.
3. Drain callback executor.
4. Release registry/resources.

## 7. Benchmark Plan (Backend Shootout)

### 7.1 Principle

Benchmark both backends with the same API surface, callback executor, registry, and workload generator to ensure apples-to-apples measurement.

### 7.2 Workloads

- `W1 single-signal ping`: one producer repeatedly updates one signal.
- `W2 multi-signal steady`: round-robin updates across `N` signals.
- `W3 burst`: randomized burst updates with idle gaps.
- `W4 contention`: many producers update different signals concurrently.
- `W5 queue-like`: emulate intercept queue completion timing pattern.

### 7.3 Metrics

- Correctness: missed, duplicate, spurious callbacks.
- Latency: `p50`, `p90`, `p99`, `p99.9`, max (write-to-callback-start).
- Throughput: callbacks/second.
- Shutdown: drain completion time and callback loss during teardown.
- CPU utilization: recorded, but not primary decision gate for demo.

### 7.4 Selection Rule

1. Must pass correctness in all required workloads (`0` misses, `0` duplicates).
2. Lower `p99.9` latency wins.
3. If `p99.9` differs by <= 5%, lower max latency wins.
4. If still tied, higher throughput wins.

## 8. Runtime Configuration and Fallback

Runtime selector:

- `ROCPROF_SIGNAL_MONITOR_BACKEND=ioctl|poll|auto`

Behavior:

- `ioctl`: force IOCTL detector, fail initialization if unavailable.
- `poll`: force polling detector.
- `auto`: attempt IOCTL first, fallback to polling on unsupported or initialization failure.

Operational rules:

- Log selected backend once at startup.
- Log fallback reason once when `auto` falls back.
- Fallback remains non-fatal to preserve profiler functionality.

## 9. Testing Strategy

### 9.1 Hardware-Independent (CI-friendly)

- Unit tests:
  - condition evaluation (`EQ`/`LT`)
  - registration/unregistration races
  - cancel-before-fire and fire-during-cancel behavior
  - shutdown with in-flight callbacks
- Deterministic integration tests with mocks:
  - identical behavior across polling and ioctl adapters
  - randomized timing with deterministic seed
  - high-volume stress without hardware dependencies

### 9.2 Hardware-Backed Validation

- Execute benchmark matrix on target HSA/KFD environment.
- Compare latency distributions and throughput between backends.
- Validate no dropped queue completion callbacks under sustained load.

## 10. Phase 1 Rollout

1. Add monitor abstraction and both detectors.
2. Integrate queue-only path via adapter.
3. Gate backend choice with environment variable.
4. Land with conservative default until benchmark report is captured.
5. Switch default backend in follow-up change based on selection rule evidence.

## 11. Risks and Mitigations

- Missed wake race in IOCTL wait path.
  - Mitigation: waiter bookkeeping and condition re-check before/after wait.
- Callback after cancellation.
  - Mitigation: atomic subscription state transitions and executor guard.
- Queue behavior regression during migration.
  - Mitigation: queue-only scope, existing semantics preserved, differential tests against baseline behavior.
- Benchmark bias due to shared harness defects.
  - Mitigation: validate harness with mock detector and known synthetic traces first.

## 12. Exit Criteria for Phase 1

- Queue path no longer depends on `hsa_amd_signal_async_handler_fn`.
- Both detectors pass hardware-independent correctness tests.
- Backend comparison benchmarks complete with reproducible results.
- Default-backend decision documented and justified by selection rule.
- No observed queue completion regressions in functional validation.
