# perf-dkms Design (Current Implementation)

**Last Updated:** 2026-04-10  
**Source of truth:** `experimental/perf-dkms/src/*` (implementation-first)  
**Scope:** This document describes the design that is currently implemented, including known reliability risks.

---

## 1. Purpose and Scope

`perf-dkms` provides an AMD GPU PMU kernel module (`amdgpu_pmu`) that exposes GPU hardware counters through Linux perf.

Primary goals in current implementation:
- Integrate GPU counters with standard perf workflows (`perf stat`, `perf record`)
- Submit PM4 commands through KFD/AQL queue plumbing
- Support dimension-aware counter reads (SE/SA/WGP etc.)
- Operate safely in atomic perf callback contexts by deferring heavy work to a workqueue

This is an **experimental** design with known corner cases and lifecycle hazards (documented below).

---

## 2. High-Level Architecture

```text
Userspace
  perf_event_open / perf stat / perf record
        |
Linux perf core
        |
amdgpu_pmu (pmu_main.c)
  - event_init/add/start/stop/read/del callbacks
  - hrtimer polling for cache refresh
        |
AQL PMU integration (aql_pmu_integration.c)
  - one global RCU-protected session
  - one global workqueue
        |
AQL session + measurement management (aql_perf.c + aql_packet_ops.c)
  - per-event measurement objects
  - shared counter tracking
  - PM4 START/READ/END packet creation/submission
        |
AQL queue/KFD bridge (aql_queue_manager.c + kfd_ioctl_bridge.c)
  - queue create/destroy, ring submit/wait
  - kernel-side ioctl bridging to KFD user-copy API
        |
GPU blocks (SQ/GL2C/TA/GRBM etc.)
```

---

## 3. Component Responsibilities

### `pmu_main.c`
- Registers PMU with perf (`perf_pmu_register`)
- Builds dynamic `events/` sysfs attributes from `counter_registry`
- Parses perf `config1` dimensions and validates limits
- Implements PMU callbacks:
  - `event_init`
  - `add` / `del`
  - `start` / `stop`
  - `read`
- Runs background hrtimer polling for measurement cache refresh
- Emits sampling tracepoints for `perf record` support

### `aql_pmu_integration.c`
- Owns global singleton session pointer (`global_aql_session`) under RCU
- Owns global workqueue used for deferred START/STOP/READ operations
- Provides API for session get/put with refcount protection
- Coordinates flush/cleanup on unload

### `aql_perf.c`
- Creates and initializes session
- Opens `/dev/kfd`
- Discovers GPU IDs via sysfs topology
- Creates per-GPU AQL queues
- Creates per-GPU architecture objects (`arch_t`)
- Binds architecture counter allocations to queue data buffers
- Owns session-level cleanup and final resource release

### `aql_packet_ops.c`
- Creates measurement objects and manages their lifecycle
- Implements PM4 START/READ/END packet generation orchestration
- Allocates/releases physical counters
- Implements current shared-counter mechanism
- Implements async work-item execution model for atomic-safe perf callbacks
- Implements cached atomic read path and synchronous read path

### `aql_queue_manager.c`
- Creates/destroys per-GPU queue resources
- Manages ring buffers and signal buffers
- Submits PM4 IB packets and waits for completion
- Uses `kthread_use_mm`-style access to queue user VA memory mappings

### `kfd_ioctl_bridge.c`
- Bridges kernel code to KFD ioctl handlers expecting user pointers
- Allocates temporary userspace pages, copies args in/out, invokes ioctl

### `aql_c/*`
- Architecture creation and event mapping
- Counter registry
- PM4 builders and packet-generation logic
- Shared helpers for dimensions and topology indexing

### `amdgpu_pmu_trace_dev.c`
- Registers miscdevice `/dev/amdgpu_pmu_trace`
- Allows userspace to emit trace events via ioctls

---

## 4. Core Runtime Objects and Lifetimes

### Global objects
- `global_aql_session` (singleton, RCU-protected)
- `aql_global_workqueue` (singleton)
- Module-global timer in `amdgpu_pmu`

### Per-session objects
- `gpu_ids[]`
- `archs[]` (per GPU)
- `queues[]` (per GPU)
- `active_measurements` list
- `shared_counters` list (current sharing model)

### Per-event object
- `struct aql_measurement`
  - stored in `event->hw.config_base`
  - includes state, counter pointers, cache, refcount, dimension options
  - refcounted because async work can outlive callback frames

### Current counter-sharing object
- `struct shared_counter_ref`
  - keyed by `counter_id` in current implementation
  - contains refcount and pointer to owner measurement

---

## 5. End-to-End Lifecycle Flows

## 5.1 Module init
1. PMU event attrs initialized from `counter_registry`
2. PMU registered with perf
3. AQL integration initialized:
   - global workqueue created
   - global session created and initialized
4. GPU architecture availability validated
5. Trace miscdevice registration attempted

## 5.2 Event init/add/start
1. `event_init` validates event type and counter ID
2. Maps perf CPU affinity to a GPU ID
3. Parses `config1` dimensions, validates against target GPU topology and counter capabilities
4. Creates measurement and stores pointer in `event->hw.config_base`
5. `add(PERF_EF_START)` or `start()` schedules async START work
6. Worker executes START path:
   - enters session mutex
   - puts measurement on active list
   - creates START packet (or uses shared counter path)
   - submits PM4 and waits
   - marks measurement ACTIVE
   - starts polling timer
   - captures baseline counter value for delta math

## 5.3 Read path (counting mode)
- Perf callback `read()` calls `aql_pmu_event_read()`
- `aql_pmu_event_read()` calls `aql_perf_measurement_read_atomic()`
- Atomic read returns cached value immediately and attempts to schedule async READ refresh
- Timer also periodically schedules READ refresh for active measurements

Result: perf-facing reads are mostly cache-based and async-refreshed.

## 5.4 Read path (synchronous worker)
Worker READ does:
1. Create READ PM4 packet
2. Submit PM4 and wait for completion
3. Read result buffer
4. If dimension-specific: index into flattened instance array
5. Else: aggregate all instances for that block
6. Convert absolute value into delta from `start_counter_value`
7. Update measurement cache

## 5.5 Stop/del/destroy
- `stop()` schedules async STOP
- `del()` reads cached final value and calls stop path (atomic-safe behavior considered)
- `event_destroy` invokes measurement destroy
- destroy attempts to ensure STOP via workqueue and then drops creator ref
- final free happens via measurement kref release once no async refs remain

## 5.6 Module exit
1. Cancel timer
2. Unregister PMU (forces perf-side event cleanup)
3. Flush pending AQL work
4. Release session and destroy workqueue
5. Free PMU resources and event attrs

---

## 6. Counter and Dimension Model

### Counter model
- `counter_registry` provides stable logical counter IDs
- Architecture maps logical counter ID to hardware event ID
- Current counters include SQ/GL2C/TA/GRBM groups

### Dimension model
- Perf `config1` encodes xcc/se/sa/wgp/cu/aggregate bits
- Validation checks both topology limits and counter-supported dimensions
- Read path supports:
  - aggregate mode (sum instances)
  - dimension-specific mode (flat index lookup)

### Count semantics
- Returned values are deltas for a measurement interval, not raw hardware absolute counts
- Baseline established at measurement start

---

## 7. Concurrency and Context Model

### Context constraints
- Perf callbacks can run in atomic/no-sleep contexts
- GPU/KFD operations can sleep and require deferred execution

### Mechanisms used
- RCU for global session pointer access
- Session mutex for lifecycle-serialized measurement transitions
- Spinlocks for measurement list, shared-counter list, per-measurement cache
- Global workqueue for deferred START/STOP/READ
- Global submit mutex for serialized PM4 submit+wait sequence

### Work item model
- `aql_create_work_item` takes measurement ref
- Worker executes operation, signals completion (if sync waiter), drops ref, frees item

---

## 8. Current Reliability Risks and Design Debt

The following are present in the current implementation:

1. Shared counter scope mismatch:
- Sharing key is only `counter_id`, not `(gpu_id, counter_id)`.
- Can incorrectly share across GPUs.

2. Shared owner pointer lifetime risk:
- Shared entry stores owner measurement pointer and reuses its counter pointer.
- If owner is destroyed first, stale pointer/UAF class risk exists.

3. Shared stop ownership bug:
- Owner stop can release physical counter while other sharers still active.

4. Timer stop semantics mismatch:
- `stop_timer_if_idle` currently cancels timer unconditionally.
- One stop can halt polling while other measurements are active.

5. READ work dedupe ineffective:
- Each READ queue request allocates a new `work_struct`, so queue-level dedupe does not prevent pile-up.

6. Destroy STOP timeout handling is weak:
- Wait result is not fully enforced; fallback can try direct STOP from user context.

7. MM context handling fragility in queue manager:
- Several hot-path accesses switch mm context without consistent strict gating to kernel-thread-only callsites.

8. Dimension encoding ambiguity for explicit zero:
- `config1 == 0` means unspecified; explicit zero-index requests are ambiguous.

9. Observability debt:
- `total_samples` is initialized/printed but not updated.
- Debug logging defaults are very verbose.

10. Trace device exposure:
- `/dev/amdgpu_pmu_trace` mode is `0666` (world writable), allowing unprivileged trace injection.

---

## 9. Testing Status (Current Tree)

Current tree includes:
- Unit tests for PM4 and packet-generation helpers in `src/aql_c/tests`
- Integration scripts and test apps under `test/` and `test_apps/`

Coverage gaps for current design:
- Cross-process shared-counter lifetime tests
- Multi-GPU sharing correctness tests
- Stress tests for READ workqueue backpressure
- Destroy/unload race and timeout-path validation

---

## 10. Design Direction (High-Level)

To improve simplicity and reliability while preserving cross-program sharing, the key direction is:
- Replace owner-measurement-based sharing with standalone counter-instance ownership keyed by GPU+counter, with explicit refcount transitions controlling START/STOP at `0->1` and `1->0`.

This document intentionally captures the **current** design first; detailed target design changes should be tracked in a separate design-update section or follow-on RFC.

