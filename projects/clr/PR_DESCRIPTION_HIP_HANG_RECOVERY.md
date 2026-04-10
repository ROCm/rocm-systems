Add `HIP_HANG_RECOVERY_ENABLE` and `HIP_MAX_SIGNAL_WAIT`. When enabled, `WaitForSignal` forces completion after the configured timeout to avoid indefinite host hangs (e.g. eviction / SDMA reset scenarios).

Default behavior unchanged when `HIP_HANG_RECOVERY_ENABLE=0`.

Made-with: chun-wan

## Motivation

**Core objective:** Expose two opt-in environment variables so applications can turn on bounded waits on HSA completion signals. That lets the host make progress after a stall—by treating the wait as complete—instead of blocking forever. In practice this supports **recovery-oriented** deployments where **rejecting or resetting stalled work** is preferable to a **full process or service hang**.

**Why this matters**

- **Application deadlock:** After **driver eviction** or **SDMA reset**, upper-level stacks often end up stuck in host-side waits (e.g. on signals backing D2H / fence paths) and **cannot shut down cleanly**.
- **RDMA Write / GPU-direct paths:** Customer workloads use **RDMA Write** into device-visible memory. HIP does not fully track that class of traffic; when an exception or reset intersects those paths, the runtime can **hang permanently** waiting on completion that will never arrive in a well-defined way.

**Design stance (production / HA)**

- **Service continuity:** Keeping the process **alive and responsive** outweighs **every single request succeeding**.
- **Graceful degradation:** **Dropping or failing individual requests** is acceptable if the **overall service** stays up and can recover.
- **Resource / liveness:** Forcing completion of a stuck wait is a **last-resort host-side unblock** so threads do not hold the runtime hostage after **hardware or driver-level** faults.

## Technical Details

- **`projects/clr/rocclr/utils/flags.hpp`**
  - `HIP_HANG_RECOVERY_ENABLE` (default `0`): master switch; when `0`, no new timeout logic runs.
  - `HIP_MAX_SIGNAL_WAIT` (default `60`): when recovery is enabled, maximum **seconds** to wait before forcing completion; `0` means **no cap** (same indefinite behavior as before).

- **`projects/clr/rocclr/device/rocm/rocvirtual.hpp` — `WaitForSignal`**
  - Unchanged when `HIP_HANG_RECOVERY_ENABLE` is off: existing 4-second polling loop and `HIP_SKIP_ABORT_ON_GPU_ERROR` / `IsGPUInError()` early-exit behavior remain as today.
  - When enabled and `HIP_MAX_SIGNAL_WAIT > 0`: track elapsed time since entering the wait; if elapsed ≥ limit, call `Hsa::signal_silent_store_relaxed(signal, 0)` and return so the caller can proceed. This **does not** add SDMA health probes, extra file logging, or queue-error suppression—only **bounded signal wait** + **force complete**.

**Note:** Forcing a signal can leave GPU state inconsistent with what the app assumes; this is intentional for the **HA / liveness** trade-off described above. Operators should enable only where that trade-off is acceptable.

## JIRA ID

<!-- If applicable, mention the JIRA ID resolved by this PR (Example: Resolves SWDEV-12345). -->
<!-- Do not post any JIRA links here. -->

_(None yet — add before merge if required by your team.)_

## Test Plan

1. **Default / regression:** Build CLR with HIP as usual. Run with **no** env or `HIP_HANG_RECOVERY_ENABLE=0`. Confirm existing behavior (indefinite wait on stuck signal still possible, same as baseline).
2. **Recovery path:** Set `HIP_HANG_RECOVERY_ENABLE=1` and `HIP_MAX_SIGNAL_WAIT` to a small value (e.g. `5`). Use a workload that previously **blocked in host wait** on a completion signal after eviction/reset (or a targeted internal repro). Confirm the host wait **returns** after the timeout instead of hanging forever.
3. **Infinite cap:** `HIP_HANG_RECOVERY_ENABLE=1` with `HIP_MAX_SIGNAL_WAIT=0` should **not** apply the forced completion path (waits remain unbounded by this feature).
4. **CI / smoke:** Run relevant ROCclr/HIP tests on a standard GPU CI configuration with defaults unchanged.

## Test Result

- **Local / targeted:** _(Fill in after you run the repro above — e.g. “Host wait returned after N s; process remained responsive.”)_
- **CI:** _(Paste pipeline link or “green on …” after upstream CI runs.)_

## Submission Checklist

- [ ] Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.
