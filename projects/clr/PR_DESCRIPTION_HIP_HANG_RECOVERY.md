# HIP hang recovery: bounded signal wait (`HIP_HANG_RECOVERY_ENABLE`, `HIP_MAX_SIGNAL_WAIT`)

## Summary

This change adds two opt-in environment variables that bound how long the ROCclr/ HIP host side waits on HSA completion signals. When the limit is reached, the wait is forced to complete so the process can make progress instead of blocking indefinitely.

---

## Core Objective

The primary goal is to enable users to utilize `HIP_HANG_RECOVERY_ENABLE` and `HIP_MAX_SIGNAL_WAIT`. This allows applications in specific scenarios to activate recovery mechanisms, permitting requests or kernel states to be rejected or reset rather than causing a total system hang.

---

## Current Pain Points & Technical Constraints

### Application Deadlock

Currently, when a Driver Eviction or SDMA Reset occurs, upper-level applications typically hang and cannot exit gracefully.

### RDMA Write Vulnerability

The issue is exacerbated because the customer is utilizing RDMA Write operations. Since HIP runtime cannot effectively manage or track the state of these specific operations, any exception triggered during the process easily leads to a permanent hang of the runtime.

---

## Key Requirements & Design Philosophy

### Service Continuity

The customer's priority is ensuring service uptime. Preventing the online environment from crashing or freezing is more critical than individual request success.

### Graceful Degradation

In high-availability scenarios, dropping individual user requests is an acceptable trade-off to ensure the overall service remains responsive and recoverable.

### Resource Management

By allowing the runtime to reject or reset stalled kernel states, the system can clear blocked resources and maintain "liveness" even after hardware or driver-level faults.

---

## Technical behavior

| Variable | Default | Meaning |
|----------|---------|---------|
| `HIP_HANG_RECOVERY_ENABLE` | `0` (off) | When set to `1`, enables bounded waiting on HSA signals in `WaitForSignal`. |
| `HIP_MAX_SIGNAL_WAIT` | `60` | With recovery enabled, maximum seconds before the runtime forces signal completion (`0` = wait indefinitely, same as legacy behavior). |

Default behavior is unchanged when `HIP_HANG_RECOVERY_ENABLE` is unset or `0`.

---

## Files touched

- `rocclr/utils/flags.hpp` — register the two flags.
- `rocclr/device/rocm/rocvirtual.hpp` — `WaitForSignal`: optional timeout loop and `signal_silent_store_relaxed` to unblock the host wait.
