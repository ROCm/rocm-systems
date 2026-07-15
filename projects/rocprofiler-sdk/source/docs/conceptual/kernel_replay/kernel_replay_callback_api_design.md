# Kernel Replay Callback Tracing API Design

## Overview

Kernel replay is redesigned as a **standalone callback tracing service** under a new
domain `ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY`. This decouples replay from
hardware counter collection, allowing tools to use replay for arbitrary purposes
(counter collection, kernel timing statistics, PC sampling, ATT, etc.).

Tools configure replay through the existing `rocprofiler_configure_callback_tracing_service()`
infrastructure -- no new `rocprofiler_configure_*` function is needed.

## Motivation

The previous API (`rocprofiler_configure_kernel_replay_counting_service()`) was:
- Tightly coupled to dispatch counter collection
- Mutually exclusive with regular dispatch counting on the same context
- Limited to fixed pass counts (no indefinite loop / early exit)
- Unable to give tools per-pass control over which services are active

## API Surface

### Domain

```c
ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY  // added to rocprofiler_callback_tracing_kind_t
```

### Operations

```c
typedef enum rocprofiler_kernel_replay_operation_t
{
    ROCPROFILER_KERNEL_REPLAY_NONE   = 0,
    ROCPROFILER_KERNEL_REPLAY_CONFIG = 1,  ///< Pass-count / loop configuration
    ROCPROFILER_KERNEL_REPLAY_PASS,        ///< Per-pass begin/end notification
    ROCPROFILER_KERNEL_REPLAY_LAST,
} rocprofiler_kernel_replay_operation_t;
```

### Payload Struct

Flat struct (no unions). Which members are meaningful depends on the operation.

```c
typedef struct rocprofiler_callback_tracing_kernel_replay_data_t
{
    uint64_t                           size;
    rocprofiler_kernel_dispatch_info_t dispatch_info;

    // CONFIG fields (writable by tool during CONFIG PHASE_ENTER)
    uint64_t pass_count;
    int (*should_continue)(uint64_t                current_pass,
                           uint64_t                total_passes,
                           rocprofiler_user_data_t user_data);

    // PASS fields (read-only, populated by SDK during PASS operation)
    uint64_t current_pass;
    uint64_t total_passes;
} rocprofiler_callback_tracing_kernel_replay_data_t;
```

### Configuration

```c
rocprofiler_configure_callback_tracing_service(
    context_id,
    ROCPROFILER_CALLBACK_TRACING_KERNEL_REPLAY,
    NULL, 0,
    tool_kernel_replay_callback,
    tool_data);
```

## Callback Flow

```
CONFIG PHASE_ENTER
  tool sets: pass_count, optionally should_continue
  SDK validates: pass_count==0 && should_continue==NULL -> error

  drain queue
  snapshot device memory

  loop (i = 0..pass_count, or indefinitely if pass_count==0):
    PASS PHASE_ENTER  (current_pass=i, total_passes=pass_count)
    submit kernel
    wait for completion
    PASS PHASE_EXIT
    if should_continue provided and returns 0 -> break
    if not last pass -> restore device memory

CONFIG PHASE_EXIT
fire application's original completion signal
```

## Pass Count Semantics

| pass_count | should_continue | Behavior |
|------------|-----------------|----------|
| N > 0      | NULL            | Fixed loop of exactly N passes |
| N > 0      | fn              | Up to N passes; fn can break early |
| 0          | fn (required)   | Indefinite loop until fn returns 0 |
| 0          | NULL            | Error: ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT |

## Interaction with Other Services

Kernel replay is independent of all other services. During a replay pass,
other services (counter collection, kernel tracing, PC sampling, ATT) fire
their callbacks as usual. The tool coordinates which service does what on
each pass through its own internal state management.

## Future Work

- **Localized context control**: `start_context`/`stop_context` function
  pointers in the payload for per-pass context overrides without changing
  global state.
- **Memory snapshot/restore operations**: `ROCPROFILER_KERNEL_REPLAY_SNAPSHOT`
  and `ROCPROFILER_KERNEL_REPLAY_RESTORE` for tool visibility into
  snapshot/restore phases.
- **Pass info delivery to service callbacks**: mechanism for other service
  callbacks to know which replay pass they are in without tool-side TLS.
