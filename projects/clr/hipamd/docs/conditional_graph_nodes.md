# HIP Graph Conditional Nodes — POC Implementation

Branch: `users/agoadavr/conditional-IB-POC`

## Overview

This document describes the proof-of-concept (POC) implementation of HIP graph
conditional nodes (IF and WHILE) on the AMD ROCm stack. The approach uses three
new vendor AQL packet types to let the GPU Command Processor (CP) evaluate a
signal-backed condition and walk a pre-built instruction buffer (IB) of kernel
dispatch packets, without scheduler kernel involvement or driver round-trips.

**Current status:** The CLR + ROCr stack targets the current firmware ABI:
COND_BRANCH is packet type 6, LOOP_BACK is packet type 7, and WHILE remains
lowered as `[body packets | LOOP_BACK]` rather than COND_BRANCH repeat mode.

---

## Public API

### `hipGraphConditionalHandle`

```c
typedef struct hipGraphConditionalHandle_st {
  uint64_t device_ptr;    // GPU-visible address of the condition value cell
  uint64_t default_value; // Value stamped into the cell at handle creation
  uint64_t signal_handle; // Backing hsa_signal_t.handle (runtime-internal)
} hipGraphConditionalHandle;
```

The handle is backed by a real `hsa_signal_t`. `device_ptr` is set to
`signal_handle + offsetof(amd_signal_t, value)` (offset +8, verified by
`static_assert`), so a kernel's `hipGraphSetConditional` store and the CP's
signal load operate on the same memory cell.

### `hipGraphConditionalHandleCreate`

```c
hipError_t hipGraphConditionalHandleCreate(
    hipGraphConditionalHandle* pHandleOut,
    hipGraph_t                 hGraph,
    unsigned int               defaultLaunchValue,
    unsigned int               flags);          // reserved, must be 0
```

Allocates a real `hsa_signal_t`, stores `defaultLaunchValue` into it, and fills
`pHandleOut`. The signal is not yet registered with the graph for lifetime
management (M1 limitation; fixed in the M2 lifetime work).

### `hipGraphAddConditionalNode`

```c
hipError_t hipGraphAddConditionalNode(
    hipGraphNode_t*           pGraphNode,
    hipGraph_t                graph,
    const hipGraphNode_t*     pDependencies,
    size_t                    numDependencies,
    hipGraphConditionalHandle handle,
    hipGraphConditionalType   type,
    unsigned int              numConditionalGraphs,
    hipGraph_t*               phConditionalGraphs);
```

Creates a conditional node and returns `numConditionalGraphs` empty body graphs
into `phConditionalGraphs`. The caller populates the body graphs with kernel
nodes before calling `hipGraphInstantiate`.

| `type`                   | `numConditionalGraphs` | Behaviour                          |
|--------------------------|------------------------|------------------------------------|
| `hipGraphCondTypeWhile`  | exactly 1              | Loop body; exits when cell == 0    |
| `hipGraphCondTypeIf`     | 1                      | No-else IF                         |
| `hipGraphCondTypeIf`     | 2                      | IF / ELSE (`[else_arm][if_arm]`)   |
| `hipGraphCondTypeSwitch` | any                    | Returns `hipErrorNotSupported`     |

### `hipGraphSetConditional` (device function)

```c
__device__ static inline void
hipGraphSetConditional(hipGraphConditionalHandle handle,
                       unsigned long long value);
```

Single volatile store to `handle.device_ptr`. No HW fence beyond the
screlease on the enclosing kernel-dispatch packet. Body kernels call this to
signal whether the loop should continue (`1`) or exit (`0`).

---

## Vendor AQL Packet Types

Defined in `projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h`.
All are 64 bytes and use `HSA_PACKET_TYPE_VENDOR_SPECIFIC` in the AQL header
with the AMD format field encoding the subtype.

| Enum value | Name | Purpose |
|---|---|---|
| 5 | `HSA_AMD_PACKET_TYPE_AQL_INDIRECT_BUFFER` | Generic IB dispatch (reserved; not emitted by this POC) |
| 6 | `HSA_AMD_PACKET_TYPE_AQL_COND_BRANCH` | Entry packet for IF/WHILE nodes |
| 7 | `HSA_AMD_PACKET_TYPE_AQL_LOOP_BACK` | Loop-back tail packet inside a WHILE body IB |

### `hsa_amd_aql_cond_branch_packet_t` (type 6)

| Offset | Field | Description |
|--------|-------|-------------|
| +0  | `header` | Vendor packet header; BARRIER=1, scacquire=screlease=SYSTEM |
| +4  | `cond_op` | `BOOL_TRUE` (10) for HIP IF/WHILE entry |
| +5  | `execution_mode` | `BRANCH_ONCE` (0) |
| +6  | `post_action` | `NONE` (0) |
| +7  | `flags` | Must be 0 |
| +8  | `condition_signal` | `hsa_signal_t` — the cond cell read by the CP |
| +16 | `test_value` | 0 for `BOOL_TRUE` |
| +24 | `true_target_offset_packets` | TRUE-arm start offset in packets |
| +28 | `true_target_size_packets` | Packets to run when condition is TRUE |
| +32 | `false_target_offset_packets` | FALSE-arm start offset in packets |
| +36 | `false_target_size_packets` | Packets to run when condition is FALSE |
| +40 | `ib_base_addr` | Base address of the instruction buffer |
| +48 | `branch_options` | Must be 0 |
| +52 | `ib_size_packets` | Total IB size in packets |
| +56 | `completion_signal` | Optional completion signal (0 for in-graph use) |

### `hsa_amd_aql_loop_back_t` (type 7)

| Offset | Field | Description |
|--------|-------|-------------|
| +0  | `header` | Vendor packet header; BARRIER=1, scacquire=screlease=SYSTEM |
| +4  | `reserved0` | Must be 0 |
| +8  | `condition_signal` | Same signal as the enclosing COND_BRANCH |
| +16 | `test_value` | 1; firmware increments the signal and exits when the result equals this |
| +24 | `cond_op` | Legacy field; HIP leaves this as `NE` |
| +28 | `ib_size_packets` | Total IB size including this packet; CP uses it to rewind the read pointer |
| +32 | `reserved1`–`reserved3` | Must be 0 |
| +56 | `completion_signal` | Optional (0 for in-graph use) |

---

## IB Layout

### WHILE

```
parent queue:
  [ ... ]
  [ COND_BRANCH ]
      condition_signal  = handle.signal
      cond_op           = BOOL_TRUE
      test_value        = 0
      ib_base_addr      = body_ib
      true_offset_pkts  = 0
      true_size_pkts    = N + 1         (N body kernels + 1 LOOP_BACK)
      false_size_pkts   = 0             (0-iteration: skip immediately)
  [ ... ]

body_ib:
  [ KERNEL_DISPATCH body[0] ]   <- body kernels call hipGraphSetConditional
  [ KERNEL_DISPATCH body[1] ]
  ...
  [ KERNEL_DISPATCH body[N-1] ] <- writes 0 to cond cell when done
  [ LOOP_BACK ]
      ib_size_packets = N + 1
```

CP behaviour: on each LOOP_BACK the CP rewinds to `body[0]` and re-runs the
body. The body is responsible for clearing the cond cell when the termination
condition is met; only then does the CP exit the IB and resume the parent queue.

### IF (single body)

```
parent queue:
  [ COND_BRANCH ]
      true_offset_pkts  = 0
      true_size_pkts    = N
      false_size_pkts   = 0

body_ib:
  [ KERNEL_DISPATCH body[0] ]
  ...
  [ KERNEL_DISPATCH body[N-1] ]
  -- no LOOP_BACK; CP returns to parent queue on IB exhaustion --
```

### IF / ELSE (two bodies)

```
body_ib:
  [ KERNEL_DISPATCH if[0]    ]  <- true arm (K packets)
  ...
  [ KERNEL_DISPATCH if[K-1]  ]
  [ KERNEL_DISPATCH else[0]  ]  <- false arm (M packets)
  ...
  [ KERNEL_DISPATCH else[M-1] ]

COND_BRANCH fields:
  true_offset_pkts  = 0     (run true arm on TRUE)
  true_size_pkts    = K
  false_offset_pkts = K     (run else arm on FALSE)
  false_size_pkts   = M
```

---

## Implementation: CLR

### `GraphConditionalNode` (`hip_graph_internal.hpp/.cpp`)

Inherits from `GraphNode` with `hipGraphNodeTypeConditional`.

**Key members:**

| Member | Type | Description |
|--------|------|-------------|
| `handle_` | `hipGraphConditionalHandle` | The condition signal handle |
| `cond_type_` | `hipGraphConditionalType` | IF or WHILE |
| `bodies_` | `vector<Graph*>` | Owned body graph(s) |
| `ib_addr_` | `void*` | Base of the IB in device-accessible memory |
| `ib_size_pkts_` | `size_t` | Total IB packets including LOOP_BACK |
| `true_off_pkts_` | `uint32_t` | `cond_branch.true_target_offset_packets` |
| `true_pkts_` | `uint32_t` | `cond_branch.true_target_size_packets` |
| `false_off_pkts_` | `uint32_t` | `cond_branch.false_target_offset_packets` |
| `false_pkts_` | `uint32_t` | `cond_branch.false_target_size_packets` |

**`BuildIB(kernArgMgr, devId)`** — called from `GraphExec::CaptureAQLPackets()`:

1. Bootstraps a kernarg pool sized for all body kernels (sum of `GetKerArgSize()`
   across all body nodes, plus `kKernArgChunkSize` headroom).
2. Walks each body graph in insertion order, calling
   `GraphNode::CaptureAndFormPacket(kernArgMgr)` on every kernel node.
3. Zeroes `completion_signal` in each captured packet.
4. Allocates the IB with `Device::hostAlloc(kKernArg segment)` so ROCclr uses
   the executable CPU HSA memory pool and grants agent access.
5. `memcpy`s all captured packets into the IB, appending an
   `hsa_amd_aql_loop_back_t` for WHILE.
6. Sets true/false target packet ranges for the parent
   `CondJumpCommand`.

**`CondJumpCommand::submit(device)`** — called at each `hipGraphLaunch`:

1. `hsa_signal_store_relaxed(cond_sig, default_value)` — resets the cell so a
   previous launch's terminal value doesn't carry over.
2. `roc::VirtualGPU::dispatchCondBranchPacket(...)` — emits the vendor packet.

**Copy constructor / `clone()`:** deep-clones each body `Graph` so the
`GraphExec` owns independent body graph instances. IB metadata is
default-initialised; `BuildIB` is re-run on the clone's first instantiation.

### `VirtualGPU::dispatchCondBranchPacket` (`rocvirtual.hpp/.cpp`)

```cpp
void dispatchCondBranchPacket(
    hsa_signal_t cond_signal,
    uint64_t ib_base_addr,
    uint32_t true_offset_pkts,
    uint32_t true_pkts,
    uint32_t false_offset_pkts,
    uint32_t false_pkts,
    uint32_t ib_size_pkts);
```

Assembles `hsa_amd_aql_cond_branch_packet_t` with:
- `BARRIER=1`, `scacquire=screlease=HSA_FENCE_SCOPE_SYSTEM`
- `cond_op = HSA_AMD_AQL_COND_BRANCH_COND_BOOL_TRUE`, `test_value = 0`

Reserves a slot on the HW queue via `hsa_queue_add_write_index_screlease`,
writes the packet at the ring-buffer location, and rings the doorbell.

### `GraphExec::CaptureAQLPackets` hook (`hip_graph_internal.cpp`)

After capturing all regular node packets, iterates `topoOrder_` and calls
`BuildIB(kernArgManager_, instantiateDeviceId_)` on every
`hipGraphNodeTypeConditional` node. This ensures the kernarg pool is already
allocated before body-kernel kernarg allocation runs inside `BuildIB`.

---

## Memory and Ordering

| Concern | Mechanism |
|---------|-----------|
| IB allocation | `deviceLocalAlloc` (largeBar VRAM) or `hostAlloc(kKernArg)` (pinned) |
| IB visibility to CP | Flagged uncached on host side; CP fetches through GFX cache |
| Cond cell update (device) | Volatile store in `hipGraphSetConditional`; ordered by body kernel's screlease |
| Cond cell reset (host) | `hsa_signal_store_relaxed` in `CondJumpCommand::submit` before each launch |
| Packet ordering | BARRIER=1 + SYSTEM fences on both COND_BRANCH and LOOP_BACK packets |

---

## Profiler / Tracing Integration

The two new API entry points are wired through the full HIP tracing stack:

| Layer | File |
|-------|------|
| Dispatch table | `hip_api_trace.hpp` — `t_hipGraphConditionalHandleCreate`, `t_hipGraphAddConditionalNode` function pointer slots; `HIP_RUNTIME_API_TABLE_STEP_VERSION` bumped to 30 |
| ABI enforcement | `hip_api_trace.cpp` — `HIP_ENFORCE_ABI` entries at offsets 537 / 538 |
| Profiler IDs | `hip_prof_str.h` — `HIP_API_ID_hipGraphConditionalHandleCreate = 480`, `HIP_API_ID_hipGraphAddConditionalNode = 481` |
| Version script | `hip_hcc.map.in` — exports both symbols under `hip_7.2` |
| Table interface | `hip_table_interface.cpp` — initialises both function pointer slots |

---

## Validation

Three standalone harnesses (built with plain `hipcc`, not part of hip-tests)
were used to validate the implementation:

### `conditional_handle_smoke.cpp` (M1 host-side)

Validates handle plumbing without launching any graph:

- `device_ptr == signal_handle + 8` (Option C layout confirmed)
- `hsa_signal_load_relaxed` reads back `defaultLaunchValue` correctly
- `hipGraphAddConditionalNode` argument validation: null output pointer →
  `hipErrorInvalidValue`; SWITCH → `hipErrorNotSupported`; WHILE with 2 bodies
  → `hipErrorInvalidValue`; WHILE with 1 body → `hipSuccess`; IF with 2 bodies
  → `hipSuccess` with distinct non-null body graphs returned

### `hip_graph_set_conditional_kernel_test.cpp` (device write path)

Launches `set_conditional_kernel<<<1,1>>>` which calls
`hipGraphSetConditional(h, value)`. After `hipDeviceSynchronize`, verifies the
updated value is observable host-side via `hsa_signal_load_relaxed`. Confirms
the volatile store in the kernel reaches the same memory cell the host signal
API reads.

### `hip_graph_conditional_ib_test.cpp` (end-to-end IB path)

Exercises `hipGraphInstantiate` + `hipGraphLaunch` for WHILE and IF nodes. All
tests fail at `hipStreamSynchronize` with `hipErrorLaunchFailure (719)` in the
expected pattern: the CP aborts the queue on the vendor COND_BRANCH packet with
`HSA_STATUS_ERROR_INVALID_PACKET_FORMAT (0x1009)` (amd_format=6). This matches
the documented firmware boundary exactly.

---

## File Map

| Concern | File |
|---------|------|
| Public API types and declarations | `projects/hip/include/hip/hip_runtime_api.h` |
| `hipGraphSetConditional` device inline | `projects/hip/include/hip/hip_runtime_api.h` |
| Handle creation + node creation | `projects/clr/hipamd/src/hip_graph.cpp` |
| `GraphConditionalNode` class definition | `projects/clr/hipamd/src/hip_graph_internal.hpp` |
| `BuildIB`, `CondJumpCommand::submit` | `projects/clr/hipamd/src/hip_graph_internal.cpp` |
| `dispatchCondBranchPacket` | `projects/clr/rocclr/device/rocm/rocvirtual.hpp/.cpp` |
| Vendor AQL packet types (ROCr) | `projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h` |
| AQL packet name printer (ROCr) | `projects/rocr-runtime/runtime/hsa-runtime/core/inc/queue.h` |

---

## Open Items

1. **WHILE repeat mode** — WHILE still uses `[body packets | LOOP_BACK]`; it is
   not lowered to COND_BRANCH repeat mode yet.
2. **ROCr packet definitions** — the ABI definitions are present in this branch
   and should be kept in sync with the firmware-owned packet layout.
3. **Handle lifetime** — the backing `hsa_signal_t` is not yet destroyed when
   the owning graph is destroyed (M1 limitation).
4. **Non-kernel body nodes** — `BuildIB` returns `hipErrorNotSupported` for any
   body node that is not a kernel node. Memcpy, memset, and child-graph nodes
   are not yet supported.
5. **IF/ELSE** — `hipGraphCondTypeIf` with 2 bodies compiles and instantiates
   but is tagged `[!mayfail]` in tests pending CP firmware support.
6. **`hipGraphCondTypeSwitch`** — returns `hipErrorNotSupported`; not planned
   for the immediate follow-on.
