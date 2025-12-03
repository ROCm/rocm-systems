# perf-dkms Kernel Module - Developer Quickstart Guide

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Setup Flow](#setup-flow)
3. [Reading Flow](#reading-flow)
4. [Teardown Flow](#teardown-flow)
5. [Key Data Structures](#key-data-structures)
6. [Counter Sharing](#counter-sharing)
7. [Dimension-Specific Monitoring](#dimension-specific-monitoring)

---

## Architecture Overview

### Layer Stack

The perf-dkms kernel module provides GPU performance counter monitoring through the Linux perf subsystem. The architecture consists of multiple layers:

```
┌─────────────────────────────────────────────────────────────────┐
│                      Userspace (perf tool)                       │
│   Example: perf stat -C 0 -e amdgpu_pmu/sq_waves/               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              Linux Perf Subsystem (kernel/events/)               │
│  PMU callbacks: event_init, add, del, start, stop, read         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│              PMU Integration Layer (pmu_main.c)                  │
│  - Registers PMU with perf subsystem                             │
│  - Routes event operations to AQL layer                          │
│  - Manages background polling timer                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│          AQL Integration Layer (aql_pmu_integration.c)           │
│  - Global session management                                     │
│  - GPU selection and mapping                                     │
│  - Workqueue for atomic context handling                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         Measurement Management (aql_packet_ops.c, aql_perf.c)    │
│  - Counter allocation & sharing                                  │
│  - PM4 packet generation (START/READ/STOP)                       │
│  - Reference counting for safe cleanup                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│         PM4 Packet Generation (aql_c/packet_generation.c)        │
│  - Generate START packets (configure counters)                   │
│  - Generate READ packets (read counter values)                   │
│  - Generate STOP packets (disable counters)                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│           KFD Submission Layer (kfd_ioctl_submit_ib_packet)      │
│  - Submits PM4 command buffers to GPU                            │
│  - Synchronous execution (blocks until GPU completes)            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      GPU Hardware                                │
│  - Executes PM4 commands                                         │
│  - Configures performance counters                               │
│  - Writes counter values to memory                               │
└─────────────────────────────────────────────────────────────────┘
```

### File Organization

```
projects/perf-dkms/
├── src/
│   ├── pmu_main.c               # PMU driver entry point & callbacks
│   ├── pmu_events.c             # Event validation & utilities
│   ├── pmu_dimension.h          # Dimension parsing & validation
│   ├── aql_pmu_integration.c    # Global session & GPU mapping
│   ├── aql_perf.c               # Session & GPU discovery
│   ├── aql_packet_ops.c         # Measurement & packet creation
│   ├── aql_error_recovery.c     # Error handling & recovery
│   └── aql_c/
│       ├── packet_generation.c  # PM4 packet builders
│       ├── pm4_packets.c        # Low-level PM4 primitives
│       ├── counter_registry.c   # Counter definitions
│       ├── arch_creator.c       # Architecture factory
│       ├── gfx12_creator.c      # GFX12-specific architecture
│       └── gfx12_events.c       # GFX12 event mappings
└── QUICKSTART.md                # This file
```

### Component Relationships

```
┌──────────────────────────────────────────────────────────────────┐
│                       amdgpu_pmu (singleton)                      │
│  - struct pmu (Linux perf PMU)                                    │
│  - hrtimer for background polling                                 │
│  - Statistics tracking                                            │
└──────────────────────────────────────────────────────────────────┘
                              │ owns
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│             global_aql_session (singleton)                        │
│  - kfd_file, kfd_process (KFD integration)                        │
│  - gpu_ids[] (discovered GPUs)                                    │
│  - archs[] (architecture per GPU)                                 │
│  - active_measurements (list)                                     │
│  - shared_counters (list for sharing)                             │
└──────────────────────────────────────────────────────────────────┘
                              │ contains
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│              aql_measurement (one per perf event)                 │
│  - perf_event* (back-reference)                                   │
│  - allocated_counter* (hardware counter)                          │
│  - shared_ref* (for counter sharing)                              │
│  - state, cached_counter_value                                    │
│  - kref (reference counting)                                      │
└──────────────────────────────────────────────────────────────────┘
                              │ uses
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│         arch_t (architecture, one per GPU)                        │
│  - block_map (SQ, GRBM, TA, GL2C, TCC blocks)                     │
│  - control_regs (register offsets)                                │
│  - num_se, num_sa, num_wgp_per_sa                                 │
└──────────────────────────────────────────────────────────────────┘
                              │ contains
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│           block_info_t (per hardware block)                       │
│  - counter_reg_info[] (array of counters)                         │
│  - dimension[] (SE/SA/WGP dimensions)                             │
│  - instance_count (for GL2C/TCC)                                  │
└──────────────────────────────────────────────────────────────────┘
                              │ contains
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│      counter_reg_info_t (per hardware counter)                    │
│  - select_addr, control_addr (config registers)                   │
│  - register_addr_lo, register_addr_hi (result)                    │
│  - allocation (state, event_id, user_id)                          │
│  - command_buffer, data_buffer (GPU memory)                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Setup Flow

When a user runs `perf stat -e amdgpu_pmu/sq_waves/ -C 0`, the following sequence occurs:

### Step-by-Step Initialization

```
User Command
    │
    ▼
perf_event_open() syscall
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. amdgpu_pmu_event_init() [pmu_main.c:379]                     │
│    - Validates event configuration                               │
│    - Extracts dimension parameters from config1                  │
│    - Maps CPU to GPU ID                                          │
│    - Calls aql_pmu_event_init()                                  │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. aql_pmu_event_init() [aql_pmu_integration.c:285]             │
│    - Checks AQL availability                                     │
│    - Selects target GPU                                          │
│    - Creates aql_measurement structure                           │
│    - Stores measurement in event->hw.config_base                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. aql_perf_measurement_create() [aql_packet_ops.c:620]         │
│    - Allocates measurement structure                             │
│    - Initializes state to MEASUREMENT_IDLE                       │
│    - Sets counter_id from event->attr.config                     │
│    - Initializes reference count (kref) to 1                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. amdgpu_pmu_add() [pmu_main.c:504]                            │
│    - Adds event to PMU (called by perf core)                     │
│    - If PERF_EF_START flag set, calls aql_pmu_event_start()     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. aql_pmu_event_start() [aql_pmu_integration.c:391]            │
│    - Queues START work on global workqueue                       │
│    - Returns immediately (atomic context safe)                   │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. aql_work_handler() [aql_packet_ops.c:1289]                   │
│    - Dequeued from workqueue (process context)                   │
│    - Calls aql_perf_measurement_start()                          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. aql_perf_measurement_start() [aql_packet_ops.c:682]          │
│    - Acquires session mutex                                      │
│    - Adds to active_measurements list                            │
│    - Calls aql_perf_create_start_packet()                        │
│    - Submits PM4 packet via kfd_ioctl_submit_ib_packet()        │
│    - Starts background timer                                     │
│    - Reads baseline counter value                                │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 8. aql_perf_create_start_packet() [aql_packet_ops.c:174]        │
│    - Checks for shared counter (find_and_install_shared_counter) │
│    - If not shared: atomically allocates counter                 │
│    - Builds counter_info_t and counter_collection_t              │
│    - Calls generate_start_packet()                               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 9. generate_start_packet() [packet_generation.c:197]            │
│    - CS partial flush (synchronization)                          │
│    - Set GRBM to broadcast mode                                  │
│    - Disable perfmon                                             │
│    - Configure counter SELECT & CONTROL registers               │
│    - Enable compute perfcount                                    │
│    - Enable perfmon (state_enable)                               │
│    - Final CS partial flush                                      │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ 10. GPU Execution                                                │
│     - Processes PM4 commands synchronously                       │
│     - Configures hardware counter registers                      │
│     - Starts counting events                                     │
└─────────────────────────────────────────────────────────────────┘
```

### Counter Allocation Details

Counter allocation uses atomic operations to prevent race conditions:

```c
// In aql_counter_try_allocate() [aql_perf.c:729]
for (i = 0; i < block->counter_count; i++) {
    counter_reg_info_t *reg = &block->counter_reg_info[i];

    // Atomic compare-and-swap: FREE -> ALLOCATED
    old_state = atomic_cmpxchg(&reg->allocation.state,
                               COUNTER_STATE_FREE,
                               COUNTER_STATE_ALLOCATED);

    if (old_state == COUNTER_STATE_FREE) {
        // Successfully claimed!
        reg->allocation.event_id = event_id;
        reg->allocation.user_id = (uint32_t)(uintptr_t)perf_event;
        return reg;
    }
}
```

---

## Reading Flow

Counter values are read through two mechanisms:

1. **Background Timer Polling** - Periodic refresh of cached values
2. **On-Demand Reads** - Direct reads when perf requests data

### Background Timer Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ Timer fires every 20ms [pmu_main.c:216]                         │
│ (configurable via timer_period_ms module parameter)              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ amdgpu_pmu_timer_handler()                                       │
│ - Iterates active_measurements list                             │
│ - For each ACTIVE measurement:                                   │
│   Creates AQL_WORK_READ work item                                │
│   Queues on global workqueue                                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ aql_work_handler() - AQL_WORK_READ case                         │
│ - Calls aql_perf_measurement_read()                             │
│ - Updates cached_counter_value under cache_lock                  │
│ - Sets cache_valid = true                                        │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ aql_perf_measurement_read() [aql_packet_ops.c:1138]             │
│ - Creates READ PM4 packet                                        │
│ - Submits to GPU synchronously                                   │
│ - Reads result buffer                                            │
│ - Aggregates or filters by dimension                             │
│ - Computes delta from start_counter_value                        │
└─────────────────────────────────────────────────────────────────┘
```

### On-Demand Read Flow

When perf calls `read()` on the event file descriptor:

```
perf read() syscall
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ amdgpu_pmu_read() [pmu_main.c:656]                              │
│ - Checks if event uses AQL hardware                              │
│ - Calls aql_pmu_event_read()                                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ aql_pmu_event_read() [aql_pmu_integration.c:447]                │
│ - Calls aql_perf_measurement_read_atomic()                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ aql_perf_measurement_read_atomic() [aql_packet_ops.c:1475]      │
│ - Returns cached_counter_value immediately                       │
│ - Schedules background refresh (non-blocking)                    │
└─────────────────────────────────────────────────────────────────┘
```

### PM4 READ Packet Generation

```
generate_read_packet() [packet_generation.c:320]
    │
    ├─→ Enable perfmon with sampling (non-destructive read)
    │
    ├─→ Set GRBM to broadcast mode
    │
    ├─→ CS partial flush (synchronization)
    │
    └─→ For each counter:
        │
        ├─→ SE-dependent blocks (SQ, TA):
        │   For each (SE, SA, WGP) in GPU topology:
        │       Set GRBM_GFX_INDEX to (SE, SA, WGP)
        │       COPY_DATA: counter_reg -> GPU memory
        │       current_addr += 8 bytes
        │
        └─→ Global blocks (GRBM, GL2C, TCC):
            For each instance:
                Set GRBM_GFX_INDEX to instance
                COPY_DATA: counter_reg -> GPU memory
                current_addr += 8 bytes
```

### Result Buffer Layout

For an SQ counter on GFX12 (4 SE × 2 SA × 4 WGP = 32 instances):

```
GPU Memory (data_buffer):
┌──────────┬──────────┬──────────┬──────────┬─────┬──────────┐
│ [0]      │ [1]      │ [2]      │ [3]      │ ... │ [31]     │
│ SE0,SA0  │ SE0,SA0  │ SE0,SA0  │ SE0,SA0  │     │ SE3,SA1  │
│ WGP0     │ WGP1     │ WGP2     │ WGP3     │     │ WGP3     │
│ (uint64) │ (uint64) │ (uint64) │ (uint64) │     │ (uint64) │
└──────────┴──────────┴──────────┴──────────┴─────┴──────────┘

Flat index calculation:
  flat_idx = (se * num_sa * num_wgp) + (sa * num_wgp) + wgp

  Example: SE=1, SA=0, WGP=2
    flat_idx = (1 * 2 * 4) + (0 * 4) + 2 = 10
```

---

## Teardown Flow

When the perf event is closed or perf exits:

### Normal Teardown (Process Context)

```
close() syscall on event fd
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ perf core calls pmu->del()                                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ amdgpu_pmu_del() [pmu_main.c:541]                               │
│ - Checks context: in_atomic() || irqs_disabled()?               │
│ - If process context: synchronous cleanup                       │
│ - Calls aql_pmu_event_read_sync() for final count               │
│ - Calls aql_pmu_event_stop()                                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ aql_pmu_event_stop() [aql_pmu_integration.c:419]                │
│ - Calls aql_perf_measurement_stop_atomic()                       │
│ - Queues STOP work on global workqueue                          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ aql_work_handler() - AQL_WORK_STOP                              │
│ - Calls aql_perf_measurement_stop()                             │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ aql_perf_measurement_stop() [aql_packet_ops.c:814]              │
│ - If owns_counter: generate & submit STOP packet                │
│ - Release allocated counter (atomic set to FREE)                │
│ - Release shared counter reference (decrement refcount)         │
│ - Remove from active_measurements list                          │
│ - Stop timer if no more active measurements                     │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ perf core calls event->destroy()                                 │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ aql_pmu_event_destroy() [aql_pmu_integration.c:365]             │
│ - Calls aql_perf_measurement_destroy()                          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ aql_perf_measurement_destroy() [aql_packet_ops.c:1207]          │
│ - Ensures measurement stopped                                    │
│ - Releases counter if still allocated                            │
│ - Releases shared reference                                      │
│ - Calls aql_measurement_put() (drops creator reference)          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ aql_measurement_release() [aql_packet_ops.c:547]                │
│ - Called when refcount reaches 0                                │
│ - Final cleanup: free measurement structure                     │
└─────────────────────────────────────────────────────────────────┘
```

### Atomic Context Teardown

If `del()` is called from atomic context (e.g., interrupt):

```
amdgpu_pmu_del() detects atomic context
    │
    ├─→ Reads cached counter value (no blocking)
    │
    ├─→ Marks event as STOPPED
    │
    └─→ Actual cleanup deferred to event->destroy() callback
```

### PM4 STOP Packet

```
generate_stop_packet() [packet_generation.c:655]
    │
    ├─→ Set GRBM to broadcast mode
    │
    ├─→ Set perfmon state to STOP
    │
    └─→ CS partial flush (synchronization)
```

---

## Key Data Structures

### struct amdgpu_pmu

Main PMU driver structure (singleton):

```c
struct amdgpu_pmu {
    struct pmu pmu;                    // Linux perf PMU
    struct hrtimer timer;              // Background polling timer
    ktime_t timer_period;              // Default: 20ms

    // Event management
    spinlock_t lock;
    struct amdgpu_pmu_event events[64];
    DECLARE_BITMAP(used_mask, 64);

    // Statistics
    atomic64_t total_events;
    atomic64_t hardware_events;
};
```

**Location**: `src/amdgpu_pmu.h`

### struct aql_perf_session

Global session managing KFD and GPU resources (singleton):

```c
struct aql_perf_session {
    // KFD integration
    struct file *kfd_file;             // /dev/kfd handle
    struct kfd_process *process;       // KFD process context

    // GPU management
    uint32_t *gpu_ids;                 // Array of GPU device IDs
    uint32_t num_gpus;                 // Number of GPUs discovered
    arch_t **archs;                    // Architecture per GPU

    // Measurement tracking
    struct list_head active_measurements;
    spinlock_t measurement_lock;

    // Counter sharing
    struct list_head shared_counters;
    spinlock_t shared_lock;

    // State
    struct mutex session_mutex;
    refcount_t ref_count;
    enum session_state state;
    uint64_t session_id;
};
```

**Location**: `src/aql_perf.h`

**Lifecycle**: Created in `aql_pmu_init()`, destroyed in `aql_pmu_cleanup()`

### struct aql_measurement

Per-event measurement tracking (one per perf event):

```c
struct aql_measurement {
    struct list_head list;             // Link in active_measurements
    struct aql_perf_session *session;  // Parent session
    uint32_t gpu_id;                   // Target GPU device ID
    uint32_t counter_id;               // Counter ID (from registry)
    struct perf_event *event;          // Back-reference to perf event

    // State tracking
    enum measurement_state state;      // IDLE, STARTING, ACTIVE, etc.
    ktime_t start_time;
    uint64_t start_counter_value;      // Baseline for delta
    uint64_t last_counter_value;

    // Counter allocation
    counter_reg_info_t *allocated_counter;  // Hardware counter
    bool owns_counter;                      // True if allocated by us
    struct shared_counter_ref *shared_ref;  // Sharing reference

    // Dimension-specific monitoring
    struct pmu_dimension_coords target_dims;
    bool dimension_specific;

    // Atomic context support
    spinlock_t cache_lock;
    uint64_t cached_counter_value;
    bool cache_valid;

    // Reference counting
    struct kref refcount;              // Prevents use-after-free
};
```

**Location**: `src/aql_perf.h`

**Lifecycle**:
- Created in `aql_perf_measurement_create()`
- Released when `kref` reaches 0 in `aql_measurement_release()`

### struct arch_t

GPU architecture definition (one per GPU):

```c
typedef struct {
    const char *name;                  // e.g., "gfx12"

    // Topology
    uint32_t num_se;                   // Shader Engines (e.g., 4)
    uint32_t num_sa;                   // Shader Arrays per SE (e.g., 2)
    uint32_t num_wgp_per_sa;          // WGPs per SA (e.g., 4)
    uint32_t num_cu;                   // Compute Units
    uint32_t num_xcc;                  // XCCs

    // Hardware blocks
    struct {
        block_info_t *blocks[HW_IP_BLOCK_LAST];
        uint32_t block_count;
    } block_map;

    // Control registers
    struct {
        uint32_t grbm_gfx_index;
        uint32_t cp_perfmon_cntl;
        uint32_t compute_perfcount_enable;
        // ... more register offsets
    } control_regs;
} arch_t;
```

**Location**: `src/aql_c/aql_structures.h`

**Creation**: `arch_create_by_name("gfx12")` in `aql_perf_session_initialize()`

### struct block_info_t

Hardware block definition (SQ, GRBM, TA, GL2C, TCC):

```c
typedef struct {
    const char *name;                  // e.g., "SQ", "GRBM"
    hw_ip_block_t block_id;           // HW_IP_BLOCK_SQ, etc.

    // Counter configuration
    counter_reg_info_t *counter_reg_info;  // Array of counters
    uint32_t counter_count;                // Number of counters

    // Topology dimensions
    dimension_t dimensions[MAX_DIMENSIONS];
    uint32_t dimension_count;

    // For global blocks (GL2C, TCC)
    uint32_t instance_count;           // Number of instances

    // Event ID range
    uint32_t event_id_max;
} block_info_t;
```

**Location**: `src/aql_c/aql_structures.h`

**Example**: GFX12 SQ block has 16 counters, dimensions [SE=4, SA=2, WGP=4]

### struct counter_reg_info_t

Individual hardware counter register layout:

```c
typedef struct {
    // Configuration registers
    uint32_t select_addr;              // Event select register
    uint32_t control_addr;             // Control register (0 if unused)

    // Result registers
    uint32_t register_addr_lo;         // Counter value low 32 bits
    uint32_t register_addr_hi;         // Counter value high 32 bits

    // Allocation tracking
    struct {
        atomic_t state;                // FREE, ALLOCATED
        uint32_t event_id;             // Event being counted
        uint32_t user_id;              // perf_event pointer
        ktime_t allocation_time;

        // GPU memory buffers (pre-allocated)
        struct kfd_data_alloc *command_buffer;
        struct kfd_data_alloc *data_buffer;
    } allocation;
} counter_reg_info_t;
```

**Location**: `src/aql_c/aql_structures.h`

**Allocation**: Buffers allocated during session init via `aql_perf_allocate_counter_buffers()`

---

## Counter Sharing

Multiple perf events monitoring the same counter type can share a single hardware counter allocation. This is essential for dimension-specific monitoring where different events target different SE/SA/WGP instances.

### Sharing Mechanism

```
Event 1: perf stat -e amdgpu_pmu/sq_waves,se=0/
Event 2: perf stat -e amdgpu_pmu/sq_waves,se=1/
Event 3: perf stat -e amdgpu_pmu/sq_waves,se=2/
                        │
                        │ All monitor counter_id = SQ_WAVES
                        ▼
┌──────────────────────────────────────────────────────────────┐
│            shared_counter_ref (refcount = 3)                  │
│  - counter_id: SQ_WAVES                                       │
│  - measurement: -> Event 1's measurement (owner)              │
│  - ref_count: 3                                               │
└──────────────────────────────────────────────────────────────┘
                        │
                        │ Points to
                        ▼
┌──────────────────────────────────────────────────────────────┐
│            allocated_counter (SQ block, counter 0)            │
│  - allocation.state: ALLOCATED                                │
│  - allocation.event_id: 0x001 (SQ_WAVES)                      │
│  - data_buffer: GPU memory for all 32 instances               │
└──────────────────────────────────────────────────────────────┘
```

### Sharing Flow

```c
// In aql_perf_create_start_packet() [aql_packet_ops.c:229]

// 1. Try to find existing shared counter
shared_ref = find_and_install_shared_counter(session, counter_id, measurement);
if (shared_ref) {
    // Found! Reuse existing allocation
    // - measurement->allocated_counter now points to shared counter
    // - measurement->shared_ref holds reference (refcount++)
    // - measurement->owns_counter = false
    *out_pm4_buffer = NULL;  // No START packet needed
    return 0;
}

// 2. First event for this counter - allocate new counter
allocated_counter = aql_counter_try_allocate(block, event_id, perf_event);

// 3. Create new shared reference
shared_ref = create_shared_counter(session, counter_id, measurement);
measurement->shared_ref = shared_ref;
measurement->owns_counter = true;  // We allocated it

// 4. Generate and submit START packet (only for first event)
```

### Reference Counting

```
Event 1 starts:  refcount = 1, counter ALLOCATED, START packet sent
Event 2 starts:  refcount = 2, counter shared, no START packet
Event 3 starts:  refcount = 3, counter shared, no START packet
Event 2 stops:   refcount = 2, counter still in use, no STOP packet
Event 3 stops:   refcount = 1, counter still in use, no STOP packet
Event 1 stops:   refcount = 0, counter FREED, STOP packet sent
```

### Why Sharing Works

All events share the **same hardware counter** which reads all GPU instances. The dimension filtering happens in software:

```c
// In aql_perf_measurement_read() [aql_packet_ops.c:1138]

// Counter reads ALL instances into result_buffer
uint64_t result_buffer[32];  // For GFX12: 4 SE × 2 SA × 4 WGP

if (measurement->dimension_specific) {
    // Filter to specific instance
    uint32_t flat_idx = encode_dimension_index(
        measurement->target_dims.se,
        measurement->target_dims.sa,
        measurement->target_dims.wgp,
        arch->num_sa,
        arch->num_wgp_per_sa
    );
    counter_value = result_buffer[flat_idx];
} else {
    // Aggregate all instances
    counter_value = aql_aggregate_counter_instances(...);
}
```

---

## Dimension-Specific Monitoring

Users can monitor specific hardware instances using dimension parameters:

### Syntax

```bash
# Monitor specific SE
perf stat -C 0 -e amdgpu_pmu/sq_waves,se=0/

# Monitor specific SA within SE
perf stat -C 0 -e amdgpu_pmu/sq_waves,se=1,sa=0/

# Monitor specific WGP within SA
perf stat -C 0 -e amdgpu_pmu/sq_waves,se=2,sa=1,wgp=3/

# Aggregate all dimensions (default)
perf stat -C 0 -e amdgpu_pmu/sq_waves,aggregate=1/
```

### Dimension Encoding

Dimensions are encoded in `event->attr.config1`:

```
config1 bit layout (64 bits):
┌─────────┬─────────┬─────────┬─────────┬─────────┬──────────────┐
│ Bits    │ Field   │ Values  │ Bits    │ Field   │ Values       │
├─────────┼─────────┼─────────┼─────────┼─────────┼──────────────┤
│ 0-7     │ XCC     │ 0-255   │ 24-31   │ WGP     │ 0-255        │
│ 8-15    │ SE      │ 0-255   │ 32-39   │ CU      │ 0-255        │
│ 16-23   │ SA      │ 0-255   │ 40      │ AGG     │ 0=off, 1=on  │
└─────────┴─────────┴─────────┴─────────┴─────────┴──────────────┘
```

### Dimension Extraction

```c
// In pmu_main.c:434
if (config1 != 0) {
    pmu_extract_dimensions(config1, &dims);
    // dims.se, dims.sa, dims.wgp extracted
    // dims.aggregate flag

    // Validate against GPU limits
    get_gpu_dimension_limits(gpu_id, &gpu_limits);
    pmu_validate_dimensions(&dims, &gpu_limits);

    // Store in measurement
    measurement->target_dims = dims;
    measurement->dimension_specific = !dims.aggregate;
}
```

### Result Filtering

When reading counters, dimension-specific events filter the result:

```c
// GFX12 example: 4 SE × 2 SA × 4 WGP = 32 instances
// Result buffer layout after GPU read:
uint64_t result_buffer[32] = {
    // [0]  SE0,SA0,WGP0
    // [1]  SE0,SA0,WGP1
    // [2]  SE0,SA0,WGP2
    // [3]  SE0,SA0,WGP3
    // [4]  SE0,SA1,WGP0
    // ...
    // [31] SE3,SA1,WGP3
};

// Event requesting SE=1, SA=0, WGP=2
uint32_t flat_idx = encode_dimension_index(1, 0, 2, 2, 4);
// flat_idx = (1 * 2 * 4) + (0 * 4) + 2 = 10

counter_value = result_buffer[10];  // Return only this instance
```

### Dimension Validation

Each counter defines which dimensions it supports:

```c
// In counter_registry.c
counter_def_t sq_waves = {
    .id = COUNTER_SQ_WAVES,
    .name = "sq_waves",
    .hw_block = HW_IP_BLOCK_SQ,
    .supported_dimensions = DIM_SE | DIM_SA | DIM_WGP,  // SE-dependent
};

counter_def_t grbm_count = {
    .id = COUNTER_GRBM_COUNT,
    .name = "grbm_count",
    .hw_block = HW_IP_BLOCK_GRBM,
    .supported_dimensions = 0,  // Global block, no dimensions
};
```

Validation rejects incompatible dimension requests:

```c
// In pmu_dimension.c
int pmu_validate_counter_dimensions(const counter_def_t *counter,
                                     const struct pmu_dimension_coords *dims)
{
    if (dims->se != DIM_UNSPECIFIED &&
        !(counter->supported_dimensions & DIM_SE)) {
        return -EINVAL;  // SE not supported for this counter
    }
    // ... similar for SA, WGP, CU
}
```

---

## Debugging Tips

### Enable Debug Logging

```bash
# Load module with debug enabled
modprobe amdgpu_pmu debug_enable=1

# Check dmesg for detailed logs
dmesg -w | grep -E "(AQL_PERF|PMU)"
```

### Monitor Active Measurements

```bash
# View session state
cat /sys/kernel/debug/amdgpu_pmu/session_state

# View active measurements
cat /sys/kernel/debug/amdgpu_pmu/measurements
```

### Trace Event Flow

Key log messages to watch for:

```
[PMU] aql_pmu_event_init: config=0x1 config1=0x0
[PMU] Mapped CPU 0 to GPU 7410
[PMU] aql_perf_measurement_create: session_id=1, gpu_id=7410
[PMU] generate_start_packet: Allocating counter from block=SQ
[PMU] Counter allocated successfully
[PMU] START packet submitted
[PMU] Timer: Scheduled read for GPU 7410
[PMU] READ_SYNC: GPU 7410, counter_value=12345
[PMU] aql_perf_measurement_stop: GPU 7410
```

### Common Issues

**Issue**: `event_init` fails with `-EINVAL`
- **Cause**: Invalid dimension values or counter doesn't support requested dimensions
- **Fix**: Check counter's `supported_dimensions` in counter registry

**Issue**: Counter reads return 0
- **Cause**: Counter not started or GPU idle
- **Fix**: Verify GPU workload is running, check START packet submission

**Issue**: Measurement cleanup warnings
- **Cause**: Reference counting imbalance or work items not flushed
- **Fix**: Ensure `aql_pmu_flush_all_measurements()` called before session cleanup

---

## Summary

The perf-dkms kernel module provides comprehensive GPU performance monitoring through:

1. **Linux perf integration** - Standard perf event interface
2. **PM4 packet submission** - Direct GPU command buffer generation
3. **Counter sharing** - Efficient resource usage for dimension-specific monitoring
4. **Atomic context safety** - Workqueue-based deferred operations
5. **Reference counting** - Safe cleanup with async work items

Key architectural principles:

- **One global session** - Shared by all events
- **Per-event measurements** - Isolated state tracking
- **Atomic counter allocation** - Lock-free race prevention
- **Software dimension filtering** - Hardware reads all instances, SW selects
- **Background polling** - Periodic cache refresh for low latency reads

For more details, consult the source code in `projects/perf-dkms/src/`.
