# AGENT_CODE_INFO.md - perf-pmu-stub/src

## Directory Purpose
This directory contains the main implementation of the PMU stub kernel module, providing the core PMU driver functionality, timer-based counter simulation, and integration with the AQL C library for AMD GPU performance monitoring packet generation.

## Key Components

### PMU Core Implementation
- **pmu_main.c**: Main module entry point, PMU driver registration, and core callbacks
- **pmu_stub.h**: Core data structures, function prototypes, and constants
- **pmu_events.c**: Event management utilities and counter simulation logic

### AQL C Library Integration
- **aql_c/**: Complete AQL packet generation library (see separate AGENT_CODE_INFO.md)
  - Architecture detection and abstraction layer
  - PM4 command generation for GPU counter programming
  - Support for GFX9, GFX10, GFX11, GFX12 architectures

### Build Configuration
- **Makefile**: Kernel module build rules, linking AQL library, and compiler flags

## Public APIs

### PMU Driver Interface (pmu_stub.h)

```c
/* Core PMU structure - maximum 64 concurrent events */
#define PMU_STUB_MAX_EVENTS 64

struct pmu_stub {
    struct pmu pmu;                          /* Base PMU structure */
    struct device *dev;                      /* Device for sysfs */
    spinlock_t lock;                         /* Protects event_list */
    struct pmu_stub_event events[PMU_STUB_MAX_EVENTS];
    DECLARE_BITMAP(used_mask, PMU_STUB_MAX_EVENTS);
    int num_events;

    /* Timer for simulating counter updates */
    struct hrtimer timer;
    ktime_t timer_period;

    /* Statistics and simulated counter values */
    atomic64_t total_events;
    atomic64_t total_samples;
    atomic64_t counter_cycles;
    atomic64_t counter_instructions;
    atomic64_t counter_cache_misses;
    atomic64_t counter_bandwidth;
};

/* Event types (simulated) */
enum pmu_stub_event_id {
    PMU_STUB_EVENT_CYCLES = 0x00,
    PMU_STUB_EVENT_INSTRUCTIONS = 0x01,
    PMU_STUB_EVENT_CACHE_MISSES = 0x02,
    PMU_STUB_EVENT_BANDWIDTH = 0x03,
    PMU_STUB_EVENT_MAX
};

/* Event management functions */
enum hrtimer_restart pmu_stub_timer_handler(struct hrtimer *timer);
int pmu_stub_get_event_idx(struct pmu_stub *pmu);
void pmu_stub_free_event_idx(struct pmu_stub *pmu, int idx);
```

### Event Utility Interface (pmu_events.c)

```c
/* Event configuration and validation */
const char *pmu_stub_get_event_name(u64 config);
const char *pmu_stub_get_event_description(u64 config);
bool pmu_stub_is_valid_event(u64 config);

/* Counter access and manipulation */
u64 pmu_stub_get_counter_value(struct pmu_stub *pmu, u64 config);
void pmu_stub_update_counter(struct pmu_stub *pmu, u64 config, s64 delta);
void pmu_stub_reset_counters(struct pmu_stub *pmu);

/* Debug and monitoring */
void pmu_stub_print_event_stats(struct pmu_stub *pmu);

/* Exported symbols for module integration */
EXPORT_SYMBOL_GPL(pmu_stub_get_event_name);
EXPORT_SYMBOL_GPL(pmu_stub_get_event_description);
EXPORT_SYMBOL_GPL(pmu_stub_is_valid_event);
```

### Module Parameters Interface

```c
/* Configurable module parameters */
static bool debug_enable = false;
module_param(debug_enable, bool, 0644);

static int timer_period_ms = 100;
module_param(timer_period_ms, int, 0644);

/* Debug macros */
#define pmu_debug(fmt, ...) /* Debug output when enabled */
#define pmu_info(fmt, ...)  /* Informational messages */
#define pmu_err(fmt, ...)   /* Error messages */
```

## Design Patterns

### Singleton PMU Instance
- Single global `pmu_stub_instance` manages all PMU operations
- Registration with Linux perf subsystem on module load
- Cleanup and unregistration on module unload

### Bitmap-Based Resource Management
```c
struct pmu_stub {
    struct pmu_stub_event events[PMU_STUB_MAX_EVENTS];
    DECLARE_BITMAP(used_mask, PMU_STUB_MAX_EVENTS);
    int num_events;
};

/* Event slot allocation using kernel bitmap functions */
int pmu_stub_get_event_idx(struct pmu_stub *pmu) {
    idx = find_first_zero_bit(pmu->used_mask, PMU_STUB_MAX_EVENTS);
    set_bit(idx, pmu->used_mask);
    return idx;
}
```

### Timer-Based Simulation Pattern
```c
/* High-resolution timer for counter simulation */
enum hrtimer_restart pmu_stub_timer_handler(struct hrtimer *timer) {
    /* Update simulated counter values */
    atomic64_add(1000, &pmu->counter_cycles);
    atomic64_add(500, &pmu->counter_instructions);
    /* Update active events */
    return HRTIMER_RESTART;
}
```

### Event State Machine
Events transition through states:
1. **Unallocated** → **Initialized** (event_init)
2. **Initialized** → **Added** (add - allocates slot, starts timer if first)
3. **Added** → **Active** (start - enables counting)
4. **Active** → **Stopped** (stop - pauses counting)
5. **Stopped** → **Active** (start - resumes counting)
6. **Added/Active/Stopped** → **Unallocated** (del - frees slot, stops timer if last)

## Dependencies

### Kernel Headers Required
- `<linux/perf_event.h>`: Core perf subsystem interface
- `<linux/hrtimer.h>`: High-resolution timer support
- `<linux/spinlock.h>`: Spinlock protection for events
- `<linux/bitmap.h>`: Bitmap operations for event allocation
- `<linux/atomic.h>`: Atomic counter operations
- `<linux/device.h>`: Device model and sysfs attributes
- `<linux/slab.h>`: Memory allocation functions

### Module Dependencies
- No external module dependencies for basic operation
- Subdirectory `aql_c/`: Complete AQL packet generation library
- Linux perf subsystem (always available in modern kernels)

## Data Flow

### Event Creation Flow
1. User space calls `perf_event_open()` with PMU type
2. `pmu_stub_event_init()` validates event configuration
3. Event passes validation, but no slot allocated yet
4. Event added to PMU via `pmu_stub_add()` which allocates slot

### Event Management Flow
1. `pmu_stub_add()`: Find free slot, allocate, start timer if first event
2. `pmu_stub_start()`: Mark event as active for counter updates
3. `pmu_stub_stop()`: Mark event as inactive, optionally update count
4. `pmu_stub_del()`: Free slot, stop timer if last event

### Counter Update Flow
1. `hrtimer` fires every `timer_period_ms` (default 100ms)
2. `pmu_stub_timer_handler()` called in interrupt context
3. Updates global atomic counters (cycles, instructions, etc.)
4. Iterates through active events, updates individual event counts
5. Reschedules timer for next period

### Build Integration Flow
1. `Makefile` defines `pmu_stub-y` with all object files
2. AQL C library files compiled as part of module
3. Single kernel object `pmu_stub.ko` created
4. All symbols linked into single module namespace

## Integration Points

### sysfs Attributes
Located at `/sys/bus/event_source/devices/pmu_stub/`:
- `format/format`: Event format descriptors (`config:0-63`)
- `events/cycles`: Cycles event configuration
- `events/instructions`: Instructions event configuration
- `events/cache_misses`: Cache misses event configuration
- `events/bandwidth`: Bandwidth event configuration
- Module parameters in `/sys/module/pmu_stub/parameters/`

### PMU Callbacks Implementation
```c
/* PMU structure with all required callbacks */
pmu->pmu = (struct pmu) {
    .name           = PMU_NAME,
    .task_ctx_nr    = perf_invalid_context,
    .event_init     = pmu_stub_event_init,
    .add            = pmu_stub_add,
    .del            = pmu_stub_del,
    .start          = pmu_stub_start,
    .stop           = pmu_stub_stop,
    .read           = pmu_stub_read,
    .attr_groups    = pmu_stub_attr_groups,
    .capabilities   = PERF_PMU_CAP_NO_INTERRUPT | PERF_PMU_CAP_NO_EXCLUDE,
};
```

### AQL Library Integration
The module integrates with the AQL C library for GPU packet generation:
- Architecture detection via `aql_detect_architecture()`
- PM4 command generation for counter programming
- AQL packet creation for GPU submission (prepared but not executed)

## Critical Business Logic

### Event Validation (pmu_stub_event_init)
1. Check event is for this PMU (type match via `event->attr.type`)
2. Validate event configuration (0-3 for four supported simulated events)
3. Reject sampling events (not supported - `is_sampling_event()`)
4. Reject exclude filters (user/kernel/hv/idle not supported)
5. Set initial hardware event state

### Counter Simulation Strategy
Timer-based simulation with realistic increment patterns:
- **Cycles**: Increments by 1000 per timer tick (100ms = 10 KHz simulation)
- **Instructions**: Increments by 500 per timer tick
- **Cache Misses**: Increments by 10 per timer tick
- **Bandwidth**: Increments by 100 per timer tick

### Event Lifecycle Management
1. **Init Phase**: Validate configuration, no hardware allocation
2. **Add Phase**: Allocate slot from bitmap, start timer if first event
3. **Start/Stop Phase**: Control active flag for counter updates
4. **Del Phase**: Free slot, stop timer if last event, update final count

### Timer Management Logic
- Timer started only when `num_events` transitions from 0 to 1
- Timer stopped only when `num_events` transitions from 1 to 0
- Prevents unnecessary timer overhead when no events are active
- Configurable period via module parameter

## Error Handling

### Error Codes and Validation
- `-ENOENT`: Event type doesn't match this PMU
- `-EINVAL`: Invalid event configuration (>= PMU_STUB_EVENT_MAX)
- `-EOPNOTSUPP`: Unsupported features (sampling, exclude filters)
- `-EAGAIN`: No free event slots available

### Recovery Mechanisms
- Proper cleanup on all error paths
- Event slot bitmap prevents resource leaks
- Error logging with pmu_err() macro for debugging
- Graceful handling of timer cancellation

## Thread Safety
- `spinlock_t lock` protects event array and bitmap operations
- `atomic64_t` counters for thread-safe counter updates
- Timer callbacks run in interrupt context (softirq)
- IRQ-safe locking (`spin_lock_irqsave`) in timer handler

## Performance Considerations
- Timer period configurable via module parameter (default 100ms)
- Batch counter updates in single timer handler
- Pre-allocated event structures (no dynamic allocation during operation)
- Bitmap operations for O(1) slot allocation
- Atomic operations for counter updates minimize lock contention

## Module Lifecycle

### Initialization (`pmu_stub_init`)
1. Allocate PMU instance with `kzalloc()`
2. Initialize spinlock and event bitmap
3. Setup high-resolution timer with `hrtimer_setup()`
4. Set timer period from module parameter
5. Configure PMU structure with callbacks and attributes
6. Register PMU with kernel via `perf_pmu_register()`
7. Store global instance pointer

### Cleanup (`pmu_stub_exit`)
1. Cancel and cleanup high-resolution timer
2. Unregister PMU from kernel via `perf_pmu_unregister()`
3. Print final statistics (total events, samples)
4. Free PMU instance memory with `kfree()`
5. Clear global instance pointer

## Testing Support
- Simulated counters for testing without real hardware
- Timer-based simulation provides realistic event behavior
- Standard perf tools compatibility (perf stat, perf record)
- Debug output controlled by `debug_enable` module parameter
- sysfs attributes for runtime monitoring and configuration

## Build Integration with AQL Library
The src/ Makefile integrates all AQL C library files:
```makefile
pmu_stub-y := pmu_main.o pmu_events.o \
              aql_c/aql_cmd_buffer.o \
              aql_c/aql_packet.o \
              aql_c/aql_arch_detect.o \
              aql_c/aql_gfx9_ops.o \
              aql_c/aql_gfx12_ops.o \
              aql_c/aql_pmc_interface.o
```