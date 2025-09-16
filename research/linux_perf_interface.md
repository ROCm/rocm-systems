# Linux Perf Interface Research: GPU Performance Monitoring Integration

## Table of Contents
1. [Perf Subsystem Architecture Overview](#perf-subsystem-architecture-overview)
2. [PMU Driver Interface and Registration](#pmu-driver-interface-and-registration)
3. [Key Kernel APIs and Data Structures](#key-kernel-apis-and-data-structures)
4. [Ring Buffer and Data Format Specifications](#ring-buffer-and-data-format-specifications)
5. [Event Configuration and Sampling Mechanisms](#event-configuration-and-sampling-mechanisms)
6. [Userspace Interface and Tool Integration](#userspace-interface-and-tool-integration)
7. [Example Code for PMU Driver Implementation](#example-code-for-pmu-driver-implementation)
8. [Specific Requirements for GPU Performance Counters](#specific-requirements-for-gpu-performance-counters)
9. [Best Practices and Common Patterns](#best-practices-and-common-patterns)

## Perf Subsystem Architecture Overview

### Core Components

The Linux perf_event subsystem is a comprehensive performance monitoring framework that provides unified access to hardware and software performance counters. The architecture implements several key abstraction layers:

#### Event Types and Classification
- **Hardware Events**: CPU cycles, instructions retired, cache misses, branch mispredictions
- **Software Events**: Context switches, page faults, CPU migrations
- **Tracepoint Events**: Kernel function entry/exit points
- **Hardware Cache Events**: L1/L2/L3 cache operations with detailed breakdowns
- **Raw Events**: Direct hardware register access for vendor-specific counters
- **Breakpoint Events**: Hardware watchpoints and breakpoints

#### Context Management
The perf subsystem uses a hierarchical context structure:

```c
struct perf_event_context {
    struct pmu             *pmu;
    raw_spinlock_t         lock;
    struct mutex           mutex;
    struct list_head       active_ctx_list;
    struct perf_event_groups  pinned_groups;
    struct perf_event_groups  flexible_groups;
    // ... other fields
};
```

- **CPU Context (`perf_cpu_context`)**: Per-CPU performance monitoring context
- **Task Context**: Per-process/thread performance monitoring
- **PMU Context**: Per-PMU-type context sharing for same PMU implementations

#### Core Data Flow
1. **Event Creation**: User calls `perf_event_open()` system call
2. **PMU Assignment**: Kernel assigns event to appropriate PMU based on event type
3. **Context Management**: Event added to appropriate CPU/task context
4. **Scheduling**: Context scheduler manages PMU resource allocation
5. **Data Collection**: Ring buffer captures performance data
6. **User Access**: mmap'd ring buffer provides low-latency data access

### PMU (Performance Monitoring Unit) Architecture

Each PMU type implements a standardized interface through `struct pmu`:

```c
struct pmu {
    struct list_head        entry;
    struct module          *module;
    struct device          *dev;
    const struct attribute_group **attr_groups;
    const char             *name;
    int                     type;

    /* PMU lifecycle callbacks */
    int  (*event_init)     (struct perf_event *event);
    void (*event_mapped)   (struct perf_event *event);
    void (*event_unmapped) (struct perf_event *event);
    int  (*add)            (struct perf_event *event, int flags);
    void (*del)            (struct perf_event *event, int flags);
    void (*start)          (struct perf_event *event, int flags);
    void (*stop)           (struct perf_event *event, int flags);
    void (*read)           (struct perf_event *event);
    // ... additional callbacks
};
```

## PMU Driver Interface and Registration

### PMU Registration Process

PMU drivers register with the kernel using `perf_pmu_register()`:

```c
int perf_pmu_register(struct pmu *pmu, const char *name, int type);
```

**Registration Steps:**
1. **Memory Allocation**: Allocate per-CPU PMU contexts
2. **Type Assignment**: Assign unique PMU type identifier
3. **Sysfs Integration**: Create sysfs entries for event discovery
4. **Context Initialization**: Set up per-CPU context structures
5. **Event Group Setup**: Initialize pinned and flexible event groups

### Critical PMU Callbacks Implementation

#### event_init Callback
Called when user creates a new perf event. Must validate event configuration and initialize hardware-specific state:

```c
static int gpu_pmu_event_init(struct perf_event *event)
{
    struct gpu_pmu *gpu_pmu = to_gpu_pmu(event->pmu);
    struct hw_perf_event *hwc = &event->hw;

    /* Validate event type and configuration */
    if (event->attr.type != event->pmu->type)
        return -ENOENT;

    /* Check if event is supported on this GPU */
    if (!gpu_pmu_event_supported(event->attr.config))
        return -EOPNOTSUPP;

    /* Initialize hardware event structure */
    hwc->config = event->attr.config;
    hwc->sample_period = event->attr.sample_period;

    /* Set up event constraints */
    event->hw.last_tag = ~0ULL;

    return 0;
}
```

#### add/del Callbacks
Manage event scheduling and PMU resource allocation:

```c
static int gpu_pmu_add(struct perf_event *event, int flags)
{
    struct gpu_pmu *gpu_pmu = to_gpu_pmu(event->pmu);
    struct hw_perf_event *hwc = &event->hw;
    int idx;

    /* Find available counter */
    idx = gpu_pmu_get_event_idx(gpu_pmu, event);
    if (idx < 0)
        return -EAGAIN;

    /* Assign counter to event */
    hwc->idx = idx;
    gpu_pmu->events[idx] = event;

    /* Configure hardware counter */
    gpu_pmu_config_counter(gpu_pmu, idx, hwc->config);

    /* Start counter if requested */
    if (flags & PERF_EF_START)
        gpu_pmu_start(event, 0);

    return 0;
}

static void gpu_pmu_del(struct perf_event *event, int flags)
{
    struct gpu_pmu *gpu_pmu = to_gpu_pmu(event->pmu);
    struct hw_perf_event *hwc = &event->hw;

    /* Stop counter if running */
    if (flags & PERF_EF_UPDATE)
        gpu_pmu_stop(event, PERF_EF_UPDATE);

    /* Release counter */
    gpu_pmu->events[hwc->idx] = NULL;
    hwc->idx = -1;

    /* Update final count */
    gpu_pmu_read(event);
}
```

#### start/stop Callbacks
Control counter operation:

```c
static void gpu_pmu_start(struct perf_event *event, int flags)
{
    struct gpu_pmu *gpu_pmu = to_gpu_pmu(event->pmu);
    struct hw_perf_event *hwc = &event->hw;

    if (WARN_ON_ONCE(!(hwc->state & PERF_HES_STOPPED)))
        return;

    /* Clear stopped state */
    hwc->state = 0;

    /* Enable counter in hardware */
    gpu_pmu_enable_counter(gpu_pmu, hwc->idx);

    /* Record start time */
    hwc->last_tag = gpu_pmu_read_timestamp(gpu_pmu);
}

static void gpu_pmu_stop(struct perf_event *event, int flags)
{
    struct gpu_pmu *gpu_pmu = to_gpu_pmu(event->pmu);
    struct hw_perf_event *hwc = &event->hw;

    if (hwc->state & PERF_HES_STOPPED)
        return;

    /* Disable counter in hardware */
    gpu_pmu_disable_counter(gpu_pmu, hwc->idx);

    /* Mark as stopped */
    hwc->state |= PERF_HES_STOPPED;

    /* Update count if requested */
    if (flags & PERF_EF_UPDATE)
        gpu_pmu_read(event);
}
```

#### read Callback
Updates event count from hardware:

```c
static void gpu_pmu_read(struct perf_event *event)
{
    struct gpu_pmu *gpu_pmu = to_gpu_pmu(event->pmu);
    struct hw_perf_event *hwc = &event->hw;
    u64 delta, prev, now;

    /* Read current counter value */
    now = gpu_pmu_read_counter(gpu_pmu, hwc->idx);
    prev = local64_read(&hwc->prev_count);

    /* Calculate delta accounting for overflow */
    delta = (now - prev) & ((1ULL << gpu_pmu->counter_width) - 1);

    /* Update stored values */
    local64_set(&hwc->prev_count, now);
    local64_add(delta, &event->count);
}
```

### Event Lifecycle State Machine

The PMU driver operates as a state machine with well-defined transitions:

1. **INACTIVE**: Event created but not scheduled
2. **ACTIVE**: Event added to PMU and potentially running
3. **RUNNING**: Counter actively collecting data
4. **STOPPED**: Counter temporarily disabled

State transitions occur through the callback functions based on scheduling decisions and user requests.

## Key Kernel APIs and Data Structures

### struct perf_event

The core event structure representing a single performance monitoring event:

```c
struct perf_event {
    struct list_head            event_entry;
    struct list_head            sibling_list;
    struct list_head            active_list;

    struct rb_node              group_node;
    u64                         group_index;
    struct list_head            group_entry;
    struct perf_event          *group_leader;
    struct pmu                 *pmu;

    enum perf_event_state       state;
    unsigned int                attach_state;
    local64_t                   count;
    atomic64_t                  child_count;

    struct perf_event_attr      attr;
    u16                         header_size;
    u16                         id_header_size;
    u16                         read_size;
    struct hw_perf_event        hw;

    struct perf_event_context  *ctx;
    atomic_long_t               refcount;
    atomic64_t                  child_total_time_enabled;
    atomic64_t                  child_total_time_running;

    struct mutex                child_mutex;
    struct list_head            child_list;
    struct perf_event          *parent;

    int                         oncpu;
    int                         cpu;

    struct list_head            owner_entry;
    struct task_struct         *owner;

    /* Ring buffer management */
    struct mutex                mmap_mutex;
    atomic_t                    mmap_count;
    struct perf_buffer         *rb;
    struct list_head            rb_entry;

    /* Sampling and overflow */
    atomic_t                    pending_wakeup;
    atomic_t                    pending_kill;
    atomic_t                    pending_disable;
    wait_queue_head_t           waitq;
    struct fasync_struct       *fasync;

    /* Event capabilities */
    int                         pending_addr;

    /* Security and context */
    void                       *security;
    struct perf_cgroup         *cgrp;
};
```

### struct perf_event_attr

Configuration structure passed from userspace:

```c
struct perf_event_attr {
    __u32                   type;           /* PERF_TYPE_* */
    __u32                   size;           /* sizeof(struct perf_event_attr) */
    __u64                   config;         /* Event-specific configuration */

    union {
        __u64               sample_period;  /* Period for sampling */
        __u64               sample_freq;    /* Frequency for sampling */
    };

    __u64                   sample_type;    /* PERF_SAMPLE_* */
    __u64                   read_format;    /* PERF_FORMAT_* */

    __u64                   disabled       :  1,   /* Event starts disabled */
                            inherit        :  1,   /* Children inherit event */
                            pinned         :  1,   /* Must always be on PMU */
                            exclusive      :  1,   /* Only this event on PMU */
                            exclude_user   :  1,   /* Don't count user space */
                            exclude_kernel :  1,   /* Don't count kernel */
                            exclude_hv     :  1,   /* Don't count hypervisor */
                            exclude_idle   :  1,   /* Don't count idle CPU */
                            mmap           :  1,   /* Include mmap data */
                            comm           :  1,   /* Include comm data */
                            freq           :  1,   /* Use freq not period */
                            inherit_stat   :  1,   /* Per task counts */
                            enable_on_exec :  1,   /* Next exec enables */
                            task           :  1,   /* Trace fork/exit */
                            watermark      :  1,   /* Wakeup watermark */
                            precise_ip     :  2,   /* Precision level */
                            mmap_data      :  1,   /* Non-exec mmap data */
                            sample_id_all  :  1,   /* Include ID in all samples */
                            exclude_host   :  1,   /* Don't count host */
                            exclude_guest  :  1,   /* Don't count guest */
                            exclude_callchain_kernel : 1,
                            exclude_callchain_user   : 1,
                            mmap2          :  1,   /* Include mmap with inode data */
                            comm_exec      :  1,   /* Flag comm events from exec */
                            use_clockid    :  1,   /* Use clockid field */
                            context_switch :  1,   /* Include context switch */
                            write_backward :  1,   /* Write ring buffer backwards */
                            namespaces     :  1,   /* Include namespace data */
                            ksymbol        :  1,   /* Include ksymbol events */
                            bpf_event      :  1,   /* Include bpf events */
                            aux_output     :  1,   /* Generate AUX records */
                            cgroup         :  1,   /* Include cgroup events */
                            text_poke      :  1,   /* Include text poke events */
                            build_id       :  1,   /* Include build ID */
                            inherit_thread :  1,   /* Children only inherit if cloned with CLONE_THREAD */
                            remove_on_exec :  1,   /* Event is removed from task on exec */
                            sigtrap        :  1,   /* Send synchronous SIGTRAP on event */
                            __reserved_1   : 26;

    union {
        __u32               wakeup_events;   /* Wakeup every n events */
        __u32               wakeup_watermark; /* Bytes before wakeup */
    };

    __u32                   bp_type;        /* Breakpoint type */
    union {
        __u64               bp_addr;        /* Breakpoint address */
        __u64               kprobe_func;    /* Kprobe function */
        __u64               uprobe_path;    /* Uprobe path */
        __u64               config1;        /* Extension of config */
    };
    union {
        __u64               bp_len;         /* Breakpoint length */
        __u64               kprobe_addr;    /* Kprobe address */
        __u64               probe_offset;   /* Probe offset */
        __u64               config2;        /* Extension of config */
    };
    __u64                   branch_sample_type; /* Branch sampling mode */
    __u64                   sample_regs_user;   /* User regs to dump on samples */
    __u32                   sample_stack_user;  /* User stack size to dump */
    __s32                   clockid;        /* Clock to use for timestamps */
    __u64                   sample_regs_intr;   /* Interrupt regs to dump */
    __u32                   aux_watermark;  /* AUX area watermark */
    __u16                   sample_max_stack; /* Max frames in callchain */
    __u16                   __reserved_2;   /* Align to u64 */
    __u32                   aux_sample_size; /* AUX area sample size */
    __u32                   __reserved_3;   /* Align to u64 */
    __u64                   sig_data;       /* User data for sigtrap */
};
```

### struct hw_perf_event

Hardware-specific event information:

```c
struct hw_perf_event {
    union {
        struct { /* Hardware event */
            u64             config;
            u64             last_tag;
            unsigned long   config_base;
            unsigned long   event_base;
            int             event_base_rdpmc;
            int             idx;
            int             last_cpu;
            int             flags;
            struct hw_perf_event_extra extra_reg;
            struct hw_perf_event_extra branch_reg;
        };
        struct { /* Software event */
            struct hrtimer  hrtimer;
        };
        struct { /* Breakpoint event */
            struct arch_hw_breakpoint info;
        };
        struct { /* AMD IBS */
            u32             config;
        };
    };

    int                         state;
    local64_t                   prev_count;
    u64                         sample_period;
    union {
        u64                     period_left;
        struct {
            u64                 addr_filters;
        };
    };
    u64                         interrupts_seq;
    u64                         interrupts;

    u64                         freq_time_stamp;
    u64                         freq_count_stamp;
};
```

## Ring Buffer and Data Format Specifications

### Ring Buffer Architecture

The perf ring buffer provides high-performance, lock-free data transfer between kernel and userspace using memory-mapped I/O.

#### Memory Layout

```
+-------------------+
| Metadata Page     | <- struct perf_event_mmap_page
+-------------------+
| Data Page 0       |
+-------------------+
| Data Page 1       |
+-------------------+
| ...               |
+-------------------+
| Data Page N-1     |
+-------------------+
```

Total size must be 1 + 2^n pages where the first page contains metadata.

#### struct perf_event_mmap_page

Control structure for ring buffer management:

```c
struct perf_event_mmap_page {
    __u32   version;                /* Version of this structure */
    __u32   compat_version;         /* Compatibility version */
    __u32   lock;                   /* Seqlock for reader/writer sync */
    __u32   index;                  /* Hardware counter index */
    __s64   offset;                 /* Hardware counter offset */
    __u64   time_enabled;           /* Time event was enabled */
    __u64   time_running;           /* Time event was running */

    union {
        __u64   capabilities;
        struct {
            __u64   cap_bit0        : 1,
                    cap_bit0_is_deprecated : 1,
                    cap_user_rdpmc  : 1,   /* Can use RDPMC instruction */
                    cap_user_time   : 1,   /* Time functionality */
                    cap_user_time_zero : 1, /* Time zero functionality */
                    cap_user_time_short : 1, /* Short time functionality */
                    cap_____res     : 58;
        };
    };

    __u16   pmc_width;              /* PMC width in bits */
    __u16   time_shift;             /* Time shift for time calculations */
    __u32   time_mult;              /* Time multiplier */
    __u64   time_offset;            /* Time offset */
    __u64   time_zero;              /* Time zero value */
    __u32   size;                   /* Size of this structure */
    __u32   __reserved_1;

    __u64   data_head;              /* Head pointer (written by kernel) */
    __u64   data_tail;              /* Tail pointer (written by user) */
    __u64   data_offset;            /* Data offset from mmap base */
    __u64   data_size;              /* Data section size */

    __u64   aux_head;               /* AUX head pointer */
    __u64   aux_tail;               /* AUX tail pointer */
    __u64   aux_offset;             /* AUX offset from mmap base */
    __u64   aux_size;               /* AUX section size */
};
```

#### Memory Barriers and Synchronization

Proper synchronization requires memory barriers:

```c
/* Reader (userspace) */
head = mmap_page->data_head;
smp_rmb(); /* Read barrier after reading head */

/* Process data between tail and head */
while (tail != head) {
    event = data + (tail & mask);
    /* Process event */
    tail += event->size;
}

smp_mb(); /* Memory barrier before updating tail */
mmap_page->data_tail = tail;

/* Writer (kernel) */
tail = READ_ONCE(rb->user_page->data_tail);
if (CIRC_SPACE(head, tail, data_size) < record_size)
    return -ENOSPC;

/* Write record */
memcpy(data + (head & mask), record, record_size);

smp_wmb(); /* Write barrier before updating head */
rb->user_page->data_head = head + record_size;
```

### Record Format

All records start with a common header:

```c
struct perf_event_header {
    __u32   type;       /* PERF_RECORD_* */
    __u16   misc;       /* Record-specific flags */
    __u16   size;       /* Size including header */
};
```

#### Sample Record Format

Sample records contain event-specific data based on `sample_type`:

```c
struct sample_record {
    struct perf_event_header header;

    /* Conditional fields based on sample_type */
    { u64                   ip;           } /* PERF_SAMPLE_IP */
    { u32                   pid, tid;     } /* PERF_SAMPLE_TID */
    { u64                   time;         } /* PERF_SAMPLE_TIME */
    { u64                   addr;         } /* PERF_SAMPLE_ADDR */
    { u64                   id;           } /* PERF_SAMPLE_ID */
    { u64                   stream_id;    } /* PERF_SAMPLE_STREAM_ID */
    { u32                   cpu, res;     } /* PERF_SAMPLE_CPU */
    { u64                   period;       } /* PERF_SAMPLE_PERIOD */

    /* Variable-length fields */
    { u64                   nr;           /* PERF_SAMPLE_CALLCHAIN */
      u64                   ips[nr];      }
    { u32                   size;         /* PERF_SAMPLE_RAW */
      char                  data[size];   }
    { u64                   abi;          /* PERF_SAMPLE_REGS_USER */
      u64                   regs[weight(mask)]; }
    { u64                   size;         /* PERF_SAMPLE_STACK_USER */
      char                  data[size];
      u64                   dyn_size;     }
    { u64                   weight;       } /* PERF_SAMPLE_WEIGHT */
    { u64                   data_src;     } /* PERF_SAMPLE_DATA_SRC */
    { u64                   transaction;  } /* PERF_SAMPLE_TRANSACTION */
};
```

### AUX Buffer Implementation

Auxiliary buffers support high-bandwidth hardware tracing:

```c
struct perf_aux_event {
    struct perf_event_header header;
    u64                     aux_offset;  /* Offset in AUX area */
    u64                     aux_size;    /* Size of AUX data */
    u64                     flags;       /* AUX-specific flags */
};
```

AUX buffers support:
- **Snapshot Mode**: Capture trace data at specific intervals
- **Free-running Mode**: Continuous tracing with circular buffer
- **Hardware Integration**: Direct hardware writing to avoid kernel overhead

## Event Configuration and Sampling Mechanisms

### Event Configuration Options

Events are configured through the `perf_event_attr` structure with multiple configuration vectors:

#### Primary Configuration (`config` field)
- **Hardware Events**: Use predefined constants (PERF_COUNT_HW_*)
- **Software Events**: Use software event constants (PERF_COUNT_SW_*)
- **Raw Events**: Direct hardware register values
- **Cache Events**: Encoded cache operations (cache_id | op_id | result_id)

```c
/* Hardware event configuration */
attr.type = PERF_TYPE_HARDWARE;
attr.config = PERF_COUNT_HW_CPU_CYCLES;

/* Raw event configuration (vendor-specific) */
attr.type = PERF_TYPE_RAW;
attr.config = 0x01c0; /* Intel raw event encoding */

/* Cache event configuration */
attr.type = PERF_TYPE_HW_CACHE;
attr.config = PERF_COUNT_HW_CACHE_L1D |
              (PERF_COUNT_HW_CACHE_OP_READ << 8) |
              (PERF_COUNT_HW_CACHE_RESULT_MISS << 16);
```

#### Extended Configuration (`config1`, `config2`)
Allow additional PMU-specific parameters:

```c
/* GPU-specific configuration example */
attr.config1 = gpu_engine_mask;     /* Which GPU engines to monitor */
attr.config2 = (gpu_instance << 8) | gpu_context_id;
```

### Sampling Mechanisms

#### Frequency-based Sampling
Automatically adjusts sampling period to achieve target frequency:

```c
attr.freq = 1;
attr.sample_freq = 1000; /* Target 1000 samples/second */
attr.sample_type = PERF_SAMPLE_IP | PERF_SAMPLE_TID | PERF_SAMPLE_TIME;
```

#### Period-based Sampling
Fixed interval sampling:

```c
attr.freq = 0;
attr.sample_period = 1000000; /* Sample every 1M events */
```

#### Watermark-based Wakeup
Control when userspace gets notified:

```c
attr.watermark = 1;
attr.wakeup_watermark = 64 * 1024; /* Wake up when 64KB available */
```

### Sample Data Types

Comprehensive sample data collection options:

```c
enum perf_event_sample_format {
    PERF_SAMPLE_IP               = 1U << 0,  /* Instruction pointer */
    PERF_SAMPLE_TID              = 1U << 1,  /* Process/thread ID */
    PERF_SAMPLE_TIME             = 1U << 2,  /* Timestamp */
    PERF_SAMPLE_ADDR             = 1U << 3,  /* Memory address */
    PERF_SAMPLE_READ             = 1U << 4,  /* Read format data */
    PERF_SAMPLE_CALLCHAIN        = 1U << 5,  /* Call stack */
    PERF_SAMPLE_ID               = 1U << 6,  /* Event ID */
    PERF_SAMPLE_CPU              = 1U << 7,  /* CPU number */
    PERF_SAMPLE_PERIOD           = 1U << 8,  /* Sample period */
    PERF_SAMPLE_STREAM_ID        = 1U << 9,  /* Stream ID */
    PERF_SAMPLE_RAW              = 1U << 10, /* Raw data */
    PERF_SAMPLE_BRANCH_STACK     = 1U << 11, /* Branch records */
    PERF_SAMPLE_REGS_USER        = 1U << 12, /* User registers */
    PERF_SAMPLE_STACK_USER       = 1U << 13, /* User stack */
    PERF_SAMPLE_WEIGHT           = 1U << 14, /* Memory latency */
    PERF_SAMPLE_DATA_SRC         = 1U << 15, /* Memory hierarchy */
    PERF_SAMPLE_IDENTIFIER       = 1U << 16, /* Unique sample ID */
    PERF_SAMPLE_TRANSACTION      = 1U << 17, /* Transaction abort info */
    PERF_SAMPLE_REGS_INTR        = 1U << 18, /* Interrupt registers */
    PERF_SAMPLE_PHYS_ADDR        = 1U << 19, /* Physical address */
    PERF_SAMPLE_AUX              = 1U << 20, /* AUX area sample */
    PERF_SAMPLE_CGROUP           = 1U << 21, /* Cgroup information */
    PERF_SAMPLE_DATA_PAGE_SIZE   = 1U << 22, /* Data page size */
    PERF_SAMPLE_CODE_PAGE_SIZE   = 1U << 23, /* Code page size */
    PERF_SAMPLE_WEIGHT_STRUCT    = 1U << 24, /* Structured weight */
};
```

### Event Multiplexing

When more events are requested than available hardware counters:

#### Round-Robin Scheduling
Events are time-sliced with configurable intervals:

```c
/* In PMU driver */
static void gpu_pmu_rotate_context(struct perf_cpu_context *cpuctx)
{
    struct perf_event_context *ctx = &cpuctx->ctx;

    /* Rotate flexible events */
    if (!list_empty(&ctx->flexible_groups.active_list))
        ctx_sched_out(ctx, cpuctx, EVENT_FLEXIBLE);

    /* Schedule new events */
    ctx_sched_in(ctx, cpuctx, EVENT_FLEXIBLE, current);
}
```

#### Scaling Calculations
Statistical scaling based on actual runtime:

```c
u64 scaled_count = raw_count * (time_enabled / time_running);
```

## Userspace Interface and Tool Integration

### perf_event_open() System Call

Primary interface for creating performance events:

```c
#include <linux/perf_event.h>
#include <sys/syscall.h>

int perf_event_open(struct perf_event_attr *attr,
                   pid_t pid, int cpu, int group_fd,
                   unsigned long flags);
```

#### Usage Patterns

**Per-process monitoring:**
```c
struct perf_event_attr attr = {
    .type = PERF_TYPE_HARDWARE,
    .config = PERF_COUNT_HW_CPU_CYCLES,
    .disabled = 1,
    .exclude_kernel = 1,
    .exclude_hv = 1,
};

int fd = perf_event_open(&attr, getpid(), -1, -1, 0);
```

**System-wide monitoring:**
```c
int fd = perf_event_open(&attr, -1, 0, -1, 0); /* CPU 0 */
```

**Event grouping:**
```c
int leader_fd = perf_event_open(&leader_attr, pid, cpu, -1, 0);
int member_fd = perf_event_open(&member_attr, pid, cpu, leader_fd, 0);
```

### Sysfs Interface

PMU drivers expose events and capabilities through sysfs:

```
/sys/bus/event_source/devices/gpu_pmu/
├── events/
│   ├── engine0-busy
│   ├── engine1-busy
│   ├── memory-bandwidth
│   └── cache-misses
├── format/
│   ├── engine
│   ├── instance
│   └── config
├── type
├── cpumask
└── capabilities/
    ├── no_interrupt
    └── no_exclude
```

#### Event Attribute Files
Each event file contains the configuration needed:

```bash
# /sys/bus/event_source/devices/gpu_pmu/events/engine0-busy
echo "config=0x0,engine=0x1"

# /sys/bus/event_source/devices/gpu_pmu/events/memory-bandwidth
echo "config=0x10,instance=0xff"
```

#### Format Attribute Files
Define configuration field layouts:

```bash
# /sys/bus/event_source/devices/gpu_pmu/format/engine
echo "config1:0-7"

# /sys/bus/event_source/devices/gpu_pmu/format/instance
echo "config1:8-15"
```

### Integration with Perf Tools

#### perf list Integration
PMU events automatically appear in `perf list` output:

```bash
$ perf list | grep gpu_pmu
  gpu_pmu/engine0-busy/                      [Kernel PMU event]
  gpu_pmu/engine1-busy/                      [Kernel PMU event]
  gpu_pmu/memory-bandwidth/                  [Kernel PMU event]
```

#### perf stat Integration
Basic counting mode:

```bash
$ perf stat -e gpu_pmu/engine0-busy/ -a sleep 5
 Performance counter stats for 'system wide':

    45,324,567      gpu_pmu/engine0-busy/

       5.002381063 seconds time elapsed
```

#### perf record Integration
Sampling mode with custom PMU events:

```bash
$ perf record -e gpu_pmu/cache-misses/u -g ./gpu_workload
$ perf report --stdio
```

### Memory Mapping Interface

Efficient data access through mmap():

```c
int mmap_size = (1 + BUFFER_SIZE_PAGES) * getpagesize();
void *mmap_base = mmap(NULL, mmap_size, PROT_READ | PROT_WRITE,
                       MAP_SHARED, fd, 0);

struct perf_event_mmap_page *metadata = mmap_base;
void *data_base = mmap_base + getpagesize();

/* Reading samples */
while (metadata->data_tail != metadata->data_head) {
    struct perf_event_header *header =
        (void*)data_base + (metadata->data_tail % BUFFER_SIZE);

    if (header->type == PERF_RECORD_SAMPLE) {
        /* Process sample */
    }

    metadata->data_tail += header->size;
}
```

## Example Code for PMU Driver Implementation

### Intel i915 GPU PMU Driver Analysis

The Intel i915 driver provides an excellent reference implementation for GPU PMU integration. Key architectural elements:

#### PMU Structure Definition

```c
struct i915_pmu {
    struct pmu base;                    /* Base PMU structure */
    struct drm_i915_private *i915;     /* i915 device instance */

    spinlock_t lock;                   /* Synchronization */
    struct hrtimer timer;              /* Sampling timer */
    u64 timer_period_ns;               /* Timer period */

    /* Per-GT tracking */
    struct i915_pmu_gt {
        u64 events[I915_PMU_LAST_EVENT]; /* Event enable mask */
        u64 samples[I915_PMU_GT_LAST];   /* Sampled values */
    } *gt;

    /* Event tracking */
    u64 enable;                        /* Global enable mask */
    u64 enable_count[I915_PMU_MASK_BITS]; /* Reference counting */

    /* CPU assignment */
    unsigned int cpu;
    struct cpuhp_state *cpuhp_state;
};
```

#### Timer-based Sampling Implementation

```c
static enum hrtimer_restart i915_sample(struct hrtimer *hrtimer)
{
    struct i915_pmu *pmu = container_of(hrtimer, struct i915_pmu, timer);
    struct drm_i915_private *i915 = pmu->i915;
    unsigned long flags;

    if (!READ_ONCE(pmu->timer_enabled))
        return HRTIMER_NORESTART;

    spin_lock_irqsave(&pmu->lock, flags);

    for_each_gt(gt, i915, i) {
        if (gt->pmu.events & SAMPLE_MASK)
            sample_gt(gt);
    }

    spin_unlock_irqrestore(&pmu->lock, flags);

    hrtimer_forward_now(hrtimer, ns_to_ktime(pmu->timer_period_ns));
    return HRTIMER_RESTART;
}

static void sample_gt(struct intel_gt *gt)
{
    struct i915_pmu_gt *pmu_gt = &gt->pmu;

    /* Sample engine busy times */
    if (pmu_gt->events & ENGINE_SAMPLE_MASK) {
        for_each_engine(engine, gt, id) {
            if (pmu_gt->events & BIT_ULL(engine_event_sample(engine, I915_SAMPLE_BUSY)))
                pmu_gt->samples[I915_SAMPLE_BUSY] +=
                    intel_engine_get_busy_time(engine);
        }
    }

    /* Sample frequency */
    if (pmu_gt->events & FREQUENCY_SAMPLE_MASK) {
        pmu_gt->samples[I915_SAMPLE_FREQ_ACT] =
            intel_rps_get_cagf(&gt->rps, intel_rps_read_actual_frequency(&gt->rps));
        pmu_gt->samples[I915_SAMPLE_FREQ_REQ] =
            intel_rps_get_cagf(&gt->rps, intel_rps_read_requested_frequency(&gt->rps));
    }

    /* Sample RC6 residency */
    if (pmu_gt->events & BIT_ULL(I915_PMU_RC6_RESIDENCY))
        pmu_gt->samples[I915_SAMPLE_RC6] = intel_rc6_residency_ns(&gt->rc6, GEN6_GT_GFX_RC6);
}
```

#### Event Configuration and Validation

```c
static int i915_pmu_event_init(struct perf_event *event)
{
    struct drm_i915_private *i915 = container_of(event->pmu, typeof(*i915), pmu.base);
    struct intel_gt *gt;
    u64 config = event->attr.config;
    int cpu, ret = 0;

    if (event->attr.type != event->pmu->type)
        return -ENOENT;

    if (has_branch_stack(event))
        return -EOPNOTSUPP;

    if (event->cpu < 0)
        return -EINVAL;

    /* Validate event configuration */
    if (is_engine_event(event)) {
        if (!engine_event_supported(event, i915))
            return -ENOENT;
    } else {
        if (!config_status(i915, config))
            return -ENOENT;
    }

    /* Assign to appropriate GT */
    if (is_engine_event(event) || is_gt_event(event)) {
        gt = intel_gt_get(to_gt(i915));
        if (!gt)
            return -ENODEV;
        event->hw.config_base = (unsigned long)gt;
    }

    /* Set up CPU context */
    cpu = cpumask_any(topology_sibling_cpumask(event->cpu));
    if (cpu >= nr_cpu_ids)
        cpu = event->cpu;

    event->hw.idx = -1;
    event->hw.config = config;

    return ret;
}
```

### AMD ROCm Integration Patterns

Based on the analysis of ROCm's existing perf integration:

#### ROCm PMU Structure Framework

```c
struct amdgpu_pmu {
    struct pmu base;                    /* Base PMU structure */
    struct amdgpu_device *adev;         /* AMDGPU device */

    /* Performance counter configuration */
    struct {
        void __iomem *regs;             /* Counter register base */
        u32 num_counters;               /* Available counter count */
        u32 counter_width;              /* Counter width in bits */
        const struct amdgpu_pmu_config *config; /* PMU configuration */
    } hw;

    /* Event management */
    struct {
        DECLARE_BITMAP(used_mask, AMDGPU_PMU_MAX_COUNTERS);
        struct perf_event *events[AMDGPU_PMU_MAX_COUNTERS];
        struct hrtimer timer;           /* Overflow timer */
        u64 timer_period;
    } events;

    /* Per-engine tracking for GPU blocks */
    struct amdgpu_pmu_block {
        const char *name;               /* Block name (GFX, SDMA, etc.) */
        void __iomem *regs;             /* Block register base */
        u32 num_instances;              /* Number of instances */
        u32 num_counters_per_instance;  /* Counters per instance */
    } blocks[AMDGPU_PMU_MAX_BLOCKS];

    spinlock_t lock;                    /* Synchronization */
    int cpu;                            /* Assigned CPU */
};
```

#### GPU Block-Specific Event Handling

```c
/* Event configuration for different GPU blocks */
enum amdgpu_pmu_event_id {
    /* Graphics Engine Events */
    AMDGPU_PMU_EVENT_GFX_BUSY               = 0x00,
    AMDGPU_PMU_EVENT_GFX_CYCLES             = 0x01,
    AMDGPU_PMU_EVENT_GFX_VERTEX_THROUGHPUT  = 0x02,
    AMDGPU_PMU_EVENT_GFX_PIXEL_THROUGHPUT   = 0x03,

    /* Memory Controller Events */
    AMDGPU_PMU_EVENT_MC_READ_REQUESTS       = 0x10,
    AMDGPU_PMU_EVENT_MC_WRITE_REQUESTS      = 0x11,
    AMDGPU_PMU_EVENT_MC_READ_BYTES          = 0x12,
    AMDGPU_PMU_EVENT_MC_WRITE_BYTES         = 0x13,

    /* SDMA Engine Events */
    AMDGPU_PMU_EVENT_SDMA_BUSY              = 0x20,
    AMDGPU_PMU_EVENT_SDMA_CYCLES            = 0x21,
    AMDGPU_PMU_EVENT_SDMA_THROUGHPUT        = 0x22,

    /* Cache Events */
    AMDGPU_PMU_EVENT_L1_CACHE_HIT           = 0x30,
    AMDGPU_PMU_EVENT_L1_CACHE_MISS          = 0x31,
    AMDGPU_PMU_EVENT_L2_CACHE_HIT           = 0x32,
    AMDGPU_PMU_EVENT_L2_CACHE_MISS          = 0x33,
};

static int amdgpu_pmu_event_init(struct perf_event *event)
{
    struct amdgpu_pmu *pmu = to_amdgpu_pmu(event->pmu);
    struct hw_perf_event *hwc = &event->hw;
    u64 config = event->attr.config;
    u32 block_id, instance, counter_id;

    if (event->attr.type != event->pmu->type)
        return -ENOENT;

    /* Decode event configuration */
    block_id = (config >> 16) & 0xFF;
    instance = (config >> 8) & 0xFF;
    counter_id = config & 0xFF;

    /* Validate block and instance */
    if (block_id >= AMDGPU_PMU_MAX_BLOCKS)
        return -EINVAL;

    if (instance >= pmu->blocks[block_id].num_instances)
        return -EINVAL;

    /* Set up hardware event */
    hwc->config = config;
    hwc->config_base = block_id;
    hwc->idx = -1;

    /* GPU events don't support period-based sampling */
    if (is_sampling_event(event))
        return -EOPNOTSUPP;

    return 0;
}
```

### Driver Registration Template

Complete registration template for AMD GPU PMU:

```c
static struct attribute *amdgpu_pmu_events_attrs[] = {
    PMU_EVENT_ATTR(gfx-busy,        gfx_busy,        "config=0x00"),
    PMU_EVENT_ATTR(gfx-cycles,      gfx_cycles,      "config=0x01"),
    PMU_EVENT_ATTR(mc-read-bytes,   mc_read_bytes,   "config=0x12"),
    PMU_EVENT_ATTR(mc-write-bytes,  mc_write_bytes,  "config=0x13"),
    PMU_EVENT_ATTR(sdma-busy,       sdma_busy,       "config=0x20"),
    PMU_EVENT_ATTR(l2-cache-hit,    l2_cache_hit,    "config=0x32"),
    PMU_EVENT_ATTR(l2-cache-miss,   l2_cache_miss,   "config=0x33"),
    NULL
};

static struct attribute *amdgpu_pmu_format_attrs[] = {
    PMU_FORMAT_ATTR(config, "config:0-63"),
    NULL
};

static const struct attribute_group amdgpu_pmu_events_attr_group = {
    .name = "events",
    .attrs = amdgpu_pmu_events_attrs,
};

static const struct attribute_group amdgpu_pmu_format_attr_group = {
    .name = "format",
    .attrs = amdgpu_pmu_format_attrs,
};

static const struct attribute_group *amdgpu_pmu_attr_groups[] = {
    &amdgpu_pmu_events_attr_group,
    &amdgpu_pmu_format_attr_group,
    NULL
};

static struct pmu amdgpu_pmu_pmu = {
    .task_ctx_nr    = perf_invalid_context,
    .event_init     = amdgpu_pmu_event_init,
    .add            = amdgpu_pmu_add,
    .del            = amdgpu_pmu_del,
    .start          = amdgpu_pmu_start,
    .stop           = amdgpu_pmu_stop,
    .read           = amdgpu_pmu_read,
    .attr_groups    = amdgpu_pmu_attr_groups,
    .capabilities   = PERF_PMU_CAP_NO_INTERRUPT | PERF_PMU_CAP_NO_EXCLUDE,
};

int amdgpu_pmu_init(struct amdgpu_device *adev)
{
    struct amdgpu_pmu *pmu;
    int ret;

    pmu = kzalloc(sizeof(*pmu), GFP_KERNEL);
    if (!pmu)
        return -ENOMEM;

    pmu->adev = adev;
    pmu->base = amdgpu_pmu_pmu;
    spin_lock_init(&pmu->lock);

    /* Initialize hardware-specific configuration */
    ret = amdgpu_pmu_hw_init(pmu);
    if (ret)
        goto err_free;

    /* Register with perf subsystem */
    ret = perf_pmu_register(&pmu->base, "amdgpu", PERF_TYPE_RAW);
    if (ret)
        goto err_hw_fini;

    adev->pmu = pmu;
    return 0;

err_hw_fini:
    amdgpu_pmu_hw_fini(pmu);
err_free:
    kfree(pmu);
    return ret;
}
```

## Specific Requirements for GPU Performance Counters

### GPU-Specific Considerations

GPU performance monitoring presents unique challenges compared to CPU PMUs:

#### Multi-Engine Architecture
Modern GPUs contain multiple execution engines that require independent monitoring:

```c
enum amdgpu_engine_type {
    AMDGPU_ENGINE_GFX,      /* Graphics engine */
    AMDGPU_ENGINE_COMPUTE,  /* Compute shader engine */
    AMDGPU_ENGINE_SDMA0,    /* System DMA engine 0 */
    AMDGPU_ENGINE_SDMA1,    /* System DMA engine 1 */
    AMDGPU_ENGINE_VCN_DEC,  /* Video decode engine */
    AMDGPU_ENGINE_VCN_ENC,  /* Video encode engine */
    AMDGPU_ENGINE_UVD,      /* Unified video decoder */
    AMDGPU_ENGINE_VCE,      /* Video compression engine */
    AMDGPU_ENGINE_MAX
};

struct amdgpu_engine_pmu {
    const char *name;
    enum amdgpu_engine_type type;
    void __iomem *mmio_base;
    u32 perfmon_offset;
    u32 num_counters;
    u32 counter_width;
    bool supports_overflow;
};
```

#### Memory Hierarchy Monitoring
GPU memory systems require specialized counter types:

```c
enum amdgpu_memory_event {
    /* L1 Cache Events */
    AMDGPU_L1_CACHE_REQUESTS,
    AMDGPU_L1_CACHE_HITS,
    AMDGPU_L1_CACHE_MISSES,
    AMDGPU_L1_CACHE_EVICTIONS,

    /* L2 Cache Events */
    AMDGPU_L2_CACHE_REQUESTS,
    AMDGPU_L2_CACHE_HITS,
    AMDGPU_L2_CACHE_MISSES,
    AMDGPU_L2_CACHE_WRITES,

    /* Memory Controller Events */
    AMDGPU_MC_READ_REQUESTS,
    AMDGPU_MC_WRITE_REQUESTS,
    AMDGPU_MC_READ_BYTES,
    AMDGPU_MC_WRITE_BYTES,
    AMDGPU_MC_READ_LATENCY,
    AMDGPU_MC_WRITE_LATENCY,

    /* VRAM Events */
    AMDGPU_VRAM_READ_BYTES,
    AMDGPU_VRAM_WRITE_BYTES,
    AMDGPU_VRAM_UTILIZATION,
};
```

#### Power and Thermal Monitoring
GPU performance is heavily influenced by power and thermal states:

```c
struct amdgpu_power_events {
    u64 gpu_power_consumption;      /* Watts */
    u64 gpu_energy_consumption;     /* Joules */
    u64 memory_power_consumption;   /* Watts */
    u64 total_board_power;          /* Watts */
    u64 gpu_temperature;            /* Celsius */
    u64 memory_temperature;         /* Celsius */
    u64 power_limit_throttling;     /* Bool/duration */
    u64 thermal_throttling;         /* Bool/duration */
};
```

### Counter Implementation Strategies

#### Hardware Counter Access Patterns

```c
/* Direct MMIO register access for performance counters */
static u64 amdgpu_pmu_read_counter(struct amdgpu_pmu *pmu, int idx)
{
    struct amdgpu_device *adev = pmu->adev;
    u32 reg_offset = pmu->hw.counter_base + (idx * 8);
    u64 value = 0;

    /* Ensure GPU is powered and accessible */
    if (amdgpu_device_is_px(adev) && !adev->is_px_enabled)
        return 0;

    /* Read 64-bit counter value */
    value = RREG32(reg_offset);
    if (pmu->hw.counter_width > 32)
        value |= ((u64)RREG32(reg_offset + 4)) << 32;

    return value;
}

static void amdgpu_pmu_write_counter(struct amdgpu_pmu *pmu, int idx, u64 value)
{
    struct amdgpu_device *adev = pmu->adev;
    u32 reg_offset = pmu->hw.counter_base + (idx * 8);

    WREG32(reg_offset, lower_32_bits(value));
    if (pmu->hw.counter_width > 32)
        WREG32(reg_offset + 4, upper_32_bits(value));
}
```

#### Event Multiplexing for Limited Counters

```c
struct amdgpu_pmu_group {
    struct list_head events;
    struct hrtimer rotation_timer;
    u64 rotation_period_ns;
    int active_events;
    int max_events;
};

static void amdgpu_pmu_rotate_events(struct amdgpu_pmu *pmu)
{
    struct amdgpu_pmu_group *group = &pmu->event_group;
    struct perf_event *event, *next;
    int available_counters = pmu->hw.num_counters;
    int scheduled = 0;

    /* Stop current events */
    list_for_each_entry(event, &group->events, hw.entry) {
        if (event->hw.idx >= 0) {
            amdgpu_pmu_stop(event, PERF_EF_UPDATE);
            event->hw.idx = -1;
            scheduled++;
        }
    }

    /* Start next batch of events */
    list_for_each_entry_safe(event, next, &group->events, hw.entry) {
        if (scheduled >= available_counters)
            break;

        if (event->hw.idx < 0) {
            if (amdgpu_pmu_add(event, PERF_EF_START) == 0)
                scheduled++;
        }
    }

    /* Move scheduled events to end of list for round-robin */
    list_splice_init(&group->events, &group->events);
}
```

### GPU Context and Process Isolation

#### GPU Context Tracking
Modern GPUs support multiple contexts that need separate monitoring:

```c
struct amdgpu_ctx_pmu {
    u32 ctx_id;                     /* GPU context identifier */
    struct pid *tgid;               /* Owning process */
    struct list_head ctx_events;   /* Events for this context */
    u64 ctx_switch_count;           /* Context switch frequency */
    ktime_t ctx_start_time;         /* Context activation time */
    ktime_t ctx_total_time;         /* Total active time */
};

static void amdgpu_pmu_ctx_switch(struct amdgpu_device *adev,
                                 u32 old_ctx_id, u32 new_ctx_id)
{
    struct amdgpu_pmu *pmu = adev->pmu;
    struct amdgpu_ctx_pmu *old_ctx, *new_ctx;
    ktime_t now = ktime_get();

    /* Update outgoing context */
    old_ctx = amdgpu_pmu_find_ctx(pmu, old_ctx_id);
    if (old_ctx) {
        old_ctx->ctx_total_time = ktime_add(old_ctx->ctx_total_time,
            ktime_sub(now, old_ctx->ctx_start_time));
    }

    /* Activate incoming context */
    new_ctx = amdgpu_pmu_find_ctx(pmu, new_ctx_id);
    if (new_ctx) {
        new_ctx->ctx_start_time = now;
        new_ctx->ctx_switch_count++;
    }

    /* Update per-context counters */
    amdgpu_pmu_update_ctx_counters(pmu, old_ctx, new_ctx);
}
```

### Interrupt and Overflow Handling

#### Counter Overflow Management
GPU counters often have limited width and require overflow handling:

```c
static void amdgpu_pmu_overflow_handler(struct amdgpu_pmu *pmu)
{
    int i;

    for (i = 0; i < pmu->hw.num_counters; i++) {
        struct perf_event *event = pmu->events.events[i];
        struct hw_perf_event *hwc;
        u64 prev, new, delta;

        if (!event)
            continue;

        hwc = &event->hw;

        /* Read current counter value */
        new = amdgpu_pmu_read_counter(pmu, i);
        prev = local64_read(&hwc->prev_count);

        /* Handle overflow */
        delta = (new - prev) & ((1ULL << pmu->hw.counter_width) - 1);

        /* Update event count */
        local64_add(delta, &event->count);
        local64_set(&hwc->prev_count, new);

        /* Check for sampling events */
        if (is_sampling_event(event)) {
            hwc->interrupts++;
            if (delta >= hwc->sample_period) {
                /* Generate sample record */
                perf_event_overflow(event, &data, regs);
            }
        }
    }
}
```

### Memory Bandwidth and Latency Measurement

#### Advanced Memory Monitoring
GPU memory performance requires specialized measurement techniques:

```c
struct amdgpu_memory_counter {
    const char *name;
    u32 config_value;
    enum {
        AMDGPU_MEM_COUNTER_BYTES,
        AMDGPU_MEM_COUNTER_TRANSACTIONS,
        AMDGPU_MEM_COUNTER_LATENCY,
        AMDGPU_MEM_COUNTER_UTILIZATION,
    } type;
    u32 instance_mask;              /* Which memory controllers */
};

static const struct amdgpu_memory_counter memory_events[] = {
    {"vram-read-bytes",    0x100, AMDGPU_MEM_COUNTER_BYTES, 0xFF},
    {"vram-write-bytes",   0x101, AMDGPU_MEM_COUNTER_BYTES, 0xFF},
    {"vram-read-latency",  0x102, AMDGPU_MEM_COUNTER_LATENCY, 0xFF},
    {"vram-write-latency", 0x103, AMDGPU_MEM_COUNTER_LATENCY, 0xFF},
    {"vram-utilization",   0x104, AMDGPU_MEM_COUNTER_UTILIZATION, 0xFF},
};

static u64 amdgpu_pmu_read_memory_counter(struct amdgpu_pmu *pmu,
                                         const struct amdgpu_memory_counter *counter)
{
    struct amdgpu_device *adev = pmu->adev;
    u64 total = 0;
    int instance;

    /* Aggregate across memory controller instances */
    for (instance = 0; instance < adev->mc.num_instances; instance++) {
        if (!(counter->instance_mask & BIT(instance)))
            continue;

        u32 reg_addr = adev->mc.instance[instance].perfmon_base +
                       counter->config_value * 8;
        u64 value = RREG32(reg_addr) |
                   ((u64)RREG32(reg_addr + 4) << 32);

        total += value;
    }

    return total;
}
```

## Best Practices and Common Patterns

### Design Principles for GPU PMU Drivers

#### 1. Minimize GPU Interference
Performance monitoring should not significantly impact GPU performance:

```c
/* Use shadow registers when possible to avoid GPU stalls */
static void amdgpu_pmu_enable_counter_shadow(struct amdgpu_pmu *pmu, int idx)
{
    struct amdgpu_device *adev = pmu->adev;

    /* Configure counter in shadow register space */
    u32 shadow_reg = adev->perfmon.shadow_base + (idx * 4);
    WREG32(shadow_reg, pmu->hw.events[idx].config);

    /* Atomically enable via control register */
    u32 ctrl_reg = adev->perfmon.control_base;
    u32 enable_mask = BIT(idx);
    WREG32_OR(ctrl_reg, enable_mask);
}

/* Batch register updates to minimize GPU interruption */
static void amdgpu_pmu_update_multiple_counters(struct amdgpu_pmu *pmu,
                                               struct counter_update *updates,
                                               int count)
{
    struct amdgpu_device *adev = pmu->adev;
    int i;

    /* Disable interrupts during batch update */
    amdgpu_pmu_disable_interrupts(pmu);

    /* Update all counters in batch */
    for (i = 0; i < count; i++) {
        u32 reg_addr = updates[i].reg_addr;
        u32 value = updates[i].value;
        WREG32(reg_addr, value);
    }

    /* Re-enable interrupts */
    amdgpu_pmu_enable_interrupts(pmu);
}
```

#### 2. Efficient Resource Management
GPU resources are precious and must be managed carefully:

```c
struct amdgpu_pmu_resource_pool {
    DECLARE_BITMAP(counter_pool, AMDGPU_PMU_MAX_COUNTERS);
    DECLARE_BITMAP(timer_pool, AMDGPU_PMU_MAX_TIMERS);
    struct mutex allocation_lock;

    /* Priority-based allocation */
    struct list_head high_priority_events;
    struct list_head normal_priority_events;
    struct list_head background_events;
};

static int amdgpu_pmu_allocate_counter(struct amdgpu_pmu *pmu,
                                      struct perf_event *event)
{
    struct amdgpu_pmu_resource_pool *pool = &pmu->resource_pool;
    int counter_idx = -1;

    mutex_lock(&pool->allocation_lock);

    /* Try to allocate based on event priority */
    if (event->attr.pinned) {
        /* High priority - try dedicated counters first */
        counter_idx = find_first_zero_bit(pool->counter_pool,
                                         AMDGPU_PMU_DEDICATED_COUNTERS);
    } else {
        /* Normal priority - use shared counter pool */
        counter_idx = find_first_zero_bit(pool->counter_pool,
                                         AMDGPU_PMU_MAX_COUNTERS);
    }

    if (counter_idx < AMDGPU_PMU_MAX_COUNTERS) {
        set_bit(counter_idx, pool->counter_pool);
        pmu->events.events[counter_idx] = event;
        event->hw.idx = counter_idx;
    }

    mutex_unlock(&pool->allocation_lock);
    return counter_idx >= 0 ? 0 : -EAGAIN;
}
```

#### 3. Power-Aware Implementation
GPU PMUs must handle power management gracefully:

```c
static int amdgpu_pmu_runtime_suspend(struct device *dev)
{
    struct amdgpu_device *adev = dev_get_drvdata(dev);
    struct amdgpu_pmu *pmu = adev->pmu;
    int i;

    if (!pmu)
        return 0;

    /* Save counter state before suspend */
    for (i = 0; i < pmu->hw.num_counters; i++) {
        if (pmu->events.events[i]) {
            pmu->suspend_state.counter_values[i] =
                amdgpu_pmu_read_counter(pmu, i);
            pmu->suspend_state.counter_configs[i] =
                RREG32(pmu->hw.config_base + i * 4);
        }
    }

    /* Stop timer */
    hrtimer_cancel(&pmu->events.timer);

    return 0;
}

static int amdgpu_pmu_runtime_resume(struct device *dev)
{
    struct amdgpu_device *adev = dev_get_drvdata(dev);
    struct amdgpu_pmu *pmu = adev->pmu;
    int i;

    if (!pmu)
        return 0;

    /* Restore counter configurations */
    for (i = 0; i < pmu->hw.num_counters; i++) {
        if (pmu->events.events[i]) {
            WREG32(pmu->hw.config_base + i * 4,
                   pmu->suspend_state.counter_configs[i]);
            amdgpu_pmu_write_counter(pmu, i,
                   pmu->suspend_state.counter_values[i]);
        }
    }

    /* Restart timer if needed */
    if (pmu->events.active_events > 0)
        hrtimer_start(&pmu->events.timer,
                     ns_to_ktime(pmu->events.timer_period),
                     HRTIMER_MODE_REL_PINNED);

    return 0;
}
```

### Security and Access Control

#### Process Isolation
Ensure proper isolation between processes:

```c
static bool amdgpu_pmu_event_access_ok(struct perf_event *event)
{
    /* System-wide events require CAP_PERFMON */
    if (event->cpu >= 0 && event->pid == -1) {
        if (!perfmon_capable())
            return false;
    }

    /* Per-process events must match current process or have ptrace access */
    if (event->pid >= 0) {
        struct task_struct *task = find_task_by_vpid(event->pid);
        if (!task)
            return false;

        if (task != current && !ptrace_may_access(task, PTRACE_MODE_READ))
            return false;
    }

    return true;
}
```

#### Sensitive Information Protection
Avoid exposing sensitive GPU information:

```c
static void amdgpu_pmu_sanitize_sample(struct perf_event *event,
                                      struct perf_sample_data *data)
{
    /* Remove sensitive addresses from samples */
    if (data->addr && !perfmon_capable()) {
        data->addr = 0;
        data->sample_flags &= ~PERF_SAMPLE_ADDR;
    }

    /* Filter raw data for unprivileged users */
    if (data->raw && !perfmon_capable()) {
        struct amdgpu_raw_data *raw = data->raw->data;
        raw->gpu_virtual_addr = 0;
        raw->context_id = 0;
    }
}
```

### Error Handling and Robustness

#### Graceful Degradation
Handle hardware failures gracefully:

```c
static void amdgpu_pmu_handle_hw_error(struct amdgpu_pmu *pmu, u32 error_code)
{
    struct amdgpu_device *adev = pmu->adev;

    switch (error_code) {
    case AMDGPU_PMU_ERROR_COUNTER_OVERFLOW:
        /* Reset overflowed counters */
        amdgpu_pmu_reset_overflow_counters(pmu);
        break;

    case AMDGPU_PMU_ERROR_MMIO_TIMEOUT:
        /* Disable PMU temporarily */
        amdgpu_pmu_disable_all_events(pmu);
        schedule_delayed_work(&pmu->recovery_work,
                             msecs_to_jiffies(1000));
        break;

    case AMDGPU_PMU_ERROR_INVALID_CONFIG:
        /* Stop problematic events */
        amdgpu_pmu_stop_invalid_events(pmu);
        break;

    default:
        dev_warn(adev->dev, "Unknown PMU error: 0x%x\n", error_code);
    }
}

static void amdgpu_pmu_recovery_work(struct work_struct *work)
{
    struct amdgpu_pmu *pmu = container_of(work, struct amdgpu_pmu,
                                         recovery_work.work);

    /* Attempt to re-initialize PMU hardware */
    if (amdgpu_pmu_hw_reinit(pmu) == 0) {
        /* Re-enable events that were running */
        amdgpu_pmu_restore_events(pmu);
        dev_info(pmu->adev->dev, "PMU recovery successful\n");
    } else {
        /* Schedule another recovery attempt */
        schedule_delayed_work(&pmu->recovery_work,
                             msecs_to_jiffies(5000));
    }
}
```

### Performance Optimization Guidelines

#### Minimize Overhead
Keep PMU overhead minimal:

```c
/* Use efficient data structures for hot paths */
struct amdgpu_pmu_fast_path {
    /* Cacheline-aligned for performance */
    struct {
        u64 counter_values[AMDGPU_PMU_MAX_COUNTERS];
        u64 prev_values[AMDGPU_PMU_MAX_COUNTERS];
        u32 event_configs[AMDGPU_PMU_MAX_COUNTERS];
        DECLARE_BITMAP(active_mask, AMDGPU_PMU_MAX_COUNTERS);
    } ____cacheline_aligned;

    /* Read-mostly data */
    void __iomem *counter_regs;
    u32 counter_stride;
    u32 num_active_counters;
};

/* Optimized counter reading with minimal branches */
static void amdgpu_pmu_read_counters_fast(struct amdgpu_pmu *pmu)
{
    struct amdgpu_pmu_fast_path *fast = &pmu->fast_path;
    int i;

    /* Use for_each_set_bit for efficient iteration */
    for_each_set_bit(i, fast->active_mask, AMDGPU_PMU_MAX_COUNTERS) {
        fast->counter_values[i] =
            readq(fast->counter_regs + i * fast->counter_stride);
    }
}
```

### Testing and Validation Framework

#### Comprehensive Test Coverage
Provide thorough testing infrastructure:

```c
#ifdef CONFIG_AMDGPU_PMU_DEBUG
static int amdgpu_pmu_self_test(struct amdgpu_pmu *pmu)
{
    struct test_event {
        u32 config;
        u64 expected_min;
        u64 expected_max;
    } test_events[] = {
        {AMDGPU_PMU_EVENT_GFX_BUSY, 0, UINT64_MAX},
        {AMDGPU_PMU_EVENT_MC_READ_BYTES, 0, UINT64_MAX},
        {AMDGPU_PMU_EVENT_L2_CACHE_HIT, 0, UINT64_MAX},
    };

    int i, ret = 0;

    dev_info(pmu->adev->dev, "Starting PMU self-test\n");

    for (i = 0; i < ARRAY_SIZE(test_events); i++) {
        ret = amdgpu_pmu_test_event(pmu, &test_events[i]);
        if (ret) {
            dev_err(pmu->adev->dev, "Self-test failed for event 0x%x\n",
                   test_events[i].config);
            break;
        }
    }

    dev_info(pmu->adev->dev, "PMU self-test %s\n",
            ret ? "FAILED" : "PASSED");
    return ret;
}
#endif
```

This comprehensive research document provides the foundation for implementing a robust AMD GPU PMU driver that integrates seamlessly with the Linux perf subsystem while following established best practices and architectural patterns.