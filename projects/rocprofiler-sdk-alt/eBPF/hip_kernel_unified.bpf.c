// SPDX-License-Identifier: GPL-2.0
/*
 * Unified HIP + Kernel Dispatch eBPF Tracer
 *
 * This eBPF program:
 * 1. Traces HIP API calls via uprobes on libamdhip64.so
 * 2. Receives kernel dispatch/completion events from userspace shim via ring buffer
 * 3. Maintains unified event stream with correlation
 *
 * Events are sent to a shared ring buffer that the userspace tracer consumes.
 */

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#define MAX_NAME_LEN 64
#define MAX_KERNEL_NAME 256
#define MAX_ARGS 8

// Event types
#define EVENT_HIP_API_ENTRY     0
#define EVENT_HIP_API_EXIT      1
#define EVENT_KERNEL_DISPATCH   2
#define EVENT_KERNEL_COMPLETE   3

// GPU trace event structure (must match shim and userspace)
struct gpu_trace_event {
    __u64 timestamp;
    __u32 pid;
    __u32 tid;
    __u32 event_type;
    __u32 correlation_id;

    // For HIP API events
    char function_name[MAX_NAME_LEN];
    __u64 args[8];
    __u32 arg_count;
    __u64 return_value;

    // For kernel dispatch events
    char kernel_name[MAX_KERNEL_NAME];
    __u64 kernel_object;
    __u32 queue_id;

    // Kernel metadata
    __u32 grid_size_x;
    __u32 grid_size_y;
    __u32 grid_size_z;
    __u32 workgroup_size_x;
    __u32 workgroup_size_y;
    __u32 workgroup_size_z;
    __u32 group_segment_size;
    __u32 private_segment_size;

    // GPU timestamps
    __u64 gpu_start_time;
    __u64 gpu_end_time;
};

// Kernel event from shim (subset of gpu_trace_event)
struct kernel_event {
    __u64 timestamp;
    __u32 pid;
    __u32 tid;
    __u32 event_type;
    __u32 queue_id;
    __u64 agent_id;    // HSA agent handle
    char kernel_name[MAX_KERNEL_NAME];
    __u64 kernel_object;
    __u32 grid_size_x, grid_size_y, grid_size_z;
    __u32 workgroup_size_x, workgroup_size_y, workgroup_size_z;
    __u32 group_segment_size;
    __u32 private_segment_size;
    __u64 gpu_start_time;
    __u64 gpu_end_time;
};

// Ring buffer for sending events to userspace
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// Array map for kernel events from shim (shim pushes here via bpf_map_update_elem)
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct kernel_event);
} kernel_events SEC(".maps");

// Per-thread correlation ID tracking
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u64);   // tid
    __type(value, __u32); // correlation_id
} correlation_map SEC(".maps");

// Global correlation ID counter
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u32);
} correlation_counter SEC(".maps");

// Map correlation ID to function ID
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 10240);
    __type(key, __u32);   // correlation_id
    __type(value, __u32); // function_id
} correlation_to_function SEC(".maps");

// Function name lookup table (indexed by function ID)
struct {
    __uint(type, BPF_MAP_TYPE_ARRAY);
    __uint(max_entries, 64);  // Max 64 HIP functions
    __type(key, __u32);
    __type(value, char[MAX_NAME_LEN]);
} function_names SEC(".maps");

// Map to store function ID per probe point
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 256);
    __type(key, __u64);   // probe address/offset
    __type(value, __u32); // function ID
} probe_function_map SEC(".maps");

// Allocate a new correlation ID
static __always_inline __u32 allocate_correlation_id(void) {
    __u32 zero = 0;
    __u32 *counter = bpf_map_lookup_elem(&correlation_counter, &zero);
    if (!counter) return 0;

    // Read current value and increment
    __u32 id = *counter;
    __u32 next = id + 1;
    bpf_map_update_elem(&correlation_counter, &zero, &next, BPF_ANY);
    return id;
}

// Get current correlation ID for this thread
static __always_inline __u32 get_correlation_id(void) {
    __u64 tid = bpf_get_current_pid_tgid();
    __u32 *corr_id = bpf_map_lookup_elem(&correlation_map, &tid);
    return corr_id ? *corr_id : 0;
}

// Set correlation ID for this thread
static __always_inline void set_correlation_id(__u32 corr_id) {
    __u64 tid = bpf_get_current_pid_tgid();
    bpf_map_update_elem(&correlation_map, &tid, &corr_id, BPF_ANY);
}

// Generic HIP API entry probe
SEC("uprobe/hip_api_entry")
int hip_api_entry(struct pt_regs *ctx) {
    struct gpu_trace_event *event;
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    __u32 tid = (__u32)pid_tgid;

    event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) return 0;

    event->timestamp = bpf_ktime_get_ns();
    event->pid = pid;
    event->tid = tid;
    event->event_type = EVENT_HIP_API_ENTRY;

    // Allocate and store correlation ID
    event->correlation_id = allocate_correlation_id();
    set_correlation_id(event->correlation_id);

    // Get function ID from BPF cookie (set during probe attachment)
    __u64 cookie = bpf_get_attach_cookie(ctx);
    __u32 func_id = (__u32)cookie;

    // Store function ID for this correlation ID (must use stack variables for map update)
    __u32 corr_id_copy = event->correlation_id;
    __u32 func_id_copy = func_id;
    bpf_map_update_elem(&correlation_to_function, &corr_id_copy, &func_id_copy, BPF_ANY);

    // Look up function name
    char *func_name = bpf_map_lookup_elem(&function_names, &func_id);
    if (func_name) {
        // Copy function name
        for (int i = 0; i < MAX_NAME_LEN - 1; i++) {
            event->function_name[i] = func_name[i];
            if (func_name[i] == 0) break;
        }
        event->function_name[MAX_NAME_LEN - 1] = 0;
    } else {
        __builtin_memcpy(event->function_name, "unknown_hip", 12);
    }

    // Capture up to 5 arguments (portable across architectures)
    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx);
    event->args[1] = PT_REGS_PARM2(ctx);
    event->args[2] = PT_REGS_PARM3(ctx);
    event->args[3] = PT_REGS_PARM4(ctx);
    event->args[4] = PT_REGS_PARM5(ctx);
    event->args[5] = 0;
    event->args[6] = 0;
    event->args[7] = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

// Generic HIP API exit probe
SEC("uretprobe/hip_api_exit")
int hip_api_exit(struct pt_regs *ctx) {
    struct gpu_trace_event *event;
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    __u32 tid = (__u32)pid_tgid;

    event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) return 0;

    event->timestamp = bpf_ktime_get_ns();
    event->pid = pid;
    event->tid = tid;
    event->event_type = EVENT_HIP_API_EXIT;

    // Get correlation ID from thread-local storage
    event->correlation_id = get_correlation_id();

    // Capture return value
    event->return_value = PT_REGS_RC(ctx);

    // Look up function name using correlation ID (must use stack variable for map lookup)
    __u32 corr_id_copy = event->correlation_id;
    __u32 *func_id_ptr = bpf_map_lookup_elem(&correlation_to_function, &corr_id_copy);
    if (func_id_ptr) {
        char *func_name = bpf_map_lookup_elem(&function_names, func_id_ptr);
        if (func_name) {
            for (int i = 0; i < MAX_NAME_LEN - 1; i++) {
                event->function_name[i] = func_name[i];
                if (func_name[i] == 0) break;
            }
            event->function_name[MAX_NAME_LEN - 1] = 0;
        } else {
            __builtin_memset(event->function_name, 0, sizeof(event->function_name));
        }

        // Clean up the correlation mapping
        bpf_map_delete_elem(&correlation_to_function, &corr_id_copy);
    } else {
        __builtin_memset(event->function_name, 0, sizeof(event->function_name));
    }

    bpf_ringbuf_submit(event, 0);
    return 0;
}


char LICENSE[] SEC("license") = "GPL";
