// SPDX-License-Identifier: GPL-2.0
#define __TARGET_ARCH_x86
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

// Maximum number of arguments to capture
#define MAX_ARGS 8
#define MAX_STRING_LEN 256

// Event structure for HIP API calls
struct hip_event {
    __u64 timestamp;
    __u32 pid;
    __u32 tid;
    __u32 event_type; // 0 = entry, 1 = exit, 2 = kernel_dispatch
    char function_name[64];
    __u64 args[MAX_ARGS];
    __u64 return_value;
    __u32 arg_count;
    __u32 function_id; // Unique ID to match entry/exit events
};

// Event structure for kernel dispatch events
struct kernel_dispatch_event {
    __u64 timestamp;
    __u32 pid;
    __u32 tid;
    __u32 event_type; // 2 = kernel_dispatch, 3 = kernel_completion
    char event_name[64]; // "cs_ioctl", "sched_run_job", "drm_sched_job_run", "kernel_completion"
    __u64 fence_context;
    __u64 fence_seqno;
    char ring_name[32];
    __u32 num_ibs;
    __u32 job_count;
    __u32 hw_job_count;
    __u64 client_id;
    char device_name[32];
    // Duration calculation fields
    __u64 start_timestamp; // When drm_sched_job_run occurred
    __u64 duration; // Execution duration (0 for start events, set for completion)
};

// Map to store events (unified for HIP and kernel dispatch events)
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

// Stack-based tracking to handle nested function calls
#define MAX_STACK_DEPTH 16

// Map to store call stack depth per thread
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u32);  // tid
    __type(value, __u32);  // depth
    __uint(max_entries, 1024);
} call_stack_depths SEC(".maps");

// Map to store timestamps: key = (tid << 16) | depth
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u64);  // (tid << 16) | depth
    __type(value, __u64);  // timestamp
    __uint(max_entries, 1024 * MAX_STACK_DEPTH);
} call_stack_timestamps SEC(".maps");

// Map to store function IDs: key = (tid << 16) | depth
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u64);  // (tid << 16) | depth
    __type(value, __u32);  // function_id
    __uint(max_entries, 1024 * MAX_STACK_DEPTH);
} call_stack_function_ids SEC(".maps");

// Map to store kernel dispatch start times for duration calculation
struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, 1024);
    __type(key, __u64); // fence_context << 32 | fence_seqno
    __type(value, __u64); // start timestamp
} kernel_dispatch_starts SEC(".maps");

// Structure to store kernel dispatch information
struct kernel_dispatch_info {
    __u64 start_timestamp;
    __u64 end_timestamp;
    __u64 duration;
    char ring_name[32];
    __u32 pid;
    __u32 tid;
};

// Helper functions
static inline void get_pid_tid(__u32 *pid, __u32 *tid) {
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    *pid = (__u32)(pid_tgid >> 32);
    *tid = (__u32)(pid_tgid & 0xFFFFFFFF);
}

static inline __u64 get_timestamp(void) {
    return bpf_ktime_get_ns();
}

static inline __u32 get_next_function_id(void) {
    // Use a simple hash of the current timestamp to generate function IDs
    // This ensures we get unique function IDs without using static variables
    // Limit to reasonable range to avoid array bounds issues
    __u64 timestamp = bpf_ktime_get_ns();
    return (__u32)((timestamp / 1000) % 1000); // Use microseconds and mod 1000
}


// Helper function to create fence key for correlation
static inline __u64 create_fence_key(__u64 context, __u64 seqno) {
    return (context << 32) | (seqno & 0xFFFFFFFF);
}

// Helper function to push entry onto call stack
static inline int push_call_stack(__u32 tid, __u64 timestamp, __u32 function_id) {
    __u32 *depth_ptr = bpf_map_lookup_elem(&call_stack_depths, &tid);
    __u32 depth = 0;

    if (depth_ptr) {
        depth = *depth_ptr;
    }

    // Check for stack overflow
    if (depth >= MAX_STACK_DEPTH) {
        return -1;
    }

    // Create key: (tid << 16) | depth
    __u64 key = ((__u64)tid << 16) | depth;

    // Store timestamp and function_id in separate maps
    bpf_map_update_elem(&call_stack_timestamps, &key, &timestamp, BPF_ANY);
    bpf_map_update_elem(&call_stack_function_ids, &key, &function_id, BPF_ANY);

    // Update depth
    __u32 new_depth = depth + 1;
    bpf_map_update_elem(&call_stack_depths, &tid, &new_depth, BPF_ANY);

    return 0;
}

// Helper function to create exit events
static int create_exit_event(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();

    // Get call stack depth for this thread
    __u32 *depth_ptr = bpf_map_lookup_elem(&call_stack_depths, &tid);
    if (!depth_ptr || *depth_ptr == 0) {
        return 0;
    }

    // Pop from stack
    __u32 depth = *depth_ptr - 1;
    __u64 key = ((__u64)tid << 16) | depth;

    // Get timestamp and function_id from separate maps
    __u64 *entry_time = bpf_map_lookup_elem(&call_stack_timestamps, &key);
    __u32 *function_id = bpf_map_lookup_elem(&call_stack_function_ids, &key);

    if (!entry_time || !function_id) {
        return 0;
    }

    // Update stack depth
    bpf_map_update_elem(&call_stack_depths, &tid, &depth, BPF_ANY);

    // Delete entries
    bpf_map_delete_elem(&call_stack_timestamps, &key);
    bpf_map_delete_elem(&call_stack_function_ids, &key);

    // Create exit event
    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 1;  // Exit
    event->function_id = *function_id;
    // Set function name to empty - user-space will resolve from function_id
    event->function_name[0] = 0;
    event->arg_count = 0;  // Exit events don't capture arguments
    event->return_value = PT_REGS_RC(ctx);

    bpf_ringbuf_submit(event, 0);

    return 0;
}

SEC("uprobe/hip_init_entry")
int hip_init_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(0 / 100 + '0');
    event->function_name[6] = (char)((0 / 10) % 10 + '0');
    event->function_name[7] = (char)(0 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_driver_get_version_entry")
int hip_driver_get_version_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(1 / 100 + '0');
    event->function_name[6] = (char)((1 / 10) % 10 + '0');
    event->function_name[7] = (char)(1 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // driverVersion
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_runtime_get_version_entry")
int hip_runtime_get_version_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(2 / 100 + '0');
    event->function_name[6] = (char)((2 / 10) % 10 + '0');
    event->function_name[7] = (char)(2 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // runtimeVersion
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_entry")
int hip_device_get_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(3 / 100 + '0');
    event->function_name[6] = (char)((3 / 10) % 10 + '0');
    event->function_name[7] = (char)(3 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // device
    event->args[1] = PT_REGS_PARM2(ctx); // ordinal
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_compute_capability_entry")
int hip_device_compute_capability_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(4 / 100 + '0');
    event->function_name[6] = (char)((4 / 10) % 10 + '0');
    event->function_name[7] = (char)(4 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // major
    event->args[1] = PT_REGS_PARM2(ctx); // minor
    event->args[2] = PT_REGS_PARM3(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_name_entry")
int hip_device_get_name_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(5 / 100 + '0');
    event->function_name[6] = (char)((5 / 10) % 10 + '0');
    event->function_name[7] = (char)(5 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // name
    event->args[1] = PT_REGS_PARM2(ctx); // len
    event->args[2] = PT_REGS_PARM3(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_uuid_entry")
int hip_device_get_uuid_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(6 / 100 + '0');
    event->function_name[6] = (char)((6 / 10) % 10 + '0');
    event->function_name[7] = (char)(6 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // uuid
    event->args[1] = PT_REGS_PARM2(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_p2_p_attribute_entry")
int hip_device_get_p2_p_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(7 / 100 + '0');
    event->function_name[6] = (char)((7 / 10) % 10 + '0');
    event->function_name[7] = (char)(7 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // value
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // srcDevice
    event->args[3] = PT_REGS_PARM4(ctx); // dstDevice
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_p_c_i_bus_id_entry")
int hip_device_get_p_c_i_bus_id_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(8 / 100 + '0');
    event->function_name[6] = (char)((8 / 10) % 10 + '0');
    event->function_name[7] = (char)(8 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pciBusId
    event->args[1] = PT_REGS_PARM2(ctx); // len
    event->args[2] = PT_REGS_PARM3(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_by_p_c_i_bus_id_entry")
int hip_device_get_by_p_c_i_bus_id_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(9 / 100 + '0');
    event->function_name[6] = (char)((9 / 10) % 10 + '0');
    event->function_name[7] = (char)(9 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // device
    event->args[1] = PT_REGS_PARM2(ctx); // pciBusId
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_total_mem_entry")
int hip_device_total_mem_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(10 / 100 + '0');
    event->function_name[6] = (char)((10 / 10) % 10 + '0');
    event->function_name[7] = (char)(10 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // bytes
    event->args[1] = PT_REGS_PARM2(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_synchronize_entry")
int hip_device_synchronize_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(11 / 100 + '0');
    event->function_name[6] = (char)((11 / 10) % 10 + '0');
    event->function_name[7] = (char)(11 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // void
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_reset_entry")
int hip_device_reset_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(12 / 100 + '0');
    event->function_name[6] = (char)((12 / 10) % 10 + '0');
    event->function_name[7] = (char)(12 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // void
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_set_device_entry")
int hip_set_device_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(13 / 100 + '0');
    event->function_name[6] = (char)((13 / 10) % 10 + '0');
    event->function_name[7] = (char)(13 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // deviceId
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_set_valid_devices_entry")
int hip_set_valid_devices_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(14 / 100 + '0');
    event->function_name[6] = (char)((14 / 10) % 10 + '0');
    event->function_name[7] = (char)(14 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // device_arr
    event->args[1] = PT_REGS_PARM2(ctx); // len
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_device_entry")
int hip_get_device_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(15 / 100 + '0');
    event->function_name[6] = (char)((15 / 10) % 10 + '0');
    event->function_name[7] = (char)(15 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // deviceId
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_device_count_entry")
int hip_get_device_count_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(16 / 100 + '0');
    event->function_name[6] = (char)((16 / 10) % 10 + '0');
    event->function_name[7] = (char)(16 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_attribute_entry")
int hip_device_get_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(17 / 100 + '0');
    event->function_name[6] = (char)((17 / 10) % 10 + '0');
    event->function_name[7] = (char)(17 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pi
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // deviceId
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_default_mem_pool_entry")
int hip_device_get_default_mem_pool_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(18 / 100 + '0');
    event->function_name[6] = (char)((18 / 10) % 10 + '0');
    event->function_name[7] = (char)(18 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // mem_pool
    event->args[1] = PT_REGS_PARM2(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_set_mem_pool_entry")
int hip_device_set_mem_pool_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(19 / 100 + '0');
    event->function_name[6] = (char)((19 / 10) % 10 + '0');
    event->function_name[7] = (char)(19 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // device
    event->args[1] = PT_REGS_PARM2(ctx); // mem_pool
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_mem_pool_entry")
int hip_device_get_mem_pool_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(20 / 100 + '0');
    event->function_name[6] = (char)((20 / 10) % 10 + '0');
    event->function_name[7] = (char)(20 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // mem_pool
    event->args[1] = PT_REGS_PARM2(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_device_properties_entry")
int hip_get_device_properties_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(21 / 100 + '0');
    event->function_name[6] = (char)((21 / 10) % 10 + '0');
    event->function_name[7] = (char)(21 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // prop
    event->args[1] = PT_REGS_PARM2(ctx); // deviceId
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_texture1_d_linear_max_width_entry")
int hip_device_get_texture1_d_linear_max_width_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(22 / 100 + '0');
    event->function_name[6] = (char)((22 / 10) % 10 + '0');
    event->function_name[7] = (char)(22 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // max_width
    event->args[1] = PT_REGS_PARM2(ctx); // desc
    event->args[2] = PT_REGS_PARM3(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_set_cache_config_entry")
int hip_device_set_cache_config_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(23 / 100 + '0');
    event->function_name[6] = (char)((23 / 10) % 10 + '0');
    event->function_name[7] = (char)(23 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // cacheConfig
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_cache_config_entry")
int hip_device_get_cache_config_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(24 / 100 + '0');
    event->function_name[6] = (char)((24 / 10) % 10 + '0');
    event->function_name[7] = (char)(24 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // cacheConfig
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_limit_entry")
int hip_device_get_limit_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(25 / 100 + '0');
    event->function_name[6] = (char)((25 / 10) % 10 + '0');
    event->function_name[7] = (char)(25 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pValue
    event->args[1] = PT_REGS_PARM2(ctx); // limit
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_set_limit_entry")
int hip_device_set_limit_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(26 / 100 + '0');
    event->function_name[6] = (char)((26 / 10) % 10 + '0');
    event->function_name[7] = (char)(26 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // limit
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_shared_mem_config_entry")
int hip_device_get_shared_mem_config_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(27 / 100 + '0');
    event->function_name[6] = (char)((27 / 10) % 10 + '0');
    event->function_name[7] = (char)(27 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // pConfig
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_device_flags_entry")
int hip_get_device_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(28 / 100 + '0');
    event->function_name[6] = (char)((28 / 10) % 10 + '0');
    event->function_name[7] = (char)(28 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_set_shared_mem_config_entry")
int hip_device_set_shared_mem_config_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(29 / 100 + '0');
    event->function_name[6] = (char)((29 / 10) % 10 + '0');
    event->function_name[7] = (char)(29 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // config
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_set_device_flags_entry")
int hip_set_device_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(30 / 100 + '0');
    event->function_name[6] = (char)((30 / 10) % 10 + '0');
    event->function_name[7] = (char)(30 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_choose_device_entry")
int hip_choose_device_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(31 / 100 + '0');
    event->function_name[6] = (char)((31 / 10) % 10 + '0');
    event->function_name[7] = (char)(31 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // device
    event->args[1] = PT_REGS_PARM2(ctx); // prop
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ipc_get_mem_handle_entry")
int hip_ipc_get_mem_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(32 / 100 + '0');
    event->function_name[6] = (char)((32 / 10) % 10 + '0');
    event->function_name[7] = (char)(32 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // handle
    event->args[1] = PT_REGS_PARM2(ctx); // devPtr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ipc_open_mem_handle_entry")
int hip_ipc_open_mem_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(33 / 100 + '0');
    event->function_name[6] = (char)((33 / 10) % 10 + '0');
    event->function_name[7] = (char)(33 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // devPtr
    event->args[1] = PT_REGS_PARM2(ctx); // handle
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ipc_close_mem_handle_entry")
int hip_ipc_close_mem_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(34 / 100 + '0');
    event->function_name[6] = (char)((34 / 10) % 10 + '0');
    event->function_name[7] = (char)(34 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // devPtr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ipc_get_event_handle_entry")
int hip_ipc_get_event_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(35 / 100 + '0');
    event->function_name[6] = (char)((35 / 10) % 10 + '0');
    event->function_name[7] = (char)(35 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // handle
    event->args[1] = PT_REGS_PARM2(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ipc_open_event_handle_entry")
int hip_ipc_open_event_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(36 / 100 + '0');
    event->function_name[6] = (char)((36 / 10) % 10 + '0');
    event->function_name[7] = (char)(36 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // event
    event->args[1] = PT_REGS_PARM2(ctx); // handle
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_func_set_attribute_entry")
int hip_func_set_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(37 / 100 + '0');
    event->function_name[6] = (char)((37 / 10) % 10 + '0');
    event->function_name[7] = (char)(37 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // func
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_func_set_cache_config_entry")
int hip_func_set_cache_config_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(38 / 100 + '0');
    event->function_name[6] = (char)((38 / 10) % 10 + '0');
    event->function_name[7] = (char)(38 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // func
    event->args[1] = PT_REGS_PARM2(ctx); // config
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_func_set_shared_mem_config_entry")
int hip_func_set_shared_mem_config_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(39 / 100 + '0');
    event->function_name[6] = (char)((39 / 10) % 10 + '0');
    event->function_name[7] = (char)(39 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // func
    event->args[1] = PT_REGS_PARM2(ctx); // config
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_last_error_entry")
int hip_get_last_error_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'h';
    event->function_name[1] = 'i';
    event->function_name[2] = 'p';
    event->function_name[3] = 'G';
    event->function_name[4] = 'e';
    event->function_name[5] = 't';
    event->function_name[6] = 'L';
    event->function_name[7] = 'a';
    event->function_name[8] = 's';
    event->function_name[9] = 't';
    event->function_name[10] = 'E';
    event->function_name[11] = 'r';
    event->function_name[12] = 'r';
    event->function_name[13] = 'o';
    event->function_name[14] = 'r';
    event->function_name[15] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // void
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_peek_at_last_error_entry")
int hip_peek_at_last_error_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(41 / 100 + '0');
    event->function_name[6] = (char)((41 / 10) % 10 + '0');
    event->function_name[7] = (char)(41 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // void
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_create_entry")
int hip_stream_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(42 / 100 + '0');
    event->function_name[6] = (char)((42 / 10) % 10 + '0');
    event->function_name[7] = (char)(42 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_create_with_flags_entry")
int hip_stream_create_with_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(43 / 100 + '0');
    event->function_name[6] = (char)((43 / 10) % 10 + '0');
    event->function_name[7] = (char)(43 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_create_with_priority_entry")
int hip_stream_create_with_priority_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(44 / 100 + '0');
    event->function_name[6] = (char)((44 / 10) % 10 + '0');
    event->function_name[7] = (char)(44 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->args[2] = PT_REGS_PARM3(ctx); // priority
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_stream_priority_range_entry")
int hip_device_get_stream_priority_range_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(45 / 100 + '0');
    event->function_name[6] = (char)((45 / 10) % 10 + '0');
    event->function_name[7] = (char)(45 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // leastPriority
    event->args[1] = PT_REGS_PARM2(ctx); // greatestPriority
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_destroy_entry")
int hip_stream_destroy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(46 / 100 + '0');
    event->function_name[6] = (char)((46 / 10) % 10 + '0');
    event->function_name[7] = (char)(46 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_query_entry")
int hip_stream_query_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(47 / 100 + '0');
    event->function_name[6] = (char)((47 / 10) % 10 + '0');
    event->function_name[7] = (char)(47 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_synchronize_entry")
int hip_stream_synchronize_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(48 / 100 + '0');
    event->function_name[6] = (char)((48 / 10) % 10 + '0');
    event->function_name[7] = (char)(48 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_wait_event_entry")
int hip_stream_wait_event_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(49 / 100 + '0');
    event->function_name[6] = (char)((49 / 10) % 10 + '0');
    event->function_name[7] = (char)(49 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // event
    event->args[2] = PT_REGS_PARM3(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_get_flags_entry")
int hip_stream_get_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(50 / 100 + '0');
    event->function_name[6] = (char)((50 / 10) % 10 + '0');
    event->function_name[7] = (char)(50 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_get_id_entry")
int hip_stream_get_id_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(51 / 100 + '0');
    event->function_name[6] = (char)((51 / 10) % 10 + '0');
    event->function_name[7] = (char)(51 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // streamId
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_get_priority_entry")
int hip_stream_get_priority_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(52 / 100 + '0');
    event->function_name[6] = (char)((52 / 10) % 10 + '0');
    event->function_name[7] = (char)(52 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // priority
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_get_device_entry")
int hip_stream_get_device_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(53 / 100 + '0');
    event->function_name[6] = (char)((53 / 10) % 10 + '0');
    event->function_name[7] = (char)(53 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ext_stream_create_with_c_u_mask_entry")
int hip_ext_stream_create_with_c_u_mask_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(54 / 100 + '0');
    event->function_name[6] = (char)((54 / 10) % 10 + '0');
    event->function_name[7] = (char)(54 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // cuMaskSize
    event->args[2] = PT_REGS_PARM3(ctx); // cuMask
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ext_stream_get_c_u_mask_entry")
int hip_ext_stream_get_c_u_mask_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(55 / 100 + '0');
    event->function_name[6] = (char)((55 / 10) % 10 + '0');
    event->function_name[7] = (char)(55 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // cuMaskSize
    event->args[2] = PT_REGS_PARM3(ctx); // cuMask
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_add_callback_entry")
int hip_stream_add_callback_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(56 / 100 + '0');
    event->function_name[6] = (char)((56 / 10) % 10 + '0');
    event->function_name[7] = (char)(56 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // callback
    event->args[2] = PT_REGS_PARM3(ctx); // userData
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_set_attribute_entry")
int hip_stream_set_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(57 / 100 + '0');
    event->function_name[6] = (char)((57 / 10) % 10 + '0');
    event->function_name[7] = (char)(57 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_get_attribute_entry")
int hip_stream_get_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(58 / 100 + '0');
    event->function_name[6] = (char)((58 / 10) % 10 + '0');
    event->function_name[7] = (char)(58 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // value_out
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_wait_value32_entry")
int hip_stream_wait_value32_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(59 / 100 + '0');
    event->function_name[6] = (char)((59 / 10) % 10 + '0');
    event->function_name[7] = (char)(59 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // ptr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(0xFFFFFFFF)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_wait_value64_entry")
int hip_stream_wait_value64_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(60 / 100 + '0');
    event->function_name[6] = (char)((60 / 10) % 10 + '0');
    event->function_name[7] = (char)(60 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // ptr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(0xFFFFFFFFFFFFFFFF)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_write_value32_entry")
int hip_stream_write_value32_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(61 / 100 + '0');
    event->function_name[6] = (char)((61 / 10) % 10 + '0');
    event->function_name[7] = (char)(61 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // ptr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_write_value64_entry")
int hip_stream_write_value64_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(62 / 100 + '0');
    event->function_name[6] = (char)((62 / 10) % 10 + '0');
    event->function_name[7] = (char)(62 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // ptr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_batch_mem_op_entry")
int hip_stream_batch_mem_op_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(63 / 100 + '0');
    event->function_name[6] = (char)((63 / 10) % 10 + '0');
    event->function_name[7] = (char)(63 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // count
    event->args[2] = PT_REGS_PARM3(ctx); // paramArray
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_batch_mem_op_node_entry")
int hip_graph_add_batch_mem_op_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(64 / 100 + '0');
    event->function_name[6] = (char)((64 / 10) % 10 + '0');
    event->function_name[7] = (char)(64 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // phGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // hGraph
    event->args[2] = PT_REGS_PARM3(ctx); // dependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_batch_mem_op_node_get_params_entry")
int hip_graph_batch_mem_op_node_get_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(65 / 100 + '0');
    event->function_name[6] = (char)((65 / 10) % 10 + '0');
    event->function_name[7] = (char)(65 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // hNode
    event->args[1] = PT_REGS_PARM2(ctx); // nodeParams_out
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_batch_mem_op_node_set_params_entry")
int hip_graph_batch_mem_op_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(66 / 100 + '0');
    event->function_name[6] = (char)((66 / 10) % 10 + '0');
    event->function_name[7] = (char)(66 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // hNode
    event->args[1] = PT_REGS_PARM2(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_batch_mem_op_node_set_params_entry")
int hip_graph_exec_batch_mem_op_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(67 / 100 + '0');
    event->function_name[6] = (char)((67 / 10) % 10 + '0');
    event->function_name[7] = (char)(67 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // hNode
    event->args[2] = PT_REGS_PARM3(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_event_create_with_flags_entry")
int hip_event_create_with_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(68 / 100 + '0');
    event->function_name[6] = (char)((68 / 10) % 10 + '0');
    event->function_name[7] = (char)(68 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // event
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_event_create_entry")
int hip_event_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(69 / 100 + '0');
    event->function_name[6] = (char)((69 / 10) % 10 + '0');
    event->function_name[7] = (char)(69 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_event_record_with_flags_entry")
int hip_event_record_with_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(70 / 100 + '0');
    event->function_name[6] = (char)((70 / 10) % 10 + '0');
    event->function_name[7] = (char)(70 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // event
    event->args[1] = PT_REGS_PARM2(ctx); // __dparm(0)
    event->args[2] = PT_REGS_PARM3(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_event_record_entry")
int hip_event_record_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(71 / 100 + '0');
    event->function_name[6] = (char)((71 / 10) % 10 + '0');
    event->function_name[7] = (char)(71 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // event
    event->args[1] = PT_REGS_PARM2(ctx); // NULL
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_event_destroy_entry")
int hip_event_destroy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(72 / 100 + '0');
    event->function_name[6] = (char)((72 / 10) % 10 + '0');
    event->function_name[7] = (char)(72 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_event_synchronize_entry")
int hip_event_synchronize_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(73 / 100 + '0');
    event->function_name[6] = (char)((73 / 10) % 10 + '0');
    event->function_name[7] = (char)(73 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_event_elapsed_time_entry")
int hip_event_elapsed_time_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(74 / 100 + '0');
    event->function_name[6] = (char)((74 / 10) % 10 + '0');
    event->function_name[7] = (char)(74 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // ms
    event->args[1] = PT_REGS_PARM2(ctx); // start
    event->args[2] = PT_REGS_PARM3(ctx); // stop
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_event_query_entry")
int hip_event_query_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(75 / 100 + '0');
    event->function_name[6] = (char)((75 / 10) % 10 + '0');
    event->function_name[7] = (char)(75 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_pointer_set_attribute_entry")
int hip_pointer_set_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(76 / 100 + '0');
    event->function_name[6] = (char)((76 / 10) % 10 + '0');
    event->function_name[7] = (char)(76 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // value
    event->args[1] = PT_REGS_PARM2(ctx); // attribute
    event->args[2] = PT_REGS_PARM3(ctx); // ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_pointer_get_attributes_entry")
int hip_pointer_get_attributes_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(77 / 100 + '0');
    event->function_name[6] = (char)((77 / 10) % 10 + '0');
    event->function_name[7] = (char)(77 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // attributes
    event->args[1] = PT_REGS_PARM2(ctx); // ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_pointer_get_attribute_entry")
int hip_pointer_get_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(78 / 100 + '0');
    event->function_name[6] = (char)((78 / 10) % 10 + '0');
    event->function_name[7] = (char)(78 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // data
    event->args[1] = PT_REGS_PARM2(ctx); // attribute
    event->args[2] = PT_REGS_PARM3(ctx); // ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_pointer_get_attributes_entry")
int hip_drv_pointer_get_attributes_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(79 / 100 + '0');
    event->function_name[6] = (char)((79 / 10) % 10 + '0');
    event->function_name[7] = (char)(79 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // numAttributes
    event->args[1] = PT_REGS_PARM2(ctx); // attributes
    event->args[2] = PT_REGS_PARM3(ctx); // data
    event->args[3] = PT_REGS_PARM4(ctx); // ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_import_external_semaphore_entry")
int hip_import_external_semaphore_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(80 / 100 + '0');
    event->function_name[6] = (char)((80 / 10) % 10 + '0');
    event->function_name[7] = (char)(80 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // extSem_out
    event->args[1] = PT_REGS_PARM2(ctx); // semHandleDesc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_signal_external_semaphores_async_entry")
int hip_signal_external_semaphores_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(81 / 100 + '0');
    event->function_name[6] = (char)((81 / 10) % 10 + '0');
    event->function_name[7] = (char)(81 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // extSemArray
    event->args[1] = PT_REGS_PARM2(ctx); // paramsArray
    event->args[2] = PT_REGS_PARM3(ctx); // numExtSems
    event->args[3] = PT_REGS_PARM4(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_wait_external_semaphores_async_entry")
int hip_wait_external_semaphores_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(82 / 100 + '0');
    event->function_name[6] = (char)((82 / 10) % 10 + '0');
    event->function_name[7] = (char)(82 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // extSemArray
    event->args[1] = PT_REGS_PARM2(ctx); // paramsArray
    event->args[2] = PT_REGS_PARM3(ctx); // numExtSems
    event->args[3] = PT_REGS_PARM4(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_destroy_external_semaphore_entry")
int hip_destroy_external_semaphore_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(83 / 100 + '0');
    event->function_name[6] = (char)((83 / 10) % 10 + '0');
    event->function_name[7] = (char)(83 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // extSem
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_import_external_memory_entry")
int hip_import_external_memory_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(84 / 100 + '0');
    event->function_name[6] = (char)((84 / 10) % 10 + '0');
    event->function_name[7] = (char)(84 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // extMem_out
    event->args[1] = PT_REGS_PARM2(ctx); // memHandleDesc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_external_memory_get_mapped_buffer_entry")
int hip_external_memory_get_mapped_buffer_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(85 / 100 + '0');
    event->function_name[6] = (char)((85 / 10) % 10 + '0');
    event->function_name[7] = (char)(85 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // devPtr
    event->args[1] = PT_REGS_PARM2(ctx); // extMem
    event->args[2] = PT_REGS_PARM3(ctx); // bufferDesc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_destroy_external_memory_entry")
int hip_destroy_external_memory_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(86 / 100 + '0');
    event->function_name[6] = (char)((86 / 10) % 10 + '0');
    event->function_name[7] = (char)(86 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // extMem
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_external_memory_get_mapped_mipmapped_array_entry")
int hip_external_memory_get_mapped_mipmapped_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(87 / 100 + '0');
    event->function_name[6] = (char)((87 / 10) % 10 + '0');
    event->function_name[7] = (char)(87 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // mipmap
    event->args[1] = PT_REGS_PARM2(ctx); // extMem
    event->args[2] = PT_REGS_PARM3(ctx); // mipmapDesc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_malloc_entry")
int hip_malloc_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(88 / 100 + '0');
    event->function_name[6] = (char)((88 / 10) % 10 + '0');
    event->function_name[7] = (char)(88 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ext_malloc_with_flags_entry")
int hip_ext_malloc_with_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(89 / 100 + '0');
    event->function_name[6] = (char)((89 / 10) % 10 + '0');
    event->function_name[7] = (char)(89 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // sizeBytes
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_malloc_host_entry")
int hip_malloc_host_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(90 / 100 + '0');
    event->function_name[6] = (char)((90 / 10) % 10 + '0');
    event->function_name[7] = (char)(90 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_alloc_host_entry")
int hip_mem_alloc_host_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(91 / 100 + '0');
    event->function_name[6] = (char)((91 / 10) % 10 + '0');
    event->function_name[7] = (char)(91 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_host_malloc_entry")
int hip_host_malloc_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(92 / 100 + '0');
    event->function_name[6] = (char)((92 / 10) % 10 + '0');
    event->function_name[7] = (char)(92 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_malloc_managed_entry")
int hip_malloc_managed_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(93 / 100 + '0');
    event->function_name[6] = (char)((93 / 10) % 10 + '0');
    event->function_name[7] = (char)(93 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dev_ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // __dparm(hipMemAttachGlobal)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_prefetch_async_entry")
int hip_mem_prefetch_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(94 / 100 + '0');
    event->function_name[6] = (char)((94 / 10) % 10 + '0');
    event->function_name[7] = (char)(94 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dev_ptr
    event->args[1] = PT_REGS_PARM2(ctx); // count
    event->args[2] = PT_REGS_PARM3(ctx); // device
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_prefetch_async_v2_entry")
int hip_mem_prefetch_async_v2_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(95 / 100 + '0');
    event->function_name[6] = (char)((95 / 10) % 10 + '0');
    event->function_name[7] = (char)(95 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dev_ptr
    event->args[1] = PT_REGS_PARM2(ctx); // count
    event->args[2] = PT_REGS_PARM3(ctx); // location
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_advise_entry")
int hip_mem_advise_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(96 / 100 + '0');
    event->function_name[6] = (char)((96 / 10) % 10 + '0');
    event->function_name[7] = (char)(96 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dev_ptr
    event->args[1] = PT_REGS_PARM2(ctx); // count
    event->args[2] = PT_REGS_PARM3(ctx); // advice
    event->args[3] = PT_REGS_PARM4(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_advise_v2_entry")
int hip_mem_advise_v2_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(97 / 100 + '0');
    event->function_name[6] = (char)((97 / 10) % 10 + '0');
    event->function_name[7] = (char)(97 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dev_ptr
    event->args[1] = PT_REGS_PARM2(ctx); // count
    event->args[2] = PT_REGS_PARM3(ctx); // advice
    event->args[3] = PT_REGS_PARM4(ctx); // location
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_range_get_attribute_entry")
int hip_mem_range_get_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(98 / 100 + '0');
    event->function_name[6] = (char)((98 / 10) % 10 + '0');
    event->function_name[7] = (char)(98 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // data
    event->args[1] = PT_REGS_PARM2(ctx); // data_size
    event->args[2] = PT_REGS_PARM3(ctx); // attribute
    event->args[3] = PT_REGS_PARM4(ctx); // dev_ptr
    event->args[4] = PT_REGS_PARM5(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_range_get_attributes_entry")
int hip_mem_range_get_attributes_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(99 / 100 + '0');
    event->function_name[6] = (char)((99 / 10) % 10 + '0');
    event->function_name[7] = (char)(99 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // data
    event->args[1] = PT_REGS_PARM2(ctx); // data_sizes
    event->args[2] = PT_REGS_PARM3(ctx); // attributes
    event->args[3] = PT_REGS_PARM4(ctx); // num_attributes
    event->args[4] = PT_REGS_PARM5(ctx); // dev_ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_attach_mem_async_entry")
int hip_stream_attach_mem_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(100 / 100 + '0');
    event->function_name[6] = (char)((100 / 10) % 10 + '0');
    event->function_name[7] = (char)(100 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // dev_ptr
    event->args[2] = PT_REGS_PARM3(ctx); // __dparm(0)
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(hipMemAttachSingle)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_malloc_async_entry")
int hip_malloc_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(101 / 100 + '0');
    event->function_name[6] = (char)((101 / 10) % 10 + '0');
    event->function_name[7] = (char)(101 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dev_ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_free_async_entry")
int hip_free_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(102 / 100 + '0');
    event->function_name[6] = (char)((102 / 10) % 10 + '0');
    event->function_name[7] = (char)(102 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // dev_ptr
    event->args[1] = PT_REGS_PARM2(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_trim_to_entry")
int hip_mem_pool_trim_to_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(103 / 100 + '0');
    event->function_name[6] = (char)((103 / 10) % 10 + '0');
    event->function_name[7] = (char)(103 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // mem_pool
    event->args[1] = PT_REGS_PARM2(ctx); // min_bytes_to_hold
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_set_attribute_entry")
int hip_mem_pool_set_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(104 / 100 + '0');
    event->function_name[6] = (char)((104 / 10) % 10 + '0');
    event->function_name[7] = (char)(104 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // mem_pool
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_get_attribute_entry")
int hip_mem_pool_get_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(105 / 100 + '0');
    event->function_name[6] = (char)((105 / 10) % 10 + '0');
    event->function_name[7] = (char)(105 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // mem_pool
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_set_access_entry")
int hip_mem_pool_set_access_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(106 / 100 + '0');
    event->function_name[6] = (char)((106 / 10) % 10 + '0');
    event->function_name[7] = (char)(106 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // mem_pool
    event->args[1] = PT_REGS_PARM2(ctx); // desc_list
    event->args[2] = PT_REGS_PARM3(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_get_access_entry")
int hip_mem_pool_get_access_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(107 / 100 + '0');
    event->function_name[6] = (char)((107 / 10) % 10 + '0');
    event->function_name[7] = (char)(107 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // flags
    event->args[1] = PT_REGS_PARM2(ctx); // mem_pool
    event->args[2] = PT_REGS_PARM3(ctx); // location
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_create_entry")
int hip_mem_pool_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(108 / 100 + '0');
    event->function_name[6] = (char)((108 / 10) % 10 + '0');
    event->function_name[7] = (char)(108 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // mem_pool
    event->args[1] = PT_REGS_PARM2(ctx); // pool_props
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_destroy_entry")
int hip_mem_pool_destroy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(109 / 100 + '0');
    event->function_name[6] = (char)((109 / 10) % 10 + '0');
    event->function_name[7] = (char)(109 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // mem_pool
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_malloc_from_pool_async_entry")
int hip_malloc_from_pool_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(110 / 100 + '0');
    event->function_name[6] = (char)((110 / 10) % 10 + '0');
    event->function_name[7] = (char)(110 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dev_ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // mem_pool
    event->args[3] = PT_REGS_PARM4(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_export_to_shareable_handle_entry")
int hip_mem_pool_export_to_shareable_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(111 / 100 + '0');
    event->function_name[6] = (char)((111 / 10) % 10 + '0');
    event->function_name[7] = (char)(111 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // shared_handle
    event->args[1] = PT_REGS_PARM2(ctx); // mem_pool
    event->args[2] = PT_REGS_PARM3(ctx); // handle_type
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_import_from_shareable_handle_entry")
int hip_mem_pool_import_from_shareable_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(112 / 100 + '0');
    event->function_name[6] = (char)((112 / 10) % 10 + '0');
    event->function_name[7] = (char)(112 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // mem_pool
    event->args[1] = PT_REGS_PARM2(ctx); // shared_handle
    event->args[2] = PT_REGS_PARM3(ctx); // handle_type
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_export_pointer_entry")
int hip_mem_pool_export_pointer_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(113 / 100 + '0');
    event->function_name[6] = (char)((113 / 10) % 10 + '0');
    event->function_name[7] = (char)(113 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // export_data
    event->args[1] = PT_REGS_PARM2(ctx); // dev_ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_pool_import_pointer_entry")
int hip_mem_pool_import_pointer_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(114 / 100 + '0');
    event->function_name[6] = (char)((114 / 10) % 10 + '0');
    event->function_name[7] = (char)(114 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dev_ptr
    event->args[1] = PT_REGS_PARM2(ctx); // mem_pool
    event->args[2] = PT_REGS_PARM3(ctx); // export_data
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_host_alloc_entry")
int hip_host_alloc_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(115 / 100 + '0');
    event->function_name[6] = (char)((115 / 10) % 10 + '0');
    event->function_name[7] = (char)(115 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_host_get_device_pointer_entry")
int hip_host_get_device_pointer_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(116 / 100 + '0');
    event->function_name[6] = (char)((116 / 10) % 10 + '0');
    event->function_name[7] = (char)(116 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // devPtr
    event->args[1] = PT_REGS_PARM2(ctx); // hstPtr
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_host_get_flags_entry")
int hip_host_get_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(117 / 100 + '0');
    event->function_name[6] = (char)((117 / 10) % 10 + '0');
    event->function_name[7] = (char)(117 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // flagsPtr
    event->args[1] = PT_REGS_PARM2(ctx); // hostPtr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_host_register_entry")
int hip_host_register_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(118 / 100 + '0');
    event->function_name[6] = (char)((118 / 10) % 10 + '0');
    event->function_name[7] = (char)(118 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hostPtr
    event->args[1] = PT_REGS_PARM2(ctx); // sizeBytes
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_host_unregister_entry")
int hip_host_unregister_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(119 / 100 + '0');
    event->function_name[6] = (char)((119 / 10) % 10 + '0');
    event->function_name[7] = (char)(119 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // hostPtr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_malloc_pitch_entry")
int hip_malloc_pitch_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(120 / 100 + '0');
    event->function_name[6] = (char)((120 / 10) % 10 + '0');
    event->function_name[7] = (char)(120 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // pitch
    event->args[2] = PT_REGS_PARM3(ctx); // width
    event->args[3] = PT_REGS_PARM4(ctx); // height
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_alloc_pitch_entry")
int hip_mem_alloc_pitch_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(121 / 100 + '0');
    event->function_name[6] = (char)((121 / 10) % 10 + '0');
    event->function_name[7] = (char)(121 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dptr
    event->args[1] = PT_REGS_PARM2(ctx); // pitch
    event->args[2] = PT_REGS_PARM3(ctx); // widthInBytes
    event->args[3] = PT_REGS_PARM4(ctx); // height
    event->args[4] = PT_REGS_PARM5(ctx); // elementSizeBytes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_free_entry")
int hip_free_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(122 / 100 + '0');
    event->function_name[6] = (char)((122 / 10) % 10 + '0');
    event->function_name[7] = (char)(122 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_free_host_entry")
int hip_free_host_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(123 / 100 + '0');
    event->function_name[6] = (char)((123 / 10) % 10 + '0');
    event->function_name[7] = (char)(123 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_host_free_entry")
int hip_host_free_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(124 / 100 + '0');
    event->function_name[6] = (char)((124 / 10) % 10 + '0');
    event->function_name[7] = (char)(124 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_entry")
int hip_memcpy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(125 / 100 + '0');
    event->function_name[6] = (char)((125 / 10) % 10 + '0');
    event->function_name[7] = (char)(125 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // kind
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_with_stream_entry")
int hip_memcpy_with_stream_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(126 / 100 + '0');
    event->function_name[6] = (char)((126 / 10) % 10 + '0');
    event->function_name[7] = (char)(126 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // kind
    event->args[4] = PT_REGS_PARM5(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_hto_d_entry")
int hip_memcpy_hto_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(127 / 100 + '0');
    event->function_name[6] = (char)((127 / 10) % 10 + '0');
    event->function_name[7] = (char)(127 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_dto_h_entry")
int hip_memcpy_dto_h_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(128 / 100 + '0');
    event->function_name[6] = (char)((128 / 10) % 10 + '0');
    event->function_name[7] = (char)(128 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_dto_d_entry")
int hip_memcpy_dto_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(129 / 100 + '0');
    event->function_name[6] = (char)((129 / 10) % 10 + '0');
    event->function_name[7] = (char)(129 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_ato_d_entry")
int hip_memcpy_ato_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(130 / 100 + '0');
    event->function_name[6] = (char)((130 / 10) % 10 + '0');
    event->function_name[7] = (char)(130 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dstDevice
    event->args[1] = PT_REGS_PARM2(ctx); // srcArray
    event->args[2] = PT_REGS_PARM3(ctx); // srcOffset
    event->args[3] = PT_REGS_PARM4(ctx); // ByteCount
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_dto_a_entry")
int hip_memcpy_dto_a_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(131 / 100 + '0');
    event->function_name[6] = (char)((131 / 10) % 10 + '0');
    event->function_name[7] = (char)(131 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dstArray
    event->args[1] = PT_REGS_PARM2(ctx); // dstOffset
    event->args[2] = PT_REGS_PARM3(ctx); // srcDevice
    event->args[3] = PT_REGS_PARM4(ctx); // ByteCount
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_ato_a_entry")
int hip_memcpy_ato_a_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(132 / 100 + '0');
    event->function_name[6] = (char)((132 / 10) % 10 + '0');
    event->function_name[7] = (char)(132 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dstArray
    event->args[1] = PT_REGS_PARM2(ctx); // dstOffset
    event->args[2] = PT_REGS_PARM3(ctx); // srcArray
    event->args[3] = PT_REGS_PARM4(ctx); // srcOffset
    event->args[4] = PT_REGS_PARM5(ctx); // ByteCount
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_hto_d_async_entry")
int hip_memcpy_hto_d_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(133 / 100 + '0');
    event->function_name[6] = (char)((133 / 10) % 10 + '0');
    event->function_name[7] = (char)(133 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_dto_h_async_entry")
int hip_memcpy_dto_h_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(134 / 100 + '0');
    event->function_name[6] = (char)((134 / 10) % 10 + '0');
    event->function_name[7] = (char)(134 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_dto_d_async_entry")
int hip_memcpy_dto_d_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(135 / 100 + '0');
    event->function_name[6] = (char)((135 / 10) % 10 + '0');
    event->function_name[7] = (char)(135 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_ato_h_async_entry")
int hip_memcpy_ato_h_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(136 / 100 + '0');
    event->function_name[6] = (char)((136 / 10) % 10 + '0');
    event->function_name[7] = (char)(136 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dstHost
    event->args[1] = PT_REGS_PARM2(ctx); // srcArray
    event->args[2] = PT_REGS_PARM3(ctx); // srcOffset
    event->args[3] = PT_REGS_PARM4(ctx); // ByteCount
    event->args[4] = PT_REGS_PARM5(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_hto_a_async_entry")
int hip_memcpy_hto_a_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(137 / 100 + '0');
    event->function_name[6] = (char)((137 / 10) % 10 + '0');
    event->function_name[7] = (char)(137 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dstArray
    event->args[1] = PT_REGS_PARM2(ctx); // dstOffset
    event->args[2] = PT_REGS_PARM3(ctx); // srcHost
    event->args[3] = PT_REGS_PARM4(ctx); // ByteCount
    event->args[4] = PT_REGS_PARM5(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_get_global_entry")
int hip_module_get_global_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(138 / 100 + '0');
    event->function_name[6] = (char)((138 / 10) % 10 + '0');
    event->function_name[7] = (char)(138 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dptr
    event->args[1] = PT_REGS_PARM2(ctx); // bytes
    event->args[2] = PT_REGS_PARM3(ctx); // hmod
    event->args[3] = PT_REGS_PARM4(ctx); // name
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_symbol_address_entry")
int hip_get_symbol_address_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(139 / 100 + '0');
    event->function_name[6] = (char)((139 / 10) % 10 + '0');
    event->function_name[7] = (char)(139 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // devPtr
    event->args[1] = PT_REGS_PARM2(ctx); // symbol
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_symbol_size_entry")
int hip_get_symbol_size_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(140 / 100 + '0');
    event->function_name[6] = (char)((140 / 10) % 10 + '0');
    event->function_name[7] = (char)(140 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // size
    event->args[1] = PT_REGS_PARM2(ctx); // symbol
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_proc_address_entry")
int hip_get_proc_address_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(141 / 100 + '0');
    event->function_name[6] = (char)((141 / 10) % 10 + '0');
    event->function_name[7] = (char)(141 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // symbol
    event->args[1] = PT_REGS_PARM2(ctx); // pfn
    event->args[2] = PT_REGS_PARM3(ctx); // hipVersion
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->args[4] = PT_REGS_PARM5(ctx); // symbolStatus
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_to_symbol_entry")
int hip_memcpy_to_symbol_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(142 / 100 + '0');
    event->function_name[6] = (char)((142 / 10) % 10 + '0');
    event->function_name[7] = (char)(142 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // symbol
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(hipMemcpyHostToDevice)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_to_symbol_async_entry")
int hip_memcpy_to_symbol_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(143 / 100 + '0');
    event->function_name[6] = (char)((143 / 10) % 10 + '0');
    event->function_name[7] = (char)(143 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // symbol
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // offset
    event->args[4] = PT_REGS_PARM5(ctx); // kind
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_from_symbol_entry")
int hip_memcpy_from_symbol_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(144 / 100 + '0');
    event->function_name[6] = (char)((144 / 10) % 10 + '0');
    event->function_name[7] = (char)(144 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // symbol
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(hipMemcpyDeviceToHost)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_from_symbol_async_entry")
int hip_memcpy_from_symbol_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(145 / 100 + '0');
    event->function_name[6] = (char)((145 / 10) % 10 + '0');
    event->function_name[7] = (char)(145 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // symbol
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // offset
    event->args[4] = PT_REGS_PARM5(ctx); // kind
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_async_entry")
int hip_memcpy_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(146 / 100 + '0');
    event->function_name[6] = (char)((146 / 10) % 10 + '0');
    event->function_name[7] = (char)(146 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // src
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // kind
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_entry")
int hip_memset_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(147 / 100 + '0');
    event->function_name[6] = (char)((147 / 10) % 10 + '0');
    event->function_name[7] = (char)(147 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d8_entry")
int hip_memset_d8_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(148 / 100 + '0');
    event->function_name[6] = (char)((148 / 10) % 10 + '0');
    event->function_name[7] = (char)(148 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dest
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->args[2] = PT_REGS_PARM3(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d8_async_entry")
int hip_memset_d8_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(149 / 100 + '0');
    event->function_name[6] = (char)((149 / 10) % 10 + '0');
    event->function_name[7] = (char)(149 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dest
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->args[2] = PT_REGS_PARM3(ctx); // count
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d16_entry")
int hip_memset_d16_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(150 / 100 + '0');
    event->function_name[6] = (char)((150 / 10) % 10 + '0');
    event->function_name[7] = (char)(150 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dest
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->args[2] = PT_REGS_PARM3(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d16_async_entry")
int hip_memset_d16_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(151 / 100 + '0');
    event->function_name[6] = (char)((151 / 10) % 10 + '0');
    event->function_name[7] = (char)(151 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dest
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->args[2] = PT_REGS_PARM3(ctx); // count
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d32_entry")
int hip_memset_d32_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(152 / 100 + '0');
    event->function_name[6] = (char)((152 / 10) % 10 + '0');
    event->function_name[7] = (char)(152 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dest
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->args[2] = PT_REGS_PARM3(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_async_entry")
int hip_memset_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(153 / 100 + '0');
    event->function_name[6] = (char)((153 / 10) % 10 + '0');
    event->function_name[7] = (char)(153 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->args[2] = PT_REGS_PARM3(ctx); // sizeBytes
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d32_async_entry")
int hip_memset_d32_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(154 / 100 + '0');
    event->function_name[6] = (char)((154 / 10) % 10 + '0');
    event->function_name[7] = (char)(154 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->args[2] = PT_REGS_PARM3(ctx); // count
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset2_d_entry")
int hip_memset2_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(155 / 100 + '0');
    event->function_name[6] = (char)((155 / 10) % 10 + '0');
    event->function_name[7] = (char)(155 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // pitch
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // width
    event->args[4] = PT_REGS_PARM5(ctx); // height
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset2_d_async_entry")
int hip_memset2_d_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(156 / 100 + '0');
    event->function_name[6] = (char)((156 / 10) % 10 + '0');
    event->function_name[7] = (char)(156 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // pitch
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // width
    event->args[4] = PT_REGS_PARM5(ctx); // height
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset3_d_entry")
int hip_memset3_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(157 / 100 + '0');
    event->function_name[6] = (char)((157 / 10) % 10 + '0');
    event->function_name[7] = (char)(157 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pitchedDevPtr
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->args[2] = PT_REGS_PARM3(ctx); // extent
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset3_d_async_entry")
int hip_memset3_d_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(158 / 100 + '0');
    event->function_name[6] = (char)((158 / 10) % 10 + '0');
    event->function_name[7] = (char)(158 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // pitchedDevPtr
    event->args[1] = PT_REGS_PARM2(ctx); // value
    event->args[2] = PT_REGS_PARM3(ctx); // extent
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d2_d8_entry")
int hip_memset_d2_d8_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(159 / 100 + '0');
    event->function_name[6] = (char)((159 / 10) % 10 + '0');
    event->function_name[7] = (char)(159 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dstPitch
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // width
    event->args[4] = PT_REGS_PARM5(ctx); // height
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d2_d8_async_entry")
int hip_memset_d2_d8_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(160 / 100 + '0');
    event->function_name[6] = (char)((160 / 10) % 10 + '0');
    event->function_name[7] = (char)(160 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dstPitch
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // width
    event->args[4] = PT_REGS_PARM5(ctx); // height
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d2_d16_entry")
int hip_memset_d2_d16_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(161 / 100 + '0');
    event->function_name[6] = (char)((161 / 10) % 10 + '0');
    event->function_name[7] = (char)(161 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dstPitch
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // width
    event->args[4] = PT_REGS_PARM5(ctx); // height
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d2_d16_async_entry")
int hip_memset_d2_d16_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(162 / 100 + '0');
    event->function_name[6] = (char)((162 / 10) % 10 + '0');
    event->function_name[7] = (char)(162 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dstPitch
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // width
    event->args[4] = PT_REGS_PARM5(ctx); // height
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d2_d32_entry")
int hip_memset_d2_d32_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(163 / 100 + '0');
    event->function_name[6] = (char)((163 / 10) % 10 + '0');
    event->function_name[7] = (char)(163 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dstPitch
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // width
    event->args[4] = PT_REGS_PARM5(ctx); // height
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memset_d2_d32_async_entry")
int hip_memset_d2_d32_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(164 / 100 + '0');
    event->function_name[6] = (char)((164 / 10) % 10 + '0');
    event->function_name[7] = (char)(164 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dstPitch
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->args[3] = PT_REGS_PARM4(ctx); // width
    event->args[4] = PT_REGS_PARM5(ctx); // height
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_get_info_entry")
int hip_mem_get_info_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(165 / 100 + '0');
    event->function_name[6] = (char)((165 / 10) % 10 + '0');
    event->function_name[7] = (char)(165 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // free
    event->args[1] = PT_REGS_PARM2(ctx); // total
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_ptr_get_info_entry")
int hip_mem_ptr_get_info_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(166 / 100 + '0');
    event->function_name[6] = (char)((166 / 10) % 10 + '0');
    event->function_name[7] = (char)(166 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_malloc_array_entry")
int hip_malloc_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(167 / 100 + '0');
    event->function_name[6] = (char)((167 / 10) % 10 + '0');
    event->function_name[7] = (char)(167 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // array
    event->args[1] = PT_REGS_PARM2(ctx); // desc
    event->args[2] = PT_REGS_PARM3(ctx); // width
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(hipArrayDefault)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_array_create_entry")
int hip_array_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(168 / 100 + '0');
    event->function_name[6] = (char)((168 / 10) % 10 + '0');
    event->function_name[7] = (char)(168 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pHandle
    event->args[1] = PT_REGS_PARM2(ctx); // pAllocateArray
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_array_destroy_entry")
int hip_array_destroy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(169 / 100 + '0');
    event->function_name[6] = (char)((169 / 10) % 10 + '0');
    event->function_name[7] = (char)(169 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // array
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_array3_d_create_entry")
int hip_array3_d_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(170 / 100 + '0');
    event->function_name[6] = (char)((170 / 10) % 10 + '0');
    event->function_name[7] = (char)(170 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // array
    event->args[1] = PT_REGS_PARM2(ctx); // pAllocateArray
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_malloc3_d_entry")
int hip_malloc3_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(171 / 100 + '0');
    event->function_name[6] = (char)((171 / 10) % 10 + '0');
    event->function_name[7] = (char)(171 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pitchedDevPtr
    event->args[1] = PT_REGS_PARM2(ctx); // extent
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_free_array_entry")
int hip_free_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(172 / 100 + '0');
    event->function_name[6] = (char)((172 / 10) % 10 + '0');
    event->function_name[7] = (char)(172 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // array
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_malloc3_d_array_entry")
int hip_malloc3_d_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(173 / 100 + '0');
    event->function_name[6] = (char)((173 / 10) % 10 + '0');
    event->function_name[7] = (char)(173 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // array
    event->args[1] = PT_REGS_PARM2(ctx); // desc
    event->args[2] = PT_REGS_PARM3(ctx); // extent
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_array_get_info_entry")
int hip_array_get_info_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(174 / 100 + '0');
    event->function_name[6] = (char)((174 / 10) % 10 + '0');
    event->function_name[7] = (char)(174 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // desc
    event->args[1] = PT_REGS_PARM2(ctx); // extent
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->args[3] = PT_REGS_PARM4(ctx); // array
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_array_get_descriptor_entry")
int hip_array_get_descriptor_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(175 / 100 + '0');
    event->function_name[6] = (char)((175 / 10) % 10 + '0');
    event->function_name[7] = (char)(175 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pArrayDescriptor
    event->args[1] = PT_REGS_PARM2(ctx); // array
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_array3_d_get_descriptor_entry")
int hip_array3_d_get_descriptor_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(176 / 100 + '0');
    event->function_name[6] = (char)((176 / 10) % 10 + '0');
    event->function_name[7] = (char)(176 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pArrayDescriptor
    event->args[1] = PT_REGS_PARM2(ctx); // array
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy2_d_entry")
int hip_memcpy2_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(177 / 100 + '0');
    event->function_name[6] = (char)((177 / 10) % 10 + '0');
    event->function_name[7] = (char)(177 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 7;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dpitch
    event->args[2] = PT_REGS_PARM3(ctx); // src
    event->args[3] = PT_REGS_PARM4(ctx); // spitch
    event->args[4] = PT_REGS_PARM5(ctx); // width
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_param2_d_entry")
int hip_memcpy_param2_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(178 / 100 + '0');
    event->function_name[6] = (char)((178 / 10) % 10 + '0');
    event->function_name[7] = (char)(178 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // pCopy
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_param2_d_async_entry")
int hip_memcpy_param2_d_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(179 / 100 + '0');
    event->function_name[6] = (char)((179 / 10) % 10 + '0');
    event->function_name[7] = (char)(179 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pCopy
    event->args[1] = PT_REGS_PARM2(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy2_d_async_entry")
int hip_memcpy2_d_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(180 / 100 + '0');
    event->function_name[6] = (char)((180 / 10) % 10 + '0');
    event->function_name[7] = (char)(180 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 8;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dpitch
    event->args[2] = PT_REGS_PARM3(ctx); // src
    event->args[3] = PT_REGS_PARM4(ctx); // spitch
    event->args[4] = PT_REGS_PARM5(ctx); // width
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy2_d_to_array_entry")
int hip_memcpy2_d_to_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(181 / 100 + '0');
    event->function_name[6] = (char)((181 / 10) % 10 + '0');
    event->function_name[7] = (char)(181 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 8;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // wOffset
    event->args[2] = PT_REGS_PARM3(ctx); // hOffset
    event->args[3] = PT_REGS_PARM4(ctx); // src
    event->args[4] = PT_REGS_PARM5(ctx); // spitch
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy2_d_to_array_async_entry")
int hip_memcpy2_d_to_array_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(182 / 100 + '0');
    event->function_name[6] = (char)((182 / 10) % 10 + '0');
    event->function_name[7] = (char)(182 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 9;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // wOffset
    event->args[2] = PT_REGS_PARM3(ctx); // hOffset
    event->args[3] = PT_REGS_PARM4(ctx); // src
    event->args[4] = PT_REGS_PARM5(ctx); // spitch
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy2_d_array_to_array_entry")
int hip_memcpy2_d_array_to_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(183 / 100 + '0');
    event->function_name[6] = (char)((183 / 10) % 10 + '0');
    event->function_name[7] = (char)(183 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 9;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // wOffsetDst
    event->args[2] = PT_REGS_PARM3(ctx); // hOffsetDst
    event->args[3] = PT_REGS_PARM4(ctx); // src
    event->args[4] = PT_REGS_PARM5(ctx); // wOffsetSrc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_to_array_entry")
int hip_memcpy_to_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(184 / 100 + '0');
    event->function_name[6] = (char)((184 / 10) % 10 + '0');
    event->function_name[7] = (char)(184 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // wOffset
    event->args[2] = PT_REGS_PARM3(ctx); // hOffset
    event->args[3] = PT_REGS_PARM4(ctx); // src
    event->args[4] = PT_REGS_PARM5(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_from_array_entry")
int hip_memcpy_from_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(185 / 100 + '0');
    event->function_name[6] = (char)((185 / 10) % 10 + '0');
    event->function_name[7] = (char)(185 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // srcArray
    event->args[2] = PT_REGS_PARM3(ctx); // wOffset
    event->args[3] = PT_REGS_PARM4(ctx); // hOffset
    event->args[4] = PT_REGS_PARM5(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy2_d_from_array_entry")
int hip_memcpy2_d_from_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(186 / 100 + '0');
    event->function_name[6] = (char)((186 / 10) % 10 + '0');
    event->function_name[7] = (char)(186 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 8;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dpitch
    event->args[2] = PT_REGS_PARM3(ctx); // src
    event->args[3] = PT_REGS_PARM4(ctx); // wOffset
    event->args[4] = PT_REGS_PARM5(ctx); // hOffset
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy2_d_from_array_async_entry")
int hip_memcpy2_d_from_array_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(187 / 100 + '0');
    event->function_name[6] = (char)((187 / 10) % 10 + '0');
    event->function_name[7] = (char)(187 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 9;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dpitch
    event->args[2] = PT_REGS_PARM3(ctx); // src
    event->args[3] = PT_REGS_PARM4(ctx); // wOffset
    event->args[4] = PT_REGS_PARM5(ctx); // hOffset
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_ato_h_entry")
int hip_memcpy_ato_h_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(188 / 100 + '0');
    event->function_name[6] = (char)((188 / 10) % 10 + '0');
    event->function_name[7] = (char)(188 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // srcArray
    event->args[2] = PT_REGS_PARM3(ctx); // srcOffset
    event->args[3] = PT_REGS_PARM4(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_hto_a_entry")
int hip_memcpy_hto_a_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(189 / 100 + '0');
    event->function_name[6] = (char)((189 / 10) % 10 + '0');
    event->function_name[7] = (char)(189 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // dstArray
    event->args[1] = PT_REGS_PARM2(ctx); // dstOffset
    event->args[2] = PT_REGS_PARM3(ctx); // srcHost
    event->args[3] = PT_REGS_PARM4(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy3_d_entry")
int hip_memcpy3_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(190 / 100 + '0');
    event->function_name[6] = (char)((190 / 10) % 10 + '0');
    event->function_name[7] = (char)(190 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // p
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy3_d_async_entry")
int hip_memcpy3_d_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(191 / 100 + '0');
    event->function_name[6] = (char)((191 / 10) % 10 + '0');
    event->function_name[7] = (char)(191 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // p
    event->args[1] = PT_REGS_PARM2(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_memcpy3_d_entry")
int hip_drv_memcpy3_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(192 / 100 + '0');
    event->function_name[6] = (char)((192 / 10) % 10 + '0');
    event->function_name[7] = (char)(192 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // pCopy
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_memcpy3_d_async_entry")
int hip_drv_memcpy3_d_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(193 / 100 + '0');
    event->function_name[6] = (char)((193 / 10) % 10 + '0');
    event->function_name[7] = (char)(193 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pCopy
    event->args[1] = PT_REGS_PARM2(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_get_address_range_entry")
int hip_mem_get_address_range_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(194 / 100 + '0');
    event->function_name[6] = (char)((194 / 10) % 10 + '0');
    event->function_name[7] = (char)(194 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pbase
    event->args[1] = PT_REGS_PARM2(ctx); // psize
    event->args[2] = PT_REGS_PARM3(ctx); // dptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_batch_async_entry")
int hip_memcpy_batch_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(195 / 100 + '0');
    event->function_name[6] = (char)((195 / 10) % 10 + '0');
    event->function_name[7] = (char)(195 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 9;
    event->args[0] = PT_REGS_PARM1(ctx); // dsts
    event->args[1] = PT_REGS_PARM2(ctx); // srcs
    event->args[2] = PT_REGS_PARM3(ctx); // sizes
    event->args[3] = PT_REGS_PARM4(ctx); // count
    event->args[4] = PT_REGS_PARM5(ctx); // attrs
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy3_d_batch_async_entry")
int hip_memcpy3_d_batch_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(196 / 100 + '0');
    event->function_name[6] = (char)((196 / 10) % 10 + '0');
    event->function_name[7] = (char)(196 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // numOps
    event->args[1] = PT_REGS_PARM2(ctx); // opList
    event->args[2] = PT_REGS_PARM3(ctx); // failIdx
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy3_d_peer_entry")
int hip_memcpy3_d_peer_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(197 / 100 + '0');
    event->function_name[6] = (char)((197 / 10) % 10 + '0');
    event->function_name[7] = (char)(197 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // p
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy3_d_peer_async_entry")
int hip_memcpy3_d_peer_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(198 / 100 + '0');
    event->function_name[6] = (char)((198 / 10) % 10 + '0');
    event->function_name[7] = (char)(198 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // p
    event->args[1] = PT_REGS_PARM2(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_can_access_peer_entry")
int hip_device_can_access_peer_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(199 / 100 + '0');
    event->function_name[6] = (char)((199 / 10) % 10 + '0');
    event->function_name[7] = (char)(199 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // canAccessPeer
    event->args[1] = PT_REGS_PARM2(ctx); // deviceId
    event->args[2] = PT_REGS_PARM3(ctx); // peerDeviceId
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_enable_peer_access_entry")
int hip_device_enable_peer_access_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(200 / 100 + '0');
    event->function_name[6] = (char)((200 / 10) % 10 + '0');
    event->function_name[7] = (char)(200 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // peerDeviceId
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_disable_peer_access_entry")
int hip_device_disable_peer_access_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(201 / 100 + '0');
    event->function_name[6] = (char)((201 / 10) % 10 + '0');
    event->function_name[7] = (char)(201 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // peerDeviceId
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_peer_entry")
int hip_memcpy_peer_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(202 / 100 + '0');
    event->function_name[6] = (char)((202 / 10) % 10 + '0');
    event->function_name[7] = (char)(202 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dstDeviceId
    event->args[2] = PT_REGS_PARM3(ctx); // src
    event->args[3] = PT_REGS_PARM4(ctx); // srcDeviceId
    event->args[4] = PT_REGS_PARM5(ctx); // sizeBytes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_memcpy_peer_async_entry")
int hip_memcpy_peer_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(203 / 100 + '0');
    event->function_name[6] = (char)((203 / 10) % 10 + '0');
    event->function_name[7] = (char)(203 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // dst
    event->args[1] = PT_REGS_PARM2(ctx); // dstDeviceId
    event->args[2] = PT_REGS_PARM3(ctx); // src
    event->args[3] = PT_REGS_PARM4(ctx); // srcDevice
    event->args[4] = PT_REGS_PARM5(ctx); // sizeBytes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_create_entry")
int hip_ctx_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(204 / 100 + '0');
    event->function_name[6] = (char)((204 / 10) % 10 + '0');
    event->function_name[7] = (char)(204 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // ctx
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->args[2] = PT_REGS_PARM3(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_destroy_entry")
int hip_ctx_destroy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(205 / 100 + '0');
    event->function_name[6] = (char)((205 / 10) % 10 + '0');
    event->function_name[7] = (char)(205 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // ctx
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_pop_current_entry")
int hip_ctx_pop_current_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(206 / 100 + '0');
    event->function_name[6] = (char)((206 / 10) % 10 + '0');
    event->function_name[7] = (char)(206 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // ctx
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_push_current_entry")
int hip_ctx_push_current_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(207 / 100 + '0');
    event->function_name[6] = (char)((207 / 10) % 10 + '0');
    event->function_name[7] = (char)(207 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // ctx
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_set_current_entry")
int hip_ctx_set_current_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(208 / 100 + '0');
    event->function_name[6] = (char)((208 / 10) % 10 + '0');
    event->function_name[7] = (char)(208 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // ctx
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_get_current_entry")
int hip_ctx_get_current_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(209 / 100 + '0');
    event->function_name[6] = (char)((209 / 10) % 10 + '0');
    event->function_name[7] = (char)(209 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // ctx
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_get_device_entry")
int hip_ctx_get_device_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(210 / 100 + '0');
    event->function_name[6] = (char)((210 / 10) % 10 + '0');
    event->function_name[7] = (char)(210 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_get_api_version_entry")
int hip_ctx_get_api_version_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(211 / 100 + '0');
    event->function_name[6] = (char)((211 / 10) % 10 + '0');
    event->function_name[7] = (char)(211 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // ctx
    event->args[1] = PT_REGS_PARM2(ctx); // apiVersion
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_get_cache_config_entry")
int hip_ctx_get_cache_config_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(212 / 100 + '0');
    event->function_name[6] = (char)((212 / 10) % 10 + '0');
    event->function_name[7] = (char)(212 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // cacheConfig
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_set_cache_config_entry")
int hip_ctx_set_cache_config_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(213 / 100 + '0');
    event->function_name[6] = (char)((213 / 10) % 10 + '0');
    event->function_name[7] = (char)(213 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // cacheConfig
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_set_shared_mem_config_entry")
int hip_ctx_set_shared_mem_config_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(214 / 100 + '0');
    event->function_name[6] = (char)((214 / 10) % 10 + '0');
    event->function_name[7] = (char)(214 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // config
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_get_shared_mem_config_entry")
int hip_ctx_get_shared_mem_config_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(215 / 100 + '0');
    event->function_name[6] = (char)((215 / 10) % 10 + '0');
    event->function_name[7] = (char)(215 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // pConfig
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_synchronize_entry")
int hip_ctx_synchronize_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(216 / 100 + '0');
    event->function_name[6] = (char)((216 / 10) % 10 + '0');
    event->function_name[7] = (char)(216 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // void
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_get_flags_entry")
int hip_ctx_get_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(217 / 100 + '0');
    event->function_name[6] = (char)((217 / 10) % 10 + '0');
    event->function_name[7] = (char)(217 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_enable_peer_access_entry")
int hip_ctx_enable_peer_access_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(218 / 100 + '0');
    event->function_name[6] = (char)((218 / 10) % 10 + '0');
    event->function_name[7] = (char)(218 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // peerCtx
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ctx_disable_peer_access_entry")
int hip_ctx_disable_peer_access_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(219 / 100 + '0');
    event->function_name[6] = (char)((219 / 10) % 10 + '0');
    event->function_name[7] = (char)(219 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // peerCtx
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_primary_ctx_get_state_entry")
int hip_device_primary_ctx_get_state_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(220 / 100 + '0');
    event->function_name[6] = (char)((220 / 10) % 10 + '0');
    event->function_name[7] = (char)(220 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // dev
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->args[2] = PT_REGS_PARM3(ctx); // active
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_primary_ctx_release_entry")
int hip_device_primary_ctx_release_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(221 / 100 + '0');
    event->function_name[6] = (char)((221 / 10) % 10 + '0');
    event->function_name[7] = (char)(221 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // dev
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_primary_ctx_retain_entry")
int hip_device_primary_ctx_retain_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(222 / 100 + '0');
    event->function_name[6] = (char)((222 / 10) % 10 + '0');
    event->function_name[7] = (char)(222 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pctx
    event->args[1] = PT_REGS_PARM2(ctx); // dev
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_primary_ctx_reset_entry")
int hip_device_primary_ctx_reset_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(223 / 100 + '0');
    event->function_name[6] = (char)((223 / 10) % 10 + '0');
    event->function_name[7] = (char)(223 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // dev
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_primary_ctx_set_flags_entry")
int hip_device_primary_ctx_set_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(224 / 100 + '0');
    event->function_name[6] = (char)((224 / 10) % 10 + '0');
    event->function_name[7] = (char)(224 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // dev
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_load_fat_binary_entry")
int hip_module_load_fat_binary_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(225 / 100 + '0');
    event->function_name[6] = (char)((225 / 10) % 10 + '0');
    event->function_name[7] = (char)(225 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // module
    event->args[1] = PT_REGS_PARM2(ctx); // fatbin
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_load_entry")
int hip_module_load_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(226 / 100 + '0');
    event->function_name[6] = (char)((226 / 10) % 10 + '0');
    event->function_name[7] = (char)(226 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // module
    event->args[1] = PT_REGS_PARM2(ctx); // fname
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_unload_entry")
int hip_module_unload_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(227 / 100 + '0');
    event->function_name[6] = (char)((227 / 10) % 10 + '0');
    event->function_name[7] = (char)(227 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // module
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_get_function_entry")
int hip_module_get_function_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(228 / 100 + '0');
    event->function_name[6] = (char)((228 / 10) % 10 + '0');
    event->function_name[7] = (char)(228 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // function
    event->args[1] = PT_REGS_PARM2(ctx); // module
    event->args[2] = PT_REGS_PARM3(ctx); // kname
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_get_function_count_entry")
int hip_module_get_function_count_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(229 / 100 + '0');
    event->function_name[6] = (char)((229 / 10) % 10 + '0');
    event->function_name[7] = (char)(229 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // count
    event->args[1] = PT_REGS_PARM2(ctx); // mod
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_func_get_attributes_entry")
int hip_func_get_attributes_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(230 / 100 + '0');
    event->function_name[6] = (char)((230 / 10) % 10 + '0');
    event->function_name[7] = (char)(230 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // attr
    event->args[1] = PT_REGS_PARM2(ctx); // func
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_func_get_attribute_entry")
int hip_func_get_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(231 / 100 + '0');
    event->function_name[6] = (char)((231 / 10) % 10 + '0');
    event->function_name[7] = (char)(231 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // value
    event->args[1] = PT_REGS_PARM2(ctx); // attrib
    event->args[2] = PT_REGS_PARM3(ctx); // hfunc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_func_by_symbol_entry")
int hip_get_func_by_symbol_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(232 / 100 + '0');
    event->function_name[6] = (char)((232 / 10) % 10 + '0');
    event->function_name[7] = (char)(232 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // functionPtr
    event->args[1] = PT_REGS_PARM2(ctx); // symbolPtr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_driver_entry_point_entry")
int hip_get_driver_entry_point_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(233 / 100 + '0');
    event->function_name[6] = (char)((233 / 10) % 10 + '0');
    event->function_name[7] = (char)(233 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // symbol
    event->args[1] = PT_REGS_PARM2(ctx); // funcPtr
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->args[3] = PT_REGS_PARM4(ctx); // driverStatus
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_get_tex_ref_entry")
int hip_module_get_tex_ref_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(234 / 100 + '0');
    event->function_name[6] = (char)((234 / 10) % 10 + '0');
    event->function_name[7] = (char)(234 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // hmod
    event->args[2] = PT_REGS_PARM3(ctx); // name
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_load_data_entry")
int hip_module_load_data_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(235 / 100 + '0');
    event->function_name[6] = (char)((235 / 10) % 10 + '0');
    event->function_name[7] = (char)(235 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // module
    event->args[1] = PT_REGS_PARM2(ctx); // image
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_load_data_ex_entry")
int hip_module_load_data_ex_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(236 / 100 + '0');
    event->function_name[6] = (char)((236 / 10) % 10 + '0');
    event->function_name[7] = (char)(236 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // module
    event->args[1] = PT_REGS_PARM2(ctx); // image
    event->args[2] = PT_REGS_PARM3(ctx); // numOptions
    event->args[3] = PT_REGS_PARM4(ctx); // options
    event->args[4] = PT_REGS_PARM5(ctx); // optionValues
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_link_add_data_entry")
int hip_link_add_data_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(237 / 100 + '0');
    event->function_name[6] = (char)((237 / 10) % 10 + '0');
    event->function_name[7] = (char)(237 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 8;
    event->args[0] = PT_REGS_PARM1(ctx); // state
    event->args[1] = PT_REGS_PARM2(ctx); // type
    event->args[2] = PT_REGS_PARM3(ctx); // data
    event->args[3] = PT_REGS_PARM4(ctx); // size
    event->args[4] = PT_REGS_PARM5(ctx); // name
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_link_add_file_entry")
int hip_link_add_file_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(238 / 100 + '0');
    event->function_name[6] = (char)((238 / 10) % 10 + '0');
    event->function_name[7] = (char)(238 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // state
    event->args[1] = PT_REGS_PARM2(ctx); // type
    event->args[2] = PT_REGS_PARM3(ctx); // path
    event->args[3] = PT_REGS_PARM4(ctx); // numOptions
    event->args[4] = PT_REGS_PARM5(ctx); // options
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_link_complete_entry")
int hip_link_complete_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(239 / 100 + '0');
    event->function_name[6] = (char)((239 / 10) % 10 + '0');
    event->function_name[7] = (char)(239 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // state
    event->args[1] = PT_REGS_PARM2(ctx); // hipBinOut
    event->args[2] = PT_REGS_PARM3(ctx); // sizeOut
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_link_create_entry")
int hip_link_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(240 / 100 + '0');
    event->function_name[6] = (char)((240 / 10) % 10 + '0');
    event->function_name[7] = (char)(240 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // numOptions
    event->args[1] = PT_REGS_PARM2(ctx); // options
    event->args[2] = PT_REGS_PARM3(ctx); // optionValues
    event->args[3] = PT_REGS_PARM4(ctx); // stateOut
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_link_destroy_entry")
int hip_link_destroy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(241 / 100 + '0');
    event->function_name[6] = (char)((241 / 10) % 10 + '0');
    event->function_name[7] = (char)(241 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // state
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_launch_kernel_entry")
int hip_module_launch_kernel_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(242 / 100 + '0');
    event->function_name[6] = (char)((242 / 10) % 10 + '0');
    event->function_name[7] = (char)(242 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 11;
    event->args[0] = PT_REGS_PARM1(ctx); // f
    event->args[1] = PT_REGS_PARM2(ctx); // gridDimX
    event->args[2] = PT_REGS_PARM3(ctx); // gridDimY
    event->args[3] = PT_REGS_PARM4(ctx); // gridDimZ
    event->args[4] = PT_REGS_PARM5(ctx); // blockDimX
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_launch_cooperative_kernel_entry")
int hip_module_launch_cooperative_kernel_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(243 / 100 + '0');
    event->function_name[6] = (char)((243 / 10) % 10 + '0');
    event->function_name[7] = (char)(243 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 10;
    event->args[0] = PT_REGS_PARM1(ctx); // f
    event->args[1] = PT_REGS_PARM2(ctx); // gridDimX
    event->args[2] = PT_REGS_PARM3(ctx); // gridDimY
    event->args[3] = PT_REGS_PARM4(ctx); // gridDimZ
    event->args[4] = PT_REGS_PARM5(ctx); // blockDimX
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_launch_cooperative_kernel_multi_device_entry")
int hip_module_launch_cooperative_kernel_multi_device_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(244 / 100 + '0');
    event->function_name[6] = (char)((244 / 10) % 10 + '0');
    event->function_name[7] = (char)(244 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // launchParamsList
    event->args[1] = PT_REGS_PARM2(ctx); // numDevices
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_launch_cooperative_kernel_entry")
int hip_launch_cooperative_kernel_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(245 / 100 + '0');
    event->function_name[6] = (char)((245 / 10) % 10 + '0');
    event->function_name[7] = (char)(245 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // f
    event->args[1] = PT_REGS_PARM2(ctx); // gridDim
    event->args[2] = PT_REGS_PARM3(ctx); // blockDimX
    event->args[3] = PT_REGS_PARM4(ctx); // kernelParams
    event->args[4] = PT_REGS_PARM5(ctx); // sharedMemBytes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_launch_cooperative_kernel_multi_device_entry")
int hip_launch_cooperative_kernel_multi_device_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(246 / 100 + '0');
    event->function_name[6] = (char)((246 / 10) % 10 + '0');
    event->function_name[7] = (char)(246 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // launchParamsList
    event->args[1] = PT_REGS_PARM2(ctx); // numDevices
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ext_launch_multi_kernel_multi_device_entry")
int hip_ext_launch_multi_kernel_multi_device_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(247 / 100 + '0');
    event->function_name[6] = (char)((247 / 10) % 10 + '0');
    event->function_name[7] = (char)(247 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // launchParamsList
    event->args[1] = PT_REGS_PARM2(ctx); // numDevices
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_launch_kernel_ex_c_entry")
int hip_launch_kernel_ex_c_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(248 / 100 + '0');
    event->function_name[6] = (char)((248 / 10) % 10 + '0');
    event->function_name[7] = (char)(248 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // config
    event->args[1] = PT_REGS_PARM2(ctx); // fPtr
    event->args[2] = PT_REGS_PARM3(ctx); // args
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_launch_kernel_ex_entry")
int hip_drv_launch_kernel_ex_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(249 / 100 + '0');
    event->function_name[6] = (char)((249 / 10) % 10 + '0');
    event->function_name[7] = (char)(249 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // config
    event->args[1] = PT_REGS_PARM2(ctx); // f
    event->args[2] = PT_REGS_PARM3(ctx); // params
    event->args[3] = PT_REGS_PARM4(ctx); // extra
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_get_handle_for_address_range_entry")
int hip_mem_get_handle_for_address_range_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(250 / 100 + '0');
    event->function_name[6] = (char)((250 / 10) % 10 + '0');
    event->function_name[7] = (char)(250 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // handle
    event->args[1] = PT_REGS_PARM2(ctx); // dptr
    event->args[2] = PT_REGS_PARM3(ctx); // size
    event->args[3] = PT_REGS_PARM4(ctx); // handleType
    event->args[4] = PT_REGS_PARM5(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_occupancy_max_potential_block_size_entry")
int hip_module_occupancy_max_potential_block_size_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(251 / 100 + '0');
    event->function_name[6] = (char)((251 / 10) % 10 + '0');
    event->function_name[7] = (char)(251 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // gridSize
    event->args[1] = PT_REGS_PARM2(ctx); // blockSize
    event->args[2] = PT_REGS_PARM3(ctx); // f
    event->args[3] = PT_REGS_PARM4(ctx); // dynSharedMemPerBlk
    event->args[4] = PT_REGS_PARM5(ctx); // blockSizeLimit
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_occupancy_max_potential_block_size_with_flags_entry")
int hip_module_occupancy_max_potential_block_size_with_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(252 / 100 + '0');
    event->function_name[6] = (char)((252 / 10) % 10 + '0');
    event->function_name[7] = (char)(252 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // gridSize
    event->args[1] = PT_REGS_PARM2(ctx); // blockSize
    event->args[2] = PT_REGS_PARM3(ctx); // f
    event->args[3] = PT_REGS_PARM4(ctx); // dynSharedMemPerBlk
    event->args[4] = PT_REGS_PARM5(ctx); // blockSizeLimit
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_occupancy_max_active_blocks_per_multiprocessor_entry")
int hip_module_occupancy_max_active_blocks_per_multiprocessor_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(253 / 100 + '0');
    event->function_name[6] = (char)((253 / 10) % 10 + '0');
    event->function_name[7] = (char)(253 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // numBlocks
    event->args[1] = PT_REGS_PARM2(ctx); // f
    event->args[2] = PT_REGS_PARM3(ctx); // blockSize
    event->args[3] = PT_REGS_PARM4(ctx); // dynSharedMemPerBlk
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_module_occupancy_max_active_blocks_per_multiprocessor_with_flags_entry")
int hip_module_occupancy_max_active_blocks_per_multiprocessor_with_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(254 / 100 + '0');
    event->function_name[6] = (char)((254 / 10) % 10 + '0');
    event->function_name[7] = (char)(254 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // numBlocks
    event->args[1] = PT_REGS_PARM2(ctx); // f
    event->args[2] = PT_REGS_PARM3(ctx); // blockSize
    event->args[3] = PT_REGS_PARM4(ctx); // dynSharedMemPerBlk
    event->args[4] = PT_REGS_PARM5(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_occupancy_max_active_blocks_per_multiprocessor_entry")
int hip_occupancy_max_active_blocks_per_multiprocessor_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(255 / 100 + '0');
    event->function_name[6] = (char)((255 / 10) % 10 + '0');
    event->function_name[7] = (char)(255 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // numBlocks
    event->args[1] = PT_REGS_PARM2(ctx); // f
    event->args[2] = PT_REGS_PARM3(ctx); // blockSize
    event->args[3] = PT_REGS_PARM4(ctx); // dynSharedMemPerBlk
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_occupancy_max_active_blocks_per_multiprocessor_with_flags_entry")
int hip_occupancy_max_active_blocks_per_multiprocessor_with_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(256 / 100 + '0');
    event->function_name[6] = (char)((256 / 10) % 10 + '0');
    event->function_name[7] = (char)(256 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // numBlocks
    event->args[1] = PT_REGS_PARM2(ctx); // f
    event->args[2] = PT_REGS_PARM3(ctx); // blockSize
    event->args[3] = PT_REGS_PARM4(ctx); // dynSharedMemPerBlk
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(hipOccupancyDefault)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_occupancy_max_potential_block_size_entry")
int hip_occupancy_max_potential_block_size_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(257 / 100 + '0');
    event->function_name[6] = (char)((257 / 10) % 10 + '0');
    event->function_name[7] = (char)(257 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // gridSize
    event->args[1] = PT_REGS_PARM2(ctx); // blockSize
    event->args[2] = PT_REGS_PARM3(ctx); // f
    event->args[3] = PT_REGS_PARM4(ctx); // dynSharedMemPerBlk
    event->args[4] = PT_REGS_PARM5(ctx); // blockSizeLimit
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_profiler_start_entry")
int hip_profiler_start_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(258 / 100 + '0');
    event->function_name[6] = (char)((258 / 10) % 10 + '0');
    event->function_name[7] = (char)(258 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 0;
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_profiler_stop_entry")
int hip_profiler_stop_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(259 / 100 + '0');
    event->function_name[6] = (char)((259 / 10) % 10 + '0');
    event->function_name[7] = (char)(259 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 0;
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_configure_call_entry")
int hip_configure_call_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(260 / 100 + '0');
    event->function_name[6] = (char)((260 / 10) % 10 + '0');
    event->function_name[7] = (char)(260 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // gridDim
    event->args[1] = PT_REGS_PARM2(ctx); // blockDim
    event->args[2] = PT_REGS_PARM3(ctx); // __dparm(0)
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_setup_argument_entry")
int hip_setup_argument_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(261 / 100 + '0');
    event->function_name[6] = (char)((261 / 10) % 10 + '0');
    event->function_name[7] = (char)(261 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // arg
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // offset
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_launch_by_ptr_entry")
int hip_launch_by_ptr_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(262 / 100 + '0');
    event->function_name[6] = (char)((262 / 10) % 10 + '0');
    event->function_name[7] = (char)(262 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // func
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_launch_kernel_entry")
int hip_launch_kernel_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'h';
    event->function_name[1] = 'i';
    event->function_name[2] = 'p';
    event->function_name[3] = 'L';
    event->function_name[4] = 'a';
    event->function_name[5] = 'u';
    event->function_name[6] = 'n';
    event->function_name[7] = 'c';
    event->function_name[8] = 'h';
    event->function_name[9] = 'K';
    event->function_name[10] = 'e';
    event->function_name[11] = 'r';
    event->function_name[12] = 'n';
    event->function_name[13] = 'e';
    event->function_name[14] = 'l';
    event->function_name[15] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // function_address
    event->args[1] = PT_REGS_PARM2(ctx); // numBlocks
    event->args[2] = PT_REGS_PARM3(ctx); // dimBlocks
    event->args[3] = PT_REGS_PARM4(ctx); // args
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_launch_host_func_entry")
int hip_launch_host_func_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(264 / 100 + '0');
    event->function_name[6] = (char)((264 / 10) % 10 + '0');
    event->function_name[7] = (char)(264 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // fn
    event->args[2] = PT_REGS_PARM3(ctx); // userData
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_memcpy2_d_unaligned_entry")
int hip_drv_memcpy2_d_unaligned_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(265 / 100 + '0');
    event->function_name[6] = (char)((265 / 10) % 10 + '0');
    event->function_name[7] = (char)(265 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // pCopy
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_ext_launch_kernel_entry")
int hip_ext_launch_kernel_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(266 / 100 + '0');
    event->function_name[6] = (char)((266 / 10) % 10 + '0');
    event->function_name[7] = (char)(266 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 9;
    event->args[0] = PT_REGS_PARM1(ctx); // function_address
    event->args[1] = PT_REGS_PARM2(ctx); // numBlocks
    event->args[2] = PT_REGS_PARM3(ctx); // dimBlocks
    event->args[3] = PT_REGS_PARM4(ctx); // args
    event->args[4] = PT_REGS_PARM5(ctx); // sharedMemBytes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_create_texture_object_entry")
int hip_create_texture_object_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(267 / 100 + '0');
    event->function_name[6] = (char)((267 / 10) % 10 + '0');
    event->function_name[7] = (char)(267 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // pTexObject
    event->args[1] = PT_REGS_PARM2(ctx); // pResDesc
    event->args[2] = PT_REGS_PARM3(ctx); // pTexDesc
    event->args[3] = PT_REGS_PARM4(ctx); // pResViewDesc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_destroy_texture_object_entry")
int hip_destroy_texture_object_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(268 / 100 + '0');
    event->function_name[6] = (char)((268 / 10) % 10 + '0');
    event->function_name[7] = (char)(268 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // textureObject
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_channel_desc_entry")
int hip_get_channel_desc_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(269 / 100 + '0');
    event->function_name[6] = (char)((269 / 10) % 10 + '0');
    event->function_name[7] = (char)(269 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // desc
    event->args[1] = PT_REGS_PARM2(ctx); // array
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_texture_object_resource_desc_entry")
int hip_get_texture_object_resource_desc_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(270 / 100 + '0');
    event->function_name[6] = (char)((270 / 10) % 10 + '0');
    event->function_name[7] = (char)(270 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pResDesc
    event->args[1] = PT_REGS_PARM2(ctx); // textureObject
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_texture_object_resource_view_desc_entry")
int hip_get_texture_object_resource_view_desc_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(271 / 100 + '0');
    event->function_name[6] = (char)((271 / 10) % 10 + '0');
    event->function_name[7] = (char)(271 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pResViewDesc
    event->args[1] = PT_REGS_PARM2(ctx); // textureObject
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_texture_object_texture_desc_entry")
int hip_get_texture_object_texture_desc_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(272 / 100 + '0');
    event->function_name[6] = (char)((272 / 10) % 10 + '0');
    event->function_name[7] = (char)(272 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pTexDesc
    event->args[1] = PT_REGS_PARM2(ctx); // textureObject
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_object_create_entry")
int hip_tex_object_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(273 / 100 + '0');
    event->function_name[6] = (char)((273 / 10) % 10 + '0');
    event->function_name[7] = (char)(273 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // pTexObject
    event->args[1] = PT_REGS_PARM2(ctx); // pResDesc
    event->args[2] = PT_REGS_PARM3(ctx); // pTexDesc
    event->args[3] = PT_REGS_PARM4(ctx); // pResViewDesc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_object_destroy_entry")
int hip_tex_object_destroy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(274 / 100 + '0');
    event->function_name[6] = (char)((274 / 10) % 10 + '0');
    event->function_name[7] = (char)(274 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // texObject
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_object_get_resource_desc_entry")
int hip_tex_object_get_resource_desc_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(275 / 100 + '0');
    event->function_name[6] = (char)((275 / 10) % 10 + '0');
    event->function_name[7] = (char)(275 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pResDesc
    event->args[1] = PT_REGS_PARM2(ctx); // texObject
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_object_get_resource_view_desc_entry")
int hip_tex_object_get_resource_view_desc_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(276 / 100 + '0');
    event->function_name[6] = (char)((276 / 10) % 10 + '0');
    event->function_name[7] = (char)(276 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pResViewDesc
    event->args[1] = PT_REGS_PARM2(ctx); // texObject
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_object_get_texture_desc_entry")
int hip_tex_object_get_texture_desc_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(277 / 100 + '0');
    event->function_name[6] = (char)((277 / 10) % 10 + '0');
    event->function_name[7] = (char)(277 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pTexDesc
    event->args[1] = PT_REGS_PARM2(ctx); // texObject
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_malloc_mipmapped_array_entry")
int hip_malloc_mipmapped_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(278 / 100 + '0');
    event->function_name[6] = (char)((278 / 10) % 10 + '0');
    event->function_name[7] = (char)(278 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // mipmappedArray
    event->args[1] = PT_REGS_PARM2(ctx); // desc
    event->args[2] = PT_REGS_PARM3(ctx); // extent
    event->args[3] = PT_REGS_PARM4(ctx); // numLevels
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_free_mipmapped_array_entry")
int hip_free_mipmapped_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(279 / 100 + '0');
    event->function_name[6] = (char)((279 / 10) % 10 + '0');
    event->function_name[7] = (char)(279 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // mipmappedArray
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_mipmapped_array_level_entry")
int hip_get_mipmapped_array_level_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(280 / 100 + '0');
    event->function_name[6] = (char)((280 / 10) % 10 + '0');
    event->function_name[7] = (char)(280 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // levelArray
    event->args[1] = PT_REGS_PARM2(ctx); // mipmappedArray
    event->args[2] = PT_REGS_PARM3(ctx); // level
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mipmapped_array_create_entry")
int hip_mipmapped_array_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(281 / 100 + '0');
    event->function_name[6] = (char)((281 / 10) % 10 + '0');
    event->function_name[7] = (char)(281 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pHandle
    event->args[1] = PT_REGS_PARM2(ctx); // pMipmappedArrayDesc
    event->args[2] = PT_REGS_PARM3(ctx); // numMipmapLevels
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mipmapped_array_destroy_entry")
int hip_mipmapped_array_destroy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(282 / 100 + '0');
    event->function_name[6] = (char)((282 / 10) % 10 + '0');
    event->function_name[7] = (char)(282 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // hMipmappedArray
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mipmapped_array_get_level_entry")
int hip_mipmapped_array_get_level_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(283 / 100 + '0');
    event->function_name[6] = (char)((283 / 10) % 10 + '0');
    event->function_name[7] = (char)(283 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pLevelArray
    event->args[1] = PT_REGS_PARM2(ctx); // hMipMappedArray
    event->args[2] = PT_REGS_PARM3(ctx); // level
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_bind_texture_to_mipmapped_array_entry")
int hip_bind_texture_to_mipmapped_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(284 / 100 + '0');
    event->function_name[6] = (char)((284 / 10) % 10 + '0');
    event->function_name[7] = (char)(284 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // tex
    event->args[1] = PT_REGS_PARM2(ctx); // mipmappedArray
    event->args[2] = PT_REGS_PARM3(ctx); // desc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_texture_reference_entry")
int hip_get_texture_reference_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(285 / 100 + '0');
    event->function_name[6] = (char)((285 / 10) % 10 + '0');
    event->function_name[7] = (char)(285 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // texref
    event->args[1] = PT_REGS_PARM2(ctx); // symbol
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_border_color_entry")
int hip_tex_ref_get_border_color_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(286 / 100 + '0');
    event->function_name[6] = (char)((286 / 10) % 10 + '0');
    event->function_name[7] = (char)(286 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pBorderColor
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_array_entry")
int hip_tex_ref_get_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(287 / 100 + '0');
    event->function_name[6] = (char)((287 / 10) % 10 + '0');
    event->function_name[7] = (char)(287 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pArray
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_address_mode_entry")
int hip_tex_ref_set_address_mode_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(288 / 100 + '0');
    event->function_name[6] = (char)((288 / 10) % 10 + '0');
    event->function_name[7] = (char)(288 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // dim
    event->args[2] = PT_REGS_PARM3(ctx); // am
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_array_entry")
int hip_tex_ref_set_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(289 / 100 + '0');
    event->function_name[6] = (char)((289 / 10) % 10 + '0');
    event->function_name[7] = (char)(289 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // tex
    event->args[1] = PT_REGS_PARM2(ctx); // array
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_filter_mode_entry")
int hip_tex_ref_set_filter_mode_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(290 / 100 + '0');
    event->function_name[6] = (char)((290 / 10) % 10 + '0');
    event->function_name[7] = (char)(290 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // fm
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_flags_entry")
int hip_tex_ref_set_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(291 / 100 + '0');
    event->function_name[6] = (char)((291 / 10) % 10 + '0');
    event->function_name[7] = (char)(291 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // Flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_format_entry")
int hip_tex_ref_set_format_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(292 / 100 + '0');
    event->function_name[6] = (char)((292 / 10) % 10 + '0');
    event->function_name[7] = (char)(292 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // fmt
    event->args[2] = PT_REGS_PARM3(ctx); // NumPackedComponents
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_bind_texture_entry")
int hip_bind_texture_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(293 / 100 + '0');
    event->function_name[6] = (char)((293 / 10) % 10 + '0');
    event->function_name[7] = (char)(293 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // offset
    event->args[1] = PT_REGS_PARM2(ctx); // tex
    event->args[2] = PT_REGS_PARM3(ctx); // devPtr
    event->args[3] = PT_REGS_PARM4(ctx); // desc
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(UINT_MAX)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_bind_texture2_d_entry")
int hip_bind_texture2_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(294 / 100 + '0');
    event->function_name[6] = (char)((294 / 10) % 10 + '0');
    event->function_name[7] = (char)(294 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 7;
    event->args[0] = PT_REGS_PARM1(ctx); // offset
    event->args[1] = PT_REGS_PARM2(ctx); // tex
    event->args[2] = PT_REGS_PARM3(ctx); // devPtr
    event->args[3] = PT_REGS_PARM4(ctx); // desc
    event->args[4] = PT_REGS_PARM5(ctx); // width
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_bind_texture_to_array_entry")
int hip_bind_texture_to_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(295 / 100 + '0');
    event->function_name[6] = (char)((295 / 10) % 10 + '0');
    event->function_name[7] = (char)(295 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // tex
    event->args[1] = PT_REGS_PARM2(ctx); // array
    event->args[2] = PT_REGS_PARM3(ctx); // desc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_get_texture_alignment_offset_entry")
int hip_get_texture_alignment_offset_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(296 / 100 + '0');
    event->function_name[6] = (char)((296 / 10) % 10 + '0');
    event->function_name[7] = (char)(296 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // offset
    event->args[1] = PT_REGS_PARM2(ctx); // texref
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_unbind_texture_entry")
int hip_unbind_texture_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(297 / 100 + '0');
    event->function_name[6] = (char)((297 / 10) % 10 + '0');
    event->function_name[7] = (char)(297 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // tex
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_address_entry")
int hip_tex_ref_get_address_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(298 / 100 + '0');
    event->function_name[6] = (char)((298 / 10) % 10 + '0');
    event->function_name[7] = (char)(298 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // dev_ptr
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_address_mode_entry")
int hip_tex_ref_get_address_mode_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(299 / 100 + '0');
    event->function_name[6] = (char)((299 / 10) % 10 + '0');
    event->function_name[7] = (char)(299 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pam
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->args[2] = PT_REGS_PARM3(ctx); // dim
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_filter_mode_entry")
int hip_tex_ref_get_filter_mode_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(300 / 100 + '0');
    event->function_name[6] = (char)((300 / 10) % 10 + '0');
    event->function_name[7] = (char)(300 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pfm
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_flags_entry")
int hip_tex_ref_get_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(301 / 100 + '0');
    event->function_name[6] = (char)((301 / 10) % 10 + '0');
    event->function_name[7] = (char)(301 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pFlags
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_format_entry")
int hip_tex_ref_get_format_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(302 / 100 + '0');
    event->function_name[6] = (char)((302 / 10) % 10 + '0');
    event->function_name[7] = (char)(302 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pFormat
    event->args[1] = PT_REGS_PARM2(ctx); // pNumChannels
    event->args[2] = PT_REGS_PARM3(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_max_anisotropy_entry")
int hip_tex_ref_get_max_anisotropy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(303 / 100 + '0');
    event->function_name[6] = (char)((303 / 10) % 10 + '0');
    event->function_name[7] = (char)(303 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pmaxAnsio
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_mipmap_filter_mode_entry")
int hip_tex_ref_get_mipmap_filter_mode_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(304 / 100 + '0');
    event->function_name[6] = (char)((304 / 10) % 10 + '0');
    event->function_name[7] = (char)(304 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pfm
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_mipmap_level_bias_entry")
int hip_tex_ref_get_mipmap_level_bias_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(305 / 100 + '0');
    event->function_name[6] = (char)((305 / 10) % 10 + '0');
    event->function_name[7] = (char)(305 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pbias
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_mipmap_level_clamp_entry")
int hip_tex_ref_get_mipmap_level_clamp_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(306 / 100 + '0');
    event->function_name[6] = (char)((306 / 10) % 10 + '0');
    event->function_name[7] = (char)(306 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pminMipmapLevelClamp
    event->args[1] = PT_REGS_PARM2(ctx); // pmaxMipmapLevelClamp
    event->args[2] = PT_REGS_PARM3(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_get_mip_mapped_array_entry")
int hip_tex_ref_get_mip_mapped_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(307 / 100 + '0');
    event->function_name[6] = (char)((307 / 10) % 10 + '0');
    event->function_name[7] = (char)(307 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pArray
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_address_entry")
int hip_tex_ref_set_address_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(308 / 100 + '0');
    event->function_name[6] = (char)((308 / 10) % 10 + '0');
    event->function_name[7] = (char)(308 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // ByteOffset
    event->args[1] = PT_REGS_PARM2(ctx); // texRef
    event->args[2] = PT_REGS_PARM3(ctx); // dptr
    event->args[3] = PT_REGS_PARM4(ctx); // bytes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_address2_d_entry")
int hip_tex_ref_set_address2_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(309 / 100 + '0');
    event->function_name[6] = (char)((309 / 10) % 10 + '0');
    event->function_name[7] = (char)(309 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // desc
    event->args[2] = PT_REGS_PARM3(ctx); // dptr
    event->args[3] = PT_REGS_PARM4(ctx); // Pitch
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_max_anisotropy_entry")
int hip_tex_ref_set_max_anisotropy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(310 / 100 + '0');
    event->function_name[6] = (char)((310 / 10) % 10 + '0');
    event->function_name[7] = (char)(310 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // maxAniso
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_border_color_entry")
int hip_tex_ref_set_border_color_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(311 / 100 + '0');
    event->function_name[6] = (char)((311 / 10) % 10 + '0');
    event->function_name[7] = (char)(311 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // pBorderColor
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_mipmap_filter_mode_entry")
int hip_tex_ref_set_mipmap_filter_mode_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(312 / 100 + '0');
    event->function_name[6] = (char)((312 / 10) % 10 + '0');
    event->function_name[7] = (char)(312 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // fm
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_mipmap_level_bias_entry")
int hip_tex_ref_set_mipmap_level_bias_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(313 / 100 + '0');
    event->function_name[6] = (char)((313 / 10) % 10 + '0');
    event->function_name[7] = (char)(313 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // bias
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_mipmap_level_clamp_entry")
int hip_tex_ref_set_mipmap_level_clamp_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(314 / 100 + '0');
    event->function_name[6] = (char)((314 / 10) % 10 + '0');
    event->function_name[7] = (char)(314 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // minMipMapLevelClamp
    event->args[2] = PT_REGS_PARM3(ctx); // maxMipMapLevelClamp
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_tex_ref_set_mipmapped_array_entry")
int hip_tex_ref_set_mipmapped_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(315 / 100 + '0');
    event->function_name[6] = (char)((315 / 10) % 10 + '0');
    event->function_name[7] = (char)(315 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // texRef
    event->args[1] = PT_REGS_PARM2(ctx); // mipmappedArray
    event->args[2] = PT_REGS_PARM3(ctx); // Flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_begin_capture_entry")
int hip_stream_begin_capture_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(316 / 100 + '0');
    event->function_name[6] = (char)((316 / 10) % 10 + '0');
    event->function_name[7] = (char)(316 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // mode
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_begin_capture_to_graph_entry")
int hip_stream_begin_capture_to_graph_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(317 / 100 + '0');
    event->function_name[6] = (char)((317 / 10) % 10 + '0');
    event->function_name[7] = (char)(317 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // dependencies
    event->args[3] = PT_REGS_PARM4(ctx); // dependencyData
    event->args[4] = PT_REGS_PARM5(ctx); // numDependencies
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_end_capture_entry")
int hip_stream_end_capture_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(318 / 100 + '0');
    event->function_name[6] = (char)((318 / 10) % 10 + '0');
    event->function_name[7] = (char)(318 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // pGraph
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_get_capture_info_entry")
int hip_stream_get_capture_info_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(319 / 100 + '0');
    event->function_name[6] = (char)((319 / 10) % 10 + '0');
    event->function_name[7] = (char)(319 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // pCaptureStatus
    event->args[2] = PT_REGS_PARM3(ctx); // pId
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_get_capture_info_v2_entry")
int hip_stream_get_capture_info_v2_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(320 / 100 + '0');
    event->function_name[6] = (char)((320 / 10) % 10 + '0');
    event->function_name[7] = (char)(320 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // captureStatus_out
    event->args[2] = PT_REGS_PARM3(ctx); // __dparm(0)
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->args[4] = PT_REGS_PARM5(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_is_capturing_entry")
int hip_stream_is_capturing_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(321 / 100 + '0');
    event->function_name[6] = (char)((321 / 10) % 10 + '0');
    event->function_name[7] = (char)(321 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // pCaptureStatus
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_stream_update_capture_dependencies_entry")
int hip_stream_update_capture_dependencies_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(322 / 100 + '0');
    event->function_name[6] = (char)((322 / 10) % 10 + '0');
    event->function_name[7] = (char)(322 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // stream
    event->args[1] = PT_REGS_PARM2(ctx); // dependencies
    event->args[2] = PT_REGS_PARM3(ctx); // numDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_thread_exchange_stream_capture_mode_entry")
int hip_thread_exchange_stream_capture_mode_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(323 / 100 + '0');
    event->function_name[6] = (char)((323 / 10) % 10 + '0');
    event->function_name[7] = (char)(323 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // mode
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_create_entry")
int hip_graph_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(324 / 100 + '0');
    event->function_name[6] = (char)((324 / 10) % 10 + '0');
    event->function_name[7] = (char)(324 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraph
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_destroy_entry")
int hip_graph_destroy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(325 / 100 + '0');
    event->function_name[6] = (char)((325 / 10) % 10 + '0');
    event->function_name[7] = (char)(325 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // graph
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_dependencies_entry")
int hip_graph_add_dependencies_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(326 / 100 + '0');
    event->function_name[6] = (char)((326 / 10) % 10 + '0');
    event->function_name[7] = (char)(326 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // graph
    event->args[1] = PT_REGS_PARM2(ctx); // from
    event->args[2] = PT_REGS_PARM3(ctx); // to
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_remove_dependencies_entry")
int hip_graph_remove_dependencies_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(327 / 100 + '0');
    event->function_name[6] = (char)((327 / 10) % 10 + '0');
    event->function_name[7] = (char)(327 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // graph
    event->args[1] = PT_REGS_PARM2(ctx); // from
    event->args[2] = PT_REGS_PARM3(ctx); // to
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_get_edges_entry")
int hip_graph_get_edges_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(328 / 100 + '0');
    event->function_name[6] = (char)((328 / 10) % 10 + '0');
    event->function_name[7] = (char)(328 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // graph
    event->args[1] = PT_REGS_PARM2(ctx); // from
    event->args[2] = PT_REGS_PARM3(ctx); // to
    event->args[3] = PT_REGS_PARM4(ctx); // numEdges
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_get_nodes_entry")
int hip_graph_get_nodes_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(329 / 100 + '0');
    event->function_name[6] = (char)((329 / 10) % 10 + '0');
    event->function_name[7] = (char)(329 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // graph
    event->args[1] = PT_REGS_PARM2(ctx); // nodes
    event->args[2] = PT_REGS_PARM3(ctx); // numNodes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_get_root_nodes_entry")
int hip_graph_get_root_nodes_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(330 / 100 + '0');
    event->function_name[6] = (char)((330 / 10) % 10 + '0');
    event->function_name[7] = (char)(330 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // graph
    event->args[1] = PT_REGS_PARM2(ctx); // pRootNodes
    event->args[2] = PT_REGS_PARM3(ctx); // pNumRootNodes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_node_get_dependencies_entry")
int hip_graph_node_get_dependencies_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(331 / 100 + '0');
    event->function_name[6] = (char)((331 / 10) % 10 + '0');
    event->function_name[7] = (char)(331 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pDependencies
    event->args[2] = PT_REGS_PARM3(ctx); // pNumDependencies
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_node_get_dependent_nodes_entry")
int hip_graph_node_get_dependent_nodes_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(332 / 100 + '0');
    event->function_name[6] = (char)((332 / 10) % 10 + '0');
    event->function_name[7] = (char)(332 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pDependentNodes
    event->args[2] = PT_REGS_PARM3(ctx); // pNumDependentNodes
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_node_get_type_entry")
int hip_graph_node_get_type_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(333 / 100 + '0');
    event->function_name[6] = (char)((333 / 10) % 10 + '0');
    event->function_name[7] = (char)(333 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pType
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_destroy_node_entry")
int hip_graph_destroy_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(334 / 100 + '0');
    event->function_name[6] = (char)((334 / 10) % 10 + '0');
    event->function_name[7] = (char)(334 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_clone_entry")
int hip_graph_clone_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(335 / 100 + '0');
    event->function_name[6] = (char)((335 / 10) % 10 + '0');
    event->function_name[7] = (char)(335 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphClone
    event->args[1] = PT_REGS_PARM2(ctx); // originalGraph
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_node_find_in_clone_entry")
int hip_graph_node_find_in_clone_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(336 / 100 + '0');
    event->function_name[6] = (char)((336 / 10) % 10 + '0');
    event->function_name[7] = (char)(336 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pNode
    event->args[1] = PT_REGS_PARM2(ctx); // originalNode
    event->args[2] = PT_REGS_PARM3(ctx); // clonedGraph
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_instantiate_entry")
int hip_graph_instantiate_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(337 / 100 + '0');
    event->function_name[6] = (char)((337 / 10) % 10 + '0');
    event->function_name[7] = (char)(337 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pErrorNode
    event->args[3] = PT_REGS_PARM4(ctx); // pLogBuffer
    event->args[4] = PT_REGS_PARM5(ctx); // bufferSize
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_instantiate_with_flags_entry")
int hip_graph_instantiate_with_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(338 / 100 + '0');
    event->function_name[6] = (char)((338 / 10) % 10 + '0');
    event->function_name[7] = (char)(338 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_instantiate_with_params_entry")
int hip_graph_instantiate_with_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(339 / 100 + '0');
    event->function_name[6] = (char)((339 / 10) % 10 + '0');
    event->function_name[7] = (char)(339 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // instantiateParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_launch_entry")
int hip_graph_launch_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(340 / 100 + '0');
    event->function_name[6] = (char)((340 / 10) % 10 + '0');
    event->function_name[7] = (char)(340 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // graphExec
    event->args[1] = PT_REGS_PARM2(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_upload_entry")
int hip_graph_upload_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(341 / 100 + '0');
    event->function_name[6] = (char)((341 / 10) % 10 + '0');
    event->function_name[7] = (char)(341 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // graphExec
    event->args[1] = PT_REGS_PARM2(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_node_entry")
int hip_graph_add_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(342 / 100 + '0');
    event->function_name[6] = (char)((342 / 10) % 10 + '0');
    event->function_name[7] = (char)(342 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_get_flags_entry")
int hip_graph_exec_get_flags_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(343 / 100 + '0');
    event->function_name[6] = (char)((343 / 10) % 10 + '0');
    event->function_name[7] = (char)(343 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // graphExec
    event->args[1] = PT_REGS_PARM2(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_node_set_params_entry")
int hip_graph_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(344 / 100 + '0');
    event->function_name[6] = (char)((344 / 10) % 10 + '0');
    event->function_name[7] = (char)(344 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_node_set_params_entry")
int hip_graph_exec_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(345 / 100 + '0');
    event->function_name[6] = (char)((345 / 10) % 10 + '0');
    event->function_name[7] = (char)(345 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // graphExec
    event->args[1] = PT_REGS_PARM2(ctx); // node
    event->args[2] = PT_REGS_PARM3(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_destroy_entry")
int hip_graph_exec_destroy_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(346 / 100 + '0');
    event->function_name[6] = (char)((346 / 10) % 10 + '0');
    event->function_name[7] = (char)(346 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // graphExec
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_update_entry")
int hip_graph_exec_update_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(347 / 100 + '0');
    event->function_name[6] = (char)((347 / 10) % 10 + '0');
    event->function_name[7] = (char)(347 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // hGraph
    event->args[2] = PT_REGS_PARM3(ctx); // hErrorNode_out
    event->args[3] = PT_REGS_PARM4(ctx); // updateResult_out
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_kernel_node_entry")
int hip_graph_add_kernel_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(348 / 100 + '0');
    event->function_name[6] = (char)((348 / 10) % 10 + '0');
    event->function_name[7] = (char)(348 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_kernel_node_get_params_entry")
int hip_graph_kernel_node_get_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(349 / 100 + '0');
    event->function_name[6] = (char)((349 / 10) % 10 + '0');
    event->function_name[7] = (char)(349 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_kernel_node_set_params_entry")
int hip_graph_kernel_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(350 / 100 + '0');
    event->function_name[6] = (char)((350 / 10) % 10 + '0');
    event->function_name[7] = (char)(350 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_kernel_node_set_params_entry")
int hip_graph_exec_kernel_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(351 / 100 + '0');
    event->function_name[6] = (char)((351 / 10) % 10 + '0');
    event->function_name[7] = (char)(351 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // node
    event->args[2] = PT_REGS_PARM3(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_graph_add_memcpy_node_entry")
int hip_drv_graph_add_memcpy_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(352 / 100 + '0');
    event->function_name[6] = (char)((352 / 10) % 10 + '0');
    event->function_name[7] = (char)(352 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // phGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // hGraph
    event->args[2] = PT_REGS_PARM3(ctx); // dependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // copyParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_memcpy_node_entry")
int hip_graph_add_memcpy_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(353 / 100 + '0');
    event->function_name[6] = (char)((353 / 10) % 10 + '0');
    event->function_name[7] = (char)(353 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // pCopyParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_memcpy_node_get_params_entry")
int hip_graph_memcpy_node_get_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(354 / 100 + '0');
    event->function_name[6] = (char)((354 / 10) % 10 + '0');
    event->function_name[7] = (char)(354 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_memcpy_node_set_params_entry")
int hip_graph_memcpy_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(355 / 100 + '0');
    event->function_name[6] = (char)((355 / 10) % 10 + '0');
    event->function_name[7] = (char)(355 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_kernel_node_set_attribute_entry")
int hip_graph_kernel_node_set_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(356 / 100 + '0');
    event->function_name[6] = (char)((356 / 10) % 10 + '0');
    event->function_name[7] = (char)(356 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hNode
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_kernel_node_get_attribute_entry")
int hip_graph_kernel_node_get_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(357 / 100 + '0');
    event->function_name[6] = (char)((357 / 10) % 10 + '0');
    event->function_name[7] = (char)(357 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hNode
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_memcpy_node_set_params_entry")
int hip_graph_exec_memcpy_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(358 / 100 + '0');
    event->function_name[6] = (char)((358 / 10) % 10 + '0');
    event->function_name[7] = (char)(358 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // node
    event->args[2] = PT_REGS_PARM3(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_memcpy_node1_d_entry")
int hip_graph_add_memcpy_node1_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(359 / 100 + '0');
    event->function_name[6] = (char)((359 / 10) % 10 + '0');
    event->function_name[7] = (char)(359 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 8;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // dst
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_memcpy_node_set_params1_d_entry")
int hip_graph_memcpy_node_set_params1_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(360 / 100 + '0');
    event->function_name[6] = (char)((360 / 10) % 10 + '0');
    event->function_name[7] = (char)(360 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // dst
    event->args[2] = PT_REGS_PARM3(ctx); // src
    event->args[3] = PT_REGS_PARM4(ctx); // count
    event->args[4] = PT_REGS_PARM5(ctx); // kind
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_memcpy_node_set_params1_d_entry")
int hip_graph_exec_memcpy_node_set_params1_d_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(361 / 100 + '0');
    event->function_name[6] = (char)((361 / 10) % 10 + '0');
    event->function_name[7] = (char)(361 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // node
    event->args[2] = PT_REGS_PARM3(ctx); // dst
    event->args[3] = PT_REGS_PARM4(ctx); // src
    event->args[4] = PT_REGS_PARM5(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_memcpy_node_from_symbol_entry")
int hip_graph_add_memcpy_node_from_symbol_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(362 / 100 + '0');
    event->function_name[6] = (char)((362 / 10) % 10 + '0');
    event->function_name[7] = (char)(362 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 9;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // dst
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_memcpy_node_set_params_from_symbol_entry")
int hip_graph_memcpy_node_set_params_from_symbol_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(363 / 100 + '0');
    event->function_name[6] = (char)((363 / 10) % 10 + '0');
    event->function_name[7] = (char)(363 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // dst
    event->args[2] = PT_REGS_PARM3(ctx); // symbol
    event->args[3] = PT_REGS_PARM4(ctx); // count
    event->args[4] = PT_REGS_PARM5(ctx); // offset
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_memcpy_node_set_params_from_symbol_entry")
int hip_graph_exec_memcpy_node_set_params_from_symbol_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(364 / 100 + '0');
    event->function_name[6] = (char)((364 / 10) % 10 + '0');
    event->function_name[7] = (char)(364 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 7;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // node
    event->args[2] = PT_REGS_PARM3(ctx); // dst
    event->args[3] = PT_REGS_PARM4(ctx); // symbol
    event->args[4] = PT_REGS_PARM5(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_memcpy_node_to_symbol_entry")
int hip_graph_add_memcpy_node_to_symbol_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(365 / 100 + '0');
    event->function_name[6] = (char)((365 / 10) % 10 + '0');
    event->function_name[7] = (char)(365 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 9;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // symbol
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_memcpy_node_set_params_to_symbol_entry")
int hip_graph_memcpy_node_set_params_to_symbol_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(366 / 100 + '0');
    event->function_name[6] = (char)((366 / 10) % 10 + '0');
    event->function_name[7] = (char)(366 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // symbol
    event->args[2] = PT_REGS_PARM3(ctx); // src
    event->args[3] = PT_REGS_PARM4(ctx); // count
    event->args[4] = PT_REGS_PARM5(ctx); // offset
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_memcpy_node_set_params_to_symbol_entry")
int hip_graph_exec_memcpy_node_set_params_to_symbol_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(367 / 100 + '0');
    event->function_name[6] = (char)((367 / 10) % 10 + '0');
    event->function_name[7] = (char)(367 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 7;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // node
    event->args[2] = PT_REGS_PARM3(ctx); // symbol
    event->args[3] = PT_REGS_PARM4(ctx); // src
    event->args[4] = PT_REGS_PARM5(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_memset_node_entry")
int hip_graph_add_memset_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(368 / 100 + '0');
    event->function_name[6] = (char)((368 / 10) % 10 + '0');
    event->function_name[7] = (char)(368 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // pMemsetParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_memset_node_get_params_entry")
int hip_graph_memset_node_get_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(369 / 100 + '0');
    event->function_name[6] = (char)((369 / 10) % 10 + '0');
    event->function_name[7] = (char)(369 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_memset_node_set_params_entry")
int hip_graph_memset_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(370 / 100 + '0');
    event->function_name[6] = (char)((370 / 10) % 10 + '0');
    event->function_name[7] = (char)(370 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_memset_node_set_params_entry")
int hip_graph_exec_memset_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(371 / 100 + '0');
    event->function_name[6] = (char)((371 / 10) % 10 + '0');
    event->function_name[7] = (char)(371 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // node
    event->args[2] = PT_REGS_PARM3(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_host_node_entry")
int hip_graph_add_host_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(372 / 100 + '0');
    event->function_name[6] = (char)((372 / 10) % 10 + '0');
    event->function_name[7] = (char)(372 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_host_node_get_params_entry")
int hip_graph_host_node_get_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(373 / 100 + '0');
    event->function_name[6] = (char)((373 / 10) % 10 + '0');
    event->function_name[7] = (char)(373 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_host_node_set_params_entry")
int hip_graph_host_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(374 / 100 + '0');
    event->function_name[6] = (char)((374 / 10) % 10 + '0');
    event->function_name[7] = (char)(374 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_host_node_set_params_entry")
int hip_graph_exec_host_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(375 / 100 + '0');
    event->function_name[6] = (char)((375 / 10) % 10 + '0');
    event->function_name[7] = (char)(375 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // node
    event->args[2] = PT_REGS_PARM3(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_child_graph_node_entry")
int hip_graph_add_child_graph_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(376 / 100 + '0');
    event->function_name[6] = (char)((376 / 10) % 10 + '0');
    event->function_name[7] = (char)(376 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // childGraph
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_child_graph_node_get_graph_entry")
int hip_graph_child_graph_node_get_graph_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(377 / 100 + '0');
    event->function_name[6] = (char)((377 / 10) % 10 + '0');
    event->function_name[7] = (char)(377 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pGraph
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_child_graph_node_set_params_entry")
int hip_graph_exec_child_graph_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(378 / 100 + '0');
    event->function_name[6] = (char)((378 / 10) % 10 + '0');
    event->function_name[7] = (char)(378 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // node
    event->args[2] = PT_REGS_PARM3(ctx); // childGraph
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_empty_node_entry")
int hip_graph_add_empty_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(379 / 100 + '0');
    event->function_name[6] = (char)((379 / 10) % 10 + '0');
    event->function_name[7] = (char)(379 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_event_record_node_entry")
int hip_graph_add_event_record_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(380 / 100 + '0');
    event->function_name[6] = (char)((380 / 10) % 10 + '0');
    event->function_name[7] = (char)(380 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_event_record_node_get_event_entry")
int hip_graph_event_record_node_get_event_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(381 / 100 + '0');
    event->function_name[6] = (char)((381 / 10) % 10 + '0');
    event->function_name[7] = (char)(381 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // event_out
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_event_record_node_set_event_entry")
int hip_graph_event_record_node_set_event_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(382 / 100 + '0');
    event->function_name[6] = (char)((382 / 10) % 10 + '0');
    event->function_name[7] = (char)(382 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_event_record_node_set_event_entry")
int hip_graph_exec_event_record_node_set_event_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(383 / 100 + '0');
    event->function_name[6] = (char)((383 / 10) % 10 + '0');
    event->function_name[7] = (char)(383 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // hNode
    event->args[2] = PT_REGS_PARM3(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_event_wait_node_entry")
int hip_graph_add_event_wait_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(384 / 100 + '0');
    event->function_name[6] = (char)((384 / 10) % 10 + '0');
    event->function_name[7] = (char)(384 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_event_wait_node_get_event_entry")
int hip_graph_event_wait_node_get_event_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(385 / 100 + '0');
    event->function_name[6] = (char)((385 / 10) % 10 + '0');
    event->function_name[7] = (char)(385 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // event_out
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_event_wait_node_set_event_entry")
int hip_graph_event_wait_node_set_event_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(386 / 100 + '0');
    event->function_name[6] = (char)((386 / 10) % 10 + '0');
    event->function_name[7] = (char)(386 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_event_wait_node_set_event_entry")
int hip_graph_exec_event_wait_node_set_event_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(387 / 100 + '0');
    event->function_name[6] = (char)((387 / 10) % 10 + '0');
    event->function_name[7] = (char)(387 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // hNode
    event->args[2] = PT_REGS_PARM3(ctx); // event
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_mem_alloc_node_entry")
int hip_graph_add_mem_alloc_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(388 / 100 + '0');
    event->function_name[6] = (char)((388 / 10) % 10 + '0');
    event->function_name[7] = (char)(388 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_mem_alloc_node_get_params_entry")
int hip_graph_mem_alloc_node_get_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(389 / 100 + '0');
    event->function_name[6] = (char)((389 / 10) % 10 + '0');
    event->function_name[7] = (char)(389 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // pNodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_mem_free_node_entry")
int hip_graph_add_mem_free_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(390 / 100 + '0');
    event->function_name[6] = (char)((390 / 10) % 10 + '0');
    event->function_name[7] = (char)(390 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // dev_ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_mem_free_node_get_params_entry")
int hip_graph_mem_free_node_get_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(391 / 100 + '0');
    event->function_name[6] = (char)((391 / 10) % 10 + '0');
    event->function_name[7] = (char)(391 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // node
    event->args[1] = PT_REGS_PARM2(ctx); // dev_ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_get_graph_mem_attribute_entry")
int hip_device_get_graph_mem_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(392 / 100 + '0');
    event->function_name[6] = (char)((392 / 10) % 10 + '0');
    event->function_name[7] = (char)(392 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // device
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_set_graph_mem_attribute_entry")
int hip_device_set_graph_mem_attribute_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(393 / 100 + '0');
    event->function_name[6] = (char)((393 / 10) % 10 + '0');
    event->function_name[7] = (char)(393 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // device
    event->args[1] = PT_REGS_PARM2(ctx); // attr
    event->args[2] = PT_REGS_PARM3(ctx); // value
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_device_graph_mem_trim_entry")
int hip_device_graph_mem_trim_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(394 / 100 + '0');
    event->function_name[6] = (char)((394 / 10) % 10 + '0');
    event->function_name[7] = (char)(394 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // device
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_user_object_create_entry")
int hip_user_object_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(395 / 100 + '0');
    event->function_name[6] = (char)((395 / 10) % 10 + '0');
    event->function_name[7] = (char)(395 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // object_out
    event->args[1] = PT_REGS_PARM2(ctx); // ptr
    event->args[2] = PT_REGS_PARM3(ctx); // destroy
    event->args[3] = PT_REGS_PARM4(ctx); // initialRefcount
    event->args[4] = PT_REGS_PARM5(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_user_object_release_entry")
int hip_user_object_release_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(396 / 100 + '0');
    event->function_name[6] = (char)((396 / 10) % 10 + '0');
    event->function_name[7] = (char)(396 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // object
    event->args[1] = PT_REGS_PARM2(ctx); // __dparm(1)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_user_object_retain_entry")
int hip_user_object_retain_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(397 / 100 + '0');
    event->function_name[6] = (char)((397 / 10) % 10 + '0');
    event->function_name[7] = (char)(397 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // object
    event->args[1] = PT_REGS_PARM2(ctx); // __dparm(1)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_retain_user_object_entry")
int hip_graph_retain_user_object_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(398 / 100 + '0');
    event->function_name[6] = (char)((398 / 10) % 10 + '0');
    event->function_name[7] = (char)(398 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // graph
    event->args[1] = PT_REGS_PARM2(ctx); // object
    event->args[2] = PT_REGS_PARM3(ctx); // __dparm(1)
    event->args[3] = PT_REGS_PARM4(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_release_user_object_entry")
int hip_graph_release_user_object_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(399 / 100 + '0');
    event->function_name[6] = (char)((399 / 10) % 10 + '0');
    event->function_name[7] = (char)(399 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // graph
    event->args[1] = PT_REGS_PARM2(ctx); // object
    event->args[2] = PT_REGS_PARM3(ctx); // __dparm(1)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_debug_dot_print_entry")
int hip_graph_debug_dot_print_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(400 / 100 + '0');
    event->function_name[6] = (char)((400 / 10) % 10 + '0');
    event->function_name[7] = (char)(400 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // graph
    event->args[1] = PT_REGS_PARM2(ctx); // path
    event->args[2] = PT_REGS_PARM3(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_kernel_node_copy_attributes_entry")
int hip_graph_kernel_node_copy_attributes_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(401 / 100 + '0');
    event->function_name[6] = (char)((401 / 10) % 10 + '0');
    event->function_name[7] = (char)(401 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // hSrc
    event->args[1] = PT_REGS_PARM2(ctx); // hDst
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_node_set_enabled_entry")
int hip_graph_node_set_enabled_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(402 / 100 + '0');
    event->function_name[6] = (char)((402 / 10) % 10 + '0');
    event->function_name[7] = (char)(402 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // hNode
    event->args[2] = PT_REGS_PARM3(ctx); // isEnabled
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_node_get_enabled_entry")
int hip_graph_node_get_enabled_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(403 / 100 + '0');
    event->function_name[6] = (char)((403 / 10) % 10 + '0');
    event->function_name[7] = (char)(403 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // hNode
    event->args[2] = PT_REGS_PARM3(ctx); // isEnabled
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_external_semaphores_wait_node_entry")
int hip_graph_add_external_semaphores_wait_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(404 / 100 + '0');
    event->function_name[6] = (char)((404 / 10) % 10 + '0');
    event->function_name[7] = (char)(404 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_add_external_semaphores_signal_node_entry")
int hip_graph_add_external_semaphores_signal_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(405 / 100 + '0');
    event->function_name[6] = (char)((405 / 10) % 10 + '0');
    event->function_name[7] = (char)(405 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // pGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // graph
    event->args[2] = PT_REGS_PARM3(ctx); // pDependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_external_semaphores_signal_node_set_params_entry")
int hip_graph_external_semaphores_signal_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(406 / 100 + '0');
    event->function_name[6] = (char)((406 / 10) % 10 + '0');
    event->function_name[7] = (char)(406 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // hNode
    event->args[1] = PT_REGS_PARM2(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_external_semaphores_wait_node_set_params_entry")
int hip_graph_external_semaphores_wait_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(407 / 100 + '0');
    event->function_name[6] = (char)((407 / 10) % 10 + '0');
    event->function_name[7] = (char)(407 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // hNode
    event->args[1] = PT_REGS_PARM2(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_external_semaphores_signal_node_get_params_entry")
int hip_graph_external_semaphores_signal_node_get_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(408 / 100 + '0');
    event->function_name[6] = (char)((408 / 10) % 10 + '0');
    event->function_name[7] = (char)(408 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // hNode
    event->args[1] = PT_REGS_PARM2(ctx); // params_out
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_external_semaphores_wait_node_get_params_entry")
int hip_graph_external_semaphores_wait_node_get_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(409 / 100 + '0');
    event->function_name[6] = (char)((409 / 10) % 10 + '0');
    event->function_name[7] = (char)(409 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // hNode
    event->args[1] = PT_REGS_PARM2(ctx); // params_out
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_external_semaphores_signal_node_set_params_entry")
int hip_graph_exec_external_semaphores_signal_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(410 / 100 + '0');
    event->function_name[6] = (char)((410 / 10) % 10 + '0');
    event->function_name[7] = (char)(410 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // hNode
    event->args[2] = PT_REGS_PARM3(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graph_exec_external_semaphores_wait_node_set_params_entry")
int hip_graph_exec_external_semaphores_wait_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(411 / 100 + '0');
    event->function_name[6] = (char)((411 / 10) % 10 + '0');
    event->function_name[7] = (char)(411 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // hNode
    event->args[2] = PT_REGS_PARM3(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_graph_memcpy_node_get_params_entry")
int hip_drv_graph_memcpy_node_get_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(412 / 100 + '0');
    event->function_name[6] = (char)((412 / 10) % 10 + '0');
    event->function_name[7] = (char)(412 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // hNode
    event->args[1] = PT_REGS_PARM2(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_graph_memcpy_node_set_params_entry")
int hip_drv_graph_memcpy_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(413 / 100 + '0');
    event->function_name[6] = (char)((413 / 10) % 10 + '0');
    event->function_name[7] = (char)(413 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // hNode
    event->args[1] = PT_REGS_PARM2(ctx); // nodeParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_graph_add_memset_node_entry")
int hip_drv_graph_add_memset_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(414 / 100 + '0');
    event->function_name[6] = (char)((414 / 10) % 10 + '0');
    event->function_name[7] = (char)(414 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 6;
    event->args[0] = PT_REGS_PARM1(ctx); // phGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // hGraph
    event->args[2] = PT_REGS_PARM3(ctx); // dependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // memsetParams
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_graph_add_mem_free_node_entry")
int hip_drv_graph_add_mem_free_node_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(415 / 100 + '0');
    event->function_name[6] = (char)((415 / 10) % 10 + '0');
    event->function_name[7] = (char)(415 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // phGraphNode
    event->args[1] = PT_REGS_PARM2(ctx); // hGraph
    event->args[2] = PT_REGS_PARM3(ctx); // dependencies
    event->args[3] = PT_REGS_PARM4(ctx); // numDependencies
    event->args[4] = PT_REGS_PARM5(ctx); // dptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_graph_exec_memcpy_node_set_params_entry")
int hip_drv_graph_exec_memcpy_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(416 / 100 + '0');
    event->function_name[6] = (char)((416 / 10) % 10 + '0');
    event->function_name[7] = (char)(416 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // hNode
    event->args[2] = PT_REGS_PARM3(ctx); // copyParams
    event->args[3] = PT_REGS_PARM4(ctx); // ctx
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_drv_graph_exec_memset_node_set_params_entry")
int hip_drv_graph_exec_memset_node_set_params_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(417 / 100 + '0');
    event->function_name[6] = (char)((417 / 10) % 10 + '0');
    event->function_name[7] = (char)(417 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // hGraphExec
    event->args[1] = PT_REGS_PARM2(ctx); // hNode
    event->args[2] = PT_REGS_PARM3(ctx); // memsetParams
    event->args[3] = PT_REGS_PARM4(ctx); // ctx
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_address_free_entry")
int hip_mem_address_free_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(418 / 100 + '0');
    event->function_name[6] = (char)((418 / 10) % 10 + '0');
    event->function_name[7] = (char)(418 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // devPtr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_address_reserve_entry")
int hip_mem_address_reserve_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(419 / 100 + '0');
    event->function_name[6] = (char)((419 / 10) % 10 + '0');
    event->function_name[7] = (char)(419 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // alignment
    event->args[3] = PT_REGS_PARM4(ctx); // addr
    event->args[4] = PT_REGS_PARM5(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_create_entry")
int hip_mem_create_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(420 / 100 + '0');
    event->function_name[6] = (char)((420 / 10) % 10 + '0');
    event->function_name[7] = (char)(420 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // handle
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // prop
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_export_to_shareable_handle_entry")
int hip_mem_export_to_shareable_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(421 / 100 + '0');
    event->function_name[6] = (char)((421 / 10) % 10 + '0');
    event->function_name[7] = (char)(421 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // shareableHandle
    event->args[1] = PT_REGS_PARM2(ctx); // handle
    event->args[2] = PT_REGS_PARM3(ctx); // handleType
    event->args[3] = PT_REGS_PARM4(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_get_access_entry")
int hip_mem_get_access_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(422 / 100 + '0');
    event->function_name[6] = (char)((422 / 10) % 10 + '0');
    event->function_name[7] = (char)(422 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // flags
    event->args[1] = PT_REGS_PARM2(ctx); // location
    event->args[2] = PT_REGS_PARM3(ctx); // ptr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_get_allocation_granularity_entry")
int hip_mem_get_allocation_granularity_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(423 / 100 + '0');
    event->function_name[6] = (char)((423 / 10) % 10 + '0');
    event->function_name[7] = (char)(423 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // granularity
    event->args[1] = PT_REGS_PARM2(ctx); // prop
    event->args[2] = PT_REGS_PARM3(ctx); // option
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_get_allocation_properties_from_handle_entry")
int hip_mem_get_allocation_properties_from_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(424 / 100 + '0');
    event->function_name[6] = (char)((424 / 10) % 10 + '0');
    event->function_name[7] = (char)(424 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // prop
    event->args[1] = PT_REGS_PARM2(ctx); // handle
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_import_from_shareable_handle_entry")
int hip_mem_import_from_shareable_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(425 / 100 + '0');
    event->function_name[6] = (char)((425 / 10) % 10 + '0');
    event->function_name[7] = (char)(425 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // handle
    event->args[1] = PT_REGS_PARM2(ctx); // osHandle
    event->args[2] = PT_REGS_PARM3(ctx); // shHandleType
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_map_entry")
int hip_mem_map_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(426 / 100 + '0');
    event->function_name[6] = (char)((426 / 10) % 10 + '0');
    event->function_name[7] = (char)(426 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 5;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // offset
    event->args[3] = PT_REGS_PARM4(ctx); // handle
    event->args[4] = PT_REGS_PARM5(ctx); // flags
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_map_array_async_entry")
int hip_mem_map_array_async_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(427 / 100 + '0');
    event->function_name[6] = (char)((427 / 10) % 10 + '0');
    event->function_name[7] = (char)(427 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // mapInfoList
    event->args[1] = PT_REGS_PARM2(ctx); // count
    event->args[2] = PT_REGS_PARM3(ctx); // stream
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_release_entry")
int hip_mem_release_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(428 / 100 + '0');
    event->function_name[6] = (char)((428 / 10) % 10 + '0');
    event->function_name[7] = (char)(428 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // handle
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_retain_allocation_handle_entry")
int hip_mem_retain_allocation_handle_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(429 / 100 + '0');
    event->function_name[6] = (char)((429 / 10) % 10 + '0');
    event->function_name[7] = (char)(429 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // handle
    event->args[1] = PT_REGS_PARM2(ctx); // addr
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_set_access_entry")
int hip_mem_set_access_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(430 / 100 + '0');
    event->function_name[6] = (char)((430 / 10) % 10 + '0');
    event->function_name[7] = (char)(430 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // desc
    event->args[3] = PT_REGS_PARM4(ctx); // count
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_mem_unmap_entry")
int hip_mem_unmap_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(431 / 100 + '0');
    event->function_name[6] = (char)((431 / 10) % 10 + '0');
    event->function_name[7] = (char)(431 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // ptr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graphics_map_resources_entry")
int hip_graphics_map_resources_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(432 / 100 + '0');
    event->function_name[6] = (char)((432 / 10) % 10 + '0');
    event->function_name[7] = (char)(432 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // count
    event->args[1] = PT_REGS_PARM2(ctx); // resources
    event->args[2] = PT_REGS_PARM3(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graphics_sub_resource_get_mapped_array_entry")
int hip_graphics_sub_resource_get_mapped_array_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(433 / 100 + '0');
    event->function_name[6] = (char)((433 / 10) % 10 + '0');
    event->function_name[7] = (char)(433 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 4;
    event->args[0] = PT_REGS_PARM1(ctx); // array
    event->args[1] = PT_REGS_PARM2(ctx); // resource
    event->args[2] = PT_REGS_PARM3(ctx); // arrayIndex
    event->args[3] = PT_REGS_PARM4(ctx); // mipLevel
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graphics_resource_get_mapped_pointer_entry")
int hip_graphics_resource_get_mapped_pointer_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(434 / 100 + '0');
    event->function_name[6] = (char)((434 / 10) % 10 + '0');
    event->function_name[7] = (char)(434 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // devPtr
    event->args[1] = PT_REGS_PARM2(ctx); // size
    event->args[2] = PT_REGS_PARM3(ctx); // resource
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graphics_unmap_resources_entry")
int hip_graphics_unmap_resources_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(435 / 100 + '0');
    event->function_name[6] = (char)((435 / 10) % 10 + '0');
    event->function_name[7] = (char)(435 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 3;
    event->args[0] = PT_REGS_PARM1(ctx); // count
    event->args[1] = PT_REGS_PARM2(ctx); // resources
    event->args[2] = PT_REGS_PARM3(ctx); // __dparm(0)
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_graphics_unregister_resource_entry")
int hip_graphics_unregister_resource_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(436 / 100 + '0');
    event->function_name[6] = (char)((436 / 10) % 10 + '0');
    event->function_name[7] = (char)(436 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // resource
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_create_surface_object_entry")
int hip_create_surface_object_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(437 / 100 + '0');
    event->function_name[6] = (char)((437 / 10) % 10 + '0');
    event->function_name[7] = (char)(437 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 2;
    event->args[0] = PT_REGS_PARM1(ctx); // pSurfObject
    event->args[1] = PT_REGS_PARM2(ctx); // pResDesc
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uprobe/hip_destroy_surface_object_entry")
int hip_destroy_surface_object_entry(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    push_call_stack(tid, timestamp, function_id);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;

    // Set function name character by character to avoid string literals
    event->function_name[0] = 'f';
    event->function_name[1] = 'u';
    event->function_name[2] = 'n';
    event->function_name[3] = 'c';
    event->function_name[4] = '_';
    event->function_name[5] = (char)(438 / 100 + '0');
    event->function_name[6] = (char)((438 / 10) % 10 + '0');
    event->function_name[7] = (char)(438 % 10 + '0');
    event->function_name[8] = 0;

    event->arg_count = 1;
    event->args[0] = PT_REGS_PARM1(ctx); // surfaceObject
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

SEC("uretprobe/hip_init_exit")
int hip_init_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_driver_get_version_exit")
int hip_driver_get_version_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_runtime_get_version_exit")
int hip_runtime_get_version_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_exit")
int hip_device_get_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_compute_capability_exit")
int hip_device_compute_capability_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_name_exit")
int hip_device_get_name_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_uuid_exit")
int hip_device_get_uuid_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_p2_p_attribute_exit")
int hip_device_get_p2_p_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_p_c_i_bus_id_exit")
int hip_device_get_p_c_i_bus_id_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_by_p_c_i_bus_id_exit")
int hip_device_get_by_p_c_i_bus_id_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_total_mem_exit")
int hip_device_total_mem_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_synchronize_exit")
int hip_device_synchronize_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_reset_exit")
int hip_device_reset_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_set_device_exit")
int hip_set_device_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_set_valid_devices_exit")
int hip_set_valid_devices_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_device_exit")
int hip_get_device_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_device_count_exit")
int hip_get_device_count_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_attribute_exit")
int hip_device_get_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_default_mem_pool_exit")
int hip_device_get_default_mem_pool_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_set_mem_pool_exit")
int hip_device_set_mem_pool_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_mem_pool_exit")
int hip_device_get_mem_pool_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_device_properties_exit")
int hip_get_device_properties_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_texture1_d_linear_max_width_exit")
int hip_device_get_texture1_d_linear_max_width_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_set_cache_config_exit")
int hip_device_set_cache_config_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_cache_config_exit")
int hip_device_get_cache_config_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_limit_exit")
int hip_device_get_limit_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_set_limit_exit")
int hip_device_set_limit_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_shared_mem_config_exit")
int hip_device_get_shared_mem_config_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_device_flags_exit")
int hip_get_device_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_set_shared_mem_config_exit")
int hip_device_set_shared_mem_config_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_set_device_flags_exit")
int hip_set_device_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_choose_device_exit")
int hip_choose_device_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ipc_get_mem_handle_exit")
int hip_ipc_get_mem_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ipc_open_mem_handle_exit")
int hip_ipc_open_mem_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ipc_close_mem_handle_exit")
int hip_ipc_close_mem_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ipc_get_event_handle_exit")
int hip_ipc_get_event_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ipc_open_event_handle_exit")
int hip_ipc_open_event_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_func_set_attribute_exit")
int hip_func_set_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_func_set_cache_config_exit")
int hip_func_set_cache_config_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_func_set_shared_mem_config_exit")
int hip_func_set_shared_mem_config_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_last_error_exit")
int hip_get_last_error_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_peek_at_last_error_exit")
int hip_peek_at_last_error_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_create_exit")
int hip_stream_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_create_with_flags_exit")
int hip_stream_create_with_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_create_with_priority_exit")
int hip_stream_create_with_priority_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_stream_priority_range_exit")
int hip_device_get_stream_priority_range_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_destroy_exit")
int hip_stream_destroy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_query_exit")
int hip_stream_query_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_synchronize_exit")
int hip_stream_synchronize_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_wait_event_exit")
int hip_stream_wait_event_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_get_flags_exit")
int hip_stream_get_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_get_id_exit")
int hip_stream_get_id_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_get_priority_exit")
int hip_stream_get_priority_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_get_device_exit")
int hip_stream_get_device_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ext_stream_create_with_c_u_mask_exit")
int hip_ext_stream_create_with_c_u_mask_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ext_stream_get_c_u_mask_exit")
int hip_ext_stream_get_c_u_mask_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_add_callback_exit")
int hip_stream_add_callback_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_set_attribute_exit")
int hip_stream_set_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_get_attribute_exit")
int hip_stream_get_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_wait_value32_exit")
int hip_stream_wait_value32_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_wait_value64_exit")
int hip_stream_wait_value64_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_write_value32_exit")
int hip_stream_write_value32_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_write_value64_exit")
int hip_stream_write_value64_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_batch_mem_op_exit")
int hip_stream_batch_mem_op_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_batch_mem_op_node_exit")
int hip_graph_add_batch_mem_op_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_batch_mem_op_node_get_params_exit")
int hip_graph_batch_mem_op_node_get_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_batch_mem_op_node_set_params_exit")
int hip_graph_batch_mem_op_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_batch_mem_op_node_set_params_exit")
int hip_graph_exec_batch_mem_op_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_event_create_with_flags_exit")
int hip_event_create_with_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_event_create_exit")
int hip_event_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_event_record_with_flags_exit")
int hip_event_record_with_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_event_record_exit")
int hip_event_record_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_event_destroy_exit")
int hip_event_destroy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_event_synchronize_exit")
int hip_event_synchronize_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_event_elapsed_time_exit")
int hip_event_elapsed_time_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_event_query_exit")
int hip_event_query_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_pointer_set_attribute_exit")
int hip_pointer_set_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_pointer_get_attributes_exit")
int hip_pointer_get_attributes_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_pointer_get_attribute_exit")
int hip_pointer_get_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_pointer_get_attributes_exit")
int hip_drv_pointer_get_attributes_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_import_external_semaphore_exit")
int hip_import_external_semaphore_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_signal_external_semaphores_async_exit")
int hip_signal_external_semaphores_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_wait_external_semaphores_async_exit")
int hip_wait_external_semaphores_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_destroy_external_semaphore_exit")
int hip_destroy_external_semaphore_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_import_external_memory_exit")
int hip_import_external_memory_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_external_memory_get_mapped_buffer_exit")
int hip_external_memory_get_mapped_buffer_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_destroy_external_memory_exit")
int hip_destroy_external_memory_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_external_memory_get_mapped_mipmapped_array_exit")
int hip_external_memory_get_mapped_mipmapped_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_malloc_exit")
int hip_malloc_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ext_malloc_with_flags_exit")
int hip_ext_malloc_with_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_malloc_host_exit")
int hip_malloc_host_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_alloc_host_exit")
int hip_mem_alloc_host_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_host_malloc_exit")
int hip_host_malloc_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_malloc_managed_exit")
int hip_malloc_managed_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_prefetch_async_exit")
int hip_mem_prefetch_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_prefetch_async_v2_exit")
int hip_mem_prefetch_async_v2_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_advise_exit")
int hip_mem_advise_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_advise_v2_exit")
int hip_mem_advise_v2_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_range_get_attribute_exit")
int hip_mem_range_get_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_range_get_attributes_exit")
int hip_mem_range_get_attributes_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_attach_mem_async_exit")
int hip_stream_attach_mem_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_malloc_async_exit")
int hip_malloc_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_free_async_exit")
int hip_free_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_trim_to_exit")
int hip_mem_pool_trim_to_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_set_attribute_exit")
int hip_mem_pool_set_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_get_attribute_exit")
int hip_mem_pool_get_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_set_access_exit")
int hip_mem_pool_set_access_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_get_access_exit")
int hip_mem_pool_get_access_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_create_exit")
int hip_mem_pool_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_destroy_exit")
int hip_mem_pool_destroy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_malloc_from_pool_async_exit")
int hip_malloc_from_pool_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_export_to_shareable_handle_exit")
int hip_mem_pool_export_to_shareable_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_import_from_shareable_handle_exit")
int hip_mem_pool_import_from_shareable_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_export_pointer_exit")
int hip_mem_pool_export_pointer_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_pool_import_pointer_exit")
int hip_mem_pool_import_pointer_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_host_alloc_exit")
int hip_host_alloc_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_host_get_device_pointer_exit")
int hip_host_get_device_pointer_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_host_get_flags_exit")
int hip_host_get_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_host_register_exit")
int hip_host_register_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_host_unregister_exit")
int hip_host_unregister_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_malloc_pitch_exit")
int hip_malloc_pitch_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_alloc_pitch_exit")
int hip_mem_alloc_pitch_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_free_exit")
int hip_free_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_free_host_exit")
int hip_free_host_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_host_free_exit")
int hip_host_free_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_exit")
int hip_memcpy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_with_stream_exit")
int hip_memcpy_with_stream_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_hto_d_exit")
int hip_memcpy_hto_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_dto_h_exit")
int hip_memcpy_dto_h_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_dto_d_exit")
int hip_memcpy_dto_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_ato_d_exit")
int hip_memcpy_ato_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_dto_a_exit")
int hip_memcpy_dto_a_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_ato_a_exit")
int hip_memcpy_ato_a_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_hto_d_async_exit")
int hip_memcpy_hto_d_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_dto_h_async_exit")
int hip_memcpy_dto_h_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_dto_d_async_exit")
int hip_memcpy_dto_d_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_ato_h_async_exit")
int hip_memcpy_ato_h_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_hto_a_async_exit")
int hip_memcpy_hto_a_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_get_global_exit")
int hip_module_get_global_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_symbol_address_exit")
int hip_get_symbol_address_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_symbol_size_exit")
int hip_get_symbol_size_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_proc_address_exit")
int hip_get_proc_address_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_to_symbol_exit")
int hip_memcpy_to_symbol_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_to_symbol_async_exit")
int hip_memcpy_to_symbol_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_from_symbol_exit")
int hip_memcpy_from_symbol_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_from_symbol_async_exit")
int hip_memcpy_from_symbol_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_async_exit")
int hip_memcpy_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_exit")
int hip_memset_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d8_exit")
int hip_memset_d8_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d8_async_exit")
int hip_memset_d8_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d16_exit")
int hip_memset_d16_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d16_async_exit")
int hip_memset_d16_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d32_exit")
int hip_memset_d32_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_async_exit")
int hip_memset_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d32_async_exit")
int hip_memset_d32_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset2_d_exit")
int hip_memset2_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset2_d_async_exit")
int hip_memset2_d_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset3_d_exit")
int hip_memset3_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset3_d_async_exit")
int hip_memset3_d_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d2_d8_exit")
int hip_memset_d2_d8_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d2_d8_async_exit")
int hip_memset_d2_d8_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d2_d16_exit")
int hip_memset_d2_d16_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d2_d16_async_exit")
int hip_memset_d2_d16_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d2_d32_exit")
int hip_memset_d2_d32_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memset_d2_d32_async_exit")
int hip_memset_d2_d32_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_get_info_exit")
int hip_mem_get_info_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_ptr_get_info_exit")
int hip_mem_ptr_get_info_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_malloc_array_exit")
int hip_malloc_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_array_create_exit")
int hip_array_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_array_destroy_exit")
int hip_array_destroy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_array3_d_create_exit")
int hip_array3_d_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_malloc3_d_exit")
int hip_malloc3_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_free_array_exit")
int hip_free_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_malloc3_d_array_exit")
int hip_malloc3_d_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_array_get_info_exit")
int hip_array_get_info_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_array_get_descriptor_exit")
int hip_array_get_descriptor_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_array3_d_get_descriptor_exit")
int hip_array3_d_get_descriptor_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy2_d_exit")
int hip_memcpy2_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_param2_d_exit")
int hip_memcpy_param2_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_param2_d_async_exit")
int hip_memcpy_param2_d_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy2_d_async_exit")
int hip_memcpy2_d_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy2_d_to_array_exit")
int hip_memcpy2_d_to_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy2_d_to_array_async_exit")
int hip_memcpy2_d_to_array_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy2_d_array_to_array_exit")
int hip_memcpy2_d_array_to_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_to_array_exit")
int hip_memcpy_to_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_from_array_exit")
int hip_memcpy_from_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy2_d_from_array_exit")
int hip_memcpy2_d_from_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy2_d_from_array_async_exit")
int hip_memcpy2_d_from_array_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_ato_h_exit")
int hip_memcpy_ato_h_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_hto_a_exit")
int hip_memcpy_hto_a_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy3_d_exit")
int hip_memcpy3_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy3_d_async_exit")
int hip_memcpy3_d_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_memcpy3_d_exit")
int hip_drv_memcpy3_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_memcpy3_d_async_exit")
int hip_drv_memcpy3_d_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_get_address_range_exit")
int hip_mem_get_address_range_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_batch_async_exit")
int hip_memcpy_batch_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy3_d_batch_async_exit")
int hip_memcpy3_d_batch_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy3_d_peer_exit")
int hip_memcpy3_d_peer_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy3_d_peer_async_exit")
int hip_memcpy3_d_peer_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_can_access_peer_exit")
int hip_device_can_access_peer_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_enable_peer_access_exit")
int hip_device_enable_peer_access_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_disable_peer_access_exit")
int hip_device_disable_peer_access_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_peer_exit")
int hip_memcpy_peer_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_memcpy_peer_async_exit")
int hip_memcpy_peer_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_create_exit")
int hip_ctx_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_destroy_exit")
int hip_ctx_destroy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_pop_current_exit")
int hip_ctx_pop_current_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_push_current_exit")
int hip_ctx_push_current_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_set_current_exit")
int hip_ctx_set_current_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_get_current_exit")
int hip_ctx_get_current_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_get_device_exit")
int hip_ctx_get_device_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_get_api_version_exit")
int hip_ctx_get_api_version_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_get_cache_config_exit")
int hip_ctx_get_cache_config_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_set_cache_config_exit")
int hip_ctx_set_cache_config_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_set_shared_mem_config_exit")
int hip_ctx_set_shared_mem_config_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_get_shared_mem_config_exit")
int hip_ctx_get_shared_mem_config_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_synchronize_exit")
int hip_ctx_synchronize_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_get_flags_exit")
int hip_ctx_get_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_enable_peer_access_exit")
int hip_ctx_enable_peer_access_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ctx_disable_peer_access_exit")
int hip_ctx_disable_peer_access_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_primary_ctx_get_state_exit")
int hip_device_primary_ctx_get_state_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_primary_ctx_release_exit")
int hip_device_primary_ctx_release_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_primary_ctx_retain_exit")
int hip_device_primary_ctx_retain_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_primary_ctx_reset_exit")
int hip_device_primary_ctx_reset_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_primary_ctx_set_flags_exit")
int hip_device_primary_ctx_set_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_load_fat_binary_exit")
int hip_module_load_fat_binary_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_load_exit")
int hip_module_load_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_unload_exit")
int hip_module_unload_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_get_function_exit")
int hip_module_get_function_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_get_function_count_exit")
int hip_module_get_function_count_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_func_get_attributes_exit")
int hip_func_get_attributes_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_func_get_attribute_exit")
int hip_func_get_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_func_by_symbol_exit")
int hip_get_func_by_symbol_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_driver_entry_point_exit")
int hip_get_driver_entry_point_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_get_tex_ref_exit")
int hip_module_get_tex_ref_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_load_data_exit")
int hip_module_load_data_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_load_data_ex_exit")
int hip_module_load_data_ex_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_link_add_data_exit")
int hip_link_add_data_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_link_add_file_exit")
int hip_link_add_file_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_link_complete_exit")
int hip_link_complete_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_link_create_exit")
int hip_link_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_link_destroy_exit")
int hip_link_destroy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_launch_kernel_exit")
int hip_module_launch_kernel_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_launch_cooperative_kernel_exit")
int hip_module_launch_cooperative_kernel_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_launch_cooperative_kernel_multi_device_exit")
int hip_module_launch_cooperative_kernel_multi_device_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_launch_cooperative_kernel_exit")
int hip_launch_cooperative_kernel_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_launch_cooperative_kernel_multi_device_exit")
int hip_launch_cooperative_kernel_multi_device_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ext_launch_multi_kernel_multi_device_exit")
int hip_ext_launch_multi_kernel_multi_device_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_launch_kernel_ex_c_exit")
int hip_launch_kernel_ex_c_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_launch_kernel_ex_exit")
int hip_drv_launch_kernel_ex_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_get_handle_for_address_range_exit")
int hip_mem_get_handle_for_address_range_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_occupancy_max_potential_block_size_exit")
int hip_module_occupancy_max_potential_block_size_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_occupancy_max_potential_block_size_with_flags_exit")
int hip_module_occupancy_max_potential_block_size_with_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_occupancy_max_active_blocks_per_multiprocessor_exit")
int hip_module_occupancy_max_active_blocks_per_multiprocessor_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_module_occupancy_max_active_blocks_per_multiprocessor_with_flags_exit")
int hip_module_occupancy_max_active_blocks_per_multiprocessor_with_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_occupancy_max_active_blocks_per_multiprocessor_exit")
int hip_occupancy_max_active_blocks_per_multiprocessor_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_occupancy_max_active_blocks_per_multiprocessor_with_flags_exit")
int hip_occupancy_max_active_blocks_per_multiprocessor_with_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_occupancy_max_potential_block_size_exit")
int hip_occupancy_max_potential_block_size_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_profiler_start_exit")
int hip_profiler_start_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_profiler_stop_exit")
int hip_profiler_stop_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_configure_call_exit")
int hip_configure_call_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_setup_argument_exit")
int hip_setup_argument_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_launch_by_ptr_exit")
int hip_launch_by_ptr_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_launch_kernel_exit")
int hip_launch_kernel_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_launch_host_func_exit")
int hip_launch_host_func_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_memcpy2_d_unaligned_exit")
int hip_drv_memcpy2_d_unaligned_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_ext_launch_kernel_exit")
int hip_ext_launch_kernel_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_create_texture_object_exit")
int hip_create_texture_object_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_destroy_texture_object_exit")
int hip_destroy_texture_object_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_channel_desc_exit")
int hip_get_channel_desc_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_texture_object_resource_desc_exit")
int hip_get_texture_object_resource_desc_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_texture_object_resource_view_desc_exit")
int hip_get_texture_object_resource_view_desc_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_texture_object_texture_desc_exit")
int hip_get_texture_object_texture_desc_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_object_create_exit")
int hip_tex_object_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_object_destroy_exit")
int hip_tex_object_destroy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_object_get_resource_desc_exit")
int hip_tex_object_get_resource_desc_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_object_get_resource_view_desc_exit")
int hip_tex_object_get_resource_view_desc_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_object_get_texture_desc_exit")
int hip_tex_object_get_texture_desc_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_malloc_mipmapped_array_exit")
int hip_malloc_mipmapped_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_free_mipmapped_array_exit")
int hip_free_mipmapped_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_mipmapped_array_level_exit")
int hip_get_mipmapped_array_level_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mipmapped_array_create_exit")
int hip_mipmapped_array_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mipmapped_array_destroy_exit")
int hip_mipmapped_array_destroy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mipmapped_array_get_level_exit")
int hip_mipmapped_array_get_level_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_bind_texture_to_mipmapped_array_exit")
int hip_bind_texture_to_mipmapped_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_texture_reference_exit")
int hip_get_texture_reference_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_border_color_exit")
int hip_tex_ref_get_border_color_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_array_exit")
int hip_tex_ref_get_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_address_mode_exit")
int hip_tex_ref_set_address_mode_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_array_exit")
int hip_tex_ref_set_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_filter_mode_exit")
int hip_tex_ref_set_filter_mode_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_flags_exit")
int hip_tex_ref_set_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_format_exit")
int hip_tex_ref_set_format_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_bind_texture_exit")
int hip_bind_texture_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_bind_texture2_d_exit")
int hip_bind_texture2_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_bind_texture_to_array_exit")
int hip_bind_texture_to_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_get_texture_alignment_offset_exit")
int hip_get_texture_alignment_offset_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_unbind_texture_exit")
int hip_unbind_texture_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_address_exit")
int hip_tex_ref_get_address_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_address_mode_exit")
int hip_tex_ref_get_address_mode_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_filter_mode_exit")
int hip_tex_ref_get_filter_mode_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_flags_exit")
int hip_tex_ref_get_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_format_exit")
int hip_tex_ref_get_format_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_max_anisotropy_exit")
int hip_tex_ref_get_max_anisotropy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_mipmap_filter_mode_exit")
int hip_tex_ref_get_mipmap_filter_mode_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_mipmap_level_bias_exit")
int hip_tex_ref_get_mipmap_level_bias_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_mipmap_level_clamp_exit")
int hip_tex_ref_get_mipmap_level_clamp_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_get_mip_mapped_array_exit")
int hip_tex_ref_get_mip_mapped_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_address_exit")
int hip_tex_ref_set_address_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_address2_d_exit")
int hip_tex_ref_set_address2_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_max_anisotropy_exit")
int hip_tex_ref_set_max_anisotropy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_border_color_exit")
int hip_tex_ref_set_border_color_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_mipmap_filter_mode_exit")
int hip_tex_ref_set_mipmap_filter_mode_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_mipmap_level_bias_exit")
int hip_tex_ref_set_mipmap_level_bias_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_mipmap_level_clamp_exit")
int hip_tex_ref_set_mipmap_level_clamp_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_tex_ref_set_mipmapped_array_exit")
int hip_tex_ref_set_mipmapped_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_begin_capture_exit")
int hip_stream_begin_capture_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_begin_capture_to_graph_exit")
int hip_stream_begin_capture_to_graph_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_end_capture_exit")
int hip_stream_end_capture_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_get_capture_info_exit")
int hip_stream_get_capture_info_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_get_capture_info_v2_exit")
int hip_stream_get_capture_info_v2_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_is_capturing_exit")
int hip_stream_is_capturing_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_stream_update_capture_dependencies_exit")
int hip_stream_update_capture_dependencies_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_thread_exchange_stream_capture_mode_exit")
int hip_thread_exchange_stream_capture_mode_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_create_exit")
int hip_graph_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_destroy_exit")
int hip_graph_destroy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_dependencies_exit")
int hip_graph_add_dependencies_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_remove_dependencies_exit")
int hip_graph_remove_dependencies_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_get_edges_exit")
int hip_graph_get_edges_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_get_nodes_exit")
int hip_graph_get_nodes_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_get_root_nodes_exit")
int hip_graph_get_root_nodes_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_node_get_dependencies_exit")
int hip_graph_node_get_dependencies_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_node_get_dependent_nodes_exit")
int hip_graph_node_get_dependent_nodes_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_node_get_type_exit")
int hip_graph_node_get_type_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_destroy_node_exit")
int hip_graph_destroy_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_clone_exit")
int hip_graph_clone_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_node_find_in_clone_exit")
int hip_graph_node_find_in_clone_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_instantiate_exit")
int hip_graph_instantiate_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_instantiate_with_flags_exit")
int hip_graph_instantiate_with_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_instantiate_with_params_exit")
int hip_graph_instantiate_with_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_launch_exit")
int hip_graph_launch_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_upload_exit")
int hip_graph_upload_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_node_exit")
int hip_graph_add_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_get_flags_exit")
int hip_graph_exec_get_flags_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_node_set_params_exit")
int hip_graph_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_node_set_params_exit")
int hip_graph_exec_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_destroy_exit")
int hip_graph_exec_destroy_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_update_exit")
int hip_graph_exec_update_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_kernel_node_exit")
int hip_graph_add_kernel_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_kernel_node_get_params_exit")
int hip_graph_kernel_node_get_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_kernel_node_set_params_exit")
int hip_graph_kernel_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_kernel_node_set_params_exit")
int hip_graph_exec_kernel_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_graph_add_memcpy_node_exit")
int hip_drv_graph_add_memcpy_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_memcpy_node_exit")
int hip_graph_add_memcpy_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_memcpy_node_get_params_exit")
int hip_graph_memcpy_node_get_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_memcpy_node_set_params_exit")
int hip_graph_memcpy_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_kernel_node_set_attribute_exit")
int hip_graph_kernel_node_set_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_kernel_node_get_attribute_exit")
int hip_graph_kernel_node_get_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_memcpy_node_set_params_exit")
int hip_graph_exec_memcpy_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_memcpy_node1_d_exit")
int hip_graph_add_memcpy_node1_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_memcpy_node_set_params1_d_exit")
int hip_graph_memcpy_node_set_params1_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_memcpy_node_set_params1_d_exit")
int hip_graph_exec_memcpy_node_set_params1_d_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_memcpy_node_from_symbol_exit")
int hip_graph_add_memcpy_node_from_symbol_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_memcpy_node_set_params_from_symbol_exit")
int hip_graph_memcpy_node_set_params_from_symbol_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_memcpy_node_set_params_from_symbol_exit")
int hip_graph_exec_memcpy_node_set_params_from_symbol_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_memcpy_node_to_symbol_exit")
int hip_graph_add_memcpy_node_to_symbol_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_memcpy_node_set_params_to_symbol_exit")
int hip_graph_memcpy_node_set_params_to_symbol_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_memcpy_node_set_params_to_symbol_exit")
int hip_graph_exec_memcpy_node_set_params_to_symbol_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_memset_node_exit")
int hip_graph_add_memset_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_memset_node_get_params_exit")
int hip_graph_memset_node_get_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_memset_node_set_params_exit")
int hip_graph_memset_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_memset_node_set_params_exit")
int hip_graph_exec_memset_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_host_node_exit")
int hip_graph_add_host_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_host_node_get_params_exit")
int hip_graph_host_node_get_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_host_node_set_params_exit")
int hip_graph_host_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_host_node_set_params_exit")
int hip_graph_exec_host_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_child_graph_node_exit")
int hip_graph_add_child_graph_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_child_graph_node_get_graph_exit")
int hip_graph_child_graph_node_get_graph_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_child_graph_node_set_params_exit")
int hip_graph_exec_child_graph_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_empty_node_exit")
int hip_graph_add_empty_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_event_record_node_exit")
int hip_graph_add_event_record_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_event_record_node_get_event_exit")
int hip_graph_event_record_node_get_event_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_event_record_node_set_event_exit")
int hip_graph_event_record_node_set_event_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_event_record_node_set_event_exit")
int hip_graph_exec_event_record_node_set_event_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_event_wait_node_exit")
int hip_graph_add_event_wait_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_event_wait_node_get_event_exit")
int hip_graph_event_wait_node_get_event_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_event_wait_node_set_event_exit")
int hip_graph_event_wait_node_set_event_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_event_wait_node_set_event_exit")
int hip_graph_exec_event_wait_node_set_event_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_mem_alloc_node_exit")
int hip_graph_add_mem_alloc_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_mem_alloc_node_get_params_exit")
int hip_graph_mem_alloc_node_get_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_mem_free_node_exit")
int hip_graph_add_mem_free_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_mem_free_node_get_params_exit")
int hip_graph_mem_free_node_get_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_get_graph_mem_attribute_exit")
int hip_device_get_graph_mem_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_set_graph_mem_attribute_exit")
int hip_device_set_graph_mem_attribute_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_device_graph_mem_trim_exit")
int hip_device_graph_mem_trim_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_user_object_create_exit")
int hip_user_object_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_user_object_release_exit")
int hip_user_object_release_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_user_object_retain_exit")
int hip_user_object_retain_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_retain_user_object_exit")
int hip_graph_retain_user_object_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_release_user_object_exit")
int hip_graph_release_user_object_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_debug_dot_print_exit")
int hip_graph_debug_dot_print_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_kernel_node_copy_attributes_exit")
int hip_graph_kernel_node_copy_attributes_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_node_set_enabled_exit")
int hip_graph_node_set_enabled_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_node_get_enabled_exit")
int hip_graph_node_get_enabled_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_external_semaphores_wait_node_exit")
int hip_graph_add_external_semaphores_wait_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_add_external_semaphores_signal_node_exit")
int hip_graph_add_external_semaphores_signal_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_external_semaphores_signal_node_set_params_exit")
int hip_graph_external_semaphores_signal_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_external_semaphores_wait_node_set_params_exit")
int hip_graph_external_semaphores_wait_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_external_semaphores_signal_node_get_params_exit")
int hip_graph_external_semaphores_signal_node_get_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_external_semaphores_wait_node_get_params_exit")
int hip_graph_external_semaphores_wait_node_get_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_external_semaphores_signal_node_set_params_exit")
int hip_graph_exec_external_semaphores_signal_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graph_exec_external_semaphores_wait_node_set_params_exit")
int hip_graph_exec_external_semaphores_wait_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_graph_memcpy_node_get_params_exit")
int hip_drv_graph_memcpy_node_get_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_graph_memcpy_node_set_params_exit")
int hip_drv_graph_memcpy_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_graph_add_memset_node_exit")
int hip_drv_graph_add_memset_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_graph_add_mem_free_node_exit")
int hip_drv_graph_add_mem_free_node_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_graph_exec_memcpy_node_set_params_exit")
int hip_drv_graph_exec_memcpy_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_drv_graph_exec_memset_node_set_params_exit")
int hip_drv_graph_exec_memset_node_set_params_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_address_free_exit")
int hip_mem_address_free_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_address_reserve_exit")
int hip_mem_address_reserve_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_create_exit")
int hip_mem_create_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_export_to_shareable_handle_exit")
int hip_mem_export_to_shareable_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_get_access_exit")
int hip_mem_get_access_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_get_allocation_granularity_exit")
int hip_mem_get_allocation_granularity_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_get_allocation_properties_from_handle_exit")
int hip_mem_get_allocation_properties_from_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_import_from_shareable_handle_exit")
int hip_mem_import_from_shareable_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_map_exit")
int hip_mem_map_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_map_array_async_exit")
int hip_mem_map_array_async_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_release_exit")
int hip_mem_release_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_retain_allocation_handle_exit")
int hip_mem_retain_allocation_handle_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_set_access_exit")
int hip_mem_set_access_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_mem_unmap_exit")
int hip_mem_unmap_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graphics_map_resources_exit")
int hip_graphics_map_resources_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graphics_sub_resource_get_mapped_array_exit")
int hip_graphics_sub_resource_get_mapped_array_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graphics_resource_get_mapped_pointer_exit")
int hip_graphics_resource_get_mapped_pointer_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graphics_unmap_resources_exit")
int hip_graphics_unmap_resources_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_graphics_unregister_resource_exit")
int hip_graphics_unregister_resource_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_create_surface_object_exit")
int hip_create_surface_object_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}

SEC("uretprobe/hip_destroy_surface_object_exit")
int hip_destroy_surface_object_exit(struct pt_regs *ctx) {
    return create_exit_event(ctx);
}
// ============================================================================
// Kernel Dispatch Tracing via AMDGPU Tracepoints
// ============================================================================

// Tracepoint structures based on /sys/kernel/debug/tracing/events/amdgpu/*/format
struct amdgpu_cs_ioctl_args {
    unsigned short common_type;
    unsigned char common_flags;
    unsigned char common_preempt_count;
    int common_pid;
    __u32 __data_loc_timeline;
    __u64 context;
    __u64 seqno;
    void *fence;
    __u32 __data_loc_ring;
    __u32 num_ibs;
};

struct amdgpu_sched_run_job_args {
    unsigned short common_type;
    unsigned char common_flags;
    unsigned char common_preempt_count;
    int common_pid;
    __u32 __data_loc_timeline;
    __u64 context;
    __u64 seqno;
    __u32 __data_loc_ring;
    __u32 num_ibs;
};

struct drm_sched_job_args {
    unsigned short common_type;
    unsigned char common_flags;
    unsigned char common_preempt_count;
    int common_pid;
    __u32 __data_loc_name;
    __u32 job_count;
    int hw_job_count;
    __u32 __data_loc_dev;
    __u64 fence_context;
    __u64 fence_seqno;
    __u64 client_id;
};

struct drm_sched_job_done_args {
    unsigned short common_type;
    unsigned char common_flags;
    unsigned char common_preempt_count;
    int common_pid;
    __u64 fence_context;
    __u64 fence_seqno;
};

// Tracepoint: amdgpu/amdgpu_cs_ioctl - Command submission (kernel dispatch initiation)
SEC("tracepoint/amdgpu/amdgpu_cs_ioctl")
int trace_amdgpu_cs_ioctl(struct amdgpu_cs_ioctl_args *ctx) {
    struct kernel_dispatch_event *event;

    event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event)
        return 0;

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = get_timestamp();
    get_pid_tid(&event->pid, &event->tid);
    event->event_type = 2; // kernel_dispatch

    __builtin_memcpy(event->event_name, "cs_ioctl", 9);

    // Read tracepoint fields directly
    event->fence_context = ctx->context;
    event->fence_seqno = ctx->seqno;
    event->num_ibs = ctx->num_ibs;

    // Read ring name (variable length string from __data_loc)
    // __data_loc contains offset and length encoded as: (length << 16) | offset
    __u32 ring_loc = ctx->__data_loc_ring;
    __u16 ring_offset = ring_loc & 0xFFFF;
    bpf_probe_read_str(event->ring_name, sizeof(event->ring_name),
                       (void *)ctx + ring_offset);

    event->job_count = 0;
    event->hw_job_count = 0;
    event->client_id = 0;
    event->device_name[0] = '\0';

    // Initialize duration fields
    event->start_timestamp = 0;
    event->duration = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

// Tracepoint: amdgpu/amdgpu_sched_run_job - Job scheduling (kernel execution start)
SEC("tracepoint/amdgpu/amdgpu_sched_run_job")
int trace_amdgpu_sched_run_job(struct amdgpu_sched_run_job_args *ctx) {
    struct kernel_dispatch_event *event;

    event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event)
        return 0;

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = get_timestamp();
    get_pid_tid(&event->pid, &event->tid);
    event->event_type = 2; // kernel_dispatch

    __builtin_memcpy(event->event_name, "sched_run_job", 14);

    // Read tracepoint fields directly
    event->fence_context = ctx->context;
    event->fence_seqno = ctx->seqno;
    event->num_ibs = ctx->num_ibs;

    // Read ring name (variable length string from __data_loc)
    __u32 ring_loc = ctx->__data_loc_ring;
    __u16 ring_offset = ring_loc & 0xFFFF;
    bpf_probe_read_str(event->ring_name, sizeof(event->ring_name),
                       (void *)ctx + ring_offset);

    event->job_count = 0;
    event->hw_job_count = 0;
    event->client_id = 0;
    event->device_name[0] = '\0';

    // Store start timestamp for duration calculation
    __u64 fence_key = (event->fence_context << 32) | event->fence_seqno;
    __u64 timestamp = event->timestamp;
    bpf_map_update_elem(&kernel_dispatch_starts, &fence_key, &timestamp, BPF_ANY);

    // Initialize duration fields
    event->start_timestamp = event->timestamp;
    event->duration = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

// Tracepoint: gpu_scheduler/drm_sched_job_run - DRM scheduler job execution
SEC("tracepoint/gpu_scheduler/drm_sched_job_run")
int trace_drm_sched_job_run(struct drm_sched_job_args *ctx) {
    struct kernel_dispatch_event *event;

    event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event)
        return 0;

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    event->timestamp = get_timestamp();
    get_pid_tid(&event->pid, &event->tid);
    event->event_type = 2; // kernel_dispatch

    __builtin_memcpy(event->event_name, "drm_sched_job_run", 18);

    // Read tracepoint fields directly
    event->fence_context = ctx->fence_context;
    event->fence_seqno = ctx->fence_seqno;
    event->job_count = ctx->job_count;
    event->hw_job_count = ctx->hw_job_count;
    event->client_id = ctx->client_id;

    // Read ring/device name (variable length strings from __data_loc)
    __u32 name_loc = ctx->__data_loc_name;
    __u16 name_offset = name_loc & 0xFFFF;
    bpf_probe_read_str(event->ring_name, sizeof(event->ring_name),
                       (void *)ctx + name_offset);

    __u32 dev_loc = ctx->__data_loc_dev;
    __u16 dev_offset = dev_loc & 0xFFFF;
    bpf_probe_read_str(event->device_name, sizeof(event->device_name),
                       (void *)ctx + dev_offset);

    event->num_ibs = 0;

    // Store start timestamp for duration calculation
    __u64 fence_key = (event->fence_context << 32) | event->fence_seqno;
    __u64 timestamp = event->timestamp;
    bpf_map_update_elem(&kernel_dispatch_starts, &fence_key, &timestamp, BPF_ANY);

    // Initialize duration fields
    event->start_timestamp = event->timestamp;
    event->duration = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}

// Tracepoint: gpu_scheduler/drm_sched_job_done - DRM scheduler job completion
SEC("tracepoint/gpu_scheduler/drm_sched_job_done")
int trace_drm_sched_job_done(struct drm_sched_job_done_args *ctx) {
    struct kernel_dispatch_event *event;

    event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event)
        return 0;

    // Initialize the entire event structure to zero first
    __builtin_memset(event, 0, sizeof(*event));

    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    event->timestamp = get_timestamp();
    event->pid = pid;
    event->tid = tid;
    event->event_type = 3; // kernel_completion
    // Set event name character by character to avoid string literals
    event->event_name[0] = 'd';
    event->event_name[1] = 'r';
    event->event_name[2] = 'm';
    event->event_name[3] = '_';
    event->event_name[4] = 's';
    event->event_name[5] = 'c';
    event->event_name[6] = 'h';
    event->event_name[7] = 'e';
    event->event_name[8] = 'd';
    event->event_name[9] = '_';
    event->event_name[10] = 'j';
    event->event_name[11] = 'o';
    event->event_name[12] = 'b';
    event->event_name[13] = '_';
    event->event_name[14] = 'd';
    event->event_name[15] = 'o';
    event->event_name[16] = 'n';
    event->event_name[17] = 'e';
    event->event_name[18] = '\0';

    event->fence_context = ctx->fence_context;
    event->fence_seqno = ctx->fence_seqno;
    event->ring_name[0] = '\0';
    event->num_ibs = 0;
    event->job_count = 0;
    event->hw_job_count = 0;
    event->client_id = 0;
    event->device_name[0] = '\0';

    // Calculate duration using fence context/sequence to find start time
    __u64 fence_key = (event->fence_context << 32) | event->fence_seqno;
    __u64 *start_timestamp = bpf_map_lookup_elem(&kernel_dispatch_starts, &fence_key);

    if (start_timestamp) {
        event->start_timestamp = *start_timestamp;
        event->duration = event->timestamp - *start_timestamp;
        // Clean up the start timestamp entry
        bpf_map_delete_elem(&kernel_dispatch_starts, &fence_key);
    } else {
        // No start time found, duration is 0
        event->start_timestamp = 0;
        event->duration = 0;
    }

    bpf_ringbuf_submit(event, 0);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
