// SPDX-License-Identifier: GPL-2.0
/*
 * HSA Hybrid Tracing Shim (Minimal LD_PRELOAD)
 *
 * This is a minimal in-process shim that:
 * 1. Reads queue pointers from eBPF maps (populated by out-of-process tracer)
 * 2. Installs write interceptors on queues
 * 3. Captures kernel dispatches and injects completion signals
 * 4. Uses async signal handlers to capture kernel timestamps
 *
 * Usage: LD_PRELOAD=./libhsa_hybrid_shim.so <app>
 * Requires: Out-of-process eBPF tracer must be running first
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/syscall.h>
#include <stdarg.h>
#include <bpf/bpf.h>
#include <bpf/libbpf.h>

#define AMD_INTERNAL_BUILD
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
/* Include hsa_api_trace.h for AmdExtTable */
#include <hsa/hsa_api_trace.h>

/*
 * ROCProfiler-SDK Registration (ONLY for HSA API table access)
 */
#define __HIP_PLATFORM_AMD__
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#define MAX_QUEUES 1024
#define MAX_SIGNALS 4096
#define MAX_KERNEL_NAME 256

// Event types (must match unified eBPF program)
#define EVENT_KERNEL_DISPATCH   2
#define EVENT_KERNEL_COMPLETE   3

// Kernel event structure to send to eBPF
typedef struct __attribute__((packed)) {
    uint64_t timestamp;
    uint32_t pid;
    uint32_t tid;
    uint32_t event_type;  // 2=dispatch, 3=complete
    uint32_t queue_id;
    uint64_t agent_id;    // HSA agent handle
    uint32_t correlation_id;  // Link to HIP API that triggered this dispatch
    char kernel_name[MAX_KERNEL_NAME];
    uint64_t kernel_object;
    uint32_t grid_size_x, grid_size_y, grid_size_z;
    uint32_t workgroup_size_x, workgroup_size_y, workgroup_size_z;
    uint32_t group_segment_size;
    uint32_t private_segment_size;
    uint64_t gpu_start_time;
    uint64_t gpu_end_time;
} kernel_event_t;

/* Forward declarations for HSA AMD extensions */
typedef void (*hsa_amd_queue_intercept_packet_writer)(const void* pkts, uint64_t pkt_count);
typedef void (*hsa_amd_queue_intercept_handler)(
    const void* pkts,
    uint64_t pkt_count,
    uint64_t user_pkt_index,
    void* data,
    hsa_amd_queue_intercept_packet_writer writer);

/* Function pointer types for dynamically loaded HSA AMD functions */
typedef hsa_status_t (*hsa_amd_profiling_set_profiler_enabled_fn)(hsa_queue_t* queue, int enable);
typedef hsa_status_t (*hsa_amd_queue_intercept_register_fn)(
    hsa_queue_t* queue,
    hsa_amd_queue_intercept_handler callback,
    void* user_data);
typedef hsa_status_t (*hsa_amd_queue_intercept_create_fn)(
    hsa_agent_t agent_handle,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t* source, void* data),
    void* data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t** queue);

/* Async signal handler type (returns bool - false means remove handler) */
typedef bool (*hsa_amd_signal_handler)(hsa_signal_value_t value, void* arg);
typedef hsa_status_t (*hsa_amd_signal_async_handler_fn)(
    hsa_signal_t signal,
    hsa_signal_condition_t cond,
    hsa_signal_value_t value,
    hsa_amd_signal_handler handler,
    void* arg);

/* Dispatch time function pointer (structure already defined in hsa_ext_amd.h) */
typedef hsa_status_t (*hsa_amd_profiling_get_dispatch_time_fn)(
    hsa_agent_t agent,
    hsa_signal_t signal,
    hsa_amd_profiling_dispatch_time_t* time);

/* Queue tracking structure */
typedef struct {
    hsa_queue_t* queue;
    hsa_agent_t agent;       // Agent this queue belongs to
    uint32_t queue_id;
    uint64_t create_time;
    int interceptor_installed;
    int active;
} queue_info_t;

/* Kernel name mapping structure */
typedef struct {
    uint64_t kernel_object;
    char kernel_name[MAX_KERNEL_NAME];
    int active;
} kernel_name_entry_t;

#define MAX_KERNEL_NAMES 4096
static kernel_name_entry_t g_kernel_names[MAX_KERNEL_NAMES];
static pthread_mutex_t g_kernel_names_mutex = PTHREAD_MUTEX_INITIALIZER;

/* Signal tracking for kernel dispatches */
typedef struct {
    hsa_signal_t signal;              // Our tracking signal
    hsa_signal_t original_signal;     // Original application signal
    hsa_agent_t agent;                // Agent for profiling API
    uint64_t dispatch_time;           // CPU timestamp at dispatch
    uint64_t write_index;
    uint32_t queue_id;
    uint32_t correlation_id;          // Correlation ID from eBPF (links to HIP API)
    char kernel_name[MAX_KERNEL_NAME];
    uint64_t kernel_object;           // Kernel object address
    uint32_t workgroup_size_x;        // Workgroup dimensions
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;
    uint32_t grid_size_x;             // Grid dimensions
    uint32_t grid_size_y;
    uint32_t grid_size_z;
    uint32_t private_segment_size;    // Scratch memory size
    uint32_t group_segment_size;      // LDS size
    int active;
    int has_original_signal;          // Whether there was an original signal
} signal_info_t;

/* Global state */
static queue_info_t g_queues[MAX_QUEUES];
static signal_info_t g_signals[MAX_SIGNALS];
static pthread_mutex_t g_queue_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t g_signal_mutex = PTHREAD_MUTEX_INITIALIZER;
static uint32_t g_next_queue_id = 0;
static int g_initialized = 0;
static int g_kernel_events_fd = -1;  // Ring buffer FD for sending kernel events to eBPF
static int g_correlation_map_fd = -1; // FD for correlation_map to read correlation IDs
static volatile int g_running = 1;
static uint64_t g_timestamp_frequency_hz = 0;  // HSA system timestamp frequency for conversion

/* Original HSA functions */
static hsa_status_t (*real_hsa_queue_create)(
    hsa_agent_t agent,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t *source, void *data),
    void *data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t **queue) = NULL;

static hsa_status_t (*real_hsa_signal_create)(
    hsa_signal_value_t initial_value,
    uint32_t num_consumers,
    const hsa_agent_t *consumers,
    hsa_signal_t *signal) = NULL;

static void (*real_hsa_signal_destroy)(hsa_signal_t signal) = NULL;

static void (*real_hsa_signal_store_screlease)(hsa_signal_t signal, hsa_signal_value_t value) = NULL;

/* HSA AMD extension functions */
static hsa_amd_profiling_set_profiler_enabled_fn real_hsa_amd_profiling_set_profiler_enabled = NULL;
static hsa_amd_queue_intercept_register_fn real_hsa_amd_queue_intercept_register = NULL;
static hsa_amd_queue_intercept_create_fn real_hsa_amd_queue_intercept_create = NULL;
static hsa_amd_signal_async_handler_fn real_hsa_amd_signal_async_handler = NULL;
static hsa_amd_profiling_get_dispatch_time_fn real_hsa_amd_profiling_get_dispatch_time = NULL;

/* HSA core functions for executable symbol iteration */
static hsa_status_t (*real_hsa_executable_freeze)(hsa_executable_t executable, const char* options) = NULL;
static hsa_status_t (*real_hsa_executable_iterate_agent_symbols)(
    hsa_executable_t executable,
    hsa_agent_t agent,
    hsa_status_t (*callback)(hsa_executable_t executable, hsa_agent_t agent, hsa_executable_symbol_t symbol, void* data),
    void* data) = NULL;
static hsa_status_t (*real_hsa_executable_symbol_get_info)(
    hsa_executable_symbol_t executable_symbol,
    hsa_executable_symbol_info_t attribute,
    void* value) = NULL;
static hsa_status_t (*real_hsa_iterate_agents)(
    hsa_status_t (*callback)(hsa_agent_t agent, void* data),
    void* data) = NULL;
static hsa_status_t (*real_hsa_system_get_info)(
    hsa_system_info_t attribute,
    void* value) = NULL;

/* Utility: Get timestamp in nanoseconds */
static inline uint64_t get_timestamp_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* Get correlation ID from eBPF map for current thread */
static inline uint32_t get_correlation_id_from_ebpf(void) {
    if (g_correlation_map_fd < 0) return 0;

    // Get current thread ID in the format eBPF uses (PID << 32 | TID)
    pid_t pid = getpid();
    pid_t tid = syscall(SYS_gettid);
    uint64_t key = ((uint64_t)pid << 32) | (uint64_t)tid;

    uint32_t correlation_id = 0;
    if (bpf_map_lookup_elem(g_correlation_map_fd, &key, &correlation_id) == 0) {
        return correlation_id;
    }

    return 0; // No correlation ID found (not in a traced HIP API call)
}

/* Convert HSA system clock ticks to nanoseconds */
static inline uint64_t hsa_ticks_to_ns(uint64_t ticks) {
    static int freq_init_attempted = 0;

    // Lazy initialization of timestamp frequency
    if (g_timestamp_frequency_hz == 0 && !freq_init_attempted) {
        freq_init_attempted = 1;  // Only try once

        if (real_hsa_system_get_info) {
            hsa_status_t status = real_hsa_system_get_info(
                HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &g_timestamp_frequency_hz);
            if (status != HSA_STATUS_SUCCESS || g_timestamp_frequency_hz == 0) {
                // Set to 1 to indicate failure
                g_timestamp_frequency_hz = 1;
            }
        } else {
            g_timestamp_frequency_hz = 1;
        }
    }

    if (g_timestamp_frequency_hz <= 1) {
        // Frequency query failed or unavailable - return raw ticks
        return ticks;
    }

    // Convert: ns = (ticks * 1000000000) / frequency_hz
    return (uint64_t)((double)ticks * 1000000000.0 / (double)g_timestamp_frequency_hz);
}

/* Register a kernel name mapping */
static void register_kernel_name(uint64_t kernel_object, const char* kernel_name) {
    if (!kernel_name || kernel_object == 0) return;
    
    pthread_mutex_lock(&g_kernel_names_mutex);
    
    // Look for existing entry first
    for (int i = 0; i < MAX_KERNEL_NAMES; i++) {
        if (g_kernel_names[i].active && g_kernel_names[i].kernel_object == kernel_object) {
            // Update existing entry
            strncpy(g_kernel_names[i].kernel_name, kernel_name, MAX_KERNEL_NAME - 1);
            g_kernel_names[i].kernel_name[MAX_KERNEL_NAME - 1] = '\0';
            pthread_mutex_unlock(&g_kernel_names_mutex);
            return;
        }
    }
    
    // Find empty slot
    for (int i = 0; i < MAX_KERNEL_NAMES; i++) {
        if (!g_kernel_names[i].active) {
            g_kernel_names[i].kernel_object = kernel_object;
            strncpy(g_kernel_names[i].kernel_name, kernel_name, MAX_KERNEL_NAME - 1);
            g_kernel_names[i].kernel_name[MAX_KERNEL_NAME - 1] = '\0';
            g_kernel_names[i].active = 1;
            pthread_mutex_unlock(&g_kernel_names_mutex);
            return;
        }
    }

    pthread_mutex_unlock(&g_kernel_names_mutex);
}

/* Look up kernel name by kernel object */
static const char* lookup_kernel_name(uint64_t kernel_object) {
    if (kernel_object == 0) return NULL;
    
    pthread_mutex_lock(&g_kernel_names_mutex);
    for (int i = 0; i < MAX_KERNEL_NAMES; i++) {
        if (g_kernel_names[i].active && g_kernel_names[i].kernel_object == kernel_object) {
            pthread_mutex_unlock(&g_kernel_names_mutex);
            return g_kernel_names[i].kernel_name;
        }
    }
    pthread_mutex_unlock(&g_kernel_names_mutex);
    return NULL;
}

/* HSA executable symbol iteration callback */
static hsa_status_t executable_symbol_callback(
    hsa_executable_t executable,
    hsa_agent_t agent,
    hsa_executable_symbol_t symbol,
    void* data)
{
    (void)executable;
    (void)agent;
    (void)data;

    if (!real_hsa_executable_symbol_get_info) {
        return HSA_STATUS_SUCCESS;
    }

    // Check if this is a kernel symbol
    hsa_symbol_kind_t symbol_type;
    hsa_status_t status = real_hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_TYPE, &symbol_type);
    
    if (status != HSA_STATUS_SUCCESS || symbol_type != HSA_SYMBOL_KIND_KERNEL) {
        return HSA_STATUS_SUCCESS;
    }

    // Get kernel object address
    uint64_t kernel_object = 0;
    status = real_hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kernel_object);
    
    if (status != HSA_STATUS_SUCCESS || kernel_object == 0) {
        return HSA_STATUS_SUCCESS;
    }

    // Get kernel name length
    uint32_t name_length = 0;
    status = real_hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME_LENGTH, &name_length);
    
    if (status != HSA_STATUS_SUCCESS || name_length == 0 || name_length >= MAX_KERNEL_NAME) {
        return HSA_STATUS_SUCCESS;
    }

    // Get kernel name
    char kernel_name[MAX_KERNEL_NAME];
    status = real_hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_NAME, kernel_name);
    
    if (status == HSA_STATUS_SUCCESS) {
        kernel_name[name_length] = '\0';  // Ensure null termination
        register_kernel_name(kernel_object, kernel_name);
    }

    return HSA_STATUS_SUCCESS;
}

/* Find queue by pointer */
static queue_info_t* find_queue_by_pointer(hsa_queue_t* queue) {
    pthread_mutex_lock(&g_queue_mutex);
    for (int i = 0; i < MAX_QUEUES; i++) {
        if (g_queues[i].active && g_queues[i].queue == queue) {
            pthread_mutex_unlock(&g_queue_mutex);
            return &g_queues[i];
        }
    }
    pthread_mutex_unlock(&g_queue_mutex);
    return NULL;
}

/* Allocate queue info */
static queue_info_t* allocate_queue_info(hsa_queue_t* queue) {
    pthread_mutex_lock(&g_queue_mutex);
    for (int i = 0; i < MAX_QUEUES; i++) {
        if (!g_queues[i].active) {
            g_queues[i].active = 1;
            g_queues[i].queue = queue;
            g_queues[i].queue_id = g_next_queue_id++;
            g_queues[i].create_time = get_timestamp_ns();
            g_queues[i].interceptor_installed = 0;
            pthread_mutex_unlock(&g_queue_mutex);
            return &g_queues[i];
        }
    }
    pthread_mutex_unlock(&g_queue_mutex);
    return NULL;
}

/* Allocate signal tracking */
static signal_info_t* allocate_signal_info(void) {
    pthread_mutex_lock(&g_signal_mutex);
    for (int i = 0; i < MAX_SIGNALS; i++) {
        if (!g_signals[i].active) {
            g_signals[i].active = 1;
            pthread_mutex_unlock(&g_signal_mutex);
            return &g_signals[i];
        }
    }
    pthread_mutex_unlock(&g_signal_mutex);
    return NULL;
}

/* Extract kernel name from AQL packet */
static void extract_kernel_name(const hsa_kernel_dispatch_packet_t* pkt, char* buf, size_t len) {
    if (pkt && pkt->kernel_object) {
        const char* kernel_name = lookup_kernel_name(pkt->kernel_object);
        if (kernel_name) {
            snprintf(buf, len, "%s", kernel_name);
        } else {
            snprintf(buf, len, "kernel_0x%llx", (unsigned long long)pkt->kernel_object);
        }
    } else {
        snprintf(buf, len, "unknown_kernel");
    }
}

/* Send kernel event to eBPF ring buffer */
static void send_kernel_event(kernel_event_t* event) {
    if (g_kernel_events_fd < 0) {
        static int warned = 0;
        if (!warned) {
            fprintf(stderr, "[SHIM] WARNING: kernel_events_fd not open, cannot send events\n");
            warned = 1;
        }
        return;
    }

    // CRITICAL FIX: Use per-CPU array with rotating slot index
    // This reduces event overwrites compared to single-slot
    static __thread uint32_t slot_index = 0;
    const uint32_t MAX_SLOTS = 16;
    uint32_t key = slot_index % MAX_SLOTS;
    slot_index++;

    int ret = bpf_map_update_elem(g_kernel_events_fd, &key, event, BPF_ANY);
    if (ret != 0) {
        static int err_count = 0;
        if (err_count < 5) {
            fprintf(stderr, "[SHIM] ERROR: Failed to send kernel event (type=%u, slot=%u): %s\n",
                    event->event_type, key, strerror(errno));
            err_count++;
        }
    }
}

/* Async Signal Handler Callback - called when kernel completes */
static bool async_signal_handler(hsa_signal_value_t value, void* data) {
    signal_info_t* sinfo = (signal_info_t*)data;
    (void)value;

    if (!sinfo || !sinfo->active) {
        return false;
    }

    /* Get accurate GPU-side timestamps using HSA profiling API */
    uint64_t gpu_start_time = sinfo->dispatch_time;
    uint64_t gpu_end_time = get_timestamp_ns();

    if (real_hsa_amd_profiling_get_dispatch_time) {
        hsa_amd_profiling_dispatch_time_t dispatch_time = {0};
        hsa_status_t status = real_hsa_amd_profiling_get_dispatch_time(
            sinfo->agent, sinfo->signal, &dispatch_time);

        if (status == HSA_STATUS_SUCCESS) {
            // Convert HSA system clock ticks to nanoseconds
            gpu_start_time = hsa_ticks_to_ns(dispatch_time.start);
            gpu_end_time = hsa_ticks_to_ns(dispatch_time.end);
        }
    }

    /* Forward to original signal if exists */
    if (sinfo->has_original_signal && sinfo->original_signal.handle != 0) {
        static void (*real_hsa_signal_subtract)(hsa_signal_t, hsa_signal_value_t) = NULL;
        if (!real_hsa_signal_subtract) {
            void* hsa_lib = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
            if (hsa_lib) {
                real_hsa_signal_subtract = (decltype(real_hsa_signal_subtract))dlsym(hsa_lib, "hsa_signal_subtract_screlease");
            }
        }
        if (real_hsa_signal_subtract) {
            real_hsa_signal_subtract(sinfo->original_signal, 1);
        }
    }

    /* Send kernel completion event to eBPF */
    kernel_event_t event = {0};
    event.timestamp = get_timestamp_ns();
    event.pid = getpid();
    event.tid = sinfo->queue_id; // Use queue_id as TID for grouping
    event.event_type = EVENT_KERNEL_COMPLETE;
    event.queue_id = sinfo->queue_id;
    event.agent_id = sinfo->agent.handle;  // Store agent ID for categorization
    event.correlation_id = sinfo->correlation_id;  // Link to HIP API
    strncpy(event.kernel_name, sinfo->kernel_name, MAX_KERNEL_NAME - 1);
    event.kernel_object = sinfo->kernel_object;
    event.grid_size_x = sinfo->grid_size_x;
    event.grid_size_y = sinfo->grid_size_y;
    event.grid_size_z = sinfo->grid_size_z;
    event.workgroup_size_x = sinfo->workgroup_size_x;
    event.workgroup_size_y = sinfo->workgroup_size_y;
    event.workgroup_size_z = sinfo->workgroup_size_z;
    event.group_segment_size = sinfo->group_segment_size;
    event.private_segment_size = sinfo->private_segment_size;
    event.gpu_start_time = gpu_start_time;
    event.gpu_end_time = gpu_end_time;

    send_kernel_event(&event);

    /* Cleanup tracking signal */
    if (real_hsa_signal_destroy) {
        real_hsa_signal_destroy(sinfo->signal);
    }

    /* Mark as inactive */
    pthread_mutex_lock(&g_signal_mutex);
    sinfo->active = 0;
    pthread_mutex_unlock(&g_signal_mutex);

    return false;
}

/* Write Interceptor Callback */
static void write_interceptor_callback(
    const void* packets,
    uint64_t num_packets,
    uint64_t user_pkt_index,
    void* data,
    hsa_amd_queue_intercept_packet_writer writer)
{
    queue_info_t* qinfo = (queue_info_t*)data;

    if (!packets || num_packets == 0) {
        writer(packets, num_packets);
        return;
    }

    /* Process each packet */
    for (uint64_t i = 0; i < num_packets; i++) {
        const hsa_kernel_dispatch_packet_t* pkt =
            (const hsa_kernel_dispatch_packet_t*)((char*)packets + i * sizeof(hsa_kernel_dispatch_packet_t));

        uint16_t header = __atomic_load_n(&pkt->header, __ATOMIC_ACQUIRE);
        uint16_t packet_type = header & HSA_PACKET_TYPE_KERNEL_DISPATCH;

        if (packet_type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
            /* Save the original completion signal (if any) */
            hsa_signal_t original_signal = pkt->completion_signal;
            int has_original = (original_signal.handle != 0);

            /* Create our own tracking signal (initialized to 1) */
            hsa_signal_t tracking_signal;
            hsa_status_t status = real_hsa_signal_create(1, 0, NULL, &tracking_signal);

            if (status == HSA_STATUS_SUCCESS) {
                signal_info_t* sinfo = allocate_signal_info();
                if (sinfo) {
                    sinfo->signal = tracking_signal;
                    sinfo->original_signal = original_signal;
                    sinfo->has_original_signal = has_original;
                    sinfo->agent = qinfo->agent;
                    sinfo->dispatch_time = get_timestamp_ns();
                    sinfo->write_index = user_pkt_index + i;
                    sinfo->queue_id = qinfo->queue_id;

                    /* Get correlation ID from eBPF map (links this dispatch to HIP API) */
                    sinfo->correlation_id = get_correlation_id_from_ebpf();

                    extract_kernel_name(pkt, sinfo->kernel_name, sizeof(sinfo->kernel_name));

                    /* Capture kernel dispatch information */
                    sinfo->kernel_object = pkt->kernel_object;
                    sinfo->workgroup_size_x = pkt->workgroup_size_x;
                    sinfo->workgroup_size_y = pkt->workgroup_size_y;
                    sinfo->workgroup_size_z = pkt->workgroup_size_z;
                    sinfo->grid_size_x = pkt->grid_size_x;
                    sinfo->grid_size_y = pkt->grid_size_y;
                    sinfo->grid_size_z = pkt->grid_size_z;
                    sinfo->private_segment_size = pkt->private_segment_size;
                    sinfo->group_segment_size = pkt->group_segment_size;

                    /* Send kernel dispatch event to eBPF */
                    kernel_event_t dispatch_event = {0};
                    dispatch_event.timestamp = sinfo->dispatch_time;
                    dispatch_event.pid = getpid();
                    dispatch_event.tid = sinfo->queue_id;
                    dispatch_event.event_type = EVENT_KERNEL_DISPATCH;
                    dispatch_event.queue_id = sinfo->queue_id;
                    dispatch_event.agent_id = sinfo->agent.handle;  // Store agent ID for categorization
                    dispatch_event.correlation_id = sinfo->correlation_id;  // Link to HIP API
                    strncpy(dispatch_event.kernel_name, sinfo->kernel_name, MAX_KERNEL_NAME - 1);
                    dispatch_event.kernel_object = sinfo->kernel_object;
                    dispatch_event.grid_size_x = sinfo->grid_size_x;
                    dispatch_event.grid_size_y = sinfo->grid_size_y;
                    dispatch_event.grid_size_z = sinfo->grid_size_z;
                    dispatch_event.workgroup_size_x = sinfo->workgroup_size_x;
                    dispatch_event.workgroup_size_y = sinfo->workgroup_size_y;
                    dispatch_event.workgroup_size_z = sinfo->workgroup_size_z;
                    dispatch_event.group_segment_size = sinfo->group_segment_size;
                    dispatch_event.private_segment_size = sinfo->private_segment_size;

                    // DON'T send dispatch events - they flood the single-slot map and overwrite completion events
                    // The HIP API uprobes already capture dispatch timing
                    // send_kernel_event(&dispatch_event);

                    /* Reset signal to 0 before kernel dispatch */
                    if (real_hsa_signal_store_screlease) {
                        real_hsa_signal_store_screlease(tracking_signal, 0);
                    }

                    /* Register async signal handler to wait for completion */
                    if (real_hsa_amd_signal_async_handler) {
                        real_hsa_amd_signal_async_handler(
                            tracking_signal,
                            HSA_SIGNAL_CONDITION_EQ,
                            -1,
                            async_signal_handler,
                            sinfo
                        );
                    }

                    /* Modify packet to use our tracking signal */
                    hsa_kernel_dispatch_packet_t modified_pkt = *pkt;
                    modified_pkt.completion_signal = tracking_signal;

                    /* Write the modified packet */
                    writer(&modified_pkt, 1);
                    continue;
                }
            }
        }

        /* For non-kernel packets or if signal creation failed, write original */
        writer((char*)packets + i * sizeof(hsa_kernel_dispatch_packet_t), 1);
    }
}

/* Install write interceptor on a queue */
static void install_write_interceptor(hsa_queue_t* queue) {
    queue_info_t* qinfo = find_queue_by_pointer(queue);
    if (!qinfo) {
        qinfo = allocate_queue_info(queue);
    }

    if (!qinfo || qinfo->interceptor_installed) {
        return;
    }

    /* Check if HSA AMD functions are available */
    if (!real_hsa_amd_profiling_set_profiler_enabled || !real_hsa_amd_queue_intercept_register) {
        return;
    }

    /* Enable profiling */
    hsa_status_t status = real_hsa_amd_profiling_set_profiler_enabled(queue, 1);
    if (status != HSA_STATUS_SUCCESS) {
        return;
    }

    /* Register write interceptor */
    status = real_hsa_amd_queue_intercept_register(queue, write_interceptor_callback, qinfo);
    if (status == HSA_STATUS_SUCCESS) {
        qinfo->interceptor_installed = 1;
    } else {
    }
}


/* Initialize shim */
static void init_shim(void) __attribute__((constructor));
static void init_shim(void) {
    if (g_initialized) return;


    /* Load HSA library */
    void* hsa_lib = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (!hsa_lib) {
        hsa_lib = dlopen("libhsa-runtime64.so.1", RTLD_LAZY);
    }

    if (!hsa_lib) {
        return;
    }

    // Load basic HSA signal functions (needed for write interceptor)
    real_hsa_queue_create = (decltype(real_hsa_queue_create))dlsym(hsa_lib, "hsa_queue_create");
    real_hsa_signal_create = (decltype(real_hsa_signal_create))dlsym(hsa_lib, "hsa_signal_create");
    real_hsa_signal_destroy = (decltype(real_hsa_signal_destroy))dlsym(hsa_lib, "hsa_signal_destroy");
    real_hsa_signal_store_screlease = (decltype(real_hsa_signal_store_screlease))dlsym(hsa_lib, "hsa_signal_store_screlease");

    // Load HSA core functions for symbol extraction
    real_hsa_executable_freeze = (decltype(real_hsa_executable_freeze))dlsym(hsa_lib, "hsa_executable_freeze");
    real_hsa_executable_iterate_agent_symbols = (decltype(real_hsa_executable_iterate_agent_symbols))dlsym(hsa_lib, "hsa_executable_iterate_agent_symbols");
    real_hsa_executable_symbol_get_info = (decltype(real_hsa_executable_symbol_get_info))dlsym(hsa_lib, "hsa_executable_symbol_get_info");
    real_hsa_iterate_agents = (decltype(real_hsa_iterate_agents))dlsym(hsa_lib, "hsa_iterate_agents");
    real_hsa_system_get_info = (decltype(real_hsa_system_get_info))dlsym(hsa_lib, "hsa_system_get_info");

    if (!real_hsa_queue_create || !real_hsa_signal_create || !real_hsa_signal_store_screlease) {
        return;
    }

    if (!real_hsa_executable_freeze || !real_hsa_executable_iterate_agent_symbols ||
        !real_hsa_executable_symbol_get_info || !real_hsa_iterate_agents) {
    }

    // Note: Timestamp frequency will be queried lazily on first use (after HSA runtime is initialized)

    // AMD extension functions will be loaded from API tables via rocprofiler_configure callback

    /* Open eBPF ring buffer for sending kernel events */
    g_kernel_events_fd = bpf_obj_get("/sys/fs/bpf/kernel_events");
    if (g_kernel_events_fd < 0) {
        fprintf(stderr, "[SHIM] WARNING: Failed to open kernel_events map: %s\n", strerror(errno));
    }

    /* Open eBPF correlation_map for reading correlation IDs */
    g_correlation_map_fd = bpf_obj_get("/sys/fs/bpf/correlation_map");
    if (g_correlation_map_fd < 0) {
        fprintf(stderr, "[SHIM] WARNING: Failed to open correlation_map: %s\n", strerror(errno));
    }

    memset(g_queues, 0, sizeof(g_queues));
    memset(g_signals, 0, sizeof(g_signals));
    memset(g_kernel_names, 0, sizeof(g_kernel_names));

    g_initialized = 1;
}

/* Cleanup on exit */
__attribute__((destructor))
static void cleanup_shim(void) {
    if (!g_initialized) return;


    g_running = 0;

    if (g_kernel_events_fd >= 0) {
        close(g_kernel_events_fd);
    }

    if (g_correlation_map_fd >= 0) {
        close(g_correlation_map_fd);
    }

}

/* Intercepted hsa_queue_create - use hsa_amd_queue_intercept_create instead */
hsa_status_t hsa_queue_create(
    hsa_agent_t agent,
    uint32_t size,
    hsa_queue_type32_t type,
    void (*callback)(hsa_status_t status, hsa_queue_t *source, void *data),
    void *data,
    uint32_t private_segment_size,
    uint32_t group_segment_size,
    hsa_queue_t **queue)
{
    if (!g_initialized) init_shim();

    hsa_status_t status;

    /* If AMD intercept create is available, use it to create an intercept queue */
    if (real_hsa_amd_queue_intercept_create) {

        /* Create intercept queue */
        status = real_hsa_amd_queue_intercept_create(
            agent, size, type, callback, data,
            private_segment_size, group_segment_size, queue);

        if (status != HSA_STATUS_SUCCESS) {
            return status;
        }

        if (!queue || !*queue) {
            return HSA_STATUS_ERROR_INVALID_QUEUE;
        }


        /* Enable profiling on the intercept queue */
        if (real_hsa_amd_profiling_set_profiler_enabled) {
            status = real_hsa_amd_profiling_set_profiler_enabled(*queue, 1);
            if (status != HSA_STATUS_SUCCESS) {
            } else {
            }
        }

        /* Register write interceptor */
        if (real_hsa_amd_queue_intercept_register) {
            queue_info_t* qinfo = allocate_queue_info(*queue);
            if (qinfo) {
                qinfo->agent = agent;  // Save agent for profiling API
                status = real_hsa_amd_queue_intercept_register(*queue, write_interceptor_callback, qinfo);
                if (status == HSA_STATUS_SUCCESS) {
                    qinfo->interceptor_installed = 1;
                } else {
                }
            }
        }

        return HSA_STATUS_SUCCESS;
    }

    /* Fallback: Use regular queue creation */
    status = real_hsa_queue_create(
        agent, size, type, callback, data,
        private_segment_size, group_segment_size, queue);

    if (status == HSA_STATUS_SUCCESS && queue && *queue) {
    }

    return status;
}

/* Context for agent iteration */
struct agent_symbol_context {
    hsa_executable_t executable;
};

/* Callback to iterate through agents and extract symbols for each */
static hsa_status_t iterate_agent_callback(hsa_agent_t agent, void* data) {
    struct agent_symbol_context* ctx = (struct agent_symbol_context*)data;

    if (!real_hsa_executable_iterate_agent_symbols) {
        return HSA_STATUS_SUCCESS;
    }

    // Get agent device type to skip non-GPU agents
    hsa_device_type_t device_type;
    hsa_status_t status = HSA_STATUS_SUCCESS;

    // We need hsa_agent_get_info function
    void* hsa_lib = dlopen("libhsa-runtime64.so.1", RTLD_LAZY | RTLD_NOLOAD);
    if (hsa_lib) {
        typedef hsa_status_t (*hsa_agent_get_info_fn)(hsa_agent_t, hsa_agent_info_t, void*);
        hsa_agent_get_info_fn get_info = (hsa_agent_get_info_fn)dlsym(hsa_lib, "hsa_agent_get_info");

        if (get_info) {
            status = get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type);

            // Only process GPU agents
            if (status == HSA_STATUS_SUCCESS && device_type == HSA_DEVICE_TYPE_GPU) {

                hsa_status_t iter_status = real_hsa_executable_iterate_agent_symbols(
                    ctx->executable, agent, executable_symbol_callback, NULL);

                if (iter_status != HSA_STATUS_SUCCESS) {
                }
            }
        }
    }

    return HSA_STATUS_SUCCESS;
}

/* Intercepted hsa_executable_freeze - extract kernel symbols when executable is frozen */
hsa_status_t hsa_executable_freeze(hsa_executable_t executable, const char* options)
{
    if (!g_initialized) init_shim();

    // Call the real function first
    hsa_status_t status = HSA_STATUS_ERROR;
    if (real_hsa_executable_freeze) {
        status = real_hsa_executable_freeze(executable, options);
    }

    // If freeze succeeded and we have the symbol iteration functions, extract kernel names
    if (status == HSA_STATUS_SUCCESS &&
        real_hsa_executable_iterate_agent_symbols &&
        real_hsa_executable_symbol_get_info &&
        real_hsa_iterate_agents) {


        // Iterate through all HSA agents and extract symbols for each
        struct agent_symbol_context ctx = { .executable = executable };
        real_hsa_iterate_agents(iterate_agent_callback, &ctx);
    }

    return status;
}

/* Helper function to exit with error */
static void die(const char* fmt, ...) __attribute__((noreturn, format(printf, 1, 2)));
static void die(const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    exit(1);
}

// AMD Extension Table is already defined in hsa_api_trace.h
static void
api_registration_callback(rocprofiler_intercept_table_t type,
                          uint64_t lib_version,
                          uint64_t lib_instance,
                          void**   tables,
                          uint64_t num_tables,
                          void*    user_data)
{
    (void) user_data;
    (void) lib_instance;


    if (type != ROCPROFILER_HSA_TABLE) {
        return;
    }

    // According to ROCprofiler SDK docs, num_tables should be 1 for HSA
    // tables[0] is HsaApiTable which contains pointers to subtables
    if (num_tables != 1) {
        return;
    }

    // Extract HsaApiTable from tables[0]
    HsaApiTable* hsa_api_table = (HsaApiTable*)tables[0];
    if (!hsa_api_table) {
        return;
    }


    // Extract AMD Extension Table from HsaApiTable
    AmdExtTable* amd_ext_table = hsa_api_table->amd_ext_;
    if (!amd_ext_table) {
        return;
    }


    // Save the function pointers we need
    real_hsa_amd_profiling_set_profiler_enabled =
        amd_ext_table->hsa_amd_profiling_set_profiler_enabled_fn;
    real_hsa_amd_queue_intercept_create =
        amd_ext_table->hsa_amd_queue_intercept_create_fn;
    real_hsa_amd_queue_intercept_register =
        amd_ext_table->hsa_amd_queue_intercept_register_fn;
    real_hsa_amd_signal_async_handler =
        amd_ext_table->hsa_amd_signal_async_handler_fn;
    real_hsa_amd_profiling_get_dispatch_time =
        amd_ext_table->hsa_amd_profiling_get_dispatch_time_fn;

    // DEBUG: Log function pointer initialization
}

extern "C"
__attribute__((visibility("default")))
rocprofiler_tool_configure_result_t *rocprofiler_configure(uint32_t version,
							   const char *runtime_version,
							   uint32_t priority,
							   rocprofiler_client_id_t *id)
{
    (void) priority;
    (void) runtime_version;
    (void) version;

    id->name = "eBPF";

    if (ROCPROFILER_STATUS_ERROR_NOT_IMPLEMENTED ==
        rocprofiler_at_intercept_table_registration(api_registration_callback,
                                                    ROCPROFILER_HSA_TABLE,
                                                    nullptr)) {
        die("Trying to register API interception : " "NOT IMPLEMENTED");
    }

    static auto cfg = rocprofiler_tool_configure_result_t {
        sizeof(rocprofiler_tool_configure_result_t),
        nullptr,
        nullptr,
        nullptr
    };

    return &cfg;
}
