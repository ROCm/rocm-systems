// SPDX-License-Identifier: GPL-2.0
/*
 * Unified HIP + Kernel Dispatch Tracer (Userspace)
 *
 * This tool:
 * 1. Loads the unified eBPF program (hip_kernel_unified.bpf.c)
 * 2. Attaches uprobes to HIP API functions in libamdhip64.so
 * 3. Receives HIP API events from eBPF uprobes
 * 4. Receives kernel dispatch/completion events from hsa_hybrid_shim via kernel_events ring buffer
 * 5. Correlates HIP APIs with kernel dispatches
 * 6. Generates unified Chrome Trace JSON output
 *
 * Usage: sudo ./hip_kernel_unified_tracer -l /opt/rocm/lib/libamdhip64.so -o trace.json
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <errno.h>
#include <time.h>
#include <fcntl.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <gelf.h>
#include <libelf.h>

#include "hip_kernel_unified.skel.h"
#include "chrome_trace_writer.h"

#define MAX_NAME_LEN 64
#define MAX_KERNEL_NAME 256
#define MAX_ARGS 8

// Event types (must match eBPF and shim)
#define EVENT_HIP_API_ENTRY     0
#define EVENT_HIP_API_EXIT      1
#define EVENT_KERNEL_DISPATCH   2
#define EVENT_KERNEL_COMPLETE   3

// Event structures (must match eBPF and shim)
struct gpu_trace_event {
    uint64_t timestamp;
    uint32_t pid;
    uint32_t tid;
    uint32_t event_type;
    uint32_t correlation_id;

    // For HIP API events
    char function_name[MAX_NAME_LEN];
    uint64_t args[MAX_ARGS];
    uint32_t arg_count;
    uint64_t return_value;

    // For kernel dispatch events
    char kernel_name[MAX_KERNEL_NAME];
    uint64_t kernel_object;
    uint32_t queue_id;

    // Kernel metadata
    uint32_t grid_size_x, grid_size_y, grid_size_z;
    uint32_t workgroup_size_x, workgroup_size_y, workgroup_size_z;
    uint32_t group_segment_size;
    uint32_t private_segment_size;

    // GPU timestamps
    uint64_t gpu_start_time;
    uint64_t gpu_end_time;
};

struct __attribute__((packed)) kernel_event {
    uint64_t timestamp;
    uint32_t pid;
    uint32_t tid;
    uint32_t event_type;
    uint32_t queue_id;
    uint64_t agent_id;    // HSA agent handle
    char kernel_name[MAX_KERNEL_NAME];
    uint64_t kernel_object;
    uint32_t grid_size_x, grid_size_y, grid_size_z;
    uint32_t workgroup_size_x, workgroup_size_y, workgroup_size_z;
    uint32_t group_segment_size;
    uint32_t private_segment_size;
    uint64_t gpu_start_time;
    uint64_t gpu_end_time;
};

// Correlation state for HIP API calls
struct hip_api_call {
    uint64_t start_timestamp;
    uint32_t pid;
    uint32_t tid;
    char function_name[MAX_NAME_LEN];
    uint64_t args[MAX_ARGS];
    uint32_t arg_count;
    int active;
};

#define MAX_CORRELATIONS 4096
static struct hip_api_call g_correlations[MAX_CORRELATIONS];

// Global state
static volatile bool g_exiting = false;
static perfetto_writer_t* g_trace_writer = NULL;
static struct hip_kernel_unified_bpf* g_skel = NULL;
static struct bpf_link** g_hip_links = NULL;
static int g_hip_link_count = 0;

// HIP function names to trace
static const char* hip_functions[] = {
    "hipMalloc", "hipFree", "hipMemcpy", "hipMemcpyAsync",
    "hipMemset", "hipMemsetAsync", "hipLaunchKernel",
    "hipStreamCreate", "hipStreamDestroy", "hipStreamSynchronize",
    "hipDeviceSynchronize", "hipGetDevice", "hipSetDevice",
    NULL
};

static void sig_handler(int sig) {
    (void)sig;
    g_exiting = true;
}

// ELF parsing functions copied from hip_tracer.c
static bool is_func_sym(const GElf_Sym *sym) {
    /* STT_FUNC and defined (not SHN_UNDEF), non-zero size preferred */
    if (GELF_ST_TYPE(sym->st_info) != STT_FUNC) return false;
    if (sym->st_shndx == SHN_UNDEF) return false;
    if (sym->st_value == 0) return false;       // avoid PLT/ifunc weirdness
    /* size can be zero on some builds; prefer >0, but don't strictly require */
    return true;
}

static bool section_is_text_like(const char *secname) {
    if (!secname) return false;
    /* Accept typical code sections; reject .plt/.plt.sec etc. */
    if (strncmp(secname, ".text", 5) == 0) return true;
    if (strcmp(secname, ".init") == 0 || strcmp(secname, ".fini") == 0) return true;
    /* Reject PLT */
    if (strncmp(secname, ".plt", 4) == 0) return false;
    return true; /* fall back to true if unsure */
}

static off_t compute_file_offset(const GElf_Sym *sym, const GElf_Shdr *shdr) {
    /* file_off = (virtual_symbol_addr - section_load_addr) + section_file_offset */
    return (off_t)(sym->st_value - shdr->sh_addr) + (off_t)shdr->sh_offset;
}

static Elf_Scn *find_symbol_section(Elf *elf, bool dynsym) {
    size_t shstrndx = 0;
    if (elf_getshdrstrndx(elf, &shstrndx) != 0) return NULL;

    Elf_Scn *scn = NULL;
    while ((scn = elf_nextscn(elf, scn))) {
        GElf_Shdr sh;
        if (!gelf_getshdr(scn, &sh)) continue;
        if ((dynsym && sh.sh_type == SHT_DYNSYM) ||
            (!dynsym && sh.sh_type == SHT_SYMTAB)) {
            return scn;
        }
    }
    return NULL;
}

static off_t lookup_one_table(Elf *elf, const char *want, Elf_Scn *sym_scn) {
    GElf_Shdr sym_sh;
    if (!gelf_getshdr(sym_scn, &sym_sh)) return -1;

    Elf_Data *sym_data = NULL;
    if (!(sym_data = elf_getdata(sym_scn, NULL))) return -1;

    /* We also need the section header string table to resolve section names. */
    size_t shstrndx = 0;
    if (elf_getshdrstrndx(elf, &shstrndx) != 0)
        return -1;

    size_t nsyms = sym_sh.sh_size / sym_sh.sh_entsize;
    for (size_t i = 0; i < nsyms; i++) {
        GElf_Sym sym;
        if (!gelf_getsym(sym_data, (int)i, &sym)) continue;

        const char *name = elf_strptr(elf, sym_sh.sh_link, sym.st_name);
        if (!name || strcmp(name, want) != 0) continue;

        if (!is_func_sym(&sym)) continue;

        /* Get the containing section to compute file offset */
        Elf_Scn *sec = elf_getscn(elf, sym.st_shndx);
        if (!sec) continue;

        GElf_Shdr shdr;
        if (!gelf_getshdr(sec, &shdr)) continue;

        const char *secname = elf_strptr(elf, shstrndx, shdr.sh_name);
        if (!section_is_text_like(secname)) continue;

        off_t off = compute_file_offset(&sym, &shdr);
        if (off >= 0) return off;
    }
    return -1;
}

/**
 * Return a file-relative offset suitable for uprobes (bpf_program__attach_uprobe).
 * Looks in .dynsym first, then .symtab. Exact name match, skips PLT/stubs.
 * Returns -1 on failure.
 */
static off_t get_function_offset(const char *lib_path, const char *func_name) {
    if (!lib_path || !func_name) { errno = EINVAL; return -1; }

    if (elf_version(EV_CURRENT) == EV_NONE) return -1;

    int fd = open(lib_path, O_RDONLY);
    if (fd < 0) return -1;

    Elf *elf = elf_begin(fd, ELF_C_READ, NULL);
    if (!elf) { close(fd); return -1; }

    /* Prefer .dynsym (shared libs' exported funcs), then fall back to .symtab */
    off_t off = -1;

    Elf_Scn *dynsym = find_symbol_section(elf, /*dynsym=*/true);
    if (dynsym) {
        off = lookup_one_table(elf, func_name, dynsym);
    }
    if (off < 0) {
        Elf_Scn *symtab = find_symbol_section(elf, /*dynsym=*/false);
        if (symtab) off = lookup_one_table(elf, func_name, symtab);
    }

    elf_end(elf);
    close(fd);
    return off;
}

// Function name mapping for correlation IDs
static const char* g_function_names[MAX_CORRELATIONS];

// Check if an argument should be formatted as decimal (not hex)
static int should_format_as_decimal(const char* func_name, int arg_index) {
    if (strcmp(func_name, "hipMalloc") == 0 && arg_index == 1) return 1; // size
    if (strcmp(func_name, "hipMemcpy") == 0 && arg_index == 2) return 1; // count
    if (strcmp(func_name, "hipMemcpyAsync") == 0 && arg_index == 2) return 1; // count
    if (strcmp(func_name, "hipMemset") == 0 && (arg_index == 1 || arg_index == 2)) return 1; // value, count
    if (strcmp(func_name, "hipMemsetAsync") == 0 && (arg_index == 1 || arg_index == 2)) return 1; // value, count
    if (strcmp(func_name, "hipSetDevice") == 0 && arg_index == 0) return 1; // deviceId
    if (strcmp(func_name, "hipGetDevice") == 0 && arg_index == 0) return 1; // deviceId
    if (strcmp(func_name, "hipLaunchKernel") == 0 && arg_index == 4) return 1; // sharedMemBytes
    return 0;
}


// Get argument names for different HIP functions
static const char* get_argument_name(const char* func_name, int arg_index) {
    if (strcmp(func_name, "hipMalloc") == 0) {
        const char* names[] = {"devPtr", "size"};
        return (arg_index < 2) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipFree") == 0) {
        const char* names[] = {"devPtr"};
        return (arg_index < 1) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipMemcpy") == 0) {
        const char* names[] = {"dst", "src", "count", "kind"};
        return (arg_index < 4) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipMemcpyAsync") == 0) {
        const char* names[] = {"dst", "src", "count", "kind", "stream"};
        return (arg_index < 5) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipMemset") == 0) {
        const char* names[] = {"devPtr", "value", "count"};
        return (arg_index < 3) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipMemsetAsync") == 0) {
        const char* names[] = {"devPtr", "value", "count", "stream"};
        return (arg_index < 4) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipLaunchKernel") == 0) {
        const char* names[] = {"func", "gridDim", "blockDim", "args", "sharedMemBytes"};
        return (arg_index < 5) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipStreamCreate") == 0) {
        const char* names[] = {"stream"};
        return (arg_index < 1) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipStreamDestroy") == 0) {
        const char* names[] = {"stream"};
        return (arg_index < 1) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipStreamSynchronize") == 0) {
        const char* names[] = {"stream"};
        return (arg_index < 1) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipSetDevice") == 0) {
        const char* names[] = {"deviceId"};
        return (arg_index < 1) ? names[arg_index] : "arg";
    } else if (strcmp(func_name, "hipGetDevice") == 0) {
        const char* names[] = {"deviceId"};
        return (arg_index < 1) ? names[arg_index] : "arg";
    }
    return "arg";
}

// Find HIP API call by correlation ID
static struct hip_api_call* find_hip_call(uint32_t correlation_id) {
    if (correlation_id >= MAX_CORRELATIONS) return NULL;
    if (g_correlations[correlation_id].active) {
        return &g_correlations[correlation_id];
    }
    return NULL;
}

// Handle events from main ring buffer (HIP API events)
static int handle_hip_event(void* ctx, void* data, size_t data_sz) {
    (void)ctx;

    if (data_sz < sizeof(struct gpu_trace_event)) {
        return 0;
    }

    struct gpu_trace_event* event = (struct gpu_trace_event*)data;

    switch (event->event_type) {
        case EVENT_HIP_API_ENTRY: {
            if (event->correlation_id < MAX_CORRELATIONS) {
                struct hip_api_call* call = &g_correlations[event->correlation_id];
                call->start_timestamp = event->timestamp;
                call->pid = event->pid;
                call->tid = event->tid;
                
                // Use stored function name or event function name
                const char* func_name = g_function_names[event->correlation_id];
                if (func_name && strlen(func_name) > 0) {
                    strncpy(call->function_name, func_name, MAX_NAME_LEN - 1);
                } else if (strlen(event->function_name) > 0) {
                    strncpy(call->function_name, event->function_name, MAX_NAME_LEN - 1);
                } else {
                    strncpy(call->function_name, "unknown_hip_function", MAX_NAME_LEN - 1);
                }
                call->function_name[MAX_NAME_LEN - 1] = '\0';
                
                // Store arguments
                call->arg_count = event->arg_count < MAX_ARGS ? event->arg_count : MAX_ARGS;
                for (int i = 0; i < call->arg_count; i++) {
                    call->args[i] = event->args[i];
                }
                
                call->active = 1;
                
                printf("[TRACE] HIP Entry: %s (corr_id=%u, pid=%u, tid=%u)\n", 
                       call->function_name, event->correlation_id, event->pid, event->tid);
            }
            break;
        }

        case EVENT_HIP_API_EXIT: {
            struct hip_api_call* call = find_hip_call(event->correlation_id);
            if (call) {
                printf("[TRACE] HIP Exit: %s (corr_id=%u, ret=0x%llx, duration=%llu ns)\n",
                       call->function_name, event->correlation_id,
                       (unsigned long long)event->return_value,
                       event->timestamp - call->start_timestamp);

                // Skip unknown functions (internal HIP runtime calls we're not tracing)
                if (strstr(call->function_name, "unknown_hip") != NULL) {
                    call->active = 0;
                    break;
                }

                if (g_trace_writer) {
                    // Build arguments JSON
                    char args_buffer[1024];
                    int pos = 0;
                    pos += snprintf(args_buffer + pos, sizeof(args_buffer) - pos, "{");
                    
                    // Add return value
                    pos += snprintf(args_buffer + pos, sizeof(args_buffer) - pos, 
                                   "\"return_value\":\"0x%llx\"", 
                                   (unsigned long long)event->return_value);
                    
                    // Add arguments
                    for (int i = 0; i < call->arg_count && i < MAX_ARGS && pos < sizeof(args_buffer) - 50; i++) {
                        const char* arg_name = get_argument_name(call->function_name, i);
                        if (should_format_as_decimal(call->function_name, i)) {
                            pos += snprintf(args_buffer + pos, sizeof(args_buffer) - pos,
                                           ",\"%s\":%llu",
                                           arg_name, (unsigned long long)call->args[i]);
                        } else {
                            pos += snprintf(args_buffer + pos, sizeof(args_buffer) - pos,
                                           ",\"%s\":\"0x%llx\"",
                                           arg_name, (unsigned long long)call->args[i]);
                        }
                    }
                    
                    pos += snprintf(args_buffer + pos, sizeof(args_buffer) - pos, "}");
                    
                    perfetto_writer_add_slice_with_args(
                        g_trace_writer,
                        call->start_timestamp,
                        event->timestamp,
                        call->function_name,
                        call->pid,
                        call->tid,
                        args_buffer
                    );
                }
                call->active = 0;
            }
            break;
        }

        default:
            break;
    }

    return 0;
}

// Track map to store agent/queue names we've already emitted
typedef struct {
    uint64_t agent_id;
    uint32_t queue_id;
} agent_queue_pair_t;

static uint64_t g_emitted_agents[256];
static int g_emitted_agent_count = 0;
static agent_queue_pair_t g_emitted_queues[1024];
static int g_emitted_queue_count = 0;

// Handle kernel events from shim
static int handle_kernel_event(void* ctx, void* data, size_t data_sz) {
    (void)ctx;


    if (data_sz < sizeof(struct kernel_event)) {
        return 0;
    }

    struct kernel_event* event = (struct kernel_event*)data;


    if (!g_trace_writer) return 0;

    // Only write on kernel completion (which has GPU timestamps)
    if (event->event_type == EVENT_KERNEL_COMPLETE) {
        uint32_t kernel_pid = (uint32_t)(event->agent_id & 0xFFFFFFFF);
        uint32_t kernel_tid = event->queue_id;

        // Check if we need to emit process name for this agent
        int agent_found = 0;
        for (int i = 0; i < g_emitted_agent_count; i++) {
            if (g_emitted_agents[i] == event->agent_id) {
                agent_found = 1;
                break;
            }
        }

        if (!agent_found && g_emitted_agent_count < 256) {
            // Add process name metadata for this agent
            char agent_name[128];
            snprintf(agent_name, sizeof(agent_name), "GPU Agent 0x%llx",
                     (unsigned long long)event->agent_id);
            perfetto_writer_add_track(g_trace_writer, kernel_pid, kernel_pid, agent_name);

            g_emitted_agents[g_emitted_agent_count++] = event->agent_id;
        }

        // Check if we need to emit thread name for this queue
        int queue_found = 0;
        for (int i = 0; i < g_emitted_queue_count; i++) {
            if (g_emitted_queues[i].agent_id == event->agent_id &&
                g_emitted_queues[i].queue_id == event->queue_id) {
                queue_found = 1;
                break;
            }
        }

        if (!queue_found && g_emitted_queue_count < 1024) {
            // Add thread name metadata for this queue
            char queue_name[128];
            snprintf(queue_name, sizeof(queue_name), "Queue %u", event->queue_id);

            // Write thread_name metadata event
            fprintf(g_trace_writer->file, ",\n    {\n");
            fprintf(g_trace_writer->file, "      \"name\": \"thread_name\",\n");
            fprintf(g_trace_writer->file, "      \"ph\": \"M\",\n");
            fprintf(g_trace_writer->file, "      \"pid\": %u,\n", kernel_pid);
            fprintf(g_trace_writer->file, "      \"tid\": %u,\n", kernel_tid);
            fprintf(g_trace_writer->file, "      \"args\": {\n");
            fprintf(g_trace_writer->file, "        \"name\": \"%s\"\n", queue_name);
            fprintf(g_trace_writer->file, "      }\n");
            fprintf(g_trace_writer->file, "    }");

            g_emitted_queues[g_emitted_queue_count].agent_id = event->agent_id;
            g_emitted_queues[g_emitted_queue_count].queue_id = event->queue_id;
            g_emitted_queue_count++;
        }
        // Build args JSON
        uint64_t total_workitems = (uint64_t)event->grid_size_x *
                                   event->grid_size_y * event->grid_size_z;
        uint64_t workgroup_size = (uint64_t)event->workgroup_size_x *
                                 event->workgroup_size_y * event->workgroup_size_z;

        char args_buffer[512];
        snprintf(args_buffer, sizeof(args_buffer),
                 "{\"kernel_object\":\"0x%llx\","
                 "\"agent_id\":\"0x%llx\","
                 "\"queue_id\":%u,"
                 "\"grid\":[%u,%u,%u],"
                 "\"workgroup\":[%u,%u,%u],"
                 "\"total_work_items\":%lu,"
                 "\"num_workgroups\":%lu,"
                 "\"lds_bytes\":%u,"
                 "\"scratch_bytes\":%u}",
                 (unsigned long long)event->kernel_object,
                 (unsigned long long)event->agent_id,
                 event->queue_id,
                 event->grid_size_x, event->grid_size_y, event->grid_size_z,
                 event->workgroup_size_x, event->workgroup_size_y, event->workgroup_size_z,
                 total_workitems,
                 workgroup_size > 0 ? total_workitems / workgroup_size : 0,
                 event->group_segment_size,
                 event->private_segment_size);

        // Write kernel event with Agent as PID and Queue as TID
        perfetto_writer_add_slice_with_args(
            g_trace_writer,
            event->gpu_start_time,
            event->gpu_end_time,
            event->kernel_name,
            kernel_pid,
            kernel_tid,
            args_buffer
        );
    }

    return 0;
}

static int libbpf_print_fn(enum libbpf_print_level level, const char* format, va_list args) {
    if (level == LIBBPF_DEBUG) {
        return 0;
    }
    return vfprintf(stderr, format, args);
}

int main(int argc, char** argv) {
    struct ring_buffer* rb_events = NULL;
    int err;
    const char* output_file = "hip_kernel_trace.json";
    const char* hip_lib = "/opt/rocm/lib/libamdhip64.so";

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-o") == 0 && i + 1 < argc) {
            output_file = argv[++i];
        } else if (strcmp(argv[i], "-l") == 0 && i + 1 < argc) {
            hip_lib = argv[++i];
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [-l libamdhip64.so] [-o output.json]\n", argv[0]);
            printf("\nOptions:\n");
            printf("  -l FILE    Path to libamdhip64.so (default: /opt/rocm/lib/libamdhip64.so)\n");
            printf("  -o FILE    Output trace file (default: hip_kernel_trace.json)\n");
            printf("  -h         Show this help\n");
            printf("\nRequires: hsa_hybrid_shim.so should be LD_PRELOAD'd in target app\n");
            printf("Example: LD_PRELOAD=./libhsa_hybrid_shim.so ./my_hip_app\n");
            return 0;
        }
    }

    // Set up signal handlers
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Set up libbpf logging
    libbpf_set_print(libbpf_print_fn);

    // Bump RLIMIT_MEMLOCK
    struct rlimit rlim = {
        .rlim_cur = RLIM_INFINITY,
        .rlim_max = RLIM_INFINITY,
    };
    if (setrlimit(RLIMIT_MEMLOCK, &rlim)) {
        fprintf(stderr, "Failed to increase RLIMIT_MEMLOCK: %s\n", strerror(errno));
        return 1;
    }

    // Open and load BPF program
    g_skel = hip_kernel_unified_bpf__open();
    if (!g_skel) {
        fprintf(stderr, "Failed to open BPF skeleton\n");
        return 1;
    }

    err = hip_kernel_unified_bpf__load(g_skel);
    if (err) {
        fprintf(stderr, "Failed to load BPF skeleton: %d\n", err);
        goto cleanup;
    }

    // Attach HIP API uprobes
    printf("Attaching HIP API uprobes to %s\n", hip_lib);
    printf("Using ELF parsing for accurate function offsets\n\n");
    
    int attached_count = 0;
    int function_names_fd = bpf_map__fd(g_skel->maps.function_names);
    int probe_function_map_fd = bpf_map__fd(g_skel->maps.probe_function_map);
    
    for (int i = 0; hip_functions[i] != NULL; i++) {
        // Get function offset using ELF parsing
        off_t offset = get_function_offset(hip_lib, hip_functions[i]);
        if (offset < 0) {
            printf("Warning: Could not find offset for %s, skipping\n", hip_functions[i]);
            continue;
        }
        
        printf("  %s at offset 0x%lx\n", hip_functions[i], offset);
        
        // Store function name in array map by function ID
        uint32_t func_id = i;
        char func_name_buf[MAX_NAME_LEN] = {0};
        strncpy(func_name_buf, hip_functions[i], MAX_NAME_LEN - 1);
        
        if (bpf_map_update_elem(function_names_fd, &func_id, func_name_buf, BPF_ANY) != 0) {
            printf("Warning: Failed to store function name for %s\n", hip_functions[i]);
        }
        
        // Store function name by index for later correlation
        g_function_names[i] = hip_functions[i];
        
        // Map offset to function ID for eBPF lookup
        uint64_t probe_offset = (uint64_t)offset;
        if (bpf_map_update_elem(probe_function_map_fd, &probe_offset, &func_id, BPF_ANY) != 0) {
            printf("Warning: Failed to store probe function mapping for %s\n", hip_functions[i]);
        }
        
        // Attach entry probe with func_id as cookie
        struct bpf_uprobe_opts entry_opts = {
            .sz = sizeof(struct bpf_uprobe_opts),
            .retprobe = false,
            .bpf_cookie = func_id
        };
        struct bpf_link* link_entry = bpf_program__attach_uprobe_opts(
            g_skel->progs.hip_api_entry, -1, hip_lib, offset, &entry_opts);

        // Attach exit probe with func_id as cookie
        struct bpf_uprobe_opts exit_opts = {
            .sz = sizeof(struct bpf_uprobe_opts),
            .retprobe = true,
            .bpf_cookie = func_id
        };
        struct bpf_link* link_exit = bpf_program__attach_uprobe_opts(
            g_skel->progs.hip_api_exit, -1, hip_lib, offset, &exit_opts);
        
        if (!link_entry || !link_exit) {
            fprintf(stderr, "Warning: Failed to attach probes for %s at offset 0x%lx\n", 
                    hip_functions[i], offset);
            if (link_entry) bpf_link__destroy(link_entry);
            if (link_exit) bpf_link__destroy(link_exit);
        } else {
            printf("    ✓ Attached entry and exit probes\n");
            attached_count++;
            g_hip_link_count += 2;
        }
    }
    
    printf("\nSuccessfully attached %d HIP functions\n", attached_count);
    if (attached_count == 0) {
        fprintf(stderr, "Error: No HIP functions were successfully attached!\n");
        return 1;
    }

    // Pin maps for shim to access
    int kernel_events_fd = bpf_map__fd(g_skel->maps.kernel_events);
    printf("Pinning kernel_events map (fd=%d) to /sys/fs/bpf/kernel_events\n", kernel_events_fd);
    if (bpf_obj_pin(kernel_events_fd, "/sys/fs/bpf/kernel_events") && errno != EEXIST) {
        fprintf(stderr, "Warning: Failed to pin kernel_events map: %s\n", strerror(errno));
    } else {
        printf("Successfully pinned kernel_events map\n");
    }

    // Initialize trace writer
    g_trace_writer = perfetto_writer_create(output_file);
    if (!g_trace_writer) {
        fprintf(stderr, "Failed to create trace writer\n");
        err = 1;
        goto cleanup;
    }

    // Add process track for better visualization
    perfetto_writer_add_track(g_trace_writer, getpid(), getpid(), "UnifiedHIPKernelTracer");

    printf("\nUnified HIP + Kernel Tracer started\n");
    printf("Output: %s\n", output_file);
    printf("Waiting for application (LD_PRELOAD=./libhsa_hybrid_shim.so <app>)...\n");
    printf("Press Ctrl+C to stop\n\n");

    // Set up ring buffer for HIP events only
    int events_fd = bpf_map__fd(g_skel->maps.events);
    rb_events = ring_buffer__new(events_fd, handle_hip_event, NULL, NULL);
    if (!rb_events) {
        fprintf(stderr, "Failed to create events ring buffer\n");
        err = 1;
        goto cleanup;
    }

    // Note: kernel_events is now an array map, not a ring buffer
    // We'll poll it directly in the main loop

    // Poll for events
    while (!g_exiting) {
        err = ring_buffer__poll(rb_events, 100);
        if (err == -EINTR) {
            err = 0;
            break;
        }
        if (err < 0) {
            fprintf(stderr, "Error polling events ring buffer: %d\n", err);
            break;
        }

        // Poll kernel events from array map
        uint32_t key = 0;
        struct kernel_event kev;
        if (bpf_map_lookup_elem(kernel_events_fd, &key, &kev) == 0 && kev.timestamp != 0) {
            // Process kernel event
            handle_kernel_event(NULL, &kev, sizeof(kev));
            
            // Clear the event to mark as processed
            memset(&kev, 0, sizeof(kev));
            bpf_map_update_elem(kernel_events_fd, &key, &kev, BPF_ANY);
        }
    }

cleanup:
    printf("\nShutting down...\n");

    if (rb_events) {
        ring_buffer__free(rb_events);
    }

    if (g_trace_writer) {
        perfetto_writer_destroy(g_trace_writer);
        printf("Trace written to: %s\n", output_file);
    }

    // Unpin maps
    unlink("/sys/fs/bpf/kernel_events");

    if (g_skel) {
        hip_kernel_unified_bpf__destroy(g_skel);
    }

    return err < 0 ? -err : 0;
}
