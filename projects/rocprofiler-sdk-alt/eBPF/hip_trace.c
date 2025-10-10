// SPDX-License-Identifier: GPL-2.0
#define _GNU_SOURCE
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <sys/resource.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <gelf.h>
#include <libelf.h>
#include "hip_trace.skel.h"
#include "chrome_trace_writer.h"
// No longer using generated code - reverting to manual approach

// Forward declarations for generated functions
const char* get_function_name_by_id(int func_id);
int attach_generated_hip_functions(struct hip_trace_bpf *skel, const char *lib_path, int *attached_count);

static volatile bool exiting = false;

// Event structure matching the eBPF program
struct hip_event {
    __u64 timestamp;
    __u32 pid;
    __u32 tid;
    __u32 event_type; // 0 = entry, 1 = exit, 2 = kernel_dispatch
    char function_name[64];
    __u64 args[8];
    __u64 return_value;
    __u32 arg_count;
    __u32 function_id; // Unique ID to match entry/exit events
};

// Kernel dispatch event structure
struct kernel_dispatch_event {
    __u64 timestamp;
    __u32 pid;
    __u32 tid;
    __u32 event_type; // 2 = kernel_dispatch, 3 = kernel_completion
    char event_name[64];
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

// Structure to store function call information for CSV output
struct function_call {
    __u32 function_id;
    char function_name[64];
    __u64 start_timestamp;
    __u64 end_timestamp;
    __u32 pid;
    __u32 tid;
    __u64 args[8];
    __u32 arg_count;
    __u64 return_value;
    int completed; // 0 = incomplete, 1 = complete
};

// Global variables for output
static FILE *csv_file = NULL;
static perfetto_writer_t *perfetto_writer = NULL;
static struct function_call *function_calls = NULL;
static size_t function_calls_size = 0;
static size_t function_calls_count = 0;
static int output_format = 0; // 0 = console, 1 = CSV, 2 = Perfetto
static bool enable_kernel_dispatches = false; // Kernel dispatch tracing disabled by default

// Function to format timestamp
static void print_timestamp(__u64 timestamp) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);

    // Convert nanoseconds to seconds and nanoseconds
    __u64 sec = timestamp / 1000000000ULL;
    __u64 nsec = timestamp % 1000000000ULL;

    printf("[%llu.%09llu] ", (unsigned long long)sec, (unsigned long long)nsec);
}

// Function to get argument names for a function
static const char* get_argument_names(const char* func_name, int arg_index) {
    if (strcmp(func_name, "hipMalloc") == 0) {
        const char* names[] = {"devPtr", "size"};
        return (arg_index < 2) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipFree") == 0) {
        const char* names[] = {"devPtr"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipMemcpy") == 0) {
        const char* names[] = {"dst", "src", "count", "kind"};
        return (arg_index < 4) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipMemcpyAsync") == 0) {
        const char* names[] = {"dst", "src", "count", "kind", "stream"};
        return (arg_index < 5) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipMemset") == 0) {
        const char* names[] = {"devPtr", "value", "count"};
        return (arg_index < 3) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipMemsetAsync") == 0) {
        const char* names[] = {"devPtr", "value", "count", "stream"};
        return (arg_index < 4) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipMemGetInfo") == 0) {
        const char* names[] = {"free", "total"};
        return (arg_index < 2) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipMemAllocManaged") == 0) {
        const char* names[] = {"devPtr", "size", "flags"};
        return (arg_index < 3) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipMemPrefetchAsync") == 0) {
        const char* names[] = {"devPtr", "count", "device"};
        return (arg_index < 3) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipMemAdvise") == 0) {
        const char* names[] = {"devPtr", "count", "advice"};
        return (arg_index < 3) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipStreamCreate") == 0) {
        const char* names[] = {"stream"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipStreamDestroy") == 0) {
        const char* names[] = {"stream"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipStreamSynchronize") == 0) {
        const char* names[] = {"stream"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipStreamWaitEvent") == 0) {
        const char* names[] = {"stream", "event"};
        return (arg_index < 2) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipStreamQuery") == 0) {
        const char* names[] = {"stream"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipEventCreate") == 0) {
        const char* names[] = {"event"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipEventDestroy") == 0) {
        const char* names[] = {"event"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipEventRecord") == 0) {
        const char* names[] = {"event", "stream"};
        return (arg_index < 2) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipEventSynchronize") == 0) {
        const char* names[] = {"event"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipEventElapsedTime") == 0) {
        const char* names[] = {"ms", "start", "end"};
        return (arg_index < 3) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipLaunchKernel") == 0) {
        const char* names[] = {"func", "gridDim", "blockDim", "args", "sharedMemBytes"};
        return (arg_index < 5) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipModuleLoad") == 0) {
        const char* names[] = {"module", "fname"};
        return (arg_index < 2) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipModuleUnload") == 0) {
        const char* names[] = {"module"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipModuleGetFunction") == 0) {
        const char* names[] = {"function", "module", "name"};
        return (arg_index < 3) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipModuleLaunchKernel") == 0) {
        const char* names[] = {"f", "gridDimX", "gridDimY", "gridDimZ", "blockDimX", "blockDimY", "blockDimZ", "sharedMemBytes"};
        return (arg_index < 8) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipSetDevice") == 0) {
        const char* names[] = {"deviceId"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipGetDevice") == 0) {
        const char* names[] = {"deviceId"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipGetDeviceCount") == 0) {
        const char* names[] = {"count"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipDeviceSynchronize") == 0) {
        return "none";
    } else if (strcmp(func_name, "hipDeviceReset") == 0) {
        return "none";
    } else if (strcmp(func_name, "hipCtxCreate") == 0) {
        const char* names[] = {"ctx", "flags", "device"};
        return (arg_index < 3) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipCtxDestroy") == 0) {
        const char* names[] = {"ctx"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipCtxSetCurrent") == 0) {
        const char* names[] = {"ctx"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipCtxGetCurrent") == 0) {
        const char* names[] = {"ctx"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipCtxSynchronize") == 0) {
        return "none";
    } else if (strcmp(func_name, "hipGetLastError") == 0) {
        return "none";
    } else if (strcmp(func_name, "hipPeekAtLastError") == 0) {
        return "none";
    } else if (strcmp(func_name, "hipGetErrorString") == 0) {
        const char* names[] = {"error"};
        return (arg_index < 1) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipProfilerStart") == 0) {
        return "none";
    } else if (strcmp(func_name, "hipProfilerStop") == 0) {
        return "none";
    } else if (strcmp(func_name, "hipDeviceGetAttribute") == 0) {
        const char* names[] = {"pi", "attr", "deviceId"};
        return (arg_index < 3) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipDeviceGetName") == 0) {
        const char* names[] = {"name", "len", "device"};
        return (arg_index < 3) ? names[arg_index] : "unknown";
    } else if (strcmp(func_name, "hipDeviceGetPCIBusId") == 0) {
        const char* names[] = {"pciBusId", "len", "device"};
        return (arg_index < 3) ? names[arg_index] : "unknown";
    }

    return "arg";
}

// Function to determine function name based on argument count and values
static const char* determine_function_name(__u32 arg_count, __u64 *args) {
    // This is a simplified approach - in practice, you'd need more sophisticated
    // heuristics to distinguish between functions with the same argument count
    switch (arg_count) {
        case 1:
            // Functions with 1 argument
            if (args && args[0] > 0x100000000) {
                // Large pointer value suggests hipGetDeviceCount
                return "hipGetDeviceCount";
            } else {
                // Smaller value suggests hipGetDevice
                return "hipGetDevice";
            }
        case 2:
            // Functions with 2 arguments - need to distinguish based on argument values
            if (args && args[0] > 0x100000000) {
                // Large pointer value suggests hipMalloc
                return "hipMalloc";
            } else {
                // Smaller value suggests hipMemGetInfo
                return "hipMemGetInfo";
            }
        case 3:
            // Functions with 3 arguments
            return "hipMemGetInfo";
        case 4:
            // Functions with 4 arguments - need to distinguish
            if (args && args[3] < 10) {
                // Small last argument suggests hipMemcpy
                return "hipMemcpy";
            } else {
                // Larger last argument suggests hipStreamCreate
                return "hipStreamCreate";
            }
        case 5:
            // Functions with 5 arguments
            return "hipLaunchKernel";
        case 6:
            // Functions with 6 arguments
            return "hipEventCreate";
        default:
            return "unknown_function";
    }
}

// Function to find or create function call entry
static struct function_call* find_or_create_function_call(__u32 function_id) {
    // First, try to find existing entry
    for (size_t i = 0; i < function_calls_count; i++) {
        if (function_calls[i].function_id == function_id) {
            return &function_calls[i];
        }
    }

    // If not found, create new entry
    if (function_calls_count >= function_calls_size) {
        function_calls_size = function_calls_size ? function_calls_size * 2 : 1024;
        function_calls = realloc(function_calls, function_calls_size * sizeof(struct function_call));
        if (!function_calls) {
            return NULL;
        }
    }

    struct function_call *call = &function_calls[function_calls_count++];
    memset(call, 0, sizeof(struct function_call));
    call->function_id = function_id;
    call->completed = 0;

    return call;
}

// Function to write CSV header
static void write_csv_header() {
    if (csv_file) {
        fprintf(csv_file, "function_id,function_name,pid,tid,start_timestamp_ns,end_timestamp_ns,duration_ns,arg_count");
        for (int i = 0; i < 8; i++) {
            fprintf(csv_file, ",arg%d_name,arg%d_value", i, i);
        }
        fprintf(csv_file, ",return_value\n");
        fflush(csv_file);
    }
}

// Function to write function call to CSV
static void write_function_call_to_csv(struct function_call *call) {
    if (!csv_file || !call->completed) {
        return;
    }

    __u64 duration = call->end_timestamp - call->start_timestamp;

    fprintf(csv_file, "%u,%s,%u,%u,%llu,%llu,%llu,%u",
            call->function_id,
            call->function_name,
            call->pid,
            call->tid,
            (unsigned long long)call->start_timestamp,
            (unsigned long long)call->end_timestamp,
            (unsigned long long)duration,
            call->arg_count);

    // Write arguments
    for (int i = 0; i < 8; i++) {
        if (i < call->arg_count) {
            const char* arg_name = get_argument_names(call->function_name, i);
            fprintf(csv_file, ",%s,0x%llx", arg_name, (unsigned long long)call->args[i]);
        } else {
            fprintf(csv_file, ",,");
        }
    }

    fprintf(csv_file, ",0x%llx\n", (unsigned long long)call->return_value);
    fflush(csv_file);
}

// Function to write function call to Perfetto
static void write_function_call_to_perfetto(struct function_call *call) {
    if (!perfetto_writer || !call->completed) {
        return;
    }

    // Skip events with zero timestamps to prevent Perfetto overflow errors
    if (call->start_timestamp == 0 || call->end_timestamp == 0 || call->pid == 0) {
        return;
    }

    // Add track for this process/thread if not already added
    static uint32_t last_pid = 0, last_tid = 0;
    if (call->pid != last_pid || call->tid != last_tid) {
        char process_name[64];
        snprintf(process_name, sizeof(process_name), "HIP Process %u", call->pid);
        perfetto_writer_add_track(perfetto_writer, call->pid, call->tid, process_name);
        last_pid = call->pid;
        last_tid = call->tid;
    }

    // Write slice begin event
    perfetto_writer_add_slice_begin(perfetto_writer, call->start_timestamp, call->function_name, call->pid, call->tid);

    // Write slice end event
    perfetto_writer_add_slice_end(perfetto_writer, call->end_timestamp, call->function_name, call->pid, call->tid);
}

// Function to write kernel dispatch events to Perfetto output
static void write_kernel_dispatch_to_perfetto(struct kernel_dispatch_event *kernel_event) {
    if (!perfetto_writer) {
        return;
    }

    // Add track for this process/thread if not already added
    static uint32_t last_pid = 0, last_tid = 0;
    if (kernel_event->pid != last_pid || kernel_event->tid != last_tid) {
        char process_name[64];
        snprintf(process_name, sizeof(process_name), "Kernel Dispatch Process %u", kernel_event->pid);
        perfetto_writer_add_track(perfetto_writer, kernel_event->pid, kernel_event->tid, process_name);
        last_pid = kernel_event->pid;
        last_tid = kernel_event->tid;
    }

    // Create event name with fence information
    char event_name[256];
    snprintf(event_name, sizeof(event_name), "%s (fence=%llu:%llu, ring=%s)",
             kernel_event->event_name,
             kernel_event->fence_context,
             kernel_event->fence_seqno,
             kernel_event->ring_name);

    // If we have duration information, write as a slice; otherwise as instant event
    if (kernel_event->duration > 0 && kernel_event->start_timestamp > 0) {
        // Write slice for kernel execution with duration
        perfetto_writer_add_slice_begin(perfetto_writer, kernel_event->start_timestamp, event_name,
                                        kernel_event->pid, kernel_event->tid);
        perfetto_writer_add_slice_end(perfetto_writer, kernel_event->timestamp, event_name,
                                      kernel_event->pid, kernel_event->tid);
    } else {
        // Write instant event for kernel dispatch (no duration)
        perfetto_writer_add_instant_event(perfetto_writer, kernel_event->timestamp, event_name,
                                          kernel_event->pid, kernel_event->tid);
    }
}

// Function to print event details
static void print_event(struct hip_event *event) {
    print_timestamp(event->timestamp);

    if (event->event_type == 0) {
        printf("ENTRY: %s (pid=%u, tid=%u) ",
               event->function_name, event->pid, event->tid);

        if (event->arg_count > 0) {
            printf("args=[");
            for (unsigned int i = 0; i < event->arg_count; i++) {
                printf("0x%llx", (unsigned long long)event->args[i]);
                if (i < event->arg_count - 1) printf(", ");
            }
            printf("]");
        }
    } else {
        printf("EXIT:  %s (pid=%u, tid=%u) ret=0x%llx",
               event->function_name, event->pid, event->tid, (unsigned long long)event->return_value);
    }

    printf("\n");
}

// Signal handler for graceful shutdown
static void sig_handler(int sig) {
    printf("\nReceived signal %d, shutting down...\n", sig);
    exiting = true;
}

// Function to find HIP library path
static char* find_hip_library() {
    const char* possible_paths[] = {
        "/opt/rocm/lib/libamdhip64.so",
        "/opt/rocm/lib/libhip_hcc.so",
        "/usr/lib/x86_64-linux-gnu/libamdhip64.so",
        "/usr/lib/x86_64-linux-gnu/libhip_hcc.so",
        "/usr/local/lib/libamdhip64.so",
        "/usr/local/lib/libhip_hcc.so",
        NULL
    };

    for (int i = 0; possible_paths[i] != NULL; i++) {
        if (access(possible_paths[i], R_OK) == 0) {
            return strdup(possible_paths[i]);
        }
    }

    return NULL;
}

// Function to get function offset using robust ELF parsing

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
off_t get_function_offset(const char *lib_path, const char *func_name) {
    if (!lib_path || !func_name) { errno = EINVAL; return -1; }

    if (elf_version(EV_CURRENT) == EV_NONE) return -1;

    int fd = open(lib_path, O_RDONLY);
    if (fd < 0) return -1;

    Elf *elf = elf_begin(fd, ELF_C_READ, NULL);
    if (!elf) { close(fd); return -1; }

    /* Prefer .dynsym (shared libs’ exported funcs), then fall back to .symtab */
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



// Function to attach uprobes to HIP functions
static int attach_uprobes(struct hip_trace_bpf *skel, const char *lib_path) {
    printf("Attaching uprobes to HIP library: %s\n", lib_path);
    printf("Using ELF parsing for accurate function offsets\n\n");

    struct bpf_link *link;
    off_t offset;

    // List of HIP functions to trace
    // Note: Reduced to key functions only to avoid recursive tracing issues
    // where one traced function calls another traced function
    const char* hip_functions[] = {
        "hipMalloc", "hipFree", "hipMemcpy", "hipMemcpyAsync", "hipMemset",
        "hipMemsetAsync", "hipMemGetInfo", "hipMemAllocManaged", "hipMemPrefetchAsync",
        "hipMemAdvise", "hipStreamCreate", "hipStreamDestroy", "hipStreamSynchronize",
        "hipStreamWaitEvent", "hipStreamQuery", "hipEventCreate", "hipEventDestroy",
        "hipEventRecord", "hipEventSynchronize", "hipEventElapsedTime", "hipLaunchKernel",
        "hipModuleLoad", "hipModuleUnload", "hipModuleGetFunction", "hipModuleLaunchKernel",
        "hipSetDevice", "hipGetDevice", "hipGetDeviceCount", "hipDeviceSynchronize",
        "hipDeviceReset"
    };

    int num_functions = sizeof(hip_functions) / sizeof(hip_functions[0]);
    int attached_count = 0;

    // Use generated attachment code for all HIP functions
    // This code is generated from HIP runtime headers
    attach_generated_hip_functions(skel, lib_path, &attached_count);

    printf("\nSuccessfully attached %d HIP functions\n", attached_count);
    return 0;
}

// Function to attach tracepoints for kernel dispatch tracing
static int attach_tracepoints(struct hip_trace_bpf *skel) {
    if (!enable_kernel_dispatches) {
        printf("Kernel dispatch tracing disabled by default (use -k to enable)\n");
        return 0;
    }

    printf("Attaching tracepoints for kernel dispatch tracing...\n");

    struct bpf_link *link;
    int err = 0;
    int attached_count = 0;

    // Attach amdgpu_cs_ioctl tracepoint
    link = bpf_program__attach_tracepoint(skel->progs.trace_amdgpu_cs_ioctl,
                                         "amdgpu", "amdgpu_cs_ioctl");
    if (!link) {
        printf("Warning: Failed to attach amdgpu_cs_ioctl tracepoint: %s\n", strerror(errno));
        // Don't fail completely - this might not be available on all systems
    } else {
        printf("✓ Attached amdgpu_cs_ioctl tracepoint\n");
        attached_count++;
    }

    // Attach amdgpu_sched_run_job tracepoint
    link = bpf_program__attach_tracepoint(skel->progs.trace_amdgpu_sched_run_job,
                                         "amdgpu", "amdgpu_sched_run_job");
    if (!link) {
        printf("Warning: Failed to attach amdgpu_sched_run_job tracepoint: %s\n", strerror(errno));
    } else {
        printf("✓ Attached amdgpu_sched_run_job tracepoint\n");
        attached_count++;
    }

    // Attach drm_sched_job_run tracepoint
    link = bpf_program__attach_tracepoint(skel->progs.trace_drm_sched_job_run,
                                         "gpu_scheduler", "drm_sched_job_run");
    if (!link) {
        printf("Warning: Failed to attach drm_sched_job_run tracepoint: %s\n", strerror(errno));
    } else {
        printf("✓ Attached drm_sched_job_run tracepoint\n");
        attached_count++;
    }

    // Attach drm_sched_job_done tracepoint
    link = bpf_program__attach_tracepoint(skel->progs.trace_drm_sched_job_done,
                                         "gpu_scheduler", "drm_sched_job_done");
    if (!link) {
        printf("Warning: Failed to attach drm_sched_job_done tracepoint: %s\n", strerror(errno));
    } else {
        printf("✓ Attached drm_sched_job_done tracepoint\n");
        attached_count++;
    }

    printf("\nSuccessfully attached %d tracepoints for kernel dispatch tracing\n", attached_count);

    // Return 0 even if some tracepoints failed - they might not be available on all systems
    return 0;
}

// Ring buffer callback function
static int handle_event(void *ctx __attribute__((unused)), void *data, size_t size __attribute__((unused))) {
    struct hip_event *event = (struct hip_event *)data;

    // Check if this is a kernel dispatch event
    if (event->event_type == 2) {
        struct kernel_dispatch_event *kernel_event = (struct kernel_dispatch_event *)data;

        // Print kernel dispatch event to console only if not using Perfetto output
        if (output_format != 2) {
            if (kernel_event->duration > 0) {
                printf("[%llu.%09llu] KERNEL_DISPATCH: %s (pid=%u, tid=%u) fence=%llu:%llu ring=%s duration=%llu ns\n",
                       kernel_event->timestamp / 1000000000,
                       kernel_event->timestamp % 1000000000,
                       kernel_event->event_name,
                       kernel_event->pid,
                       kernel_event->tid,
                       kernel_event->fence_context,
                       kernel_event->fence_seqno,
                       kernel_event->ring_name,
                       kernel_event->duration);
            } else {
                printf("[%llu.%09llu] KERNEL_DISPATCH: %s (pid=%u, tid=%u) fence=%llu:%llu ring=%s num_ibs=%u job_count=%u hw_job_count=%u client_id=%llu dev=%s\n",
                       kernel_event->timestamp / 1000000000,
                       kernel_event->timestamp % 1000000000,
                       kernel_event->event_name,
                       kernel_event->pid,
                       kernel_event->tid,
                       kernel_event->fence_context,
                       kernel_event->fence_seqno,
                       kernel_event->ring_name,
                       kernel_event->num_ibs,
                       kernel_event->job_count,
                       kernel_event->hw_job_count,
                       kernel_event->client_id,
                       kernel_event->device_name);
            }
        }

        // Write kernel dispatch event to Perfetto output
        if (output_format == 2 && perfetto_writer) {
            write_kernel_dispatch_to_perfetto(kernel_event);
        }

        return 0;
    }

    // Print HIP API event to console only if not using Perfetto output
    if (output_format != 2) {
        print_event(event);
    }

    // Process event for CSV output
    struct function_call *call = find_or_create_function_call(event->function_id);
    if (!call) {
        return 0;
    }

    if (event->event_type == 0) { // Entry
        // Parse function ID from function_name (e.g., "func_016" -> 16)
        int func_id = -1;
        if (strncmp(event->function_name, "func_", 5) == 0) {
            // Handle both "func_16" and "func_016" formats
            const char* id_str = event->function_name + 5;
            // Skip leading zeros
            while (*id_str == '0' && *(id_str + 1) != '\0') {
                id_str++;
            }
            func_id = atoi(id_str);
        }

        // Get actual function name from function ID mapping
        const char* func_name = NULL;
        if (func_id >= 0) {
            // Add bounds checking for function ID
            if (func_id < 500) { // Allow for all HIP functions (should be around 400+)
                func_name = get_function_name_by_id(func_id);
            } else {
                printf("Warning: Function ID %d out of range (max 499)\n", func_id);
                func_name = "unknown";
            }
            // Check if the returned pointer is valid
            if (!func_name || func_name == (void*)-1) {
                func_name = "unknown_function";
            }
        } else if (event->function_name[0]) {
            func_name = event->function_name;
        } else {
            func_name = determine_function_name(event->arg_count, event->args);
        }

        // Ensure func_name is not NULL before copying
        if (!func_name) {
            func_name = "unknown_function";
        }

        strncpy(call->function_name, func_name, sizeof(call->function_name) - 1);
        call->function_name[sizeof(call->function_name) - 1] = '\0';

        call->start_timestamp = event->timestamp;
        call->pid = event->pid;
        call->tid = event->tid;
        call->arg_count = event->arg_count;

        for (int i = 0; i < event->arg_count && i < 8; i++) {
            call->args[i] = event->args[i];
        }
    } else { // Exit
        call->end_timestamp = event->timestamp;
        call->return_value = event->return_value;
        call->completed = 1;

        // Write completed function call to output format
        if (output_format == 1) {
            write_function_call_to_csv(call);
        } else if (output_format == 2) {
            write_function_call_to_perfetto(call);
        }
    }

    return 0;
}

// Function to process ring buffer events
static int process_events(struct hip_trace_bpf *skel) {
    struct ring_buffer *rb = NULL;
    int err;

    // Create ring buffer
    rb = ring_buffer__new(bpf_map__fd(skel->maps.events), handle_event, NULL, NULL);

    if (!rb) {
        err = -1;
        printf("Failed to create ring buffer\n");
        goto cleanup;
    }

    printf("HIP API tracing started. Press Ctrl+C to stop.\n");
    printf("Format: [timestamp] ENTRY/EXIT: function_name (pid, tid) args/ret\n");
    if (output_format == 1) {
        printf("CSV output: enabled\n\n");
    } else if (output_format == 2) {
        printf("Perfetto output: enabled\n\n");
    } else {
        printf("Console output: enabled\n\n");
    }

    // Process events
    while (!exiting) {
        err = ring_buffer__poll(rb, 100); // 100ms timeout
        if (err == -EINTR) {
            err = 0;
            break;
        }
        if (err < 0) {
            printf("Error polling ring buffer: %d\n", err);
            break;
        }
    }

cleanup:
    if (rb) {
        ring_buffer__free(rb);
    }
    return err;
}

// Function to print usage
static void print_usage(const char *prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("Options:\n");
    printf("  -p <pid>     Attach to specific process ID\n");
    printf("  -l <lib>     Path to HIP library (default: auto-detect)\n");
    printf("  -o <file>    Output file for function calls\n");
    printf("  -f <format>  Output format: csv, perfetto (default: console)\n");
    printf("  -k           Enable kernel dispatch tracing\n");
    printf("  -h           Show this help message\n");
    printf("\n");
    printf("Examples:\n");
    printf("  %s                           # Trace all HIP processes (console)\n", prog_name);
    printf("  %s -p 1234                   # Trace process 1234\n", prog_name);
    printf("  %s -l /path/to/libhip.so     # Use specific HIP library\n", prog_name);
    printf("  %s -o trace.csv -f csv       # Output to CSV file\n", prog_name);
    printf("  %s -k -o trace.json -f perfetto  # Enable kernel dispatches\n", prog_name);
}

int main(int argc, char **argv) {
    struct hip_trace_bpf *skel;
    int err = 0;
    char *hip_lib_path = NULL;
    int opt;

    // Parse command line arguments
    char *output_file = NULL;
    char *format_str = NULL;

    while ((opt = getopt(argc, argv, "p:l:o:f:kh")) != -1) {
        switch (opt) {
        case 'p':
            // target_pid = atoi(optarg); // TODO: implement PID filtering
            break;
        case 'l':
            hip_lib_path = strdup(optarg);
            break;
        case 'o':
            output_file = strdup(optarg);
            break;
        case 'f':
            format_str = strdup(optarg);
            break;
        case 'k':
            enable_kernel_dispatches = true;
            break;
        case 'h':
            print_usage(argv[0]);
            return 0;
        default:
            print_usage(argv[0]);
            return 1;
        }
    }

    // Set output format and open output file
    if (output_file) {
        if (format_str) {
            if (strcmp(format_str, "csv") == 0) {
                output_format = 1;
                csv_file = fopen(output_file, "w");
                if (!csv_file) {
                    fprintf(stderr, "Failed to open CSV file: %s\n", output_file);
                    return 1;
                }
                printf("CSV output: %s\n", output_file);
                write_csv_header();
            } else if (strcmp(format_str, "perfetto") == 0) {
                output_format = 2;
                perfetto_writer = perfetto_writer_create(output_file);
                if (!perfetto_writer) {
                    fprintf(stderr, "Failed to create Perfetto writer for file: %s\n", output_file);
                    return 1;
                }
                printf("Perfetto output: %s\n", output_file);
            } else {
                fprintf(stderr, "Unknown format: %s. Use 'csv' or 'perfetto'\n", format_str);
                return 1;
            }
        } else {
            // Default to CSV if no format specified
            output_format = 1;
            csv_file = fopen(output_file, "w");
            if (!csv_file) {
                fprintf(stderr, "Failed to open output file: %s\n", output_file);
                return 1;
            }
            printf("CSV output: %s\n", output_file);
            write_csv_header();
        }
    }

    // Set up signal handlers
    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    // Find HIP library if not specified
    if (!hip_lib_path) {
        hip_lib_path = find_hip_library();
        if (!hip_lib_path) {
            printf("Error: Could not find HIP library. Please specify with -l option.\n");
            printf("Common locations:\n");
            printf("  /opt/rocm/lib/libamdhip64.so\n");
            printf("  /opt/rocm/lib/libhip_hcc.so\n");
            printf("  /usr/lib/x86_64-linux-gnu/libamdhip64.so\n");
            return 1;
        }
    }

    printf("Using HIP library: %s\n", hip_lib_path);

    // Load and verify eBPF program
    skel = hip_trace_bpf__open();
    if (!skel) {
        printf("Failed to open eBPF program\n");
        err = -1;
        goto cleanup;
    }

    // Load eBPF program
    err = hip_trace_bpf__load(skel);
    if (err) {
        printf("Failed to load eBPF program: %d\n", err);
        goto cleanup;
    }

    // Attach uprobes
    err = attach_uprobes(skel, hip_lib_path);
    if (err) {
        printf("Failed to attach uprobes: %d\n", err);
        goto cleanup;
    }

    // Attach tracepoints for kernel dispatch tracing
    err = attach_tracepoints(skel);
    if (err) {
        printf("Failed to attach tracepoints: %d\n", err);
        goto cleanup;
    }

    // Process events
    err = process_events(skel);

cleanup:
    if (skel) {
        hip_trace_bpf__destroy(skel);
    }
    if (hip_lib_path) {
        free(hip_lib_path);
    }
    if (output_file) {
        free(output_file);
    }
    if (format_str) {
        free(format_str);
    }
    if (csv_file) {
        fclose(csv_file);
    }
    if (perfetto_writer) {
        perfetto_writer_finalize(perfetto_writer);
        perfetto_writer_destroy(perfetto_writer);
    }
    if (function_calls) {
        free(function_calls);
    }

    return err < 0 ? -err : 0;
}
