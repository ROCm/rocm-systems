#!/usr/bin/env python3
"""
Generate eBPF programs for HIP API functions from HIP runtime headers.
This script parses hip_runtime_api.h and generates the necessary eBPF code.
"""

import re
import sys
import os
from typing import List, Dict, Tuple

def extract_hip_functions(header_file: str) -> List[Dict]:
    """Extract HIP function signatures from the header file."""
    functions = []
    seen_functions = set()

    with open(header_file, 'r') as f:
        content = f.read()

    # Pattern to match hipError_t hipFunctionName(...) declarations
    # This captures the function name and parameters, handling multi-line declarations
    # The pattern now handles functions that span multiple lines
    pattern = r'hipError_t\s+(hip\w+)\s*\(([^;]*?)\)\s*;'

    matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)

    for func_name, params in matches:
        # Skip duplicates
        if func_name in seen_functions:
            continue
        seen_functions.add(func_name)

        # Parse parameters
        param_list = []
        if params.strip():
            # Remove newlines and extra whitespace
            params_clean = re.sub(r'\s+', ' ', params.strip())
            # Split by comma and clean up
            param_parts = [p.strip() for p in params_clean.split(',')]
            for part in param_parts:
                # Extract parameter name (last word)
                words = part.split()
                if words:
                    param_name = words[-1]
                    # Remove array brackets and pointers
                    param_name = re.sub(r'[\[\]*]', '', param_name)
                    param_list.append(param_name)

        functions.append({
            'name': func_name,
            'params': param_list,
            'param_count': len(param_list)
        })

    return functions

def generate_ebpf_programs(functions: List[Dict]) -> str:
    """Generate eBPF program code for the functions."""

    # Add required headers and includes
    header_code = '''#define __TARGET_ARCH_x86
#include "../vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

// Helper functions (these should be defined in the main eBPF file)
static __always_inline void get_pid_tid(__u32 *pid, __u32 *tid) {
    *pid = bpf_get_current_pid_tgid() >> 32;
    *tid = bpf_get_current_pid_tgid() & 0xFFFFFFFF;
}

static __always_inline __u64 get_timestamp() {
    return bpf_ktime_get_ns();
}

static __always_inline __u32 get_next_function_id() {
    return bpf_get_current_pid_tgid() & 0xFFFFFFFF;
}

// Event structure (should match the main eBPF file)
struct hip_event {
    __u64 timestamp;
    __u32 pid, tid;
    __u32 event_type; // 0 = entry, 1 = exit
    char function_name[64];
    __u64 args[8];
    __u64 return_value;
    __u32 arg_count;
    __u32 function_id;
};

// Maps (these should be defined in the main eBPF file)
struct {
    __uint(type, BPF_MAP_TYPE_RINGBUF);
    __uint(max_entries, 256 * 1024);
} events SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u32); // tid
    __type(value, __u64); // timestamp
    __uint(max_entries, 1024);
} entry_times SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u32); // tid
    __type(value, char[64]); // function name
    __uint(max_entries, 1024);
} function_names SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __type(key, __u32); // tid
    __type(value, __u32); // function id
    __uint(max_entries, 1024);
} function_ids SEC(".maps");

// Helper function to create exit event (should be defined in main eBPF file)
static __always_inline int create_exit_event(struct pt_regs *ctx) {
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 exit_timestamp = get_timestamp();

    // Get entry time
    __u64 *entry_time = bpf_map_lookup_elem(&entry_times, &tid);
    if (!entry_time) {
        return 0;
    }

    // Get function name and ID
    char *func_name = bpf_map_lookup_elem(&function_names, &tid);
    if (!func_name) {
        return 0;
    }

    __u32 *function_id = bpf_map_lookup_elem(&function_ids, &tid);
    if (!function_id) {
        return 0;
    }

    // Create event
    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {
        return 0;
    }

    event->timestamp = exit_timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 1; // exit
    event->function_id = *function_id;
    bpf_probe_read_user_str(event->function_name, sizeof(event->function_name), func_name);

    // Capture return value
    event->return_value = PT_REGS_RC(ctx);
    event->arg_count = 0; // No args for exit

    bpf_ringbuf_submit(event, 0);

    // Clean up maps
    bpf_map_delete_elem(&entry_times, &tid);
    bpf_map_delete_elem(&function_names, &tid);
    bpf_map_delete_elem(&function_ids, &tid);

    return 0;
}

'''

    # Generate entry programs
    entry_programs = []
    exit_programs = []

    for func in functions:
        func_name = func['name']
        param_count = func['param_count']
        params = func['params']

        # Convert function name to eBPF program name
        # eBPF programs use underscores between words, e.g., hipDeviceSynchronize -> hip_device_synchronize
        prog_name = func_name.lower()
        # Insert underscores before capital letters (except the first one)
        import re
        prog_name = re.sub(r'(?<!^)(?=[A-Z])', '_', prog_name).lower()

        # Generate entry program
        entry_code = f'''SEC("uprobe/{prog_name}_entry")
int {prog_name}_entry(struct pt_regs *ctx) {{
    __u32 pid, tid;
    get_pid_tid(&pid, &tid);

    __u64 timestamp = get_timestamp();
    __u32 function_id = get_next_function_id();

    bpf_map_update_elem(&entry_times, &tid, &timestamp, BPF_ANY);

    char func_name[64] = "{func_name}";
    bpf_map_update_elem(&function_names, &tid, func_name, BPF_ANY);
    bpf_map_update_elem(&function_ids, &tid, &function_id, BPF_ANY);

    struct hip_event *event = bpf_ringbuf_reserve(&events, sizeof(*event), 0);
    if (!event) {{
        return 0;
    }}

    event->timestamp = timestamp;
    event->pid = pid;
    event->tid = tid;
    event->event_type = 0;
    event->function_id = function_id;
    bpf_probe_read_user_str(event->function_name, sizeof(event->function_name), func_name);

    event->arg_count = {param_count};'''

        # Add parameter capture (limit to 5 parameters to avoid PT_REGS_PARM6+ issues)
        for i, param in enumerate(params[:5]):  # Limit to 5 parameters
            entry_code += f'''
    event->args[{i}] = PT_REGS_PARM{i+1}(ctx); // {param}'''

        entry_code += '''
    event->return_value = 0;

    bpf_ringbuf_submit(event, 0);
    return 0;
}'''

        entry_programs.append(entry_code)

        # Generate exit program
        exit_code = f'''SEC("uretprobe/{prog_name}_exit")
int {prog_name}_exit(struct pt_regs *ctx) {{
    return create_exit_event(ctx);
}}'''

        exit_programs.append(exit_code)

    return header_code + '\n\n'.join(entry_programs) + '\n\n' + '\n\n'.join(exit_programs)

def generate_user_space_attachment(functions: List[Dict]) -> str:
    """Generate user-space attachment code."""

    attachment_code = '''#include <stdio.h>
#include <stdbool.h>
#include <string.h>
#include <bpf/libbpf.h>
#include "hip_trace.skel.h"

// Forward declaration
long get_function_offset(const char *lib_path, const char *func_name);

// Function ID to name mapping
const char* get_function_name_by_id(int func_id) {
    static const char* function_names[] = {'''

    for i, func in enumerate(functions):
        attachment_code += f'''
        "{func['name']}",  // {i}'''

    attachment_code += '''
    };

    int num_functions = sizeof(function_names) / sizeof(function_names[0]);
    if (func_id >= 0 && func_id < num_functions) {
        return function_names[func_id];
    }
    return "unknown";
}

// Function to attach all generated HIP functions
int attach_generated_hip_functions(struct hip_trace_bpf *skel, const char *lib_path, int *attached_count) {
    struct bpf_link *link;
    long offset;

    // Attach specific programs for all HIP functions
    struct {
        const char *func_name;
        const char *entry_prog;
        const char *exit_prog;
    } hip_function_programs[] = {'''

    for func in functions:
        func_name = func['name']
        # Convert function name to eBPF program name
        # eBPF programs use underscores between words, e.g., hipDeviceSynchronize -> hip_device_synchronize
        prog_name = func_name
        # Insert underscores before capital letters (except the first one)
        prog_name = re.sub(r'(?<!^)(?=[A-Z])', '_', prog_name).lower()
        attachment_code += f'''
        {{"{func_name}", "{prog_name}_entry", "{prog_name}_exit"}},'''

    attachment_code += '''
    };

    int num_programs = sizeof(hip_function_programs) / sizeof(hip_function_programs[0]);

    for (int i = 0; i < num_programs; i++) {
        offset = get_function_offset(lib_path, hip_function_programs[i].func_name);
        if (offset > 0) {
            // Find the specific programs
            struct bpf_program *entry_prog = bpf_object__find_program_by_name(skel->obj, hip_function_programs[i].entry_prog);
            struct bpf_program *exit_prog = bpf_object__find_program_by_name(skel->obj, hip_function_programs[i].exit_prog);

            if (entry_prog && exit_prog) {
                link = bpf_program__attach_uprobe(entry_prog, false, -1, lib_path, offset);
                if (link) {
                    bpf_program__attach_uprobe(exit_prog, true, -1, lib_path, offset);
                    printf("Attached uprobes for %s (offset 0x%lx)\\n", hip_function_programs[i].func_name, offset);
                    (*attached_count)++;
                } else {
                    printf("Failed to attach uprobes for %s (offset 0x%lx)\\n", hip_function_programs[i].func_name, offset);
                }
            } else {
                printf("Failed to find programs for %s\\n", hip_function_programs[i].func_name);
            }
        } else {
            printf("Failed to find offset for %s\\n", hip_function_programs[i].func_name);
        }
    }

    return 0;
}'''

    return attachment_code

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 generate_hip_functions_from_headers.py <header_file> <output_dir>")
        sys.exit(1)

    header_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(header_file):
        print(f"Error: Header file {header_file} not found")
        sys.exit(1)

    # Extract functions
    print(f"Parsing HIP functions from {header_file}...")
    all_functions = extract_hip_functions(header_file)
    print(f"Found {len(all_functions)} HIP functions")

    # Include all HIP Runtime API functions
    # Filter out deprecated or internal functions
    excluded_functions = {
        'hip_init',  # Internal function
        'hipDrvGetErrorName', 'hipDrvGetErrorString',  # Driver API functions
        'hipExtGetLinkTypeAndHopCount', 'hipExtGetLastError',  # Extension functions
    }

    functions = [f for f in all_functions if f['name'] not in excluded_functions]
    print(f"Including {len(functions)} HIP Runtime API functions")

    # Generate eBPF programs for all HIP functions
    print("Generating eBPF programs for all HIP functions...")
    ebpf_code = generate_ebpf_programs(functions)

    # Write eBPF code
    ebpf_file = os.path.join(output_dir, "hip_functions_generated.bpf.c")
    with open(ebpf_file, 'w') as f:
        f.write("// Auto-generated eBPF programs for HIP functions\n")
        f.write("// Generated from HIP runtime headers\n\n")
        f.write(ebpf_code)

    # Generate user-space attachment code
    print("Generating user-space attachment code...")
    attachment_code = generate_user_space_attachment(functions)

    # Write attachment code
    attachment_file = os.path.join(output_dir, "hip_functions_generated.c")
    with open(attachment_file, 'w') as f:
        f.write("// Auto-generated user-space attachment code for HIP functions\n")
        f.write("// Generated from HIP runtime headers\n\n")
        f.write(attachment_code)

    print(f"Generated {len(functions)} HIP function programs")
    print(f"eBPF code written to: {ebpf_file}")
    print(f"Attachment code written to: {attachment_file}")

    # Print summary
    print("\nFunctions found:")
    for func in functions[:10]:  # Show first 10
        print(f"  {func['name']} ({func['param_count']} params)")
    if len(functions) > 10:
        print(f"  ... and {len(functions) - 10} more")

if __name__ == "__main__":
    main()
