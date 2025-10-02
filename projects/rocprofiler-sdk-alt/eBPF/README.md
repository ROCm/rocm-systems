# HIP eBPF Tracing Tool

A powerful eBPF-based tool for tracing HIP (Heterogeneous-compute Interface for Portability) API calls in real-time without modifying the target application. This tool captures function entry/exit events, arguments, return values, and execution durations with nanosecond precision.

## Features

- **Comprehensive HIP API tracing** - traces all 439+ HIP Runtime API functions using eBPF uprobes
- **Multiple output formats**: Console, CSV, and Perfetto trace format
- **Dynamic function generation** - automatically parses HIP headers to generate tracing code
- **Nanosecond precision** timestamps and duration measurements
- **Argument capture** with proper names and values
- **Process and thread tracking** for multi-threaded applications
- **Non-intrusive** - no application modification required
- **High performance** - minimal overhead on target application
- **Perfetto visualization** - view traces on ui.perfetto.dev
- **Call stack tracking** - handles nested function calls without recursion
- **Kernel dispatch duration tracking** - captures GPU kernel execution times

## Requirements

- Linux kernel with eBPF support (5.4+)
- Root privileges (for eBPF program loading)
- HIP runtime library
- Build tools: `clang`, `bpftool`, `libbpf-dev`
- CMake 3.10+ (required for build)
- Python 3 (for dynamic function generation)
- **System limits**: Increased file descriptor limit (65536) and memory lock limit for comprehensive tracing of all HIP functions

## Quick Start

### Build with CMake

```bash
# Configure and build
cmake -B build .
cmake --build build --parallel 16

# Run the tool (use wrapper script for automatic limit increase)
./hip_trace_wrapper.sh -l /opt/rocm/lib/libamdhip64.so -o trace.json -f perfetto
```

### Complete Example

```bash
# 1. Build the tool
cmake -B build .
cmake --build build --parallel 16

# 2. Run with your HIP application (CSV output)
sudo ./build/hip_trace -l /opt/rocm/lib/libamdhip64.so -o trace.csv -f csv &
TRACE_PID=$!

# 3. Run your HIP application
./your_hip_application

# 4. Stop tracing
kill $TRACE_PID

# 5. Analyze results
head -10 trace.csv
cut -d',' -f2 trace.csv | sort | uniq -c

# Alternative: Perfetto output for visualization
sudo ./build/hip_trace -l /opt/rocm/lib/libamdhip64.so -o trace.json -f perfetto &
TRACE_PID=$!
./your_hip_application
kill $TRACE_PID
# Open trace.json in ui.perfetto.dev
```

## Usage

### Command Line Options

```bash
./hip_trace [options]

Options:
  -p <pid>     Attach to specific process ID (not implemented yet)
  -l <lib>     Path to HIP library (default: auto-detect)
  -o <file>    Output file for function calls
  -f <format>  Output format: csv, perfetto (default: console)
  -h           Show help message
```

### Usage Examples

#### Basic Tracing
```bash
# Trace all HIP processes with console output
sudo ./build/hip_trace -l /opt/rocm/lib/libamdhip64.so

# Auto-detect HIP library
sudo ./build/hip_trace
```

#### CSV Output
```bash
# Generate CSV file for analysis
sudo ./build/hip_trace -l /opt/rocm/lib/libamdhip64.so -o trace.csv -f csv

# Run in background and analyze later
sudo ./build/hip_trace -o trace.csv -f csv &
TRACE_PID=$!
./your_hip_app
kill $TRACE_PID
# Analyze results
head -10 trace.csv
cut -d',' -f2 trace.csv | sort | uniq -c
```

#### Perfetto Output
```bash
# Generate Perfetto trace file for visualization
sudo ./build/hip_trace -l /opt/rocm/lib/libamdhip64.so -o trace.json -f perfetto

# Run in background and visualize later
sudo ./build/hip_trace -o trace.json -f perfetto &
TRACE_PID=$!
./your_hip_app
kill $TRACE_PID
# Open trace.json in ui.perfetto.dev
```

#### Common HIP Library Paths
```bash
# ROCm 7.1.0
sudo ./build/hip_trace -l /opt/rocm-7.1.0/lib/libamdhip64.so.7.1.70100 -o trace.csv -f csv

# Standard ROCm installation
sudo ./build/hip_trace -l /opt/rocm/lib/libamdhip64.so -o trace.csv -f csv

# System installation
sudo ./build/hip_trace -l /usr/lib/x86_64-linux-gnu/libamdhip64.so -o trace.csv -f csv
```

#### Testing with Sample Application
```bash
# Test with the provided test program
sudo ./build/hip_trace -l /opt/rocm/lib/libamdhip64.so -o test_trace.csv -f csv &
TRACE_PID=$!
sleep 2
./test_hip_program
kill $TRACE_PID
# Analyze results
head -10 test_trace.csv
cut -d',' -f2 test_trace.csv | sort | uniq -c
```

## Output Format

### Console Output
```
[timestamp] ENTRY: function_name (pid, tid) args=[arg1, arg2, ...]
[timestamp] EXIT:  function_name (pid, tid) ret=return_value
```

### CSV Output
The CSV file contains the following columns:
- `function_id` - Unique identifier for each function call
- `function_name` - Name of the HIP function
- `pid` - Process ID
- `tid` - Thread ID
- `start_timestamp_ns` - Function entry timestamp (nanoseconds)
- `end_timestamp_ns` - Function exit timestamp (nanoseconds)
- `duration_ns` - Function execution duration (nanoseconds)
- `arg_count` - Number of arguments captured
- `arg0_name, arg0_value, arg1_name, arg1_value, ...` - Function arguments
- `return_value` - Function return value

### Perfetto Output
The Perfetto trace file (`.json`) contains:
- **Track events** for each HIP function call
- **Slice begin/end events** showing function execution duration
- **Process and thread information** for multi-threaded applications
- **Timestamps** with nanosecond precision
- **Function names** and argument information

The Perfetto format enables:
- **Interactive visualization** on ui.perfetto.dev
- **Timeline analysis** of HIP API calls
- **Performance profiling** with detailed timing information
- **Multi-process/thread correlation** for complex applications

## Supported HIP Functions

Comprehensive support for **all 439 HIP Runtime API functions** (100% coverage) across all major categories:

### Memory Management (25+ functions)
- `hipMalloc`, `hipFree`, `hipMemcpy`, `hipMemcpyAsync`
- `hipMemset`, `hipMemsetAsync`, `hipMemGetInfo`
- `hipMemPrefetchAsync`, `hipMemAdvise`, `hipMemAllocHost`
- `hipHostMalloc`, `hipMallocManaged`, `hipMallocHost`
- `hipMemAddressFree`, `hipMemAddressReserve`, `hipMemCreate`
- `hipMemExportToShareableHandle`, `hipMemGetAccess`
- `hipMemGetAllocationGranularity`, `hipMemImportFromShareableHandle`
- `hipMemMap`, `hipMemMapArrayAsync`, `hipMemRelease`
- `hipMemRetainAllocationHandle`, `hipMemSetAccess`, `hipMemUnmap`

### Stream Management (20+ functions)
- `hipStreamCreate`, `hipStreamDestroy`, `hipStreamSynchronize`
- `hipStreamWaitEvent`, `hipStreamQuery`, `hipStreamCreateWithFlags`
- `hipStreamCreateWithPriority`, `hipStreamGetFlags`, `hipStreamGetId`
- `hipStreamGetPriority`, `hipStreamGetDevice`, `hipStreamAddCallback`
- `hipStreamSetAttribute`, `hipStreamGetAttribute`
- `hipStreamWaitValue32`, `hipStreamWaitValue64`
- `hipStreamWriteValue32`, `hipStreamWriteValue64`
- `hipStreamBatchMemOp`, `hipExtStreamCreateWithCUMask`

### Event Management (10+ functions)
- `hipEventCreate`, `hipEventDestroy`, `hipEventRecord`
- `hipEventSynchronize`, `hipEventElapsedTime`, `hipEventQuery`
- `hipEventCreateWithFlags`, `hipEventRecordWithFlags`

### Kernel Launch & Execution
- `hipLaunchKernel` - Launch kernel (with call stack tracking)
- `hipFuncSetAttribute`, `hipFuncSetCacheConfig`, `hipFuncSetSharedMemConfig`
- **Kernel dispatch tracking** - Captures GPU kernel execution duration via DRM scheduler events

### Device Management (30+ functions)
- `hipSetDevice`, `hipGetDevice`, `hipGetDeviceCount`
- `hipDeviceSynchronize`, `hipDeviceReset`, `hipDeviceGet`
- `hipDeviceComputeCapability`, `hipDeviceGetName`, `hipDeviceGetUuid`
- `hipDeviceGetP2PAttribute`, `hipDeviceGetPCIBusId`, `hipDeviceGetByPCIBusId`
- `hipDeviceTotalMem`, `hipDeviceGetAttribute`, `hipDeviceGetDefaultMemPool`
- `hipDeviceSetMemPool`, `hipDeviceGetMemPool`, `hipGetDeviceProperties`
- `hipDeviceSetCacheConfig`, `hipDeviceGetCacheConfig`
- `hipDeviceGetLimit`, `hipDeviceSetLimit`
- `hipDeviceGetSharedMemConfig`, `hipDeviceSetSharedMemConfig`
- `hipGetDeviceFlags`, `hipSetDeviceFlags`, `hipChooseDevice`

### Graph Management (50+ functions)
- `hipGraphCreate`, `hipGraphDestroy`, `hipGraphAddKernelNode`
- `hipGraphAddMemcpyNode`, `hipGraphAddMemsetNode`, `hipGraphAddMemFreeNode`
- `hipGraphAddHostNode`, `hipGraphAddChildGraphNode`, `hipGraphAddEmptyNode`
- `hipGraphAddEventRecordNode`, `hipGraphAddEventWaitNode`
- `hipGraphAddExternalSemaphoresWaitNode`, `hipGraphAddExternalSemaphoresSignalNode`
- `hipGraphExecCreate`, `hipGraphExecDestroy`, `hipGraphExecUpdate`
- `hipGraphExecKernelNodeSetParams`, `hipGraphExecMemcpyNodeSetParams`
- `hipGraphExecMemsetNodeSetParams`, `hipGraphExecHostNodeSetParams`
- `hipGraphExecChildGraphNodeSetParams`, `hipGraphExecEventRecordNodeSetParams`
- `hipGraphExecEventWaitNodeSetParams`, `hipGraphExecExternalSemaphoresWaitNodeSetParams`
- `hipGraphExecExternalSemaphoresSignalNodeSetParams`
- `hipGraphLaunch`, `hipGraphInstantiate`, `hipGraphInstantiateWithFlags`
- `hipGraphKernelNodeCopyAttributes`, `hipGraphNodeSetEnabled`, `hipGraphNodeGetEnabled`
- `hipGraphRetainUserObject`, `hipGraphReleaseUserObject`
- `hipGraphDebugDotPrint`, `hipGraphAddBatchMemOpNode`
- `hipGraphBatchMemOpNodeGetParams`, `hipGraphBatchMemOpNodeSetParams`
- `hipGraphExecBatchMemOpNodeSetParams`

### IPC & External Resources (15+ functions)
- `hipIpcGetMemHandle`, `hipIpcOpenMemHandle`, `hipIpcCloseMemHandle`
- `hipIpcGetEventHandle`, `hipIpcOpenEventHandle`
- `hipImportExternalSemaphore`, `hipSignalExternalSemaphoresAsync`
- `hipWaitExternalSemaphoresAsync`, `hipDestroyExternalSemaphore`
- `hipImportExternalMemory`, `hipExternalMemoryGetMappedBuffer`
- `hipDestroyExternalMemory`, `hipExternalMemoryGetMappedMipmappedArray`

### Graphics & Surface Management (10+ functions)
- `hipGraphicsMapResources`, `hipGraphicsSubResourceGetMappedArray`
- `hipGraphicsResourceGetMappedPointer`, `hipGraphicsUnmapResources`
- `hipGraphicsUnregisterResource`, `hipCreateSurfaceObject`
- `hipDestroySurfaceObject`

### User Objects & Advanced Features (10+ functions)
- `hipUserObjectCreate`, `hipUserObjectRelease`, `hipUserObjectRetain`
- `hipPointerSetAttribute`, `hipPointerGetAttributes`, `hipPointerGetAttribute`
- `hipDrvPointerGetAttributes`, `hipGetLastError`, `hipPeekAtLastError`
- `hipInit`, `hipDriverGetVersion`, `hipRuntimeGetVersion`

## Analysis Tools

### CSV Analysis
```bash
# Basic analysis commands
head -10 trace.csv                    # View first 10 entries
cut -d',' -f2 trace.csv | sort | uniq -c  # Count function calls
sort -t',' -k7 -nr trace.csv | head -10   # Longest running calls
```

### Perfetto Visualization
```bash
# Generate Perfetto trace
sudo ./build/hip_trace -o trace.json -f perfetto &
TRACE_PID=$!
./your_hip_application
kill $TRACE_PID

# Open in Perfetto UI
# Navigate to https://ui.perfetto.dev
# Click "Open trace file" and select trace.json
```

Perfetto UI provides:
- **Interactive timeline** with zoom and pan
- **Function call visualization** as colored slices
- **Performance metrics** and statistics
- **Multi-thread analysis** with thread tracks
- **Search and filtering** capabilities
- **Export options** for further analysis

### Python Analysis
```python
import pandas as pd
import numpy as np

# Load the CSV data
df = pd.read_csv('trace.csv')

# Basic statistics
print("Function call summary:")
print(df.groupby('function_name')['duration_ns'].describe())

# Performance analysis
print("\nPerformance metrics:")
print(f"Total calls: {len(df)}")
print(f"Average duration: {df['duration_ns'].mean() / 1e6:.3f} ms")
print(f"Max duration: {df['duration_ns'].max() / 1e6:.3f} ms")

# Memory allocation analysis
if 'hipMalloc' in df['function_name'].values:
    malloc_calls = df[df['function_name'] == 'hipMalloc']
    print(f"\nMemory allocations: {len(malloc_calls)}")
    print(f"Average allocation time: {malloc_calls['duration_ns'].mean() / 1e6:.3f} ms")
```

### Command Line Analysis
```bash
# Count function calls
cut -d',' -f2 trace.csv | sort | uniq -c

# Find longest running calls
sort -t',' -k7 -nr trace.csv | head -10

# Analyze by process
awk -F',' 'NR>1 {print $3}' trace.csv | sort | uniq -c

# Calculate average duration by function
awk -F',' 'NR>1 {sum[$2]+=$7; count[$2]++} END {for(f in sum) print f, sum[f]/count[f]/1e6 " ms"}' trace.csv
```

## Architecture

The tool consists of:

1. **eBPF Program** (`hip_trace.bpf.c`) - Kernel-space tracing logic with:
   - 878 uprobes (439 functions × 2) for HIP API tracing
   - Call stack tracking to handle nested function calls
   - Kernel dispatch tracepoints for GPU execution duration
2. **User-space Loader** (`hip_trace.c`) - Program loading and event processing
3. **Dynamic Code Generation** (`generate_hip_functions_from_headers.py`) - Parses HIP headers and generates eBPF programs
4. **Build System** - CMake-based build with automatic function generation
5. **Wrapper Script** (`hip_trace_wrapper.sh`) - Handles system limit requirements
6. **Analysis Tools** - CSV analysis and visualization scripts

## How It Works

1. **Dynamic Code Generation**: Python script parses HIP headers and generates eBPF programs for all 439 HIP Runtime API functions
2. **Uprobe Attachment**: Attaches 878 eBPF uprobes (entry + exit) to HIP library functions
   - **Call Stack Tracking**: Uses depth-based keys `(tid << 16) | depth` to handle nested function calls
   - **Recursion Prevention**: Prevents infinite recursion when traced functions call other traced functions
3. **Event Capture**: Captures function entry/exit events with arguments and timestamps
4. **Kernel Dispatch Tracking**: Monitors DRM scheduler events (`drm_sched_job_run`, `drm_sched_job_done`) for GPU kernel execution duration
5. **Data Processing**: Processes events in user-space with function name mapping
6. **Output Generation**: Writes structured data to console, CSV, or Perfetto JSON format

For detailed technical explanation, see [eBPF_HIP_Tracing_Technical_Guide.md](eBPF_HIP_Tracing_Technical_Guide.md).

## Performance Impact

- **Minimal overhead** on target application
- **High throughput** event processing
- **Efficient memory usage** with ring buffers
- **Real-time processing** without blocking target application

## Troubleshooting

### Common Issues

1. **Permission denied**: Run with `sudo` (eBPF requires root privileges)
2. **Library not found**: Specify HIP library path with `-l` option
3. **No events captured**: Ensure target application uses HIP functions
4. **Build errors**: Install required dependencies (`libbpf-dev`, `clang`, `bpftool`)
5. **Recursive calls**: Fixed with call stack tracking - no longer occurs
6. **Missing kernel dispatch duration**: Ensure `drm_sched_job_done` tracepoint is enabled

### Build Issues

#### CMake Build Fails
```bash
# Clean and rebuild
rm -rf build
cmake -B build .
cmake --build build --parallel 16
```

#### Missing Dependencies
```bash
# Install required packages
sudo apt update
sudo apt install libbpf-dev clang bpftool libelf-dev zlib1g-dev python3
```

#### System Limit Issues
```bash
# If you get "Too many open files" error, use the wrapper script
./hip_trace_wrapper.sh -l /opt/rocm/lib/libamdhip64.so -o trace.json -f perfetto

# Or manually increase limits
sudo sh -c 'ulimit -n 65536 && ./build/hip_trace -l /opt/rocm/lib/libamdhip64.so -o trace.json -f perfetto'
```

### Runtime Issues

#### No Events Captured
```bash
# Check if HIP library is found
ldd /path/to/your/hip/app | grep hip

# Verify library path
sudo ./build/hip_trace -l /opt/rocm/lib/libamdhip64.so -o trace.csv
```

#### High CPU Usage
```bash
# Reduce sampling rate by modifying the eBPF program
# Or use filtering in the user-space code
```

### Debug Mode

Enable verbose output by modifying the source code or using debug builds:
```bash
# Debug build
cmake -B build -DCMAKE_BUILD_TYPE=Debug .
cmake --build build
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is licensed under the GPL-2.0 License - see the LICENSE file for details.

## Acknowledgments

- Built with [libbpf](https://github.com/libbpf/libbpf)
- Uses eBPF uprobes for user-space tracing
- Inspired by modern observability tools

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the technical guide
3. Open an issue on the repository

## Quick Reference

### Build Commands
```bash
# CMake build (required)
cmake -B build .
cmake --build build --parallel 16
```

### Run Commands
```bash
# Basic tracing (use wrapper script for automatic limit increase)
./hip_trace_wrapper.sh -l /opt/rocm/lib/libamdhip64.so

# CSV output
./hip_trace_wrapper.sh -l /opt/rocm/lib/libamdhip64.so -o trace.csv -f csv

# Perfetto output
./hip_trace_wrapper.sh -l /opt/rocm/lib/libamdhip64.so -o trace.json -f perfetto

# Background tracing
./hip_trace_wrapper.sh -o trace.csv -f csv &
TRACE_PID=$!
./your_hip_app
kill $TRACE_PID
```

### Analysis Commands
```bash
# Function call count
cut -d',' -f2 trace.csv | sort | uniq -c

# Longest calls
sort -t',' -k7 -nr trace.csv | head -10

# View sample data
head -10 trace.csv
```