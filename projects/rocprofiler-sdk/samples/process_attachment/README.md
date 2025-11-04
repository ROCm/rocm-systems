# Process Attachment Sample

This sample demonstrates how to attach profiling tools to already-running HIP applications using the ROCProfiler SDK. This is useful for profiling long-running applications without restarting them.

## Overview

The sample consists of three components:

1. **process-attachment-target** - A sample HIP application that runs continuous GPU workloads
2. **process-attachment-client** - A utility that attaches profiling tools to running processes
3. **process-attachment-tool** - A shared library that gets dynamically loaded into target processes to collect profiling data

## Components

### 1. Target Application (main.cpp)

A long-running HIP application that executes various GPU workloads in a continuous loop:

- **Vector Addition** - Simple element-wise vector operations
- **Matrix Multiplication** - 512x512 matrix multiply operations
- **Compute Intensive Workload** - Iterative trigonometric and math operations
- **ROCTX Markers** - Annotated regions for profiling visibility

The application runs workload cycles with configurable duration and includes ROCTX markers for identifying different execution phases.

### 2. Attachment Client (client.cpp)

A command-line utility that:

- Loads the `librocprofv3-attach.so` library
- Attaches to a running process by PID
- Injects the profiling tool library into the target process
- Monitors the profiling session
- Cleanly detaches when profiling is complete

### 3. Profiling Tool (tool.cpp, tool.hpp)

A shared library that:

- Gets dynamically loaded into the target process
- Initializes ROCProfiler SDK within the target
- Collects kernel and HIP API trace data
- Provides statistics about the profiling session

## Building

This sample is built as part of the rocprofiler-sdk samples suite:

```bash
cd projects/rocprofiler-sdk
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target process-attachment-target
cmake --build build --target process-attachment-client
cmake --build build --target process-attachment-tool
```

Or build all samples:

```bash
cmake --build build --parallel
```

## Usage

### Basic Workflow

1. **Start the target application:**

```bash
./process-attachment-target -t 300
```

This starts the HIP workload application that will run for 300 seconds. The application will print its PID.

Example output:
```
=== Target Application for Process Attachment ===
PID: 12345
Will run for 300 seconds
Found 1 HIP device(s)
Using device: AMD Radeon RX 7900 XTX
```

2. **Attach the profiler (in a separate terminal):**

```bash
./process-attachment-client \
    --tool-lib ./libprocess-attachment-tool.so \
    --pid 12345 \
    --duration 30
```

This attaches the profiling tool to process 12345 and profiles for 30 seconds.

### Target Application Options

```
Usage: ./process-attachment-target [options]

Options:
  -t, --time <seconds>     Run for specified time (default: 60 seconds)
  -h, --help              Show help message

Examples:
  ./process-attachment-target                 # Run for 60 seconds
  ./process-attachment-target -t 300          # Run for 5 minutes
```

### Client Application Options

```
Usage: ./process-attachment-client [options] <pid>

Options:
  --tool-lib <path>        Path to tool library (required)
  --pid <pid>             Target process ID (required)
  -d, --duration <seconds> Attach for specified duration (default: 10)
  --interactive            Interactive mode - press Enter to detach
  -h, --help              Show help message

Examples:
  ./process-attachment-client --tool-lib ./tool.so --pid 1234
  ./process-attachment-client --tool-lib ./tool.so --pid 1234 -d 30
  ./process-attachment-client --tool-lib ./tool.so --pid 1234 --interactive
```

### Interactive Mode

In interactive mode, the profiler remains attached until you press Enter:

```bash
./process-attachment-client \
    --tool-lib ./libprocess-attachment-tool.so \
    --pid 12345 \
    --interactive
```

Press Enter when you want to stop profiling and detach.

## Example Session

**Terminal 1 - Start target application:**
```bash
$ ./process-attachment-target -t 600
=== Target Application for Process Attachment ===
PID: 42157
Will run for 600 seconds
Found 1 HIP device(s)
Using device: AMD Radeon RX 7900 XTX
Global memory: 24564 MB
Allocated 4 device buffers of 4 MB each
=== Cycle 1 ===
Completed cycle 1 (PID: 42157)
=== Cycle 2 ===
...
```

**Terminal 2 - Attach profiler:**
```bash
$ ./process-attachment-client --tool-lib ./libprocess-attachment-tool.so --pid 42157 -d 30
Process Attachment Client
=========================
Target PID: 42157
Tool Library: ./libprocess-attachment-tool.so
Attachment Mode: timed (30 seconds)
Loaded attachment library: /opt/rocm/lib/rocprofiler-sdk/librocprofv3-attach.so
Attaching to process 42157...
Tool library: ./libprocess-attachment-tool.so
Successfully attached to process 42157
Profiling for 30 seconds...
Profiling... 10 seconds elapsed
Profiling... 20 seconds elapsed
Profiling duration of 30 seconds completed
Detaching from process 42157...
Successfully detached from process
Process attachment session completed
```

## Implementation Details

### Process Attachment Flow

1. **Client initialization:**
   - Loads `librocprofv3-attach.so`
   - Resolves `attach()` and `detach()` function symbols

2. **Attachment:**
   - Sets `ROCPROF_ATTACH_TOOL_LIBRARY` environment variable
   - Calls `attach(pid)` to inject tool into target process
   - Target process loads the tool library dynamically

3. **Profiling:**
   - Tool library initializes ROCProfiler SDK in target process
   - Configures kernel and API tracing
   - Collects profiling data

4. **Detachment:**
   - Client calls `detach()`
   - Tool library finalizes and outputs statistics
   - Tool library is unloaded from target process

### ROCTX Integration

The target application uses ROCTX markers to annotate execution:

```cpp
roctxRangePushA("Vector Addition Workload");
// ... GPU work ...
roctxRangePop();

roctxMark("Vector addition kernel launch");
```

These markers appear in the profiling output for better analysis.

### Workload Characteristics

Each workload cycle includes:

- **Vector Addition:** ~1M element vectors, memory-bound operation
- **Matrix Multiplication:** 512x512 matrices, compute-bound operation
- **Compute Intensive:** 1M elements with 1000 iterations each, highly compute-bound

The workloads are separated by sleep intervals to create distinct profiling regions.

## Requirements

- ROCm 7.0 or later
- HIP-capable AMD GPU
- ROCProfiler SDK with process attachment support
- `librocprofv3-attach.so` library

## Testing

The sample includes automated tests:

```bash
# Test target application runs correctly
ctest -R process-attachment-target-test

# Test full attachment scenario (if enabled)
ctest -R process-attachment-integration-test
```

## Troubleshooting

### Common Issues

1. **"Failed to load rocprofv3-attach library"**
   - Ensure ROCm is installed and `librocprofv3-attach.so` is available
   - Check `LD_LIBRARY_PATH` includes ROCm library paths

2. **"Target process is not accessible"**
   - Verify the PID is correct
   - Ensure you have permissions to attach to the process
   - Check if process is still running

3. **"Attachment failed"**
   - Verify the tool library path is correct and accessible
   - Check target process is a HIP application
   - Review ROCm installation and version compatibility

4. **"Target process has exited unexpectedly"**
   - Normal if target application completes before profiling finishes
   - Increase target runtime with `-t` option

## Files

- `main.cpp` - Target HIP application with continuous workloads
- `client.cpp` - Process attachment client utility
- `tool.cpp` - Profiling tool library implementation
- `tool.hpp` - Tool library interface
- `CMakeLists.txt` - Build configuration
- `README.md` - This file


## See Also

- [ROCProfiler SDK Documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/)

