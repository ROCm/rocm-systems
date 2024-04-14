# rocprofv3 user guide

ROCProfiler SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm platform. It supports application tracing to provide a big picture of the GPU application execution and kernel profiling to provide low-level hardware details from the performance counters.

The ROCProfiler SDK library provides runtime-independent APIs for tracing runtime calls and asynchronous activities such as GPU kernel dispatches and memory moves. The tracing includes callback APIs for runtime API tracing and activity APIs for asynchronous activity records logging. You can use these APIs to develop a tracing tool or implement tracing in the application.

This document discusses the command-line tool `rocprofv3` in detail. It is based on the APIs from the ROCProfiler SDK library.

## Installation

To install ROCProfiler SDK from the source, follow the instructions provided in the sections below:

### Prerequisites

- Linux operating system. Here is the list of operating systems tested for ROCProfiler SDK support:

  - Ubuntu 20.04
  - Ubuntu 22.04
  - OpenSUSE 15.4
  - RedHat 8.8

  To check a system’s operating system and version, see the `/etc/os-release` and `/usr/lib/os-release` files:

  ```bash
  $ cat /etc/os-release
  NAME="Ubuntu"
  VERSION="20.04.4 LTS (Focal Fossa)"
  ID=ubuntu
  ...
  VERSION_ID="20.04"
  ...
  ```

- Cmake version 3.21 or higher.

- To install a new version of Cmake, we recommend using PyPi (Python’s pip):

    ```bash
    pip install --user 'cmake==3.21.0'
    export PATH=${HOME}/.local/bin:${PATH}
    ```

### Build

To build ROCProfiler SDK, use:

```bash
git clone https://git@github.com:ROCm/rocprofiler-sdk.git rocprofiler-sdk-source
```

```bash
cmake --build rocprofiler-sdk-build --target all --parallel 8
```

To see the various build options along with their default values, use:

```bash
$ cmake -LA

REPRODUCIBLE_RUNTIME_USE_MPI:BOOL=OFF
ROCPROFILER_BLACK_FORMAT_EXE:FILEPATH=ROCPROFILER_BLACK_FORMAT_EXE-NOTFOUND
ROCPROFILER_BUILD_CI:BOOL=OFF
ROCPROFILER_BUILD_CODECOV:BOOL=OFF
ROCPROFILER_BUILD_DEBUG:BOOL=OFF
ROCPROFILER_BUILD_DEVELOPER:BOOL=OFF
ROCPROFILER_BUILD_DOCS:BOOL=OFF
ROCPROFILER_BUILD_FMT:BOOL=ON
ROCPROFILER_BUILD_GHC_FS:BOOL=ON
ROCPROFILER_BUILD_GLOG:BOOL=ON
ROCPROFILER_BUILD_GTEST:BOOL=ON
ROCPROFILER_BUILD_RELEASE:BOOL=OFF
ROCPROFILER_BUILD_SAMPLES:BOOL=OFF
ROCPROFILER_BUILD_STACK_PROTECTOR:BOOL=ON
ROCPROFILER_BUILD_STATIC_LIBGCC:BOOL=OFF
ROCPROFILER_BUILD_STATIC_LIBSTDCXX:BOOL=OFF
ROCPROFILER_BUILD_TESTS:BOOL=ON
ROCPROFILER_BUILD_WERROR:BOOL=OFF
ROCPROFILER_CLANG_FORMAT_EXE:FILEPATH=ROCPROFILER_CLANG_FORMAT_EXE-NOTFOUND
ROCPROFILER_CLANG_TIDY_COMMAND:FILEPATH=ROCPROFILER_CLANG_TIDY_COMMAND-NOTFOUND
ROCPROFILER_CMAKE_FORMAT_EXE:FILEPATH=ROCPROFILER_CMAKE_FORMAT_EXE-NOTFOUND
ROCPROFILER_CPACK_SYSTEM_NAME:STRING=Linux
ROCPROFILER_DEBUG_TRACE:BOOL=OFF
ROCPROFILER_DEFAULT_ROCM_PATH:PATH=/opt/rocm-6.1.0-13278
ROCPROFILER_ENABLE_CLANG_TIDY:BOOL=OFF
ROCPROFILER_LD_AQLPROFILE:BOOL=OFF
ROCPROFILER_MEMCHECK:STRING=
ROCPROFILER_REGENERATE_COUNTERS_PARSER:BOOL=OFF
ROCPROFILER_UNSAFE_NO_VERSION_CHECK:BOOL=OFF
```

### Install

To install ROCProfiler SDK from the `rocprofiler-sdk-build` directory, run:

```bash
cmake --build rocprofiler-sdk-build --target install
```

### Test

To run the build tests, `cd` into the `rocprofiler-sdk-build` directory and run:

```bash
ctest -R
```

## Usage

`rocprofv3` is a CLI tool that helps you quickly optimize applications and understand the low-level kernel details without requiring any modification in the source code. It is being developed to be backward compatible with its predecessor, `rocprof`, and to provide more features to help users profile their applications with better accuracy. 

The following sections demonstrate the use of `rocprofv3` for application tracing and kernel profiling using various command-line options.

`rocprofv3` is installed with ROCm under `/opt/rocm/bin`. To use the tool from anywhere in the system, export `PATH` variable:

```bash
export PATH=$PATH:/opt/rocm/bin
```

Before you can start tracing or profiling your HIP application using `rocprofv3`, build the application using:

```bash
cmake -B <build-directory> <source-directory> -DCMAKE_PREFIX_PATH=/opt/rocm

cmake --build <build-directory> --target all --parallel <N>
```

### Options

Below is the list of `rocprofv3` command-line options. Some options are used for application tracing and some for kernel profiling while the output control options control the presentation and redirection of the generated output.

| Option | Description | Use |
|--------|-------------|-----|
| -d \| --output-directory | Specifies the path for the output files. | Output control |
| --hip-trace | Collects HIP runtime traces. | Application tracing |
| --hip-runtime-trace | Collects HIP runtime API traces. | Application tracing |
| --hip-compiler-trace | Collects HIP compiler-generated code traces. | Application tracing |
| --scratch-memory-trace | Collects scratch memory operations traces. | Application tracing |
| --hsa-trace | Collects HSA API traces. | Application tracing |
| --hsa-core-trace | Collects HSA API traces (core API). | Application tracing |
| --hsa-amd-trace | Collects HSA API traces (AMD-extension API). | Application tracing |
| --hsa-image-trace | Collects HSA API Ttaces (Image-extension API). | Application tracing |
| --hsa-finalizer-trace | Collects HSA API traces (Finalizer-extension API). | Application tracing |
| -i | Specifies the input file. | Kernel profiling |
| -L \| --list-metrics | List metrics for counter collection. | Kernel profiling |
| --kernel-trace | Collects kernel dispatch traces. | Application tracing |
| -M \| --mangled-kernels | Overrides the default demangling of kernel names. | Output control |
| --marker-trace | Collects marker (ROC-TX) traces. | Application tracing |
| --memory-copy-trace | Collects memory copy traces. | Application tracing |
| -o \| --output-file | Specifies the name of the output file. Note that this name is appended to the default names (_api_trace or counter_collection.csv) of the generated files'. | Output control |
| --sys-trace | Collects HIP, HSA, memory copy, marker, and kernel dispatch traces. | Application Tracing |
| -T \| --truncate-kernels | Truncates the demangled kernel names for improved readability. | Output control |

You can also see all the `rocprofv3` options using:

```bash
rocprofv3 --help 
```

### Application tracing

Application tracing provides the big picture of a program’s execution by collecting data on the execution times of API calls and GPU commands, such as kernel execution, async memory copy, and barrier packets. This information can be used as the first step in the profiling process to answer important questions, such as how much percentage of time was spent on memory copy and which kernel took the longest time to execute. 

To use `rocprofv3` for application tracing, run:

```bash
rocprofv3 <tracing_option> <app_relative_path>
```

#### HIP trace

HIP trace comprises execution traces for the entire application at the HIP level. This includes HIP API functions and their asynchronous activities at the runtime level. In general, HIP APIs directly interact with the user program. It is easier to analyze HIP traces as you can directly map them to the program.

To trace HIP runtime APIs, use:

```bash
rocprofv3 --hip-trace < app_relative_path >
```

The above command generates a `hip_api_trace.csv` file prefixed with the process ID.

```bash
$ cat 238_hip_api_trace.csv

"Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
"HIP_RUNTIME_API","hipGetDevicePropertiesR0600",238,238,1,1191915574691984,1191915687784011
"HIP_RUNTIME_API","hipMalloc",238,238,2,1191915691312459,1191915691388696
"HIP_RUNTIME_API","hipMalloc",238,238,3,1191915691390637,1191915691423279
"HIP_RUNTIME_API","hipMemcpy",238,238,4,1191915691439107,1191916547828448
"HIP_RUNTIME_API","hipLaunchKernel",238,238,5,1191916547842972,1191916548408842
"HIP_RUNTIME_API","hipMemcpy",238,238,6,1191916548412677,1191916550217834
"HIP_RUNTIME_API","hipFree",238,238,7,1191916562618151,1191916562789093
"HIP_RUNTIME_API","hipFree",238,238,8,1191916562790923,1191916562836351
```

To trace HIP compile time APIs, use:

```bash
rocprofv3 --hip-compiler-trace < app_relative_path >
```

The above command generates a `hip_api_trace.csv` file prefixed with the process ID.

```bash
$ cat 208_hip_api_trace.csv

"Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
"HIP_COMPILER_API","__hipRegisterFatBinary",208,208,1,1508780270085955,1508780270096795
"HIP_COMPILER_API","__hipRegisterFunction",208,208,2,1508780270104242,1508780270115355
"HIP_COMPILER_API","__hipPushCallConfiguration",208,208,3,1508780613897816,1508780613898701
"HIP_COMPILER_API","__hipPopCallConfiguration",208,208,4,1508780613901714,1508780613902200 
```

To describe the fields in the output file, see [Output file fields](#output-file-fields).

#### HSA trace

The HIP runtime library is implemented with the low-level HSA runtime. HSA API tracing is more suited for advanced users who want to understand the application behavior at the lower level. In general, tracing at the HIP level is recommended for most users. You should use HSA trace only if you are familiar with HSA runtime. 

HSA trace contains the start and end time of HSA runtime API calls and their asynchronous activities.

```bash
rocprofv3 --hsa-trace < app_relative_path >
```

The above command generates a `hsa_api_trace.csv` file prefixed with process ID.
Note: the contents of this file have been truncated for demonstration purposes.

```bash
$ cat 197_hsa_api_trace.csv

"Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
"HSA_CORE_API","hsa_system_get_major_extension_table",197,197,1,1507843974724237,1507843974724947
"HSA_CORE_API","hsa_agent_get_info",197,197,3,1507843974754471,1507843974755014
"HSA_AMD_EXT_API","hsa_amd_memory_pool_get_info",197,197,5,1507843974761705,1507843974762398
"HSA_AMD_EXT_API","hsa_amd_memory_pool_get_info",197,197,6,1507843974763901,1507843974764030
"HSA_AMD_EXT_API","hsa_amd_memory_pool_get_info",197,197,7,1507843974765121,1507843974765224
"HSA_AMD_EXT_API","hsa_amd_memory_pool_get_info",197,197,8,1507843974766196,1507843974766328
"HSA_AMD_EXT_API","hsa_amd_memory_pool_get_info",197,197,9,1507843974767534,1507843974767641
"HSA_AMD_EXT_API","hsa_amd_memory_pool_get_info",197,197,10,1507843974768639,1507843974768779
"HSA_AMD_EXT_API","hsa_amd_agent_iterate_memory_pools",197,197,4,1507843974758768,1507843974769238
"HSA_CORE_API","hsa_agent_get_info",197,197,11,1507843974771091,1507843974771537
```

To describe the fields in the output file, see [Output file fields](#output-file-fields).

#### Marker trace

In certain situations, such as debugging performance issues in large-scale GPU programs, API-level tracing may be too fine-grained to provide a big picture of the program execution. In such cases, defining specific tasks to be traced is helpful.

To specify the tasks for tracing, enclose the respective source code with the API calls provided by the ROCTX library. This process is also known as instrumentation. As the scope of code for instrumentation is defined using the enclosing API calls, it is called a range. A range is a programmer-defined task that has a well-defined start and end code scope. You can also fine-grained the scope specified within a range using further nested ranges. The `rocprofv3` tool also reports the timelines for these nested ranges.

Here is a list of useful APIs for code instrumentation.

- `roctxMark`: Inserts a marker in the code with a message. Creating marks can help you see when a line of code is executed.
- `roctxRangeStart`: Starts a range. Different threads can start ranges.
- `roctxRangePush`: Starts a new nested range.
- `roctxRangePop`: Stops the current nested range.
- `roctxRangeStop`: Stops the given range.

See how to use `rocTX` APIs in the MatrixTranspose application below:

```bash
roctxMark("before hipLaunchKernel");
int rangeId = roctxRangeStart("hipLaunchKernel range");
roctxRangePush("hipLaunchKernel");

// Launching kernel from host
hipLaunchKernelGGL(matrixTranspose, dim3(WIDTH/THREADS_PER_BLOCK_X, WIDTH/THREADS_PER_BLOCK_Y), dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y), 0,0,gpuTransposeMatrix,gpuMatrix, WIDTH);

roctxMark("after hipLaunchKernel");

// Memory transfer from device to host
roctxRangePush("hipMemcpy");

hipMemcpy(TransposeMatrix, gpuTransposeMatrix, NUM * sizeof(float), hipMemcpyDeviceToHost);

roctxRangePop();  // for "hipMemcpy"
roctxRangePop();  // for "hipLaunchKernel"
roctxRangeStop(rangeId);
```

To trace the API calls enclosed within the range, use:

```bash
rocprofv3 --marker-trace < app_relative_path >
```

Running the above command generates a `marker_api_trace.csv` file prefixed with the process ID.

```bash
$ cat 210_marker_api_trace.csv

"Domain","Function","Process_Id","Thread_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
"MARKER_CORE_API","before hipLaunchKernel",717,717,1,1520113899312225,1520113899312225
"MARKER_CORE_API","after hipLaunchKernel",717,717,4,1520113900128482,1520113900128482
"MARKER_CORE_API","hipMemcpy",717,717,5,1520113900141100,1520113901483408
"MARKER_CORE_API","hipLaunchKernel",717,717,3,1520113899684965,1520113901491622
"MARKER_CORE_API","hipLaunchKernel range",717,0,2,1520113899682208,1520113901495882
```

For the description of the fields in the output file, see [Output file fields](#output-file-fields).

#### Kernel trace

To trace kernel dispatch traces, use:

```bash
rocprofv3 --kernel-trace < app_relative_path >
```

The above command generates a `kernel_trace.csv` file prefixed with the process ID.

```bash
$ cat 199_kernel_trace.csv

"Kind","Agent_Id","Queue_Id","Kernel_Id","Kernel_Name","Correlation_Id","Start_Timestamp","End_Timestamp","Private_Segment_Size","Group_Segment_Size","Workgroup_Size_X","Workgroup_Size_Y","Workgroup_Size_Z","Grid_Size_X","Grid_Size_Y","Grid_Size_Z"
"KERNEL_DISPATCH",1,139690710949888,15,"matrixTranspose(float*, float*, int)",0,671599758568,671599825328,0,0,4,4,1,1024,1024,1
```

To describe the fields in the output file, see [Output file fields](#output-file-fields).

#### Memory copy trace

To trace memory moves across the application, use:

```bash
rocprofv3 –-memory-copy-trace < app_relative_path >
```

The above command generates a `memory_copy_trace.csv` file prefixed with the process ID.

```bash
$ cat 197_memory_copy_trace.csv

"Kind","Direction","Source_Agent_Id","Destination_Agent_Id","Correlation_Id","Start_Timestamp","End_Timestamp"
"MEMORY_COPY","HOST_TO_DEVICE",0,1,0,14955949675563,14955950239443
"MEMORY_COPY","DEVICE_TO_HOST",1,0,0,14955952733485,14955953315285
```

To describe the fields in the output file, see [Output file fields](#output-file-fields).

#### Sys trace

This is an all-inclusive option to collect all the above-mentioned traces.

```bash
rocprofv3 –-sys-trace < app_relative_path >
```

Running the above command generates `hip_api_trace.csv`, `hsa_api_trace.csv`, `kernel_trace.csv`, `memory_copy_trace.csv`, and `marker_api_trace.csv` (if `rocTX` APIs are specified in the application) files prefixed with the process Id.

### Kernel profiling

The application tracing functionality allows you to evaluate the duration of kernel execution but is of little help in providing insight into kernel execution details. The kernel profiling functionality allows you to select kernels for profiling and choose the basic counters or derived metrics to be collected for each kernel execution, thus providing a greater insight into kernel execution.

For more information on counters available on MI200, refer to the [MI200 Performance Counters and Metrics](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch/mi300-mi200-performance-counters.html).

#### Input file

To collect the desired basic counters or derived metrics, you can just mention them in an input file below. The line consisting of the counter or metric names must begin with `pmc`.

```bash
$ cat input.txt

pmc: GPUBusy SQ_WAVES
pmc: GRBM_GUI_ACTIVE
```

The GPU hardware resources limit the number of basic counters or derived metrics that can be collected in one run of profiling. If too many counters or metrics are selected, the kernels need to be executed multiple times to collect them. For multi-pass execution, include multiple `pmc` rows in the input file. Counters or metrics in each `pmc` row can be collected in each kernel run.

#### Kernel profiling output

To supply the input file for kernel profiling, use:

```bash
rocprofv3 -i input.txt <app_relative_path>
```

Running the above command generates a `./pmc_n/counter_collection.csv` file prefixed with the process ID. For each `pmc` row, a directory `pmc_n` containing a `counter_collection.csv` file is generated, where n = 1 for the first row and so on.

Each row of the CSV file is an instance of kernel execution. Here is a truncated version of the output file from `pmc_1`.

```bash
$ cat pmc_1/218_counter_collection.csv

"Correlation_Id","Dispatch_Id","Agent_Id","Queue_Id","Process_Id","Thread_Id","Grid_Size","Kernel_Name","Workgroup_Size","LDS_Block_Size","Scratch_Size","VGPR_Count","SGPR_Count","Counter_Name","Counter_Value"
0,1,1,139892123975680,5619,5619,1048576,"matrixTranspose(float*, float*, int)",16,0,0,8,16,"SQ_WAVES",65536
```

### Output file fields

The various fields or the columns in the output CSV files generated for application tracing and kernel profiling are described here:

| Field | Description |
|-------|-------------|
| Agent_Id | GPU identifier to which the kernel was submitted. |
| Correlation_Id | Unique identifier for correlation between HIP and HSA async calls during activity tracing. |
| Start_Timestamp | Begin time in nanoseconds (`ns`) when the kernel begins execution. |
| End_Timestamp | End time in ns when the kernel finishes execution. |
| Queue_Id | ROCm queue unique identifier to which the kernel was submitted. |
| Private_Segment_Size | The amount of memory required for the combined private, spill, and arg segments for a work item in bytes. |
| Group_Segment_Size | The group segment memory required by a workgroup in bytes. This does not include any dynamically allocated group segment memory that may be added when the kernel is dispatched. |
| Workgroup_Size | Size of the workgroup as declared by the compute shader. |
| Workgroup_Size_n | Size of the workgroup in the nth dimension as declared by the compute shader, where n = X, Y, or Z. |
| Grid_Size | Number of thread blocks required to launch the kernel. |
| Grid_Size_n | Number of thread blocks in the nth dimension required to launch the kernel, where n = X, Y, or Z. |
| LDS_Block_Size | Thread block size for the kernel's Local Data Share (`LDS`) memory. |
| Scratch_Size | Kernel’s scratch memory size. |
| SGPR_Count | Kernel's Scalar General-Purpose Register (`SGPR`) count. |
| VGPR_Count | Kernel's Vector General-Purpose Register (`VGPR`) count. |

### Sample programs

After the ROCm build is installed:

- Sample programs are installed here:

    ```bash
    /opt/rocm/share/rocprofiler-sdk/samples
    ```

- `rocprofv3` tool is installed here:

    ```bash
    /opt/rocm/bin
    ```

To build samples from any directory, run the following:

```bash
cmake -B <build directory> /opt/rocm/share/rocprofiler-sdk/samples -DCMAKE_PREFIX_PATH=/opt/rocm

cmake --build <build directory> --target all --parallel 8
```

To run the built samples, `cd` into the `<build directory>` mentioned in the build commands above and run:

```bash
ctest -V
```

**Note:** Running a few of these tests will require pandas and pytest to be installed first.

```bash
/usr/local/bin/python -m pip install -r requirements.txt
```
