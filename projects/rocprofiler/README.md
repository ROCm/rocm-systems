# ROCm Profiling Tools

## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2022 Advanced Micro Devices, Inc. All Rights Reserved.

## Introduction

ROCProfiler is AMD’s tooling infrastructure that provides a hardware specific low level performance analysis interface for the profiling and the tracing of GPU compute applications.

## ROCProfiler V1

Profiling with metrics and traces based on perfcounters (PMC) and traces (SPM).
Implementation is based on AqlProfile HSA extension.
The last API library version for ROCProfiler v1 is 8.0.0

The library source tree:

- doc  - Documentation
- include/rocprofiler/rocprofiler.h - Library public API
- include/rocprofiler/v2/rocprofiler.h - V2 Beta Library public API
- include/rocprofiler/v2/rocprofiler_plugins.h - V2 Beta Tool's Plugins Library public API
- src  - Library sources
  - core - Library API sources
  - util - Library utils sources
  - xml - XML parser
- test - Library test suite
  - ctrl - Test controll
  - util - Test utils
  - simple_convolution - Simple convolution test kernel

## Build environment

Roctracer & Rocprofiler need to be installed in the same directory.

```bash
export CMAKE_PREFIX_PATH=<path_to_hsa-runtime_includes>:<path_to_hsa-runtime_library>
export CMAKE_BUILD_TYPE=<debug|release> # release by default
export CMAKE_DEBUG_TRACE=1 # 1 to enable debug tracing
```

To build with the current installed ROCM:

```bash
cd .../rocprofiler
./build.sh ## (for clean build use `-cb`)
```

To run the test:

```bash
cd .../rocprofiler/build
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH # paths to ROC profiler and oher libraries
export HSA_TOOLS_LIB=librocprofiler64.so.1 # ROC profiler library loaded by HSA runtime
export ROCP_TOOL_LIB=test/librocprof-tool.so # tool library loaded by ROC profiler
export ROCP_METRICS=metrics.xml # ROC profiler metrics config file
export ROCP_INPUT=input.xml # input file for the tool library
export ROCP_OUTPUT_DIR=./ # output directory for the tool library, for metrics results file 'results.txt' and trace files
./<your_test>
```

Internal 'simple_convolution' test run script:

```bash
cd .../rocprofiler/build
./run.sh
```

- To enabled error messages logging to '/tmp/rocprofiler_log.txt':

```bash
export ROCPROFILER_LOG=1
```

- To enable verbose tracing:

```bash
export ROCPROFILER_TRACE=1
```

## Supported AMD GPU Architectures (V1)

  The following AMD GPU architectures are supported with ROCprofiler V1:

- gfx8 (Fiji/Ellesmere)
- gfx900 (AMD Vega 10)
- gfx906 (AMD Vega 7nm also referred to as AMD Vega 20)
- gfx908 (AMD Instinct™ MI100 accelerator)
- gfx90a (AMD Instinct™ MI200)

***
Note: ROCProfiler V1 tool usage documentation is available at [Click Here](doc/rocprof_tool.md)
***

## ROCProfiler V2

The first API library version for ROCProfiler v2 is 9.0.0

***
Note: ROCProfilerV2 is currently considered a beta version and is subject to change in future releases
***

### ROCProfilerV2 Modules

- Counters
- Hardware
- Generic Buffer
- Session
- Filter
- Tools
- Plugins
- Samples
- Tests

## Getting started

### Requirements

- makecache
- Gtest Development Package (Ubuntu: libgtest-dev)
- libsystemd-dev, libelf-dev, libnuma-dev, libpciaccess-dev on ubuntu or their corresponding packages on any other OS
- Cppheaderparser, websockets, matplotlib, lxml, barectf Python3 Packages
- Python packages can be installed using:

  ```bash
  pip3 install -r requirements.txt
  ```

### Build

The user has two options for building:

- Option 1 (It will install in the path saved in ROCM_PATH environment variable or /opt/rocm if ROCM_PATH is empty):

  - Run

    Normal Build

    ```bash
    ./build.sh --build OR ./build.sh -b
    ```

    Clean Build

    ```bash
    ./build.sh --clean-build OR ./build.sh -cb
    ```

- Option 2 (Where ROCM_PATH envronment need to be set with the current installation directory of rocm), run the following:

  - Creating the build directory

    ```bash
    mkdir build && cd build
    ```

  - Configuring the rocprofv2 build

    ```bash
    cmake -DCMAKE_PREFIX_PATH=$ROCM_PATH -DCMAKE_MODULE_PATH=$ROCM_PATH/hip/cmake -DROCPROFILER_BUILD_TESTS=1 -DROCPROFILER_BUILD_SAMPLES=1 <CMAKE_OPTIONS> ..
    ```

  - Building the main runtime of the rocprofv2 project

    ```bash
    cmake --build . -- -j
    ```

  - Optionally, for building API documentation

    ```bash
    cmake --build . -- -j doc
    ```

  - Optionally, for building packages (DEB, RPM, TGZ)
    Note: Requires rpm package on ubuntu

    ```bash
    cmake --build . -- -j package
    ```

### Install

- Optionally, run the following to install

  ```bash
  cd build
  cmake --build . -- -j install
  ```

## Features & Usage

### rocsys

A command line utility to control a session (launch/start/stop/exit), with the required application to be traced or profiled in a rocprofv2 context. Usage:

- Launch the application with the required profiling and tracing options with giving a session identifier to be used later

  ```bash
  rocsys --session session_name launch mpiexec -n 2 rocprofv2 -i samples/input.txt Histogram
  ```

- Start a session with a given identifier created at launch

  ```bash
  rocsys --session session_name start
  ```

- Stop a session with a given identifier created at launch

  ```bash
  rocsys –session session_name stop
  ```

- Exit a session with a given identifier created at launch

  ```bash
  rocsys –session session_name exit
  ```

### ROCProf Versioning Support

Currently, rocprof can support both versions, rocprof and rocprofv2, that can be done using `--tool-version`

```bash
rocprof --tool-version <VERSION_REQUIRED> <rocprof/v2_options> <app_relative_path>
```

- `--tool-version 1` means it will just use rocprof V1.
- `--tool-version 2` means it will just use rocprofv2.

To know what version you are using right now, along with more information about the rocm version, use the following:

```bash
rocprof --version
```

### Counters and Metric Collection

HW counters and derived metrics can be collected using following option:

```bash
rocprofv2 -i samples/input.txt <app_relative_path>
```

input.txt content Example (Details of what is needed inside input.txt will be mentioned with every feature):

  `pmc: SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE SQ_INSTS_VALU`

### Application Trace Support

Different trace options are available while profiling an app:

- HIP API & asynchronous activity tracing

  ```bash
  rocprofv2 --hip-api <app_relative_path> ## For synchronous HIP API Activity tracing
  rocprofv2 --hip-activity <app_relative_path> ## For both Synchronous & ASynchronous HIP API Activity tracing
  rocprofv2 --hip-trace <app_relative_path> ## Same as --hip-activity, added for backward compatibility
  ```

- HSA API & asynchronous activity tracing

  ```bash
  rocprofv2 --hsa-api <app_relative_path> ## For synchronous HSA API Activity tracing
  rocprofv2 --hsa-activity <app_relative_path> ## For both Synchronous & ASynchronous HSA API Activity tracing
  rocprofv2 --hsa-trace <app_relative_path> ## Same as --hsa-activity, added for backward compatibility
  ```

- Kernel dispatches tracing

  ```bash
  rocprofv2 --kernel-trace <app_relative_path> ## Kernel Dispatch Tracing
  ```

- HIP & HSA API and asynchronous activity and kernel dispatches tracing

  ```bash
  rocprofv2 --sys-trace <app_relative_path> ## Same as combining --hip-trace & --hsa-trace & --kernel-trace
  ```

- For complete usage options, please run rocprofv2 help

  ```bash
  rocprofv2 --help
  ```

### Plugin Support

We have a template for adding new plugins. New plugins can be written on top of rocprofv2 to support the desired output format using include/rocprofiler/v2/rocprofiler_plugins.h header file. These plugins are modular in nature and can easily be decoupled from the code based on need. Installation files:

```string
rocprofiler-plugins_2.0.0-local_amd64.deb
rocprofiler-plugins-2.0.0-local.x86_64.rpm
```

Plugins may have multiple versions, the user can specify which version of the plugin to use by running the following command:

```bash
rocprofv2 --plugin <plugin_name> --plugin-version <plugin_version_required> <rocprofv2_options> <app_relative_path>
```

- File plugin: outputs the data in txt files. File plugin have two versions, by default version 2 is the current default.
Usage:

  ```bash
  rocprofv2 --plugin file -i samples/input.txt -d output_dir <app_relative_path> # -d is optional, but can be used to define the directory output for output results
  ```

  File plugin version 1 output header will be similar to the legacy rocprof v1 output:

  ```text
  Index,KernelName,gpu-id,queue-id,queue-index,pid,tid,grd,wgr,lds,scr,arch_vgpr,accum_vgpr,sgpr,wave_size,sig,obj,DispatchNs,BeginNs,EndNs,CompleteNs,Counters
  ```

  File plugin version 2 output header:

  ```text
  Dispatch_ID,GPU_ID,Queue_ID,PID,TID,Grid_Size,Workgroup_Size,LDS_Per_Workgroup,Scratch_Per_Workitem,Arch_VGPR,Accum_VGPR,SGPR,Wave_Size,Kernel_Name,Start_Timestamp,End_Timestamp,Correlation_ID,Counters
  ```

- Perfetto plugin: outputs the data in protobuf format. Protobuf files can be viewed using ui.perfetto.dev or using trace_processor.
Usage:

  ```bash
  rocprofv2 --plugin perfetto --hsa-trace -d output_dir <app_relative_path> # -d is optional, but can be used to define the directory output for output results
  ```

- CTF plugin: Outputs the data in ctf format(a binary trace format). CTF binary output can be viewed using TraceCompass or babeltrace.
Usage:

  ```bash
  rocprofv2 --plugin ctf --hip-trace -d output_dir <app_relative_path> # -d is optional, but can be used to define the directory output for output results
  ```

- ATT (Advanced thread tracer) plugin: advanced hardware traces data in binary format. Please refer ATT section.
Tool used to collect fine-grained hardware metrics. Provides ISA-level instruction hotspot analysis via hardware tracing.

  - Install plugin package. See Plugin Support section for installation
  - Run the following to view the trace. Att-specific options must come right after the assembly file.
  - On ROCm 6.0, ATT enables automatic capture of the ISA during kernel execution, and does not require recompiling. It is recommeneded to leave at "auto".

    ```bash
    rocprofv2 -i input.txt --plugin att auto --mode csv <app_relative_path>
    # Or using a user-supplied ISA:
    # rocprofv2 -i input.txt --plugin att <app_assembly_file> --mode csv <app_relative_path>
    ```

  - app_relative_path
    Path for the running application
  - ATT plugin optional parameters
    - --att_kernel "filename": Kernel filename(s) (glob) to use. A CSV file (or UI folder) will be generated for each kernel.txt file. Default: all in current folder.
    - --mode [csv, network, file, off (default)]
      - off
          Runs trace collection but not analysis, so it can be analyzed at a later time. Run rocprofv2 ATT with the same parameters (+ --mode csv), removing the application binary, to analyze previously generated traces.
      - csv
          Dumps the analyzed assembly into a CSV format, with the hitcount and total cycles cost. Recommended mode for most users.
      - network (deprecated)
          Opens the server with the browser UI.
          att needs 2 ports available (e.g. 8000, 18000). There is an option (default: --ports "8000,18000") to change these.
          In case rocprofv2 is running on a different machine, use port forwarding `ssh -L 8000:localhost:8000 <user@IP>` so the browser can be used locally. For docker, use --network=host --ipc=host -p8000:8000 -p18000:18000
      - file (deprecated)
          Dumps the analyzed json files to disk for vieweing at a later time. Run python3 httpserver.py from within the generated ui/ folder to view the trace, similarly to network mode. The folder can be copied to another machine, and will run without rocm.
  - input.txt
      Required. Used to select specific compute units and other trace parameters.
      For first time users, using the following input file:

      ```bash
      # vectoradd
      att: TARGET_CU=1
      SE_MASK=0x1
      SIMD_SELECT=0x3
      ```

      ```bash
      # histogram
      att: TARGET_CU=0
      SE_MASK=0xFF
      SIMD_SELECT=0xF // 0xF for GFX9, SIMD_SELECT=0 for Navi
      ```

      Possible contents:
    - att: TARGET_CU=1 // or some other CU [0,15] - WGP for Navi [0,8]
    - SE_MASK=0x1 // bitmask of shader engines. The fewer, the easier on the hardware. Default enables 1 out of 4 shader engines.
    - SIMD_SELECT=0xF // GFX9: bitmask of SIMDs. Navi: SIMD Index [0-3]. Recommended 0xF for GFX9 and 0x0 for Navi.
    - DISPATCH=ID // collect trace only for the given dispatch_ID. Multiple lines for can be added.
    - DISPATCH=ID,RN // collect trace only for the given dispatch_ID and MPI rank RN. Multiple lines with varying combinations of RN and ID can be added.
    - KERNEL=kernname // Profile only kernels containing the string kernname (c++ mangled name). Multiple lines can be added.
    - PERFCOUNTERS_CTRL=0x3 // Multiplier period for counter collection [0~31]. 0=fastest. GFX9 only.
    - PERFCOUNTER_MASK=0xFFF // Bitmask for perfcounter collection. GFX9 only.
    - PERFCOUNTER=counter_name // Add a SQ counter to be collected with ATT; period defined by PERFCOUNTERS_CTRL. GFX9 only.
    - BUFFER_SIZE=[size] // Sets size of the ATT buffer collection, per dispatch, in megabytes (shared among all shader engines).
    - ISA_CAPTURE_MODE=[0,1,2] // Set codeobj capture mode during kernel dispatch.
        - 0 = capture symbols only.
        - 1 = capture symbols for file:// and make a copy of memory://
        - 2 = Copy file:// and memory://
    - ISA_DUMP_MODE=[0,1,2,3] // Set how captured codeobj information is dumped when a trace record arrives.
        - 0 = Default. Dump everything.
        - 1 = Dump only the code object containing the kernel address in the kernel dispatch packet.
        - 2 = Dump a single kernel symbol matching the kernel dispatch packet.
        - 3 = Disables ISA Dumping.
    - By default, kernel names are truncated for ATT.To disable, please see the kernel name truncation section below.

  - Example for vectoradd.

    ```bash
    # -g adds debugging symbols to the binary. Required only for tracking disassembly back to c++.
    hipcc -g vectoradd_hip.cpp -o vectoradd_hip.exe
    # "auto" means to use the automatically captured ISA, e.g. vectoradd_float_v0_isa.s dumped along with .att files.
    # "--mode csv" dumps the result to "att_output_vectoradd_float_v0.csv".
    rocprofv2 -i input.txt --plugin att auto --mode csv ./vectoradd_hip.exe
    ```
    ```bash
    # Alternatively, using --save-temps to generate the ISA
    hipcc -g --save-temps vectoradd_hip.cpp -o vectoradd_hip.exe
    # Replace "auto" with <generated_gpu_isa.s> for user-supplied ISA. Typically they match the wildcards *amdgcn-amd-amdhsa*.s.
    # Special attention to the correct architecture for the ISA, such as "gfx1100" (navi31).
    rocprofv2 -i input.txt --plugin att vectoradd_hip-hip-amdgcn-amd-amdhsa-gfx1100.s --mode csv ./vectoradd_hip.exe
    ```

    Instruction latencies will be in att_output_vectoradd_float_v0.csv

    ```bash
    # Use -d option to specify the generated data directory, and -o to specify dir and filename is the csv:
    rocprofv2 -d mydir -o test/mycsv -i input.txt --plugin att auto --mode csv ./vectoradd_hip.exe
    # Generates raw files inside mydir/ and the parsed data on test/mycsv_vectoradd_float_v0.csv
    ```

  ***
  Note: For MPI or long running applications, we recommend to run collection, and later run the parser with already collected data:
  Run only collection: The assembly file is not used. Use mpirun [...] rocprofv2 [...] if needed.

  ```bash
  # Run only data collection, not the parser
  rocprofv2 -i input.txt --plugin att auto --mode off ./vectoradd_hip.exe
  ```

  Remove the binary/application from the command line.

  ```bash
  # Only runs the parser on previously collected data.
  rocprofv2 -i input.txt --plugin att auto --mode csv
  ```

  Note 2: By default, ATT only collects a SINGLE kernel dispatch for the whole application, which is the first dispatch matching the given filters (DISPATCH=<id> or KERNEL=<name>). To collect multiple dispatches in a single application run, use:

  ```bash
  export ROCPROFILER_MAX_ATT_PROFILES=<max_collections>
  ```

  ***

### Flush Interval

Flush interval can be used to control the interval time in milliseconds between the buffers flush for the tool. However, if the buffers are full the flush will be called on its own. This can be used as in the next example:

```bash
rocprofv2 --flush-interval <TIME_INTERVAL_IN_MILLISECONDS> <rest_of_rocprofv2_arguments> <app_relative_path>
```

### Trace Period

Trace period can be used to control when the profiling or tracing is enabled using two arguments, the first one is the delay time, which is the time spent idle without tracing or profiling. The second argument is the profiling or the tracing time, which is the active time where the profiling and tracing are working, so basically, the session will work in the following timeline:

```string
<DELAY_TIME> => <PROFILING_OR_TRACING_SESSION_START> => <ACTIVE_PROFILING_OR_TRACING_TIME> => <PROFILING_OR_TRACING_SESSION_STOP>
```

  This feature can be used using the following command:

```bash
rocprofv2 --trace-period <delay>:<active_time>:<interval> <rest_of_rocprofv2_arguments> <app_relative_path>
```

- delay: Time delay to start profiling (ms).
- active_time: How long to profile for (ms).
- interval: If set, profiling sessions will start (loop) every "interval", and run for "active_time", until the application ends. Must be higher than "active_time".

### Device Profiling

A device profiling session allows the user to profile the GPU device for counters irrespective of the running applications on the GPU. This is different from application profiling. device profiling session doesn't care about the host running processes and threads. It directly provides low level profiling information.

### Session Support

  A session is a unique identifier for a profiling/tracing/pc-sampling task. A ROCProfilerV2 Session has enough information about what needs to be collected or traced and it allows the user to start/stop profiling/tracing whenever required. More details on the API can be found in the API specification documentation that can be installed using rocprofiler-doc package. Samples also can be found for how to use the API in samples directory.

## Tests

 We make use of the GoogleTest (Gtest) framework to automatically find and add test cases to the CMAKE testing environment. ROCProfilerV2 testing is categorized as following:

- unittests (Gtest Based) : These includes tests for core classes. Any newly added functionality should have a unit test written to it.

- featuretests (standalone and Gtest Based): These includes both API tests and tool tests. Tool is tested against different applications to make sure we have right output in evry run.

- memorytests (standalone): This includes running address sanitizer for memory leaks, corruptions.

installation:
rocprofiler-tests_9.0.0-local_amd64.deb
rocprofiler-tests-9.0.0-local.x86_64.rpm

### List and Run tests

#### Run unit tests on the commandline

```bash
./build/tests/unittests/runUnitTests
```

#### Run profilerfeaturetests on the commandline

```bash
./build/tests/featuretests/profiler/runFeatureTests
```

#### Run tracer featuretests on the commandline

```bash
./build/tests/featuretests/tracer/runTracerFeatureTests
```

#### Run all tests

```bash
rocprofv2 -t
```

OR

```bash
ctest
```

### Guidelines for adding new tests

- Prefer to enhance an existing test as opposed to writing a new one. Tests have overhead to start and many small tests spend precious test time on startup and initialization issues.
- Make the test run standalone without requirement for command-line arguments. This makes it easier to debug since the name of the test is shown in the test report and if you know the name of the test you can the run the test.

## Logging

To enable error messages logging to '/tmp/rocprofiler_log.txt':

```bash
export ROCPROFILER_LOG=1
```

## Kernel Name Truncation

By default kernel names are not truncated. To enable truncation for readability:

```
export ROCPROFILER_TRUNCATE_KERNEL_PATH=1
```

## Documentation

We make use of doxygen to automatically generate API documentation. Generated document can be found in the following path:

ROCM_PATH by default is /opt/rocm
It can be set by the user in different location if needed.
<ROCM_PATH>/share/doc/rocprofv2

installation:

```string
rocprofiler-docs_9.0.0-local_amd64.deb
rocprofiler-docs-9.0.0-local.x86_64.rpm
```

## Samples

- Profiling: Profiling Samples depending on replay mode
- Tracing: Tracing Samples

installation:

```string
rocprofiler-samples_9.0.0-local_amd64.deb
rocprofiler-samples-9.0.0-local.x86_64.rpm
```

usage:

samples can be run as independent executables once installed

## Project Structure

- bin: ROCProf scripts along with V1 post processing scripts
- doc: Documentation settings for doxygen, V1 API Specifications pdf document.
- include:
  - rocprofiler.h: V1 API Header File
  - v2:
    - rocprofiler.h: V2 API Header File
    - rocprofiler_plugin.h: V2 Tool Plugins API
- plugin
  - file: File Plugin
  - perfetto: Perfetto Plugin
  - att: Adavced thread tracer Plugin
  - ctf: CTF Plugin
- samples: Samples of how to use the API, and also input.txt input file samples for counter collection and ATT.
- script: Scripts needed for tracing
- src: Source files of the project
  - api: API implementation for rocprofv2
  - core: Core source files needed for the V1/V2 API
    - counters: Basic and Derived Counters
    - hardware: Hardware support
    - hsa: Provides support for profiler and tracer to communicate with HSA
      - queues: Intercepting HSA Queues
      - packets: Packets Preparation for profiling
    - memory: Memory Pool used in buffers that saves the output data
    - session: Session Logic
      - filter: Type of profiling or tracing and its properties
      - tracer: Tracing support of the session
      - profiler: Profiling support of the session
      - spm: SPM support of the session
      - att: ATT support of the session
  - tools: Tools needed to run profiling and tracing
    - rocsys: Controlling Session from another CLI
  - utils: Utilities needed by the project
- tests: Tests folder
- CMakeLists.txt: Handles cmake list for the whole project
- build.sh: To easily build and compile rocprofiler
- CHANGELOG.md: Changes that are happening per release

## Support

Please report in the Github Issues.

## Limitations

- Navi3x requires a stable power state for counter collection.
  Currently, this state needs to be set by the user.
  To do so, set "power_dpm_force_performance_level" to be writeable for non-root users, then set performance level to profile_standard:

  ```bash
  sudo chmod 777 /sys/class/drm/card0/device/power_dpm_force_performance_level
  echo profile_standard >> /sys/class/drm/card0/device/power_dpm_force_performance_level
  ```

  Recommended: "profile_standard" for counter collection and "auto" for all other profiling. Use rocm-smi to verify the current power state. For multiGPU systems (includes integrated graphics), replace "card0" by the desired card.
- Timestamps may be incorrect with HIP_OPS when the system has been in sleep state.
- HIP_OPS are mutually exclusive with HSA_OPS.

## Supported AMD GPU Architectures (V2)

  The following AMD GPU architectures are supported with ROCprofiler V2:

- gfx900 (AMD Vega 10)
- gfx906 (AMD Vega 7nm also referred to as AMD Vega 20)
- gfx908 (AMD Instinct™ MI100 accelerator)
- gfx90a (AMD Instinct™ MI200)
- gfx94x (AMD Instinct™ MI300)
- gfx10xx ([Navi2x] AMD Radeon(TM) Graphics)
- gfx11xx ([Navi3x] AMD Radeon(TM) Graphics)
