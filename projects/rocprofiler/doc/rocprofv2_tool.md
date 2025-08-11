# ROCProfiler v2

## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2022 Advanced Micro Devices, Inc. All Rights Reserved.


## Introduction

ROCProfilerV2 is a newly developed design for AMD’s tooling infrastructure that provides a hardware specific low level performance analysis interface for profiling of GPU compute applications.
The first API library version for ROCProfiler v2 is 9.0.0

**Note: ROCProfilerV2 is currently considered a beta version and is subject to change in future releases**

## Modules

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
- libelf-dev, libnuma-dev on ubuntu or their corresponding packages on any other OS
- Cppheaderparser, websockets, matplotlib, lxml, barectf Python3 Packages

### Build

The user has two options for building:

- Option 1 (It will install in the path saved in ROCM_PATH environment variable or /opt/rocm if ROCM_PATH is empty):

  - Run
  ```bash
   # Normal Build
  ./build.sh --build OR ./build.sh -b
  ```
  ```bash
   # Clean Build
  ./build.sh --clean-build OR ./build.sh -cb
  ```

- Option 2 (Where ROCM_PATH environment need to be set with the current installation directory of rocm), run the following:
  ```bash
  # Creating the build directory
  mkdir build && cd build

  # Configuring the rocprofv2 build
  cmake -DCMAKE_PREFIX_PATH=$ROCM_PATH -DCMAKE_MODULE_PATH=$ROCM_PATH/hip/cmake <CMAKE_OPTIONS> ..

  # Building the main runtime of the rocprofv2 project
  cmake --build . -- -j

  # Optionally, for building API documentation
  cmake --build . -- -j doc

  # Optionally, for building ROCProfiler V2 samples
  cmake --build . -- -j samples

  # Optionally, for building packages (DEB, RPM, TGZ)
  cmake --build . -- -j tests

  # Optionally, for building packages (DEB, RPM, TGZ)
  # Note: Requires rpm package on ubuntu
  cmake --build . -- -j package
  ```

### Install

- Optionally, run the following to install
  ```bash
  # Install rocprofv2 in the ROCM_PATH path
  ./rocprofv2 --install
  ```

  OR, if you are using option 2 in building
  ```bash
  cd build
  # Install rocprofv2 in the ROCM_PATH path
  cmake --build . -- -j install
  ```

## Tool Usage and Features

- rocsys: This is a frontend command line utility to launch/start/stop/exit a session with the required application to be traced or profiled in rocprofv2 context. 
  Usage:
  ```bash
    # Launch the application with the required profiling and tracing options with giving a session identifier to be used later
    rocsys --session session_name launch mpiexec -n 2 ./rocprofv2 -i samples/input.txt Histogram
  
    # Start a session with a given identifier created at launch
    rocsys --session session_name start

    # Stop a session with a given identifier created at launch
    rocsys –session session_name stop

    # Exit a session with a given identifier created at launch
    rocsys –session session_name exit
  ```

- rocprofv2:

  - Counters and Metric Collection: HW counters and derived metrics can be collected using following option:
  ```bash
      rocprofv2 -i samples/input.txt <app_relative_path>
      input.txt
  ```
      input.txt content Example (Details of what is needed inside input.txt will be mentioned with every feature):
  ```bash
      pmc: SQ_WAVES GRBM_COUNT GRBM_GUI_ACTIVE SQ_INSTS_VALU
  ```

  - Application Trace Support: Different trace options are available while profiling an app:
  ```bash
      # HIP API & asynchronous activity tracing
      rocprofv2 --hip-api <app_relative_path> ## For synchronous HIP API Activity tracing
      rocprofv2 --hip-activity <app_relative_path> ## For both Synchronous & ASynchronous HIP API Activity tracing
      rocprofv2 --hip-trace <app_relative_path> ## Same as --hip-activity, added for backward compatibility

      # HSA API & asynchronous activity tracing
      rocprofv2 --hsa-api <app_relative_path> ## For synchronous HSA API Activity tracing
      rocprofv2 --hsa-activity <app_relative_path> ## For both Synchronous & ASynchronous HSA API Activity tracing
      rocprofv2 --hsa-trace <app_relative_path> ## Same as --hsa-activity, added for backward compatibility

      # Kernel dispatches tracing
      rocprofv2 --kernel-trace <app_relative_path> ## Kernel Dispatch Tracing

      # HIP & HSA API and asynchronous activity and kernel dispatches tracing
      rocprofv2 --sys-trace <app_relative_path> ## Same as combining --hip-trace & --hsa-trace & --kernel-trace
  ```

    For complete usage options, please run rocprofv2 help
  ```bash
      rocprofv2 --help
  ```

- Plugin Support: We have a template for adding new plugins. New plugins can be written on top of rocprofv2 to support the desired output format using include/rocprofiler/v2/rocprofiler_plugins.h header file. These plugins are modular in nature and can easily be decoupled from the code based on need. E.g.
  - file plugin: outputs the data in txt files.
  - Perfetto plugin: outputs the data in protobuf format.
    - Protobuf files can be viewed using ui.perfetto.dev or using trace_processor
  - CTF plugin: Outputs the data in ctf format(a binary trace format)
  - CTF binary output can be viewed using TraceCompass or babeltrace.

      installation:
  ```bash
      rocprofiler-plugins_9.0.0-local_amd64.deb
      rocprofiler-plugins-9.0.0-local.x86_64.rpm
  ```
      usage:
  ```bash
          # plugin_name can be file, perfetto , ctf
          ./rocprofv2 --plugin plugin_name -i samples/input.txt -d output_dir <app_relative_path> # -d is optional, but can be used to define the directory output for output results
  ```

- Device Profiling: A device profiling session allows the user to profile the GPU device for counters irrespective of the running applications on the GPU. This is different from application profiling. device profiling session doesn't care about the host running processes and threads. It directly provides low level profiling information.

- Session Support: A session is a unique identifier for a profiling/tracing/pc-sampling task. A ROCProfilerV2 Session has enough information about what needs to be collected or traced and it allows the user to start/stop profiling/tracing whenever required. More details on the API can be found in the API specification documentation that can be installed using rocprofiler-doc package. Samples also can be found for how to use the API in samples directory.

## Tests

 We make use of the GoogleTest (Gtest) framework to automatically find and add test cases to the CMAKE testing environment. ROCProfilerV2 testing is categorized as following:

- unittests (Gtest Based) : These includes tests for core classes. Any newly added functionality should have a unit test written to it.

- featuretests (standalone and Gtest Based): These includes both API tests and tool tests. Tool is tested against different applications to make sure we have right output in every run.

- memorytests (standalone): This includes running address sanitizer for memory leaks, corruptions.

  installation:
  ```bash
      rocprofiler-tests_9.0.0-local_amd64.deb
      rocprofiler-tests-9.0.0-local.x86_64.rpm
  ```

- Optionally, for tests: run the following:

- Option 1, using rocprofv2 script:
  ```bash
      cd build && ./rocprofv2 -t
  ```

- Option 2, using cmake directly:
  ```bash
      cd build && cmake --build . -- -j check
  ```

## Documentation

We make use of doxygen to automatically generate API documentation. Generated document can be found in the following path:

  ```bash
      # ROCM_PATH by default is /opt/rocm
      # It can be set by the user in different location if needed.
      <ROCM_PATH>/share/doc/rocprofv2
  ```

   installation:
  ```bash
      rocprofiler-docs_9.0.0-local_amd64.deb
      rocprofiler-docs-9.0.0-local.x86_64.rpm
  ```

## Samples

- Profiling: Profiling Samples depending on replay mode
- Tracing: Tracing Samples

installation:
  ```bash
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
  - ctf: CTF Plugin
- samples: Samples of how to use the API, and also input.txt input file samples for counter collection.
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
  - tools: Tools needed to run profiling and tracing
    - rocsys: Controlling Session from another CLI
  - utils: Utilities needed by the project
- tests: Tests folder
- CMakeLists.txt: Handles cmake list for the whole project
- build.sh: To easily build and compile rocprofiler
- CHANGELOG.md: Changes that are happening per release

## Support

Please report in the Github Issues

## Limitations
