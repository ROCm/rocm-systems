# ROC-tracer

> [!IMPORTANT]
We are phasing out development and support for roctracer/rocprofiler/rocprof/rocprofv2 in favor of rocprofiler-sdk/rocprofv3 in upcoming ROCm releases. Going forward, only critical defect fixes will be addressed for older versions of profiling tools and libraries. We encourage all users to upgrade to the latest version, rocprofiler-sdk library and rocprofv3 tool, to ensure continued support and access to new features.

> [!NOTE]
> The published documentation is available at [ROCTracer](https://rocm.docs.amd.com/projects/roctracer/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents.

- **ROC-tracer library: Runtimes Generic Callback/Activity APIs**

  The goal of the implementation is to provide a generic independent from specific runtime profiler to trace API and asynchronous activity.

  The API provides functionality for registering the runtimes API callbacks and asynchronous activity records pool support.

- **ROC-TX library: Code Annotation Events API**

  Includes API for:

  - `roctxMark`
  - `roctxRangePush`
  - `roctxRangePop`

## Usage

### `rocTracer` API

To use the rocTracer API you need the API header and to link your application with `roctracer` .so library:

- `/opt/rocm/include/roctracer/roctracer.h`

  API header.

- `/opt/rocm/lib/libroctracer64.so`

  .so library.

### `rocTX` API

To use the rocTX API you need the API header and to link your application with `roctx` .so library:

- `/opt/rocm/include/roctracer/roctx.h`

  API header.

- `/opt/rocm/lib/libroctx64.so`

  .so library.

## Library source tree

- `doc`

  Documentation.

- `inc`

  Include header files.

  - `roctracer.h`

    `rocTracer` library public API header.

  - `roctx.h`
  
    `rocTX` library public API header.

- `src`
  
  Library sources.

  - `core`

    `rocTracer` library API sources.

  - `roctx`

    `rocTX` library API sources.

  - `util`

    Library utils sources.

- `test`

  Test suit.

  - `MatrixTranspose`

    Test based on HIP MatrixTranspose sample.

## Documentation

- API description:
  - ['roctracer' / 'rocTX' profiling C API specification](doc/roctracer_spec.md)
- Code examples:
  - [HIP API ops, GPU Activity Tracing](doc/roctracer_spec.md#41-hip-api-ops-gpu-activity-tracing)
  - [MatrixTranspose HIP sample with all APIs/activity tracing enabled](doc/roctracer_spec.md#42-matrixtranspose-hip-sample-with-all-apisactivity-tracing-enabled)

## Build and run tests

- ROCm is required

- Packages required:

  1. For Ubuntu 18.04 and Ubuntu 20.04 the following adds the needed packages:

   ````shell
   apt install python3 python3-pip gcc g++ libatomic1 make rocm-llvm-dev \
    cmake doxygen graphviz texlive-full
   ````

  2. For CentOS 8.1 and RHEL 8.1 the following adds the needed packages:

   ````shell
   yum install -y python3 python3-pip gcc gcc-g++ make rocm-llvm-devel \
    cmake libatomic doxygen graphviz texlive \
    texlive-xtab texlive-multirow texlive-sectsty \
    texlive-tocloft texlive-tabu texlive-adjustbox
   ````

  3. For SLES 15 Service Pack 15 the following adds the needed packages:

   ````shell
   zypper in python3 python3-pip gcc gcc-g++ make rocm-llvm-devel \
    cmake libatomic doxygen graphviz \
    texlive-scheme-medium texlive-hanging texlive-stackengine \
    texlive-tocloft texlive-etoc texlive-tabu
   ````

- Python modules requirements: `CppHeaderParser`, `argparse`.

  To install:

  ```sh
  pip3 install CppHeaderParser argparse
  ```

- Clone development branch of `roctracer`:

  ```sh
  git clone -b amd-master https://github.com/ROCm-Developer-Tools/roctracer
  ```

- To build `roctracer` library:

   ```sh
   cd <your path>/roctracer
   ./build.sh
   ```

- To build and run test:

  ```sh
  cd <your path>/roctracer/build
  make mytest
  run.sh
  ```

## Installation

Install by:

  ```sh
  make install
  ```

  or:

  ```sh
  make package && dpkg -i *.deb
  ```
