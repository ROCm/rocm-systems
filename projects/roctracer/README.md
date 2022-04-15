# ROC-tracer

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

- `/opt/rocm/roctracer/include/roctracer.h`

  API header.

- `/opt/rocm/lib/libroctracer64.so`

  .so library.

### `rocTX` API

To use the rocTX API you need the API header and to link your application with `roctx` .so library:

- `/opt/rocm/roctracer/include/roctx.h`

  API header.

- `/opt/rocm/lib/libroctx64.so`

  .so library.

## Library source tree

- `doc`

  Documentation.

- `inc`

  Include header files.

  - `roctracer.h`

    `rocTacer` library public API header.

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
  - ['roctracer' / 'rocTX' profiling C API specification](docroctracer_spec.md)
- Code examples:
  - [test/MatrixTranspose_test/MatrixTranspose.cpp](testMatrixTranspose_test/MatrixTranspose.cpp)
  - [test/MatrixTranspose/MatrixTranspose.cpp](test/MatrixTranspose/MatrixTranspose.cpp)

## Build and run tests

- ROCm is required

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
   mkdir build && cd build
   cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
   make
   ```

- To build and run test:

  ```sh
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
