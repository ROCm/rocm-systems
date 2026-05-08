.. meta::
  :description: The SAXPY tutorial on HIP
  :keywords: AMD, ROCm, HIP, SAXPY, tutorial

*******************************************************************************
Working with HIP Tests
*******************************************************************************

This tutorial introduces the HIP tests component available alongside the HIP API and runtime.
HIP tests are not installed as part of the standard ROCm installation, but you
can build ``rocm-systems/projects/hip-tests`` as described in :ref:`build-tests`.

The catch tests and samples provided in ``hip-tests`` are both a diagnostic tool and a learning toolkit.
You can use the tools and tests to validate your own system environment and HIP installation. You can
also review the code and see how specific aspects of the HIP API are used under different conditions. 

Running HIP tests
=================

The HIP tests are catch tests provided alongside the HIP component, letting you test your
system and the HIP runtime. In addition, HIP tests provide some sample applications and
utilities that can be individually built and run as needed. 

The top level of ``rocm-systems/projects/hip-tests`` is organized around one purpose: it is the official unit‑test
and sample suite for HIP. The ``rocm-systems/projects/hip-tests/catch`` folder contains the lightweight Catch2-based
C++ testing library unit test framework used by HIP tests. It provides the test harness for HIP API tests and lets
you see how HIP functions are used and validated in isolation.

The catch tests of ``hip-tests`` serves two complementary purposes: provides a comprehensive diagnostic suite for your
HIP installation, and executable examples of how to write catch tests for your own code. The tests show:

* How to structure a HIP unit test
* How to wrap HIP API calls with assertions
* How to test kernels deterministically
* How to test memory behavior, events, streams, and error handling
* How to isolate failures cleanly
* How to write reproducible GPU tests that don’t depend on timing or scheduling quirks

Follow the instructions in :ref:`build-tests`, and run the tests using the ``ctest`` command from the ``build`` folder. The command runs more than 4,000 tests and writes results in the ``build/Testing/Temporary/LastTest.log`` file. You can also build and run individual tests as explained in the build instructions. 

Building and running HIP samples
================================

The ROCm/hip-tests/samples folder is organized into a set of small utilities that demonstrate how to use HIP APIs. Each sample is a utility that illustrates a concept, pattern, or technique. The folder is divided into three major groups:

* ``samples/0_Intro``: Includes basic HIP API usage and simple kernels.

  - ``samples/0_Intro/bit_extract``: Runs a small HIP kernel that extracts specific bits from integers. It demonstrates writing a small kernel, launching the kernel, managing memory, and validating results on the host.
  - ``samples/0_Intro/generic_target``: Demonstrates building code for generic GPU targets rather than for specific architectures. This is useful when you want to distribute HIP binaries that run on a variety of AMD GPUs without recompiling. 
  - ``samples/0_Intro/module_api``: Shows how to use the HIP module API, which is analogous to CUDA’s driver API. It demonstrates loading precompiled code objects (.co or .hsaco) and launching kernels via the module API instead of the runtime API.
  - ``samples/0_Intro/module_api_global``: Extends the module API example to show how to access global variables defined inside a GPU module, use global variables to configure the kernel, and copy data to and from those global variables. 
  - ``samples/0_Intro/square``: Implements a small kernel that squares each element of an array. It demonstrates minimal kernel structure, memory allocation, launch parameters, and verifying results on the host. 

* ``samples/1_Utils``: Provides reusable helper utilities and patterns.

  - ``samples/1_Utils/hipDispatchLatency``: This utility benchmarks the latency of dispatching a HIP kernel, focusing on the performance cost of launching kernels rather than the cost of running them.
  - ``samples/1_Utils/hipInfo``: Provides information about the HIP runtime and the available GPU devices. ``hipInfo`` is similar to ``rocminfo``, but returns only what the HIP runtime exposes via ``hipGetDeviceProperties()`` and related APIs. 

* ``samples/2_Cookbook``: Demonstrates more advanced examples and build techniques. The concepts covered by the Cookbook include a variety of operations such as matrix operations, memory‑access patterns, warp/wavefront shuffle and data exchange, streams, concurrency, and multi‑GPU, HIP events and timing, occupancy, architecture, and device‑specific behavior. The following is a brief description of the available modules. 

  - ``samples/2_Cookbook/0_MatrixTranspose``: Demonstrates implementing and optimizing a matrix transpose kernel. Helps to understand coalesced versus non‑coalesced memory access and shared‑memory tiling.
  - ``samples/2_Cookbook/1_hipEvent``: Uses HIP event timing for profiling and benchmarking kernels for performance measurement.
  - ``samples/2_Cookbook/3_shared_memory``: Demonstrates how to use shared memory to accelerate memory‑bound kernels.
  - ``samples/2_Cookbook/4_shfl``: Demonstrates warp‑level shuffle operations used for reductions, scans, and warp‑synchronous algorithms.
  - ``samples/2_Cookbook/5_2dshfl``: Extends shuffle operations to two-dimensional patterns for stencil operations, and two-dimensional data exchange without shared memory.
  - ``samples/2_Cookbook/6_dynamic+shared``: Shows how to allocate dynamic shared memory at kernel launch. This is useful when memory requirements depend on runtime parameters.
  - ``samples/2_Cookbook/7_streams``: Demonstrates HIP streams to enable overlapping compute and memory transfers to hide launch latency and manage pipelined workloads.
  - ``samples/2_Cookbook/8_peer2peer``: Shows P2P GPU memory access for multi‑GPU systems.
  - ``samples/2_Cookbook/9_unroll``: Demonstrates loop unrolling techniques useful for tuning compute‑heavy kernels to improve performance.
  - ``samples/2_Cookbook/10_inline_asm``: Shows how to embed inline GPU assembly code in HIP kernels. Used for optimizations or accessing instructions not exposed in the HIP language. 
  - ``samples/2_Cookbook/11_texture_driver``: Demonstrates texture objects for image processing, sampling, and spatial filtering workloads using the HIP driver API.
  - ``samples/2_Cookbook/12_cmake_hip_add_executable``: Shows how to use ``hip_add_executable()`` in CMake for simple HIP projects.
  - ``samples/2_Cookbook/13_occupancy``: Illustrates how to compute kernel occupancy. Useful for tuning block sizes and understanding hardware limits. 
  - ``samples/2_Cookbook/14_gpu_arch``: Shows how to query GPU architecture and adapt behavior for different devices. This is useful for writing kernels that behave differently on different devices.
  - ``samples/2_Cookbook/15_MatrixTranspose``: 
  - ``samples/2_Cookbook/16_assembly_to_executable``: Demonstrates assembling GPU ISA into an executable code object.
  - ``samples/2_Cookbook/17_llvm_ir_to_executable``: Shows how to take LLVM IR and turn it into a GPU executable.
  - ``samples/2_Cookbook/18_cmake_hip_device``: Demonstrates device‑side compilation with HIP in CMake. Useful for multi‑file device code.
  - ``samples/2_Cookbook/19_cmake_lang``: Shows how to use HIP as a first‑class CMake language.
  - ``samples/2_Cookbook/21_cmake_hip_cxx_lang``: Demonstrates mixing HIP with C++ and Clang toolchains for hybrid CPU/GPU projects.
  - ``samples/2_Cookbook/22_cmake_hip_lang``: Shows how to use the hip-lang CMake integration for advanced CMake setups and cross‑platform HIP builds.
  - ``samples/2_Cookbook/23_cmake_hiprtc``: Demonstrates HIP runtime compilation (HIPRTC) using CMake.

  .. note::

    There are no ``2_XXX`` or ``20_XXX`` code samples in the Cookbook. ``22_cmake_hip_lang`` is not built as part of the batch build process, and must be manually built. 

Building samples
----------------

To build all the hip-tests/samples use the following commands:

.. code:: bash

  cd hip-tests
  mkdir -p build && cd build
  cmake ../samples
  make build_samples


Alternatively, to build specific samples use the following commands rather than ``build_samples``:

.. code:: bash

  make build_intro
  make build_utils
  make build_cookbook

Finally, to build individual samples use the following commands:

.. code:: bash

  cd samples/0_Intro/bit_extract
  mkdir -p build && cd build
  cmake ..
  make all
