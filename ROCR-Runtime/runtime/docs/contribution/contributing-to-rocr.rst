.. meta::
   :description: HSA runtime implementation
   :keywords: ROCR, ROCm, library, tool, runtime

.. _contributing-to-rocr:

Contributing to ROCR
========================

This document contains useful information required to contribute to ROCR.

.. _runtime-design:

Runtime design
-----------------

ROCR consists of the following primary layers:

1. :ref:`C interface adaptors <c-interface-adaptors>`

2. C++ interface classes and common functions

3. Device-specific implementations

The first layer provides interfaces to make ROCR APIs available to the user applications.
The second and third layers comprise of the internal ROCR implementation, which is available for contribution.

Additionally, the runtime is dependent on a small utility library that provides simple common functions, limited operating system, compiler abstraction, and atomic operation interfaces.

The following sections list the important files present in the second and third layer.

C++ interface classes and common functions
----------------------------------------------

The C++ interface layer provides abstract interface classes encapsulating commands to HSA signals, agents, and queues. This layer also contains the implementation of device-independent commands, such as ``hsa_init``, ``hsa_system_get_info``, and a default signal and queue implementation.

Files present in this layer:

- ``runtime.h`` (cpp)

- ``agent.h``

- ``queue.h``

- ``signal.h``

- ``memory_region.h`` (cpp)

- ``checked.h``

- ``memory_database.h`` (cpp)

- ``default_signal.h`` (cpp)

Device-specific implementations
----------------------------------

The device-specific layer contains implementations of the C++ interface classes that implement HSA functionality for ROCm supported devices.

Files present in this layer:

- ``amd_cpu_agent.h`` (cpp)

- ``amd_gpu_agent.h`` (cpp)

- ``amd_hw_aql_command_processor.h`` (cpp)

- ``amd_memory_region.h`` (cpp)

- ``amd_memory_registration.h`` (cpp)

- ``amd_topology.h`` (cpp)

- ``host_queue.h`` (cpp)

- ``interrupt_signal.h`` (cpp)

- ``hsa_ext_private_amd.h`` (cpp)

Source and include directories
--------------------------------

- ``core``: Source code for AMD’s implementation of the core HSA Runtime API’s

- ``cmake_modules``: CMake support modules and files

- ``inc``: Public and AMD-specific header files exposing the HSA Runtime`s interfaces

- ``libamdhsacode``: Code object definitions and interfaces

- ``loader``: Loads code objects

- ``utils``: Utilities required to build the core runtime