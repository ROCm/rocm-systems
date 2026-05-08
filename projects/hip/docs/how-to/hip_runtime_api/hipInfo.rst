.. meta::
   :description: hipInfo documentation
   :keywords: AMD, ROCm, HIP, rocminfo, GPU diagnostics, device properties

.. _hipInfo:

Using hipInfo
=============

``hipInfo`` is a diagnostic utility that displays information about AMD GPU devices available on your system. It queries and reports device properties, architectural features, memory information, and peer-to-peer capabilities through the HIP runtime API. ``hipInfo`` is similar to ``rocminfo``, though they are different as described in :ref:`hip-v-rocm-info`. For more information, see `rocminfo <https://rocm.docs.amd.com/projects/rocminfo/en/latest/index.html>`__.

``hipInfo`` can be used for:

* Verifying GPU detection and configuration
* Evaluating device capabilities for HIP application development
* Troubleshooting GPU-related issues
* Gathering system information for bug reports or support requests

.. note::

   ``hipInfo`` is installed as part of the `HIP SDK for Windows <https://rocm.docs.amd.com/projects/install-on-windows/en/latest/>`_. However, on Linux systems it is not installed with HIP, but can be separately built as described in `Building HIP tests <https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html#build-hip-tests>`_. 

hipInfo command
----------------

The command requires no arguments and will automatically detect and report information for all available AMD GPU devices.

.. code-block:: bash

   $ ./hipInfo

Output Information
------------------

``hipInfo`` reports support for various GPU features, such as:

* Atomic operations (int32, int64, float)
* Double precision floating point
* Warp/wavefront operations (vote, ballot, shuffle)
* Thread fence operations
* 3D grid support
* Texture capabilities

The tool also displays the following categories of information for each detected GPU:

- Device Identification

   * **Name**: GPU device name
   * **pciBusID, pciDeviceID, pciDomainID**: PCI location identifiers
   * **gcnArchName**: AMD GCN architecture name (for example: gfx906, gfx90a, gfx1100)

- Compute Capabilities

   * **multiProcessorCount**: Number of compute units
   * **maxThreadsPerMultiProcessor**: Maximum threads per compute unit
   * **maxThreadsPerBlock**: Maximum threads per work-group
   * **maxThreadsDim**: Maximum work-group dimensions (x, y, z)
   * **maxGridSize**: Maximum grid dimensions (x, y, z)
   * **warpSize**: Wavefront size (typically 64 for AMD GPUs)
   * **regsPerBlock**: Registers available per block
   * **major/minor**: Compute capability version

- Memory Information

   * **totalGlobalMem**: Total device memory
   * **totalConstMem**: Total constant memory
   * **sharedMemPerBlock**: Shared memory per block
   * **maxSharedMemoryPerMultiProcessor**: Shared memory per compute unit
   * **l2CacheSize**: L2 cache size
   * **memoryBusWidth**: Memory bus width in bits
   * **memInfo.total/free**: Current memory usage statistics

- Clock Rates

   * **clockRate**: GPU core clock rate
   * **memoryClockRate**: Memory clock rate
   * **clockInstructionRate**: Instruction clock rate

- AMD-Specific Properties

   * **isLargeBar**: Whether large BAR (Base Address Register) is enabled
   * **asicRevision**: ASIC revision number
   * **maxAvailableVgprsPerThread**: Maximum vector GPRs per thread
   * **hostNativeAtomicSupported**: Host-side atomic operation support

- Other Properties

   * **cooperativeLaunch**: Cooperative group launch support
   * **cooperativeMultiDeviceLaunch**: Multi-device cooperative launch support
   * **concurrentKernels**: Concurrent kernel execution support
   * **isIntegrated**: Whether GPU is integrated or discrete
   * **peers/non-peers**: P2P (peer-to-peer) capable device relationships

Example Output
--------------

.. code-block:: text

   --------------------------------------------------------------------------------
   device#                           0
   Name:                             AMD Radeon GPU
   pciBusID:                         103
   pciDeviceID:                      0
   pciDomainID:                      0
   multiProcessorCount:              64
   maxThreadsPerMultiProcessor:      2560
   clockRate:                        1800 Mhz
   memoryClockRate:                  1000 Mhz
   memoryBusWidth:                   4096
   totalGlobalMem:                   31.98 GB
   sharedMemPerBlock:                64.00 KB
   warpSize:                         64
   major:                            9
   minor:                            0
   gcnArchName:                      gfx906:sramecc+:xnack-
   memInfo.total:                    31.98 GB
   memInfo.free:                     31.96 GB (100%)

.. _hip-v-rocm-info:

Comparing hipinfo and rocminfo
------------------------------

While both ``hipInfo`` and ``rocminfo`` provide GPU device information, they serve different purposes:

**hipInfo**:

* Uses the HIP runtime API (``hipGetDeviceProperties``)
* Shows device properties from the perspective of HIP application programming
* Displays information relevant to HIP kernel development
* Reports current memory usage and runtime state
* Provides information in a developer-friendly format

**rocminfo**:

* Uses lower-level HSA (Heterogeneous System Architecture) interfaces
* Provides more comprehensive system-level hardware details
* Shows additional information about CPU agents, memory pools, and caches
* Reports hardware topology and NUMA relationships
* Includes ISA details and agent characteristics

Use ``hipInfo`` when you need to quickly assess GPU compute capabilities and memory, check device capabilities for HIP programming decisions, and verify device properties available to your HIP applications. 

Use ``rocminfo`` to understand detailed system topology with comprehensive hardware specifications, verify HSA runtime functionality, or diagnose low-level hardware detection or configuration issues. For more information, see `ROCmInfo documentation <https://rocm.docs.amd.com/projects/rocminfo/en/latest/index.html>`__.
