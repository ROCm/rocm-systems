.. meta::
   :description: ROCm Compute Profiler terminology and definitions
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, glossary, definitions, terms, profiler, tool,
              Instinct, accelerator, AMD

***********
Definitions
***********

The following table briefly defines some terminology used in ROCm Compute Profiler interfaces
and in this documentation.

.. include:: ./includes/terms.rst

.. include:: ./includes/normalization-units.rst

.. _memory-spaces:

Memory spaces
=============

AMD Instinct™ MI-Series GPUs can access memory through multiple address spaces
which may map to different physical memory locations on the system. The
following table provides a view into how various types of memory used
in HIP map onto these constructs:

.. list-table::
   :header-rows: 1

   * - LLVM Address Space
     - Hardware Memory Space
     - HIP Terminology

   * - Generic
     - Flat
     - N/A

   * - Global
     - Global
     - Global

   * - Local
     - LDS
     - LDS/Shared

   * - Private
     - Scratch
     - Private

   * - Constant
     - Same as global
     - Constant

The following is a high-level description of the address spaces in the AMDGPU
backend of LLVM:

.. list-table::
   :header-rows: 1

   * - Address space
     - Description

   * - Global
     - Memory that can be seen by all threads in a process, and may be backed by
       the local GPU's HBM, a remote GPU's HBM, or the CPU's
       DRAM.

   * - Local
     - Memory that is only visible to a particular workgroup. On AMD Instinct
       GPU hardware, this is stored in :doc:`LDS <cdna/local-data-share>`
       memory.

   * - Private
     - Memory that is only visible to a particular [work-item](workitem)
       (thread), stored in the scratch space on AMD Instinct GPUs.

   * - Constant
     - Read-only memory that is in the global address space and stored on the
       local GPU's HBM.

   * - Generic
     - Used when the compiler cannot statically prove that a pointer is
       addressing memory in a single (non-generic) address space. Mapped to Flat
       on AMD Instinct GPUs, the pointer could dynamically address
       global, local, private or constant memory.

`LLVM's documentation for AMDGPU Backend <https://llvm.org/docs/AMDGPUUsage.html#address-spaces>`_
has the most up-to-date information. Refer to this source for a more complete
explanation.

.. _memory-type:

Memory type
===========

AMD Instinct GPUs contain a number of different memory allocation types to enable the HIP language :doc:`memory coherency model <hip:how-to/hip_runtime_api/memory_management/coherence_control>`.
These memory types are broadly similar between AMD Instinct GPU
generations, but may differ in exact implementation.

In addition, these memory types might differ between GPUs on the same
system, even when accessing the same memory allocation.

For example, AMD Instinct :ref:`MI2XX <mixxx-note>` Series GPU accessing fine-grained
memory allocated local to that device may see the allocation as coherently
cacheable, while a remote GPU might see the same allocation as
uncached.

These memory types include:

.. list-table::
   :header-rows: 1

   * - Memory type
     - Description

   * - Uncached Memory (UC)
     - Memory that will not be cached in this GPU. On
       AMD Instinct :ref:`MI2XX <mixxx-note>` Series GPU, this corresponds to fine-grained
       (or coherent) memory allocated on a remote GPU or the host,
       for example, using ``hipHostMalloc`` or ``hipMallocManaged`` with default
       allocation flags.

   * - Non-hardware-Coherent Memory (NC)
     - Memory that will be cached by the GPU, and is only guaranteed to
       be consistent at kernel boundaries / after software-driven
       synchronization events. On AMD Instinct :ref:`MI2XX <mixxx-note>` Series GPUs, this
       type of memory maps to, coarse-grained ``hipHostMalloc`` memory that is, allocated with the ``hipHostMallocNonCoherent``
       flag or ``hipMalloc`` memory allocated on a remote GPU.

   * - Coherently Cachable (CC)
     - Memory for which only reads from the GPU where the memory was
       allocated will be cached. Writes to CC memory are uncached, and trigger
       invalidations of any line within the GPU. On AMD Instinct :ref:`MI2XX <mixxx-note>` Series GPUs, this type of memory maps to
       fine-grained memory allocated on the local GPU using, for
       example, the ``hipExtMallocWithFlags`` API using the
       ``hipDeviceMallocFinegrained`` flag.

   * - Read/Write Coherent Memory (RW)
     - Memory that will be cached by the GPU, but may be invalidated by
       writes from remote devices at kernel boundaries / after software-driven
       synchronization events. On AMD Instinct :ref:`MI2XX <mixxx-note>` Series GPUs, this
       corresponds to coarse-grained memory allocated locally to the
       GPU, using for example, the default ``hipMalloc`` allocator.

For more details on coarse and fine-grained memory allocations and what
type of memory is returned by various combinations of memory allocators, flags
and arguments, see `Crusher quick-start guide <https://docs.olcf.ornl.gov/systems/crusher_quick_start_guide.html#floating-point-fp-atomic-operations-and-coarse-fine-grained-memory-allocations>`_.
