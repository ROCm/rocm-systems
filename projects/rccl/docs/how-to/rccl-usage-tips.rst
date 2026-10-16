.. meta::
   :description: Usage tips for the RCCL library of collective communication primitives
   :keywords: RCCL, ROCm, library, API, peer-to-peer, transport

.. _rccl-usage-tips:


*****************************************
RCCL usage tips
*****************************************

This topic describes common RCCL configuration options and usage tips.

Profiling
=========

For fine-grained profiling of collective operations, use the RCCL **profiler plugin** API and related tooling rather than legacy in-tree profilers.

MSCCL and MSCCL++ integration has been removed from RCCL. The legacy API symbols ``mscclLoadAlgo``,
``mscclRunAlgo``, and ``mscclUnloadAlgo`` remain as no-ops for link compatibility.

Enabling peer-to-peer transport
===============================

To enable peer-to-peer access on machines with PCIe-connected GPUs,
set the HSA environment variable as follows:

.. code-block:: shell

   HSA_FORCE_FINE_GRAIN_PCIE=1

This feature requires GPUs that support peer-to-peer access along with
proper large BAR addressing support.

Symmetric memory and ``NCCL_P2P_LEVEL``
=======================================

RCCL can accelerate some collectives (for example, allreduce, allgather, and
reduce-scatter) through a *symmetric memory* path. This path uses
:doc:`Virtual Memory Management <../api-reference/api-library>`-backed
buffers that are registered as symmetric windows (``ncclCommWindowRegister``
with the ``NCCL_WIN_COLL_SYMMETRIC`` flag) so that every participating rank can
address the buffer directly. ``ncclMemAlloc`` allocates memory suitable for such
a registration, but it does not create a symmetric window on its own.

Whether a communicator can use symmetric memory is decided once at
``ncclCommInitRank`` time. The prerequisites are:

- All local ranks are **peer-to-peer capable** with each other (on AMD
  GPUs this means they are on the same host over PCIe or XGMI,
  or are reachable through a Multi-Node Infinity Fabric clique).
- Virtual Memory Management is enabled (``NCCL_CUMEM_ENABLE=1``).
- Symmetric windows are enabled (``NCCL_WIN_ENABLE=1``, the default).
- Either GPU-Initiated Networking (GIN) is available, or the communicator is a
  single locality (one-LSA) team.

.. note::

   Symmetric-memory availability does **not** depend on the
   ``NCCL_P2P_LEVEL`` distance setting. This matches the behavior introduced in
   upstream NCCL 2.28.7, which removed the topology-distance check from the
   symmetric-memory decision. Restricting peer-to-peer reach with a value such
   as ``NCCL_P2P_LEVEL=PHB`` (or even disabling distance-based P2P entirely with
   ``NCCL_P2P_DISABLE=1``) changes how the *flat* P2P transport schedule is
   built, but it does **not** by itself turn off the symmetric path: as long as
   the GPUs are CUDA peer-to-peer capable, symmetric memory remains eligible.

To confirm whether symmetric memory was enabled for a run, inspect the init
logs:

.. code-block:: shell

   NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT ./your_app

When symmetric memory is **not** available, RCCL logs a line that begins with
``Symmetric memory is not supported`` and reports which prerequisite was
missing (for example, ``cuMemEnable`` or ``globalGinSupport``). If you expect
the symmetric path but it is disabled, check those prerequisites rather than
``NCCL_P2P_LEVEL``.

.. _device-symmetric-memory:

Device-side symmetric memory access (``ncclFindWindow``)
========================================================

In addition to the host-side collective API, RCCL exposes a *device-side* API
that lets your own GPU kernels read and write peer ranks' symmetric buffers
directly, without a host-driven collective. This is the mechanism that backs
RCCL's fused symmetric kernels. It requires the same symmetric-memory
prerequisites described above.

Working with symmetric memory from a kernel involves three pieces:

- A **device communicator** (``ncclDevComm``), created on the host with
  ``ncclDevCommCreate`` and passed to the kernel by value. It carries the rank
  layout, barriers, and the device-side window registry.
- One or more **symmetric windows**, registered on the host with
  ``ncclCommWindowRegister`` using the ``NCCL_WIN_COLL_SYMMETRIC`` flag.
  ``ncclMemAlloc`` only allocates and maps symmetric-capable memory; it does not
  populate the window registry, so the explicit registration call is required.
- The **device API** (available through ``nccl_device.h``) that turns a window
  plus an offset into a pointer to a specific peer's copy of the buffer.

When a kernel already holds the ``ncclWindow_t`` handle (for example, because
the host passed it as a launch argument), it can call ``ncclGetLsaPointer``
directly. When a kernel only has a raw device pointer, it can resolve the
backing window from the device-side registry with ``ncclFindWindow`` (introduced
in NCCL 2.28.7), with no host round-trip:

.. code-block:: cpp

   template <typename Coop>
   ncclWindow_t ncclFindWindow(Coop coop, ncclDevComm const& comm, void const* ptr);

- ``coop`` is a cooperative-thread group (for example ``ncclCoopCta()``). The
  lookup is warp-coalesced, so **every thread in the group must call it**; the
  same window handle is returned to all participating threads.
- ``comm`` is the device communicator passed to the kernel.
- ``ptr`` is any address that falls within a registered symmetric window's
  ``[base, base + size)`` range.

``ncclFindWindow`` walks the communicator's window registry, which
``ncclCommWindowRegister`` populates, and returns the matching ``ncclWindow_t``.
``ncclDevCommCreate`` does not fill this registry; it only publishes a pointer
to it into the device communicator. Passing a pointer that no registered
window covers is undefined behavior rather than a recoverable error: the lookup
returns only from its hit path, so a miss runs past the end of the registry and
faults. Make sure the pointer is covered by a registration instead of testing
the result against ``nullptr``. Once you have a window, resolve a specific
peer's copy of the buffer with one of the pointer helpers:

- ``ncclGetLsaPointer(window, offset, peer)`` returns a pointer to ``peer``'s
  copy of the buffer at ``offset`` bytes into the window, for peers reachable
  through Local Symmetric Access (LSA, that is, direct intra-node
  peer-to-peer).
- ``ncclGetPeerPointer(window, offset, team, peer)`` is the team-relative form.
- ``ncclGetMultimemPointer`` / ``ncclGetLsaMultimemPointer`` return a multicast
  pointer when multimem is available.

The ``offset`` is measured from the window base, so a buffer registered at its
base address is accessed at ``offset = 0``.

The kernel below is handed only its local buffer pointer. It resolves the
backing window on the device, then reads the value stored by its ring neighbor:

.. code-block:: cpp

   #include <nccl_device.h>

   __global__ void readPeerValue(void* localPtr, int* out, ncclDevComm_t comm)
   {
       // Synchronize the local symmetric team before touching peer memory.
       ncclLsaBarrierSession<ncclCoopCta> barrier(
           ncclCoopCta(), comm, ncclTeamLsa(comm), comm.lsaBarrier, blockIdx.x);
       barrier.sync(ncclCoopCta(), cuda::memory_order_relaxed);

       // Device-side registry lookup (all threads participate).
       ncclWindow_t window = ncclFindWindow(ncclCoopCta(), comm, localPtr);

       if (threadIdx.x == 0) {
           // ncclGetLsaPointer indexes the LSA team, so the peer must be an LSA
           // rank. Use ncclGetPeerPointer(window, 0, team, peer) to address a
           // rank of some other team by its rank in that team.
           int  peer     = (comm.lsaRank + 1) % comm.lsaSize;
           int* peerData = reinterpret_cast<int*>(ncclGetLsaPointer(window, 0, peer));
           out[0] = peerData[0];
       }

       barrier.sync(ncclCoopCta(), cuda::memory_order_release);
   }

On the host, register the buffer as a symmetric window and build the device
communicator before launching:

.. code-block:: cpp

   ncclCommWindowRegister(comm, buffer, bytes, &window, NCCL_WIN_COLL_SYMMETRIC);

   ncclDevCommRequirements_t req = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
   req.lsaBarrierCount = 1;
   ncclDevComm_t devComm;
   ncclDevCommCreate(comm, &req, &devComm);

   readPeerValue<<<1, 64, 0, stream>>>(buffer, out, devComm);

The device communicator holds a pointer to the communicator's live window
registry, so ``ncclFindWindow`` resolves any window registered with
``NCCL_WIN_COLL_SYMMETRIC``, whether it was registered before or after
``ncclDevCommCreate``. The only ordering requirement is that the registration
completes before the kernel that calls ``ncclFindWindow`` runs, on a
communicator for which symmetric memory is available. If ``ncclDevCommCreate``
returns ``ncclInvalidUsage``, or a kernel
faults inside the lookup, confirm the prerequisites above
(``NCCL_CUMEM_ENABLE=1``, ``NCCL_WIN_ENABLE=1``, peer-to-peer capable ranks),
that the buffer was registered with ``NCCL_WIN_COLL_SYMMETRIC``, and that the
pointer passed to ``ncclFindWindow`` lies within that registration.

Ignoring CPU affinity with multi-node
=====================================

Depending on the job launcher and the requirements of your workload, performance as the communication workload scales
can be improved by setting ``NCCL_IGNORE_CPU_AFFINITY``.  This allows the RCCL communication library to 
ignore the job's supplied CPU affinity and use the GPU affinity only.

.. code-block:: shell

   NCCL_IGNORE_CPU_AFFINITY=1

For general usage, this environment variable is not set so it doesn't interfere with the user or launcher
supplied preferences.

Improving performance on the MI200 series
=========================================

On MI200 series (gfx90a) systems, such as MI210, MI250, and MI250X, running
ROCm 7.13 or later, set ``HSA_NO_SCRATCH_RECLAIM=1`` when running RCCL:

.. code-block:: shell

   export HSA_NO_SCRATCH_RECLAIM=1

Without this setting, per-launch scratch-memory reclaim in the runtime adds a
fixed overhead to every collective launch. This overhead dominates
small-message (under 16 MB) latency and can degrade it by roughly
5-10x compared to earlier ROCm releases. Setting ``HSA_NO_SCRATCH_RECLAIM=1``
removes the overhead and restores the expected small-message latency.

Improving performance on the MI300X and MI350X
===============================================

This section outlines ways to improve RCCL performance on MI300X and MI350X
systems, including guidelines for systems with fewer than eight GPUs, the most
efficient GPU partition modes, and channel tuning for multi-node
configurations. Where behavior applies to both architectures, both are called
out together; where behavior differs, the difference is described in its own
subsection.

Configuration with fewer than eight GPUs
----------------------------------------

On a system with eight MI300X accelerators, each pair of accelerators is
connected with dedicated Infinity Fabric™ links in a fully connected topology.
For collective operations, this can achieve good performance when all eight
accelerators (and all Infinity Fabric links) are used. When fewer than eight
GPUs are used, however, this can only achieve a fraction of the potential
bandwidth on the system. However, if your workload warrants using fewer than
eight MI300X accelerators on a system, you can set the run-time variable
``NCCL_MIN_NCHANNELS`` to increase the number of channels. For example:

.. code-block:: shell

   export NCCL_MIN_NCHANNELS=32

Increasing the number of channels can benefit performance, but it also increases
GPU utilization for collective operations.
Additionally, RCCL pre-defines a higher number of channels when only two or four
accelerators are in use on a 8\*MI300X system. In this situation, RCCL uses 32
channels with two MI300X accelerators and 24 channels for four MI300X
accelerators.

.. _nps4_cpx_mi300_rccl:

NPS4 and CPX partition modes
----------------------------

The term compute partitioning modes, or Modular Chiplet Platform (MCP), refers to the
logical partitioning of XCDs into devices in the ROCm stack. The names are
derived from the number of logical partitions that are created out of the eight
XCDs. In the default mode, SPX (Single Partition X-celerator), all eight XCDs are
viewed as a single logical compute element, meaning that the :doc:`amd-smi <amdsmi:index>`
utility will show a single MI300X or MI350X device. In CPX (Core Partitioned X-celerator)
mode, each XCD appears as a separate logical GPU, for example, as eight separate
GPUs in :doc:`amd-smi <amdsmi:index>` per MI300X or MI350X device. CPX mode can be viewed as
having explicit scheduling privileges for each individual compute element (XCD).
Both the MI300X and MI350X support SPX and CPX compute partitioning modes.

While compute partitioning modes change the space on which you can assign work
to compute units, the memory partitioning modes (known as Non-Uniform Memory
Access (NUMA) Per Socket (NPS)) change the number of NUMA domains that a device
exposes. In other words, it changes the number of HBM stacks which are
accessible to a compute unit, and therefore the size of its memory space. However,
for the MI300X and MI350X, the number of memory partitions must be less than or equal to
the number of compute partitions. On the MI300X, NPS4 (viewing pairs of HBM stacks as a
disparate element), for example, is only enabled when in CPX mode (viewing each
XCD as a disparate element). The MI350X doesn't support NPS4; see
`MI350X partition mode differences`_ for the memory partitioning modes it
supports instead.

- Compute partition modes 

  - In SPX mode, workgroups launched to the device are distributed
    round-robin to the XCDs in the device, meaning that the programmer cannot
    have explicit control over which XCD a workgroup is assigned to.

  - In CPX mode, workgroups are launched to a single XCD, meaning the
    programmer has explicit control over work placement onto the XCDs.
  
- Memory partition modes 

  - In NPS1 mode (compatible with CPX and SPX), the entire memory is accessible
    to all XCDs.

  - In NPS4 mode (compatible with CPX), each memory quadrant of the memory is
    directly visible to the logical devices in its quadrant. An XCD can still
    access all portions of memory through multi-GPU programming techniques.

The MI300X CPX mode can be accessed using the following :doc:`amdsmi:index`
commands.

.. code-block:: shell

   amd-smi set --gpu all --compute-partition CPX
   amd-smi set --gpu all --memory-partition NPS4

MI350X/MI355X partition modes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For MI350X/MI355X, RCCL supports the DPX compute partitioning mode and the NPS2 memory partitioning mode.

.. code-block:: shell

   amd-smi set --gpu all --compute-partition DPX
   amd-smi set --gpu all --memory-partition NPS2

RCCL performance with CPX and NPS4 on MI300X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following benchmark results were measured on MI300X systems. To run RCCL
allreduce on 64 GPUs with CPX+NPS4 mode on the MI300X, use this
example:

.. code-block:: shell

   mpirun -np 64 --bind-to numa rccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1

To run RCCL allreduce on 8 GPUs in the same OAM with CPX+NPS4 mode on the
MI300X, use this example:

.. code-block:: shell

   export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

   mpirun -np 8 --bind-to numa rccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1

RCCL delivers improved allreduce performance in CPX mode for TP=8 (8 GPUs in
the same OAM) on the MI300X.

.. code-block:: shell

   export HIP_FORCE_DEV_KERNARG=1
   export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

   mpirun -np 8 --bind-to numa rccl-tests/build/all_reduce_perf -b 32 -e 1G -f 2 -g 1 -G 2 -w 20 -n 50

Here are the benchmark results for in-place (where the output buffer is used as
the input buffer) and out-of-place allreduce bus bandwidth.

.. figure:: ../data/how-to/rccl-usage-tips/in-place_allreduce.png
    :alt: In-place allreduce benchmark results
    :align: center

.. figure:: ../data/how-to/rccl-usage-tips/out-of-place_allreduce.png
    :alt: Out-of-place allreduce benchmark results
    :align: center

A significant performance improvement is achievable with optimized CPX mode,
which peaks at ~340 GB/s with a single OAM. The difference in bus bandwidth
between the unoptimized and optimized modes increases as the buffer size grows.

Using RCCL and CPX in PyTorch on MI300X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The following benchmark results were measured on MI300X systems. The PyTorch
all_reduce benchmark is used to reproduce the performance reported
by RCCL-Tests with the RCCL and CPX optimizations.

.. note::

   To use RCCL with CPX mode in PyTorch, check the RCCL version used by PyTorch.

   For a virtualenv with a .whl-based PyTorch setup (such as nightly/rocm6.2),
   this would be in 
   ``<path-to-your-venv>/lib/<python-version>/site-packages/torch/lib/librccl.so``
   This is the version of RCCL that is packaged as part of ROCm version 6.2.

   RCCL for CPX mode was enabled in ROCm 6.3.0. To use the CPX features, replace
   the existing ``librccl.so`` with one from ROCm 6.3.0 or newer or from a local
   build of the RCCL develop branch.

To test the effects of RCCL on PyTorch, the `stas00 all reduce benchmark <https://github.com/stas00/ml-engineering/blob/master/network/benchmarks/all_reduce_bench.py>`_
was used. The following command is used to run a single OAM allreduce
benchmark:

.. code-block:: shell

   export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   python -u -m torch.distributed.run --nproc_per_node=8 --rdzv_endpoint localhost:6000  --rdzv_backend c10d all_reduce_bench.py

For better performance, the ``HIP_FORCE_DEV_KERNARG`` and
``TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK`` environment variables are
set during the benchmark in the following manner:

.. code-block:: shell

   export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=1
   export HIP_FORCE_DEV_KERNARG=1
   export ROCR_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
   python -u -m torch.distributed.run --nproc_per_node=8 --rdzv_endpoint localhost:6000  --rdzv_backend c10d all_reduce_bench.py

The default allreduce PyTorch benchmark peak bus bandwidth performance is
~170 GB/s on a single OAM with ROCm 6.2.4, while the optimized run for CPX on a
single OAM peaks at ~315 GB/s.

RCCL channel tuning for multi-node MI350X (ROCm 7.14 / RCCL 2.30.4)
--------------------------------------------------------------------

Starting with RCCL 2.30.4 (ROCm 7.14), the default number of communication
channels for multi-node collectives on MI350X/MI355X in SPX mode has been reduced from 64 to 48,
leaving additional Compute Units (CUs) free for compute kernels that run
concurrently with communication. For computation-bound workloads that rely
heavily on communication-computation overlap, staying with this default may
be beneficial since it leaves more CUs available for compute, but this is a
deliberate trade-off against raw communication bandwidth, so users measuring
standalone rccl-tests bandwidth (with no concurrent compute contending for
CUs) or running communication-bound workloads with negligible concurrent
compute may see a regression compared to ROCm 7.2.1/RCCL 2.27.7. This expected
behavior can be recovered by setting both ``NCCL_MAX_NCHANNELS=64`` and
``NCCL_MIN_NCHANNELS=64`` (64 is the effective ceiling for multi-node jobs).
In either case, the actual channel count in use should be confirmed via
``NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,TUNING,COLL`` before drawing
performance conclusions.

As of ROCm 10.0, the default communication channel count will revert back to
64 CUs.

.. list-table:: Default max multi-node channel (CU) limits on MI350X by ROCm version
   :header-rows: 1
   :widths: 20 30

   * - ROCm version
     - Default max multi-node CUs
   * - 7.14
     - 48
   * - 10.0
     - 64

Context tracking on GPUs
----------------------------------------
Context tracking is disabled by default for optimal performance. However, enabling of context tracking can significantly improve performance
in certain scenarios. To enable context tracking, set the following environment variable:

.. code-block:: shell


   export RCCL_ENABLE_CONTEXT_TRACKING=1

.. _suspend-resume:

Suspending and resuming a communicator
======================================

A long-lived application can hold several RCCL communicators that are only used
during specific phases. While a communicator is idle, the GPU memory it holds
for channel buffers, transport FIFOs, and similar resources stays reserved and
is unavailable to the rest of the application. RCCL provides an API to release
those resources while a communicator is idle and to reacquire them later
without destroying and recreating the communicator.

The relevant functions, declared in ``rccl.h``, are described in full in
:ref:`communicator-suspend-resume`:

- ``ncclCommSuspend`` releases the resources selected by its ``flags``
  argument. Pass ``NCCL_SUSPEND_MEM`` to release dynamic GPU memory
  allocations. After this call, the communicator can't be used until it's
  resumed: a collective issued while it's suspended is rejected with
  ``ncclInvalidUsage``.
- ``ncclCommResume`` reacquires every resource that the matching
  ``ncclCommSuspend`` call released, after which the communicator can run
  collectives again.
- ``ncclCommMemStats`` reports per-communicator memory counters, such as the
  amount of GPU memory that can be suspended and whether the communicator is
  currently suspended.

Requirements
------------

Releasing the physical backing of a suspended communicator while keeping its
GPU virtual address space requires cuMem virtual memory management (VMM)
support. VMM is available only when all of the following conditions are met:

- ``NCCL_CUMEM_ENABLE`` is set to ``1``, or left at its default of ``-2``, which
  enables VMM automatically when the platform supports it. Auto-detection is
  limited to gfx1250; on other architectures set ``1`` explicitly.
- The HIP/ROCm runtime provides the cuMem VMM APIs: ROCm 7.12 or later, or a
  ROCm 7.0.x build that includes the cuMem backport.
- The Linux kernel is version 6.8 or later.
- The GPU and driver report VMM support.

Without VMM support, ``ncclCommSuspend`` and ``ncclCommResume`` still succeed,
but they can't release the physical GPU memory, so the operation is
effectively a no-op.

Example
-------

The following example suspends an idle communicator, queries how much GPU
memory was freed, and later resumes it:

.. code-block:: cpp

   // comm is an initialized ncclComm_t that is currently idle.
   uint64_t suspendable = 0, suspended = 0;

   NCCLCHECK(ncclCommMemStats(comm, ncclStatGpuMemSuspend, &suspendable));
   // suspendable is bytes of GPU memory Suspend can release; 0 means none right
   // now. This query is informational; Suspend does not require suspendable > 0

   // Release dynamic GPU memory held by the communicator.
   NCCLCHECK(ncclCommSuspend(comm, NCCL_SUSPEND_MEM));

   NCCLCHECK(ncclCommMemStats(comm, ncclStatGpuMemSuspended, &suspended));
   // suspended == 1 while the communicator is suspended.

   // ... run other work that needs the freed GPU memory ...

   // Reacquire the resources before using the communicator again.
   NCCLCHECK(ncclCommResume(comm));

To suspend or resume several communicators together, wrap the calls in
``ncclGroupStart`` and ``ncclGroupEnd`` 
(see :ref:`communicator-suspend-resume`):

.. code-block:: cpp

   NCCLCHECK(ncclGroupStart());
   NCCLCHECK(ncclCommSuspend(commA, NCCL_SUSPEND_MEM));
   NCCLCHECK(ncclCommSuspend(commB, NCCL_SUSPEND_MEM));
   NCCLCHECK(ncclGroupEnd());

   // ... later, the matching resume ...

   NCCLCHECK(ncclGroupStart());
   NCCLCHECK(ncclCommResume(commA));
   NCCLCHECK(ncclCommResume(commB));
   NCCLCHECK(ncclGroupEnd());

.. note::

   When a single process owns one communicator per device through
   ``ncclCommInitAll``, ``ncclCommSuspend`` and ``ncclCommResume`` need to be
   wrapped around a group API.

