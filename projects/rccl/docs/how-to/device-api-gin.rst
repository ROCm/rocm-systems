.. meta::
   :description: Configure and use the RCCL device API with GPU-initiated networking
   :keywords: RCCL, ROCm, device API, GIN, GPU-initiated networking

.. _device-api-gin:

Use the RCCL device API and GIN
********************************

The experimental RCCL device API lets GPU kernels communicate through a
device communicator (``ncclDevComm``). GPU-initiated networking (GIN) extends
that API with one-sided puts, signals, counters, and barriers across nodes.
RCCL 2.31 incorporates Device API and GIN enhancements from upstream
NCCL 2.31.2. This page describes those APIs and the limits of the AMD
host-proxy backend.

``NCCL_GIN_TYPE`` values for AMD backends are **not compatible with 2.30.7**:
rocSHMEM GDA moved from 5 to 6 and Anvil SDMA from 6 to 7 because NCCL 2.31
placed EFA GDA at 5. The IB proxy remains ``NCCL_GIN_TYPE=2``.

Requirements
============

GIN requires a supported network backend, symmetric memory, and collective
communicator and window setup. The validated AMD path is ``GIN_IB_PROXY``:

.. code-block:: shell

   export NCCL_GIN_TYPE=2
   export NCCL_CUMEM_ENABLE=1
   export NCCL_DMABUF_ENABLE=1
   export NCCL_IB_MERGE_NICS=0

The ROCm runtime and NIC driver must support exporting and registering VMM
allocations. ``NCCL_DMABUF_ENABLE=1`` is the validated registration path;
peer-memory registration can also be available on supported systems. Full GIN
connections require cross-rail reachability, so do not force
``NCCL_CROSS_NIC=0`` for world-team operations. Leave ``ginQueueDepth`` at its
default value of zero; the AMD host-proxy backend does not support a custom QP
depth.

The following snippets abbreviate error handling with ``NCCLCHECK``. Replace it
with the application's normal ``ncclResult_t`` error handling.

Before using GIN, query communicator properties and confirm that the selected
backend and Device API are available:

.. code-block:: cpp

   ncclCommProperties properties = NCCL_COMM_PROPERTIES_INITIALIZER;
   NCCLCHECK(ncclCommQueryProperties(comm, &properties));
   if (!properties.deviceApiSupport ||
       properties.ginType != NCCL_GIN_TYPE_PROXY) {
     // Select a non-GIN path or report that this configuration is unsupported.
   }

Allocate and register every communication buffer collectively:

.. code-block:: cpp

   void* buffer = nullptr;
   ncclWindow_t window = nullptr;
   NCCLCHECK(ncclMemAlloc(&buffer, bytes));
   NCCLCHECK(ncclCommWindowRegister(
       comm, buffer, bytes, &window, NCCL_WIN_COLL_SYMMETRIC));

Create a device communicator
============================

Always initialize requirements with
``NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER``. The initializer records the header
version used to compile the application.

.. code-block:: cpp

   ncclDevCommRequirements reqs = NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER;
   reqs.ginConnectionType = NCCL_GIN_CONNECTION_FULL;
   reqs.ginContextCount = 4;       // Hint; inspect devComm.ginContextCount.
   reqs.ginSignalCount = 1;
   reqs.ginCounterCount = 1;
   reqs.barrierCount = 1;          // Generic ncclBarrierSession.
   reqs.worldGinBarrierCount = 1;

   ncclDevComm devComm{};
   NCCLCHECK(ncclDevCommCreate(comm, &reqs, &devComm));

GIN contexts and their signal, counter, and queue resources are allocated per
device communicator. Two device communicators backed by the same host
communicator therefore don't alias those resources. ``ginContextCount`` is a
request; use the value returned in ``devComm.ginContextCount`` when assigning
work to contexts.

``ginTrafficClass`` overrides the host communicator traffic class for this
device communicator. ``NCCL_IB_SL`` independently overrides the InfiniBand
service level, and ``NCCL_IB_TC`` independently overrides the RoCE traffic
class. Set ``reqs.ginTrafficClass`` only when the fabric administrator provides
an appropriate value.

Device code creates an ``ncclGin`` object for one returned context:

.. code-block:: cpp

   ncclGin gin{devComm, contextIndex, NCCL_GIN_RESOURCE_SHARING_GPU};

For example, a CTA can issue a put and wait for local queue completion:

.. code-block:: cpp

   __global__ void putKernel(
       ncclDevComm devComm, ncclWindow_t source, ncclWindow_t destination,
       size_t bytes, int peer) {
     ncclGin gin{devComm, /*contextIndex=*/0};
     if (threadIdx.x == 0) {
       gin.put(ncclTeamWorld(devComm), peer,
               destination, /*destinationOffset=*/0,
               source, /*sourceOffset=*/0, bytes,
               ncclGin_SignalInc{/*signal=*/0});
     }
     gin.flush(ncclCoopCta());
   }

``flush`` makes the local source buffer reusable. It does not by itself prove
remote visibility; the peer must wait for the associated signal before using
the destination bytes.

``NCCL_GIN_RESOURCE_SHARING_GPU`` permits sharing across the GPU;
``NCCL_GIN_RESOURCE_SHARING_CTA`` limits sharing to a CTA. These modes select
resource-sharing behavior on direct device backends. The AMD GIN host-proxy
backend uses the same proxy queue behavior for both modes.

Use world-team barriers and timeouts
====================================

Reserve ``worldGinBarrierCount`` slots to construct a world-team GIN barrier
without manually allocating a barrier handle:

.. code-block:: cpp

   ncclGinBarrierSession<ncclCoopCta> barrier{
       ncclCoopCta(), gin, ncclTeamTagWorld{}, barrierIndex};

   ncclResult_t result = barrier.sync(
       ncclCoopCta(), cuda::memory_order_acq_rel,
       ncclGinFenceLevel::Relaxed, timeoutCycles);

The timeout overload returns ``ncclTimeout`` if all team members don't arrive
within ``timeoutCycles``. Barrier resources for ``ncclGinBarrierSession``,
``ncclLsaBarrierSession``, and ``ncclBarrierSession`` are separate. Reserve
``barrierCount`` for generic ``ncclBarrierSession`` objects; use
``lsaBarrierCount``, ``railGinBarrierCount``, or ``worldGinBarrierCount`` for
the corresponding specialized session. ``barrierIndex`` must be smaller than
the selected count. Concurrent CTAs must use distinct indices, typically
``blockIdx.x``.

Destroy resources
=================

Destroy the device communicator before deregistering its application windows,
then free the backing allocations. These calls are collective where their
corresponding create or registration operation is collective.

.. code-block:: cpp

   NCCLCHECK(ncclDevCommDestroy(comm, &devComm));
   NCCLCHECK(ncclCommWindowDeregister(comm, window));
   NCCLCHECK(ncclMemFree(buffer));

Version and backend notes
=========================

``ncclDevComm`` is versioned. The upstream NCCL 2.30.3 and 2.30.4 release notes
require applications using GIN APIs to be rebuilt with the matching release.
RCCL accepts compatible layouts within the 2.30 family, but applications using
pre-2.30 GIN device code must be rebuilt with compatible RCCL headers. The
runtime rejects pre-2.30 requirements that request indexed GIN resources.

The 128-byte, versioned GIN proxy descriptor and per-context proxy progress are
internal implementation details and require no application configuration.
The ``max_rd_atomic`` and ``max_dest_rd_atomic`` changes and the
``doca-gpunetio`` update in the NCCL 2.30 release apply to NVIDIA's GDAKI
backend, not the AMD GIN host-proxy backend.

The source includes experimental proxy ``get`` and nonblocking-flush paths, but
they are not yet validated for production use on the AMD host-proxy backend.
GIN access to elastic and multi-segment windows is also unavailable on that
backend until a complete multi-segment VMM range can be exported for DMA-BUF
registration. Use the validated put/signal path with single-segment symmetric
windows.
