.. meta::
  :description: This topic describes how to partition GPU compute resources with HIP execution contexts (green contexts).
  :keywords: AMD, ROCm, HIP, execution context, green context, resource partition, CU mask, WGP, hipGreenCtxCreate, hipExecutionCtx_t

.. _execution_context_how-to:

*******************************************************************************
Using execution contexts
*******************************************************************************

Execution contexts, also known as green contexts, let you partition a device's
compute resources into isolated subsets and confine work to a specific
partition. Each execution context owns a slice of the device's compute units
(CUs), and streams created from that context can only schedule work on that
slice. This is useful for latency-sensitive services that must reserve capacity,
for running mixed workloads side by side without letting one starve another, and
for reproducible benchmarking on a fixed number of CUs.

An execution context is the HIP equivalent of a CUDA green context. The workflow
mirrors the CUDA driver API: query the device's resources, split them into
groups, generate a descriptor, and create a context from that descriptor.

Resources and partitions
===============================================================================

The unit of partitioning is the streaming multiprocessor resource, represented
by :cpp:enumerator:`hipDevResourceTypeSm`. On AMD hardware, this maps to compute
units. A :cpp:struct:`hipDevResource` describes a set of these units, and every
partitioning workflow starts by obtaining the full device resource with
:cpp:func:`hipDeviceGetDevResource`:

.. code-block:: cpp

   hipDevResource resource;
   hipDeviceGetDevResource(device, &resource, hipDevResourceTypeSm);

The resource returned is intersected with any global CU masks set through the
``HSA_CU_MASK`` and ``ROC_GLOBAL_CU_MASK`` environment variables, so it reflects
the CUs your process can actually use.

Workgroup processor alignment
-------------------------------------------------------------------------------

On AMD GPUs, CUs are grouped into Workgroup Processors (WGPs), and the hardware
schedules cooperating CUs together. This granularity is reported by
:cpp:member:`hipDevSmResource::smCoscheduledAlignment`:

.. list-table::
  :header-rows: 1
  :widths: 40 30 30

  * - Architecture
    - Mode
    - ``smCoscheduledAlignment``
  * - RDNA, CDNA2 and later
    - WGP mode
    - 2 CUs
  * - GFX9, earlier CDNA
    - CU mode
    - 1 CU

This alignment is the minimum partition granularity. On a device where
``smCoscheduledAlignment`` is 2, request CU counts in multiples of two to avoid
wasted units. Read :cpp:member:`hipDevSmResource::smCoscheduledAlignment` before
splitting rather than assuming a fixed value.

.. note::

  CU partitions are not enforced as mutually exclusive. Creating an execution
  context does not, on its own, prevent other work from running on the same CUs.
  True isolation requires you to carve the device into disjoint partitions with
  :cpp:func:`hipDevSmResourceSplitByCount` or :cpp:func:`hipDevSmResourceSplit`
  and confine each workload to its own partition.

Splitting resources
-------------------------------------------------------------------------------

Two functions divide an SM resource into smaller groups:

- :cpp:func:`hipDevSmResourceSplitByCount` divides a resource into as many
  equally sized groups as possible, each holding at least a requested minimum
  number of CUs. Call it with ``nbGroups`` set to 0, or ``result`` set to
  ``nullptr``, to run in query mode and learn how many groups are achievable before
  allocating the output array.
- :cpp:func:`hipDevSmResourceSplit` divides a resource into a fixed number of
  groups using a per-group parameter array, giving finer control over each
  group's size and coscheduling preferences.

Both functions can write any leftover CUs to an optional ``remainder`` resource,
which you can split further or leave unused.

Creating and using an execution context
===============================================================================

Once you have a resource group, combine it into a descriptor with
:cpp:func:`hipDevResourceGenerateDesc`, then create the context with
:cpp:func:`hipGreenCtxCreate`. A descriptor may contain at most one SM resource,
and all resources in it must belong to the same device.

.. code-block:: cpp

   hipDevResourceDesc_t desc;
   hipDevResourceGenerateDesc(&desc, &groups[0], 1);

   hipExecutionCtx_t ctx;
   hipGreenCtxCreate(&ctx, desc, device, hipGreenCtxDefaultStream);

The ``hipGreenCtxDefaultStream`` flag is required. To schedule work on the
context's partition, create a stream bound to it with
:cpp:func:`hipExecutionCtxStreamCreate` and launch kernels on that stream as
usual:

.. code-block:: cpp

   hipStream_t stream;
   hipExecutionCtxStreamCreate(&stream, ctx, hipStreamNonBlocking, 0);

   myKernel<<<blocks, threads, 0, stream>>>(args);

Work submitted to ``stream`` runs only on the CUs owned by ``ctx``. You can
query the partition backing any stream with :cpp:func:`hipStreamGetDevResource`,
which returns the execution context's CU partition for context streams, the
explicit mask for streams created with :cpp:func:`hipExtStreamCreateWithCUMask`,
and the full device resource for ordinary streams.

Synchronization
-------------------------------------------------------------------------------

Execution contexts provide their own synchronization primitives that act across
all streams in a context:

- :cpp:func:`hipExecutionCtxSynchronize` blocks the host until all work in the
  context's streams completes.
- :cpp:func:`hipExecutionCtxRecordEvent` records an event that signals when all
  work submitted to the context's streams at call time has finished. Streams
  added after the call are not included.
- :cpp:func:`hipExecutionCtxWaitEvent` makes all future work in the context,
  including streams created while the event is pending, wait on an event.

When you are finished, release the context with
:cpp:func:`hipExecutionCtxDestroy`. Streams created from a destroyed context
become orphaned and return :cpp:enumerator:`hipErrorContextIsDestroyed` on
subsequent operations other than :cpp:func:`hipStreamDestroy`, so destroy the
streams first.

Example
-------------------------------------------------------------------------------

The following example partitions a device into two groups and runs independent
work on each partition concurrently.

.. code-block:: cpp

   int device = 0;
   hipDevResource full;
   hipDeviceGetDevResource(device, &full, hipDevResourceTypeSm);

   unsigned int nbGroups = 2;
   hipDevResource groups[2];
   hipDevSmResourceSplitByCount(groups, &nbGroups, &full, nullptr, 0,
                                full.sm.smCoscheduledAlignment);

   hipStream_t streams[2];
   hipExecutionCtx_t ctxs[2];
   for (unsigned int i = 0; i < nbGroups; ++i) {
       hipDevResourceDesc_t desc;
       hipDevResourceGenerateDesc(&desc, &groups[i], 1);
       hipGreenCtxCreate(&ctxs[i], desc, device, hipGreenCtxDefaultStream);
       hipExecutionCtxStreamCreate(&streams[i], ctxs[i], hipStreamNonBlocking, 0);
   }

   // Each kernel runs on its own CU partition.
   kernelA<<<blocks, threads, 0, streams[0]>>>(argsA);
   kernelB<<<blocks, threads, 0, streams[1]>>>(argsB);

   for (unsigned int i = 0; i < nbGroups; ++i) {
       hipExecutionCtxSynchronize(ctxs[i]);
       hipStreamDestroy(streams[i]);
       hipExecutionCtxDestroy(ctxs[i]);
   }

Migrating from CUDA green contexts
===============================================================================

HIP execution contexts follow the CUDA green context model, so ports are largely
mechanical. The handle type differs: HIP uses :cpp:type:`hipExecutionCtx_t`
where CUDA uses ``CUgreenCtx``. The main API correspondences are:

.. list-table::
  :header-rows: 1
  :widths: 50 50

  * - CUDA
    - HIP
  * - ``cuDeviceGetDevResource``
    - :cpp:func:`hipDeviceGetDevResource`
  * - ``cuDevSmResourceSplitByCount``
    - :cpp:func:`hipDevSmResourceSplitByCount`
  * - ``cuDevResourceGenerateDesc``
    - :cpp:func:`hipDevResourceGenerateDesc`
  * - ``cuGreenCtxCreate``
    - :cpp:func:`hipGreenCtxCreate`
  * - ``cuGreenCtxDestroy``
    - :cpp:func:`hipExecutionCtxDestroy`
  * - ``cuGreenCtxStreamCreate``
    - :cpp:func:`hipExecutionCtxStreamCreate`
  * - ``cuGreenCtxRecordEvent``
    - :cpp:func:`hipExecutionCtxRecordEvent`
  * - ``cuGreenCtxWaitEvent``
    - :cpp:func:`hipExecutionCtxWaitEvent`
  * - ``cuStreamGetGreenCtx``
    - :cpp:func:`hipStreamGetDevResource`

The most important behavioral difference is alignment. CUDA aligns partitions to
SM granularity, while HIP aligns to the WGP granularity reported by
:cpp:member:`hipDevSmResource::smCoscheduledAlignment`, which is 2 CUs on RDNA
and CDNA2+ devices. Query this value and size partitions accordingly instead of
porting fixed SM counts directly.

For the complete list of types and functions, see
:ref:`execution_context_management_reference`.
