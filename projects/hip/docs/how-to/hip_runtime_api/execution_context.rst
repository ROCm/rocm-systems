.. meta::
    :description: How to use HIP execution contexts to partition GPU compute resources
    :keywords: AMD, ROCm, HIP, execution context, compute unit, CU, resource partitioning, work queue

.. _execution_context:

*******************************************************************************
Execution contexts
*******************************************************************************

By default, every kernel you launch competes for the whole GPU. The runtime
decides which compute units (CUs) a kernel runs on, and kernels that run at the
same time share the pool of CUs and work queues. This is efficient for
throughput, but it means a small, time-critical kernel can be held up behind a
large kernel that already occupies the device.

An execution context lets you carve out a fixed slice of the GPU and bind work
to it. When you create an execution context, you attach it to a chosen set of
device resources, currently CUs and work queues. Any kernel launched on a stream
that belongs to that context is confined to those resources, no matter how the
kernel is configured. You set this up entirely on the host; the kernel source
does not change.

Reserving resources this way is useful whenever you want predictable access to
part of the GPU. For example, you can hold back a handful of CUs so a latency
sensitive kernel always has somewhere to start, or you can cap a kernel to a
smaller CU count to measure how it scales, without touching its code.

In the HIP runtime, an execution context is either the device's primary context,
which the runtime uses implicitly, or a resource-partitioned context you create
with :cpp:func:`hipGreenCtxCreate`. This feature corresponds to CUDA green
contexts.

.. note::

    The device resource structures use the field name ``smCount`` and the
    resource type ``hipDevResourceTypeSm``. HIP keeps these ``sm`` (streaming
    multiprocessor) identifiers so that code written against CUDA compiles
    unchanged. On AMD GPUs the equivalent physical resource is the compute unit.
    This page writes "compute unit" or "CU" in the text and keeps the literal
    identifiers in code.

.. note::

    The snippets on this page are written to show the calling sequence. Build and
    run them on a ROCm system before depending on them. A complete, buildable
    program is provided as the
    `HIP-Basic execution context example <https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/execution_context>`_.

The full API listing is in :ref:`execution_context_management_reference`.

When execution contexts help
===============================================================================

Two properties of normal kernel scheduling motivate execution contexts.

First, you cannot ask for a specific number of CUs. The number a kernel ends up
using follows indirectly from its grid and block dimensions and its per-CU
occupancy. There is no launch parameter that says "run on N CUs."

Second, kernels that overlap in time draw from the same shared CUs. If one kernel
is already spread across the device when a second, more urgent kernel arrives,
the urgent kernel has to wait for CUs to free up as the first kernel's thread
blocks retire.

Picture a service that keeps a background kernel running continuously while
occasionally needing to run a short, urgent kernel with minimal delay. If the
background kernel is allowed to fill the GPU, the urgent kernel stalls until
enough thread blocks of the background kernel finish. Raising the urgent kernel's
stream priority helps, but it still waits for in-flight thread blocks to drain.

Execution contexts remove the contention at its source. Put the background kernel
on a context that owns most of the CUs, and the urgent kernel on a context that
owns a small, separate set. The background kernel can never spill onto the urgent
kernel's CUs, so the urgent kernel finds free resources the moment it launches.
The trade-off is that neither kernel can use the entire GPU any longer, so each
may take a little longer in isolation, but the urgent work stops being blocked.

The split is your decision and depends on the workload. Choose the CU counts for
each context when you create it, and tune them by measurement.

Work queues
-------------------------------------------------------------------------------

CUs are one kind of resource an execution context can own. Work queues are
another. A work queue is an abstraction the driver uses to dispatch GPU work;
tasks that land on the same work queue can end up serialized even when they are
logically independent, because sharing a queue introduces an ordering dependency
between them.

That matters because reserving CUs alone does not always guarantee overlap. If
the urgent kernel and the background kernel are dispatched through the same work
queue, the urgent kernel can still wait behind the background kernel even though
it has its own CUs. You do not choose work queues directly, but an execution
context lets you state how many concurrent stream-ordered workloads you expect.
The driver treats that number as a hint and tries to keep work from different
contexts on separate queues. The device-wide upper bound on work queues is
controlled by the ``GPU_MAX_HW_QUEUES`` environment variable.

.. note::

    Partitioning CUs and work queues reduces the causes of interference between
    contexts, but it does not force independent kernels to run at the same time.
    Concurrency still depends on the workloads and the device state.

Relationship to hardware partitioning
-------------------------------------------------------------------------------

AMD Instinct accelerators can be split into logical devices through hardware
compute partitioning modes such as SPX and CPX. That split happens at the system
level, before an application starts, and is typically used to divide a GPU among
separate applications.

Execution contexts work at a finer grain and inside a single process. Hardware
partitioning decides how the GPU is shared between applications; execution
contexts decide how one application's streams share the CUs it has been given. On
a GPU already running in a partitioned mode, an execution context draws from the
CUs of whichever partition the application is using. HIP has no equivalent of a
multi-process service; execution contexts are a within-process mechanism.

Device resources and descriptors
===============================================================================

An execution context is built from device resources. A device resource
(``hipDevResource``) names a slice of a specific GPU, and a resource descriptor
(``hipDevResourceDesc_t``) bundles one or more resources together. The execution
context you create from a descriptor can use exactly the resources that
descriptor holds, and nothing else.

The ``hipDevResource`` structure carries a single resource, tagged by type:

.. code-block:: cpp

    typedef struct hipDevResource_st {
        hipDevResourceType type;
        // internal padding
        union {
            hipDevSmResource              sm;
            hipDevWorkqueueConfigResource wqConfig;
            hipDevWorkqueueResource       wq;
        };
        struct hipDevResource_st* nextResource;
    } hipDevResource;

Three resource types are defined:

- ``hipDevResourceTypeSm`` for a set of compute units.
- ``hipDevResourceTypeWorkqueueConfig`` for a work queue configuration.
- ``hipDevResourceTypeWorkqueue`` for an existing work queue resource.

``hipDevResourceTypeInvalid`` marks an unset resource.

Query a device with ``hipDeviceGetDevResource`` and it reports all three:
a CU resource covering every CU on the GPU, a work queue configuration covering
all its work queues, and the matching work queue resource. You can also ask an
execution context or a stream what resources it holds, using
``hipExecutionCtxGetDevResource`` and ``hipStreamGetDevResource``. An execution
context can hold several resource types at once; a stream only ever carries a
CU resource.

Compute unit resource
-------------------------------------------------------------------------------

The CU resource (``hipDevSmResource``) describes a group of compute units:

.. code-block:: cpp

    typedef struct hipDevSmResource {
        unsigned int smCount;                // number of CUs in this resource
        unsigned int minSmPartitionSize;     // smallest CU count this resource can be split into
        unsigned int smCoscheduledAlignment; // CUs guaranteed to be co-scheduled together
        unsigned int flags;                  // 0 (default) or hipDevSmResourceGroupBackfill
    } hipDevSmResource;

You never fill these fields in yourself. ``hipDeviceGetDevResource`` sets them
when you query a device, and the split APIs set them on the resources they
produce. Treat ``minSmPartitionSize`` and ``smCoscheduledAlignment`` as
architecture-dependent values to read at runtime, not constants to hard-code.

The resource returned by ``hipDeviceGetDevResource`` is intersected with any
global CU mask set through the ``ROC_GLOBAL_CU_MASK`` environment variable, so it
reflects the CUs your process can actually use.

Workgroup processor alignment
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

On AMD GPUs, CUs are grouped into workgroup processors (WGPs), and cooperating
CUs are scheduled together. ``smCoscheduledAlignment`` reports this granularity,
which is also the minimum partition granularity:

.. list-table::
    :header-rows: 1

    * - Mode
      - ``smCoscheduledAlignment``
    * - WGP mode (typical on RDNA and recent CDNA)
      - 2 CUs
    * - CU mode
      - 1 CU

When the alignment is 2, request CU counts in multiples of two to avoid wasting
units. Read ``smCoscheduledAlignment`` at runtime rather than assuming a value,
since it depends on the device and its mode.

.. note::

    Creating an execution context does not, by itself, stop other work from
    running on the same CUs; the partitions are not hardware-enforced as mutually
    exclusive. To isolate workloads, split the device into disjoint partitions
    and confine each workload to its own context.

Work queue configuration resource
-------------------------------------------------------------------------------

The work queue configuration resource (``hipDevWorkqueueConfigResource``) is one
you populate directly:

.. code-block:: cpp

    typedef struct hipDevWorkqueueConfigResource {
        int                        device;             // device that owns the work queues
        unsigned int               wqConcurrencyLimit; // expected concurrent stream-ordered workloads
        hipDevWorkqueueConfigScope sharingScope;       // how work queues are shared
    } hipDevWorkqueueConfigResource;

``sharingScope`` takes one of two values. ``hipDevWorkqueueConfigScopeDeviceCtx``,
the default, shares work queues across all contexts.
``hipDevWorkqueueConfigScopeGreenCtxBalanced`` asks the driver to keep work queues
from different execution contexts apart where it can, guided by
``wqConcurrencyLimit``.

There is no split API for work queue resources: set the fields yourself, or read
a device's configuration with ``hipDeviceGetDevResource``. The plain work queue
resource (``hipDevResourceTypeWorkqueue``) exposes no fields you can set.

.. tip::

    Zero-initialize every device resource structure before you use it.

Creating an execution context
===============================================================================

Building an execution context takes four steps:

#. Read the resources you want to start from, usually the device's full set.
#. Split the CU resource into the partitions you need.
#. Bundle the resulting resources into a descriptor.
#. Create the execution context from that descriptor.

Once the context exists, create a stream on it. Work you launch on that
stream, including a kernel launched with the triple-chevron syntax, is limited to
the context's resources.

Step 1: Read the available resources
-------------------------------------------------------------------------------

Start by populating a ``hipDevResource`` from a device, an execution context, or
a stream:

.. code-block:: cpp

    hipError_t hipDeviceGetDevResource(hipDevice_t device, hipDevResource* resource,
                                       hipDevResourceType type);
    hipError_t hipExecutionCtxGetDevResource(hipExecutionCtx_t ctx, hipDevResource* resource,
                                             hipDevResourceType type);
    hipError_t hipStreamGetDevResource(hipStream_t hStream, hipDevResource* resource,
                                       hipDevResourceType type);

Each accepts any resource type, except ``hipStreamGetDevResource``, which is
limited to CU resources.

Reading a device's CUs looks like this:

.. code-block:: cpp

    int current_device = 0;
    HIP_CHECK(hipSetDevice(current_device));

    hipDevResource cu_resources = {};
    HIP_CHECK(hipDeviceGetDevResource(current_device, &cu_resources, hipDevResourceTypeSm));

    std::cout << "Available CUs: " << cu_resources.sm.smCount << "\n";
    std::cout << "Min. partition size: " << cu_resources.sm.minSmPartitionSize << "\n";
    std::cout << "Co-scheduled alignment: " << cu_resources.sm.smCoscheduledAlignment << "\n";

Reading the work queue configuration is similar:

.. code-block:: cpp

    hipDevResource wq_config = {};
    HIP_CHECK(hipDeviceGetDevResource(current_device, &wq_config, hipDevResourceTypeWorkqueueConfig));

    std::cout << "WQ concurrency limit: " << wq_config.wqConfig.wqConcurrencyLimit << "\n";
    std::cout << "WQ sharing scope: " << wq_config.wqConfig.sharingScope << "\n";

For a device, ``wqConcurrencyLimit`` reflects ``GPU_MAX_HW_QUEUES`` or its
default.

Step 2: Split the CU resource
-------------------------------------------------------------------------------

Divide the CU resource with one of two APIs. ``hipDevSmResourceSplitByCount``
produces equal-sized partitions; ``hipDevSmResourceSplit`` produces partitions of
different sizes in a single call. Either way, CUs that do not fit the requested
partitions land in an optional remainder. Both APIs only operate on CU resources.

Equal-sized partitions
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

.. code-block:: cpp

    hipError_t hipDevSmResourceSplitByCount(hipDevResource* result, unsigned int* nbGroups,
                                            const hipDevResource* input, hipDevResource* remainder,
                                            unsigned int flags, unsigned int minCount);

You pass in the number of groups you want (``*nbGroups``) and the minimum CUs per
group (``minCount``). The call may return fewer groups than you asked for, each
with at least ``minCount`` CUs, because the hardware imposes granularity and
alignment rules. The exact rounding depends on the device's
``minSmPartitionSize`` and ``smCoscheduledAlignment``, so read those at runtime.

.. list-table:: Why the result can differ from the request
    :header-rows: 1

    * - Situation
      - Outcome
    * - You request more groups than fit at ``minCount``
      - The count is reduced to what fits; leftover CUs go to the remainder.
    * - ``minCount`` is not a multiple of the alignment
      - Each group is rounded up to a valid size; fewer CUs remain.

A request for five groups:

.. code-block:: cpp

    hipDevResource avail = {};
    // Populate avail with hipDeviceGetDevResource.

    unsigned int min_cu_count = 8;
    unsigned int group_count  = 5; // may be lowered by the call

    hipDevResource result[5] = {};
    hipDevResource remaining = {};

    HIP_CHECK(hipDevSmResourceSplitByCount(&result[0], &group_count, &avail,
                                           &remaining, 0 /* flags */, min_cu_count));

    std::cout << "Got " << group_count << " groups of " << result[0].sm.smCount
              << " CUs, " << remaining.sm.smCount << " CUs left over\n";

Points to keep in mind:

- Pass ``result = nullptr`` to find out how many groups you would get, without
  producing them.
- Pass ``remainder = nullptr`` to discard the leftover CUs.
- The remainder does not carry the same guarantees as the equal-sized groups.
- ``flags`` is ``0`` by default. ``hipDevSmResourceSplitIgnoreSmCoscheduling`` and
  ``hipDevSmResourceSplitMaxPotentialClusterSize`` are also defined.
- To repartition a resulting resource, first turn it into a descriptor and an
  execution context (steps 3 and 4).

.. note::

    ``hipDevSmResourceSplitIgnoreSmCoscheduling`` is defined but not yet supported
    by the runtime. Passing it returns ``hipErrorNotSupported``.

Different-sized partitions
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''''

When contexts need different CU counts, one ``hipDevSmResourceSplitByCount`` call
is not enough, since it only makes equal groups. ``hipDevSmResourceSplit`` builds
groups of independent sizes at once:

.. code-block:: cpp

    hipError_t hipDevSmResourceSplit(hipDevResource* result, unsigned int nbGroups,
                                     const hipDevResource* input, hipDevResource* remainder,
                                     unsigned int flags, hipDevSmResourceGroupParams* groupParams);

Each of the ``nbGroups`` output resources is shaped by a matching
``groupParams`` entry. A remainder is optional. Every produced group has at least
some CUs; a group is never empty.

.. code-block:: cpp

    typedef struct hipDevSmResourceGroupParams_st {
        unsigned int smCount;                     // CU count, or 0 for discovery mode
        unsigned int coscheduledSmCount;          // co-scheduled CU count for clusters
        unsigned int preferredCoscheduledSmCount; // preferred co-scheduled CU count (hint)
        unsigned int flags;                       // 0 or hipDevSmResourceGroupBackfill
    } hipDevSmResourceGroupParams;

Give each group an ``smCount`` that is a multiple of two. If your kernels use
thread block clusters, set ``coscheduledSmCount`` to the largest cluster the
group must support, since a cluster's thread blocks are always co-scheduled.
``preferredCoscheduledSmCount`` is a hint to fold groups into larger ones when
possible, and setting ``flags`` to ``hipDevSmResourceGroupBackfill`` lets a group
absorb extra CUs beyond its requested size.

To let the runtime pick a size, use discovery mode: set an entry's ``smCount`` to
zero, and the call fills in a valid count. Entries are processed in order, from
index 0 to ``nbGroups - 1``, so earlier entries claim CUs first.

.. list-table:: hipDevSmResourceSplit arguments
    :header-rows: 1

    * - Argument
      - Meaning
    * - ``result``
      - ``nullptr`` for a dry run, or a valid pointer to receive the groups.
    * - ``nbGroups``
      - How many groups to create.
    * - ``input``
      - The CU resource being split.
    * - ``remainder``
      - ``nullptr`` to drop leftover CUs.
    * - ``flags``
      - ``0``.
    * - ``groupParams[i].smCount``
      - ``0`` for discovery, or a specific CU count.
    * - ``groupParams[i].coscheduledSmCount``
      - ``0`` for the default, or a co-scheduled CU count.
    * - ``groupParams[i].preferredCoscheduledSmCount``
      - ``0`` for the default, or a preferred co-scheduled CU count.
    * - ``groupParams[i].flags``
      - ``0`` or ``hipDevSmResourceGroupBackfill``.

What the return value means depends on ``result``:

- With a valid ``result``, the call succeeds only if every requested group was
  created; otherwise it returns an error.
- With ``result = nullptr``, the call can report success even for a
  configuration that would fail with a real output. Use this to probe what the
  device allows.

When the call succeeds with a valid ``result``, each ``result[i].sm.smCount`` is
an even number in the range ``[2, input.sm.smCount]``.

The table below sketches ``groupParams`` for a few common goals, with CU counts
left as placeholders you fill in for your device.

.. list-table:: Example splits
    :header-rows: 1

    * - Goal
      - ``nbGroups``
      - ``remainder``
      - ``smCount``
      - ``coscheduledSmCount``
      - ``flags``
    * - One group of X CUs, discard the rest. Clusters allowed.
      - 1
      - ``nullptr``
      - X
      - 0
      - 0
    * - One group of X CUs, the rest kept as remainder. No clusters.
      - 1
      - not ``nullptr``
      - X
      - 2
      - 0
    * - Two groups of X and Y CUs with clusters of a chosen size.
      - 2
      - ``nullptr``
      - X, then Y
      - chosen size
      - 0
    * - As many CUs as possible in one group, plus a remainder.
      - 1
      - not ``nullptr``
      - 0 (discovery)
      - chosen size
      - 0

A two-way uneven split:

.. code-block:: cpp

    hipDevResource cu_resources = {};
    HIP_CHECK(hipDeviceGetDevResource(0, &cu_resources, hipDevResourceTypeSm));

    hipDevResource result[2] = {};
    hipDevSmResourceGroupParams group_params[2] = {
        {/*smCount=*/16, /*coscheduledSmCount=*/0, /*preferredCoscheduledSmCount=*/0, /*flags=*/0},
        {/*smCount=*/8,  /*coscheduledSmCount=*/0, /*preferredCoscheduledSmCount=*/0, /*flags=*/0}};

    HIP_CHECK(hipDevSmResourceSplit(&result[0], 2, &cu_resources,
                                    nullptr /* remainder */, 0 /* flags */, &group_params[0]));

Leaving ``coscheduledSmCount`` or ``preferredCoscheduledSmCount`` at zero requests
the architecture default, which matches the device's ``smCoscheduledAlignment``.
To see the value that was chosen, read the ``groupParams`` entry back after a
successful call.

Adding a work queue resource
-------------------------------------------------------------------------------

To reserve work queues alongside CUs, fill in a work queue configuration resource
yourself and place it next to the CU resources you plan to bundle:

.. code-block:: cpp

    hipDevResource resources[2] = {};
    // Populate resources[0] with a split API (one group).

    resources[1].type                       = hipDevResourceTypeWorkqueueConfig;
    resources[1].wqConfig.device            = 0;
    resources[1].wqConfig.sharingScope      = hipDevWorkqueueConfigScopeGreenCtxBalanced;
    resources[1].wqConfig.wqConcurrencyLimit = 4;

A concurrency limit of four tells the driver you expect up to four concurrent
stream-ordered workloads, and it assigns work queues to respect that where it
can.

Step 3: Build a descriptor
-------------------------------------------------------------------------------

Gather the resources for the context into a descriptor with
``hipDevResourceGenerateDesc``:

.. code-block:: cpp

    hipError_t hipDevResourceGenerateDesc(hipDevResourceDesc_t* phDesc,
                                          hipDevResource* resources, unsigned int nbResources);

The resources you bundle must sit next to each other in the array:

.. code-block:: cpp

    hipDevResource result[5] = {};
    // Populate result with a split API.

    hipDevResourceDesc_t desc = {};
    HIP_CHECK(hipDevResourceGenerateDesc(&desc, &result[2], 3)); // bundles result[2], [3], [4]

The call requires that:

- Every resource belongs to the same device.
- CU resources combined together come from the same split call and share the same
  ``coscheduledSmCount``, unless they are remainders.
- At most one work queue configuration or work queue resource is present.

Step 4: Create the context
-------------------------------------------------------------------------------

Turn the descriptor into an execution context with ``hipGreenCtxCreate``. The
context can use only the resources the descriptor holds:

.. code-block:: cpp

    hipError_t hipGreenCtxCreate(hipExecutionCtx_t* ctx, hipDevResourceDesc_t desc,
                                 int device, unsigned int flags);

Pass ``0`` for ``flags``. Initialize the device's primary context first, with
``hipInitDevice`` or ``hipSetDevice``, so that primary context setup does not add
overhead to this call:

.. code-block:: cpp

    int current_device = 0;
    HIP_CHECK(hipSetDevice(current_device));

    hipDevResourceDesc_t desc = {};
    // Generate desc with hipDevResourceGenerateDesc.

    hipExecutionCtx_t exec_ctx = {};
    HIP_CHECK(hipGreenCtxCreate(&exec_ctx, desc, current_device, 0));

To confirm what the context received, call ``hipExecutionCtxGetDevResource`` on it
for each resource type.

You can create several contexts by repeating these steps. Usually each context
owns a disjoint set of CUs, but you can also let two contexts share some CUs by
including the same resource in both descriptors. Overlapping CUs like this is
occasionally useful; apply it deliberately.

Running work on a context
===============================================================================

To send a kernel to an execution context, create a stream on the context with
``hipExecutionCtxStreamCreate``. Anything launched on that stream is bound to the
context's resources:

.. code-block:: cpp

    hipError_t hipExecutionCtxStreamCreate(hipStream_t* stream, hipExecutionCtx_t ctx,
                                           unsigned int flags, int priority);

.. code-block:: cpp

    hipStream_t stream;
    int priority = 0;
    HIP_CHECK(hipExecutionCtxStreamCreate(&stream, exec_ctx, hipStreamDefault, priority));

    my_kernel<<<grid_dim, block_dim, 0, stream>>>();
    HIP_CHECK(hipGetLastError());

On an execution context, the default stream flag behaves like
``hipStreamNonBlocking``.

You can query the CU partition backing any stream with
``hipStreamGetDevResource``. It returns the execution context's CU partition for
a context stream, the explicit mask for a stream created with
``hipExtStreamCreateWithCUMask``, and the full device resource for an ordinary
stream. Only ``hipDevResourceTypeSm`` is supported; other types return
``hipErrorInvalidResourceType``.

Graphs
-------------------------------------------------------------------------------

With a :doc:`graph <./hipgraph>`, the stream you launch the graph on does not
decide the resources, unlike a direct launch; that stream only tracks
dependencies. Instead, each node's execution context is fixed when the node is
created. Under stream capture, a node inherits the execution context of the
captured stream. When you build a graph through the graph APIs, set the execution
context on each node explicitly.

Thread block clusters
-------------------------------------------------------------------------------

A kernel that uses thread block clusters runs on an execution context stream like
any other kernel and is bound to the context's CUs. Use the occupancy queries,
``hipOccupancyMaxPotentialClusterSize`` and ``hipOccupancyMaxActiveClusters``, to
size clusters. When you give one of these a launch configuration whose ``stream``
belongs to an execution context, it accounts for that context's CUs.

Other context operations
===============================================================================

To synchronize with events across a whole context, use
``hipExecutionCtxRecordEvent`` and ``hipExecutionCtxWaitEvent``. Recording
captures all of the context's outstanding work in one event; waiting makes later
work on the context depend on that event. When a context has several streams,
this is simpler than recording or waiting on each stream separately.

``hipExecutionCtxSynchronize`` blocks the host until the context finishes its
work. Called on the device's primary context, obtained with
``hipDeviceGetExecutionCtx``, it also waits on every execution context created on
that device.

``hipExecutionCtxGetDevice`` returns the device behind a context, and
``hipExecutionCtxGetId`` returns its unique identifier. Release a context you
created with ``hipExecutionCtxDestroy``.

Destroy a context's streams before the context itself. A stream created from a
destroyed context is orphaned: operations on it other than ``hipStreamDestroy``
return ``hipErrorContextIsDestroyed``.

Migrating from CUDA green contexts
===============================================================================

HIP execution contexts follow the CUDA green context model, so ports are mostly
mechanical. The handle type differs: HIP uses ``hipExecutionCtx_t`` where CUDA
uses ``CUgreenCtx``. The main function correspondences are:

.. list-table::
    :header-rows: 1

    * - CUDA
      - HIP
    * - ``cuDeviceGetDevResource``
      - ``hipDeviceGetDevResource``
    * - ``cuDevSmResourceSplitByCount``
      - ``hipDevSmResourceSplitByCount``
    * - ``cuDevResourceGenerateDesc``
      - ``hipDevResourceGenerateDesc``
    * - ``cuGreenCtxCreate``
      - ``hipGreenCtxCreate``
    * - ``cuGreenCtxDestroy``
      - ``hipExecutionCtxDestroy``
    * - ``cuGreenCtxStreamCreate``
      - ``hipExecutionCtxStreamCreate``
    * - ``cuGreenCtxRecordEvent``
      - ``hipExecutionCtxRecordEvent``
    * - ``cuGreenCtxWaitEvent``
      - ``hipExecutionCtxWaitEvent``
    * - ``cuStreamGetGreenCtx``
      - ``hipStreamGetDevResource``

The most important behavioral difference is alignment. CUDA aligns partitions to
SM granularity, while HIP aligns to the WGP granularity reported by
``smCoscheduledAlignment``, which is 2 CUs in WGP mode. Query this value and size
partitions accordingly instead of porting fixed SM counts directly.

Worked example
===============================================================================

This example reserves CUs for an urgent kernel so it is not blocked by a
long-running one, and measures the difference. A large background kernel runs
concurrently with a small, latency-sensitive critical kernel, and the critical
kernel's runtime is timed in two configurations:

- **Baseline**: both kernels run on ordinary streams and share all of the
  device's CUs, so the critical kernel contends with the background kernel.
- **Partitioned**: the CUs are split into two execution contexts, so the
  critical kernel runs on its own CUs while the background kernel is confined to
  the rest.

Without execution contexts, the critical kernel waits for the background
kernel's thread blocks to retire before CUs open up, even with a higher stream
priority. With execution contexts, each kernel owns a distinct set of CUs, so the
critical kernel starts right away. Both kernels give up access to the full GPU,
so each may run slightly longer alone, but the critical kernel is no longer held
back.

A busy kernel stands in for a compute-bound workload. Oversubscribing the grid,
launching many more thread blocks than the device runs at once, keeps every CU
occupied for the whole measurement:

.. code-block:: cpp

    __global__ void busy_kernel(unsigned int* out, unsigned int iterations)
    {
        volatile unsigned int acc = 0;
        for(unsigned int i = 0; i < iterations; ++i)
        {
            acc += i;
        }
        if(threadIdx.x == 0)
        {
            out[blockIdx.x] = acc;
        }
    }

A helper launches the background kernel, then times the critical kernel with HIP
events while the background kernel is still running. Passing two streams that
belong to disjoint execution contexts confines each kernel to its own CUs;
passing two ordinary streams lets them contend:

.. code-block:: cpp

    float time_critical_with_background(hipStream_t   background_stream,
                                        hipStream_t   critical_stream,
                                        unsigned int* d_out_background,
                                        unsigned int* d_out_critical)
    {
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        // Launch the background kernel; it keeps running on its own stream.
        busy_kernel<<<background_grid, block_size, 0, background_stream>>>(
            d_out_background, background_iterations);
        HIP_CHECK(hipGetLastError());

        // Time the critical kernel while the background kernel runs.
        HIP_CHECK(hipEventRecord(start, critical_stream));
        busy_kernel<<<critical_grid, block_size, 0, critical_stream>>>(
            d_out_critical, critical_iterations);
        HIP_CHECK(hipGetLastError());
        HIP_CHECK(hipEventRecord(stop, critical_stream));
        HIP_CHECK(hipEventSynchronize(stop));

        float critical_ms = 0.0f;
        HIP_CHECK(hipEventElapsedTime(&critical_ms, start, stop));
        HIP_CHECK(hipStreamSynchronize(background_stream));

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
        return critical_ms;
    }

The baseline runs both kernels on ordinary non-blocking streams, so they share
the whole device:

.. code-block:: cpp

    hipStream_t shared_background, shared_critical;
    HIP_CHECK(hipStreamCreateWithFlags(&shared_background, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&shared_critical, hipStreamNonBlocking));

    float baseline_ms = time_critical_with_background(
        shared_background, shared_critical, d_out_background, d_out_critical);

    HIP_CHECK(hipStreamDestroy(shared_background));
    HIP_CHECK(hipStreamDestroy(shared_critical));

The partitioned run splits the CUs into a group for the background kernel and a
disjoint group for the critical kernel, then follows the four-step setup: read
the device's CUs, split them, wrap each group in a descriptor, and create an
execution context per group. A stream on each context confines its kernel to that
context's CUs:

.. code-block:: cpp

    // Step 1: Read the device's CUs.
    hipDevResource cu_resources = {};
    HIP_CHECK(hipDeviceGetDevResource(0, &cu_resources, hipDevResourceTypeSm));
    unsigned int total_cus    = cu_resources.sm.smCount;
    unsigned int critical_cus = total_cus / 4;

    // Step 2: Split into a background group and a smaller critical group.
    hipDevResource result[2] = {};
    hipDevSmResourceGroupParams group_params[2] = {
        {/*smCount=*/total_cus - critical_cus, 0, 0, 0},
        {/*smCount=*/critical_cus,             0, 0, 0}};
    HIP_CHECK(hipDevSmResourceSplit(&result[0], 2, &cu_resources, nullptr, 0, &group_params[0]));

    // Step 3: Wrap each group in a descriptor.
    hipDevResourceDesc_t desc_background = {};
    hipDevResourceDesc_t desc_critical   = {};
    HIP_CHECK(hipDevResourceGenerateDesc(&desc_background, &result[0], 1));
    HIP_CHECK(hipDevResourceGenerateDesc(&desc_critical, &result[1], 1));

    // Step 4: Create an execution context per group.
    hipExecutionCtx_t ctx_background = {};
    hipExecutionCtx_t ctx_critical   = {};
    HIP_CHECK(hipGreenCtxCreate(&ctx_background, desc_background, 0, 0));
    HIP_CHECK(hipGreenCtxCreate(&ctx_critical, desc_critical, 0, 0));

    // A stream on each context, then run the same timed measurement.
    hipStream_t strm_background, strm_critical;
    HIP_CHECK(hipExecutionCtxStreamCreate(&strm_background, ctx_background, hipStreamDefault, 0));
    HIP_CHECK(hipExecutionCtxStreamCreate(&strm_critical, ctx_critical, hipStreamDefault, 0));

    float partitioned_ms = time_critical_with_background(
        strm_background, strm_critical, d_out_background, d_out_critical);

    // Release everything.
    HIP_CHECK(hipStreamDestroy(strm_background));
    HIP_CHECK(hipStreamDestroy(strm_critical));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_background));
    HIP_CHECK(hipExecutionCtxDestroy(ctx_critical));

Comparing ``baseline_ms`` with ``partitioned_ms`` shows the critical kernel
finishing sooner once it has its own CUs. Settle on the CU split by measuring your
own workload. The
`HIP-Basic execution context example <https://github.com/ROCm/rocm-examples/tree/develop/HIP-Basic/execution_context>`_
contains a complete, buildable version that sweeps several partition sizes and
also provides a CUDA green context backend.
