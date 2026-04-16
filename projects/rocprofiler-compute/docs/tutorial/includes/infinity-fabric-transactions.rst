.. _infinity-fabric-example:

Infinity Fabric transactions
============================

 For this example, consider the
 :dev-sample:`Infinity Fabric™ sample <fabric.hip>` distributed as a part of
 ROCm Compute Profiler.

This following code snippet launches a simple read-only kernel.

.. code-block:: cpp

   // the main streaming kernel
   __global__ void kernel(int* x, size_t N, int zero) {
     int sum = 0;
     const size_t offset_start = threadIdx.x + blockIdx.x * blockDim.x;
     for (int i = 0; i < 10; ++i) {
       for (size_t offset = offset_start; offset < N; offset += blockDim.x * gridDim.x) {
         sum += x[offset];
       }
     }
     if (sum != 0) {
       x[offset_start] = sum;
     }
   }

This happens twice -- once as a warm-up and once for analysis. Note that the
buffer ``x`` is initialized to all zeros via a call to ``hipMemcpy`` on the
host before the kernel is ever launched. Therefore, the following conditional
is identically false -- and thus we expect no writes.

.. code-block:: cpp

   if (sum != 0) { ...

.. note::

   The actual sample included with ROCm Compute Profiler also includes the ability to select
   different operation types (such as atomics, writes). This abbreviated version
   is presented here for reference only.

Finally, this sample code lets the user control the
:ref:`granularity of an allocation <memory-type>`, the owner of an allocation
(local HBM, CPU DRAM or remote HBM), and the size of an allocation (the default
is :math:`\sim4`\ GiB) via command line arguments. In doing so, we can explore
the impact of these parameters on the L2-Fabric metrics reported by ROCm Compute Profiler to
further understand their meaning.

.. note::

   All results in this section were generated an a node of Infinity
   Fabric connected MI250 accelerators using ROCm version 5.6.0, and ROCm Compute Profiler
   version 2.0.0. Although results may vary with ROCm versions and accelerator
   connectivity, we expect the lessons learned here to be broadly applicable.

.. _infinity-fabric-ex1:

Experiment 1:  Coarse-grained, accelerator-local HBM reads
-----------------------------------------------------------

In our first experiment, we consider the simplest possible case, a
``hipMalloc``\ ’d buffer that is local to our current accelerator:

.. code-block:: shell-session

   $ rocprof-compute profile -n coarse_grained_local --no-roof -- ./fabric -t 1 -o 0
   Using:
     mtype:CoarseGrained
     mowner:Device
     mspace:Global
     mop:Read
     mdata:Unsigned
     remoteId:-1
   <...>
   $ rocprof-compute analyze -p workloads/coarse_grained_local/mi200 -b 17.2.0 17.2.1 17.2.2 17.4.0 17.4.1 17.4.2 17.5.0 17.5.1 17.5.2 17.5.3 17.5.4 -n per_kernel --dispatch 2
   <...>
   17. L2 Cache
   17.2 L2 - Fabric Transactions
   ╒═════════╤═════════════════════╤════════════════╤════════════════╤════════════════╤══════════════════╕
   │ Index   │ Metric              │            Avg │            Min │            Max │ Unit             │
   ╞═════════╪═════════════════════╪════════════════╪════════════════╪════════════════╪══════════════════╡
   │ 17.2.0  │ L2-Fabric Read BW   │ 42947428672.00 │ 42947428672.00 │ 42947428672.00 │ Bytes per kernel │
   ├─────────┼─────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.1  │ HBM Read Traffic    │         100.00 │         100.00 │         100.00 │ Pct              │
   ├─────────┼─────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.2  │ Remote Read Traffic │           0.00 │           0.00 │           0.00 │ Pct              │
   ╘═════════╧═════════════════════╧════════════════╧════════════════╧════════════════╧══════════════════╛
   17.4 L2 - Fabric Interface Stalls
   ╒═════════╤═══════════════════════════════╤════════════════════════╤═══════════════╤═══════╤═══════╤═══════╤════════╕
   │ Index   │ Metric                        │ Type                   │ Transaction   │   Avg │   Min │   Max │ Unit   │
   ╞═════════╪═══════════════════════════════╪════════════════════════╪═══════════════╪═══════╪═══════╪═══════╪════════╡
   │ 17.4.0  │ Read - PCIe Stall             │ PCIe Stall             │ Read          │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼───────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.1  │ Read - Infinity Fabric™ Stall │ Infinity Fabric™ Stall │ Read          │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼───────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.2  │ Read - HBM Stall              │ HBM Stall              │ Read          │  0.07 │  0.07 │  0.07 │ Pct    │
   ╘═════════╧═══════════════════════════════╧════════════════════════╧═══════════════╧═══════╧═══════╧═══════╧════════╛
   17.5 L2 - Fabric Detailed Transaction Breakdown
   ╒═════════╤═════════════════╤══════════════╤══════════════╤══════════════╤════════════════╕
   │ Index   │ Metric          │          Avg │          Min │          Max │ Unit           │
   ╞═════════╪═════════════════╪══════════════╪══════════════╪══════════════╪════════════════╡
   │ 17.5.0  │ Read (32B)      │         0.00 │         0.00 │         0.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.1  │ Read (Uncached) │      1450.00 │      1450.00 │      1450.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.2  │ Read (64B)      │ 671053573.00 │ 671053573.00 │ 671053573.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.3  │ HBM Read        │ 671053565.00 │ 671053565.00 │ 671053565.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.4  │ Remote Read     │         8.00 │         8.00 │         8.00 │ Req per kernel │
   ╘═════════╧═════════════════╧══════════════╧══════════════╧══════════════╧════════════════╛

Here, you can make the following observations.

- The vast majority of L2-Fabric requests (>99%) are 64B
  read requests (**17.5.2**).

- Nearly 100% of the read requests (**17.2.1**) are homed in on the
  accelerator-local HBM (**17.5.3**), while some small fraction of these reads are
  routed to a “remote” device (**17.5.4**).

- These drive a :math:`\sim40`\ GiB per kernel read-bandwidth (**17.2.0**).

In addition, we see a small amount of :ref:`uncached <memory-type>` reads
(**17.5.1**), these correspond to things like:

* The assembly code to execute the kernel

* Kernel arguments

* Coordinate parameters (such as ``blockDim.z``) that were not initialized by the
  hardware, etc. and may account for some of our "remote" read requests
  (**17.5.4**), for example, reading from CPU DRAM

The above list is not exhaustive, nor are all of these guaranteed to be
"uncached" – the exact implementation depends on the accelerator and
ROCm versions used. These read requests could be interrogated further in
the :ref:`Scalar L1 Data Cache <desc-sl1d>` and
:ref:`Instruction Cache <desc-l1i>` metric sections.

.. note::

   The Traffic metrics in Sec **17.2** are presented as a percentage of the total
   number of requests. For example, "HBM Read Traffic" is the percent of read requests
   (**17.5.0** - **17.5.2**) that were directed to the accelerators' local HBM (**17.5.3**).

.. _infinity-fabric-ex2:

Experiment 2: Fine-grained, accelerator-local HBM reads
---------------------------------------------------------

In this experiment, we change the :ref:`granularity <memory-type>` of our
device-allocation to be fine-grained device memory, local to the current
accelerator. Our code uses the ``hipExtMallocWithFlag`` API with the
``hipDeviceMallocFinegrained`` flag to accomplish this.

.. note::

   On some systems (such as those with only PCIe® connected accelerators), you need
   to set the environment variable ``HSA_FORCE_FINE_GRAIN_PCIE=1`` to enable
   this memory type.

.. code-block:: shell-session

   $ rocprof-compute profile -n fine_grained_local --no-roof -- ./fabric -t 0 -o 0
   Using:
     mtype:FineGrained
     mowner:Device
     mspace:Global
     mop:Read
     mdata:Unsigned
     remoteId:-1
   <...>
   $ rocprof-compute analyze -p workloads/fine_grained_local/mi200 -b 17.2.0 17.2.1 17.2.2 17.2.3 17.4.0 17.4.1 17.4.2 17.5.0 17.5.1 17.5.2 17.5.3 17.5.4  -n per_kernel --dispatch 2
   <...>
   17. L2 Cache
   17.2 L2 - Fabric Transactions
   ╒═════════╤═══════════════════════╤════════════════╤════════════════╤════════════════╤══════════════════╕
   │ Index   │ Metric                │            Avg │            Min │            Max │ Unit             │
   ╞═════════╪═══════════════════════╪════════════════╪════════════════╪════════════════╪══════════════════╡
   │ 17.2.0  │ L2-Fabric Read BW     │ 42948661824.00 │ 42948661824.00 │ 42948661824.00 │ Bytes per kernel │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.1  │ HBM Read Traffic      │         100.00 │         100.00 │         100.00 │ Pct              │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.2  │ Remote Read Traffic   │           0.00 │           0.00 │           0.00 │ Pct              │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.3  │ Uncached Read Traffic │           0.00 │           0.00 │           0.00 │ Pct              │
   ╘═════════╧═══════════════════════╧════════════════╧════════════════╧════════════════╧══════════════════╛
   17.4 L2 - Fabric Interface Stalls
   ╒═════════╤═══════════════════════════════╤════════════════════════╤═══════════════╤═══════╤═══════╤═══════╤════════╕
   │ Index   │ Metric                        │ Type                   │ Transaction   │   Avg │   Min │   Max │ Unit   │
   ╞═════════╪═══════════════════════════════╪════════════════════════╪═══════════════╪═══════╪═══════╪═══════╪════════╡
   │ 17.4.0  │ Read - PCIe Stall             │ PCIe Stall             │ Read          │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼───────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.1  │ Read - Infinity Fabric™ Stall │ Infinity Fabric™ Stall │ Read          │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼───────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.2  │ Read - HBM Stall              │ HBM Stall              │ Read          │  0.07 │  0.07 │  0.07 │ Pct    │
   ╘═════════╧═══════════════════════════════╧════════════════════════╧═══════════════╧═══════╧═══════╧═══════╧════════╛
   17.5 L2 - Fabric Detailed Transaction Breakdown
   ╒═════════╤═════════════════╤══════════════╤══════════════╤══════════════╤════════════════╕
   │ Index   │ Metric          │          Avg │          Min │          Max │ Unit           │
   ╞═════════╪═════════════════╪══════════════╪══════════════╪══════════════╪════════════════╡
   │ 17.5.0  │ Read (32B)      │         0.00 │         0.00 │         0.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.1  │ Read (Uncached) │      1334.00 │      1334.00 │      1334.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.2  │ Read (64B)      │ 671072841.00 │ 671072841.00 │ 671072841.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.3  │ HBM Read        │ 671072835.00 │ 671072835.00 │ 671072835.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.4  │ Remote Read     │         6.00 │         6.00 │         6.00 │ Req per kernel │
   ╘═════════╧═════════════════╧══════════════╧══════════════╧══════════════╧════════════════╛

Comparing with our :ref:`previous example <infinity-fabric-ex1>`, we see a
relatively similar result, namely:

- The vast majority of L2-Fabric requests are 64B read requests (**17.5.2**)

- Nearly all these read requests are directed to the accelerator-local HBM (**17.2.1**)

In addition, we now see a small percentage of HBM Read Stalls (**17.4.2**),
as streaming fine-grained memory is putting more stress on Infinity
Fabric.

.. note::

   The stalls in Sec 17.4 are presented as a percentage of the total number
   active L2 cycles, summed over :doc:`all L2 channels </conceptual/l2-cache>`.

.. _infinity-fabric-ex3:

Experiment 3: Fine-grained, remote-accelerator HBM reads
----------------------------------------------------------

In this experiment, we move our :ref:`fine-grained <memory-type>` allocation to
be owned by a remote accelerator. We accomplish this by first changing
the HIP device using, for instance, the ``hipSetDevice(1)`` API, then allocating
fine-grained memory (as described :ref:`previously <infinity-fabric-ex2>`), and
finally resetting the device back to the default, for instance,
``hipSetDevice(0)``.

Although we have not changed our code significantly, we do see a
substantial change in the L2-Fabric metrics:

.. code-block:: shell-session

   $ rocprof-compute profile -n fine_grained_remote --no-roof -- ./fabric -t 0 -o 2
   Using:
     mtype:FineGrained
     mowner:Remote
     mspace:Global
     mop:Read
     mdata:Unsigned
     remoteId:-1
   <...>
   $ rocprof-compute analyze -p workloads/fine_grained_remote/mi200 -b 17.2.0 17.2.1 17.2.2 17.2.3 17.4.0 17.4.1 17.4.2 17.5.0 17.5.1 17.5.2 17.5.3 17.5.4  -n per_kernel --dispatch 2
   <...>
   17. L2 Cache
   17.2 L2 - Fabric Transactions
   ╒═════════╤═══════════════════════╤════════════════╤════════════════╤════════════════╤══════════════════╕
   │ Index   │ Metric                │            Avg │            Min │            Max │ Unit             │
   ╞═════════╪═══════════════════════╪════════════════╪════════════════╪════════════════╪══════════════════╡
   │ 17.2.0  │ L2-Fabric Read BW     │ 42949692736.00 │ 42949692736.00 │ 42949692736.00 │ Bytes per kernel │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.1  │ HBM Read Traffic      │           0.00 │           0.00 │           0.00 │ Pct              │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.2  │ Remote Read Traffic   │         100.00 │         100.00 │         100.00 │ Pct              │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.3  │ Uncached Read Traffic │         200.00 │         200.00 │         200.00 │ Pct              │
   ╘═════════╧═══════════════════════╧════════════════╧════════════════╧════════════════╧══════════════════╛
   17.4 L2 - Fabric Interface Stalls
   ╒═════════╤═══════════════════════════════╤════════════════════════╤═══════════════╤═══════╤═══════╤═══════╤════════╕
   │ Index   │ Metric                        │ Type                   │ Transaction   │   Avg │   Min │   Max │ Unit   │
   ╞═════════╪═══════════════════════════════╪════════════════════════╪═══════════════╪═══════╪═══════╪═══════╪════════╡
   │ 17.4.0  │ Read - PCIe Stall             │ PCIe Stall             │ Read          │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼───────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.1  │ Read - Infinity Fabric™ Stall │ Infinity Fabric™ Stall │ Read          │ 17.85 │ 17.85 │ 17.85 │ Pct    │
   ├─────────┼───────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.2  │ Read - HBM Stall              │ HBM Stall              │ Read          │  0.00 │  0.00 │  0.00 │ Pct    │
   ╘═════════╧═══════════════════════════════╧════════════════════════╧═══════════════╧═══════╧═══════╧═══════╧════════╛
   17.5 L2 - Fabric Detailed Transaction Breakdown
   ╒═════════╤═════════════════╤═══════════════╤═══════════════╤═══════════════╤════════════════╕
   │ Index   │ Metric          │           Avg │           Min │           Max │ Unit           │
   ╞═════════╪═════════════════╪═══════════════╪═══════════════╪═══════════════╪════════════════╡
   │ 17.5.0  │ Read (32B)      │          0.00 │          0.00 │          0.00 │ Req per kernel │
   ├─────────┼─────────────────┼───────────────┼───────────────┼───────────────┼────────────────┤
   │ 17.5.1  │ Read (Uncached) │ 1342177894.00 │ 1342177894.00 │ 1342177894.00 │ Req per kernel │
   ├─────────┼─────────────────┼───────────────┼───────────────┼───────────────┼────────────────┤
   │ 17.5.2  │ Read (64B)      │  671088949.00 │  671088949.00 │  671088949.00 │ Req per kernel │
   ├─────────┼─────────────────┼───────────────┼───────────────┼───────────────┼────────────────┤
   │ 17.5.3  │ HBM Read        │        307.00 │        307.00 │        307.00 │ Req per kernel │
   ├─────────┼─────────────────┼───────────────┼───────────────┼───────────────┼────────────────┤
   │ 17.5.4  │ Remote Read     │  671088642.00 │  671088642.00 │  671088642.00 │ Req per kernel │
   ╘═════════╧═════════════════╧═══════════════╧═══════════════╧═══════════════╧════════════════╛

First, we see that while we still observe approximately the same number
of 64B Read Requests (**17.5.2**), we now see an even larger number of
Uncached Read Requests (**17.5.3**). Some simple division reveals:

.. math::

   342177894.00 / 671088949.00 ≈ 2

That is, each 64B Read Request is *also* counted as two Uncached Read
Requests, as reflected in the :ref:`request-flow diagram <l2-request-flow>`.
This is also why the Uncached Read Traffic metric (**17.2.3**) is at the
counter-intuitive value of 200%!

In addition, observe that:

- We no longer see any significant number of HBM Read Requests (**17.2.1**,
  **17.5.3**), nor HBM Read Stalls (**17.4.2**), but instead,

- we see that almost all of these requests are considered “remote”
  (**17.2.2**, **17.5.4**) are being routed to another
  accelerator, or the CPU — in this case HIP Device 1 — and,

- we see a significantly larger percentage of AMD Infinity Fabric Read Stalls
  (**17.4.1**) as compared to the HBM Read Stalls in the
  :ref:`previous example <infinity-fabric-ex2>`.

These stalls correspond to reads that are going out over the AMD
Infinity Fabric connection to another MI250 accelerator. In
addition, because these are crossing between accelerators, we expect
significantly lower achievable bandwidths as compared to the local
accelerator’s HBM – this is reflected (indirectly) in the magnitude of
the stall metric (**17.4.1**). Finally, we note that if our system contained
only PCIe connected accelerators, these observations will differ.

.. _infinity-fabric-ex4:

Experiment 4: Fine-grained, CPU-DRAM reads
--------------------------------------------

In this experiment, we move our :ref:`fine-grained <memory-type>` allocation to
be owned by the CPU’s DRAM. We accomplish this by allocating host-pinned
fine-grained memory using the ``hipHostMalloc`` API:

.. code-block:: shell-session

   $ rocprof-compute profile -n fine_grained_host --no-roof -- ./fabric -t 0 -o 1
   Using:
     mtype:FineGrained
     mowner:Host
     mspace:Global
     mop:Read
     mdata:Unsigned
     remoteId:-1
   <...>
   $ rocprof-compute analyze -p workloads/fine_grained_host/mi200 -b 17.2.0 17.2.1 17.2.2 17.2.3 17.4.0 17.4.1 17.4.2 17.5.0 17.5.1 17.5.2 17.5.3 17.5.4  -n per_kernel --dispatch 2
   <...>
   17. L2 Cache
   17.2 L2 - Fabric Transactions
   ╒═════════╤═══════════════════════╤════════════════╤════════════════╤════════════════╤══════════════════╕
   │ Index   │ Metric                │            Avg │            Min │            Max │ Unit             │
   ╞═════════╪═══════════════════════╪════════════════╪════════════════╪════════════════╪══════════════════╡
   │ 17.2.0  │ L2-Fabric Read BW     │ 42949691264.00 │ 42949691264.00 │ 42949691264.00 │ Bytes per kernel │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.1  │ HBM Read Traffic      │           0.00 │           0.00 │           0.00 │ Pct              │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.2  │ Remote Read Traffic   │         100.00 │         100.00 │         100.00 │ Pct              │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.3  │ Uncached Read Traffic │         200.00 │         200.00 │         200.00 │ Pct              │
   ╘═════════╧═══════════════════════╧════════════════╧════════════════╧════════════════╧══════════════════╛
   17.4 L2 - Fabric Interface Stalls
   ╒═════════╤═══════════════════════════════╤════════════════════════╤═══════════════╤═══════╤═══════╤═══════╤════════╕
   │ Index   │ Metric                        │ Type                   │ Transaction   │   Avg │   Min │   Max │ Unit   │
   ╞═════════╪═══════════════════════════════╪════════════════════════╪═══════════════╪═══════╪═══════╪═══════╪════════╡
   │ 17.4.0  │ Read - PCIe Stall             │ PCIe Stall             │ Read          │ 91.29 │ 91.29 │ 91.29 │ Pct    │
   ├─────────┼───────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.1  │ Read - Infinity Fabric™ Stall │ Infinity Fabric™ Stall │ Read          │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼───────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.2  │ Read - HBM Stall              │ HBM Stall              │ Read          │  0.00 │  0.00 │  0.00 │ Pct    │
   ╘═════════╧═══════════════════════════════╧════════════════════════╧═══════════════╧═══════╧═══════╧═══════╧════════╛
   17.5 L2 - Fabric Detailed Transaction Breakdown
   ╒═════════╤═════════════════╤═══════════════╤═══════════════╤═══════════════╤════════════════╕
   │ Index   │ Metric          │           Avg │           Min │           Max │ Unit           │
   ╞═════════╪═════════════════╪═══════════════╪═══════════════╪═══════════════╪════════════════╡
   │ 17.5.0  │ Read (32B)      │          0.00 │          0.00 │          0.00 │ Req per kernel │
   ├─────────┼─────────────────┼───────────────┼───────────────┼───────────────┼────────────────┤
   │ 17.5.1  │ Read (Uncached) │ 1342177848.00 │ 1342177848.00 │ 1342177848.00 │ Req per kernel │
   ├─────────┼─────────────────┼───────────────┼───────────────┼───────────────┼────────────────┤
   │ 17.5.2  │ Read (64B)      │  671088926.00 │  671088926.00 │  671088926.00 │ Req per kernel │
   ├─────────┼─────────────────┼───────────────┼───────────────┼───────────────┼────────────────┤
   │ 17.5.3  │ HBM Read        │        284.00 │        284.00 │        284.00 │ Req per kernel │
   ├─────────┼─────────────────┼───────────────┼───────────────┼───────────────┼────────────────┤
   │ 17.5.4  │ Remote Read     │  671088642.00 │  671088642.00 │  671088642.00 │ Req per kernel │
   ╘═════════╧═════════════════╧═══════════════╧═══════════════╧═══════════════╧════════════════╛

Here we see *almost* the same results as in the
:ref:`previous experiment <infinity-fabric-ex3>`, however now as we are crossing
a PCIe bus to the CPU, we see that the Infinity Fabric Read stalls (**17.4.1**)
have shifted to be a PCIe stall (**17.4.2**). In addition, as (on this
system) the PCIe bus has a lower peak bandwidth than the AMD Infinity
Fabric connection between two accelerators, we once again observe an
increase in the percentage of stalls on this interface.

.. note::

   Had we performed this same experiment on an
   `MI250X system <https://www.amd.com/system/files/documents/amd-cdna2-white-paper.pdf>`_,
   these transactions would again have been marked as Infinity Fabric Read
   stalls (**17.4.1**), as the CPU is connected to the accelerator via AMD Infinity
   Fabric.

.. _infinity-fabric-ex5:

Experiment 5: Coarse-grained, CPU-DRAM reads
----------------------------------------------

In our next fabric experiment, we change our CPU memory allocation to be
:ref:`coarse-grained <memory-type>`. We accomplish this by passing the
``hipHostMalloc`` API the ``hipHostMallocNonCoherent`` flag, to mark the
allocation as coarse-grained:

.. code-block:: shell-session

   $ rocprof-compute profile -n coarse_grained_host --no-roof -- ./fabric -t 1 -o 1
   Using:
     mtype:CoarseGrained
     mowner:Host
     mspace:Global
     mop:Read
     mdata:Unsigned
     remoteId:-1
   <...>
   $ rocprof-compute analyze -p workloads/coarse_grained_host/mi200 -b 17.2.0 17.2.1 17.2.2 17.2.3 17.4.0 17.4.1 17.4.2 17.5.0 17.5.1 17.5.2 17.5.3 17.5.4  -n per_kernel --dispatch 2
   <...>
   17. L2 Cache
   17.2 L2 - Fabric Transactions
   ╒═════════╤═══════════════════════╤════════════════╤════════════════╤════════════════╤══════════════════╕
   │ Index   │ Metric                │            Avg │            Min │            Max │ Unit             │
   ╞═════════╪═══════════════════════╪════════════════╪════════════════╪════════════════╪══════════════════╡
   │ 17.2.0  │ L2-Fabric Read BW     │ 42949691264.00 │ 42949691264.00 │ 42949691264.00 │ Bytes per kernel │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.1  │ HBM Read Traffic      │           0.00 │           0.00 │           0.00 │ Pct              │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.2  │ Remote Read Traffic   │         100.00 │         100.00 │         100.00 │ Pct              │
   ├─────────┼───────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.3  │ Uncached Read Traffic │           0.00 │           0.00 │           0.00 │ Pct              │
   ╘═════════╧═══════════════════════╧════════════════╧════════════════╧════════════════╧══════════════════╛
   17.4 L2 - Fabric Interface Stalls
   ╒═════════╤═══════════════════════════════╤════════════════════════╤═══════════════╤═══════╤═══════╤═══════╤════════╕
   │ Index   │ Metric                        │ Type                   │ Transaction   │   Avg │   Min │   Max │ Unit   │
   ╞═════════╪═══════════════════════════════╪════════════════════════╪═══════════════╪═══════╪═══════╪═══════╪════════╡
   │ 17.4.0  │ Read - PCIe Stall             │ PCIe Stall             │ Read          │ 91.27 │ 91.27 │ 91.27 │ Pct    │
   ├─────────┼───────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.1  │ Read - Infinity Fabric™ Stall │ Infinity Fabric™ Stall │ Read          │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼───────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.2  │ Read - HBM Stall              │ HBM Stall              │ Read          │  0.00 │  0.00 │  0.00 │ Pct    │
   ╘═════════╧═══════════════════════════════╧════════════════════════╧═══════════════╧═══════╧═══════╧═══════╧════════╛
   17.5 L2 - Fabric Detailed Transaction Breakdown
   ╒═════════╤═════════════════╤══════════════╤══════════════╤══════════════╤════════════════╕
   │ Index   │ Metric          │          Avg │          Min │          Max │ Unit           │
   ╞═════════╪═════════════════╪══════════════╪══════════════╪══════════════╪════════════════╡
   │ 17.5.0  │ Read (32B)      │         0.00 │         0.00 │         0.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.1  │ Read (Uncached) │       562.00 │       562.00 │       562.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.2  │ Read (64B)      │ 671088926.00 │ 671088926.00 │ 671088926.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.3  │ HBM Read        │       281.00 │       281.00 │       281.00 │ Req per kernel │
   ├─────────┼─────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.4  │ Remote Read     │ 671088645.00 │ 671088645.00 │ 671088645.00 │ Req per kernel │
   ╘═════════╧═════════════════╧══════════════╧══════════════╧══════════════╧════════════════╛

Here we see a similar result to our
:ref:`previous experiment <infinity-fabric-ex4>`, with one key difference: our
accesses are no longer marked as Uncached Read requests (**17.2.3, 17.5.1**), but instead
are 64B read requests (**17.5.2**), as observed in our
:ref:`Coarse-grained, accelerator-local HBM <infinity-fabric-ex1>` experiment.

.. _infinity-fabric-ex6:

Experiment 6: Fine-grained, CPU-DRAM writes
--------------------------------------------

Thus far in our exploration of the L2-Fabric interface, we have
primarily focused on read operations. However, in
:ref:`our request flow diagram <l2-request-flow>`, we note that writes are
counted separately. To observe this, we use the ``-p`` flag to trigger write
operations to fine-grained memory allocated on the host:

.. code-block:: shell-session

   $ rocprof-compute profile -n fine_grained_host_write --no-roof -- ./fabric -t 0 -o 1 -p 1
   Using:
     mtype:FineGrained
     mowner:Host
     mspace:Global
     mop:Write
     mdata:Unsigned
     remoteId:-1
   <...>
   $ rocprof-compute analyze -p workloads/fine_grained_host_writes/mi200 -b 17.2.4 17.2.5 17.2.6 17.2.7 17.2.8 17.4.3 17.4.4 17.4.5 17.4.6 17.5.5 17.5.6 17.5.7 17.5.8 17.5.9 17.5.10 -n per_kernel --dispatch 2
   <...>
   17. L2 Cache
   17.2 L2 - Fabric Transactions
   ╒═════════╤═══════════════════════════════════╤════════════════╤════════════════╤════════════════╤══════════════════╕
   │ Index   │ Metric                            │            Avg │            Min │            Max │ Unit             │
   ╞═════════╪═══════════════════════════════════╪════════════════╪════════════════╪════════════════╪══════════════════╡
   │ 17.2.4  │ L2-Fabric Write and Atomic BW     │ 42949672960.00 │ 42949672960.00 │ 42949672960.00 │ Bytes per kernel │
   ├─────────┼───────────────────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.5  │ HBM Write and Atomic Traffic      │           0.00 │           0.00 │           0.00 │ Pct              │
   ├─────────┼───────────────────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.6  │ Remote Write and Atomic Traffic   │         100.00 │         100.00 │         100.00 │ Pct              │
   ├─────────┼───────────────────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.7  │ Atomic Traffic                    │           0.00 │           0.00 │           0.00 │ Pct              │
   ├─────────┼───────────────────────────────────┼────────────────┼────────────────┼────────────────┼──────────────────┤
   │ 17.2.8  │ Uncached Write and Atomic Traffic │         100.00 │         100.00 │         100.00 │ Pct              │
   ╘═════════╧═══════════════════════════════════╧════════════════╧════════════════╧════════════════╧══════════════════╛
   17.4 L2 - Fabric Interface Stalls
   ╒═════════╤════════════════════════════════╤════════════════════════╤═══════════════╤═══════╤═══════╤═══════╤════════╕
   │ Index   │ Metric                         │ Type                   │ Transaction   │   Avg │   Min │   Max │ Unit   │
   ╞═════════╪════════════════════════════════╪════════════════════════╪═══════════════╪═══════╪═══════╪═══════╪════════╡
   │ 17.4.3  │ Write - PCIe Stall             │ PCIe Stall             │ Write         │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.4  │ Write - Infinity Fabric™ Stall │ Infinity Fabric™ Stall │ Write         │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.5  │ Write - HBM Stall              │ HBM Stall              │ Write         │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.6  │ Write - Credit Starvation      │ Credit Starvation      │ Write         │  0.00 │  0.00 │  0.00 │ Pct    │
   ╘═════════╧════════════════════════════════╧════════════════════════╧═══════════════╧═══════╧═══════╧═══════╧════════╛
   17.5 L2 - Fabric Detailed Transaction Breakdown
   ╒═════════╤═════════════════════════╤══════════════╤══════════════╤══════════════╤════════════════╕
   │ Index   │ Metric                  │          Avg │          Min │          Max │ Unit           │
   ╞═════════╪═════════════════════════╪══════════════╪══════════════╪══════════════╪════════════════╡
   │ 17.5.5  │ Write (32B)             │         0.00 │         0.00 │         0.00 │ Req per kernel │
   ├─────────┼─────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.6  │ Write (Uncached)        │ 671088640.00 │ 671088640.00 │ 671088640.00 │ Req per kernel │
   ├─────────┼─────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.7  │ Write (64B)             │ 671088640.00 │ 671088640.00 │ 671088640.00 │ Req per kernel │
   ├─────────┼─────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.8  │ HBM Write and Atomic    │         0.00 │         0.00 │         0.00 │ Req per kernel │
   ├─────────┼─────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.9  │ Remote Write and Atomic │ 671088640.00 │ 671088640.00 │ 671088640.00 │ Req per kernel │
   ├─────────┼─────────────────────────┼──────────────┼──────────────┼──────────────┼────────────────┤
   │ 17.5.10 │ Atomic                  │         0.00 │         0.00 │         0.00 │ Req per kernel │
   ╘═════════╧═════════════════════════╧══════════════╧══════════════╧══════════════╧════════════════╛

Here we notice a few changes in our request pattern:

* As expected, the requests have changed from 64B Reads to 64B Write requests
  (**17.5.7**),

* these requests are homed in on a “remote” destination (**17.2.6, 17.5.9**), as
  expected, and

* these are also counted as a single Uncached Write request (**17.5.6**).

In addition, there are rather significant changes in the bandwidth values
reported:

- The “L2-Fabric Write and Atomic” bandwidth metric (**17.2.4**)
  reports about 40GiB of data written across Infinity Fabric while

- The “Remote Write and Traffic” metric (**17.2.5**) indicates that nearly
  100% of these request are being directed to a remote source.

The precise meaning of these metrics are explored in the
:ref:`subsequent experiment <infinity-fabric-ex7>`.

Finally, we note that we see no write stalls on the PCIe bus
(**17.4.3**). This is because writes over a PCIe bus `are
non-posted <https://members.pcisig.com/wg/PCI-SIG/document/10912>`_,
that is, they do not require acknowledgement.

.. _infinity-fabric-ex7:

Experiment 7: Fine-grained, CPU-DRAM atomicAdd
------------------------------------------------

Next, we change our experiment to instead target ``atomicAdd``
operations to the CPU’s DRAM.

.. code-block:: shell-session

   $ rocprof-compute profile -n fine_grained_host_add --no-roof -- ./fabric -t 0 -o 1 -p 2
   Using:
     mtype:FineGrained
     mowner:Host
     mspace:Global
     mop:Add
     mdata:Unsigned
     remoteId:-1
   <...>
   $ rocprof-compute analyze -p workloads/fine_grained_host_add/mi200 -b 17.2.4 17.2.5 17.2.6 17.2.7 17.2.8 17.4.3 17.4.4 17.4.5 17.4.6 17.5.5 17.5.6 17.5.7 17.5.8 17.5.9 17.5.10 -n per_kernel --dispatch 2
   <...>
   17. L2 Cache
   17.2 L2 - Fabric Transactions
   ╒═════════╤═══════════════════════════════════╤══════════════╤══════════════╤══════════════╤══════════════════╕
   │ Index   │ Metric                            │          Avg │          Min │          Max │ Unit             │
   ╞═════════╪═══════════════════════════════════╪══════════════╪══════════════╪══════════════╪══════════════════╡
   │ 17.2.4  │ L2-Fabric Write and Atomic BW     │ 429496736.00 │ 429496736.00 │ 429496736.00 │ Bytes per kernel │
   ├─────────┼───────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤
   │ 17.2.5  │ HBM Write and Atomic Traffic      │         0.00 │         0.00 │         0.00 │ Pct              │
   ├─────────┼───────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤
   │ 17.2.6  │ Remote Write and Atomic Traffic   │       100.00 │       100.00 │       100.00 │ Pct              │
   ├─────────┼───────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤
   │ 17.2.7  │ Atomic Traffic                    │       100.00 │       100.00 │       100.00 │ Pct              │
   ├─────────┼───────────────────────────────────┼──────────────┼──────────────┼──────────────┼──────────────────┤
   │ 17.2.8  │ Uncached Write and Atomic Traffic │       100.00 │       100.00 │       100.00 │ Pct              │
   ╘═════════╧═══════════════════════════════════╧══════════════╧══════════════╧══════════════╧══════════════════╛
   17.4 L2 - Fabric Interface Stalls
   ╒═════════╤════════════════════════════════╤════════════════════════╤═══════════════╤═══════╤═══════╤═══════╤════════╕
   │ Index   │ Metric                         │ Type                   │ Transaction   │   Avg │   Min │   Max │ Unit   │
   ╞═════════╪════════════════════════════════╪════════════════════════╪═══════════════╪═══════╪═══════╪═══════╪════════╡
   │ 17.4.3  │ Write - PCIe Stall             │ PCIe Stall             │ Write         │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.4  │ Write - Infinity Fabric™ Stall │ Infinity Fabric™ Stall │ Write         │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.5  │ Write - HBM Stall              │ HBM Stall              │ Write         │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────┼────────────────────────┼───────────────┼───────┼───────┼───────┼────────┤
   │ 17.4.6  │ Write - Credit Starvation      │ Credit Starvation      │ Write         │  0.00 │  0.00 │  0.00 │ Pct    │
   ╘═════════╧════════════════════════════════╧════════════════════════╧═══════════════╧═══════╧═══════╧═══════╧════════╛
   17.5 L2 - Fabric Detailed Transaction Breakdown
   ╒═════════╤═════════════════════════╤═════════════╤═════════════╤═════════════╤════════════════╕
   │ Index   │ Metric                  │         Avg │         Min │         Max │ Unit           │
   ╞═════════╪═════════════════════════╪═════════════╪═════════════╪═════════════╪════════════════╡
   │ 17.5.5  │ Write (32B)             │ 13421773.00 │ 13421773.00 │ 13421773.00 │ Req per kernel │
   ├─────────┼─────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
   │ 17.5.6  │ Write (Uncached)        │ 13421773.00 │ 13421773.00 │ 13421773.00 │ Req per kernel │
   ├─────────┼─────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
   │ 17.5.7  │ Write (64B)             │        0.00 │        0.00 │        0.00 │ Req per kernel │
   ├─────────┼─────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
   │ 17.5.8  │ HBM Write and Atomic    │        0.00 │        0.00 │        0.00 │ Req per kernel │
   ├─────────┼─────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
   │ 17.5.9  │ Remote Write and Atomic │ 13421773.00 │ 13421773.00 │ 13421773.00 │ Req per kernel │
   ├─────────┼─────────────────────────┼─────────────┼─────────────┼─────────────┼────────────────┤
   │ 17.5.10 │ Atomic                  │ 13421773.00 │ 13421773.00 │ 13421773.00 │ Req per kernel │
   ╘═════════╧═════════════════════════╧═════════════╧═════════════╧═════════════╧════════════════╛

In this case, there is quite a lot to unpack:

- For the first time, the 32B Write requests (**17.5.5**) are heavily used.

- These correspond to Atomic requests (**17.2.7, 17.5.10**), and are counted as
  Uncached Writes (**17.5.6**).

- The L2-Fabric Write and Atomic bandwidth metric (**17.2.4**) shows about 0.4
  GiB of traffic. For convenience, the sample reduces the default problem size
  for this case due to the speed of atomics across a PCIe bus, and finally,

- The traffic is directed to a remote device (**17.2.6, 17.5.9**).

Let's consider what an “atomic” request means in this context. Recall
that we are discussing memory traffic flowing from the L2 cache, the
device-wide coherence point on current CDNA accelerators such as the
MI250, to for example, the CPU’s DRAM. In this light, we see that these
requests correspond to *system scope* atomics, and specifically in the
case of the MI250, to fine-grained memory.


.. rubric:: Disclaimer

PCIe® is a registered trademark of PCI-SIG Corporation.

..
   `Leave as possible future experiment to add


   ### Experiment #2 - Non-temporal writes

   If we take the same code (for convenience only) as previously described, we can demonstrate how to achieve 'streaming' writes, as described in the [L2 Cache Access metrics](L2_cache_metrics) section.
   To see this, we use the Clang built-in [`__builtin_nontemporal_store`](https://clang.llvm.org/docs/LanguageExtensions.html#non-temporal-load-store-builtins), for example

   ```
   template<typename T>
   __device__ void store (T* ptr, T val) {
    __builtin_nontemporal_store(val, ptr);
   }
   ```

   On an AMD MI2XX accelerator, for FP32 values this will generate a `global_store_dword` instruction, with the `glc` and `slc` bits set, described in [section 10.1](https://developer.amd.com/wp-content/resources/CDNA2_Shader_ISA_4February2022.pdf) of the CDNA2 ISA guide.`
