.. _lds-examples:

LDS examples
============

For this example, consider the
:dev-sample:`LDS sample <lds.hip>` distributed as a part of ROCm Compute Profiler. This
code contains two kernels to explore how both :doc:`LDS </conceptual/local-data-share>` bandwidth and
bank conflicts are calculated in ROCm Compute Profiler.

This example was compiled and run on an MI250 accelerator using ROCm
v5.6.0, and ROCm Compute Profiler v2.0.0.

.. code-block:: shell-session

   $ hipcc -O3 lds.hip -o lds

Finally, we generate our ``rocprof-compute profile`` as:

.. code-block:: shell-session

   $ rocprof-compute profile -n lds --no-roof -- ./lds

.. _lds-bandwidth:

LDS bandwidth
-------------

To explore our *theoretical LDS bandwidth* metric, we use a simple
kernel:

.. code-block:: cpp

   constexpr unsigned max_threads = 256;
   __global__ void load(int* out, int flag) {
     __shared__ int array[max_threads];
     int index = threadIdx.x;
     // fake a store to the LDS array to avoid unwanted behavior
     if (flag)
       array[max_threads - index] = index;
     __syncthreads();
     int x = array[index];
     if (x == int(-1234567))
       out[threadIdx.x] = x;
   }

Here we:

* Create an array of 256 integers in :doc:`LDS </conceptual/local-data-share>`

* Fake a write to the LDS using the ``flag`` variable (always set to zero on the
  host) to avoid dead-code elimination

* Read a single integer per work-item from ``threadIdx.x`` of the LDS array

* If the integer is equal to a magic number (always false), write the value out
  to global memory to again, avoid dead-code elimination

Finally, we launch this kernel repeatedly, varying the number of threads
in our workgroup:

.. code-block:: cpp

   void bandwidth_demo(int N) {
     for (int i = 1; i <= N; ++i)
       load<<<1,i>>>(nullptr, 0);
     hipDeviceSynchronize();
   }

Next, let’s analyze the first of our bandwidth kernel dispatches:

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/lds/mi200/ -b 12.2.1 --dispatch 0 -n per_kernel
   <...>
   12. Local Data Share (LDS)
   12.2 LDS Stats
   ╒═════════╤═══════════════════════╤════════╤════════╤════════╤══════════════════╕
   │ Index   │ Metric                │    Avg │    Min │    Max │ Unit             │
   ╞═════════╪═══════════════════════╪════════╪════════╪════════╪══════════════════╡
   │ 12.2.1  │ Theoretical Bandwidth │ 256.00 │ 256.00 │ 256.00 │ Bytes per kernel │
   ╘═════════╧═══════════════════════╧════════╧════════╧════════╧══════════════════╛

Here we see that our Theoretical Bandwidth metric (**12.2.1**) is reporting
256 Bytes were loaded even though we launched a single work-item
workgroup, and thus only loaded a single integer from LDS. Why is this?

Recall our definition of this metric:

   Indicates the maximum amount of bytes that could have been loaded
   from/stored to/atomically updated in the LDS per
   :ref:`normalization unit <normalization-units>`.

Here we see that this instruction *could* have loaded up to 256 bytes of
data (4 bytes for each work-item in the wavefront), and therefore this
is the expected value for this metric in ROCm Compute Profiler, hence why this metric
is named the “theoretical” bandwidth.

To further illustrate this point we plot the relationship of the
theoretical bandwidth metric (**12.2.1**) as compared to the effective (or
achieved) bandwidth of this kernel, varying the number of work-items
launched from 1 to 256:

.. figure:: ../data/profiling-by-example/ldsbandwidth.png
   :align: center
   :alt: Comparison of effective bandwidth versus the theoretical bandwidth
         metric in ROCm Compute Profiler for our simple example.
   :width: 800

   Comparison of effective bandwidth versus the theoretical bandwidth
   metric in ROCm Compute Profiler for our simple example.

Here we see that the theoretical bandwidth metric follows a step-function. It
increases only when another wavefront issues an LDS instruction for up to 256
bytes of data. Such increases are marked in the plot using dashed lines. In
contrast, the effective bandwidth increases linearly, by 4 bytes, with the
number of work-items in the kernel, N.

.. _lds-bank-conflicts:

Bank conflicts
--------------

Next we explore bank conflicts using a slight modification of our bandwidth
kernel:

.. code-block:: cpp

   constexpr unsigned nbanks = 32;
   __global__ void conflicts(int* out, int flag) {
     constexpr unsigned nelements = nbanks * max_threads;
     __shared__ int array[nelements];
     // each thread reads from the same bank
     int index = threadIdx.x * nbanks;
     // fake a store to the LDS array to avoid unwanted behavior
     if (flag)
       array[max_threads - index] = index;
     __syncthreads();
     int x = array[index];
     if (x == int(-1234567))
       out[threadIdx.x] = x;
   }

Here we:

* Allocate an :doc:`LDS </conceptual/local-data-share>` array of size
  :math:`32*256*4{B}=32{KiB}`

* Fake a write to the LDS using the ``flag``
  variable (always set to zero on the host) to avoid dead-code elimination

* Read a single integer per work-item from index
  ``threadIdx.x * nbanks`` of the LDS array

* If the integer is equal to a
  magic number (always false), write the value out to global memory to,
  again, avoid dead-code elimination.

On the host, we again repeatedly launch this kernel, varying the number
of work-items:

.. code-block:: cpp

   void conflicts_demo(int N) {
     for (int i = 1; i <= N; ++i)
       conflicts<<<1,i>>>(nullptr, 0);
     hipDeviceSynchronize();
   }

Analyzing our first ``conflicts`` kernel (i.e., a single work-item), we
see:

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/lds/mi200/ -b 12.2.4 12.2.6 --dispatch 256 -n per_kernel
   <...>
   --------------------------------------------------------------------------------
   12. Local Data Share (LDS)
   12.2 LDS Stats
   ╒═════════╤════════════════╤═══════╤═══════╤═══════╤═══════════════════╕
   │ Index   │ Metric         │   Avg │   Min │   Max │ Unit              │
   ╞═════════╪════════════════╪═══════╪═══════╪═══════╪═══════════════════╡
   │ 12.2.4  │ Index Accesses │  2.00 │  2.00 │  2.00 │ Cycles per kernel │
   ├─────────┼────────────────┼───────┼───────┼───────┼───────────────────┤
   │ 12.2.6  │ Bank Conflict  │  0.00 │  0.00 │  0.00 │ Cycles per kernel │
   ╘═════════╧════════════════╧═══════╧═══════╧═══════╧═══════════════════╛

In our :ref:`previous example <lds-bank-conflicts>`, we showed how a load
from a single work-item is considered to have a theoretical bandwidth of
256B. Recall, the :doc:`LDS </conceptual/local-data-share>` can load up to :math:`128B` per
cycle (i.e, 32 banks x 4B / bank / cycle). Hence, we see that loading an 4B
integer spends two cycles accessing the LDS
(:math:`2\ {cycle} = (256B) / (128\ B/{cycle})`).

Looking at the next ``conflicts`` dispatch (i.e., two work-items) yields:

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/lds/mi200/ -b 12.2.4 12.2.6 --dispatch 257 -n per_kernel
   <...>
   --------------------------------------------------------------------------------
   12. Local Data Share (LDS)
   12.2 LDS Stats
   ╒═════════╤════════════════╤═══════╤═══════╤═══════╤═══════════════════╕
   │ Index   │ Metric         │   Avg │   Min │   Max │ Unit              │
   ╞═════════╪════════════════╪═══════╪═══════╪═══════╪═══════════════════╡
   │ 12.2.4  │ Index Accesses │  3.00 │  3.00 │  3.00 │ Cycles per kernel │
   ├─────────┼────────────────┼───────┼───────┼───────┼───────────────────┤
   │ 12.2.6  │ Bank Conflict  │  1.00 │  1.00 │  1.00 │ Cycles per kernel │
   ╘═════════╧════════════════╧═══════╧═══════╧═══════╧═══════════════════╛

Here we see a bank conflict! What happened?

Recall that the index for each thread was calculated as:

.. code-block:: cpp

   int index = threadIdx.x * nbanks;

Or, precisely 32 elements, and each element is 4B wide (for a standard
integer). That is, each thread strides back to the same bank in the LDS,
such that each work-item we add to the dispatch results in another bank
conflict!

Recalling our discussion of bank conflicts in our
:doc:`LDS </conceptual/local-data-share>` description:

A bank conflict occurs when two (or more) work-items in a wavefront
want to read, write, or atomically update different addresses that
map to the same bank in the same cycle. In this case, the conflict
detection hardware will determined a new schedule such that the
access is split into multiple cycles with no conflicts in any
single cycle.

Here we see the conflict resolution hardware in action! Because we have
engineered our kernel to generate conflicts, we expect our bank conflict
metric to scale linearly with the number of work-items:

.. figure:: ../data/profiling-by-example/ldsconflicts.png
   :align: center
   :alt: Comparison of LDS conflict cycles versus access cycles for our simple
         example.
   :width: 800

   Comparison of LDS conflict cycles versus access cycles for our simple
   example.

Here we show the comparison of the Index Accesses (**12.2.4**), to the Bank
Conflicts (**12.2.6**) for the first 20 kernel invocations. We see that each grows
linearly, and there is a constant gap of 2 cycles between them (i.e., the first
access is never considered a conflict).

Finally, we can use these two metrics to derive the Bank Conflict Rate (**12.1.4**).
Since within an Index Access we have 32 banks that may need to be updated, we
use:

$$
Bank\ Conflict\ Rate = 100 * ((Bank\ Conflicts / 32) / (Index\ Accesses - Bank\ Conflicts))
$$

Plotting this, we see:

.. figure:: ../data/profiling-by-example/ldsconflictrate.png
   :align: center
   :alt: LDS bank conflict rate example
   :width: 800

   LDS Bank Conflict rate for our simple example.

The bank conflict rate linearly increases with the number of work-items
within a wavefront that are active, *approaching* 100%, but never quite
reaching it.
