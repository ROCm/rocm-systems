.. _occupancy-example:

Occupancy limiters example
==========================

For this example, consider the
:dev-sample:`occupancy <occupancy.hip>` included with ROCm Compute Profiler. We will
investigate the use of the resource allocation panel in the
:ref:`Workgroup Manager <desc-spi>`’s metrics section to determine occupancy
limiters. This code contains several kernels to explore how both various
kernel resources impact achieved occupancy, and how this is reported in
ROCm Compute Profiler.

This example was compiled and run on a MI250 accelerator using ROCm
v5.6.0, and ROCm Compute Profiler v2.0.0:

.. code-block:: shell

   $ hipcc -O3 occupancy.hip -o occupancy --save-temps

We have again included the ``--save-temps`` flag to get the
corresponding assembly.

Finally, we generate our ROCm Compute Profiler profile as:

.. code-block:: shell

   $ rocprof-compute profile -n occupancy --no-roof -- ./occupancy

.. _occupancy-experiment-design:

Design note
-----------

For our occupancy test, we need to create a kernel that is resource
heavy, in various ways. For this purpose, we use the following (somewhat
funny-looking) kernel:

.. code-block:: cpp

   constexpr int bound = 16;
   __launch_bounds__(256)
   __global__ void vgprbound(int N, double* ptr) {
       double intermediates[bound];
       for (int i = 0 ; i < bound; ++i) intermediates[i] = N * threadIdx.x;
       double x = ptr[threadIdx.x];
       for (int i = 0; i < 100; ++i) {
           x += sin(pow(__shfl(x, i % warpSize) * intermediates[(i - 1) % bound], intermediates[i % bound]));
           intermediates[i % bound] = x;
       }
       if (x == N) ptr[threadIdx.x] = x;
   }

Here we try to use as many :ref:`VGPRs <desc-valu>` as possible, to this end:

* We create a small array of double precision floats, that we size to try
  to fit into registers (i.e., ``bound``, this may need to be tuned
  depending on the ROCm version).

* We specify ``__launch_bounds___(256)``
  to increase the number of VPGRs available to the kernel (by limiting the
  number of wavefronts that can be resident on a
  :doc:`CU </conceptual/compute-unit>`).

* Write a unique non-compile time constant to each element of the array.

* Repeatedly permute and call relatively expensive math functions on our
  array elements.

* Keep the compiler from optimizing out any operations by faking a write to the
  ``ptr`` based on a run-time conditional.

This yields a total of 122 VGPRs, but it is expected this number will
depend on the exact ROCm/compiler version.

.. code-block:: asm

           .size   _Z9vgprboundiPd, .Lfunc_end1-_Z9vgprboundiPd
                                           ; -- End function
           .section        .AMDGPU.csdata
   ; Kernel info:
   ; codeLenInByte = 4732
   ; NumSgprs: 68
   ; NumVgprs: 122
   ; NumAgprs: 0
   ; <...>
   ; AccumOffset: 124

We will use various permutations of this kernel to limit occupancy, and
more importantly for the purposes of this example, demonstrate how this
is reported in ROCm Compute Profiler.

.. _vgpr-occupancy:

VGPR limited
------------

For our first test, we use the ``vgprbound`` kernel discussed in the
:ref:`design note <occupancy-experiment-design>`. After profiling, we run
the analyze step on this kernel:

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/occupancy/mi200/ -b 2.1.15 6.2 7.1.5 7.1.6 7.1.7 --dispatch 1
   <...>
   --------------------------------------------------------------------------------
   0. Top Stat
   ╒════╤═════════════════════════╤═════════╤══════════════╤══════════════╤══════════════╤════════╕
   │    │ KernelName              │   Count │      Sum(ns) │     Mean(ns) │   Median(ns) │    Pct │
   ╞════╪═════════════════════════╪═════════╪══════════════╪══════════════╪══════════════╪════════╡
   │  0 │ vgprbound(int, double*) │    1.00 │ 923093822.50 │ 923093822.50 │ 923093822.50 │ 100.00 │
   ╘════╧═════════════════════════╧═════════╧══════════════╧══════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   2. System Speed-of-Light
   2.1 Speed-of-Light
   ╒═════════╤═════════════════════╤═════════╤════════════╤═════════╤═══════════════╕
   │ Index   │ Metric              │     Avg │ Unit       │    Peak │   Pct of Peak │
   ╞═════════╪═════════════════════╪═════════╪════════════╪═════════╪═══════════════╡
   │ 2.1.15  │ Wavefront Occupancy │ 1661.24 │ Wavefronts │ 3328.00 │         49.92 │
   ╘═════════╧═════════════════════╧═════════╧════════════╧═════════╧═══════════════╛


   --------------------------------------------------------------------------------
   6. Workgroup Manager (SPI)
   6.2 Workgroup Manager - Resource Allocation
   ╒═════════╤════════════════════════════════════════╤═══════╤═══════╤═══════╤════════╕
   │ Index   │ Metric                                 │   Avg │   Min │   Max │ Unit   │
   ╞═════════╪════════════════════════════════════════╪═══════╪═══════╪═══════╪════════╡
   │ 6.2.0   │ Not-scheduled Rate (Workgroup Manager) │  0.64 │  0.64 │  0.64 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.1   │ Not-scheduled Rate (Scheduler-Pipe)    │ 24.94 │ 24.94 │ 24.94 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.2   │ Scheduler-Pipe Stall Rate              │ 24.49 │ 24.49 │ 24.49 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.3   │ Scratch Stall Rate                     │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.4   │ Insufficient SIMD Waveslots            │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.5   │ Insufficient SIMD VGPRs                │ 94.90 │ 94.90 │ 94.90 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.6   │ Insufficient SIMD SGPRs                │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.7   │ Insufficient CU LDS                    │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.8   │ Insufficient CU Barriers               │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.9   │ Reached CU Workgroup Limit             │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.10  │ Reached CU Wavefront Limit             │  0.00 │  0.00 │  0.00 │ Pct    │
   ╘═════════╧════════════════════════════════════════╧═══════╧═══════╧═══════╧════════╛


   --------------------------------------------------------------------------------
   7. Wavefront
   7.1 Wavefront Launch Stats
   ╒═════════╤══════════╤════════╤════════╤════════╤═══════════╕
   │ Index   │ Metric   │    Avg │    Min │    Max │ Unit      │
   ╞═════════╪══════════╪════════╪════════╪════════╪═══════════╡
   │ 7.1.5   │ VGPRs    │ 124.00 │ 124.00 │ 124.00 │ Registers │
   ├─────────┼──────────┼────────┼────────┼────────┼───────────┤
   │ 7.1.6   │ AGPRs    │   4.00 │   4.00 │   4.00 │ Registers │
   ├─────────┼──────────┼────────┼────────┼────────┼───────────┤
   │ 7.1.7   │ SGPRs    │  80.00 │  80.00 │  80.00 │ Registers │
   ╘═════════╧══════════╧════════╧════════╧════════╧═══════════╛

Here we see that the kernel indeed does use *around* (but not exactly)
122 VGPRs, with the difference due to granularity of VGPR allocations.
In addition, we see that we have allocated 4 “:ref:`AGPRs <desc-agprs>`”. We
note that on current CDNA2 accelerators, the ``AccumOffset`` field of
the assembly metadata:

.. code-block:: asm

   ; AccumOffset: 124

denotes the divide between ``VGPRs`` and ``AGPRs``.

Next, we examine our wavefront occupancy (**2.1.15**), and see that we are
reaching only :math:`\sim50\%` of peak occupancy. As a result, we see
that:

- We are not scheduling workgroups :math:`\sim25\%` of
  :ref:`total scheduler-pipe cycles <total-pipe-cycles>` (**6.2.1**); recall
  from the discussion of the `workgroup manager <desc-spi>`, 25% is the maximum.

- The scheduler-pipe is stalled (**6.2.2**) from scheduling workgroups due to
  resource constraints for the same :math:`\sim25\%` of the time.

- And finally, :math:`\sim91\%` of those stalls are due to a lack of SIMDs
  with the appropriate number of VGPRs available (6.2.5).

That is, the reason we can’t reach full occupancy is due to our VGPR
usage, as expected!

LDS limited
-----------

To examine an LDS limited example, we must change our kernel slightly:

.. code-block:: cpp

   constexpr size_t fully_allocate_lds = 64ul * 1024ul / sizeof(double);
   __launch_bounds__(256)
   __global__ void ldsbound(int N, double* ptr) {
       __shared__ double intermediates[fully_allocate_lds];
       for (int i = threadIdx.x ; i < fully_allocate_lds; i += blockDim.x) intermediates[i] = N * threadIdx.x;
       __syncthreads();
       double x = ptr[threadIdx.x];
       for (int i = threadIdx.x; i < fully_allocate_lds; i += blockDim.x) {
           x += sin(pow(__shfl(x, i % warpSize) * intermediates[(i - 1) % fully_allocate_lds], intermediates[i % fully_allocate_lds]));
           __syncthreads();
           intermediates[i % fully_allocate_lds] = x;
       }
       if (x == N) ptr[threadIdx.x] = x;
   }

Where we now:

* Allocate an 64 KiB LDS array per workgroup, and

* Use our allocated LDS array instead of a register array

Analyzing this:

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/occupancy/mi200/ -b 2.1.15 6.2 7.1.5 7.1.6 7.1.7 7.1.8 --dispatch 3
   <...>
   --------------------------------------------------------------------------------
   2. System Speed-of-Light
   2.1 Speed-of-Light
   ╒═════════╤═════════════════════╤════════╤════════════╤═════════╤═══════════════╕
   │ Index   │ Metric              │    Avg │ Unit       │    Peak │   Pct of Peak │
   ╞═════════╪═════════════════════╪════════╪════════════╪═════════╪═══════════════╡
   │ 2.1.15  │ Wavefront Occupancy │ 415.52 │ Wavefronts │ 3328.00 │         12.49 │
   ╘═════════╧═════════════════════╧════════╧════════════╧═════════╧═══════════════╛


   --------------------------------------------------------------------------------
   6. Workgroup Manager (SPI)
   6.2 Workgroup Manager - Resource Allocation
   ╒═════════╤════════════════════════════════════════╤═══════╤═══════╤═══════╤════════╕
   │ Index   │ Metric                                 │   Avg │   Min │   Max │ Unit   │
   ╞═════════╪════════════════════════════════════════╪═══════╪═══════╪═══════╪════════╡
   │ 6.2.0   │ Not-scheduled Rate (Workgroup Manager) │  0.13 │  0.13 │  0.13 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.1   │ Not-scheduled Rate (Scheduler-Pipe)    │ 24.87 │ 24.87 │ 24.87 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.2   │ Scheduler-Pipe Stall Rate              │ 24.84 │ 24.84 │ 24.84 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.3   │ Scratch Stall Rate                     │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.4   │ Insufficient SIMD Waveslots            │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.5   │ Insufficient SIMD VGPRs                │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.6   │ Insufficient SIMD SGPRs                │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.7   │ Insufficient CU LDS                    │ 96.47 │ 96.47 │ 96.47 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.8   │ Insufficient CU Barriers               │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.9   │ Reached CU Workgroup Limit             │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.10  │ Reached CU Wavefront Limit             │  0.00 │  0.00 │  0.00 │ Pct    │
   ╘═════════╧════════════════════════════════════════╧═══════╧═══════╧═══════╧════════╛


   --------------------------------------------------------------------------------
   7. Wavefront
   7.1 Wavefront Launch Stats
   ╒═════════╤════════════════╤══════════╤══════════╤══════════╤═══════════╕
   │ Index   │ Metric         │      Avg │      Min │      Max │ Unit      │
   ╞═════════╪════════════════╪══════════╪══════════╪══════════╪═══════════╡
   │ 7.1.5   │ VGPRs          │    96.00 │    96.00 │    96.00 │ Registers │
   ├─────────┼────────────────┼──────────┼──────────┼──────────┼───────────┤
   │ 7.1.6   │ AGPRs          │     0.00 │     0.00 │     0.00 │ Registers │
   ├─────────┼────────────────┼──────────┼──────────┼──────────┼───────────┤
   │ 7.1.7   │ SGPRs          │    80.00 │    80.00 │    80.00 │ Registers │
   ├─────────┼────────────────┼──────────┼──────────┼──────────┼───────────┤
   │ 7.1.8   │ LDS Allocation │ 65536.00 │ 65536.00 │ 65536.00 │ Bytes     │
   ╘═════════╧════════════════╧══════════╧══════════╧══════════╧═══════════╛

We see that our VGPR allocation has gone down to 96 registers, but now
we see our 64KiB LDS allocation (**7.1.8**). In addition, we see a similar
non-schedule rate (**6.2.1**) and stall rate (**6.2.2**) as in our
:ref:`VGPR example <vgpr-occupancy>`. However, our occupancy limiter has now
shifted from VGPRs (**6.2.5**) to LDS (**6.2.7**).

We note that although we see the around the same scheduler/stall rates
(with our LDS limiter), our wave occupancy (**2.1.15**) is significantly
lower (:math:`\sim12\%`)! This is important to remember: the occupancy
limiter metrics in the resource allocation section tell you what the
limiter was, but *not* how much the occupancy was limited. These metrics
should always be analyzed in concert with the wavefront occupancy
metric!

.. _sgpr-occupancy:

SGPR limited
------------

Finally, we modify our kernel once more to make it limited by
:ref:`SGPRs <desc-salu>`:

.. code-block:: cpp

   constexpr int sgprlim = 1;
   __launch_bounds__(1024, 8)
   __global__ void sgprbound(int N, double* ptr) {
       double intermediates[sgprlim];
       for (int i = 0 ; i < sgprlim; ++i) intermediates[i] = i;
       double x = ptr[0];
       #pragma unroll 1
       for (int i = 0; i < 100; ++i) {
           x += sin(pow(intermediates[(i - 1) % sgprlim], intermediates[i % sgprlim]));
           intermediates[i % sgprlim] = x;
       }
       if (x == N) ptr[0] = x;
   }

The major changes here are to: - make as much as possible provably
uniform across the wave (notice the lack of ``threadIdx.x`` in the
``intermediates`` initialization and elsewhere), - addition of
``__launch_bounds__(1024, 8)``, which reduces our maximum VGPRs to 64
(such that 8 waves can fit per SIMD), but causes some register spills
(i.e., :ref:`scratch <memory-spaces>` usage), and - lower the ``bound`` (here we
use ``sgprlim``) of the array to reduce VGPR/Scratch usage.

This results in the following assembly metadata for this kernel:

.. code-block:: asm

           .size   _Z9sgprboundiPd, .Lfunc_end3-_Z9sgprboundiPd
                                           ; -- End function
           .section        .AMDGPU.csdata
   ; Kernel info:
   ; codeLenInByte = 4872
   ; NumSgprs: 76
   ; NumVgprs: 64
   ; NumAgprs: 0
   ; TotalNumVgprs: 64
   ; ScratchSize: 60
   ; <...>
   ; AccumOffset: 64
   ; Occupancy: 8

Analyzing this workload yields:

.. code-block:: shell-session

   $ rocprof-compute analyze -p workloads/occupancy/mi200/ -b 2.1.15 6.2 7.1.5 7.1.6 7.1.7 7.1.8 7.1.9 --dispatch 5
   <...>
   --------------------------------------------------------------------------------
   0. Top Stat
   ╒════╤═════════════════════════╤═════════╤══════════════╤══════════════╤══════════════╤════════╕
   │    │ KernelName              │   Count │      Sum(ns) │     Mean(ns) │   Median(ns) │    Pct │
   ╞════╪═════════════════════════╪═════════╪══════════════╪══════════════╪══════════════╪════════╡
   │  0 │ sgprbound(int, double*) │    1.00 │ 782069812.00 │ 782069812.00 │ 782069812.00 │ 100.00 │
   ╘════╧═════════════════════════╧═════════╧══════════════╧══════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   2. System Speed-of-Light
   2.1 Speed-of-Light
   ╒═════════╤═════════════════════╤═════════╤════════════╤═════════╤═══════════════╕
   │ Index   │ Metric              │     Avg │ Unit       │    Peak │   Pct of Peak │
   ╞═════════╪═════════════════════╪═════════╪════════════╪═════════╪═══════════════╡
   │ 2.1.15  │ Wavefront Occupancy │ 3291.76 │ Wavefronts │ 3328.00 │         98.91 │
   ╘═════════╧═════════════════════╧═════════╧════════════╧═════════╧═══════════════╛


   --------------------------------------------------------------------------------
   6. Workgroup Manager (SPI)
   6.2 Workgroup Manager - Resource Allocation
   ╒═════════╤════════════════════════════════════════╤═══════╤═══════╤═══════╤════════╕
   │ Index   │ Metric                                 │   Avg │   Min │   Max │ Unit   │
   ╞═════════╪════════════════════════════════════════╪═══════╪═══════╪═══════╪════════╡
   │ 6.2.0   │ Not-scheduled Rate (Workgroup Manager) │  7.72 │  7.72 │  7.72 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.1   │ Not-scheduled Rate (Scheduler-Pipe)    │ 15.17 │ 15.17 │ 15.17 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.2   │ Scheduler-Pipe Stall Rate              │  7.38 │  7.38 │  7.38 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.3   │ Scratch Stall Rate                     │ 39.76 │ 39.76 │ 39.76 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.4   │ Insufficient SIMD Waveslots            │ 26.32 │ 26.32 │ 26.32 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.5   │ Insufficient SIMD VGPRs                │ 26.32 │ 26.32 │ 26.32 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.6   │ Insufficient SIMD SGPRs                │ 25.52 │ 25.52 │ 25.52 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.7   │ Insufficient CU LDS                    │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.8   │ Insufficient CU Barriers               │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.9   │ Reached CU Workgroup Limit             │  0.00 │  0.00 │  0.00 │ Pct    │
   ├─────────┼────────────────────────────────────────┼───────┼───────┼───────┼────────┤
   │ 6.2.10  │ Reached CU Wavefront Limit             │  0.00 │  0.00 │  0.00 │ Pct    │
   ╘═════════╧════════════════════════════════════════╧═══════╧═══════╧═══════╧════════╛


   --------------------------------------------------------------------------------
   7. Wavefront
   7.1 Wavefront Launch Stats
   ╒═════════╤════════════════════╤═══════╤═══════╤═══════╤════════════════╕
   │ Index   │ Metric             │   Avg │   Min │   Max │ Unit           │
   ╞═════════╪════════════════════╪═══════╪═══════╪═══════╪════════════════╡
   │ 7.1.5   │ VGPRs              │ 64.00 │ 64.00 │ 64.00 │ Registers      │
   ├─────────┼────────────────────┼───────┼───────┼───────┼────────────────┤
   │ 7.1.6   │ AGPRs              │  0.00 │  0.00 │  0.00 │ Registers      │
   ├─────────┼────────────────────┼───────┼───────┼───────┼────────────────┤
   │ 7.1.7   │ SGPRs              │ 80.00 │ 80.00 │ 80.00 │ Registers      │
   ├─────────┼────────────────────┼───────┼───────┼───────┼────────────────┤
   │ 7.1.8   │ LDS Allocation     │  0.00 │  0.00 │  0.00 │ Bytes          │
   ├─────────┼────────────────────┼───────┼───────┼───────┼────────────────┤
   │ 7.1.9   │ Scratch Allocation │ 60.00 │ 60.00 │ 60.00 │ Bytes/workitem │
   ╘═════════╧════════════════════╧═══════╧═══════╧═══════╧════════════════╛

Here we see that our wavefront launch stats (**7.1**) have changed to
reflect the metadata seen in the ``--save-temps`` output. Of particular
interest, we see:

* The SGPR allocation (**7.1.7**) is 80 registers, slightly more than the 76
  requested by the compiler due to allocation granularity, and

* We have a :ref:`"scratch" <memory-spaces>`, that is, private memory,
  allocation of 60 bytes per work-item.

Analyzing the resource allocation block (**6.2**) we now see that for the
first time, the "Not-scheduled Rate (Workgroup Manager)" metric (**6.2.0**)
has become non-zero. This is because the workgroup manager is
responsible for management of scratch, which we see also contributes to
our occupancy limiters in the "Scratch Stall Rate" (**6.2.3**). Note that
the sum of the workgroup manager not-scheduled rate and the
scheduler-pipe non-scheduled rate is still :math:`\sim25\%`, as in our
previous examples.

Next, we see that the scheduler-pipe stall rate (**6.2.2**), that is, how often
we could not schedule a workgroup to a CU, was only about
:math:`\sim8\%`. This hints that perhaps, our kernel is not
*particularly* occupancy limited by resources. Indeed, checking the
wave occupancy metric (**2.1.15**) shows that this kernel is reaching nearly
99% occupancy.

Finally, we inspect the occupancy limiter metrics and see a roughly even
split between :ref:`waveslots <desc-valu>` (**6.2.4**), :ref:`VGPRs <desc-valu>`
(**6.2.5**), and :ref:`SGPRs <desc-salu>` (**6.2.6**) along with the scratch stalls
(**6.2.3**) previously mentioned.

This is yet another reminder to view occupancy holistically. While these
metrics tell you why a workgroup cannot be scheduled, they do *not* tell
you what your occupancy was (consult wavefront occupancy) *nor* whether
increasing occupancy will be beneficial to performance.
