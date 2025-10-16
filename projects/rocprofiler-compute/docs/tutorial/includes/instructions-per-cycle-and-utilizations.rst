.. _ipc-example:

Instructions-per-cycle and utilizations example
===============================================

For this example, consider the
:dev-sample:`instructions-per-cycle (IPC) example <ipc.hip>` included with
ROCm Compute Profiler.

This example is compiled using ``c++17`` support:

.. code-block:: shell

   $ hipcc -O3 ipc.hip -o ipc -std=c++17

and was run on an MI250 CDNA2 accelerator:

.. code-block:: shell

   $ rocprof-compute profile -n ipc --no-roof -- ./ipc

The results shown in this section are *generally* applicable to CDNA
accelerators, but may vary between generations and specific products.

.. _ipc-experiment-design-note:

Design note
-----------

The kernels in this example all execute a specific assembly operation
``N`` times (1000, by default), for instance the ``vmov`` kernel:

.. code-block:: cpp

   template<int N=1000>
   __device__ void vmov_op() {
       int dummy;
       if constexpr (N >= 1) {
           asm volatile("v_mov_b32 v0, v1\n" : : "{v31}"(dummy));
           vmov_op<N - 1>();
       }
   }

   template<int N=1000>
   __global__ void vmov() {
       vmov_op<N>();
   }

The kernels are then launched twice, once for a warm-up run, and once
for measurement.

.. _ipc-valu-utilization:

VALU utilization and IPC
------------------------

Now we can use our test to measure the achieved instructions-per-cycle
of various types of instructions. We start with a simple :ref:`VALU <desc-valu>`
operation, i.e., a ``v_mov_b32`` instruction, e.g.:

.. code-block:: asm

   v_mov_b32 v0, v1

This instruction simply copies the contents from the source register
(``v1``) to the destination register (``v0``). Investigating this kernel
with ROCm Compute Profiler, we see:

.. code-block:: shell-session

   $ rocprof-compute analyze -p workloads/ipc/mi200/ --dispatch 7 -b 11.2
   <...>
   --------------------------------------------------------------------------------
   0. Top Stat
   ╒════╤═══════════════════════════════╤═════════╤═════════════╤═════════════╤══════════════╤════════╕
   │    │ KernelName                    │   Count │     Sum(ns) │    Mean(ns) │   Median(ns) │    Pct │
   ╞════╪═══════════════════════════════╪═════════╪═════════════╪═════════════╪══════════════╪════════╡
   │  0 │ void vmov<1000>() [clone .kd] │    1.00 │ 99317423.00 │ 99317423.00 │  99317423.00 │ 100.00 │
   ╘════╧═══════════════════════════════╧═════════╧═════════════╧═════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   11. Compute Units - Compute Pipeline
   11.2 Pipeline Stats
   ╒═════════╤═════════════════════╤═══════╤═══════╤═══════╤══════════════╕
   │ Index   │ Metric              │ Avg   │ Min   │ Max   │ Unit         │
   ╞═════════╪═════════════════════╪═══════╪═══════╪═══════╪══════════════╡
   │ 11.2.0  │ IPC                 │ 1.0   │ 1.0   │ 1.0   │ Instr/cycle  │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.1  │ IPC (Issued)        │ 1.0   │ 1.0   │ 1.0   │ Instr/cycle  │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.2  │ SALU Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.3  │ VALU Util           │ 99.98 │ 99.98 │ 99.98 │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.4  │ VMEM Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.5  │ Branch Util         │ 0.1   │ 0.1   │ 0.1   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.6  │ VALU Active Threads │ 64.0  │ 64.0  │ 64.0  │ Threads      │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.7  │ MFMA Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.8  │ MFMA Instr Cycles   │       │       │       │ Cycles/instr │
   ╘═════════╧═════════════════════╧═══════╧═══════╧═══════╧══════════════╛

Here we see that:

1. Both the IPC (**11.2.0**) and “Issued” IPC (**11.2.1**) metrics are
   :math:`\sim 1`
2. The VALU Utilization metric (**11.2.3**) is also :math:`\sim100\%`, and
   finally
3. The VALU Active Threads metric (**11.2.4**) is 64, i.e., the wavefront
   size on CDNA accelerators, as all threads in the wavefront are
   active.

We will explore the difference between the IPC (**11.2.0**) and “Issued” IPC
(**11.2.1**) metrics in the :ref:`next section <issued-ipc>`.

Additionally, we notice a small (0.1%) Branch utilization (**11.2.5**).
Inspecting the assembly of this kernel shows there are no branch
operations, however recalling the note in the :ref:`Pipeline
statistics <pipeline-stats>` section:

 The branch utilization <…> includes time spent in other instruction
 types (namely: ``s_endpgm``) that are *typically* a very small
 percentage of the overall kernel execution.

We see that this is coming from execution of the ``s_endpgm``
instruction at the end of every wavefront.

.. note::

   Technically, the cycle counts used in the denominators of our IPC metrics are
   actually in units of quad-cycles, a group of 4 consecutive cycles. However, a
   typical :ref:`VALU <desc-valu>` instruction on CDNA accelerators runs for a
   single quad-cycle (see :gcn-crash-course:`30`). Therefore, for simplicity, we
   simply report these metrics as "instructions per cycle".

.. _issued-ipc:

Exploring “issued” IPC via MFMA operations
------------------------------------------

.. warning::

   The MFMA assembly operations used in this example are inherently not portable
   to older CDNA architectures.

Unlike the simple quad-cycle ``v_mov_b32`` operation discussed in our
:ref:`previous example <ipc-valu-utilization>`, some operations take many
quad-cycles to execute. For example, using the
`AMD Matrix Instruction Calculator <https://github.com/RadeonOpenCompute/amd_matrix_instruction_calculator#example-of-querying-instruction-information>`_
we can see that some :ref:`MFMA <desc-mfma>` operations take 64 cycles, e.g.:

.. code-block:: shell

   $ ./matrix_calculator.py --arch CDNA2 --detail-instruction --instruction v_mfma_f32_32x32x8bf16_1k
   Architecture: CDNA2
   Instruction: V_MFMA_F32_32X32X8BF16_1K
   <...>
       Execution statistics:
           FLOPs: 16384
           Execution cycles: 64
           FLOPs/CU/cycle: 1024
           Can co-execute with VALU: True
           VALU co-execution cycles possible: 60

What happens to our IPC when we utilize this ``v_mfma_f32_32x32x8bf16_1k``
instruction on a CDNA2 accelerator? To find out, we turn to our ``mfma`` kernel
in the IPC example:

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/ipc/mi200/ --dispatch 8 -b 11.2 --decimal 4
   <...>
   --------------------------------------------------------------------------------
   0. Top Stat
   ╒════╤═══════════════════════════════╤═════════╤═════════════════╤═════════════════╤═════════════════╤══════════╕
   │    │ KernelName                    │   Count │         Sum(ns) │        Mean(ns) │      Median(ns) │      Pct │
   ╞════╪═══════════════════════════════╪═════════╪═════════════════╪═════════════════╪═════════════════╪══════════╡
   │  0 │ void mfma<1000>() [clone .kd] │  1.0000 │ 1623167595.0000 │ 1623167595.0000 │ 1623167595.0000 │ 100.0000 │
   ╘════╧═══════════════════════════════╧═════════╧═════════════════╧═════════════════╧═════════════════╧══════════╛


   --------------------------------------------------------------------------------
   11. Compute Units - Compute Pipeline
   11.2 Pipeline Stats
   ╒═════════╤═════════════════════╤═════════╤═════════╤═════════╤══════════════╕
   │ Index   │ Metric              │     Avg │     Min │     Max │ Unit         │
   ╞═════════╪═════════════════════╪═════════╪═════════╪═════════╪══════════════╡
   │ 11.2.0  │ IPC                 │  0.0626 │  0.0626 │  0.0626 │ Instr/cycle  │
   ├─────────┼─────────────────────┼─────────┼─────────┼─────────┼──────────────┤
   │ 11.2.1  │ IPC (Issued)        │  1.0000 │  1.0000 │  1.0000 │ Instr/cycle  │
   ├─────────┼─────────────────────┼─────────┼─────────┼─────────┼──────────────┤
   │ 11.2.2  │ SALU Util           │  0.0000 │  0.0000 │  0.0000 │ Pct          │
   ├─────────┼─────────────────────┼─────────┼─────────┼─────────┼──────────────┤
   │ 11.2.3  │ VALU Util           │  6.2496 │  6.2496 │  6.2496 │ Pct          │
   ├─────────┼─────────────────────┼─────────┼─────────┼─────────┼──────────────┤
   │ 11.2.4  │ VMEM Util           │  0.0000 │  0.0000 │  0.0000 │ Pct          │
   ├─────────┼─────────────────────┼─────────┼─────────┼─────────┼──────────────┤
   │ 11.2.5  │ Branch Util         │  0.0062 │  0.0062 │  0.0062 │ Pct          │
   ├─────────┼─────────────────────┼─────────┼─────────┼─────────┼──────────────┤
   │ 11.2.6  │ VALU Active Threads │ 64.0000 │ 64.0000 │ 64.0000 │ Threads      │
   ├─────────┼─────────────────────┼─────────┼─────────┼─────────┼──────────────┤
   │ 11.2.7  │ MFMA Util           │ 99.9939 │ 99.9939 │ 99.9939 │ Pct          │
   ├─────────┼─────────────────────┼─────────┼─────────┼─────────┼──────────────┤
   │ 11.2.8  │ MFMA Instr Cycles   │ 64.0000 │ 64.0000 │ 64.0000 │ Cycles/instr │
   ╘═════════╧═════════════════════╧═════════╧═════════╧═════════╧══════════════╛

In contrast to our :ref:`VALU IPC example <ipc-valu-utilization>`, we now see
that the IPC metric (**11.2.0**) and Issued IPC (**11.2.1**) metric differ
substantially. First, we see the VALU utilization (**11.2.3**) has decreased
substantially, from nearly 100% to :math:`\sim6.25\%`. We note that this matches
the ratio of: :math:`((Execution\ cycles) - (VALU\ coexecution\ cycles)) / (Execution\ cycles)`
reported by the matrix calculator, while the MFMA utilization (**11.2.7**)
has increased to nearly 100%.

Recall that our ``v_mfma_f32_32x32x8bf16_1k`` instruction takes 64 cycles to
execute, or 16 quad-cycles, matching our observed MFMA Instruction
Cycles (**11.2.8**). That is, we have a single instruction executed every 16
quad-cycles, or :math:`1/16 = 0.0625`, which is almost identical to our IPC
metric (**11.2.0**). Why then is the Issued IPC metric (**11.2.1**) equal to 1.0?

Instead of simply counting the number of instructions issued and
dividing by the number of cycles the :doc:`CUs </conceptual/compute-unit>` on
the accelerator were active (as is done for **11.2.0**), this metric is formulated
differently, and instead counts the number of
(non-:ref:`internal <ipc-internal-instructions>`) instructions issued divided
by the number of (quad-) cycles where the :ref:`scheduler <desc-scheduler>` was
actively working on issuing instructions. Thus the Issued IPC metric
(**11.2.1**) gives more of a sense of “what percent of the total number of
:ref:`scheduler <desc-scheduler>` cycles did a wave schedule an instruction?”
while the IPC metric (**11.2.0**) indicates the ratio of the number of
instructions executed over the total
:ref:`active CU cycles <total-active-cu-cycles>`.

.. warning::

   There are further complications of the Issued IPC metric (**11.2.1**) that make
   its use more complicated. We will be explore that in the
   :ref:`following section <ipc-internal-instructions>`. For these reasons,
   ROCm Compute Profiler typically promotes use of the regular IPC metric (**11.2.0**), e.g., in
   the top-level Speed-of-Light chart.

.. _ipc-internal-instructions:

Internal instructions and IPC
-----------------------------

Next, we explore the concept of an “internal” instruction. From
:gcn-crash-course:`29`, we see a few candidates for internal instructions, and
we choose a ``s_nop`` instruction, which according to the
:mi200-isa-pdf:`CDNA2 ISA guide <>`:

 Does nothing; it can be repeated in hardware up to eight times.

Here we choose to use the following no-op to make our point:

.. code-block:: asm

   s_nop 0x0

Running this kernel through ROCm Compute Profiler yields:

.. code-block:: shell-session

   $ rocprof-compute analyze -p workloads/ipc/mi200/ --dispatch 9 -b 11.2
   <...>
   --------------------------------------------------------------------------------
   0. Top Stat
   ╒════╤═══════════════════════════════╤═════════╤═════════════╤═════════════╤══════════════╤════════╕
   │    │ KernelName                    │   Count │     Sum(ns) │    Mean(ns) │   Median(ns) │    Pct │
   ╞════╪═══════════════════════════════╪═════════╪═════════════╪═════════════╪══════════════╪════════╡
   │  0 │ void snop<1000>() [clone .kd] │    1.00 │ 14221851.50 │ 14221851.50 │  14221851.50 │ 100.00 │
   ╘════╧═══════════════════════════════╧═════════╧═════════════╧═════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   11. Compute Units - Compute Pipeline
   11.2 Pipeline Stats
   ╒═════════╤═════════════════════╤═══════╤═══════╤═══════╤══════════════╕
   │ Index   │ Metric              │ Avg   │ Min   │ Max   │ Unit         │
   ╞═════════╪═════════════════════╪═══════╪═══════╪═══════╪══════════════╡
   │ 11.2.0  │ IPC                 │ 6.79  │ 6.79  │ 6.79  │ Instr/cycle  │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.1  │ IPC (Issued)        │ 1.0   │ 1.0   │ 1.0   │ Instr/cycle  │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.2  │ SALU Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.3  │ VALU Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.4  │ VMEM Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.5  │ Branch Util         │ 0.68  │ 0.68  │ 0.68  │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.6  │ VALU Active Threads │       │       │       │ Threads      │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.7  │ MFMA Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.8  │ MFMA Instr Cycles   │       │       │       │ Cycles/instr │
   ╘═════════╧═════════════════════╧═══════╧═══════╧═══════╧══════════════╛

First, we see that the IPC metric (**11.2.0**) tops our theoretical maximum
of 5 instructions per cycle (discussed in the :ref:`scheduler <desc-scheduler>`
section). How can this be?

Recall that :gcn-crash-course:`27` say “no functional unit” for the internal
instructions. This removes the limitation on the IPC. If we are *only*
issuing internal instructions, we are not issuing to any execution
units! However, workloads such as these are almost *entirely* artificial
(that is, repeatedly issuing internal instructions almost exclusively). In
practice, a maximum of IPC of 5 is expected in almost all cases.

Secondly, note that our “Issued” IPC (**11.2.1**) is still identical to
the one here. Again, this has to do with the details of “internal”
instructions. Recall in our :ref:`previous example <issued-ipc>` we defined
this metric as explicitly excluding internal instruction counts. The
logical question then is, "what *is* this metric counting in our
``s_nop`` kernel?"

The generated assembly looks something like:

.. code-block:: asm

   ;;#ASMSTART
   s_nop 0x0
   ;;#ASMEND
   ;;#ASMSTART
   s_nop 0x0
   ;;#ASMEND
   ;;<... omitting many more ...>
   s_endpgm
   .section        .rodata,#alloc
   .p2align        6, 0x0
   .amdhsa_kernel _Z4snopILi1000EEvv

Of particular interest here is the ``s_endpgm`` instruction, of which
the `CDNA2 ISA
guide <https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf>`__
states:

   End of program; terminate wavefront.

This is not on our list of internal instructions from
:gcn-crash-course:`The AMD GCN Architecture <>`, and is therefore counted as part
of our Issued IPC (**11.2.1**). Thus, the issued IPC being equal to one here
indicates that we issued an ``s_endpgm`` instruction every cycle the
:ref:`scheduler <desc-scheduler>` was active for non-internal instructions, which
is expected as this was our *only* non-internal instruction.

SALU Utilization
----------------

Next, we explore a simple :ref:`SALU <desc-salu>` kernel in our on-going IPC and
utilization example. For this case, we select a simple scalar move
operation, for instance:

.. code-block:: asm

   s_mov_b32 s0, s1

which, in analogue to our :ref:`v_mov <ipc-valu-utilization>` example, copies the
contents of the source scalar register (``s1``) to the destination
scalar register (``s0``). Running this kernel through ROCm Compute Profiler yields:

.. code-block:: shell-session

   $ rocprof-compute analyze -p workloads/ipc/mi200/ --dispatch 10 -b 11.2
   <...>
   --------------------------------------------------------------------------------
   0. Top Stat
   ╒════╤═══════════════════════════════╤═════════╤═════════════╤═════════════╤══════════════╤════════╕
   │    │ KernelName                    │   Count │     Sum(ns) │    Mean(ns) │   Median(ns) │    Pct │
   ╞════╪═══════════════════════════════╪═════════╪═════════════╪═════════════╪══════════════╪════════╡
   │  0 │ void smov<1000>() [clone .kd] │    1.00 │ 96246554.00 │ 96246554.00 │  96246554.00 │ 100.00 │
   ╘════╧═══════════════════════════════╧═════════╧═════════════╧═════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   11. Compute Units - Compute Pipeline
   11.2 Pipeline Stats
   ╒═════════╤═════════════════════╤═══════╤═══════╤═══════╤══════════════╕
   │ Index   │ Metric              │ Avg   │ Min   │ Max   │ Unit         │
   ╞═════════╪═════════════════════╪═══════╪═══════╪═══════╪══════════════╡
   │ 11.2.0  │ IPC                 │ 1.0   │ 1.0   │ 1.0   │ Instr/cycle  │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.1  │ IPC (Issued)        │ 1.0   │ 1.0   │ 1.0   │ Instr/cycle  │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.2  │ SALU Util           │ 99.98 │ 99.98 │ 99.98 │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.3  │ VALU Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.4  │ VMEM Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.5  │ Branch Util         │ 0.1   │ 0.1   │ 0.1   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.6  │ VALU Active Threads │       │       │       │ Threads      │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.7  │ MFMA Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.8  │ MFMA Instr Cycles   │       │       │       │ Cycles/instr │
   ╘═════════╧═════════════════════╧═══════╧═══════╧═══════╧══════════════╛

Here we see that:

- Both our IPC (**11.2.0**) and Issued IPC (**11.2.1**) are
  :math:`\sim1.0` as expected, and

- The SALU Utilization (**11.2.2**) was
  nearly 100% as it was active for almost the entire kernel.

VALU Active Threads
-------------------

For our final IPC/Utilization example, we consider a slight modification
of our :ref:`v_mov <ipc-valu-utilization>` example:

.. code-block:: cpp

   template<int N=1000>
   __global__ void vmov_with_divergence() {
       if (threadIdx.x % 64 == 0)
           vmov_op<N>();
   }

That is, we wrap our :ref:`VALU <desc-valu>` operation inside a conditional
where only one lane in our wavefront is active. Running this kernel
through ROCm Compute Profiler yields:

.. code-block:: shell-session

   $ rocprof-compute analyze -p workloads/ipc/mi200/ --dispatch 11 -b 11.2
   <...>
   --------------------------------------------------------------------------------
   0. Top Stat
   ╒════╤══════════════════════════════════════════╤═════════╤═════════════╤═════════════╤══════════════╤════════╕
   │    │ KernelName                               │   Count │     Sum(ns) │    Mean(ns) │   Median(ns) │    Pct │
   ╞════╪══════════════════════════════════════════╪═════════╪═════════════╪═════════════╪══════════════╪════════╡
   │  0 │ void vmov_with_divergence<1000>() [clone │    1.00 │ 97125097.00 │ 97125097.00 │  97125097.00 │ 100.00 │
   │    │  .kd]                                    │         │             │             │              │        │
   ╘════╧══════════════════════════════════════════╧═════════╧═════════════╧═════════════╧══════════════╧════════╛


   --------------------------------------------------------------------------------
   11. Compute Units - Compute Pipeline
   11.2 Pipeline Stats
   ╒═════════╤═════════════════════╤═══════╤═══════╤═══════╤══════════════╕
   │ Index   │ Metric              │ Avg   │ Min   │ Max   │ Unit         │
   ╞═════════╪═════════════════════╪═══════╪═══════╪═══════╪══════════════╡
   │ 11.2.0  │ IPC                 │ 1.0   │ 1.0   │ 1.0   │ Instr/cycle  │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.1  │ IPC (Issued)        │ 1.0   │ 1.0   │ 1.0   │ Instr/cycle  │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.2  │ SALU Util           │ 0.1   │ 0.1   │ 0.1   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.3  │ VALU Util           │ 99.98 │ 99.98 │ 99.98 │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.4  │ VMEM Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.5  │ Branch Util         │ 0.2   │ 0.2   │ 0.2   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.6  │ VALU Active Threads │ 1.13  │ 1.13  │ 1.13  │ Threads      │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.7  │ MFMA Util           │ 0.0   │ 0.0   │ 0.0   │ Pct          │
   ├─────────┼─────────────────────┼───────┼───────┼───────┼──────────────┤
   │ 11.2.8  │ MFMA Instr Cycles   │       │       │       │ Cycles/instr │
   ╘═════════╧═════════════════════╧═══════╧═══════╧═══════╧══════════════╛

Here we see that once again, our VALU Utilization (**11.2.3**) is nearly
100%. However, we note that the VALU Active Threads metric (**11.2.6**) is
:math:`\sim 1`, which matches our conditional in the source code. So
VALU Active Threads reports the average number of lanes of our wavefront
that are active over all :ref:`VALU <desc-valu>` instructions, or thread
“convergence” (i.e., 1 - :ref:`divergence <desc-divergence>`).

.. note::

   1. The act of evaluating a vector conditional in this example typically triggers VALU operations, contributing to why the VALU Active Threads metric is not identically one.
   2. This metric is a time (cycle) averaged value, and thus contains an implicit dependence on the duration of various VALU instructions.

   Nonetheless, this metric serves as a useful measure of thread-convergence.

Finally, we note that our branch utilization (**11.2.5**) has increased
slightly from our baseline, as we now have a branch (checking the value
of ``threadIdx.x``).
