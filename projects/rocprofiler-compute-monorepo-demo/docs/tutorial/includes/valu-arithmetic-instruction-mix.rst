.. _valu-arith-instruction-mix-ex:

VALU arithmetic instruction mix
===============================

 For this example, consider the
 :dev-sample:`instruction mix sample <instmix.hip>` distributed as a part
 of ROCm Compute Profiler.

.. note::

   The examples in the section are expected to work on all CDNA™ accelerators.
   However, the actual experiment results in this section were collected on an
   :ref:`MI2XX <mixxx-note>` accelerator.

.. _valu-experiment-design:

Design note
-----------

This code uses a number of inline assembly instructions to cleanly
identify the types of instructions being issued, as well as to avoid
optimization / dead-code elimination by the compiler. While inline
assembly is inherently not portable, this example is expected to work on
all GCN™ GPUs and CDNA accelerators.

We reproduce a sample of the kernel as follows:

.. code-block:: cpp

   // fp32: add, mul, transcendental and fma
   float f1, f2;
   asm volatile(
       "v_add_f32_e32 %0, %1, %0\n"
       "v_mul_f32_e32 %0, %1, %0\n"
       "v_sqrt_f32 %0, %1\n"
       "v_fma_f32 %0, %1, %0, %1\n"
       : "=v"(f1)
       : "v"(f2));

These instructions correspond to:

* A 32-bit floating point addition,

* a 32-bit floating point multiplication,

* a 32-bit floating point square-root transcendental operation, and

* a 32-bit floating point fused multiply-add operation.

For more detail, refer to the `CDNA2 ISA
Guide <https://www.amd.com/system/files/TechDocs/instinct-mi200-cdna2-instruction-set-architecture.pdf>`__.

Instruction mix
^^^^^^^^^^^^^^^

 This example was compiled and run on a MI250 accelerator using ROCm
 v5.6.0, and ROCm Compute Profiler v2.0.0.

.. code-block:: shell

   $ hipcc -O3 instmix.hip -o instmix

Generate the profile for this example using the following command.

.. code-block:: shell

   $ rocprof-compute profile -n instmix --no-roof -- ./instmix

Analyze the instruction mix section.

.. code-block:: shell

   $ rocprof-compute analyze -p workloads/instmix/mi200/ -b 10.2
   <...>
   10. Compute Units - Instruction Mix
   10.2 VALU Arithmetic Instr Mix
   ╒═════════╤════════════╤═════════╤════════════════╕
   │ Index   │ Metric     │   Count │ Unit           │
   ╞═════════╪════════════╪═════════╪════════════════╡
   │ 10.2.0  │ INT32      │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.1  │ INT64      │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.2  │ F16-ADD    │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.3  │ F16-MUL    │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.4  │ F16-FMA    │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.5  │ F16-Trans  │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.6  │ F32-ADD    │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.7  │ F32-MUL    │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.8  │ F32-FMA    │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.9  │ F32-Trans  │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.10 │ F64-ADD    │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.11 │ F64-MUL    │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.12 │ F64-FMA    │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.13 │ F64-Trans  │    1.00 │ Instr per wave │
   ├─────────┼────────────┼─────────┼────────────────┤
   │ 10.2.14 │ Conversion │    1.00 │ Instr per wave │
   ╘═════════╧════════════╧═════════╧════════════════╛

This shows that we have exactly one of each type of VALU arithmetic instruction
by construction.
