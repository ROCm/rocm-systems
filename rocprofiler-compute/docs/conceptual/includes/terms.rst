.. _desc-workgroup:

.. _desc-work-item:

.. _desc-wavefront:

.. _desc-divergence:

.. _kernel-time:

.. _kernel-cycles:

.. _total-active-cu-cycles:

.. _total-cu-cycles:

.. _total-se-cycles:

.. _total-simd-cycles:

.. _total-pipe-cycles:

.. _total-l1i-cycles:

.. _total-active-l2-cycles:

.. _total-l2-cycles:

.. _total-sl1d-cycles:

.. _thread-requests:

.. list-table::
   :header-rows: 1

   * - Name

     - Description

     - Unit

   * - Kernel time

     - The number of seconds the accelerator was executing a kernel, from the
       :doc:`command processor <command-processor>`'s (CP) start-of-kernel
       timestamp (a number of cycles after the CP beings processing the packet)
       to the CP's end-of-kernel timestamp (a number of cycles before the CP
       stops processing the packet).

     - Seconds

   * - Kernel cycles

     - The number of cycles the accelerator was active doing *any* work, as
       measured by the :doc:`command processor <command-processor>` (CP).

     - Cycles

   * - Total CU cycles

     - The number of cycles the accelerator was active doing *any* work
       (that is, kernel cycles), multiplied by the number of
       :doc:`compute units <compute-unit>` on the accelerator. A
       measure of the total possible active cycles the compute units could be
       doing work, useful for the normalization of metrics inside the CU.

     - Cycles

   * - Total active CU cycles

     - The number of cycles a CU on the accelerator was active doing *any*
       work, summed over all :doc:`compute units <compute-unit>` on the
       accelerator.

     - Cycles

   * - Total SIMD cycles

     - The number of cycles the accelerator was active doing *any* work (that
       is, kernel cycles), multiplied by the number of
       :doc:`SIMDs <compute-unit>` on the accelerator. A measure of the
       total possible active cycles the SIMDs could be doing work, useful for
       the normalization of metrics inside the CU.

     - Cycles

   * - Total L2 cycles

     - The number of cycles the accelerator was active doing *any* work (that
       is, kernel cycles), multiplied by the number of :doc:`L2 <l2-cache>`
       channels on the accelerator. A measure of the total possible active
       cycles the L2 channels could be doing work, useful for the normalization
       of metrics inside the L2.

     - Cycles

   * - Total active L2 cycles

     - The number of cycles a channel of the L2 cache was active doing *any*
       work, summed over all :doc:`L2 <l2-cache>` channels on the accelerator.

     - Cycles

   * - Total sL1D cycles

     - The number of cycles the accelerator was active doing *any* work (that
       is, kernel cycles), multiplied by the number of
       :ref:`scalar L1 data caches <desc-sl1d>` on the accelerator. A measure of
       the total possible active cycles the sL1Ds could be doing work, useful
       for the normalization of metrics inside the sL1D.

     - Cycles

   * - Total L1I cycles

     - The number of cycles the accelerator was active doing *any* work (that
       is, kernel cycles), multiplied by the number of
       :ref:`L1 instruction caches <desc-l1i>` (L1I) on the accelerator. A
       measure of the total possible active cycles the L1Is could be doing
       work, useful for the normalization of metrics inside the L1I.

     - Cycles

   * - Total scheduler-pipe cycles

     - The number of cycles the accelerator was active doing *any* work (that
       is, kernel cycles), multiplied by the number of
       :doc:`scheduler pipes <command-processor>` on the accelerator. A measure
       of the total possible active cycles the scheduler-pipes could be doing
       work, useful for the normalization of metrics inside the
       :ref:`workgroup manager <desc-spi>` and
       :doc:`command processor <command-processor>`.

     - Cycles

   * - Total shader-engine cycles

     - The total number of cycles the accelerator was active doing *any* work,
       multiplied by the number of :doc:`shader engines <shader-engine>` on the
       accelerator. A measure of the total possible active cycles the shader
       engines could be doing work, useful for the normalization of
       metrics inside the :ref:`workgroup manager <desc-spi>`.

     - Cycles

   * - Thread-requests

     - The number of unique memory addresses accessed by a single memory
       instruction. On AMD Instinct accelerators, this has a maximum of 64
       (that is, the size of the :ref:`wavefront <wavefront>`).

     - Addresses

   * - Work-item

     - A single *thread*, or lane, of execution that executes in lockstep with
       the rest of the work-items comprising a :ref:`wavefront <wavefront>`
       of execution.

     - N/A

   * - Wavefront

     - A group of work-items, or threads, that execute in lockstep on the
       :doc:`compute unit <compute-unit>`. On AMD Instinct accelerators, the
       wavefront size is always 64 work-items.

     - N/A

   * - Workgroup

     - A group of wavefronts that execute on the same
       :doc:`compute unit <compute-unit>`, and can cooperatively execute and
       share data via the use of synchronization primitives,
       :doc:`LDS <local-data-share>`, atomics, and others.

     - N/A

   * - Divergence

     - Divergence within a wavefront occurs when not all work-items are active
       when executing an instruction, that is, due to non-uniform control flow
       within a wavefront. Can reduce execution efficiency by causing,
       for instance, the :ref:`VALU <desc-valu>` to need to execute both
       branches of a conditional with different sets of work-items active.

     - N/A
