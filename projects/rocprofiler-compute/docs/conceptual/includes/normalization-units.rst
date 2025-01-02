.. _normalization-units:

Normalization units
===================

A user-configurable unit by which you can choose to normalize data. Options
include:

.. list-table::
   :header-rows: 1

   * - Name
     - Description

   * - ``per_wave``
     - The total value of the measured counter or metric that occurred per
       kernel invocation divided by the total number of
       :ref:`wavefronts <desc-wavefront>` launched in the kernel.

   * - ``per_cycle``
     - The total value of the measured counter or metric that occurred per
       kernel invocation divided by the
       :ref:`kernel cycles <kernel-cycles>`, that is, the total number of
       cycles the kernel executed as measured by the
       :doc:`command processor <command-processor>`.

   * - ``per_kernel``
     - The total value of the measured counter or metric that occurred per
       kernel invocation.

   * - ``per_second``
     - The total value of the measured counter or metric that occurred per
       kernel invocation divided by the :ref:`kernel time <kernel-time>`,
       that is, the total runtime of the kernel in seconds, as measured by the
       :doc:`command processor <command-processor>`.

By default, ROCm Compute Profiler uses the ``per_wave`` normalization.

.. tip::

   The best normalization may vary depending on your use case. For instance, a
   ``per_second`` normalization might be useful for FLOP or bandwidth
   comparisons, while a ``per_wave`` normalization could be useful to see how many
   (and what types) of instructions are used per wavefront. A ``per_kernel``
   normalization can be useful to get the total aggregate values of metrics for
   comparison between different configurations.
