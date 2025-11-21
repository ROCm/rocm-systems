.. meta::
   :description: ROCm Compute Profiler performance model: Local data share (LDS)
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, local, data, share, LDS

**********************
Local data share (LDS)
**********************

.. _lds-sol:

LDS Speed-of-Light
==================

.. warning::

   The theoretical maximum throughput for some metrics in this section are
   currently computed with the maximum achievable clock frequency, as reported
   by ``rocminfo``, for an accelerator. This may not be realistic for all
   workloads.

The :ref:`LDS <desc-lds>` speed-of-light chart shows a number of key metrics for
the LDS as a comparison with the peak achievable values of those metrics.

.. jinja:: lds-sol
   :file: _templates/metrics_table.j2

.. rubric:: Footnotes

.. [#lds-workload] Here we assume the typical case where the workload evenly distributes
   LDS operations over all SIMDs in a CU (that is, waves on different SIMDs are
   executing similar code). For highly unbalanced workloads, where e.g., one
   SIMD pair in the CU does not issue LDS instructions at all, this metric is
   better interpreted as the percentage of SIMDs issuing LDS instructions on
   :ref:`SIMD pairs <desc-lds>` that are actively using the LDS, averaged over
   the lifetime of the kernel.

.. [#lds-bank-conflict] The maximum value of the bank conflict rate is less than 100%
   (specifically: 96.875%), as the first cycle in the
   :ref:`LDS scheduler <desc-lds>` is never considered contended.

.. _lds-stats:

Statistics
==========

The LDS statistics panel gives a more detailed view of the hardware:

.. jinja:: lds-stats
   :file: _templates/metrics_table.j2
