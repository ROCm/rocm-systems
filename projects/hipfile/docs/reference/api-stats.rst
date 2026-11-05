.. meta::
   :description: API reference for hipFile I/O statistics, covering the leveled statistics structures and the functions that query them.
   :keywords: hipFile, ROCm, statistics, stats, I/O, API, GPU, reference

************************
Statistics API reference
************************

The hipFile statistics API reports I/O and operation counters accumulated by the
driver. Statistics are organized into three cumulative levels: Level 1 provides
basic I/O and operation counters, Level 2 adds I/O size histograms, and Level 3
adds per-GPU statistics.

Fields that hipFile does not track are zero-filled, and calculated fields such as
bandwidth and average latency are derived from the counters that are tracked. See
the Doxygen descriptions for the tracking status of individual fields.

Statistics types and functions
==============================

The maximum-GPU and UUID-length macros, the operation counter and leveled
statistics structures, and the functions to query Level 1, Level 2, and Level 3
statistics from the hipFile driver.

.. doxygengroup:: stats
   :content-only:
   :members:
