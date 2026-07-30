.. meta::
   :description: How to analyze ROCm Compute Profiler data with ROCm Optiq
   :keywords: ROCm Compute Profiler, ROCm Optiq, ROCm, profiler, tool, Instinct,
              accelerator, AMD, analysis, analysis database, SQLite

.. _analyze-with-rocm-optiq:

***********************
Analyze with ROCm Optiq
***********************

.. note::

   The ROCm Compute Profiler integration with ROCm Optiq is in beta.

ROCm Optiq is a graphical application for interactively visualizing ROCm Compute
Profiler analysis data.

ROCm Compute Profiler and ROCm Optiq work together through an analysis database. First, generate an analysis database using the ``db``
:ref:`analysis output format <analysis-output-format>`, then open it in ROCm Optiq.
See :ref:`analysis database schema <analysis-database>` for what it contains.

Before using this workflow, review the `ROCm Optiq system requirements`_ and confirm that
your ROCm Optiq version supports the analysis database schema emitted by your installed
ROCm Compute Profiler version. Version and platform compatibility requirements can change
independently.

See the `ROCm Optiq analysis guide`_ for the current opening workflow and interface
details.

.. _ROCm Optiq analysis guide: https://rocm.docs.amd.com/projects/roc-optiq/en/latest/how-to/view-analysis.html
.. _ROCm Optiq system requirements: https://rocm.docs.amd.com/projects/roc-optiq/en/latest/install/optiq-install.html#system-requirements
