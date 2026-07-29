.. meta::
   :description: How to analyze ROCm Compute Profiler data with ROCm Optiq (Beta)
   :keywords: ROCm Compute Profiler, ROCm Optiq, ROCm, profiler, tool, Instinct,
              accelerator, AMD, analysis, analysis database, SQLite

.. _analyze-with-rocm-optiq:

******************************
Analyze with ROCm Optiq (Beta)
******************************

ROCm Optiq (Beta) is a separate graphical application for interactively visualizing ROCm
Compute Profiler analysis data. It opens the analysis database emitted by
ROCm Compute Profiler analysis.

Before using this workflow, review the `ROCm Optiq system requirements`_ and confirm that
your ROCm Optiq version supports the analysis database schema emitted by your installed
ROCm Compute Profiler version. Version and platform compatibility requirements can change
independently.

1. Generate a ROCm Compute Profiler analysis database from an existing
   workload directory:

   .. code-block:: shell-session

      $ rocprof-compute analyze --output-name <analysis_name> --output-format db -p <workload_directory>

   ``<analysis_name>`` can contain only alphanumeric characters, underscores, and hyphens.
   The command creates ``<analysis_name>.db`` in the current working directory.

2. After confirming a compatible ROCm Compute Profiler and ROCm Optiq version combination,
   start ROCm Optiq and open ``<analysis_name>.db`` using **File > Open** or by dragging the
   file into the application window. See the `ROCm Optiq analysis guide`_ for the current
   opening workflow and interface details.

See :doc:`CLI analysis <cli>` for complete output-format behavior and
:ref:`analysis database schema <analysis-database>` details.

.. _ROCm Optiq analysis guide: https://rocm.docs.amd.com/projects/roc-optiq/en/latest/how-to/view-analysis.html
.. _ROCm Optiq system requirements: https://rocm.docs.amd.com/projects/roc-optiq/en/latest/install/optiq-install.html#system-requirements
