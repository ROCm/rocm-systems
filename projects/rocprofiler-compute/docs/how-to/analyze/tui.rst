.. meta::
   :description: ROCm Compute Profiler analysis: Text-based User Interface
   :keywords: Omniperf, ROCm, profiler, tool, Instinct, accelerator, GUI, standalone, filter

****************************************
Text-based User Interface (TUI) analysis
****************************************

ROCm Compute Profiler's analyze mode now supports a lightweight Text-based User Interface (TUI)
that provides an interactive terminal experience for enhanced usability. You can use the TUI
interface as a more visually engaging and interactive alternative to explore analysis results
compared to the standard :doc:`cli`. It provides enhanced visual feedback and easy navigation without
needing the extra setup of a full graphical interface. This analysis option is implemented as a
terminal-based interface that offers real-time visual feedback, keyboard shortcuts for common
actions, and improved readability with formatted output.

.. note::

   TUI is currently in an early access state. While functional, you may encounter minor issues or limitations.
   For the limitations identified, see :ref:`Current limitations <tui-limitation>`
   Running production workloads is not recommended.

Launch the TUI analyzer
----------------------------------

1. Use the ``--tui`` flag with the analysis command to launch the ROCm Compute Profiler TUI analyzer.
For example:

.. code-block:: shell-session

   $ rocprof-compute analyze --tui

2. To start the analysis, use the dropdown menu at the top left of the screen to select a single
workload from ``rocprof-compute profile`` generated output directories.

.. image:: ../../data/analyze/tui.png
   :align: center
   :alt: ROCm Compute Profiler TUI home screen
   :width: 800

3. You can see the center window update with collapsed contents. Uncollapse to view tables, charts,
and graphs visualizing the analysis data.

4. After the analysis results are loaded, you can start interactive analysis with detailed metrics.
The TUI supports basic keyboard shortcuts, including quit application commands for easy navigation.

TUI analysis structure
----------------------------------

Unlike the :doc:`cli` plain style interfaces, the TUI restructures the analysis workflow into four
hierarchical categories to provide a more organized, top-down analysis approach:

1. Top Stat
2. High Level analysis
3. Detailed block analysis
4. Source Level analysis

You are recommended to follow this top-down hierarchical structure to conduct a thorough performance
analysis, starting with the broad overview and progressively drilling down to specific details.

.. _tui-limitation:

Current limitations
----------------------------------
The TUI implementation has several limitations that will be addressed in future releases:

* **PC sampling**: Source Level analysis does not have PC sampling enabled by default during the
  profiling stage. Refer to :doc:`../pc_sampling` for details on how to build and enable PC sampling
  manually.

* **Filtering capabilities**: Advanced filtering options such as kernel filtering and dispatch
  filtering are currently not supported. These features will be available in upcoming releases.
