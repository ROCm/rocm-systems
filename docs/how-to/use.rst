.. meta::
   :description: ROCm Compute Profiler basic usage
   :keywords: ROCm Compute Profiler, ROCm, profiler, tool, Instinct, accelerator, AMD,
              basics, usage, operations

***********
Basic usage
***********

The following section outlines basic ROCm Compute Profiler workflows, modes, options, and
operations.

Command line profiler
=====================

Launch and profile the target application using the command line profiler.

The command line profiler launches the target application, calls the
ROCProfiler API via the ``rocprof`` binary, and collects profile results for
the specified kernels, dispatches, and hardware components. If not
specified, ROCm Compute Profiler defaults to collecting all available counters for all
kernels and dispatches launched by the your executable.

To collect the default set of data for all kernels in the target
application, launch, for example:

.. code-block:: shell

   $ rocprof-compute profile -n vcopy_data -- ./vcopy -n 1048576 -b 256

This runs the app, launches each kernel, and generates profiling results. By
default, results are written to a subdirectory with your accelerator's name;
for example, ``./workloads/vcopy_data/MI200/``, where name is configurable
via the ``-n`` argument.

.. note::

   To collect all requested profile information, ROCm Compute Profiler might replay kernels
   multiple times.

.. _basic-filter-data-collection:

Customize data collection
-------------------------

Options are available to specify for which kernels and metrics data should be
collected. Note that you can apply filtering in either the profiling or
analysis stage. Filtering at profiling collection often speeds up your
aggregate profiling run time.

Common filters to customize data collection include:

``-k``, ``--kernel``
   Enables filtering kernels by name.

``-d``, ``--dispatch``
   Enables filtering based on dispatch ID.

``-b``, ``--block``
   Enables collection metrics for only the specified analysis report blocks.

See :ref:`Filtering <filtering>` for an in-depth walkthrough.

To view available metrics by hardware block, use the ``profile`` mode ``--list-metrics``
option with an optional system architecture argument (inferred if not provided):

.. code-block:: shell

   $ rocprof-compute profile --list-metrics
   $ rocprof-compute profile --list-metrics <sys_arch>

.. _basic-analyze-cli:

Analyze in the command line
---------------------------

After generating a local output folder (for example,
``./workloads/vcopy_data/MI200``), use the command line tool to quickly
interface with profiling results. View different metrics derived from your
profiled results and get immediate access all metrics organized by hardware
blocks.

If you don't apply kernel, dispatch, or analysis report block filters at this stage,
analysis is reflective of the entirety of the profiling data.

To interact with profiling results from a different session, provide the
workload path.

``-p``, ``--path``
   Enables you to analyze existing profiling data in the ROCm Compute Profiler CLI.

See :doc:`analyze/cli` for more detailed information.

.. _basic-analyze-grafana:

Analyze in the Grafana GUI
--------------------------

To conduct a more in-depth analysis of profiling results, it's suggested to use
a Grafana GUI with ROCm Compute Profiler. To interact with profiling results, import your
data to the MongoDB instance included in the ROCm Compute Profiler Dockerfile. See
:doc:`/install/grafana-setup`.

To interact with Grafana data, stored in the ROCm Compute Profiler database, enter
``database`` :ref:`mode <modes-database>`; for example:

.. code-block:: shell

   $ rocprof-compute database --import [CONNECTION OPTIONS]

See :doc:`/how-to/analyze/grafana-gui` for more detailed information.

.. _modes:

Modes
=====

Modes change the fundamental behavior of the ROCm Compute Profiler command line tool.
Depending on which mode you choose, different command line options become
available.

.. _modes-profile:

Profile mode
------------

``profile``
   Launches the target application on the local system using
   :doc:`ROCProfiler <rocprofiler:index>`. Depending on the profiling options
   chosen, selected kernels, dispatches, and or hardware components used by the
   application are profiled. It stores results locally in an output folder:
   ``./workloads/\<name>``.

   .. code-block:: shell

      $ rocprof-compute profile --help

See :doc:`profile/mode` to learn about this mode in depth and to get started
profiling with ROCm Compute Profiler.

.. _modes-analyze:

Analyze mode
------------

``analyze``
   Loads profiling data from the ``--path`` (``-p``) directory into the ROCm Compute Profiler
   CLI analyzer where you have immediate access to profiling results and
   generated metrics. It generates metrics from the entirety of your profiled
   application or a subset identified through the ROCm Compute Profiler CLI analysis filters.

   To generate a lightweight GUI interface, you can add the ``--gui`` flag to your
   analysis command.

   Analyze mode now supports a lightweight Text-based User Interface (TUI) that
   provides an interactive terminal experience for enhanced usability. To enable TUI mode,
   use the ``--tui`` flag when running the analyze command:

   .. code-block:: shell

      $ rocprof-compute analyze --tui

   This mode is a middle ground to the highly detailed ROCm Compute Profiler Grafana GUI and
   is great if you want immediate access to a hardware component you’re already
   familiar with.

   .. code-block:: shell

      $ rocprof-compute analyze --help

   Analyze mode now supports a lightweight Text-based User Interface (TUI) that
   provides an interactive terminal experience for enhanced usability. To enable TUI mode,
   use the ``--tui`` flag when running the analyze command:

   .. code-block:: shell

      $ rocprof-compute analyze --tui

See :doc:`analyze/mode` to learn about these modes in depth and to get started
with analysis using ROCm Compute Profiler.

.. _modes-database:

Database mode
-------------

``database``
   The Grafana analyzer GUI is built on a MongoDB database. ``--import``
   profiling results to the DB to interact with the workload in Grafana or
   ``--remove`` the workload from the DB.

   Connection options need to be specified. See :doc:`/how-to/analyze/grafana-gui` for
   more details.

   .. code-block:: shell

      $ rocprof-compute database --help

See :doc:`/install/grafana-setup` to learn about setting up a Grafana server and
database instance to make your profiling data more digestible and shareable.

.. _global-options:

Global options
==============

The ROCm Compute Profiler command line tool has a set of *global* utility options that are
available across all modes.

``-v``, ``--version``
   Prints the ROCm Compute Profiler version and exits.

``-V``, ``--verbose``
   Increases output verbosity. Use multiple times for higher levels of
   verbosity.

``-q``, ``--quiet``
   Reduces output verbosity and runs quietly.

``-s``, ``--specs``
   Prints system specs and exits.

.. note::

   ROCm Compute Profiler also recognizes the project variable, ``ROCPROFCOMPUTE_COLOR`` should you
   choose to disable colorful output. To disable default colorful behavior, set
   this variable to ``0``.

.. _basic-operations:

Basic operations
================

The following table lists ROCm Compute Profiler's basic operations, their
:ref:`modes <modes>`, and required arguments.

.. list-table::
   :header-rows: 1

   * - Operation description
     - Mode
     - Required arguments

   * - :doc:`Profile a workload </how-to/profile/mode>`
     - ``profile``
     - ``--name``, ``-- <profile_cmd>``

   * - :ref:`Standalone roofline analysis <standalone-roofline>`
     - ``profile``
     - ``--name``, ``--roof-only``, ``--roofline-data-type <data_type>``, ``-- <profile_cmd>``

   * - :ref:`Import a workload to database <grafana-gui-import>`
     - ``database``
     - ``--import``, ``--host``, ``--username``, ``--workload``, ``--team``

   * - :ref:`Remove a workload from database <grafana-gui-remove>`
     - ``database``
     - ``--remove``, ``--host``, ``--username``, ``--workload``, ``--team``

   * - :doc:`Launch standalone GUI from CLI </how-to/analyze/standalone-gui>`
     - ``analyze``
     - ``--path``, ``--gui``

   * - :doc:`Interact with profiling results from CLI </how-to/analyze/cli>`
     - ``analyze``
     - ``--path``
