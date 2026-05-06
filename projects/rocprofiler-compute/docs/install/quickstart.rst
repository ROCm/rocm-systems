.. meta::
   :description:  Quickstart guide for ROCm Compute Profiler (rocprofiler-compute)
   :keywords: Omniperf, ROCm, profiler, tool, Instinct, AMD, Profile, Analyze, CLI, performance counters, quickstart, guide

**********
Quickstart
**********

This guide provides instructions for getting started and using ROCm Compute Profiler. It covers the steps required to profile GPU workloads and analyze performance data to identify bottlenecks and optimize applications.

There are two main phases to use the tool:

1. :ref:`Profiling <profile-quickstart>`
2. :ref:`Analysis <analysis-quickstart>`

Prerequisites
=============

Ensure ROCm is installed and follow the steps:

1. Check the GPU and driver.

   .. code-block:: shell-session

      amd-smi          # Monitor GPU health, temperature, utilization
      rocminfo         # Display ROCm platform and GPU properties

   If these commands fail:

   - Verify that the GPU driver is loaded:

   .. code-block:: shell-session

      lsmod | grep amdgpu

   - Load the driver if needed:

   .. code-block:: shell-session

      sudo modprobe amdgpu

   - Verify that the device nodes exist:

   .. code-block:: shell-session

      ls /dev/kfd /dev/dri

   - Ensure that the user name is added to the ``render`` and ``video`` groups:

   .. code-block:: shell-session

      sudo usermod -aG render,video $USER
      # Log out and back in for changes to take effect

   - If ``rocminfo`` or ``amd-smi`` commands are not found, set ROCm environment:

   .. code-block:: shell-session

      export PATH=/opt/rocm/bin:$PATH
      export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}

2. Check the Python environment.

   .. code-block:: shell-session

      python3 --version   # Requires Python 3.8+

3. Check the installation dependencies.

   .. code-block:: shell-session

      pip install -r <ROCM_PATH>/libexec/rocprofiler-compute/requirements.txt

   **Note:** Replace ``<ROCM_PATH>`` with the ROCm installation path (e.g., ``/opt/rocm`` or ``/opt/rocm-7.3.0``).

For detailed installation instructions, refer to :doc:`/install/core-install`.

.. _profile-quickstart:

Profiling
=========

Profiling is the process of collecting performance counters from a GPU application during execution. ROCm Compute Profiler captures detailed metrics regarding kernel execution, memory usage, roofline analysis, and hardware utilization to facilitate performance understanding and optimization.

The following examples reference sample applications located in the samples folder of the GitHub repository:
https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute/sample

**Compile HIP sample:: Build the HIP sample into an executable named 'vcopy'**

.. code-block:: shell-session

   hipcc vcopy.cpp -o vcopy

**Profile Command:**

.. code-block:: shell-session

   rocprof-compute profile --name <workload_name> [profile options] [roofline options] -- <workload_cmd>

**Example:**

.. code-block:: shell

    rocprof-compute profile --name vcopy -- ./vcopy -n 1048576 -b 256

**Explanation:**

- ``rocprof-compute profile``: Starts a profiling session for a compute workload.
- ``--name vcopy``: Labels this run as 'vcopy' for easy identification and comparison.
- ``--``: Separates rocprof-compute options from the application arguments.
- ``./vcopy -n 1048576 -b 256``: Executes the application with the following parameters:

  - ``-n 1048576``: Number of elements.
  - ``-b 256``: Block size (threads per block).

What happens during profiling?
------------------------------

The application runs multiple times to collect all required performance counters; it executes multiple times during profiling. Roofline analysis runs automatically unless you disable it using ``--no-roof``.

After profiling, the generated files can be found inside:

.. code-block:: shell-session

   workloads/vcopy/MI200/

For detailed information on all profiling options, refer to :doc:`../how-to/profile/mode`.

During the profiling phase, roofline microbenchmarks also run to collect
hardware peak data (saved as ``roofline.csv``). To generate roofline HTML
charts from this data, run ``rocprof-compute analyze`` on the output directory.
For detailed information on roofline analysis, refer to :ref:`Standalone roofline <standalone-roofline>`.

For more details and options, run:

.. code-block:: shell-session

   rocprof-compute profile --help

Profiling examples
------------------------

Common use cases when profiling a workload are:

Collect roofline data and generate HTML charts
++++++++++++++++++++++++++++++++++++++++++++++++

Profile mode collects roofline microbenchmark data (``roofline.csv``). To
generate interactive HTML roofline charts, run analyze mode on the output:

.. code-block:: shell-session

    $ rocprof-compute profile --name vcopy --roof-only -- ./vcopy -n 1048576 -b 256
    $ rocprof-compute analyze -p workloads/vcopy/MI200/ --roofline-data-type FP32


Collect the counters to compute the metric for compute throughput utilization, skipping roofline
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
.. code-block:: shell-session

    $ rocprof-compute profile --name vcopy --set compute_thruput_util --no-roof -- ./vcopy -n 1048576 -b 256

List the available blocks/metrics for profiling
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

The blocks/metrics are listed by page, because the list is long. Note the index for each section.

.. code-block:: shell-session

    $ rocprof-compute profile --list-available-metrics | more

Using block 2 for system speed-of-light profiling
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: shell-session

    $ rocprof-compute profile --name vcopy -b 2 -- ./vcopy -n 1048576 -b 256


Attach to a running process for live profiling
+++++++++++++++++++++++++++++++++++++++++++++++++

Dynamic process attachment can be performed with specific block IDs, verbose output, and no roofline data.

.. code-block:: shell-session

    $ rocprof-compute profile -n try_live_attach_detach -b 3.1.1 4.1.1 5.1.1 --no-roof -VVV --attach-pid <process id>

Use multiple blocks (5 and 7) for detailed metric collection
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: shell-session

    $ rocprof-compute profile --name vcopy -b 5 7 -- ./vcopy -n 1048576 -b 256


.. _analysis-quickstart:

Analysis
=========

Analysis phase refers to the process of examining profiling data to understand GPU kernel performance, identify bottlenecks, and determine optimization opportunities. ROCm Compute Profiler provides multiple analysis modes to accommodate different workflows.

.. list-table::
  :header-rows: 1
  :widths: 25 25 25

  * - Mode
    - When to Use
    - Links to docs
  * - :doc:`CLI (Command Line Interface) </how-to/analyze/cli>`
    - Fast, scriptable insights; great for automation and quick checks.
    - `CLI analysis <https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-compute/docs/how-to/analyze/cli.rst>`_
  * - :doc:`GUI (Standalone Graphical Interface) </how-to/analyze/standalone-gui>`
    - Interactive exploration, visual drill-down, and detailed charts.
    - `Standalone GUI analysis <https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-compute/docs/how-to/analyze/standalone-gui.rst>`_
  * - :doc:`TUI (Textual User Interface) </how-to/analyze/tui>`
    - Lightweight, keyboard-driven experience for terminals.
    - `Text-based User Interface (TUI) analysis <https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-compute/docs/how-to/analyze/tui.rst>`_

**Analysis Command:**

.. code-block:: shell-session

   rocprof-compute analyze -p <workloads_directory>

**Example:**

.. code-block:: shell-session

   rocprof-compute analyze -p workloads/vcopy/MI200/

**Explanation:**

- ``rocprof-compute analyze``: Starts analysis mode to process profiling results.
- ``-p workloads/vcopy/MI200``: The path points to the workload directory:

  - ``workloads/``: Root folder for profiling runs.
  - ``vcopy/``: The name the user provided while launching the profiling run.
  - ``MI200``: Device-Name.

For more details on analysis options, refer to :doc:`Analyze <../how-to/analyze/mode>`.

Analysis examples
-----------------------

Common use cases when analyzing a workload are:

Show a list of metrics supported for analysis
+++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: shell-session

   rocprof-compute analyze -p workloads/vcopy/MI200/ --list-available-metrics | more


Show or display System speed-of-light (2) and roofline (4) analysis
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: shell-session

   rocprof-compute analyze -p workloads/vcopy/MI200/ -b 2 4

Analyze dispatches 12 and 34 from mixbench workload with 3 decimal precision:
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: shell-session

   rocprof-compute analyze -p workloads/mixbench/MI200/ --dispatch 12 34 --decimal 3

Compare two workloads to evaluate the impact of code optimizations
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

.. code-block:: shell-session

   rocprof-compute profile -n vcopy_optimized -- ./vcopy_optimized -n 1048576 -b 256
   rocprof-compute analyze -p workloads/vcopy/MI200/ -p workloads/vcopy_optimized/MI200/

For more details and options, run:

.. code-block:: shell-session

   rocprof-compute analyze --help
