.. meta::
   :description:  Quickstart guide for ROCm Compute Profiler (rocprofiler-compute)
   :keywords: Omniperf, ROCm, profiler, tool, Instinct, AMD, Profile, Analyze, CLI, performance counters, quickstart, guide

**********
Quickstart
**********

This guide provides instructions for using **rocprof-compute**, AMD's ROCm Compute Profiler. It covers the steps required to profile GPU workloads and analyze performance data to identify bottlenecks and optimize applications.

The following sections provide brief steps to get started with rocprof-compute. There are 2 main phases to use the tool:

1. Profiling
2. Analysis

Prerequisites
=============

Ensure ROCm is installed. Check:

1. Check GPU and Driver
-----------------------

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

2. Check Python Environment
---------------------------

.. code-block:: shell-session

   python3 --version   # Requires Python 3.8+

3. Install Dependencies
-----------------------

.. code-block:: shell-session

      pip install -r <ROCM_PATH>/libexec/rocprofiler-compute/requirements.txt

   **Note:** Replace ``<ROCM_PATH>`` with the ROCm installation path (e.g., ``/opt/rocm`` or ``/opt/rocm-7.3.0``).

For detailed installation instructions, refer to :doc:`/install/core-install`.

Profiling
=========

Profiling is the process of collecting performance counters from a GPU application during execution. ROCm Compute Profiler captures detailed metrics regarding kernel execution, memory usage, roofline analysis, and hardware utilization to facilitate performance understanding and optimization.

The following examples reference sample applications available in the samples folder of the GitHub repository:
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

The application runs multiple times to collect all required performance counters; it executes multiple times during profiling. Roofline analysis runs automatically unless disabled with ``--no-roof``.

After profiling, the generated files can be found inside:

.. code-block:: shell-session

   workloads/vcopy/MI200/

For detailed information on all profiling options, refer to the full documentation:
:doc:`Profiling </how-to/profile>`

During the profiling phase, roofline analysis also executes multiple iterations to collect the necessary performance data. For detailed information on roofline analysis, refer to the full documentation:
:doc:`Roofline Mode </how-to/profile/mode>`

For more details and options, run:

.. code-block:: shell-session

   rocprof-compute profile --help

Other Profiling Examples
------------------------

Profiles the workload and collects only roofline data for performance analysis:

.. code-block:: shell-session

    $ rocprof-compute profile --name vcopy --roof-only -- ./vcopy -n 1048576 -b 256

Profiles the workload and collects the counters to compute the metric for compute throughput utilization, skipping roofline:

.. code-block:: shell-session

    $ rocprof-compute profile --name vcopy --set compute_thruput_util --no-roof -- ./vcopy -n 1048576 -b 256

Lists the available blocks/metrics available for profiling, by page, because list is long. Note the index for each section:

.. code-block:: shell-session

    $ rocprof-compute profile --list-available-metrics | more

Profiles the workload using block 2 for system speed of light profiling:

.. code-block:: shell-session

    $ rocprof-compute profile --name vcopy -b 2 -- ./vcopy -n 1048576 -b 256

Attaches to a running process for live profiling with specific block IDs, verbose output, and no roofline data:

.. code-block:: shell-session

    $ rocprof-compute profile -n try_live_attach_detach -b 3.1.1 4.1.1 5.1.1 --no-roof -VVV --attach-pid <process id>

Profiles the workload using multiple block (5 and 7) for detailed metric collection:

.. code-block:: shell-session

    $ rocprof-compute profile --name vcopy -b 5 7 -- ./vcopy -n 1048576 -b 256

Analyzing
=========

Analysis refers to the process of examining profiling data to understand GPU kernel performance, identifying bottlenecks, and determine optimization opportunities. ROCm Compute Profiler provides multiple analysis modes to accommodate different workflows.

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
- ``-p workloads/vcopy/MI200``: Path points to the workload directory:

  - ``workloads/``: Root folder for profiling runs.
  - ``vcopy/``: The name the user provided while launching the profiling run.
  - ``MI200``: Device-Name.

For more details on analysis options, refer to the full documentation:
:doc:`Analyze </how-to/analyze>`

Other Analysis Examples
-----------------------

Show a list of metrics supported for analysis:

.. code-block:: shell-session

   rocprof-compute analyze -p workloads/vcopy/MI200/ --list-available-metrics | more

Show or display System speed-of-light (2) and roofline (4) analysis:

.. code-block:: shell-session

   rocprof-compute analyze -p workloads/vcopy/MI200/ -b 2 4

Analyzes dispatches 12 and 34 from mixbench workload with 3 decimal precision:

.. code-block:: shell-session

   rocprof-compute analyze -p workloads/mixbench/MI200/ --dispatch 12 34 --decimal 3

Compares two workloads to evaluate the impact of code optimizations:

.. code-block:: shell-session

   rocprof-compute profile -n vcopy_optimized -- ./vcopy_optimized -n 1048576 -b 256
   rocprof-compute analyze -p workloads/vcopy/MI200/ -p workloads/vcopy_optimized/MI200/

For more details and options, run:

.. code-block:: shell-session

   rocprof-compute analyze --help
