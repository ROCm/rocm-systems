.. meta::
   :description:  Quickstart guide for ROCm Compute Profiler (rocprofiler-compute)
   :keywords: Omniperf, ROCm, profiler, tool, Instinct, AMD, Profile, Analyze, CLI,  performance counters, quickstart, guide

**********
Quickstart
**********

This guide will help you quickly start using **rocprof-compute**, AMD’s ROCm Compute Profiler. By following these steps, you’ll learn how to profile GPU workloads and analyze performance data to identify bottlenecks and optimize your applications.

The following sections provide brief steps to get started with rocprof-compute. There are 2 main phases to use the tool:

1. Profiling 
2. Analyzing

Prerequisites
=============

Ensure ROCm installation is complete. Check:

**AMD System Management Interface**

.. code-block:: shell-session

   amd-smi

Purpose: Monitors GPU health, temperature, and utilization.
If fails: Verify ROCm install, GPU driver, kernel modules:

.. code-block:: shell-session

   lsmod | grep amdgpu

Check device nodes:

.. code-block:: shell-session

   ls /dev/kfd /dev/dri

**ROCm Info**

.. code-block:: shell-session

   rocminfo

Purpose: Displays ROCm platform details and GPU properties.
If fails: Check PATH, permissions (add user to render and video groups), set ``LD_LIBRARY_PATH``, reinstall ROCm if needed.

**Python & rocprof-compute-tool**

.. code-block:: shell-session

   python3 --version

Install dependencies:

.. code-block:: shell-session

   pip install -r /opt/rocm-<version>/libexec/rocprofiler-compute/requirements.txt

If missing libs: use requirements file at the above path.

Profiling
=========

Profiling is the process of collecting performance counters from your GPU application while it runs. ROCm Compute Profiler captures detailed metrics about kernel execution, memory usage, roofline and hardware utilization to help you understand and optimize performance.

The following examples will refer to some sample applications which you can get from the samples folder in our GitHub repository:  
https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-compute/sample

**Compile HIP sample:: Build the HIP sample into an executable named 'vcopy'**
   
.. code-block:: shell-session

   hipcc vcopy.cpp -o vcopy

**Profile Command:**

.. code-block:: shell-session

   rocprof-compute profile --name <workload_name> [profile options] [roofline options] -- <workload_cmd>

**Example:**

.. code-block:: shell-session

   rocprof-compute profile --name vcopy -- ./vcopy -n 1048576 -b 256

**Explanation:**

- ``rocprof-compute profile``: Starts a profiling session for a compute workload.
- ``--name vcopy``: Labels this run as 'vcopy' for easy identification and comparison.
- ``--``: Separates rocprof-compute options from your application arguments.
- ``./vcopy -n 1048576 -b 256``: Runs your app with:
  - ``-n 1048576``: Number of elements.
  - ``-b 256``: Block size (threads per block).

What happens during profiling?
------------------------------

Your application runs multiple times to collect all counters. Roofline analysis runs automatically unless disabled.  
While the profiling is collecting, you will see your application (vcopy) being executed several times. The tool requires several runs to collect all the performance counters required.

After profiling, you can find the generated files inside

.. code-block:: shell-session

   workloads/vcopy/MI200/

Above, we are running a basic example. For more details on all the profiling options, refer to the full documentation:  
:doc:`Profiling </how-to/profile>`

Also, you will notice that during the profiling phase, the roofline will run several iterations as well, to collect roofline data. For details on roofline, refer to the full documentation:
:doc:`Roofline Mode </how-to/profile/mode>`

For more details and options, run:

.. code-block:: shell-session

   rocprof-compute profile --help

Other Profiling Examples
------------------------

Profiles the workload and collects only roofline data for performance analysis:

.. code-block:: shell-session

    $ rocprof-compute profile --name vcopy --roof-only -- ./vcopy -n 1048576 -b 256

Profiles the workload and collects the metric for compute throughput utilization, skip roofline:

.. code-block:: shell-session

    $ rocprof-compute profile --name vcopy --set compute_thruput_util --no-roof -- ./vcopy -n 1048576 -b 256

List the available blocks/metrics available for profiling, by page, because list is long. Note the index for each section:

.. code-block:: shell-session

    $ rocprof-compute profile --list-available-metrics | more

Profiles the workload using block 2 (index) for system speed of light profiling:

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

Analysis is the process of examining profiling data to understand GPU kernel performance, identify bottlenecks, and find optimization opportunities. ROCm Compute Profiler provides multiple analysis modes to suit different workflows.

.. list-table::
  :header-rows: 1
  :widths: 25 50 25

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
  - ``vcopy/``: The name user provided during profiling run.
  - ``MI200``: Device-specific folder profiling auto-created.

For more details on analysis options, refer to the full documentation:   
:doc:`Analyze </how-to/analyze>`

Other Analysis Examples
-----------------------

Show a list of metrics supported for analysis:

.. code-block:: shell-session

   rocprof-compute analyze -p workloads/vcopy/MI200/ --list-available-metrics | more

Show System speed-of-light (2) and roofline (4) analysis:

.. code-block:: shell-session

   rocprof-compute analyze -p workloads/vcopy/MI200/ -b 2 4

Show memory chart (3) analysis:

.. code-block:: shell-session

   rocprof-compute analyze -p workloads/vcopy/MI200/ -b 3

Compare vcopy against vcopy_optimized (you made changes to app to optimize, recompiled):

.. code-block:: shell-session

   rocprof-compute profile -n vcopy_optimized -- ./vcopy_optimized -n 1048576 -b 256
   rocprof-compute analyze -p workloads/vcopy/MI200/ -p workloads/vcopy_optimized/MI200/

For more details and options, run:

.. code-block:: shell-session

   rocprof-compute analyze --help