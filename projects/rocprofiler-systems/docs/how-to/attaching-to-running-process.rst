.. meta::
   :description: ROCm Systems Profiler attach to running process documentation
   :keywords: rocprofiler-systems, rocprofsys, ROCm, profiler, attach, running process, dynamic profiling, tool, Instinct, accelerator, AMD

****************************************************
Attaching to a running process
****************************************************

`ROCm Systems Profiler <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems>`_ can attach
to and profile an already running process using the ``rocprof-sys-attach`` executable.
This is useful for profiling long-running applications, daemons, or processes
that are difficult to restart with instrumentation.

.. important::

   **The target process must be started with the ``ROCP_TOOL_ATTACH=1`` environment
   variable set** to enable attachment support. Without this, attachment will fail.

   .. code-block:: shell

      # Start your application with attachment support enabled
      ROCP_TOOL_ATTACH=1 ./my_application

   Alternatively, if using a version of ``rocprofiler-register`` built with
   ``ROCPROFILER_REGISTER_BUILD_DEFAULT_ATTACHMENT=ON``, this is not required.

.. note::

   The target process should be compiled with debug symbols or frame pointers
   for meaningful profiling results. Additionally, the process must be running
   with appropriate permissions for attachment (see ``ptrace`` requirements).

.. admonition:: Current limitation: no reattach

   Once you detach from a process, you **cannot reattach** to the same process.
   A second attach to the same PID will result in an error. Support for
   reattaching to a previously detached process is planned for a future release.

When to use rocprof-sys-attach
========================================

Use ``rocprof-sys-attach`` when:

* The application is already running and cannot be easily restarted
* You want to profile a specific phase of a long-running application
* The application is started by an external system (e.g., job scheduler)
* You need to attach profiling dynamically based on runtime conditions

The rocprof-sys-attach executable
========================================

View the help menu of ``rocprof-sys-attach`` with the ``-h`` / ``--help`` option:

.. code-block:: shell

   $ rocprof-sys-attach --help
   Usage: rocprof-sys-attach -p <pid> [OPTIONS]

   Attach to a running process for profiling.

   Options:
     -p <pid>             Process ID to attach to (required)
     -o, --output PATH    Output path for profiling results
     -F, --format FORMAT[,FORMAT,...]
                          Output format(s): perfetto, rocpd
     -h, --help           Show this help message

   Environment variables:
     ROCPROFSYS_OUTPUT_PATH       Output directory for profiling data
     ROCPROFSYS_TRACE             Enable perfetto trace output
     ROCPROFSYS_USE_ROCPD         Enable rocpd database output
     ROCPROF_ATTACH_TOOL_LIBRARY  Path to the tool library

   Once attached, press ENTER to detach from the process.

Command-line options
----------------------------------------

``-p <pid>`` (required)
   The process ID of the running application to attach to. You can find the PID
   using tools like ``ps``, ``pgrep``, or ``top``.

``-o, --output PATH``
   Specifies the output directory for profiling results. If not specified,
   results are written to the default location (``rocprof-sys-output/``).

``-F, --format FORMAT[,FORMAT,...]``
   Specifies the output format(s) for profiling data. Multiple formats can be
   specified as a comma-separated list. Available formats:

   * ``perfetto`` - Generates a Perfetto trace file (``.proto``) that can be
     visualized in the `Perfetto UI <https://ui.perfetto.dev>`_
   * ``rocpd`` - Generates a RocPD SQLite database file (``.db``) for
     programmatic analysis

Basic usage
========================================

1. **Find the process ID** of the running application:

   .. code-block:: shell

      $ pgrep -f my_application
      12345

2. **Attach to the process** with basic profiling:

   .. code-block:: shell

      $ rocprof-sys-attach -p 12345

3. **Wait for the profiling period** you want to capture, then press ENTER to
   detach and finalize the profiling data.

Examples
========================================

Attach with Perfetto trace output
----------------------------------------

Generate a Perfetto trace file for visualization:

.. code-block:: shell

   $ rocprof-sys-attach -p 12345 -F perfetto

After detaching, the trace file will be available at:
``rocprof-sys-output/<timestamp>/perfetto-trace-12345.proto``

Attach with custom output path
----------------------------------------

Specify a custom output directory:

.. code-block:: shell

   $ rocprof-sys-attach -p 12345 -o /path/to/results -F perfetto

Attach with multiple output formats
----------------------------------------

Generate both Perfetto trace and RocPD database:

.. code-block:: shell

   $ rocprof-sys-attach -p 12345 -o ./profiling-results -F perfetto,rocpd

This generates:

* ``./profiling-results/<timestamp>/perfetto-trace-12345.proto``
* ``./profiling-results/<timestamp>/rocpd-12345.db``

Using environment variables
----------------------------------------

You can also configure profiling using environment variables:

.. code-block:: shell

   $ ROCPROFSYS_OUTPUT_PATH=/my/output ROCPROFSYS_TRACE=true rocprof-sys-attach -p 12345

Workflow example
========================================

Here is a complete workflow for attaching to a running GPU application:

.. code-block:: shell-session

   # Terminal 1: Start your application with attachment support enabled
   $ ROCP_TOOL_ATTACH=1 ./my_gpu_application
   [my_gpu_application] Starting computation...

   # Terminal 2: Find the PID and attach
   $ pgrep -f my_gpu_application
   98765

   $ rocprof-sys-attach -p 98765 -o ./results -F perfetto,rocpd
   [rocprof-sys-attach] Using tool library: /opt/rocm/lib/librocprof-sys-dl.so
   [rocprof-sys-attach] Output path: ./results
   [rocprof-sys-attach] Output format: perfetto rocpd
   [rocprof-sys-attach] Trying to attach to process 98765
   [rocprof-sys-attach] Attached to process 98765. Press ENTER to detach.

   # Let the application run for the profiling period...
   # Press ENTER when ready to stop profiling

   [rocprof-sys-attach] Detached from process 98765
   [rocprof-sys-attach] Output written to: ./results
   [rocprof-sys-attach]   - Perfetto trace: perfetto-trace-98765.proto
   [rocprof-sys-attach]   - RocPD database: rocpd-98765.db

   # View the trace in Perfetto UI
   $ firefox https://ui.perfetto.dev
   # Drag and drop the .proto file to visualize

Troubleshooting
========================================

Attachment fails immediately
----------------------------------------

If attachment fails, ensure the target process was started with attachment support:

.. code-block:: shell

   # The target process MUST be started with this environment variable
   ROCP_TOOL_ATTACH=1 ./my_application

Without ``ROCP_TOOL_ATTACH=1``, the target process does not initialize the
attachment infrastructure and cannot be profiled dynamically.

Permission denied
----------------------------------------

If you receive a permission error when attaching, ensure:

1. You have appropriate permissions to attach to the process (same user or root)
2. The ``ptrace`` scope allows attachment:

   .. code-block:: shell

      # Check current setting
      $ cat /proc/sys/kernel/yama/ptrace_scope

      # Temporarily allow attachment (requires root)
      $ sudo sysctl kernel.yama.ptrace_scope=0

Process not found
----------------------------------------

If the process cannot be found:

1. Verify the PID is correct: ``ps -p <pid>``
2. Ensure the process is still running
3. Check if the process is in a different namespace (containers)

Reattach fails
----------------------------------------

If you previously detached from a process and try to attach again to the same
PID, the attach will fail. This is a current limitation; reattach support is
planned for a future release. To profile the same application again, restart
the application and attach to the new process.

See also
========================================

* :doc:`Sampling the call stack <./sampling-call-stack>` - Alternative profiling method
  using ``rocprof-sys-sample``
* :doc:`Understanding the output <./understanding-rocprof-sys-output>` - How to
  interpret profiling results
* :doc:`Configuring runtime options <./configuring-runtime-options>` - Environment
  variables for fine-tuning profiling behavior
