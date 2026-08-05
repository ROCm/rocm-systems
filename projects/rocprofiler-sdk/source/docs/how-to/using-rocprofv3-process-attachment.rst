
.. meta::
  :description: Guide for using rocprofv3 process attachment
  :keywords: ROCprofiler-SDK, process attachment, ptrace, dynamic profiling

.. _rocprofv3-process-attachment:

==========================================
Dynamic process attachment using rocprofv3
==========================================

For profiling long-running applications or services where restarting the application is not feasible, ``rocprofv3`` provides dynamic process attachment using the ``--attach`` option. This feature facilitates attaching the profiler to a running application without the need to restart it. The attachment is performed using the ``ptrace`` system call, which enables the profiler to monitor and collect performance data from the target process.

Here is an example syntax for dynamic process attachment:

.. code-block:: bash

   rocprofv3 --attach <PID> [--hip-trace] [--output-format <format>]

Here are the options used in the preceding example:

- ``<PID>``: Process ID of the target application.

- ``--hip-trace``: This optional flag enables HIP API tracing.

- ``--output-format``: The desired output format such as rocpd, csv, or json.

.. note::

   In process-attachment mode, ``rocprofv3`` may generate output files asynchronously during detachment. As a result, output files might not be fully written immediately when ``rocprofv3`` returns. If your workflow needs output files to be complete before continuing, for example, if a script processes or removes the output directory right after detach, use ``--attach-sync-output``. This makes detach wait for output generation to finish, which can increase detach time.

**Basic attachment syntax:**

.. code-block:: bash

    rocprofv3 -p <PID> <tracing_options>
    # or
    rocprofv3 --pid <PID> <tracing_options>
    # or
    rocprofv3 --attach <PID> <tracing_options>

Basic dynamic process attachment
---------------------------------

Follow these steps to attach the profiler to a running process and profile:

1. Start the target application in the background:

   .. code-block:: bash

      ./myapp -n 1 &

2. Get the process ID (PID) of the running application:

   .. code-block:: bash

      echo $(pgrep myapp)
      OR
      ps aux | grep myapp

3. Attach ``rocprofv3`` to the running application:

   .. code-block:: bash

      rocprofv3 --attach <PID> --hip-trace --output-format rocpd

4. Detach the profiler when done:

   To detach the profiler from the target application, press "Enter" in the terminal where ``rocprofv3`` is running. You can also send SIGINT (Ctrl+C) to ``rocprofv3`` to detach from the target.

5. The profiling data will be saved in the format specified using ``output-format``.

.. _duration-specific:

Duration-specific dynamic process attachment
---------------------------------------------

Follow these steps to attach the profiler to a running process and profile for a specific duration such as 5 seconds:

1. Start the target application in the background:

   .. code-block:: bash

      ./myapp -n 1 &

2. Get the process ID (PID) of the running application:

   .. code-block:: bash

      echo $(pgrep myapp)
      OR
      ps aux | grep myapp

3. Attach ``rocprofv3`` to the running application:

   .. code-block:: bash

      rocprofv3 --attach <PID> --attach-duration-msec 5000 --sys-trace --output-format csv

4. The profiler will automatically detach after the specified duration (5 seconds in this case). Note that the duration is to be specified in milliseconds (ms).

5. The profiling data will be saved in the format specified using ``output-format``. For example, if you specify ``--output-format csv``, the data will be saved as a CSV file.

Dynamic process attachment with counter collection
---------------------------------------------------

The dynamic process attachment functionality works with all tracing and profiling options available in ``rocprofv3``, providing the same comprehensive analysis capabilities as standard application launching.

The following example attaches the profiler to a process with PID "12345", collects counters ``SQ_WAVES`` and ``GRBM_COUNT``, and saves profiling data in a CSV file:

.. code-block:: bash

   rocprofv3 --pid 12345 --pmc SQ_WAVES GRBM_COUNT --output-format csv

Reattaching to the same process
--------------------------------

The dynamic process attachment functionality supports reattachment, allowing attaching multiple times to the same PID over a process's lifetime. You need to provide the same PID to ``rocprofv3`` to reattach.

There are some restrictions on what the options are allowed to change when reattaching. Typically, tracing, PC sampling, ATT, counter collection, and other options that decide the data to be collected can't be changed. ``rocprofv3`` throws a ``RuntimeError`` if it detects a configuration change that isn't supported.

By default, the output file generation runs asynchronously after detachment, allowing for faster tool detachment. This implies that the output files might not be immediately available when ``rocprofv3`` exits. If the output file generation from the previous attachment is still in progress, ``rocprofv3`` blocks reattachment until the ongoing output generation completes.

.. class:: details

   Full list of options that mustn't change:

   - ALL options ending with ``trace``
   - ALL options starting with ``pc_sampling``
   - ALL options starting with ``att``
   - ``pmc``
   - ``pmc_groups``
   - ``output_config``
   - ``extra_counters``
   - ``kernel_include_regex``
   - ``kernel_exclude_regex``
   - ``kernel_iteration_range``

Synchronous output generation for scripts
------------------------------------------

For use cases requiring output files to be fully written before detachment completes, such as scripts that process or delete output directories immediately after detachment, you can enable synchronous output generation using the ``--attach-sync-output`` flag. This causes ``tool_detach`` to wait for all output files to be written before returning, ensuring output files are complete when the ``rocprofv3`` process exits.

For example, consider a script that attaches for a fixed duration and then terminates the workload as soon as ``rocprofv3`` returns:

.. code-block:: bash

   # Start the workload in the background
   ./myapp &
   WL_PID=$!

   # Attach for 5 seconds, then detach
   rocprofv3 --pid "$WL_PID" --attach-duration-msec 5000 \
       --output-format csv -o profile -d "$PWD"

   # Workload is killed immediately after rocprofv3 returns
   kill "$WL_PID"

With the default asynchronous output generation, the output thread runs inside the target process. Killing the workload right after ``rocprofv3`` returns can terminate that thread before it finishes writing, producing truncated or incomplete output files. Add ``--attach-sync-output`` so that detachment waits for output generation to finish before returning:

.. code-block:: bash

   ./myapp &
   WL_PID=$!

   rocprofv3 --pid "$WL_PID" --attach-duration-msec 5000 \
       --attach-sync-output \
       --output-format csv -o profile -d "$PWD"

   # Safe: output files are fully written before rocprofv3 returns
   kill "$WL_PID"

Attaching to a process tree
----------------------------

By default, ``rocprofv3 --attach`` attaches to the target process **and all of its descendant processes**. This is useful for profiling applications that spawn child processes, such as multiprocess MPI jobs or launchers that fork workers.

To attach to a PID and all its children (default behavior), use:

.. code-block:: bash

   rocprofv3 --attach 1234 --hip-trace

To attach only to the specified PID and skip its descendants, use ``--attach-children=false``:

.. code-block:: bash

   rocprofv3 --attach 1234 --attach-children=false --hip-trace

The child process tree is enumerated once at attach time using ``/proc``. Processes that are spawned after the attachment has begun are not automatically profiled.

Key considerations
-------------------

Here are some important points to be noted while using dynamic process attachment:

- The target process must be running and actively using GPU resources for meaningful profiling data.

- Attachment requires appropriate system permissions. It might even need elevated privileges depending on the target process.

- To use attachment in a docker container, add the ``ptrace`` capability to the container (``SYS_PTRACE``).

- The profiler collects data for the entire remaining lifetime of the process or until the configured collection period expires. To learn how to configure the collection period, see :ref:`duration-specific`.

- When attaching to a process tree, if attachment to an individual child process fails (for example, because it exited between enumeration and attach), the error is logged and attachment continues with the remaining processes.
