
.. meta::
  :description: Guide for using rocprofv3 process attachment
  :keywords: ROCprofiler-SDK, process attachment, ptrace, dynamic profiling

.. _rocprofv3_process_attachment:

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

Key considerations
-------------------

Here are some important points to be noted while using dynamic process attachment:

- The target process must be running and actively using GPU resources for meaningful profiling data.

- Attachment requires appropriate system permissions. It might even need elevated privileges depending on the target process.

- To use attachment in a docker container, add the ``ptrace`` capability to the container (``SYS_PTRACE``).

- The profiler collects data for the entire remaining lifetime of the process or until the configured collection period expires. To learn how to configure the collection period, see :ref:`duration-specific`.
