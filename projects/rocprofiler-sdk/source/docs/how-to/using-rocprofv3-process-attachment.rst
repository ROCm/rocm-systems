
.. meta::
  :description: Guide for using rocprofv3 process attachment
  :keywords: ROCprofiler-SDK, process attachment, ptrace, dynamic profiling

.. _rocprofv3_process_attachment:
=================================

Using rocprofv3 Process Attachment:
=================================

``rocprofv3`` supports dynamic process attachment using the ``--attach`` option. This feature allows users to attach the profiler to an already running application without needing to restart it. The attachment is performed using the ``ptrace`` system call, which enables the profiler to monitor and collect performance data from the target process.
This capability is particularly useful for profiling long-running applications or services where restarting the application is not feasible.

Here is an example of the syntax for attaching to a running process:
.. code-block:: bash

   rocprofv3 --attach <PID> [--hip-trace] [--output-format <format>]

Where ``<PID>`` is the process ID of the target application. The optional ``--hip-trace`` flag enables HIP API tracing, and the ``--output-format`` option allows specifying the desired output format (e.g., rocpd, csv, json).

**Basic attachment syntax:**

.. code-block:: bash

    rocprofv3 -p <PID> <tracing_options>
    # or
    rocprofv3 --pid <PID> <tracing_options>  
    # or
    rocprofv3 --attach <PID> <tracing_options>

**Example:  Attach to a running process and profile**

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
   Press `Enter` in the terminal where ``rocprofv3`` is running to detach the profiler from the target application. Sending SIGINT (`Ctrl+C`) can also be sent to ``rocprofv3`` to detach from the target.

5. The profiling data will be saved in the specified output format.

**Example: Attach to a running process and profile for a specific duration (e.g., 5 seconds):**

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

4. The profiler will automatically detach after the specified duration (5 seconds in this case).
5. The profiling data will be saved in the specified output format.

   For example, if you used `--output-format csv`, the data will be saved as a CSV file.


**Example: Attach with counter collection:**

.. code-block:: bash

    rocprofv3 --pid 12345 --pmc SQ_WAVES GRBM_COUNT --output-format csv


The attachment functionality works with all tracing and profiling options available in ``rocprofv3``, providing the same comprehensive analysis capabilities as standard application launching.

**Important considerations for process attachment:**

- The target process must be running and actively using GPU resources for meaningful profiling data
- Attachment requires appropriate system permissions (may need elevated privileges depending on the target process)
- Attachment in a docker container requires the ptrace capability to be added for the container (`SYS_PTRACE`)
- The profiler will collect data for the entire remaining lifetime of the process or until the configured collection period expires
- Use ``--attach-duration-msec`` to specify how long to profile the attached process (in milliseconds)
