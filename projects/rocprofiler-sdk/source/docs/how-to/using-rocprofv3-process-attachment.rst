
.. meta::
  :description: Guide for using rocprofv3 process attachment
  :keywords: ROCprofiler-SDK, process attachment, ptrace, dynamic profiling

.. _rocprofv3_process_attachment:
=================================

Using rocprofv3 Process Attachment:
=================================

``rocprofv3`` supports dynamic process attachment using the ``--attach`` option. This feature allows users to attach the profiler to an already running application without needing to restart it. The attachment is performed using the ``ptrace`` system call, which enables the profiler to monitor and collect performance data from the target process.
This capability is particularly useful for profiling long-running applications or services where restarting the application is not feasible.

To attach ``rocprofv3`` to a running process, use the following command:
.. code-block:: bash

   rocprofv3 --attach <pid> [--hip-trace] [--output-format <format>] -- <application>

Where ``<pid>`` is the process ID of the target application. The optional ``--hip-trace`` flag enables HIP API tracing, and the ``--output-format`` option allows specifying the desired output format (e.g., rocpd, csv, json).

**Example: Attaching to a Running Application**
1. Start the target application in the background:
.. code-block:: bash

   ./myapp -n 1 &

1. Get the process ID (PID) of the running application:
.. code-block:: bash

   echo $(pgrep myapp)

1. Attach ``rocprofv3`` to the running application:
.. code-block:: bash

   rocprofv3 --attach <PID> --hip-trace --output-format rocpd -- ./myapp

1. Detach the profiler when done:
   Press `Enter` in the terminal where ``rocprofv3`` is running to detach the profiler from the target application.

**Notes:**
- Ensure that you have the necessary permissions to attach to the target process. You may need to run the command as the same user or with elevated privileges (e.g., using `sudo`).