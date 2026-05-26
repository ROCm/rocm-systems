
.. meta::
  :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software
  :keywords: ROCprofiler-SDK tool usage, rocprofv3 user manual, rocprofv3 usage, rocprofv3 user guide, using rocprofv3, ROCprofiler-SDK tool user guide, ROCprofiler-SDK tool user manual, using ROCprofiler-SDK tool, ROCprofiler-SDK command-line tool, ROCprofiler-SDK CLI, ROCprofiler-SDK command line tool

.. _rocprofv3-io-options:

==============================
rocprofv3 I/O control options
==============================

``rocprofv3`` provides the following options to control the output.

.. _output-prefix-keys:

Output prefix keys
-------------------

Output prefix keys are useful in multiple use cases but are most helpful when dealing with multiple profiling runs or large MPI jobs. Here is the list of available keys:

.. list-table::
   :header-rows: 1

   * - String
     - Encoding
   * - ``%argv%``
     - Entire command-line condensed into a single string
   * - ``%argt%``
     - Similar to ``%argv%`` except basename of the first command-line argument
   * - ``%args%``
     - All command-line arguments condensed into a single string
   * - ``%tag%``
     - Basename of the first command-line argument
   * - ``%hostname%``
     - Hostname of the machine (``gethostname()``)
   * - ``%pid%``
     - Process identifier (``getpid()``)
   * - ``%ppid%``
     - Parent process identifier (``getppid()``)
   * - ``%pgid%``
     - Process group identifier (``getpgid(getpid())``)
   * - ``%psid%``
     - Process session identifier  (``getsid(getpid())``)
   * - ``%psize%``
     - Number of sibling processes (reads ``/proc/<PPID>/tasks/<PPID>/children``)
   * - ``%job%``
     - Value of ``SLURM_JOB_ID`` environment variable if exists, else 0
   * - ``%rank%``
     - Value of ``SLURM_PROCID`` environment variable if exists, else ``MPI_Comm_rank``, or 0 for non-mpi
   * - ``%size%``
     - ``MPI_Comm_size`` or 1 for non-mpi
   * - ``%nid%``
     - ``%rank%`` if possible, otherwise ``%pid%``
   * - ``%launch_time%``
     - Launch date and/or time
   * - ``%env{NAME}%``
     - Value of ``NAME`` environment variable (``getenv(NAME)``)
   * - ``$env{NAME}``
     - Alternative syntax to ``%env{NAME}%``
   * - ``%p``
     - Shorthand for ``%pid%``
   * - ``%j``
     - Shorthand for ``%job%``
   * - ``%r``
     - Shorthand for ``%rank%``
   * - ``%s``
     - Shorthand for ``%size%``

Output directory
-----------------

To specify the output directory, use ``--output-directory`` or ``-d`` option. If not specified, the default output path is ``%hostname%/%pid%``.

.. code-block:: shell

   rocprofv3 --hip-trace --output-directory output_dir --output-format csv -- <application_path>

The preceding command generates an ``output_dir/%hostname%/%pid%_hip_api_trace.csv`` file.

.. _output_field_format:

The output directory option supports many placeholders such as:

- ``%hostname%``: Machine host name
- ``%pid%``: Process ID
- ``%env{NAME}%``: Consistent with other output key formats (starts and ends with `%`)
- ``$ENV{NAME}``: Similar to CMake
- ``%q{NAME}%``: Compatibility with NVIDIA

To see the complete list, refer to :ref:`output-prefix-keys`.

The following example shows how to use the output directory option with placeholders:

.. code-block:: bash

   mpirun -n 2 rocprofv3 --hip-trace -d %h.%p.%env{OMPI_COMM_WORLD_RANK}% --output-format csv -- <application_path>

The preceding command runs the application with ``rocprofv3`` and generates the trace file for each rank. The trace files are prefixed with hostname, process ID, and MPI rank.

Assuming the hostname as ``ubuntu-latest`` and the process IDs as 3000020 and 3000019, the output file names are:

.. code-block:: bash

    ubuntu-latest.3000020.1/ubuntu-latest/3000020_agent_info.csv
    ubuntu-latest.3000019.0/ubuntu-latest/3000019_agent_info.csv
    ubuntu-latest.3000020.1/ubuntu-latest/3000020_hip_api_trace.csv
    ubuntu-latest.3000019.0/ubuntu-latest/3000019_hip_api_trace.csv

Output file
------------

To specify the output file name, use ``--output-file`` or ``-o`` option. If not specified, the output file is prefixed with the process ID by default.

.. code-block:: shell

   rocprofv3 --hip-trace --output-file output --output-format csv -- <application_path>

The preceding command generates an ``output_hip_api_trace.csv`` file.

The output file name can also include placeholders such as ``%hostname%`` and ``%pid%``. For example:

.. code-block:: shell

   rocprofv3 --hip-trace --output-file %hostname%/%pid%_hip_api_trace --output-format csv -- <application_path>

The preceding command generates an ``%hostname%/%pid%_hip_api_trace.csv`` file.

.. _collection-period:

Collection period
------------------

The collection period is the time interval during which the profiling data is collected. You can specify the collection period using the ``--collection-period`` or ``-P`` option.
You can also specify multiple configurations, each defined by a triplet in the format ``start_delay:collection_time:repeat``.

The triplet is defined as follows:

- **Start delay time**: The time after which the profiling data collection starts.
- **Collection time**: The time period during which the profiling data is collected.
- **Repeat**: The number of times the cycle is repeated. A repeat value of 0 indicates that the cycle will repeat indefinitely.

.. code-block:: shell

   rocprofv3 --collection-period 5:1:1 --hip-trace -- <application_path>

The preceding command collects the profiling data for 1 second, starting 5 seconds after the application starts, and this cycle will be repeated once.

The collection period can be specified in different units, such as seconds, milliseconds, microseconds, and nanoseconds. The default unit is "seconds". You can change the unit using the ``--collection-period-unit`` option.

The available time units are:

``--collection-period-unit``: ``hour``, ``min``, ``sec``, ``msec``, ``usec``, ``nsec``.

To specify the time unit as milliseconds, use:

.. code-block:: shell

   rocprofv3 --collection-period 5:1:0 --collection-period-unit msec --hip-trace -- <application_path>

Perfetto-specific options
--------------------------

The following options are specific to Perfetto tracing and are used to control the Perfetto data collection behavior:

- **--perfetto-buffer-fill-policy {discard,ring_buffer}**: Policy for handling new records when Perfetto reaches the buffer limit.

  - **RING_BUFFER (default)**: The buffer behaves like a ring buffer. Once full, writes wrap over and replace the oldest trace data in the buffer.

  - **DISCARD**: The buffer stops accepting data once full. Further write attempts are dropped.

- **--perfetto-buffer-size KB**: The buffer size for Perfetto output in KB. Default: 1 GB. If set, stops the tracing session after N bytes have been written. Used to cap the trace size.

- **--perfetto-backend {inprocess,system}**: Perfetto data collection backend. ``system`` mode requires starting traced and perfetto daemons. By default Perfetto keeps the full trace buffers in memory.

- **--perfetto-shmem-size-hint KB**: Perfetto shared memory size hint in KB. Default: 64 KB. This option gives you control over shared memory buffer sizing. You can tweak this option to avoid data losses when data is produced at a higher rate.
