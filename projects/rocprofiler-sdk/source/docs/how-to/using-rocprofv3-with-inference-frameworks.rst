.. meta::
  :description: Documentation for using rocprofv3 with LLM inference frameworks such as vLLM and ATOM
  :keywords: ROCprofiler-SDK tool, LLM inference, vLLM, ATOM, tensor parallel, rocprofv3, rocprofv3 tool usage, ROCprofiler-SDK command line tool, ROCprofiler-SDK CLI


.. _using-rocprofv3-with-inference-frameworks:

==================================================
Using rocprofv3 with LLM inference frameworks
==================================================

`vLLM <https://github.com/vllm-project/vllm>`_ and `ATOM <https://github.com/ROCm/ATOM>`_ run inference in child processes: an engine process and, for multi-GPU configurations such as tensor parallelism, one worker process per GPU. Because ``rocprofv3`` collects data in-process, it profiles each of these processes and writes each process's output files when that process exits.

When these frameworks shut down, each parent process waits a grace period for its child processes to exit, then terminates them with ``SIGTERM`` followed by ``SIGKILL``. By default the grace period is about 5 seconds. If ``rocprofv3`` needs longer to write a process's output, that process is terminated first, and its output files are missing or incomplete.

To collect complete output, use a framework version that lets you configure the shutdown grace period, and set it long enough for ``rocprofv3`` to write its output.

Required versions
==================

.. list-table::
   :header-rows: 1
   :widths: 15 25 60

   * - Framework
     - Minimum version
     - Environment variable
   * - vLLM
     - 0.24.0
     - ``VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS``: Seconds the engine core waits for its workers to exit, and the client waits for the engine core to exit, before terminating them. Default: 5.
   * - ATOM
     - 0.1.7
     - ``ATOM_SHUTDOWN_TIMEOUT_S``: Seconds the server waits for its EngineCore processes, and each EngineCore waits for its ModelRunner workers, before terminating them. Default: 5.

Earlier versions use a hardcoded grace period and ignore these environment variables, so setting them has no effect.

The version string of a development or downstream build doesn't always show whether the build includes the change. To check the installed package directly, run:

.. code-block:: bash

    # vLLM: prints True when VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS is supported
    python3 -c "import vllm.envs as envs; print(hasattr(envs, 'VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS'))"

    # ATOM: prints True when ATOM_SHUTDOWN_TIMEOUT_S is supported
    python3 -c "import atom.utils.envs as envs; print(hasattr(envs, 'ATOM_SHUTDOWN_TIMEOUT_S'))"

Profiling vLLM
===============

Set ``VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS`` in the environment of the ``rocprofv3`` command so that every vLLM process inherits it. This applies to any vLLM entry point, such as ``vllm bench``, ``vllm serve``, or a Python script that uses the ``LLM`` class:

.. code-block:: bash

    export VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=300
    rocprofv3 --kernel-trace --disable-signal-handlers --output-format csv -d vllm-profile -- \
        vllm bench latency --model <model> --tensor-parallel-size <N>

The preceding command generates one set of output files per process, prefixed with the process ID. Each worker process writes a ``kernel_trace.csv`` for the GPU it drives.

The vLLM log shows whether the workers exited within the grace period:

.. code-block:: shell

    # Workers finished writing output and exited within the grace period
    [shutdown] Executor: all workers exited gracefully

    # Grace period expired; worker output is incomplete or missing
    [shutdown] Executor: workers still running after grace period; sending SIGTERM count=<N>

Profiling ATOM
===============

Set ``ATOM_SHUTDOWN_TIMEOUT_S`` in the environment of the ``rocprofv3`` command:

.. code-block:: bash

    export ATOM_SHUTDOWN_TIMEOUT_S=300
    rocprofv3 --kernel-trace --disable-signal-handlers --output-format csv -d atom-profile -- \
        python3 -m atom.examples.simple_inference --model <model> -tp <N>

When the grace period expires before a worker exits, ATOM logs a warning similar to the following and terminates the worker:

.. code-block:: shell

    worker pid=<pid> still alive after <timeout>s (ATOM_SHUTDOWN_TIMEOUT_S); terminating

Choosing a timeout
===================

The timeout is an upper bound. Workers that finish writing output exit immediately, so a large value doesn't slow down a run that shuts down cleanly. Set the timeout according to how much data ``rocprofv3`` collects:

- Kernel tracing with CSV output typically finishes in seconds, but can still exceed the 5-second default.
- Collecting many tracing domains together, such as ``--sys-trace`` with ``--rccl-trace`` and ``--pmc``, increases the output each worker writes. Perfetto output (``--output-format pftrace``) and thread trace can take several minutes per worker. In these cases, use a timeout well above 300 seconds, or reduce the collected data, for example by using :ref:`collection periods <collection-period>` or limiting the output formats.

If output is still missing, check the framework log for the termination messages shown in the preceding sections, and increase the timeout.
