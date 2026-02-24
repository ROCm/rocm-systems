.. meta::
   :description: ROCm Systems Profiler communication runtime profiling documentation
   :keywords: rocprof-sys, rocprofiler-systems, ROCm, MPI, RCCL, UCX, communication, profiler, tracking, distributed, AMD

****************************************************
Communication Runtime Profiling
****************************************************

`ROCm Systems Profiler <https://github.com/ROCm/rocm-systems/tree/develop/projects/rocprofiler-systems>`_ profiles several widely used communication runtimes and libraries, including MPI, RCCL, and UCX.

These runtimes operate at different layers of the communication stack—from high-level programming models to low-level transport mechanisms. ROCm Systems Profiler provides coordinated tracing across these layers to enable end-to-end analysis of communication behavior, overheads, and performance bottlenecks.

Communication Runtime Layers
============================

The supported communication runtimes span multiple layers of the parallel computing stack:

**High-Level Programming Models**

* **MPI (Message Passing Interface)**: The de facto standard for distributed memory parallel programming, providing point-to-point and collective communication primitives for CPU-based applications.

**GPU Collective Communication Libraries**

* **RCCL (ROCm Communication Collectives Library)**: AMD's GPU-aware collective communication library, optimized for multi-GPU communication within and across nodes. RCCL is designed to work seamlessly with ROCm and provides highly optimized implementations of collective operations like AllReduce, AllGather, and Broadcast.

**Low-Level Communication Frameworks**

* **UCX (Unified Communication X)**: A high-performance communication framework that provides low-level abstractions for RDMA, shared memory, and other transport mechanisms. UCX is often used as a backend for higher-level libraries like MPI and RCCL, providing efficient point-to-point communication, RMA (Remote Memory Access) operations, and active messages.

.. note::

   **Automatic Detection and Default Behavior:**

   * **MPI** (``ROCPROFSYS_USE_MPIP``): Enabled by default (``ON``). When using binary instrumentation, ROCm Systems Profiler automatically detects MPI symbols in the target application and enables MPI support.
   * **UCX** (``ROCPROFSYS_USE_UCX``): Disabled by default (``OFF``). Must be explicitly enabled to trace UCX operations. This is a runtime user-configurable option.
   * **RCCL** (``ROCPROFSYS_USE_RCCLP``): Disabled by default (``OFF``). Must be explicitly enabled to trace RCCL operations.

   These settings can be controlled at runtime using their respective environment variables to enable or disable tracing as needed.

Profiling MPI
=============

MPI support is enabled through the ``ROCPROFSYS_USE_MPIP`` configuration setting, which is **enabled by default**. ROCm Systems Profiler can be built with full (``ROCPROFSYS_USE_MPI=ON``) or partial (``ROCPROFSYS_USE_MPI_HEADERS=ON``) MPI support using the build-time configuration options. By default, ROCm Systems Profiler uses partial MPI support with the OpenMPI headers. For detailed information on building rocprofiler-systems with MPI support, see the :doc:`installation guide <../install/install>`.

When using binary instrumentation with ``rocprof-sys-instrument``, MPI functions are automatically detected in the target application. If MPI symbols (such as ``MPI_Init``, ``MPI_Init_thread``, ``MPI_Finalize``) are found, MPI support is automatically enabled.

Configuration
-------------

Since MPI profiling is enabled by default, you typically don't need to explicitly set ``ROCPROFSYS_USE_MPIP=ON``. However, if you need to disable MPI tracing, you can do so with:

.. code-block:: shell

   # MPI profiling is enabled by default - no action needed
   export ROCPROFSYS_TRACE=ON
   export ROCPROFSYS_PROFILE=ON

   # To explicitly disable MPI profiling if needed:
   export ROCPROFSYS_USE_MPIP=OFF

When MPI support is enabled, rocprofiler-systems automatically intercepts MPI function calls using GOTCHA wrappers, allowing you to trace MPI communication patterns and timing.

Usage with MPI Applications
----------------------------

When profiling MPI applications, use ``rocprof-sys-sample`` instead of ``rocprof-sys-instrument`` with runtime instrumentation to avoid compatibility issues with MPI process launching:

.. code-block:: shell

   # Recommended: Using rocprof-sys-sample
   mpirun -n 4 rocprof-sys-sample -- ./my_mpi_app

   # Alternative: Binary rewrite approach
   rocprof-sys-instrument -o my_mpi_app.inst -- ./my_mpi_app
   mpirun -n 4 rocprof-sys-run -- ./my_mpi_app.inst

.. note::

   Runtime instrumentation (``rocprof-sys-instrument`` without ``-o``) requires a fork and ``ptrace``, which is generally incompatible with how MPI applications spawn processes, particularly with OpenMPI.

MPI Profiling Output
--------------------

When MPI profiling is enabled, ROCm Systems Profiler generates:

* **ROCm Profiling Data (rocpd)**: When ``ROCPROFSYS_USE_ROCPD=ON`` is set, profiling data is output in a SQLite3 database format for advanced analysis. See :ref:`rocprof_sys_rocpd_output` for details on this output format. You can visualize MPI operations in a timeline view showing communication patterns, operation durations, and concurrency using `ROCm Optiq <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`_.
* **Perfetto traces**: Visualize MPI operations on a timeline, showing communication patterns, operation durations, and concurrency
* **Timemory profiles**: Statistical summaries of MPI function call counts, total time, and performance metrics
* **Communication data**: Track message sizes, communication volumes, and data movement patterns for point-to-point and collective operations

The traces include detailed information about:

* MPI ranks and communicators
* Message sizes and datatypes
* Source and destination ranks (for point-to-point operations)
* Root ranks (for collective operations)
* Tags for message matching

ROCm Systems Profiler provides automatic output labeling based on MPI rank IDs:

* When full MPI support is enabled (``ROCPROFSYS_USE_MPI=ON``), output files are labeled with the ``MPI_COMM_WORLD`` rank ID
* The ``ROCPROFSYS_USE_PID`` setting controls whether process IDs or MPI rank IDs are used for output labeling

For detailed information on building rocprofiler-systems with MPI support, see the :doc:`installation guide <../install/install>`.

Profiling RCCL
==============

RCCL profiling provides insights into GPU-to-GPU communication patterns and collective operation performance.

.. important::

   Unlike MPI and UCX, RCCL profiling is **disabled by default** and must be explicitly enabled using ``ROCPROFSYS_USE_RCCLP=ON``.

When enabled, rocprofiler-systems captures:

* RCCL API calls (ncclAllReduce, ncclBroadcast, ncclReduce, etc.)
* Communication data volumes and patterns
* Timing information for collective operations

Configuration
-------------

To enable RCCL tracing and profiling:

.. code-block:: shell

   export ROCPROFSYS_USE_RCCLP=ON
   export ROCPROFSYS_TRACE=ON
   export ROCPROFSYS_PROFILE=ON
   export ROCPROFSYS_ROCM_DOMAINS=hip_runtime_api,kernel_dispatch,memory_copy


RCCL Profiling Output
-------------

When RCCL profiling is enabled, rocprofiler-systems generates:

* **ROCm Profiling Data (rocpd)**: When ``ROCPROFSYS_USE_ROCPD=ON`` is set, RCCL profiling
  data is output in a SQLite3 database format with per-GPU device attribution for
  multi-GPU systems. See :ref:`rocprof_sys_rocpd_output` for details on this output
  format. You can visualize RCCL operations in a timeline view showing communication
  patterns, operation durations, and concurrency using `ROCm Optiq
  <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`_.
* **Perfetto traces**: Visualize RCCL operations on a timeline, showing communication patterns, operation durations, and concurrency
* **Communication data**: Track send/receive volumes separately for each GPU in multi-GPU configurations

The image below shows an example of a Perfetto trace with RCCL communication data and API tracing enabled:

.. image:: ../data/rccl-comm-recv.png
   :alt: Perfetto tracks with RCCL Communication Data and API tracing

In the Perfetto trace, you can observe:

* RCCL collective operations on dedicated tracks
* Per-GPU communication volume, direction, and patterns in multi-GPU systems
* Overlap between computation and communication
* Synchronization points and barriers

.. note::

In ROCm versions prior to 7.12, there is a known issue which causes the application to exit with an error. However, the trace data can still be found in the output directory. This issue has been resolved in ROCm 7.12 and later versions.

Profiling UCX
=============

.. important::

   Unlike MPI, UCX profiling is **disabled by default** and must be explicitly enabled using ``ROCPROFSYS_USE_UCX=ON``.

UCX is a low-level communication framework that provides the foundation for efficient data movement in high-performance computing applications. UCX profiling enables detailed analysis of low-level communication primitives, RDMA operations, and transport-layer behavior.

When enabled, rocprofiler-systems automatically intercepts and traces UCX function calls when an application uses UCX — either directly or indirectly through higher-level libraries like MPI or RCCL.

Configuration
-------------

UCX profiling must be explicitly enabled at runtime. To enable UCX tracing and profiling:

.. code-block:: shell

   # UCX profiling is disabled by default - must be explicitly enabled
   export ROCPROFSYS_USE_UCX=ON
   export ROCPROFSYS_TRACE=ON
   export ROCPROFSYS_PROFILE=ON

   # To explicitly disable UCX profiling (default behavior):
   export ROCPROFSYS_USE_UCX=OFF


UCX Operation Categories
-------------------------

rocprofiler-systems captures the following categories of UCX operations:

**Tag-Matching Communication**

Tag-matching provides a flexible mechanism for point-to-point communication with user-defined tags for message matching:

* ``ucp_tag_send_nbx`` - Non-blocking tagged send
* ``ucp_tag_recv_nbx`` - Non-blocking tagged receive
* ``ucp_tag_send_sync_nbx`` - Synchronous tagged send

**Remote Memory Access (RMA)**

RMA operations enable direct access to remote memory without involving the remote CPU:

* ``ucp_put_nbx`` - Non-blocking remote put operation
* ``ucp_get_nbx`` - Non-blocking remote get operation
* ``ucp_put_nbi``, ``ucp_get_nbi`` - Non-blocking implicit operations

**Active Messages**

Active messages provide low-latency communication with handler execution on the receiver:

* ``ucp_am_send_nbx`` - Non-blocking active message send
* ``ucp_am_recv_data_nbx`` - Non-blocking active message receive

**Atomic Operations**

UCX provides various atomic operations for lock-free algorithms and synchronization:

* ``ucp_atomic_add32``, ``ucp_atomic_add64`` - Atomic addition
* ``ucp_atomic_fadd32``, ``ucp_atomic_fadd64`` - Fetch-and-add
* ``ucp_atomic_swap32``, ``ucp_atomic_swap64`` - Atomic swap
* ``ucp_atomic_cswap32``, ``ucp_atomic_cswap64`` - Compare-and-swap

**Stream Operations**

Stream operations provide ordered, connection-oriented communication:

* ``ucp_stream_send_nbx`` - Non-blocking stream send
* ``ucp_stream_recv_nbx`` - Non-blocking stream receive

Usage with UCX Applications
----------------------------

UCX profiling works transparently with applications that use UCX directly or indirectly through higher-level libraries:

.. code-block:: shell

   # Example 1: Direct UCX application
   rocprof-sys-sample -- ./my_ucx_app

   # Example 2: MPI application using UCX as transport
   export ROCPROFSYS_USE_MPIP=ON
   export ROCPROFSYS_USE_UCX=ON
   mpirun -n 4 rocprof-sys-sample -- ./my_mpi_ucx_app

.. note::

   For MPI applications, the presence of UCX libraries alone does not ensure UCX is used at runtime. When MPI is launched with the UCX PML ( ``-mca pml ucx`` ), initialization may fail due to UCX version or transport capability mismatches, causing MPI to fall back to an alternative (non-UCX) communication path.
   Users can verify that UCX is successfully selected at runtime by enabling MPI PML verbosity, for example using ``--mca pml_base_verbose <level>``, which reports the chosen PML during MPI initialization. Additional UCX-specific logging (e.g., ``UCX_LOG_LEVEL=info``) can also be used to confirm that UCX transports are initialized and active.


UCX Profiling Output
---------------------

When UCX profiling is enabled, rocprofiler-systems generates:

* **ROCm Profiling Data (rocpd)**: When ``ROCPROFSYS_USE_ROCPD=ON`` is set, profiling data is output in a SQLite3 database format for advanced analysis. See :ref:`rocprof_sys_rocpd_output` for details on this output format. You can visualize MPI operations in a timeline view showing communication patterns, operation durations, and concurrency using `ROCm Optiq <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`_.
* **Perfetto traces**: Visualize UCX operations on a timeline, showing communication patterns, operation durations, and concurrency
* **Timemory profiles**: Statistical summaries of UCX function call counts, total time, and performance metrics
* **Communication data**: Track message sizes, communication volumes, and data movement patterns

The image below shows an example of a Perfetto trace with UCX communication data and API tracing enabled:

.. image:: ../data/rocprof-sys-ucx.png
   :alt: Perfetto tracks with UCX Communication Data and API tracing

The traces include detailed information about:

* Endpoint handles and worker contexts
* Buffer addresses and data sizes
* Tag values and masks (for tag-matching operations)
* Remote addresses and memory keys (for RMA operations)
* Message IDs and headers (for active messages)


Multi-Layer Communication Analysis
===================================

One of the key strengths of ROCm Systems Profiler is the ability to profile multiple communication layers simultaneously, providing a comprehensive view of the communication stack.

Since MPI profiling is enabled by default while UCX and RCCL require explicit enablement, profiling applications that use multiple layers requires enabling the specific layers you want to trace:

.. code-block:: shell

   # MPI is enabled by default
   # Explicitly enable UCX and RCCL profiling
   export ROCPROFSYS_USE_UCX=ON
   export ROCPROFSYS_USE_RCCLP=ON
   export ROCPROFSYS_TRACE=ON
   export ROCPROFSYS_PROFILE=ON

For complete control over all communication layers:

.. code-block:: shell

   # Explicitly configure all communication runtime profiling
   export ROCPROFSYS_USE_MPIP=ON
   export ROCPROFSYS_USE_RCCLP=ON
   export ROCPROFSYS_USE_UCX=ON
   export ROCPROFSYS_TRACE=ON
   export ROCPROFSYS_PROFILE=ON

This multi-layer profiling enables:

* **Understanding communication hierarchies**: See how high-level MPI calls translate to lower-level UCX operations
* **Identifying optimization opportunities**: Detect inefficiencies at different abstraction layers
* **Analyzing GPU-CPU coordination**: Observe interactions between CPU-based MPI communication and GPU-based RCCL collectives
* **Performance debugging**: Trace the full path of data movement from application-level calls to transport-level operations

Best Practices
==============

When profiling communication-intensive applications, consider the following recommendations:

**Start with High-Level Profiling**

* Begin by enabling only MPI or RCCL profiling to understand the overall communication patterns
* Use flat profiles to identify high-overhead communication operations
* Look for functions with high call counts or large cumulative times

**Add Lower-Level Details**

* Enable UCX profiling (``ROCPROFSYS_USE_UCX=ON``) to understand transport-layer behavior and RDMA utilization
* Use hierarchical profiles to correlate high-level operations with low-level primitives

**Minimize Overhead**

* Tracing communication operations incurs runtime overhead from intercepting each communication call and recording detailed metadata, particularly for high-frequency MPI/UCX communication paths; use sampling mode when precise traces are not required as statistical sampling can provide sufficient insights without the full overhead of complete tracing..
* For large-scale runs, consider enabling profiling on a subset of ranks
* Use ``ROCPROFSYS_SAMPLING_FREQ`` to control sampling rate and balance detail vs. overhead

**Analyze in Context**

* Combine communication profiling with GPU profiling (``ROCPROFSYS_ROCM_DOMAINS``) for heterogeneous applications
* Use ``ROCPROFSYS_TIMEMORY_COMPONENTS`` to add CPU metrics and memory statistics
* Enable process sampling (``ROCPROFSYS_USE_PROCESS_SAMPLING``) for system-level insights

**Leverage Visualization**

* Use the `Rocm Optiq <https://rocm.docs.amd.com/projects/roc-optiq/en/latest/what-is-optiq.html>`_ for rocpd database output and the Perfetto UI for perfetto traces, to visualize communication timelines and identify bottlenecks
* Look for communication/computation overlap opportunities
* Identify load imbalance by comparing traces across ranks

Example Configuration
=====================

Here is a complete configuration example for comprehensive communication profiling:

.. code-block:: shell

   # Enable all communication runtime profiling
   ROCPROFSYS_USE_MPIP                = ON
   ROCPROFSYS_USE_RCCLP               = ON
   ROCPROFSYS_USE_UCX                 = ON

   # Enable tracing and profiling
   ROCPROFSYS_TRACE                   = ON
   ROCPROFSYS_PROFILE                 = ON

   # GPU profiling
   ROCPROFSYS_ROCM_DOMAINS            = hip_runtime_api,kernel_dispatch,memory_copy

   # Sampling configuration
   ROCPROFSYS_USE_SAMPLING            = ON
   ROCPROFSYS_SAMPLING_FREQ           = 50

   # Output configuration
   ROCPROFSYS_OUTPUT_PATH             = comm-profile-output
   ROCPROFSYS_OUTPUT_PREFIX           = %tag%/
   ROCPROFSYS_USE_PID                 = OFF

   # Additional metrics
   ROCPROFSYS_TIMEMORY_COMPONENTS     = wall_clock peak_rss

   # Verbosity
   ROCPROFSYS_VERBOSE                 = 1

This configuration can be saved to a file (for example, ``comm-profile.cfg``) and loaded using:

.. code-block:: shell

   export ROCPROFSYS_CONFIG_FILE=/path/to/comm-profile.cfg

For additional configuration options and details, see :doc:`Configuring runtime options <./configuring-runtime-options>`.
