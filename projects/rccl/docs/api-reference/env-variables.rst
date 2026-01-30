.. meta::
   :description: RCCL is a stand-alone library that provides multi-GPU and multi-node collective communication primitives optimized for AMD GPUs
   :keywords: RCCL, ROCm, library, API, reference, environment variable, environment

.. _env-variables:

********************************************************************
RCCL environment variables
********************************************************************

This section describes the most important RCCL environment variables,
which are grouped by functionality.

Configuration and setup
========================

The configuration and setup environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_CONF_FILE``
        | Specifies the path to the RCCL configuration file.
      - | String path to configuration file
        | Default: ``~/.rccl.conf`` or ``/etc/rccl.conf``

    * - | ``NCCL_HOSTID``
        | Sets the host identifier for multi-node communication.
      - | String value for host identification
        | Used for host hash generation

Logging and debugging
=====================

The logging and debugging environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 35,65

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_DEBUG``
        | Controls debug logging in RCCL for troubleshooting and monitoring collective communication operations. 
      - | These are the logging levels in RCCL set via ``NCCL_DEBUG``. Each logging level contains all logging for levels below it. The default logging level is ``ERROR``.
        |
        | ``NONE``: No logging is printed.
        | ``ERROR``: These messages report when a fatal condition has occurred in RCCL and the operation can't continue.
        | ``VERSION``: ``librccl`` version info is printed during the initialization phase.
        | ``WARN``: Prints warnings about unusual conditions that could lead to unexpected results.
        | ``INFO``: Prints standard logging messages about status and operations performed.
        | ``ABORT``: Unused.
        | ``TRACE``: Prints trace-level logging of function calls and parameters. Only active when ``librccl`` is built using ``ENABLE_TRACE``.

    * - | ``NCCL_DEBUG_SUBSYS``
        | Controls which subsystems generate debug output.
      - | These are the logging subsystems set via ``NCCL_DEBUG_SUBSYS``. These can be set as a comma-separated list, and can be inverted using the ``^`` prefix. The default subsystem set is ``INIT``, ``BOOTSTRAP``, and ``ENV``.
        |
        | ``INIT``: Prints during the initialization phase.
        | ``COLL``: Prints during execution of collectives.
        | ``P2P``: Prints logs related to peer-to-peer setup or communication.
        | ``SHM``: Prints logs related to shared memory.
        | ``NET``: Prints logs related to network setup or communication.
        | ``GRAPH``: Prints logs related to parsing the topology of the network.
        | ``TUNING``: Prints logs related to the tuner plugin.
        | ``ENV``: Prints logs related to environment variables.
        | ``ALLOC``: Prints logs related to memory allocation.
        | ``CALL``: Prints logs for function calls (``TRACE`` only).
        | ``PROXY``: Prints logs related to the proxy thread.
        | ``NVLS``: Not valid for AMD/RCCL.
        | ``BOOTSTRAP``: Prints logs related to the bootstrapping phase of initialization.
        | ``REG``: Prints logs related to registration and deregistration of transport initialization.
        | ``PROFILE``: Prints logs related to the profiling/timing info.
        | ``RAS``: Prints logs related to RAS.
        | ``VERBS``: Prints logs related to IB/Verbs.
        | ``ALL``: Activates all logging subsystems.

    * - | ``NCCL_WARN_ENABLE_DEBUG_INFO``
        | Converts all ``WARN`` level logs to ``INFO`` level logs.
      - | ``0``: Default value. Variable is not enabled.
        | ``1``: Enable the variable.

    * - | ``NCCL_DEBUG_TIMESTAMP_LEVELS``
        | The timestamp levels for ``NCCL_DEBUG``.
      - | A set of ``NCCL_DEBUG`` levels can have a timestamp prepended set as a comma-separated list which can be inverted using the ``^`` prefix. The default set is ``WARN``.

    * - | ``NCCL_DEBUG_TIMESTAMP_FORMAT``
        | The timestamp format for ``NCCL_DEBUG``.
      - | Set the format of the timestamp in ``printf`` style. The default format is ``"[%F %T] "``.

    * - | ``NCCL_DEBUG_FILE``
        | Write logs to a file rather than ``stdout``.
      - | The filename can be formatted using ``%h`` for hostname, ``%p`` for pid, and ``%%`` to escape the ``%`` character. It is recommended to use ``%p`` to output to individual files per pid to avoid mixing or potentially overwriting the output. Example usage: ``NCCL_DEBUG_FILE=debugfile.%h.%p``

Algorithm and protocol control
==============================

The algorithm and protocol control environment variables for RCCL are
collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_ALGO``
        | Forces specific algorithm selection for collectives.
      - | Algorithm name string
        | Used to override automatic algorithm selection

    * - | ``NCCL_PROTO``
        | Forces specific protocol selection for communication.
      - | Protocol name string
        | Used to override automatic protocol selection

Network and topology
====================

The network and topology environment variables for RCCL are collected
in the following table.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``NCCL_IB_HCA``
        | Specifies InfiniBand device:port to use.
      - | Device specification string
        | Prefix with ``^`` for exclusion, ``=`` for exact match

    * - | ``NCCL_IB_GID_INDEX``
        | Defines the Global ID index used in RoCE mode.
      - | Integer value (default: ``-1``)
        | See InfiniBand ``show_gids`` command for valid values

    * - | ``NCCL_SOCKET_IFNAME``
        | Specifies which IP interfaces to use for communication.
      - | Interface prefix string or list
        | Multiple prefixes separated by ``,``
        | Prefix with ``^`` for exclusion, ``=`` for exact match
        | Example: ``eth`` (all eth interfaces), ``=eth0`` (exact match)

    * - | ``NCCL_SOCKET_FAMILY``
        | Forces IPv4/IPv6 interface selection.
      - | ``AF_INET``: Force IPv4
        | ``AF_INET6``: Force IPv6
        | Unset: Use first available

    * - | ``NCCL_NET_MERGE_LEVEL``
        | Controls network device merging behavior.
      - | Integer value specifying merge level
        | Default: ``PATH_PORT``

    * - | ``NCCL_NET_FORCE_MERGE``
        | Forces merging of network devices.
      - | String specifying forced merge configuration

    * - | ``NCCL_RINGS``
        | Defines custom ring topology.
      - | Ring topology specification string
        | Overrides automatic topology detection

    * - | ``RCCL_TREES``
        | Defines custom tree topology.
      - | Tree topology specification string
        | Alternative to ring topology

    * - | ``NCCL_RINGS_REMAP``
        | Controls ring remapping for specific topologies.
      - | Remapping specification string
        | Used with Rome 4P2H topology

Development and testing (advanced)
==================================

The development and testing environment variables for RCCL are
collected in the following table. These variables are primarily
intended for debugging and development purposes.

.. list-table::
    :header-rows: 1
    :widths: 40,60

    * - **Environment variable**
      - **Values**

    * - | ``CUDA_LAUNCH_BLOCKING``
        | Controls CUDA kernel launch blocking behavior.
      - | ``0``: Non-blocking launches
        | ``1`` or non-zero: Blocking launches

    * - | ``NCCL_COMM_ID``
        | Enables multi-process mode in test applications.
      - | Any non-empty value enables multi-process mode
        | Used with test executables for distributed testing
