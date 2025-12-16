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
    :widths: 70,30

    * - **Environment variable**
      - **Value**

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
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``RCCL_LOG_LEVEL``
        | Controls RCCL logging verbosity.
      - | Integer value (default: ``1``)
        | Higher values increase logging detail

    * - | ``NCCL_DEBUG_SUBSYS``
        | Controls which subsystems generate debug output.
      - | Comma-separated list of subsystems (e.g., ``INIT,COLL``)
        | Prefix with ``^`` to invert selection

Algorithm and protocol control
==============================

The algorithm and protocol control environment variables for RCCL are
collected in the following table.

.. list-table::
    :header-rows: 1
    :widths: 70,30

    * - **Environment variable**
      - **Value**

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
    :widths: 70,30

    * - **Environment variable**
      - **Value**

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
    :widths: 70,30

    * - **Environment variable**
      - **Value**

    * - | ``CUDA_LAUNCH_BLOCKING``
        | Controls CUDA kernel launch blocking behavior.
      - | ``0``: Non-blocking launches
        | ``1`` or non-zero: Blocking launches

    * - | ``NCCL_COMM_ID``
        | Enables multi-process mode in test applications.
      - | Any non-empty value enables multi-process mode
        | Used with test executables for distributed testing
