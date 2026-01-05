.. meta::
   :description: How to use the RCCL Tuner plugin API
   :keywords: RCCL, ROCm, library, API, Tuner, plugin

.. _using-rccl-tuner-plugin:

*******************************
Using the RCCL Tuner plugin API
*******************************

An external plugin enables users to hand-tailor the selection of an algorithm,
protocol, and number of channels (thread blocks) based on an input configuration specifying the
message size, number of nodes and GPUs, and link types (for instance, PCIe, XGMI, or NET).
One advantage of this plugin is that each user can create and maintain their own hand-tailored tuner
without relying on RCCL to develop and maintain it. This topic describes the API required to implement
an external tuner plugin for RCCL.

The following usage notes are relevant when using the RCCL Tuner plugin API:

*  The API allows partial outputs: tuners can set only the algorithm and protocol and let RCCL set the remaining fields,
   such as the number of channels.
*  If ``getCollInfo()`` fails, RCCL uses its default internal mechanisms to determine the best collective configuration.
*  ``getCollInfo`` is called for each collective invocation per communicator, so special care
   must be taken to avoid introducing excessive latency.
*  The supported RCCL algorithms are ``NCCL_ALGO_TREE``, and ``NCCL_ALGO_RING``.
*  The supported RCCL protocols are ``NCCL_PROTO_SIMPLE``, ``NCCL_PROTO_LL`` and ``NCCL_PROTO_LL128``.

   *  Until support is present for network collectives, use the example in the ``pluginGetCollInfo`` API implementation
      to ignore other algorithms as follows:

      .. code-block:: cpp

         if ((a == NCCL_ALGO_COLLNET_DIRECT || a == NCCL_ALGO_COLLNET_CHAIN) && collNetSupport != 1) continue;
         if ((a == NCCL_ALGO_NVLS || a == NCCL_ALGO_NVLS_TREE) && nvlsSupport != 1) continue;
         if (a == NCCL_ALGO_NVLS && collNetSupport != 1) continue;

.. note::
   
   The `example plugin <https://github.com/ROCm/rccl/blob/develop/ext-tuner/example/plugin.c>`_
   uses math models to approximate the bandwidth and latency of the available selection of algorithms and protocols
   and select the one with the lowest calculated latency. It is customized for the AMD Instinct MI300 accelerators and RoCEv2 networks
   on a limited number of nodes. This example, which is intended for demonstration purposes only, is not meant to be inclusive of all potential AMD GPUs and network configuration.

API description
================

To build a custom tuner, implement the ``ncclTuner_v1_t`` structure.

Structure: ncclTuner_v1_t
---------------------------

**Fields**

*  ``name``
  
   *  **Type**: ``const char*``
   *  **Description**: The name of the tuner, which can be used for logging purposes when ``NCCL_DEBUG=info`` and ``NCCL_DEBUG_SUBSYS=tune`` are set.

**Functions**

*  ``init`` (called upon communicator initialization with ``ncclCommInitRank``)

   Initializes the tuner states. Each communicator initializes its tuner. ``nNodes`` x ``nRanks`` = the total number of GPUs participating in the collective communication.

   *  **Parameters**:

      * ``nRanks`` (``size_t``): The number of devices (GPUs).
      * ``nNodes`` (``size_t``): The number of operating system nodes (physical nodes or VMs).
      * ``logFunction`` (``ncclDebugLogger_t``): A log function for certain debugging info.

   *  **Return**:

      *  **Type**: ``ncclResult_t``
      *  **Description**: The result of the initialization.

*  ``getCollInfo`` (called for each collective call per communicator)

   Retrieves information about the collective algorithm, protocol, and number of channels for the given input parameters.

   *  **Parameters**:

      * ``collType`` (``ncclFunc_t``): The collective type, for example, ``allreduce``, ``allgather``, etc.
      * ``nBytes`` (``size_t``): The size of the collective in bytes.
      * ``collNetSupport`` (``int``): Whether ``collNet`` supports this type.
      * ``nvlsSupport`` (``int``): Whether NVLink SHARP supports this type.
      * ``numPipeOps`` (``int``): The number of operations in the group.
  
   *  **Outputs**:

      * ``algorithm`` (``int*``): The selected algorithm to be used for the given collective.
      * ``protocol`` (``int*``): The selected protocol to be used for the given collective.
      * ``nChannels`` (``int*``): The number of channels (and SMs) to be used.
     
   *  **Description**:

      If ``getCollInfo()`` does not return ``ncclSuccess``, RCCL falls back to its default tuning for the given collective.
      The tuner is allowed to leave fields unset, in which case RCCL automatically sets those fields.

   *  **Return**:

      *  **Type**: ``ncclResult_t``
      *  **Description**: The result of the operation.

*  ``destroy`` (called upon communicator finalization with ``ncclCommFinalize``)

   Terminates the plugin and cleans up any resources allocated by the tuner.

   *  **Return**:

      *  **Type**: ``ncclResult_t`` 
      *  **Description**: The result of the cleanup process.

Build and usage instructions
============================

To use the external plugin, implement the desired algorithm and protocol selection technique using the API described above.
As a reference, the `following example <https://github.com/ROCm/rccl/blob/develop/ext-tuner/example/plugin.c>`_ is based on the
MI300 tuning table by default.

Building and using the example libnccl-tuner.so file
-----------------------------------------------------

#. Build the ``libnccl-tuner.so`` file following `the example Makefile <https://github.com/ROCm/rccl/blob/develop/ext-tuner/example/Makefile>`_.

   .. code-block:: shell

      cd $RCCL_HOME/ext-tuner/example/
      make

#. Tell RCCL to use the custom ``libnccl-tuner.so`` file by setting the following environment variable
   to the file path:

   .. code-block:: shell

      export NCCL_TUNER_PLUGIN=$RCCL_HOME/ext-tuner/example/libnccl-tuner.so
