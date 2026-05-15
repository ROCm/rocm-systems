.. meta::
  :description: rocSHMEM environment variables reference
  :keywords: rocSHMEM, ROCm, API, environment variables, environment, reference

.. _rocshmem-api-env-variables:

********************************************************************
rocSHMEM environment variables
********************************************************************

This section describes the important environment variables used to
control the behavior of rocSHMEM.

.. list-table::
    :header-rows: 1
    :widths: 35,14,51

    * - **Environment variable**
      - **Default value**
      - **Value**

    * - | ``ROCSHMEM_HEAP_SIZE``
        | Defines the size of the rocSHMEM symmetric heap in bytes (per PE).
      - ``1073741824`` (1 GB)
      - | Size in bytes (per PE).
        | Note: the heap is on GPU memory.

    * - | ``ROCSHMEM_MAX_NUM_CONTEXTS``
        | Defines the number of contexts an application can use.
      - ``32``
      - Maximum number of contexts.

    * - | ``ROCSHMEM_MAX_NUM_TEAMS``
        | Defines the number of teams an application can use.
      - ``40``
      - Maximum number of teams.

    * - | ``ROCSHMEM_BACKEND``
        | When rocSHMEM is compiled for all backends, this enviroment variable
        | selects which backend to execute. The default value is an empty string and rocSHMEM auto-selects the most appropriate backend.
      - `` ``
      - | ``ipc``: IPC Backend
        | ``ro``: Reverse Offload Backend
        | ``gda``: GPU Direct Async Backend

    * - | ``ROCSHMEM_UNIQUEID_WITH_MPI``
        | Defines whether rocSHMEM is expected to use MPI when using the uniqueId based initialization.
      - ``0``
      - | ``0``: Do not use MPI.
        | ``1``: Use MPI.

    * - | ``ROCSHMEM_DISABLE_MIXED_IPC``
        | Defines whether to force using the network conduit even when IPC is available.
      - ``0``
      - | ``0``: Use IPC when available.
        | ``1``: Force network conduit.

    * - | ``ROCSHMEM_USE_IB_HCA``
        | Forces the NIC that this PE uses. When this value is set NIC auto-detection and mapping is disabled, the NIC specified in the variable
        | will be selected. The default value is an empty string and rocSHMEM auto-detects the most appropriate NIC.
      - `` ``
      - | Example value: ``bnxt_re0``

    * - | ``ROCSHMEM_HCA_LIST``
        | Comma separated list of NIC names that can be used by rocSHMEM. Unlike ``ROCSHMEM_USE_IB_HCA``, when this variable is set,
        | NIC auto-detection and mapping still executes, but NICs that are not in the list are discarded before auto-detection runs.
        | Prefixing the list with ``^`` turns the list in an *exclude* list, NICs that are in the list are discarded before auto-detection runs.
        | The default value is an empty string and rocSHMEM auto-detects the most appropriate NIC.
      - `` ``
      - | Example value: ``bnxt_re1,bnxt_re11``, ``^mlx5_0,mlx5_3``

    * - | ``ROCSHMEM_BOOTSTRAP_SOCKET_IFNAME``
        | Chooses the interface to bootstrap rocSHMEM with.
        | Only valid when not using MPI.
        | The default value is an empty string and rocSHMEM auto-detects the most appropriate interface.
      - `` ``
      - | Example value: ``eno8303``

    * - | ``ROCSHMEM_GDA_PROVIDER``
        | When rocSHMEM is compiled with support for multiple NIC vendors,
        | the enviroment variable selects the desired provider.
        | The default value is an empty string and rocSHMEM auto-detects the most appropriate NIC.
      - `` ``
      - | ``bnxt``: Broadcom Thor 2
        | ``pensando``: AMD Pensando Pollara
        | ``ionic``: AMD Pensando Pollara (alias)
        | ``mlx5``: Mellanox ConnectX-7

    * - | ``ROCSHMEM_GDA_ALTERNATE_QP_PORTS``
        | Enables or disables alternating QP mappings across rocSHMEM contexts.
      - ``1``
      - | ``0``: Disabled.
        | ``1``: Enabled. This helps saturate bandwidth on multiport bonded interfaces.

    * - | ``ROCSHMEM_GDA_TRAFFIC_CLASS``
        | When using an NIC with an Ethernet link layer, this sets the traffic class for the QPs.
      - ``0``
      - The traffic class number.

    * - | ``ROCSHMEM_GDA_PCIE_RELAXED_ORDERING``
        | Enables PCIe Relaxed Ordering when registering the symmetric heap with the RDMA NICs.
      - ``0``
      - | ``0``: Disabled.
        | ``1``: Enabled.

    * - | ``ROCSHMEM_GDA_ENABLE_DMABUF``
        | Enable dmabuf support for memory registration.
      - ``0``
      - | ``0``: Disabled.
        | ``1``: Enabled.

