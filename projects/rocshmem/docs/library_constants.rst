.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-library-constants:

------------------
Library constants
------------------

Version and vendor constants
----------------------------

.. c:macro:: ROCSHMEM_MAJOR_VERSION

  An integer constant whose value is the major version of the
  OpenSHMEM specification supported by this library.
  Returned by ``rocshmem_info_get_version``.

.. c:macro:: ROCSHMEM_MINOR_VERSION

  An integer constant whose value is the minor version of the
  OpenSHMEM specification supported by this library.
  Returned by ``rocshmem_info_get_version``.

.. c:macro:: ROCSHMEM_MAX_NAME_LEN

  An integer constant that specifies the minimum size (in bytes) of the
  buffer passed to ``rocshmem_info_get_name``.  The value includes space
  for the null terminator.

.. c:macro:: ROCSHMEM_VENDOR_STRING

  A string constant that contains the vendor-defined name of the library.
  Its length, including the null terminator, is at most
  ``ROCSHMEM_MAX_NAME_LEN`` bytes.
  Copied into the caller's buffer by ``rocshmem_info_get_name``.

.. c:macro:: ROCSHMEM_VERSION

  A string constant that contains the full version of the library
  in ``"major.minor.patch"`` format (e.g. ``"3.3.0"``).

.. c:macro:: ROCSHMEM_VENDOR_MAJOR_VERSION

  An integer constant whose value is the vendor major version of rocSHMEM.
  Returned by ``rocshmem_vendor_get_version_info``.

.. c:macro:: ROCSHMEM_VENDOR_MINOR_VERSION

  An integer constant whose value is the vendor minor version of rocSHMEM.
  Returned by ``rocshmem_vendor_get_version_info``.

.. c:macro:: ROCSHMEM_VENDOR_PATCH_VERSION

  An integer constant whose value is the vendor patch version of rocSHMEM.
  Returned by ``rocshmem_vendor_get_version_info``.

Status and operation codes
--------------------------

.. cpp:enum:: ROCSHMEM_STATUS

  Return codes used by routines that report success or failure as an enum.

  .. cpp:enumerator:: ROCSHMEM_SUCCESS

    The operation completed successfully. Value ``0``.

  .. cpp:enumerator:: ROCSHMEM_ERROR

    The operation failed. Value ``1``.

.. cpp:enum:: ROCSHMEM_OP

  Reduction operators used internally by the collective routines. Application
  code normally selects the operator through the ``OPNAME`` component of a
  reduction routine's name rather than by passing one of these values.

  Enumerators: ``ROCSHMEM_SUM``, ``ROCSHMEM_MAX``, ``ROCSHMEM_MIN``,
  ``ROCSHMEM_PROD``, ``ROCSHMEM_AND``, ``ROCSHMEM_OR``, ``ROCSHMEM_XOR``,
  ``ROCSHMEM_REPLACE``.

Initialization constants
------------------------

.. c:macro:: ROCSHMEM_INIT_WITH_MPI_COMM

  Passed as the ``flags`` argument of ``rocshmem_init_attr`` to initialize
  from an MPI communicator. Value ``0``.

.. c:macro:: ROCSHMEM_INIT_WITH_UNIQUEID

  Passed as the ``flags`` argument of ``rocshmem_init_attr`` to initialize
  from a unique ID, with no MPI dependency. Value ``1``.

.. c:macro:: ROCSHMEM_UNIQUE_ID_BYTES

  Size in bytes of a ``rocshmem_uniqueid_t``, the byte array that uniquely
  identifies a process during unique-ID bootstrap. Value ``128``.

Thread support levels
---------------------

.. cpp:enum:: rocshmem_thread_ops

  Thread support levels, reported by ``rocshmem_query_thread``.

  .. cpp:enumerator:: ROCSHMEM_THREAD_SINGLE

    Only one thread may call into rocSHMEM.

  .. cpp:enumerator:: ROCSHMEM_THREAD_FUNNELED

    Multiple threads exist, but only the thread that initialized rocSHMEM
    calls into it.

  .. cpp:enumerator:: ROCSHMEM_THREAD_WG_FUNNELED

    Calls are funneled through a single thread per work-group.

  .. cpp:enumerator:: ROCSHMEM_THREAD_SERIALIZED

    Multiple threads may call into rocSHMEM, but not concurrently.

  .. cpp:enumerator:: ROCSHMEM_THREAD_MULTIPLE

    Multiple threads may call into rocSHMEM concurrently without external
    synchronization.

Context option constants
------------------------

``ROCSHMEM_CTX_ZERO``, ``ROCSHMEM_CTX_NOSTORE``, ``ROCSHMEM_CTX_SERIALIZED``,
``ROCSHMEM_CTX_WG_PRIVATE``, and ``ROCSHMEM_CTX_SHARED`` are the bit flags
accepted by the context creation routines. See :ref:`rocshmem-api-ctx` for
their meanings and values.

Team constants
--------------

``ROCSHMEM_TEAM_WORLD``, ``ROCSHMEM_TEAM_SHARED``, and
``ROCSHMEM_TEAM_INVALID`` are the predefined team handles;
``ROCSHMEM_TEAM_DEFAULT_CONFIGS`` and ``ROCSHMEM_TEAM_NUM_CONTEXTS`` are the
team configuration masks. See :ref:`rocshmem-api-teams`.

Comparison and signal constants
-------------------------------

``ROCSHMEM_CMP_EQ``, ``ROCSHMEM_CMP_NE``, ``ROCSHMEM_CMP_GT``,
``ROCSHMEM_CMP_GE``, ``ROCSHMEM_CMP_LT``, and ``ROCSHMEM_CMP_LE`` are the
point-to-point comparison constants; see :ref:`rocshmem-api-pt2pt-sync`.

``ROCSHMEM_SIGNAL_SET`` and ``ROCSHMEM_SIGNAL_ADD`` are the signal update
operators; see :ref:`rocshmem-api-sigops`.

Collective sizing constants
---------------------------

These constants size the symmetric work and synchronization arrays used by the
OpenSHMEM-style collective interfaces.

.. list-table:: Collective sizing constants
    :widths: 40 15 45
    :header-rows: 1

    * - Constant
      - Value
      - Description
    * - ``ROCSHMEM_REDUCE_MIN_WRKDATA_SIZE``
      - ``1024``
      - Minimum number of elements in a reduction work array.
    * - ``ROCSHMEM_BARRIER_SYNC_SIZE``
      - ``256``
      - Number of elements in a barrier synchronization array.
    * - ``ROCSHMEM_REDUCE_SYNC_SIZE``
      - ``256``
      - Number of elements in a reduction synchronization array.
    * - ``ROCSHMEM_BCAST_SYNC_SIZE``
      - ``256``
      - Number of elements in a broadcast synchronization array. Equal to
        ``ROCSHMEM_BARRIER_SYNC_SIZE``.
    * - ``ROCSHMEM_ALLTOALL_SYNC_SIZE``
      - ``257``
      - Number of elements in an all-to-all synchronization array.
    * - ``ROCSHMEM_FCOLLECT_SYNC_SIZE``
      - ``257``
      - Number of elements in an fcollect synchronization array. Equal to
        ``ROCSHMEM_ALLTOALL_SYNC_SIZE``.
    * - ``ROCSHMEM_SYNC_VALUE``
      - ``0``
      - Value with which every element of a synchronization array must be
        initialized before first use.

Tile element type constants
---------------------------

.. cpp:enum:: ROCSHMEM_TILE_ELEMENT_TYPE

  Element type tags carried in the ``flags`` word of the tile routines. The
  tile API derives the correct tag from the tensor's element type, so
  application code does not normally reference these values directly.

  Enumerators: ``ROCSHMEM_TILE_ELEMENT_UNKNOWN`` (``0``),
  ``ROCSHMEM_TILE_ELEMENT_INT8``, ``ROCSHMEM_TILE_ELEMENT_UINT8``,
  ``ROCSHMEM_TILE_ELEMENT_INT16``, ``ROCSHMEM_TILE_ELEMENT_UINT16``,
  ``ROCSHMEM_TILE_ELEMENT_INT32``, ``ROCSHMEM_TILE_ELEMENT_UINT32``,
  ``ROCSHMEM_TILE_ELEMENT_INT64``, ``ROCSHMEM_TILE_ELEMENT_UINT64``,
  ``ROCSHMEM_TILE_ELEMENT_FLOAT``, ``ROCSHMEM_TILE_ELEMENT_DOUBLE``,
  ``ROCSHMEM_TILE_ELEMENT_SHORT``, ``ROCSHMEM_TILE_ELEMENT_USHORT``,
  ``ROCSHMEM_TILE_ELEMENT_INT``, ``ROCSHMEM_TILE_ELEMENT_UINT``,
  ``ROCSHMEM_TILE_ELEMENT_LONG``, ``ROCSHMEM_TILE_ELEMENT_ULONG``,
  ``ROCSHMEM_TILE_ELEMENT_LONGLONG``, ``ROCSHMEM_TILE_ELEMENT_ULONGLONG``
  (``18``).

.. c:macro:: ROCSHMEM_TILE_ELEMENT_TYPE_SHIFT

  Bit position at which the element type tag is packed into the tile ``flags``
  word. Value ``56``.

.. c:macro:: ROCSHMEM_TILE_ELEMENT_TYPE_MASK

  Mask selecting the element type tag within the tile ``flags`` word. Value
  ``0xff`` shifted left by ``ROCSHMEM_TILE_ELEMENT_TYPE_SHIFT``.
