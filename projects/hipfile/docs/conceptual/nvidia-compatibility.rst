.. meta::
   :description: How hipfile provides NVIDIA cuFile compatibility through a type conversion layer and wrapper backend when built for the NVIDIA platform.
   :keywords: hipfile, cuFile, NVIDIA, compatibility, type conversion, HIPIFY, ROCm, portability

===========================
NVIDIA cuFile compatibility
===========================

hipfile supports building for NVIDIA platforms by wrapping the NVIDIA cuFile library. When you configure the build with ``CMAKE_HIP_PLATFORM=nvidia``, hipfile replaces its AMD backend with a thin wrapper that delegates all I/O operations to cuFile. This enables applications written against the hipfile API to run on both AMD and NVIDIA GPUs without source-level changes.

For instructions on configuring a build for the NVIDIA platform, see :doc:`/install/build-from-source`.

How the NVIDIA backend works
****************************

The NVIDIA backend is activated at build time through the CMake option ``CMAKE_HIP_PLATFORM=nvidia``, which sets the ``AIS_BUILD_NVIDIA_DETAIL`` build flag. When this flag is active, the library compiles the sources under ``src/nvidia_detail/`` instead of ``src/amd_detail/``. The resulting ``libhipfile`` shared library links against NVIDIA's ``cuFile`` library and forwards each hipfile API call to the corresponding cuFile function.

On AMD platforms, hipfile uses its own fastpath and fallback backends for direct-to-GPU I/O. On NVIDIA platforms, all I/O is handled by cuFile. The application-facing API (``hipfile.h``) remains identical on both platforms.

Type conversion layer
*********************

The header ``src/nvidia_detail/hipfile-cufile.h`` defines a bidirectional type conversion layer between hipfile types and cuFile types. This layer translates every enum, struct, and error code so that the NVIDIA backend can accept hipfile parameters and return hipfile results while internally communicating with cuFile.

The conversions fall into two categories:

cuFile to hipfile
-----------------

These functions convert cuFile types into their hipfile equivalents, used when returning results from cuFile calls back to the application:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Conversion function
     - Purpose
   * - ``toHipFileOpError``
     - Convert ``CUfileOpError`` to ``hipFileOpError_t``
   * - ``toHipFileError``
     - Convert ``CUfileError_t`` to ``hipFileError_t``
   * - ``toHipFileDriverStatusFlags``
     - Convert driver status flag enums
   * - ``toHipFileDriverControlFlags``
     - Convert driver control flag enums
   * - ``toHipFileFeatureFlags``
     - Convert feature flag enums
   * - ``toHipFileFileHandleType``
     - Convert file handle type enums
   * - ``toHipFileDriverProps``
     - Convert ``CUfileDrvProps_t`` to ``hipFileDriverProps_t``
   * - ``toHipFileRDMAInfo``
     - Convert RDMA info structs
   * - ``toHipFileDescr``
     - Convert file descriptor structs
   * - ``toHipFileOpcode``
     - Convert batch opcode enums
   * - ``toHipFileStatus``
     - Convert batch status enums
   * - ``toHipFileBatchMode``
     - Convert batch mode enums
   * - ``toHipFileIOParams``
     - Convert batch I/O parameter structs
   * - ``toHipFileIOEvents``
     - Convert batch I/O event structs

hipfile to cuFile
-----------------

These functions convert hipfile types into cuFile types, used when passing application parameters into cuFile calls:

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Conversion function
     - Purpose
   * - ``toCUfileOpError``
     - Convert ``hipFileOpError_t`` to ``CUfileOpError``
   * - ``toCUfileError``
     - Convert ``hipFileError_t`` to ``CUfileError_t``
   * - ``toCUfileDriverStatusFlags``
     - Convert driver status flag enums
   * - ``toCUfileDriverControlFlags``
     - Convert driver control flag enums
   * - ``toCUfileFeatureFlags``
     - Convert feature flag enums
   * - ``toCUfileFileHandleType``
     - Convert file handle type enums
   * - ``toCUfileDrvProps``
     - Convert driver properties structs
   * - ``toCufileRDMAInfo``
     - Convert RDMA info structs
   * - ``toCUfileDescr``
     - Convert file descriptor structs
   * - ``toCUfileOpcode``
     - Convert batch opcode enums
   * - ``toCUfileStatus``
     - Convert batch status enums
   * - ``toCUfileBatchMode``
     - Convert batch mode enums
   * - ``toCUfileIOParams``
     - Convert batch I/O parameter structs
   * - ``toCUfileIOEvents``
     - Convert batch I/O event structs
   * - ``toCUFileSizeTConfigParameter``
     - Convert size configuration parameter enums
   * - ``toCUFileBoolConfigParameter``
     - Convert boolean configuration parameter enums
   * - ``toCUFileStringConfigParameter``
     - Convert string configuration parameter enums

API parity goals
****************

hipfile aims for full API parity with cuFile. The public header ``hipfile.h`` defines types, enums, error codes, and function signatures that mirror cuFile's API surface. This includes:

- **Error handling**: ``hipFileOpError_t`` values map one-to-one with cuFile error codes. Error codes are offset by ``HIPFILE_BASE_ERR`` (5000) to avoid collisions.
- **Driver API**: ``hipFileDriverOpen()``, ``hipFileDriverClose()``, ``hipFileDriverGetProperties()``, and related configuration functions correspond to their cuFile equivalents.
- **File and buffer registration**: ``hipFileHandleRegister()``, ``hipFileBufRegister()``, and their deregistration counterparts follow the same registration model as cuFile.
- **Synchronous I/O**: ``hipFileRead()`` and ``hipFileWrite()`` have the same parameter layout and return-value semantics.
- **Batch and async APIs**: Batch operations (``hipFileBatchIOSetUp``, ``hipFileBatchIOSubmit``, etc.) and async stream operations follow the cuFile model.
- **Properties and configuration parameters**: The ``hipFileDriverProps_t`` struct and the configuration parameter enums (``hipFileSizeTConfigParameter_t``, ``hipFileBoolConfigParameter_t``, ``hipFileStringConfigParameter_t``) have corresponding cuFile types.

.. note::

   Some ``hipFileDriverProps_t`` fields have platform-specific alignment requirements. For example, ``poll_thresh_size`` and ``per_buffer_cache_size`` must be multiples of 4K on NVIDIA, and ``max_direct_io_size`` must be a multiple of 64K on NVIDIA. These constraints are documented in the ``hipfile.h`` header comments.

Known differences
*****************

The AMD and NVIDIA backends differ in their internal implementation:

- On AMD, hipfile uses a backend scoring system with a fastpath (``hipAmdFileRead`` and ``hipAmdFileWrite`` in the HIP runtime) and a POSIX fallback. On NVIDIA, all I/O is delegated directly to cuFile without backend scoring.
- The AMD backend validates filesystem type (ext4 with ordered journaling or xfs) and collects per-GPU, per-backend I/O statistics. These features are specific to the AMD implementation and are not part of the NVIDIA wrapper.
- Environment variables such as ``HIPFILE_ALLOW_COMPAT_MODE``, ``HIPFILE_FORCE_COMPAT_MODE``, and ``HIPFILE_UNSUPPORTED_FILE_SYSTEMS`` control AMD backend behavior and have no effect when built with the NVIDIA backend.

HIPIFY support
**************

The `amd-develop` branch of `ROCm/HIPIFY <https://github.com/ROCm/HIPIFY>`_ includes support for automatically translating cuFile API calls to hipfile API calls. A cuFile-to-hipfile API mapping is available in the `HIPIFY documentation <https://github.com/ROCm/HIPIFY/blob/amd-develop/docs/reference/tables/cuFile_API_supported_by_HIP.md>`_.

.. note::

   The hipfile changes in HIPIFY are not yet included in a public HIPIFY release. Use the ``amd-develop`` branch to access this functionality.
