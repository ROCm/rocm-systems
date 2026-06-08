.. meta::
  :description: Complete API reference for all public C functions and types in hipFile, organized by functional area.
  :keywords: hipFile, API, reference, ROCm, GPU I/O, direct storage, C API, functions, types

=============
API reference
=============

This page documents all public C API functions and types provided by hipFile, organized by functional area. For error code details, see :doc:`/reference/error-codes`. For driver lifecycle concepts, see :doc:`/conceptual/driver-lifecycle`.

.. note::

   Functions returning ``hipFileError_t`` have the ``[[nodiscard]]`` attribute in C++17 and C23 or later. Ignoring return values from these functions generates compiler warnings in supported language standards.

Core and versioning
*******************

Platform-independent types and compile-time version macros for hipFile.

Types
-----

``hoff_t``
   Platform-independent offset type. Defined as ``off_t`` on Linux and ``__int64`` on Windows.

``ssize_t``
   Signed size type. On Windows, defined as ``SSIZE_T`` from ``BaseTsd.h``. On Linux, provided by ``<sys/types.h>``.

Macros
------

``HIPFILE_VERSION_MAJOR``
   hipFile major version number. Currently ``0``.

``HIPFILE_VERSION_MINOR``
   hipFile minor version number. Currently ``2``.

``HIPFILE_VERSION_PATCH``
   hipFile patch version number. Currently ``0``.

Functions
---------

``hipFileGetVersion``
   Return the runtime version of the hipFile shared library.

   .. code-block:: c

      hipFileError_t hipFileGetVersion(unsigned *major, unsigned *minor, unsigned *patch)

   :param major: [out] Pointer that receives the major version number.
   :param minor: [out] Pointer that receives the minor version number.
   :param patch: [out] Pointer that receives the patch version number.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

.. _error-handling:

Error handling
**************

Error codes returned by hipFile API calls, a combined error struct carrying both a ``hipFileOpError_t`` and a ``hipError_t``, and helper macros for classifying and describing errors.

For the full list of error codes and their meanings, see :doc:`/reference/error-codes`.

Types
-----

``hipFileOpError_t``
   Enumeration of hipFile function return codes. An error code of ``-1`` indicates that a C or POSIX error has occurred and ``errno`` is likely to have been set. All hipFile-specific error codes start at ``HIPFILE_BASE_ERR`` (5000). Common values include:

   - ``hipFileSuccess`` (0): Operation completed successfully
   - ``hipFileDriverNotInitialized`` (5001): GPU I/O driver is not loaded
   - ``hipFileInvalidValue`` (5022): One or more arguments have an invalid value
   - ``hipFileInternalError`` (5030): Internal GPU I/O library error

   See :doc:`/reference/error-codes` for the complete enumeration.

``hipFileError_t``
   Error status returned from hipFile API calls. Contains two fields:

   - ``err`` (``hipFileOpError_t``): Errors related to hipFile or the GPU I/O driver
   - ``hip_drv_err`` (``hipError_t``): Errors related to the GPU driver

   .. code-block:: c

      typedef struct hipFileError {
          hipFileOpError_t err;
          hipError_t       hip_drv_err;
      } hipFileError_t;

   This struct has the ``[[nodiscard]]`` attribute in C++ >= 17 and C >= 23.

Functions
---------

``hipFileGetOpErrorString``
   Return a descriptive string for a hipFile error code.

   .. code-block:: c

      const char *hipFileGetOpErrorString(hipFileOpError_t status)

   :param status: Return code provided by hipFile.
   :returns: A human-readable description of the error encountered.

Macros
------

``IS_HIPFILE_ERR(hip_op_err)``
   Determine if an error code is a hipFile error (as opposed to a system or HIP error).

   .. code-block:: c

      bool IS_HIPFILE_ERR(hipFileOpError_t hip_op_err)

   :returns: ``true`` if the absolute value of the error code is greater than ``HIPFILE_BASE_ERR``, ``false`` otherwise.

``HIPFILE_ERRSTR(hip_op_err)``
   Get a descriptive error string for a hipFile error code. Calls ``hipFileGetOpErrorString`` with the absolute value of the error code.

   .. code-block:: c

      const char *HIPFILE_ERRSTR(hipFileOpError_t hip_op_err)

   :returns: A string description of the hipFile error.

``IS_HIP_DRV_ERR(hip_err)``
   Determine if an error is a HIP driver error.

   .. code-block:: c

      bool IS_HIP_DRV_ERR(hipFileError_t hip_err)

   :returns: ``true`` if ``hip_err.err`` equals ``hipFileHipDriverError``, ``false`` otherwise.

``HIP_DRV_ERR(hip_err)``
   Extract the ``hipError_t`` component from a ``hipFileError_t``.

   .. code-block:: c

      hipError_t HIP_DRV_ERR(hipFileError_t hip_err)

   :returns: The ``hip_drv_err`` field of the provided error struct.


GPU I/O driver API
*****************

Lifecycle and configuration of the GPU I/O driver for the current process. Controls reference counting, polling mode, maximum I/O sizes, cache sizes, and pinned memory limits. Also exposes properties describing supported file systems and transport features.

For details on driver lifecycle semantics, see :doc:`/conceptual/driver-lifecycle`.

.. note::

   On the AMD backend, ``hipFileDriverGetProperties``, all ``hipFileDriverSet*`` functions, ``hipFileBatchIOGetStatus``, ``hipFileBatchIOCancel``, and all ``hipFileGet/SetParameter*`` functions currently return ``hipFileInternalError`` (5030). ``hipFileBatchIODestroy`` is a no-op on AMD. ``hipFileBatchIOSetUp``, ``hipFileBatchIOSubmit``, ``hipFileReadAsync``, ``hipFileWriteAsync``, stream registration, and synchronous I/O are implemented. On the NVIDIA backend, most APIs delegate to cuFile. Alignment constraints called out for driver setters apply to NVIDIA builds.

Types
-----

``hipFileDriverStatusFlags_t``
   Enumeration of file systems and storage protocols supported by GPU I/O on the system. Used as bit positions in the ``driver_status_flags`` bitfield of ``hipFileDriverProps_t``.

   .. code-block:: c

      typedef enum hipFileDriverStatusFlags {
          hipFileLustreSupported       = 0,
          hipFileWekaFSSupported       = 1,
          hipFileNFSSupported          = 2,
          hipFileGPFSSupported         = 3,
          hipFileNVMeSupported         = 4,
          hipFileNVMeoFSupported       = 5,
          hipFileSCSISupported         = 6,
          hipFileScaleFluxCSDSupported = 7,
          hipFileNVMeshSupported       = 8,
          hipFileBeeGFSSupported       = 9,
          /* 10 is reserved */
          hipFileNVMeP2PSupported      = 11,
          hipFileScatefsSupported      = 12,
      } hipFileDriverStatusFlags_t;

``hipFileDriverControlFlags_t``
   Control flags for the GPU I/O driver.

   .. code-block:: c

      typedef enum hipFileDriverControlFlags {
          hipFileUsePollMode     = 0,
          hipFileAllowCompatMode = 1,
      } hipFileDriverControlFlags_t;

   - ``hipFileUsePollMode``: Enable polling for I/O completion
   - ``hipFileAllowCompatMode``: Allow GPU I/O to fall back to POSIX I/O

``hipFileFeatureFlags_t``
   GPU I/O transport and feature flags supported by the system. Used as bit positions in the ``feature_flags`` field of ``hipFileDriverProps_t``.

   .. code-block:: c

      typedef enum hipFileFeatureFlags {
          hipFileDynRoutingSupported  = 0,
          hipFileBatchIOSupported     = 1,
          hipFileStreamsSupported      = 2,
          hipFileParallelIOSupported  = 3,
      } hipFileFeatureFlags_t;

``hipFileDriverProps_t``
   GPU I/O configuration structure returned by ``hipFileDriverGetProperties``.

   .. code-block:: c

      typedef struct hipFileDriverProps {
          struct {
              unsigned major_version;
              unsigned minor_version;
              uint64_t poll_thresh_size;
              uint64_t max_direct_io_size;
              unsigned driver_status_flags;
              unsigned driver_control_flags;
          } nvfs;
          unsigned feature_flags;
          uint64_t max_device_cache_size;
          uint64_t per_buffer_cache_size;
          uint64_t max_device_pinned_mem_size;
          unsigned max_batch_io_count;
          unsigned max_batch_io_timeout_msecs;
      } hipFileDriverProps_t;

   Fields:

   - ``nvfs.major_version``: Major version of the GPU I/O driver
   - ``nvfs.minor_version``: Minor version of the GPU I/O driver
   - ``nvfs.poll_thresh_size``: Maximum I/O size (in KiB) for which polling is used when poll mode is enabled
   - ``nvfs.max_direct_io_size``: Maximum I/O size (in KiB) used by the GPU I/O driver
   - ``nvfs.driver_status_flags``: Bitfield mapping to ``hipFileDriverStatusFlags_t``
   - ``nvfs.driver_control_flags``: Bitfield mapping to ``hipFileDriverControlFlags_t``
   - ``feature_flags``: Bitfield mapping to ``hipFileFeatureFlags_t``
   - ``max_device_cache_size``: Maximum GPU memory (in KiB) for bounce buffers
   - ``per_buffer_cache_size``: GPU memory (in KiB) allocated for each bounce buffer
   - ``max_device_pinned_mem_size``: Maximum GPU memory (in KiB) that can be pinned
   - ``max_batch_io_count``: Maximum number of batch operations that can be submitted at once
   - ``max_batch_io_timeout_msecs``: Timeout (in milliseconds) for a batch operation to complete

Functions
---------

``hipFileDriverOpen``
   Initialize the GPU I/O driver for this process. Each call increments the library's reference count. If the reference count transitions from zero to one, the library's state is initialized.

   Calling ``hipFileDriverOpen`` is optional. The first call to ``hipFileBufRegister`` or ``hipFileHandleRegister`` triggers library initialization and increments the reference count.

   .. code-block:: c

      hipFileError_t hipFileDriverOpen(void)

   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileDriverClose``
   Close the GPU I/O driver for this process. Each call decrements the library's reference count. If the reference count transitions from one to zero, the library's state is destroyed.

   Explicitly closing the library is not required; the library's state is destroyed automatically at program exit.

   .. code-block:: c

      hipFileError_t hipFileDriverClose(void)

   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileUseCount``
   Obtain the current reference count for the library.

   .. code-block:: c

      int64_t hipFileUseCount(void)

   :returns: The library's current reference count.

``hipFileDriverGetProperties``
   Get a list of GPU I/O driver properties.

   .. code-block:: c

      hipFileError_t hipFileDriverGetProperties(hipFileDriverProps_t *props)

   :param props: Pointer to a ``hipFileDriverProps_t`` structure that receives the current driver properties.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileDriverSetPollMode``
   Enable or disable polling mode for GPU I/O.

   .. code-block:: c

      hipFileError_t hipFileDriverSetPollMode(bool poll, size_t poll_threshold_size)

   :param poll: ``true`` to enable polling, ``false`` to disable.
   :param poll_threshold_size: Maximum I/O size (in KiB) for which polling is used when enabled. On NVIDIA builds, this must be a multiple of 4K.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileDriverSetMaxDirectIOSize``
   Set the maximum I/O chunk size.

   .. code-block:: c

      hipFileError_t hipFileDriverSetMaxDirectIOSize(size_t max_direct_io_size)

   :param max_direct_io_size: Maximum I/O chunk size (in KiB) for each I/O request. Must be in 64K increments on NVIDIA builds.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileDriverSetMaxCacheSize``
   Set the maximum amount of GPU memory that can be used for bounce buffers.

   .. code-block:: c

      hipFileError_t hipFileDriverSetMaxCacheSize(size_t max_cache_size)

   :param max_cache_size: Maximum GPU memory (in KiB) that can be reserved for bounce buffers. Must be in 4K increments on NVIDIA builds.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileDriverSetMaxPinnedMemSize``
   Set the maximum amount of GPU memory that can be pinned.

   .. code-block:: c

      hipFileError_t hipFileDriverSetMaxPinnedMemSize(size_t max_pinned_size)

   :param max_pinned_size: Maximum GPU memory (in KiB) that can be pinned. Must be in 4K increments on NVIDIA builds.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.


Userspace RDMA API
******************

Configuration structure for userspace RDMA operations and flag macros for characterizing RDMA operations.

Types
-----

``hipFileRDMAInfo_t``
   Userspace RDMA configuration structure.

   .. code-block:: c

      typedef struct hipFileRDMAInfo {
          int         version;
          int         desc_len;
          const char *desc_str;
      } hipFileRDMAInfo_t;

   Fields:

   - ``version``: Version of the RDMA API to use
   - ``desc_len``: Length of the description string
   - ``desc_str``: Describes the configuration of the RDMA operation

Macros
------

``HIPFILE_RDMA_REGISTER``
   Flag indicating that the RDMA operation is registered. Value: ``1``.

``HIPFILE_RDMA_RELAXED_ORDERING``
   Flag indicating that the RDMA operation has relaxed ordering. Value: ``(1 << 1)``.


File handle API
***************

Register and deregister open files and GPU memory buffers for use with GPU I/O. Perform synchronous read and write operations directly between files and GPU buffers.

Types
-----

``hipFileFSOps_t``
   I/O operations for RDMA file systems. This structure provides function pointers for userspace file system operations.

   .. code-block:: c

      typedef struct hipFileFSOps {
          const char *(*fs_type)(void *handle);
          int (*getRDMADeviceList)(void *handle, struct sockaddr **hostaddrs);
          int (*getRDMADevicePriority)(void *handle, char *, size_t, hoff_t, struct sockaddr *hostaddr);
          ssize_t (*read)(void *handle, char *, size_t, hoff_t, hipFileRDMAInfo_t *);
          ssize_t (*write)(void *handle, const char *, size_t, hoff_t, hipFileRDMAInfo_t *);
      } hipFileFSOps_t;

   Function pointer fields:

   - ``fs_type``: Type of remote file system used. If ``NULL``, ``fstat`` is used for discovery.
   - ``getRDMADeviceList``: Get a list of host RDMA addresses. If ``NULL``, use any address.
   - ``getRDMADevicePriority``: Get the assigned priority of an RDMA device. Returns ``-1`` if no preference.
   - ``read``: Read from the remote file system. If ``NULL``, use the Linux VFS.
   - ``write``: Write to the remote file system. If ``NULL``, use the Linux VFS.

``hipFileFileHandleType_t``
   Type of file handle being used.

   .. code-block:: c

      typedef enum hipFileFileHandleType {
          hipFileHandleTypeOpaqueFD    = 1,
          hipFileHandleTypeOpaqueWin32 = 2,
          hipFileHandleTypeUserspaceFS = 3,
      } hipFileFileHandleType_t;

   - ``hipFileHandleTypeOpaqueFD``: POSIX file descriptor
   - ``hipFileHandleTypeOpaqueWin32``: Win32 HANDLE file handle
   - ``hipFileHandleTypeUserspaceFS``: Userspace RDMA file system

``hipFileDescr_t``
   Top-level structure for performing GPU I/O. The ``type`` field determines which union member to populate:

   - ``hipFileHandleTypeOpaqueFD``: Set ``handle.fd`` to a non-negative file descriptor; ``fs_ops`` is ignored
   - ``hipFileHandleTypeOpaqueWin32``: Set ``handle.hFile`` to a non-NULL handle; ``fs_ops`` is ignored
   - ``hipFileHandleTypeUserspaceFS``: Set ``handle.fd`` to a non-negative file descriptor and ``fs_ops`` to a non-NULL pointer

   .. code-block:: c

      typedef struct hipFileDescr {
          hipFileFileHandleType_t type;
          union {
              int   fd;
              void *hFile;
          } handle;
          const hipFileFSOps_t *fs_ops;
      } hipFileDescr_t;

``hipFileHandle_t``
   Opaque file handle used by the GPU I/O driver and library.

   .. code-block:: c

      typedef void *hipFileHandle_t;

Functions
---------

``hipFileHandleRegister``
   Register an open file for GPU I/O. If the library has not already been initialized, the first call to this function initializes the library and increments the reference count.

   .. code-block:: c

      hipFileError_t hipFileHandleRegister(hipFileHandle_t *fh, hipFileDescr_t *descr)

   :param fh: [out] Pointer that receives the opaque file handle on success.
   :param descr: [in] Parameters describing the file to register.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

   .. note::

      On the AMD backend, only ``hipFileHandleTypeOpaqueFD`` is supported. Other handle types return ``hipFileIONotSupported``.

``hipFileHandleDeregister``
   Deregister a file from GPU I/O.

   .. code-block:: c

      void hipFileHandleDeregister(hipFileHandle_t fh)

   :param fh: [in] The opaque file handle to deregister.

``hipFileBufRegister``
   Register a GPU memory region to be used with GPU I/O. The memory region should be allocated by the caller before being passed to this function. If the library has not already been initialized, the first call to this function initializes the library and increments the reference count.

   .. code-block:: c

      hipFileError_t hipFileBufRegister(const void *buffer_base, size_t length, int flags)

   :param buffer_base: [in] Base address of the GPU memory buffer.
   :param length: [in] Size of the allocated buffer in bytes.
   :param flags: [in] Control flags for this buffer.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileBufDeregister``
   Deregister a GPU memory region from being used with GPU I/O.

   .. code-block:: c

      hipFileError_t hipFileBufDeregister(const void *buffer_base)

   :param buffer_base: [in] Base address of the GPU memory buffer to deregister.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileRead``
   Synchronously read data from a file into a GPU buffer.

   .. code-block:: c

      ssize_t hipFileRead(hipFileHandle_t fh, void *buffer_base, size_t size,
                          hoff_t file_offset, hoff_t buffer_offset)

   :param fh: [in] The opaque file handle for the registered file.
   :param buffer_base: [in] Base address of the GPU memory buffer.
   :param size: [in] Number of bytes that should be read.
   :param file_offset: [in] Offset into the file to read from.
   :param buffer_offset: [in] Offset into the GPU buffer where data should be written.
   :returns: The return value uses a signed convention:

      - ``>= 0``: Number of bytes successfully read
      - ``-1``: System error (check ``errno`` for the specific error)
      - Other negative value: The negated value of the corresponding ``hipFileOpError_t`` error code. Use ``HIPFILE_ERRSTR(-retval)`` to get a human-readable description.

   .. warning::

      The maximum transfer size per call may be limited by the GPU I/O driver configuration. Large transfers may need to be split across multiple calls.

``hipFileWrite``
   Synchronously write data from a GPU buffer to a file.

   .. code-block:: c

      ssize_t hipFileWrite(hipFileHandle_t fh, const void *buffer_base, size_t size,
                           hoff_t file_offset, hoff_t buffer_offset)

   :param fh: [in] The opaque file handle for the registered file.
   :param buffer_base: [in] Base address of the GPU memory buffer.
   :param size: [in] Number of bytes that should be written.
   :param file_offset: [in] Offset into the file to write to.
   :param buffer_offset: [in] Offset into the GPU buffer where data should be read from.
   :returns: The return value uses a signed convention:

      - ``>= 0``: Number of bytes successfully written
      - ``-1``: System error (check ``errno`` for the specific error)
      - Other negative value: The negated value of the corresponding ``hipFileOpError_t`` error code. Use ``HIPFILE_ERRSTR(-retval)`` to get a human-readable description.

   .. warning::

      The maximum transfer size per call may be limited by the GPU I/O driver configuration. Large transfers may need to be split across multiple calls.


Async API
*********

Stream-based asynchronous GPU I/O operations. Streams must be registered before use. Async operations enqueue work on HIP streams.

Functions
---------

``hipFileStreamRegister``
   Register a HIP stream for use with asynchronous GPU I/O operations.

   .. code-block:: c

      hipFileError_t hipFileStreamRegister(hipStream_t stream, unsigned flags)

   :param stream: [in] A valid HIP stream.
   :param flags: [in] Stream usage flags.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileStreamDeregister``
   Deregister a HIP stream from asynchronous GPU I/O use.

   .. code-block:: c

      hipFileError_t hipFileStreamDeregister(hipStream_t stream)

   :param stream: [in] The HIP stream to deregister.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileReadAsync``
   Asynchronously read data from a file into a GPU buffer on a registered HIP stream.

   .. code-block:: c

      hipFileError_t hipFileReadAsync(hipFileHandle_t fh, void *buffer_base, size_t *size,
                                      hoff_t *file_offset, hoff_t *buffer_offset,
                                      ssize_t *bytes_read, hipStream_t stream)

   :param fh: [in] The opaque file handle for the registered file.
   :param buffer_base: [in] Base address of the GPU memory buffer.
   :param size: [in] Pointer to the number of bytes to read.
   :param file_offset: [in] Pointer to the offset into the file to read from.
   :param buffer_offset: [in] Pointer to the offset into the GPU buffer.
   :param bytes_read: [out] Pointer that receives the number of bytes actually read upon completion.
   :param stream: [in] A registered HIP stream to enqueue the operation on.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileWriteAsync``
   Asynchronously write data from a GPU buffer to a file on a registered HIP stream.

   .. code-block:: c

      hipFileError_t hipFileWriteAsync(hipFileHandle_t fh, const void *buffer_base, size_t *size,
                                       hoff_t *file_offset, hoff_t *buffer_offset,
                                       ssize_t *bytes_written, hipStream_t stream)

   :param fh: [in] The opaque file handle for the registered file.
   :param buffer_base: [in] Base address of the GPU memory buffer.
   :param size: [in] Pointer to the number of bytes to write.
   :param file_offset: [in] Pointer to the offset into the file to write to.
   :param buffer_offset: [in] Pointer to the offset into the GPU buffer.
   :param bytes_written: [out] Pointer that receives the number of bytes actually written upon completion.
   :param stream: [in] A registered HIP stream to enqueue the operation on.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.


Batch API
*********

Submit, monitor, and cancel batches of GPU I/O operations. A batch context has a fixed capacity of up to 128 operations.

Types
-----

``hipFileOpcode_t``
   The direction of data movement in a batch I/O request.

   .. code-block:: c

      typedef enum hipFileOpcode {
          hipFileBatchRead  = 0,
          hipFileBatchWrite = 1,
      } hipFileOpcode_t;

``hipFileStatus_t``
   The status of a batch I/O operation. Values are bit flags.

   .. code-block:: c

      typedef enum hipFileStatus {
          hipFileWaiting  = 1 << 0,
          hipFilePending  = 1 << 1,
          hipFileInvalid  = 1 << 2,
          hipFileCanceled = 1 << 3,
          hipFileComplete = 1 << 4,
          hipFileTimeout  = 1 << 5,
          hipFileFailed   = 1 << 6,
      } hipFileStatus_t;

   - ``hipFileWaiting``: Batch I/O operation pending acceptance
   - ``hipFilePending``: Batch I/O operation accepted and queued
   - ``hipFileInvalid``: Batch I/O operation was rejected for being invalid or could not be queued
   - ``hipFileCanceled``: Batch I/O operation was canceled
   - ``hipFileComplete``: Batch I/O operation completed
   - ``hipFileTimeout``: Batch I/O operation timed out
   - ``hipFileFailed``: Batch I/O operation failed

``hipFileBatchMode_t``
   Mode of a batch I/O operation.

   .. code-block:: c

      typedef enum hipFileBatchMode {
          hipFileBatch = 1,
      } hipFileBatchMode_t;

``hipFileIOParams_t``
   Input parameters for a batch I/O request. Contains a mode field and a union with batch-specific parameters.

   .. code-block:: c

      typedef struct hipFileIOParams {
          hipFileBatchMode_t mode;
          union {
              struct {
                  void   *devPtr_base;
                  int64_t file_offset;
                  int64_t devPtr_offset;
                  size_t  size;
              } batch;
          } u;
          hipFileHandle_t fh;
          hipFileOpcode_t opcode;
          void *cookie;
      } hipFileIOParams_t;

   Fields within ``u.batch``:

   - ``devPtr_base``: Base address of the GPU buffer for read or write
   - ``file_offset``: Offset into the file
   - ``devPtr_offset``: Offset of the GPU buffer
   - ``size``: Number of bytes to read or write

``hipFileIOEvents_t``
   Output events for batch I/O operations, containing status and transfer results.

``hipFileBatchHandle_t``
   Opaque handle for a batch I/O context.

Functions
---------

``hipFileBatchIOSetUp``
   Create a batch I/O context with a specified maximum number of concurrent operations.

   .. code-block:: c

      hipFileError_t hipFileBatchIOSetUp(hipFileBatchHandle_t *handle, unsigned max_nr)

   :param handle: [out] Pointer that receives the batch context handle.
   :param max_nr: [in] Maximum number of outstanding operations for this context. Must not exceed 128.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileBatchIOSubmit``
   Submit one or more operations to a batch context.

   .. code-block:: c

      hipFileError_t hipFileBatchIOSubmit(hipFileBatchHandle_t handle,
                                          unsigned nr, hipFileIOParams_t *params,
                                          unsigned flags)

   :param handle: [in] The batch context handle.
   :param nr: [in] Number of operations to submit.
   :param params: [in] Pointer to an array of I/O parameter structures.
   :param flags: [in] Submission flags.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileBatchIOGetStatus``
   Query the status of operations in a batch context. Not implemented on the AMD backend (returns ``hipFileInternalError``).

   .. code-block:: c

      hipFileError_t hipFileBatchIOGetStatus(hipFileBatchHandle_t handle,
                                              unsigned min_nr,
                                              unsigned *nr,
                                              hipFileIOEvents_t *events,
                                              struct timespec *timeout)

   :param handle: [in] The batch context handle.
   :param min_nr: [in] Minimum number of completed events to wait for.
   :param nr: [in, out] On input, maximum number of events to return. On output, actual number returned.
   :param events: [out] Pointer to an array of event structures to populate.
   :param timeout: [in] Maximum time to wait for events. May be ``NULL`` for non-blocking.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileBatchIOCancel``
   Cancel outstanding operations in a batch context. Not implemented on the AMD backend (returns ``hipFileInternalError``).

   .. code-block:: c

      hipFileError_t hipFileBatchIOCancel(hipFileBatchHandle_t handle)

   :param handle: [in] The batch context handle.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileBatchIODestroy``
   Destroy a batch context and release all associated resources. On the AMD backend, this function is currently a no-op.

   .. code-block:: c

      void hipFileBatchIODestroy(hipFileBatchHandle_t handle)

   :param handle: [in] The batch context handle to destroy.


Properties API
**************

Get and set typed runtime configuration parameters for the GPU I/O driver.

Types
-----

``hipFileSizeTConfigParameter_t``
   Enumeration of ``size_t``-typed configuration parameters.

``hipFileBoolConfigParameter_t``
   Enumeration of boolean-typed configuration parameters.

``hipFileStringConfigParameter_t``
   Enumeration of string-typed configuration parameters.

Functions
---------

``hipFileGetParameterSizeT``
   Get a ``size_t``-typed configuration parameter.

   .. code-block:: c

      hipFileError_t hipFileGetParameterSizeT(hipFileSizeTConfigParameter_t param,
                                               size_t *value)

   :param param: [in] The configuration parameter to query.
   :param value: [out] Pointer that receives the current value of the parameter.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileGetParameterBool``
   Get a boolean-typed configuration parameter.

   .. code-block:: c

      hipFileError_t hipFileGetParameterBool(hipFileBoolConfigParameter_t param,
                                              bool *value)

   :param param: [in] The configuration parameter to query.
   :param value: [out] Pointer that receives the current value of the parameter.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileGetParameterString``
   Get a string-typed configuration parameter.

   .. code-block:: c

      hipFileError_t hipFileGetParameterString(hipFileStringConfigParameter_t param,
                                                char *desc_str, int len)

   :param param: [in] The configuration parameter to query.
   :param desc_str: [out] Buffer to receive the string value.
   :param len: [in] Size of ``desc_str`` in bytes.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileSetParameterSizeT``
   Set a ``size_t``-typed configuration parameter.

   .. code-block:: c

      hipFileError_t hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param,
                                               size_t value)

   :param param: [in] The configuration parameter to set.
   :param value: [in] The new value for the parameter.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileSetParameterBool``
   Set a boolean-typed configuration parameter.

   .. code-block:: c

      hipFileError_t hipFileSetParameterBool(hipFileBoolConfigParameter_t param,
                                              bool value)

   :param param: [in] The configuration parameter to set.
   :param value: [in] The new value for the parameter.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.

``hipFileSetParameterString``
   Set a string-typed configuration parameter.

   .. code-block:: c

      hipFileError_t hipFileSetParameterString(hipFileStringConfigParameter_t param,
                                                const char *value)

   :param param: [in] The configuration parameter to set.
   :param value: [in] The new string value for the parameter.
   :returns: A ``hipFileError_t`` indicating success or the error encountered.


Statistics API (internal)
***********************

.. note::

   The functions in this section are defined in the internal header ``hipfile-stats.h``. They are not part of the public ``hipfile.h`` API. They are documented here because the ``ais-stats`` tool uses them. For tool usage, see :doc:`/reference/ais-stats-tool`.

Types
-----

``hipFileStatsContext_t``
   Opaque pointer to a statistics collection context.

``hipFileStatsError_t``
   Error codes returned by statistics collection API functions.

   .. code-block:: c

      typedef enum hipFileStatsError {
          hipFileStatsSuccess,
          hipFileStatsInvalidArgument,
          hipFileStatsTargetProcessNotFound,
          hipFileStatsTargetProcessNotAccessible,
          hipFileStatsReportGenerationFailed,
      } hipFileStatsError_t;

   - ``hipFileStatsSuccess``: Operation completed successfully
   - ``hipFileStatsInvalidArgument``: Invalid argument passed to function
   - ``hipFileStatsTargetProcessNotFound``: Target process with given PID not found
   - ``hipFileStatsTargetProcessNotAccessible``: Cannot access target process
   - ``hipFileStatsReportGenerationFailed``: Failed to generate or write report

Functions
---------

``hipFileStatsCreateContext``
   Create a new statistics collection context for a target process. The returned context must be freed with ``hipFileStatsCloseContext``.

   .. code-block:: c

      hipFileStatsError_t hipFileStatsCreateContext(hipFileStatsContext_t **context,
                                                     int targetPid)

   :param context: [out] Pointer to store the created context handle.
   :param targetPid: [in] Process ID of the target process to monitor.
   :returns: ``hipFileStatsSuccess`` on success, or an error code otherwise.

``hipFileStatsCloseContext``
   Close and free a statistics collection context. Safe to call with a ``NULL`` pointer.

   .. code-block:: c

      void hipFileStatsCloseContext(hipFileStatsContext_t *context)

   :param context: [in] Statistics context handle to close (may be ``NULL``).

``hipFileStatsConnectToTargetProcess``
   Establish a connection to the target process for statistics collection. Must be called before ``hipFileStatsGenerateReport``.

   .. code-block:: c

      hipFileStatsError_t hipFileStatsConnectToTargetProcess(hipFileStatsContext_t *context)

   :param context: [in] Statistics context handle.
   :returns: ``hipFileStatsSuccess`` on success, or an error code otherwise.

``hipFileStatsPollTargetProcess``
   Poll the target process for updated statistics. If ``block`` is ``true``, waits indefinitely for the target process to complete.

   .. code-block:: c

      hipFileStatsError_t hipFileStatsPollTargetProcess(const hipFileStatsContext_t *context,
                                                         bool block)

   :param context: [in] Statistics context handle.
   :param block: [in] Whether to block until the target process completes.
   :returns: ``hipFileStatsSuccess`` on success, or an error code otherwise.

``hipFileStatsGenerateReport``
   Generate a formatted statistics report from collected data and write it to a file descriptor.

   .. code-block:: c

      hipFileStatsError_t hipFileStatsGenerateReport(const hipFileStatsContext_t *context,
                                                      int fd)

   :param context: [in] Statistics context handle.
   :param fd: [in] File descriptor to write the report to.
   :returns: ``hipFileStatsSuccess`` on success, or an error code otherwise.


Python bindings
***************

hipFile provides Cython-based Python bindings exposing the C API. The bindings provide high-level context-manager classes, ``hipMalloc`` and ``hipFree`` helpers, error handling through ``HipFileException``, and typed enums. The GIL is released during all C calls to allow multi-threaded use.

For detailed Python API usage, see :doc:`/tutorials/python-gpu-io`.

Classes
-------

``Driver``
   Context manager wrapping ``hipFileDriverOpen`` and ``hipFileDriverClose``. Manages the driver lifecycle automatically.

``FileHandle``
   Context manager wrapping ``hipFileHandleRegister`` and ``hipFileHandleDeregister``. Represents a registered file for GPU I/O.

``Buffer``
   Context manager wrapping ``hipFileBufRegister`` and ``hipFileBufDeregister``. Represents a registered GPU memory buffer.

``HipFileException``
   Exception class raised when hipFile C API calls return errors.

Enums
-----

``FileHandleType``
   Python enum corresponding to ``hipFileFileHandleType_t``.

``OpError``
   Python enum corresponding to ``hipFileOpError_t``.

Functions
---------

``get_version()``
   Return the hipFile version as a tuple.

``driver_get_properties()``
   Return the GPU I/O driver properties.
