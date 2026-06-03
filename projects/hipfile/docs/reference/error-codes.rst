.. meta::
   :description: Complete reference for hipfile error codes, the hipFileError_t dual-error struct, and error-handling macros.
   :keywords: hipfile, error codes, hipFileOpError_t, hipFileError_t, ROCm, GPU IO, error handling

=====================
Error codes reference
=====================

This page documents the error codes, error types, and error-handling macros provided by hipfile. For the complete API reference, see :doc:`/reference/api-reference`.

``hipFileOpError_t`` values
***************************

hipfile defines a base error value and an enumeration of operation error codes. All hipfile-specific error codes are offset from ``HIPFILE_BASE_ERR``, which is defined as ``5000``.

An error code of ``-1`` indicates that a C or POSIX error has occurred and ``errno`` is likely to have been set.

.. note::

   Values ``5021`` (``HIPFILE_BASE_ERR + 21``) and ``5032`` (``HIPFILE_BASE_ERR + 32``) are intentionally omitted from the enumeration.

.. list-table:: ``hipFileOpError_t`` error codes
   :header-rows: 1
   :widths: 10 40 50

   * - Value
     - Symbolic name
     - Description
   * - 0
     - ``hipFileSuccess``
     - hipfile operation completed successfully.
   * - 5001
     - ``hipFileDriverNotInitialized``
     - GPU IO driver is not loaded.
   * - 5002
     - ``hipFileDriverInvalidProps``
     - Invalid GPU IO driver property provided.
   * - 5003
     - ``hipFileDriverUnsupportedLimit``
     - GPU IO driver property value is unsupported.
   * - 5004
     - ``hipFileDriverVersionMismatch``
     - hipfile version does not match GPU IO driver version.
   * - 5005
     - ``hipFileDriverVersionReadError``
     - Unable to read the GPU IO driver version.
   * - 5006
     - ``hipFileDriverClosing``
     - GPU IO driver is closing and not accepting new requests.
   * - 5007
     - ``hipFilePlatformNotSupported``
     - hipfile is not supported on the current platform.
   * - 5008
     - ``hipFileIONotSupported``
     - hipfile is not supported on the selected file.
   * - 5009
     - ``hipFileDeviceNotSupported``
     - The selected GPU does not support hipfile.
   * - 5010
     - ``hipFileDriverError``
     - GPU IO driver error.
   * - 5011
     - ``hipFileHipDriverError``
     - GPU driver error. Inspect the ``hipError_t`` value for additional information.
   * - 5012
     - ``hipFileHipPointerInvalid``
     - Invalid GPU pointer.
   * - 5013
     - ``hipFileHipMemoryTypeInvalid``
     - Memory type backing pointer is incompatible with hipfile.
   * - 5014
     - ``hipFileHipPointerRangeError``
     - Pointer range exceeds allocated memory region.
   * - 5015
     - ``hipFileHipContextMismatch``
     - GPU driver context mismatch.
   * - 5016
     - ``hipFileInvalidMappingSize``
     - Accessing memory beyond pinned memory buffer.
   * - 5017
     - ``hipFileInvalidMappingRange``
     - Accessing memory beyond mapped memory region.
   * - 5018
     - ``hipFileInvalidFileType``
     - Unsupported file type.
   * - 5019
     - ``hipFileInvalidFileOpenFlag``
     - Unsupported file open flags.
   * - 5020
     - ``hipFileDIONotSet``
     - ``O_DIRECT`` flag not set.
   * - 5021
     - *(intentionally unused)*
     -
   * - 5022
     - ``hipFileInvalidValue``
     - One or more arguments have an invalid value.
   * - 5023
     - ``hipFileMemoryAlreadyRegistered``
     - Device pointer is already registered.
   * - 5024
     - ``hipFileMemoryNotRegistered``
     - Device pointer is not registered.
   * - 5025
     - ``hipFilePermissionDenied``
     - Permission error on device or file access.
   * - 5026
     - ``hipFileDriverAlreadyOpen``
     - GPU IO driver is already open.
   * - 5027
     - ``hipFileHandleNotRegistered``
     - File handle for GPU IO is not registered.
   * - 5028
     - ``hipFileHandleAlreadyRegistered``
     - File handle for GPU IO is already registered.
   * - 5029
     - ``hipFileDeviceNotFound``
     - Selected device not found.
   * - 5030
     - ``hipFileInternalError``
     - Internal GPU IO library error.
   * - 5031
     - ``hipFileGetNewFDFailed``
     - Unable to obtain a new file descriptor.
   * - 5032
     - *(intentionally unused)*
     -
   * - 5033
     - ``hipFileDriverSetupError``
     - GPU IO driver initialization error.
   * - 5034
     - ``hipFileIODisabled``
     - GPU IO config file prohibits GPU IO on specified file.
   * - 5035
     - ``hipFileBatchSubmitFailed``
     - Failed to submit request for batch operation.
   * - 5036
     - ``hipFileGPUMemoryPinningFailed``
     - Failed to allocate pinned device memory.
   * - 5037
     - ``hipFileBatchFull``
     - Batch operation queue is full.
   * - 5038
     - ``hipFileAsyncNotSupported``
     - hipfile async IO is not supported.
   * - 5039
     - ``hipFileIOMaxError``
     - Internal flag that marks the largest hipfile error code.

``hipFileGetOpErrorString``
---------------------------

Returns a human-readable description for a given ``hipFileOpError_t`` value.

.. code-block:: c

   const char *hipFileGetOpErrorString(hipFileOpError_t status)

:param status: The hipfile error code to describe.
:returns: A string describing the error.

``hipFileError_t`` dual-error struct
************************************

Most hipfile API functions return ``hipFileError_t``, a struct that carries two independent error values:

.. code-block:: c

   typedef struct hipFileError {
       hipFileOpError_t err;         // Errors related to hipfile or the GPU IO driver
       hipError_t       hip_drv_err; // Errors related to the GPU driver (HIP runtime)
   } hipFileError_t;

``err``
   Contains a ``hipFileOpError_t`` value indicating whether the hipfile operation itself succeeded or which hipfile-level error occurred.

``hip_drv_err``
   Contains a ``hipError_t`` value from the HIP runtime. This field is meaningful when ``err`` equals ``hipFileHipDriverError`` (5011), signaling that the root cause is a GPU driver error rather than a hipfile-level error.

.. note::

   In C++17 and C23 (and later), ``hipFileError_t`` has the ``[[nodiscard]]`` attribute. Ignoring return values from hipfile API functions generates a compiler warning under these language standards.

Error-handling macros
*********************

hipfile provides four preprocessor macros for classifying and describing errors.

``IS_HIPFILE_ERR``
------------------

Determines whether an error code is a hipfile-specific error (that is, its absolute value exceeds ``HIPFILE_BASE_ERR``).

.. code-block:: c

   bool IS_HIPFILE_ERR(hipFileOpError_t hip_op_err)

:param hip_op_err: A ``hipFileOpError_t`` value.
:returns: ``true`` if the absolute value of ``hip_op_err`` is greater than ``HIPFILE_BASE_ERR``, ``false`` otherwise.

**Example:**

.. code-block:: c

   ssize_t ret = hipFileRead(fh, buf, size, file_off, buf_off);
   if (ret < 0 && IS_HIPFILE_ERR(ret)) {
       printf("hipfile error: %s\n", HIPFILE_ERRSTR(ret));
   }

``HIPFILE_ERRSTR``
------------------

Returns a descriptive string for a hipfile error code. This macro calls ``hipFileGetOpErrorString`` with the absolute value of the provided error code.

.. code-block:: c

   const char *HIPFILE_ERRSTR(hipFileOpError_t hip_op_err)

:param hip_op_err: A ``hipFileOpError_t`` value.
:returns: A string description of the hipfile error.

``IS_HIP_DRV_ERR``
------------------

Determines whether a ``hipFileError_t`` indicates that the error originates from the HIP GPU driver rather than from hipfile itself. This macro checks whether the ``err`` field equals ``hipFileHipDriverError``.

.. code-block:: c

   bool IS_HIP_DRV_ERR(hipFileError_t err)

:param err: A ``hipFileError_t`` struct.
:returns: ``true`` if ``err.err`` equals ``hipFileHipDriverError``, ``false`` otherwise.

``HIP_DRV_ERR``
---------------

Extracts the ``hipError_t`` component from a ``hipFileError_t`` struct. Use this macro after ``IS_HIP_DRV_ERR`` confirms that a GPU driver error is present.

.. code-block:: c

   hipError_t HIP_DRV_ERR(hipFileError_t err)

:param err: A ``hipFileError_t`` struct.
:returns: The ``hip_drv_err`` field of the struct.

**Example:**

.. code-block:: c

   hipFileError_t status = hipFileDriverOpen();
   if (status.err != hipFileSuccess) {
       if (IS_HIP_DRV_ERR(status)) {
           hipError_t hip_err = HIP_DRV_ERR(status);
           printf("HIP driver error: %d\n", hip_err);
       } else {
           printf("hipfile error: %s\n", HIPFILE_ERRSTR(status.err));
       }
   }

Interpreting synchronous IO return values
*****************************************

The synchronous functions ``hipFileRead`` and ``hipFileWrite`` return an ``ssize_t`` value rather than a ``hipFileError_t`` struct. Interpret these return values as follows:

- **>= 0**: The number of bytes successfully transferred.
- **-1**: A system-level error occurred. Check ``errno`` for details.
- **Any other negative value**: The negative of a ``hipFileOpError_t`` value. Use ``IS_HIPFILE_ERR`` and ``HIPFILE_ERRSTR`` to classify and describe the error.
