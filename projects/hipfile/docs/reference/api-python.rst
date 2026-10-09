.. meta::
   :description: Python API reference for hipFile, covering Driver, FileHandle, Buffer, HipFileException, enumerations, and module-level functions.
   :keywords: hipFile, Python, API, ROCm, Driver, FileHandle, Buffer, HipFileException, OpError, FileHandleType, GPU, I/O

*********************
Python API reference
*********************

The hipFile Python package (``hipfile``) provides Cython-backed bindings for
driver lifecycle management, file handle registration, GPU buffer
registration, and error handling. All classes listed below mirror the
corresponding C API surface in ``hipfile.h``.

For a guided walkthrough of common operations with these bindings, see
:doc:`/how-to/use-python-api`.

Module-level functions
======================

``get_version``
---------------

.. code-block:: python

   hipfile.get_version() -> tuple[int, int, int]

Return the hipFile library version as a ``(major, minor, patch)`` tuple.

Raises ``HipFileException`` if the version query fails.

``driver_get_properties``
-------------------------

.. code-block:: python

   hipfile.driver_get_properties() -> dict[str, int]

Return the current hipFile driver properties as a dictionary mapping
property names to their integer values.

Raises ``HipFileException`` if the properties query fails.

``Driver``
============

.. code-block:: python

   from hipfile import Driver

Manage the hipFile driver lifecycle. The driver is reference-counted
internally; multiple ``open`` calls are balanced by an equal number of
``close`` calls.

``Driver`` supports the context-manager protocol:

.. code-block:: python

   with Driver() as drv:
       ...  # driver is open
   # driver is closed

Methods
-------

``open()``
^^^^^^^^^^

Open the hipFile driver. Raises ``HipFileException`` on failure.

``close()``
^^^^^^^^^^^

Close the hipFile driver. Raises ``HipFileException`` on failure.

``use_count()`` (static)
^^^^^^^^^^^^^^^^^^^^^^^^

Return the current driver reference count as an ``int``.

``FileHandle``
================

.. code-block:: python

   from hipfile import FileHandle

Manage a file descriptor registered with the hipFile driver. Wraps
``os.open`` and ``os.close`` together with hipFile handle registration so
that GPU-accelerated reads and writes can be performed through a single
object.

``FileHandle`` supports the context-manager protocol:

.. code-block:: python

   with FileHandle(path, os.O_RDONLY | os.O_DIRECT) as fh:
       fh.read(buf, size, 0, 0)

Constructor
-----------

.. code-block:: python

   FileHandle(
       path: str | os.PathLike[str],
       flags: int,
       mode: int = 0o644,
       handle_type: FileHandleType = FileHandleType.OPAQUE_FD,
   )

The file is not opened until ``open()`` is called or the context manager
is entered.

.. note::

   Setting ``handle_type`` to ``FileHandleType.OPAQUE_WIN32`` raises
   ``NotImplementedError``.

Properties
----------

``path``
^^^^^^^^

Filesystem path (read-only).

``flags``
^^^^^^^^^

Flags passed to ``os.open`` (read-only).

``mode``
^^^^^^^^

File creation permission bits (read-only). Defaults to ``0o644``.

``handle_type``
^^^^^^^^^^^^^^^

``FileHandleType`` used for registration. Can be set before the file is
opened. Raises ``RuntimeError`` if the handle is already open,
``ValueError`` if the value is not a ``FileHandleType`` member, or
``NotImplementedError`` if set to ``OPAQUE_WIN32``.

``handle``
^^^^^^^^^^

Opaque hipFile handle as an ``int``, or ``None`` if the file is not open
(read-only).

Methods
-------

``open()``
^^^^^^^^^^

Open the file and register it with the hipFile driver. Raises
``RuntimeError`` if already open or ``HipFileException`` if registration
fails.

``close()``
^^^^^^^^^^^

Deregister the hipFile handle and close the file descriptor. Safe to call
multiple times; subsequent calls are no-ops.

``read(buffer, size, file_offset, buffer_offset)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Read ``size`` bytes from the file at ``file_offset`` into ``buffer`` at
``buffer_offset``. Returns the number of bytes read.

Raises ``RuntimeError`` if the handle is not open, ``OSError`` on a
system-level I/O error, or ``HipFileException`` on a hipFile or HIP driver
error.

``write(buffer, size, file_offset, buffer_offset)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Write ``size`` bytes from ``buffer`` at ``buffer_offset`` to the file at
``file_offset``. Returns the number of bytes written.

Raises ``RuntimeError`` if the handle is not open, ``OSError`` on a
system-level I/O error, or ``HipFileException`` on a hipFile or HIP driver
error.

``Buffer``
===========

.. code-block:: python

   from hipfile import Buffer

Manage registration of a GPU memory buffer with the hipFile driver.
``Buffer`` does not own the underlying GPU allocation; it only manages the
hipFile registration lifetime.

``Buffer`` supports the context-manager protocol:

.. code-block:: python

   with Buffer.from_ctypes_void_p(ptr, length, 0) as buf:
       fh.read(buf, length, 0, 0)

Class methods
-------------

``from_ctypes_void_p(ctypes_void_p, length, flags)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a ``Buffer`` from a ``ctypes.c_void_p``. Raises ``ValueError`` if
the pointer is null.

Constructor
-----------

.. code-block:: python

   Buffer(buffer_ptr: int, length: int, flags: int)

Initialize a ``Buffer`` from a raw integer pointer.

Properties
----------

``ptr``
^^^^^^^

Integer address of the GPU memory (read-only).

Methods
-------

``register()``
^^^^^^^^^^^^^^

Register the buffer with the hipFile driver. Raises ``HipFileException``
on failure.

``deregister()``
^^^^^^^^^^^^^^^^

Deregister the buffer from the hipFile driver. This is a no-op if the
buffer is not currently registered. Raises ``HipFileException`` on failure.

``HipFileException``
========================

.. code-block:: python

   from hipfile import HipFileException

Exception raised when a hipFile operation fails. Wraps both the
``hipFileOpError_t`` code and, when the error is ``HIP_DRIVER_ERROR``, the
underlying ``hipError_t`` from the HIP runtime.

Constructor
-----------

.. code-block:: python

   HipFileException(hipfile_err: int, hip_err: int)

Properties
----------

``hipfile_err``
^^^^^^^^^^^^^^^

``hipFileOpError_t`` code for this error.

``hip_err``
^^^^^^^^^^^

``hipError_t`` code from the HIP runtime. Only meaningful when
``hipfile_err`` equals ``OpError.HIP_DRIVER_ERROR``.

``__str__``
-----------

Returns a human-readable description of the error, including the
descriptive string from ``hipFileGetOpErrorString``. When the error is
``OpError.HIP_DRIVER_ERROR``, the HIP error code is appended.

Enumerations
================

``OpError``
-----------

.. code-block:: python

   from hipfile import OpError

``IntEnum`` mirroring ``hipFileOpError_t``. Values are sourced from the C
enum through the Cython layer and are not redefined in Python. Members
include:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Member
     - C equivalent
   * - ``SUCCESS``
     - ``hipFileSuccess``
   * - ``DRIVER_NOT_INITIALIZED``
     - ``hipFileDriverNotInitialized``
   * - ``DRIVER_INVALID_PROPS``
     - ``hipFileDriverInvalidProps``
   * - ``DRIVER_UNSUPPORTED_LIMIT``
     - ``hipFileDriverUnsupportedLimit``
   * - ``DRIVER_VERSION_MISMATCH``
     - ``hipFileDriverVersionMismatch``
   * - ``DRIVER_VERSION_READ_ERROR``
     - ``hipFileDriverVersionReadError``
   * - ``DRIVER_CLOSING``
     - ``hipFileDriverClosing``
   * - ``PLATFORM_NOT_SUPPORTED``
     - ``hipFilePlatformNotSupported``
   * - ``IO_NOT_SUPPORTED``
     - ``hipFileIONotSupported``
   * - ``DEVICE_NOT_SUPPORTED``
     - ``hipFileDeviceNotSupported``
   * - ``DRIVER_ERROR``
     - ``hipFileDriverError``
   * - ``HIP_DRIVER_ERROR``
     - ``hipFileHipDriverError``
   * - ``HIP_POINTER_INVALID``
     - ``hipFileHipPointerInvalid``
   * - ``HIP_MEMORY_TYPE_INVALID``
     - ``hipFileHipMemoryTypeInvalid``
   * - ``HIP_POINTER_RANGE_ERROR``
     - ``hipFileHipPointerRangeError``
   * - ``HIP_CONTEXT_MISMATCH``
     - ``hipFileHipContextMismatch``
   * - ``INVALID_MAPPING_SIZE``
     - ``hipFileInvalidMappingSize``
   * - ``INVALID_MAPPING_RANGE``
     - ``hipFileInvalidMappingRange``
   * - ``INVALID_FILE_TYPE``
     - ``hipFileInvalidFileType``
   * - ``INVALID_FILE_OPEN_FLAG``
     - ``hipFileInvalidFileOpenFlag``
   * - ``DIO_NOT_SET``
     - ``hipFileDIONotSet``
   * - ``INVALID_VALUE``
     - ``hipFileInvalidValue``
   * - ``MEMORY_ALREADY_REGISTERED``
     - ``hipFileMemoryAlreadyRegistered``
   * - ``MEMORY_NOT_REGISTERED``
     - ``hipFileMemoryNotRegistered``
   * - ``PERMISSION_DENIED``
     - ``hipFilePermissionDenied``
   * - ``DRIVER_ALREADY_OPEN``
     - ``hipFileDriverAlreadyOpen``
   * - ``HANDLE_NOT_REGISTERED``
     - ``hipFileHandleNotRegistered``
   * - ``HANDLE_ALREADY_REGISTERED``
     - ``hipFileHandleAlreadyRegistered``
   * - ``DEVICE_NOT_FOUND``
     - ``hipFileDeviceNotFound``
   * - ``INTERNAL_ERROR``
     - ``hipFileInternalError``
   * - ``GET_NEW_FD_FAILED``
     - ``hipFileGetNewFDFailed``
   * - ``DRIVER_SETUP_ERROR``
     - ``hipFileDriverSetupError``
   * - ``IO_DISABLED``
     - ``hipFileIODisabled``
   * - ``BATCH_SUBMIT_FAILED``
     - ``hipFileBatchSubmitFailed``
   * - ``GPU_MEMORY_PINNING_FAILED``
     - ``hipFileGPUMemoryPinningFailed``
   * - ``BATCH_FULL``
     - ``hipFileBatchFull``
   * - ``ASYNC_NOT_SUPPORTED``
     - ``hipFileAsyncNotSupported``
   * - ``IO_MAX_ERROR``
     - ``hipFileIOMaxError``

``FileHandleType``
------------------

.. code-block:: python

   from hipfile import FileHandleType

``IntEnum`` mirroring ``hipFileFileHandleType_t``. Members:

.. list-table::
   :header-rows: 1
   :widths: 30 40 30

   * - Member
     - C equivalent
     - Notes
   * - ``OPAQUE_FD``
     - ``hipFileHandleTypeOpaqueFD``
     - POSIX file descriptor
   * - ``OPAQUE_WIN32``
     - ``hipFileHandleTypeOpaqueWin32``
     - Raises ``NotImplementedError``
   * - ``USERSPACE_FS``
     - ``hipFileHandleTypeUserspaceFS``
     - Userspace RDMA filesystem
