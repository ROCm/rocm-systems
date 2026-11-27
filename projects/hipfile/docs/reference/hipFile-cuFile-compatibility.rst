.. meta::
   :description: cuFile compatibility
   :keywords: hipFile, ROCm, NVIDIA, cuFile, compatibility, hipify

*****************************
hipFile cuFile compatibility
*****************************

hipFile supports NVIDIA platforms by delegating I/O operations to the cuFile library. There are some differences between hipFile and cuFile that should be taken into account prior to using hipFile on NVIDIA platforms.

Numeric error codes might differ between cuFile and hipFile, and they might also differ across hipFile releases. Avoid using numeric error codes in your code. See :doc:`/reference/api-errors` for more information about hipFile error codes.

The side effects of ``hipFileDriverOpen()``, ``hipFileDriverClose()``, and the use count reported by ``hipFileUseCount()`` might not match cuFile's behavior. See :doc:`/reference/api-driver` for more information about hipFile's driver lifecycle API.

hipFile uses a platform-independent ``hoff_t`` type in place of ``off_t``. On POSIX systems ``hoff_t`` is defined as ``off_t``. On Windows it is defined as ``__int64``.

There is no hipFile counterpart to cuFile's ``cuFileDriverClose_v2`` alias. Both ``cuFileDriverClose`` and ``cuFileDriverClose_v2`` map to ``hipFileDriverClose()``.

There is no hipFile counterpart to cuFile's ``sockaddr_t`` typedef. Use ``struct sockaddr`` or add a custom implementation of the typedef.

Stream registration with ``hipFileStreamRegister()`` is optional in both hipFile and cuFile, and neither keeps persistent state for an unregistered stream, so only streams you explicitly registered need to be deregistered before they are destroyed. hipFile runs asynchronous I/O on an unregistered stream on a slower blocking path, while cuFile allocates per-transfer resources. By default hipFile uses that blocking path for registered streams too; see :ref:`hipfile-force-blocking-async`.

`HIPIFY <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html>`_ provides information about converting CUDA code to ROCm code. A `cuFile-to-hipFile API map <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/cuFile_API_supported_by_HIP.html>`_ is available.
