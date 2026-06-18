.. meta::
   :description: Differences between hipFile and NVIDIA cuFile when porting or running on the NVIDIA stack.
   :keywords: hipFile, ROCm, NVIDIA, cuFile, compatibility, HIPIFY, I/O

******************************
hipFile cuFile compatibility
******************************

hipFile supports NVIDIA platforms by delegating I/O operations to the cuFile library. There are some differences between hipFile and cuFile that you should account for before you use hipFile on NVIDIA platforms.

Numeric error codes might differ between cuFile and hipFile, and they might also differ across hipFile releases. Avoid using numeric error codes in your code. See :doc:`/reference/api-errors` for more information about hipFile error codes.

The side effects of ``hipFileDriverOpen()``, ``hipFileDriverClose()``, and the use count reported by ``hipFileUseCount()`` might not match cuFile's behavior. See :doc:`/reference/api-driver` for more information about hipFile's driver lifecycle API.

hipFile uses a platform-independent ``hoff_t`` type in place of ``off_t``. On POSIX systems ``hoff_t`` is defined as ``off_t``. On Windows it is defined as ``__int64``.

There is no hipFile counterpart to cuFile's ``cuFileDriverClose_v2`` alias. Both ``cuFileDriverClose`` and ``cuFileDriverClose_v2`` map to ``hipFileDriverClose()``.

There is no hipFile counterpart to cuFile's ``sockaddr_t`` typedef. Use ``struct sockaddr`` or add a custom implementation of the typedef.

`HIPIFY <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html>`_ provides information about converting CUDA code to ROCm code. A cuFile-to-hipFile API map is available in the `HIPIFY reference tables <https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/cuFile_API_supported_by_HIP.md>`_.
