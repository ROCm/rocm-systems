.. meta::
  :description: rocSHMEM intra-kernel networking runtime for AMD dGPUs on the ROCm platform.
  :keywords: rocSHMEM, API, ROCm, documentation, HIP, Networking, Communication

.. _rocshmem-api-library-constants:

------------------
Library constants
------------------

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
