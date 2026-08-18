.. meta::
  :description: ROCdbgapi provides support for a debugger and other tools to perform low-level control of running code and inspection of GPU architectures
  :keywords: ROCdbgapi limitations, ROCdbgapi compatibility, ROCdbgapi support, AMD Debugger API limitations, AMD Debugger API compatibility, AMD Debugger API support

.. _support-limit:

Support and limitations
-------------------------

Refer to the following sections in the :doc:`ROCdbgapi API Specification <../doxygen/html/index>` documentation for:

- `Supported AMD GPU architectures <../doxygen/html/index.html#supported_amd_gpu_architectures>`_
- `Known limitations and restrictions <../doxygen/html/index.html#known_limitations>`_

The ROCdbgapi library is compatible with the following interface versions:

- **AMD GPU driver version**

  - The driver requirement is ``KFD_IOCTL_MAJOR_VERSION=1`` and ``KFD_IOCTL_MINOR_VERSION>=13`` in ``src/linux/kfd_ioctl.h``, which conforms to `semver <http://semver.org/>`_.

- **ROCm runtime r_debug ABI version**

  - See ``ROCR_RDEBUG_VERSION`` in ``src/rocr_rdebug.h``.
