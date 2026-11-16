.. meta::
   :description: Frameworks and applications that use hipFile for GPU-direct I/O
   :keywords: hipFile, integrations, LMCache, fastsafetensors, rocAL, PyTorch, fio, IOR, GDS, ROCm, AMD

*******************************************
Frameworks and applications using hipFile
*******************************************

Several frameworks and applications already call hipFile to move data between storage and GPU memory. Most of them reach hipFile through a direct storage-to-GPU-memory abstraction they already had, and select hipFile at build time or at run time when running on ROCm. This page lists the integrations AMD is aware of and points at each project's own documentation for configuration details.

LMCache
=======

`LMCache <https://docs.lmcache.ai/index.html>`_ is a KV cache layer for LLM serving. It offloads KV cache chunks to CPU memory, local disk, or remote storage. Its GDS storage backend reads and writes those chunks directly between GPU memory and storage.

Set the ``gds_backend`` configuration key to ``hipfile`` or set the ``LMCACHE_GDS_BACKEND`` environment variable to ``hipfile``.  See the `LMCache GDS backend documentation <https://docs.lmcache.ai/kv_cache/storage_backends/gds.html>`_ for more information.

fastsafetensors
===============

`fastsafetensors <https://pypi.org/project/fastsafetensors/>`_ loads tensor data from safetensors model files into GPU memory. It uses hipFile for direct storage-to-GPU I/O on ROCm when ``libhipfile.so`` is available.

If ``libhipfile.so`` isn't on the dynamic linker search path, fastsafetensors uses the bounce-buffer path.  Ensure that ``ROCM_PATH`` has been set before installing the package.

rocAL
=====

`rocAL <https://rocm.docs.amd.com/projects/rocAL/en/latest/>`_ is AMD's augmentation library, the data loading and augmentation pipeline used for training. Its numpy reader can read ``.npy`` files directly into GPU memory using hipFile instead of reading them through the host.

rocAL's build looks for the hipFile CMake package and, when it finds one, compiles the reader with ``ENABLE_HIPFILE=1`` and links against ``hip::hipfile``. You need to set ``ROCAL_USE_HIPFILE=1`` to enable this feature.

PyTorch
=======

`PyTorch <https://github.com/pytorch/pytorch>`_ is a machine learning framework. On ROCm, its ``torch.cuda.gds`` APIs use hipFile. GdsFile reads and writes GPU storage with ``hipFileRead()`` and ``hipFileWrite()``. The buffer and file registration helpers use the matching hipFile APIs.

Build options in PyTorch's HIP port keep their original names. ``torch.cuda.gds.is_available()`` reports whether it was built with GDS support in a ROCm build. See :doc:`cuFile compatibility <hipFile-cuFile-compatibility>`  and `PyTorch's ROCm notes <https://docs.pytorch.org/docs/main/notes/hip.html>`_ for more information.

fio
===

`fio <https://github.com/axboe/fio>`_ is a flexible I/O tester commonly used for storage benchmarking. fio versions 3.43 and later include a ``libhipfile`` engine that lets you drive hipFile I/O workloads from fio job files.

For build instructions, see :doc:`/how-to/use-with-fio`.

IOR
===

`IOR <https://ior.readthedocs.io/en/latest/>`_  is a parallel file system I/O benchmark used in HPC, often with mdtest from the same repository. Its hipFile backend, `src/gpu_io_hipfile.c`, reads and writes directly between GPU memory and storage on ROCm.

Build IOR from the `main branch <https://github.com/hpc/ior>`_. Run ``./bootstrap`` first, then run ``./configure`` with the  ``--with-gpu-direct-backend=hipfile`` option before building IOR. If ``./configure`` doesn't find hipFile or the HIP runtime, add the ``--with-hipfile=hipfile_prefix`` and ``--with-hip=hip_prefix`` options to the ``./configure`` command.
