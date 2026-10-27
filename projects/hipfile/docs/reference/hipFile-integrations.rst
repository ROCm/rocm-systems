.. meta::
   :description: Frameworks and applications that use hipFile for GPU-direct I/O
   :keywords: hipFile, integrations, LMCache, fastsafetensors, rocAL, PyTorch, fio, IOR, GDS, ROCm, AMD

*******************************************
Frameworks and applications using hipFile
*******************************************

Several frameworks and applications already call hipFile to move data between storage and GPU memory. Most of them reach hipFile through a direct storage-to-GPU-memory abstraction they already had, and select hipFile at build time or at run time when running on ROCm. This page lists the integrations AMD is aware of and points at each project's own documentation for configuration details.

LMCache
=======

`LMCache <https://github.com/LMCache/LMCache>`_ is a KV cache layer for LLM serving that offloads KV cache chunks to CPU memory, local disk, or remote storage. Its GDS storage backend reads and writes those chunks directly between GPU memory and NVMe, and loads ``libhipfile.so`` on ROCm.

The backend is selected with the ``gds_backend`` configuration key (``LMCACHE_GDS_BACKEND`` as an environment variable), which accepts ``hipfile`` as well as ``auto``. The remaining GDS options, such as ``gds_path`` and ``gds_buffer_size``, apply to both backends. See the `LMCache GDS backend documentation <https://docs.lmcache.ai/kv_cache/storage_backends/gds.html>`_.

fastsafetensors
===============

`fastsafetensors <https://github.com/foundation-model-stack/fastsafetensors>`_ is a high-performance loader for safetensors model files. It reads tensor data from the file straight into GPU memory, which avoids the host bounce buffer and the page cache copy that the default safetensors deserializer performs.

On ROCm, its GDS code path uses hipFile transparently when ``libhipfile.so`` is available; otherwise it falls back to its own bounce-buffer path, which is also what passing ``nogds=True`` to ``SafeTensorsFileLoader`` or ``fastsafe_open()`` forces. Build it against ROCm by setting ``ROCM_PATH`` when installing the package.

rocAL
=====

`rocAL <https://github.com/ROCm/rocAL>`_ is AMD's augmentation library, the data loading and augmentation pipeline used for training. Its numpy reader can read ``.npy`` files directly into GPU memory using hipFile instead of reading them through the host.

rocAL's build looks for the hipFile CMake package and, when it finds one, compiles the reader with ``ENABLE_HIPFILE=1`` and links against ``hip::hipfile``. The feature is off by default at run time; set ``ROCAL_USE_HIPFILE=1`` to enable it.

PyTorch
=======

The ``torch.cuda.gds`` APIs, including ``GdsFile`` and the buffer registration helpers, are implemented with hipFile on ROCm. Each wrapper calls the matching hipFile function: ``hipFileRead()``, ``hipFileWrite()``, ``hipFileBufRegister()``, ``hipFileBufDeregister()``, ``hipFileHandleRegister()``, and ``hipFileHandleDeregister()``.

As elsewhere in PyTorch's HIP port, the build option keeps its original name, so ``USE_CUFILE=0`` is what disables GDS support in a ROCm build, and ``torch.cuda.gds.is_available()`` reports whether it was built in. If you are porting code that depends on specific error codes, review :doc:`cuFile compatibility <hipFile-cuFile-compatibility>` first. PyTorch's own ROCm notes cover the rest.

fio
===

`fio <https://github.com/axboe/fio>`_ is a flexible I/O tester widely used for storage benchmarking. AMD maintains a fork at `ROCm/fio <https://github.com/ROCm/fio>`_ that adds a ``libhipfile`` I/O engine, so fio job files can drive hipFile reads and writes against GPU memory. This is the usual way to measure hipFile throughput on a given system.

For build and run instructions, see :doc:`/how-to/use-with-fio`.

IOR
===

`IOR <https://github.com/hpc/ior>`_ is the parallel file system I/O benchmark used throughout HPC, usually alongside its companion metadata benchmark mdtest. IOR selects a GPU I/O backend at configure time, and now has a hipFile one (``src/gpu_io_hipfile.c``), so it can read and write directly between GPU memory and storage on ROCm.

Configure IOR with ``--with-gpu-direct-backend=hipfile`` to build the backend in; the backend defaults to ``none``, so this flag is required. Add ``--with-hipfile=<prefix>`` and ``--with-hip=<prefix>`` if hipFile and the HIP runtime aren't found automatically. It is not in an IOR release yet, so build from the project's ``main`` branch: the initial merge was followed by correctness fixes and by a fix for a build regression, both in October 2026.
