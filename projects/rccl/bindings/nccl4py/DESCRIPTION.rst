**************************************************
nccl4py: Pythonic RCCL Communication for GPU Clusters
**************************************************

`nccl4py <https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl/bindings/nccl4py>`_ provides a Pythonic interface to the
ROCm Communication Collectives Library (RCCL), AMD's drop-in replacement for
NVIDIA NCCL on ROCm. It bridges Python's simplicity with RCCL's GPU-accelerated
multi-GPU and multi-node communication primitives, so distributed Python
workloads can target AMD GPUs through the same API surface used on NVIDIA
hardware.

This package is a fork of the upstream NVIDIA ``nccl4py`` v0.2.0 with the
adaptations needed to run on top of ``librccl.so``: a HIP-backed ``cuda.core``
shim, RCCL-only collective wrappers (``ncclAllReduceWithBias``,
``ncclAllToAllv``), and a ROCm-native shared-library loader. The package name
``nccl4py`` is deliberately kept so that code written against the upstream
bindings runs unchanged on ROCm.

* `Homepage <https://github.com/ROCm/rocm-systems/tree/develop/projects/rccl>`_
* `Repository <https://github.com/ROCm/rocm-systems>`_
* `Documentation <https://rocm.docs.amd.com/projects/rccl/en/latest/>`_
* `Issue tracker <https://github.com/ROCm/rocm-systems/issues>`_

``nccl4py`` is under active development. Feedback and suggestions are welcome.


Installation
============

.. code-block:: bash

   pip install nccl4py

The package depends on ``hip-python`` and resolves ``librccl.so`` at runtime
via ``dlopen``; no CUDA or HIP toolchain is required at build time.
