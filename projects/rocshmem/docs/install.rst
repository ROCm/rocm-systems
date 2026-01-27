.. meta::
  :description: Instruction on how to install rocSHMEM.
  :keywords: rocSHMEM, ROCm, install, build, dependencies, MPI, UCX, Open MPI

.. _install-rocshmem:

---------------------------
Installing rocSHMEM
---------------------------

This topic describes how to install rocSHMEM.

Requirements
------------

* ROCm 6.4.0 or later, including the :doc:`HIP runtime <hip:index>`. For more information, see `ROCm installation for Linux <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/>`_.

* The following AMD GPUs have been fully tested for compatibility with rocSHMEM:

  * MI250X

  * MI300X

  * MI350X (Requires ROCm 7.0 or later)

  .. note::

    Other AMD GPUs might function with unknown limitations. For the complete list of supported hardware, see `ROCm System Requirements <https://rocm.docs.amd.com/projects/install-on-linux-internal/en/latest/reference/system-requirements.html>`_.

* The RO backend requires ROCm-aware Open MPI and UCX. When using the IPC or GDA backends, MPI is optional.
  For more information about installing ROCm-aware Open MPI and UCX, see :ref:`install-dependencies`.

* Inter-node communication requires AMD Pollara IONIC, Broadcom Thor 2, or CX7 Infiniband NICs.

Available network backends
--------------------------

rocSHMEM supports the following network backends:

* The **IPC (Inter-Process Communication)** backend enables fast communication between GPUs on the same host using ROCm inter-process mechanisms. It does not support inter-node communication.
* The **RO (Reverse Offload)** backend enables communication between GPUs on different nodes through a NIC, using a host-based proxy to forward communication orders to and from the GPU. RO is built on an MPI-RMA compatibility layer.
* The **GDA (GPU Direct Async)** backend enables communication between GPUs on different nodes through a NIC. In this backend, the GPU directly interacts with the NIC with no host (CPU) involvement in the critical path of communication.

You can activate IPC, RO, and GDA backends in the same rocSHMEM build.

.. note::

  When RO + IPC is active, all atomic operations use the RO backend, even for intra-node communication.
  When GDA + IPC is active, all atomic operations use the GDA backend, even for intra-node communication.

Installing from a package manager
---------------------------------

On Ubuntu, you can install rocSHMEM by running:

.. code-block:: bash

   apt install rocshmem-dev

.. note::

  This installation method requires ROCm 6.4 or later. You must manually build dependencies such as Open MPI and UCX, because the distribution packaged versions don't include full accelerator support. For more information, see :ref:`install-dependencies`.

.. _install-dependencies:

Building dependencies
---------------------

GDA NIC dependencies
^^^^^^^^^^^^^^^^^^^^

- GDA on Mellanox NICs should work on any recent version of rdma-core.
- GDA on Broadcom Thor requires driver version 233.2.108.0 and firmware version 233.2.104.0 or later.


Building rocSHMEM with MPI (Optional)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

rocSHMEM requires ROCm-Aware Open MPI and UCX for the RO backend.
MPI is optional with the IPC and GDA backends.
Other MPI implementations, such as MPICH, have not been fully tested.

To build and configure ROCm-Aware UCX 1.17.0 or later, run:

.. code-block:: bash

  git clone https://github.com/ROCm/ucx.git -b v1.17.x
  cd ucx
  ./autogen.sh
  ./configure --prefix=<prefix_dir> --with-rocm=<rocm_path> --enable-mt
  make -j 8
  make -j 8 install

To build Open MPI 5.0.7 or later with UCX support, run:

.. code-block:: bash

  git clone --recursive https://github.com/open-mpi/ompi.git -b v5.0.x
  cd ompi
  ./autogen.pl
  ./configure --prefix=<prefix_dir> --with-rocm=<rocm_path> --with-ucx=<ucx_path>
  make -j 8
  make -j 8 install

Alternatively, you can use a script to install dependencies:

.. code-block:: bash

  export BUILD_DIR=/path/to/not_rocshmem_src_or_build/dependencies
  /path/to/rocshmem_src/scripts/install_dependencies.sh

.. note::

  Configuration options vary by platform. Review the script to ensure it is compatible with your system.

For more information about OpenMPI-UCX support, see
`GPU-enabled Message Passing Interface <https://rocm.docs.amd.com/en/latest/how-to/gpu-enabled-mpi.html>`_.

Installing from source
--------------------------------

You can choose from three communication backends at build time for rocSHMEM: IPC, RO, and GDA.
Backend can be combined during build time.

MPI is not required to build rocSHMEM. To disable MPI, pass
the following flag to the build configuration scripts ``-DUSE_EXTERNAL_MPI=OFF``.
However, this will disable the functional and unit
tests, as they required MPI to run.

All backends build
^^^^^^^^^^^^^^^^^^

To build and install rocSHMEM with all three backends, run:

.. code-block:: bash

  git clone git@github.com:ROCm/rocSHMEM.git
  cd rocSHMEM
  mkdir build
  cd build
  ../scripts/build_configs/all_backends

The build script passes configuration options to CMake to set up a canonical build.

.. note::

 This builds rocSHMEM with all backends. You can select IPC, RO, GDA, or any combination at runtime. However, this portability can reduce performance, so the other build scripts are recommended if you need maximum performance.

GDA backend build
^^^^^^^^^^^^^^^^^

To build and install rocSHMEM with the GDA backends, run:


.. code-block:: bash

  git clone git@github.com:ROCm/rocSHMEM.git
  cd rocSHMEM
  mkdir build
  cd build

  # Choose one of the following scripts for your NIC vendor:
  ../scripts/build_configs/gda_bnxt  # Broadcom
  ../scripts/build_configs/gda_ionic # AMD Pollara
  ../scripts/build_configs/gda_mlx5  # Mellanox


The build script passes configuration options to CMake to set up a canonical build.

RO and IPC backend build
^^^^^^^^^^^^^^^^^^^^^^^^

To build and install rocSHMEM with the hybrid RO (off-node) and IPC (on-node) backends, run:


.. code-block:: bash

  git clone git@github.com:ROCm/rocSHMEM.git
  cd rocSHMEM
  mkdir build
  cd build
  ../scripts/build_configs/ro_ipc

The build script passes configuration options to CMake to set up a canonical build.

.. note::

  The only officially supported configuration for the RO backend uses Open MPI and UCX with a CX7 InfiniBand adapter. For more information, see :ref:`install-dependencies`. Other configurations, such as MPI implementations that are thread-safe and support GPU buffers, might work but are considered experimental.


IPC only backend build
^^^^^^^^^^^^^^^^^^^^^^

To build and install rocSHMEM with the IPC on-node, GPU-to-GPU backend, run:

.. code-block:: bash

  git clone git@github.com:ROCm/rocSHMEM.git
  cd rocSHMEM
  mkdir build
  cd build
  ../scripts/build_configs/ipc_single

The build script passes configuration options to CMake to setup a single-node build.
This is similar to the default build in ROCm 6.4.

.. note::

  The default configuration changed from IPC only in ROCm 6.4 (built with the ``ipc_single`` script) to RO and IPC in ROCm 7.0 (built with the ``ro_ipc`` script).
  Other experimental configuration scripts are available in ``./scripts/build_configs``, but only ``ipc_single`` and ``ro_ipc``
  are officially supported.

Installation prefix
^^^^^^^^^^^^^^^^^^^

By default, the build scripts install the library to ``~/rocshmem``. You can customize the installation path by adding
the desired path as the script parameter. For example, to relocate the default configuration:

.. code-block:: bash

  ../scripts/build_configs/ro_ipc /path/to/install

