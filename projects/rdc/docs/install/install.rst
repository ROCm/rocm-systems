.. meta::
  :description: The ROCm Data Center tool (RDC) addresses key infrastructure challenges regarding AMD GPUs in cluster and data center environments and simplifies their administration
  :keywords: RDC installation, Install RDC, Install ROCm Data Center tool, Building ROCm Data Center, Building RDC

.. _rdc-install:

******************
RDC installation
******************

RDC is part of the AMD ROCm software and available on the distributions supported by AMD ROCm. This topic provides information required to install RDC from prebuilt packages and source.

Prerequisites
==============

To install RDC from source, ensure that your system meets the following requirements:

- **Supported platforms:** AMD ROCm-supported platform. See the `list of supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_.

- **Dependencies:**

  - CMake >= 3.15

  - g++ (5.4.0)

  - gRPC and protoc

  - libcap-dev

  - :doc:`AMD ROCm platform <rocm:index>` including:

    - :doc:`AMDSMI library <amdsmi:index>`
    - `ROCK kernel driver <https://github.com/ROCm/ROCK-Kernel-Driver>`_

  For building latest documentation:

  - Doxygen (1.8.11)

  - LaTeX (pdfTeX 3.14159265-2.6-1.40.16)

  .. code-block:: shell

    $ sudo apt install libcap-dev
    $ sudo apt install -y doxygen

Build RDC from source
======================

The following sections provide steps to build RDC from source.

Build gRPC and Protoc
----------------------

gRPC and Protoc must be built from source as the prebuilt packages are not available for the same. Here are the steps:

1. Install the required tools:

   .. code-block:: shell

    sudo apt-get update
    sudo apt-get install automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev curl libcap-dev

2. Clone and build gRPC:

   .. code-block:: shell

    git clone -b v1.74.0 https://github.com/grpc/grpc --depth=1 --shallow-submodules --recurse-submodules
    cd grpc
    export GRPC_ROOT=/opt/grpc
    cmake -B build \
        -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="$GRPC_ROOT" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release
    make -C build -j $(nproc)
    sudo make -C build install
    echo "$GRPC_ROOT" | sudo tee /etc/ld.so.conf.d/grpc.conf
    sudo ldconfig
    cd ..

Build RDC
-----------

1. Clone the RDC repository:

   .. code-block:: shell

    git clone https://github.com/ROCm/rdc
    cd rdc

2. Configure the build:

   .. code-block:: shell

    cmake -B build -DGRPC_ROOT="$GRPC_ROOT"

3. You can also enable the following optional features:

   - ROCm profiler:

     .. code-block:: shell

        cmake -B build -DBUILD_PROFILER=ON

   - ROCm Validation Suite (RVS):

     .. code-block:: shell

        cmake -B build -DBUILD_RVS=ON

   - RDC library only (without ``rdci`` and ``rdcd``):

     .. code-block:: shell

        cmake -B build -DBUILD_STANDALONE=OFF

   - RDC library without ROCm runtime:

     .. code-block:: shell

        cmake -B build -DBUILD_RUNTIME=OFF

4. Build and install:

   .. code-block:: shell

    make -C build -j $(nproc)
    sudo make -C build install

5. Update system library path:

   .. code-block:: shell

    export RDC_LIB_DIR=/opt/rocm/lib/rdc
    export GRPC_LIB_DIR="/opt/grpc/lib"
    echo "${RDC_LIB_DIR}" | sudo tee /etc/ld.so.conf.d/x86_64-librdc_client.conf
    echo "${GRPC_LIB_DIR}" | sudo tee -a /etc/ld.so.conf.d/x86_64-librdc_client.conf
    sudo ldconfig

Installing RDC using prebuilt packages
=======================================

RDC is packaged as part of the ROCm software repository. To install RDC using prebuilt package, first :doc:`install the AMD ROCm software <rocm-install-on-linux:index>`, then use the following instructions:

.. tab-set::

    .. tab-item:: Ubuntu
        :sync: ubuntu-tab

        .. code-block:: shell

            $ sudo apt-get install rdc
            # or, to install a specific version
            $ sudo apt-get install rdc<x.y.z>

    .. tab-item:: SLES 15 Service Pack 3
        :sync: sles-tab

        .. code-block:: shell

            $ sudo zypper install rdc
            # or, to install a specific version
            $ sudo zypper install rdc<x.y.z>
