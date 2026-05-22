.. meta::
  :description: documentation of the installation, configuration, and use of the ROCm Data Center tool
  :keywords: ROCm Data Center tool, RDC, ROCm, API, reference, data type, support

.. _rdc-handbook:

**********************
Build RDC from source
**********************

To build RDC as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build RDC standalone using the instructions given in the following sections.

Build and install RDC
=====================

To build RDC from source, install the dependencies and follow the steps given in the following sections.

Dependencies
-----------------

- CMake >= 3.15

- g++ (5.4.0)

- gRPC and protoc

- libcap-dev

   .. code-block:: shell

     $ sudo apt install libcap-dev

For building latest documentation:

- Doxygen (1.8.11)

    .. code-block:: shell

      $ sudo apt install -y doxygen

- LaTeX (pdfTeX 3.14159265-2.6-1.40.16)

Build gRPC and Protoc
----------------------

gRPC and Protoc must be built from source as the prebuilt packages are not available for the same. Here are the steps:

1. Install the required tools:

   .. code-block:: shell

    sudo apt-get update
    sudo apt-get install automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang libc++-dev curl libcap-dev

2. Clone and build gRPC:

   .. code-block:: shell

    git clone -b v1.67.1 https://github.com/grpc/grpc --depth=1 --shallow-submodules --recurse-submodules
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

    git clone https://github.com/ROCm/rocm-systems --recursive
    cd rocm-systems/projects/rdc

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

Build documentation
-------------------

You can generate PDF documentation after a successful build. The reference manual, refman.pdf, appears in the latex directory.

.. code-block:: shell

    $ make doc
    $ cd latex
    $ make

Build unit tests for RDC tool
-----------------------------

.. code-block:: shell

    $ cd rocm-systems/projects/rdc/tests/rdc_tests
    $ mkdir -p build; cd build
    $ cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT"..
    $ make

    # To run the tests

    $ cd build/rdctst_tests
    $ ./rdctst

Test
----

.. code-block:: shell

    # Run rdcd daemon
    $ LD_LIBRARY_PATH=$PWD/rdc_libs/  ./server/rdcd -u

    # In another console run the RDC command-line
    $ LD_LIBRARY_PATH=$PWD/rdc_libs/  ./rdci/rdci discovery -l -u
