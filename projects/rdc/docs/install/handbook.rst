.. meta::
  :description: documentation of the installation, configuration, and use of the ROCm Data Center tool
  :keywords: ROCm Data Center tool, RDC, ROCm, API, reference, data type, support

.. _rdc-handbook:

******************
Build and test RDC
******************

To build RDC as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/docs/development/README.md>`__.
TheRock is the recommended way to build ROCm components from source.

Alternatively, you can build RDC standalone using the following
instructions.

Build and install RDC
=====================

To build and install, clone the RDC source code from GitHub and use CMake.

.. code-block:: shell

    $ git clone 'https://github.com/ROCm/rocm-systems' --recursive
    $ cd rocm-systems/projects/rdc
    $ mkdir -p build; cd build
    $ cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT"..
    $ make
    #Install library file and header and the default location is /opt/rocm
    $ make install


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
