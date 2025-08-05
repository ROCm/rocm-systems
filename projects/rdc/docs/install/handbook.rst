.. meta::
  :description: documentation of the installation, configuration, and use of the ROCm Data Center tool
  :keywords: ROCm Data Center tool, RDC, ROCm, API, reference, data type, support

.. _rdc-handbook:

***************************************************
Building and testing RDC
***************************************************

RDC is open source and available under the MIT License. This section is helpful for open source developers. Third-party integrators may also find this information useful.


Build and Install RDC
=====================

To build and install, clone the RDC source code from GitHub and use CMake.

.. code-block:: shell

    $ git clone <GitHub for RDC>
    $ cd rdc
    $ mkdir -p build; cd build
    $ cmake -DROCM_DIR=/opt/rocm -DGRPC_ROOT="$GRPC_PROTOC_ROOT"..
    $ make
    #Install library file and header and the default location is /opt/rocm
    $ make install


Build Documentation
-------------------

You can generate PDF documentation after a successful build. The reference manual, refman.pdf, appears in the latex directory.

.. code-block:: shell

    $ make doc
    $ cd latex
    $ make


Build Unit Tests for RDC Tool
-----------------------------

.. code-block:: shell

    $ cd rdc/tests/rdc_tests
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


