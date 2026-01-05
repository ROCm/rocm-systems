.. meta::
   :description: Information on how to build the RCCL library from source code
   :keywords: RCCL, ROCm, library, API, build, install

.. _building-from-source:

*********************************************
Building and installing RCCL from source code
*********************************************

To build RCCL directly from the source code, follow these steps. This guide also includes
instructions explaining how to test the build.
For information on using the quick start install script to build RCCL, see :doc:`installation`.

Requirements
============

The following prerequisites are required to build RCCL:

1. ROCm-supported GPUs
2. Having the ROCm stack installed on the system, including the :doc:`HIP runtime <hip:index>` and the HIP-Clang compiler.

Building the library using CMake:
---------------------------------

To build the library from source, follow these steps:

.. code-block:: shell

    git clone --recursive https://github.com/ROCm/rccl.git
    cd rccl
    mkdir build
    cd build
    cmake ..
    make -j 16      # Or some other suitable number of parallel jobs

If you have already cloned the repository, you can checkout the external submodules manually.

.. code-block:: shell

    git submodule update --init --recursive --depth=1

You can substitute a different installation path by providing the path as a parameter
to ``CMAKE_INSTALL_PREFIX``, for example:

.. code-block:: shell

    cmake -DCMAKE_INSTALL_PREFIX=$PWD/rccl-install -DCMAKE_BUILD_TYPE=Release ..

.. note::

    Ensure ROCm CMake is installed using the command ``apt install rocm-cmake``. By default,
    CMake builds the component in debug mode unless ``DCMAKE_BUILD_TYPE`` is specified.


Building the RCCL package and install package:
----------------------------------------------

After you have cloned the repository and built the library as described in the previous section,
use this command to build the package:

.. code-block:: shell

    cd rccl/build
    make package
    sudo dpkg -i *.deb

.. note::
   
   The RCCL package install process requires ``sudo`` or root access because it creates a directory
   named ``rccl`` in ``/opt/rocm/``. This is an optional step. RCCL can be used directly by including the path containing ``librccl.so``.

Testing RCCL
============

The RCCL unit tests are implemented using the Googletest framework in RCCL. These unit tests require Googletest 1.10
or higher to build and run (this dependency can be installed using the ``-d`` option for ``install.sh``).
To run the RCCL unit tests, go to the ``build`` folder and the ``test`` subfolder,
then run the appropriate RCCL unit test executables.

The RCCL unit test names follow this format:

.. code-block:: shell

    CollectiveCall.[Type of test]

Filtering of the RCCL unit tests can be done using environment variables
and by passing the ``--gtest_filter`` command line flag:

.. code-block:: shell

    UT_DATATYPES=ncclBfloat16 UT_REDOPS=prod ./rccl-UnitTests --gtest_filter="AllReduce.C*"

This command runs only the ``AllReduce`` correctness tests with the ``float16`` datatype.
A list of the available environment variables for filtering appears at the top of every run.
See the `Googletest documentation <https://google.github.io/googletest/advanced.html#running-a-subset-of-the-tests>`_
for more information on how to form advanced filters.

There are also other performance and error-checking tests for RCCL. They are maintained separately at `<https://github.com/ROCm/rccl-tests>`_.

.. note::

    For more information on how to build and run rccl-tests, see the `rccl-tests README file <https://github.com/ROCm/rccl-tests/blob/develop/README.md>`_ .
