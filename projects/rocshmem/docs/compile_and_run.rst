.. meta::
   :description: Information on how to compile and run rocSHMEM applications.
   :keywords: rocSHMEM, ROCm, library, API, compile, link, hipcc

.. _running-applications:

--------------------------------------------------
Compiling and running rocSHMEM applications
--------------------------------------------------

This topic explains how to compile and run rocSHMEM applications.

Compiling and linking with rocSHMEM
-----------------------------------

rocSHMEM is a library that can be statically linked to your application during compilation with ``hipcc``. For more information, see :doc:`HIPCC <hipcc:index>`.

When compiling your application with ``hipcc``, you must include the rocSHMEM header files and the rocSHMEM library.
Because rocSHMEM depends on MPI (Message Passing Interface), you must manually add the arguments for MPI linkage instead of using ``mpicc``.

When using ``hipcc`` directly without a build system, it's recommended to perform the compilation and linking steps separately.

Example compile and link commands are provided at the top of the example files in the ``examples`` directory:

.. code-block:: bash

  # Compile
  hipcc -c -fgpu-rdc -x hip rocshmem_allreduce_test.cc \
    -I/opt/rocm/include                                \
    -I$ROCSHMEM_INSTALL_DIR/include                    \
    -I$OPENMPI_UCX_INSTALL_DIR/include/

  # Link
  hipcc -fgpu-rdc --hip-link rocshmem_allreduce_test.o -o rocshmem_allreduce_test \
    $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a                                       \
    $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so                                        \
    -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

If your project uses CMake, see
`Using CMake with AMD ROCm <https://rocmdocs.amd.com/en/latest/conceptual/cmake-packages.html>`_.

Running a rocSHMEM application
------------------------------

Applications using rocSHMEM typically deploy multiple processes, usually one per GPU.
The MPI launcher, for example, ``mpiexec`` with Open MPI, is used to start the required number
of processes. For example, to launch two ``getmem`` example processes (available when compiled from source):

.. code-block:: bash

  mpiexec --map-by numa --mca pml ucx --mca osc ucx -np 2 ./build/examples/rocshmem_getmem_test

See the `Open MPI documentation <https://docs.open-mpi.org/en/main/>`_ for more information about ``mpiexec`` command line parameters.

.. note::

  Some systems may have multiple MPI installations, some of which do not
  have GPU support enabled. You must use the ``mpiexec`` from the expected
  MPI library, especially when using the MPI built by yourself
  as part of :ref:`install-dependencies`.
