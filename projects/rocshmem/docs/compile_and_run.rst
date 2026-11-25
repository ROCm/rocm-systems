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
    --offload-arch=<target>[:<xnack>]                  \
    -I/opt/rocm/include                                \
    -I$ROCSHMEM_INSTALL_DIR/include                    \
    -I$OPENMPI_UCX_INSTALL_DIR/include/

  # Link
  hipcc -fgpu-rdc --hip-link rocshmem_allreduce_test.o -o rocshmem_allreduce_test \
    --offload-arch=<target>[:<xnack>]                                             \
    $ROCSHMEM_INSTALL_DIR/lib/librocshmem.a                                       \
    $OPENMPI_UCX_INSTALL_DIR/lib/libmpi.so                                        \
    -L/opt/rocm/lib -lamdhip64 -lhsa-runtime64

Pass the same ``--offload-arch`` to both the compile and the link step. To find
the correct value for your machine, run:

.. code-block:: bash

  rocminfo | grep amdgcn

which prints a string such as ``amdgcn-amd-amdhsa--gfx942:sramecc+:xnack-``.
The architecture is ``gfx942`` and XNACK is disabled, so the flag is
``--offload-arch=gfx942:xnack-``.

.. note::

  The XNACK suffix must match the runtime configuration of the target GPU.
  XNACK is runtime-configurable and can differ between machines; a mismatch
  causes device code to fail to load. The same consideration applies to
  ``GPU_TARGETS`` when building rocSHMEM itself - see
  :doc:`Build from source <./build>`.

If your project uses CMake, see
`Using CMake with AMD ROCm <https://rocmdocs.amd.com/en/docs-7.2.4/conceptual/cmake-packages.html>`_.

Using the tile API
------------------

Device code that calls the tile routines must include ``rocshmem_device.hpp``
after ``rocshmem.hpp``. The header provides the template wrappers that forward
to the precompiled bitcode and is only meaningful during device compilation:

.. code-block:: cpp

  #include <rocshmem/rocshmem.hpp>
  #ifdef __HIP_DEVICE_COMPILE__
  #include <rocshmem/rocshmem_device.hpp>
  #endif

See :doc:`Tile routines <./api/tile>`.

Launching kernels that use the LDS-staged copy path
---------------------------------------------------

On gfx1250 builds configured with ``-DUSE_LDS_STAGE=ON`` or ``-DUSE_TDM=ON``,
work-group and wave ``put``/``get`` can be staged through LDS. The staging
buffer is supplied by the application as dynamic LDS, so the kernel launch must
reserve it:

.. code-block:: cpp

  size_t shared_bytes = rocshmem_query_lds_stage_bytes();
  hipLaunchKernelGGL(my_kernel, grid, block, shared_bytes, stream, /* args */);

and the kernel must register it:

.. code-block:: cpp

  extern __shared__ uint8_t lds_stage[];
  rocshmem_set_lds_stage(lds_stage, rocshmem_query_lds_stage_bytes());

On builds without either option ``rocshmem_query_lds_stage_bytes()`` returns
``0`` and ``rocshmem_set_lds_stage()`` is a no-op, so the same source compiles
and runs unchanged. See :doc:`LDS-staged copy routines <./api/lds_stage>`.

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
