.. meta::
   :description: How to perform RCCL unit testing
   :keywords: RCCL, ROCm, library, API,

.. _integrate_rccl:

*************************
Perform RCCL unit testing
*************************

These tests check both the performance and the correctness of RCCL operations.

Build the tests
===============

Use the ``make`` or ``make -j`` command to build the tests.

If HIP is not installed in ``/opt/rocm``, you may specify ``HIP_HOME``. 
Similarly, if RCCL (``librccl.so``) is not installed in ``/opt/rocm/lib/``, you may specify ``NCCL_HOME`` and ``CUSTOM_RCCL_LIB``.

Here's an example:

.. code-block:: shell

   $ make HIP_HOME=/path/to/hip NCCL_HOME=/path/to/rccl

Configure MPI support
---------------------

RCCL tests rely on MPI to work on multiple processes, hence multiple nodes. 
To compile RCCL tests with MPI support, you need to set ``MPI=1`` and set ``MPI_HOME`` to the path where MPI is installed.

Here's an example:

.. code-block:: shell

   $ make MPI=1 MPI_HOME=/path/to/mpi HIP_HOME=/path/to/hip NCCL_HOME=/path/to/rccl

Build with CMake
----------------

RCCL tests can also be built using CMake. A typical sequence would be:

.. code-block:: shell

   $ mkdir build
   $ cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/rocm ..
   $ make

When using CMake to build RCCL tests with a custom/user-built ``librccl.so``, ensure that RCCL has been installed (using ``make install``) and not pointing to the RCCL ``build`` directory since CMake checks for the CMake target and config files. 

.. Clarify this
This is not necessary as one can modify ``LD_LIBRARY_PATH`` to point to the custom/user-built ``librccl.so`` when running RCCL Tests.

When you build with CMake, the application automatically checks for an MPI installation during the build. 
The tests can be compiled with MPI support by adding the ``-DUSE_MPI=ON`` flag to the CMake command line.

.. tip::

   You can link against a specific MPI library by using one of these options:

   - Setting the environment variable ``MPI_HOME``
   - Adding the path to the MPI library to the CMake prefix path with ``-DCMAKE_PREFIX_PATH``
   - Including the paths to MPI ``bin`` and ``lib`` in the ``PATH`` and ``LD_LIBRARY_PATH`` environment variables, respectively

Here's an example:

.. code-block:: shell

   $ mkdir build
   $ cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/mpi;/path/to/rocm" -DUSE_MPI=ON ..
   $ make

By default, for both ``Makefile``- and CMake-based builds, RCCL tests link against all supported GPU targets (defined in ``src/Makefile``, and as ``DEFAULT_GPUS`` in ``CMakeLists.txt``).

To target specific GPUs, and potentially reduce build time, use:

- ``GPU_TARGETS`` as a comma-separated string listing the GPUs to target for a ``Makefile``-based build. Here's an example where the RCCL tests are built using ``Makefile`` specifically for the GFX942 and GFX950 GPUs:
    
  .. code-block:: shell
   
     $ GPU_TARGETS="gfx942,gfx950" make MPI=1 MPI_HOME=/path/to/mpi NCCL_HOME=/opt/rocm
    
- ``-DGPU_TARGETS`` as a semicolon-separated string listing the GPUs to target for the CMake-based build. Here's an example where the RCCL tests are built using CMake for the GFX90A, GFX942, and GFX1200 GPUs:
    
  .. code-block:: shell
   
     $ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/mpi;/path/to/rocm" -DUSE_MPI=ON -DGPU_TARGETS="gfx90a;gfx942;gfx1200;" ..
    
- For CMake builds, the flag ``DBUILD_LOCAL_GPU_TARGET_ONLY`` queries and builds for the local GPU target only (similar to RCCL).
    
  .. code-block:: shell
    
     $ cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="/path/to/mpi;/path/to/rocm" -DUSE_MPI=ON -DBUILD_LOCAL_GPU_TARGET_ONLY=ON ..
    

``-DBUILD_LOCAL_GPU_TARGET_ONLY`` won't work with ``docker build``-based setups, as the Docker build engine can't query the local GPU architecture. 
Use ``-DGPU_TARGETS`` for CMake-based builds or ``GPU_TARGETS`` for Makefile-based builds when building RCCL tests using a Dockerfile and ``docker build``.

Run RCCL tests
==============

RCCL tests can run on multiple processes, threads, and HIP devices per thread. 
The number of processes is managed by MPI; it's not passed to the tests as an argument. 
The total number of ranks (``=HIP devices``) will be equal to ``(number of processes)\*(number of threads)\*(number of GPUs per thread)``.

Unit tests
----------

Unit tests for RCCL tests are implemented with pytest (Python3 is also required). Several notes for the unit tests:

- The ``LD_LIBRARY_PATH`` environment variable must be set to include ``/path/to/rccl-install/lib/`` to run the unit tests.
- The ``HSA_FORCE_FINE_GRAIN_PCIE`` environment variable must be set to ``1`` to run unit tests which use a fine-grained memory type.

The unit tests can be invoked within the `RCCL tests GitHub <https://github.com/ROCm/rccl-tests>`_ root, or in the `test <https://github.com/ROCm/rccl-tests/tree/develop/test>`_ subfolder. Here's an example call to the unit tests:

.. code-block:: shell

   $ LD_LIBRARY_PATH=/path/to/rccl-install/lib/ HSA_FORCE_FINE_GRAIN_PCIE=1 python3 -m pytest

Quick examples
--------------

Run on single node with 8 GPUs (``-g 8``), scanning from 8 Bytes to 128MBytes:

.. code-block:: shell

   $ ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

Run 64 MPI processes on nodes with 8 GPUs each, for a total of 64 GPUs spread across 8 nodes (the rccl-tests binaries must be compiled with ``MPI=1`` for this case):

.. code-block:: shell

   $ mpirun -np 64 -N 8 ./build/all_reduce_perf -b 8 -e 8G -f 2 -g 1

.. tip::

   For performance-oriented runs, on both single-node and multi-node, you should use one MPI process per GPU and ``-g 1``. 
   So, a run on 8 GPUs looks like:

   .. code-block:: shell
      
      $ mpirun -np 8 --bind-to numa ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 1

   Running with one MPI process per GPU ensures a 1:1 mapping for CPUs and GPUs, which can be beneficial for smaller message sizes and better represents the real-world use of RCCL in Deep Learning frameworks like PyTorch and TensorFlow.

Performance
-----------

See the `Performance reported by RCCL tests <https://github.com/ROCm/rccl-tests/blob/develop/doc/PERFORMANCE.md>`_ for an in-depth view of the performance numbers captured by the RCCL tests.

Environment variables
---------------------

On some earlier versions of ROCm (before ROCm 6.4.0), setting ``HSA_NO_SCRATCH_RECLAIM=1`` as part of the environment is necessary to achieve better performance on MI300 GPUs. 
When running without MPI, use a command similar to this:

.. code-block:: shell
   
   HSA_NO_SCRATCH_RECLAIM=1 ./build/all_reduce_perf -b 8 -e 128M -f 2 -g 8

For MPI (using MPICH), you need to use a command similar to this:

.. code-block:: shell

   mpirun.mpich -np 8 -env NCCL_DEBUG=VERSION -env HSA_NO_SCRATCH_RECLAIM=1 ./build/all_reduce_perf -b 8M -e 128M -i 8388608 -g 1 -d bfloat16

Arguments
---------

All tests support the same set of arguments:

- Number of GPUs:
  
  - ``-t,--nthreads <num threads>`` number of threads per process. Default: ``1``
  - ``-g,--ngpus <GPUs per thread>`` number of gpus per thread. Default: ``1``
- Sizes to scan:
  
  - ``-b,--minbytes <min size in bytes>`` minimum size to start with. Default: ``32M``
  - ``-e,--maxbytes <max size in bytes>`` maximum size to end at. Default: ``32M``
  - Increments can be either fixed or a multiplication factor. Only one of those should be used:

    - ``-i,--stepbytes <increment size>`` fixed increment between sizes. Default: ``1M``
    - ``-f,--stepfactor <increment factor>`` multiplication factor between sizes. Default: ``disabled``

- RCCL operations arguments:

  - ``-o,--op <sum/prod/min/max/avg/all>`` Specify which reduction operation to perform. Only relevant for reduction operations like Allreduce, Reduce or ReduceScatter. Default: ``Sum``
  - ``-d,--datatype <nccltype/all>`` Specify which datatype to use. Default: ``Float``
  - ``-r,--root <root/all>`` Specify which root to use. Only for operations with a root like broadcast or reduce. Default: ``0``
  - ``-y,--memory_type <coarse/fine/host/managed>`` Default: ``Coarse``
  - ``-u,--cumask <d0,d1,d2,d3>`` Default: ``None``

- Performance:

  - ``-n,--iters <iteration count>`` number of iterations. Default: ``20``
  - ``-w,--warmup_iters <warmup iteration count>`` number of warmup iterations (not timed). Default: ``5``
  - ``-m,--agg_iters <aggregation count>`` number of operations to aggregate together in each iteration. Default: ``1``
  - ``-N,--run_cycles <cycle count>`` run & print each cycle. Default: ``1; 0=infinite``
  - ``-a,--average <0/1/2/3>`` Report performance as an average across all ranks (MPI=1 only). <0=Rank0,1=Avg,2=Min,3=Max>. Default: ``1``

- Test operation:

  - ``-p,--parallel_init <0/1>`` use threads to initialize NCCL in parallel. Default: ``0``
  - ``-c,--check <check iteration count>`` perform count iterations, checking correctness of results on each iteration. This can be quite slow on large numbers of GPUs. Default: ``1``
  - ``-z,--blocking <0/1>`` Make RCCL collective blocking, i.e. have CPUs wait and sync after each collective. Default: ``0``
  - ``-G,--hipgraph <num graph launches>`` Capture iterations as a HIP graph and then replay specified number of times. Default: ``0``
  - ``-C,--report_cputime <0/1>]`` Report CPU time instead of latency. Default: ``0``
  - ``-R,--local_register <0/1/2>`` enable local (1) or symmetric (2) buffer registration on send/recv buffers. Default: ``0``
  - ``-T,--timeout <time in seconds>`` timeout each test after specified number of seconds. Default: ``disabled``
  - ``-F,--cache_flush <cache flush after every -F iteration>`` Enable cache flush after every -F iteration. Default: ``0`` (No cache flush)
  - ``-O,--out_of_place <0=in-place only, 1=out-of-place only>``. Default: ``both``
  - ``-q,--delay <delay>`` Delay between out-of-place and in-place runs (in microseconds). Default: ``10``
  
- Parsing RCCL-Tests output:

  - ``-Z,--output_format <csv|json>`` Parse RCCL-Tests output as a CSV or JSON. Default: ``disabled``
  - ``-x,--output_file <output file name>`` RCCL-Tests output file name. Default: ``disabled``
  - ``-M,--output_algo_proto_channels <0/1>`` Report Algorithm/Protocol/Channels for each message size. Default: ``0``

Run multiple operations in parallel
-----------------------------------

RCCL tests allow you to partition a set of GPUs into smaller sets, each executing the same operation in parallel. 
To split the GPUs, RCCL computes a "color" for each rank based on the ``NCCL_TESTS_SPLIT`` environment variable, then all ranks
with the same color will end up in the same group. The resulting group is printed next to each GPU at the beginning of the test.

``NCCL_TESTS_SPLIT`` takes the syntax: ``<operation><value>``. 
``<operation>`` can be: ``AND``, ``OR``, ``MOD`` or ``DIV``. The ``&``, ``|``, ``%``, and ``/`` symbols are also supported. 
The value can be either decimal, hexadecimal (prefixed by ``0x``), or binary (prefixed by ``0b``).

``NCCL_TESTS_SPLIT_MASK="<value>"`` is equivalent to ``NCCL_TESTS_SPLIT="&<value>"``.

Here are a few examples:

- ``NCCL_TESTS_SPLIT="AND 0x7"`` or ``NCCL_TESTS_SPLIT="MOD 8"``: On systems with 8 GPUs, run 8 parallel operations, each with 1 GPU per node (purely communicating over the inter-node network).
- ``NCCL_TESTS_SPLIT="OR 0x7"`` or ``NCCL_TESTS_SPLIT="DIV 8"``: On systems with 8 GPUs, run one operation per node, purely intra-node.
- ``NCCL_TESTS_SPLIT="AND 0x1"`` or ``NCCL_TESTS_SPLIT="MOD 2"``: Run two operations, each operation using every other rank.

Note that the reported bandwidth is per group, hence to get the total bandwidth used by all groups, one must multiply by the number of groups.