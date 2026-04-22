.. meta::
  :description: Documentation of the MPI usage for rocprofv3
  :keywords: ROCprofiler-SDK tool, mpirun, rocprofv3, rocprofv3 tool usage, mpich, ROCprofiler-SDK command line tool, ROCprofiler-SDK CLI


.. _using-rocprofv3-with-mpi:

Using rocprofv3 with MPI
+++++++++++++++++++++++++++++

Message Passing Interface (MPI) is a standardized and portable message-passing system designed to function on a wide variety of parallel computing architectures. MPI is widely used for developing parallel applications and is considered the de facto standard for communication in high-performance computing (HPC) environments.
MPI applications are parallel programs that run across multiple processes, which can be distributed over one or more nodes.

For MPI applications or other job launchers such as `SLURM <https://slurm.schedmd.com/documentation.html>`_, place ``rocprofv3`` inside the job launcher. The following example demonstrates how to use ``rocprofv3`` with MPI:

.. code-block:: bash

    mpirun -n 4 rocprofv3 --hip-trace --output-format csv -- <application_path>

The preceding command runs the application with ``rocprofv3`` and generates the trace file for each rank. The trace files are prefixed with the process ID.

.. code-block:: bash

    2293213_agent_info.csv
    2293213_hip_api_trace.csv
    2293214_agent_info.csv
    2293214_hip_api_trace.csv
    2293212_agent_info.csv
    2293212_hip_api_trace.csv
    2293215_agent_info.csv
    2293215_hip_api_trace.csv

Since the data collection is performed in-process, it's ideal to collect data from within the processes launched by MPI. When ``rocprofv3`` is run outside of ``mpirun``, the tool library is loaded into the `mpirun` executable..
Collecting data outside of ``mpirun`` works but fetches agent info for the ``mpirun`` process too. For example:

.. code-block:: bash

    rocprofv3 --hip-trace -d %h.%p.%env{OMPI_COMM_WORLD_RANK}% --output-format csv -- mpirun -n 2  <application_path>

In the preceding example, an extra agent info file is generated for the ``mpirun`` process. The trace files are prefixed with the hostname, process ID, and the MPI rank.

.. code-block:: bash

    ubuntu-latest.3000020.1/3000020_agent_info.csv
    ubuntu-latest.3000020.0/3000019_agent_info.csv
    ubuntu-latest.3000020.1/3000020_hip_api_trace.csv
    ubuntu-latest.3000020.0/3000019_hip_api_trace.csv

ROCTx annotations
===================

For an MPI application, you can use ROCTx annotations to mark the start and end of the MPI code region. The following example demonstrates how to use ROCTx annotations with MPI:

.. code-block:: cpp

    #include <roctx.h>
    #include <mpi.h>
    ...

    void run(int rank, int tid, int dev_id, int argc, char** argv)
    {
        auto roctx_run_id = roctxRangeStart("run");

        const auto mark = [rank, tid, dev_id](std::string_view suffix) {
            auto _ss = std::stringstream{};
            _ss << "run/rank-" << rank << "/thread-" << tid << "/device-" << dev_id << "/" << suffix;
            roctxMark(_ss.str().c_str());
        };

        mark("begin");

        constexpr unsigned int M = 4960 * 2;
        constexpr unsigned int N = 4960 * 2;

        unsigned long long nitr = 0;
        unsigned long long nsync = 0;

        if(argc > 2) nitr = atoll(argv[2]);
        if(argc > 3) nsync = atoll(argv[3]);

        hipStream_t stream = {};

        printf("[transpose] Rank %i, thread %i assigned to device %i\n", rank, tid, dev_id);
        HIP_API_CALL(hipSetDevice(dev_id));
        HIP_API_CALL(hipStreamCreate(&stream));

        auto_lock_t _lk{print_lock};
        std::cout << "[transpose][" << rank << "][" << tid << "] M: " << M << " N: " << N << std::endl;
        _lk.unlock();

        std::default_random_engine         _engine{std::random_device{}() * (rank + 1) * (tid + 1)};
        std::uniform_int_distribution<int> _dist{0, 1000};

        ...

        auto t1 = std::chrono::high_resolution_clock::now();
        for(size_t i = 0; i < nitr; ++i)
        {
            roctxRangePush("run/iteration");
            transpose<<<grid, block, 0, stream>>>(in, out, M, N);
            check_hip_error();
            if(i % nsync == (nsync - 1))
            {
                roctxRangePush("run/iteration/sync");
                HIP_API_CALL(hipStreamSynchronize(stream));
                roctxRangePop();
            }
            roctxRangePop();
        }
        auto t2 = std::chrono::high_resolution_clock::now();
        HIP_API_CALL(hipStreamSynchronize(stream));
        HIP_API_CALL(hipMemcpyAsync(out_matrix, out, size, hipMemcpyDeviceToHost, stream));
        double time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        float  GB   = (float) size * nitr * 2 / (1 << 30);

        print_lock.lock();
        std::cout << "[transpose][" << rank << "][" << tid << "] Runtime of transpose is " << time
                  << " sec\n";
        std::cout << "[transpose][" << rank << "][" << tid
                  << "] The average performance of transpose is " << GB / time << " GBytes/sec"
                  << std::endl;
        print_lock.unlock();

        ...

        mark("end");

        roctxRangeStop(roctx_run_id);
    }

This preceding sample generates output similar to the following:

.. code-block:: shell

    "MARKER_CORE_API","run/rank-0/thread-0/device-0/begin",2936128,2936128,5,432927100747635,432927100747635
    "MARKER_CORE_API","run/rank-0/thread-1/device-1/begin",2936128,2936397,7,432927100811475,432927100811475
    "MARKER_CORE_API","run/iteration",2936128,2936397,22,432928615598809,432928648197081
    "MARKER_CORE_API","run/iteration",2936128,2936397,61,432928648229081,432928648234041
    "MARKER_CORE_API","run/iteration",2936128,2936397,67,432928648234701,432928648239621
    "MARKER_CORE_API","run/iteration",2936128,2936397,73,432928648239971,432928648244141
    "MARKER_CORE_API","run/iteration/sync",2936128,2936397,84,432928648249791,432928664871094
    ...

    "MARKER_CORE_API","run/iteration",2936128,2936128,6313,432929397644269,432929397648369
    "MARKER_CORE_API","run/iteration/sync",2936128,2936128,6324,432929397653119,432929401455250
    "MARKER_CORE_API","run/iteration",2936128,2936128,6319,432929397648779,432929401455640
    "MARKER_CORE_API","run/rank-0/thread-1/device-1/end",2936128,2936397,6339,432929527301990,432929527301990
    "MARKER_CORE_API","run",2936128,2936397,6,432927100787035,432929527313480
    "MARKER_CORE_API","run/rank-0/thread-0/device-0/end",2936128,2936128,6342,432929612438185,432929612438185
    "MARKER_CORE_API","run",2936128,2936128,4,432927100729745,432929612448285

Output format features
=======================

To collect the profiles of the individual MPI processes, use ``rocprofv3`` with output directory option to send output to unique files.

.. code-block:: bash

    mpirun -n 2 rocprofv3 --hip-trace -d %h.%p.%env{OMPI_COMM_WORLD_RANK}% --output-format csv --  <application_path>

To see the placeholders supported by the output directory option, see :ref:`output directory placeholders <output_field_format>`.

Assuming the hostname as `ubuntu-latest`, the process IDs as 3000020 and 3000019, the generated output file names are:

.. code-block:: bash

    ubuntu-latest.3000020.1/ubuntu-latest/3000020_agent_info.csv
    ubuntu-latest.3000019.0/ubuntu-latest/3000019_agent_info.csv
    ubuntu-latest.3000020.1/ubuntu-latest/3000020_hip_api_trace.csv
    ubuntu-latest.3000019.0/ubuntu-latest/3000019_hip_api_trace.csv

Selective rank profiling
==========================

When running large-scale MPI jobs, collecting profiling data from all ranks can generate excessive output and may not be necessary. The ``--profile-mpi-ranks`` option allows you to specify which MPI ranks should provide profile and trace output, while the tool still runs on all ranks to maintain program correctness.

Specifying ranks to profile
-----------------------------

Use the ``--profile-mpi-ranks`` option with a comma-separated list of ranks and ranges:

.. code-block:: bash

    # Profile only rank 0
    mpirun -n 16 rocprofv3 --hip-trace --profile-mpi-ranks 0 -- <application_path>

    # Profile ranks 0-3 and rank 8
    mpirun -n 16 rocprofv3 --hip-trace --profile-mpi-ranks 0-3,8 -- <application_path>

    # Profile ranks 0, 4, 8, and 12
    mpirun -n 16 rocprofv3 --hip-trace --profile-mpi-ranks 0,4,8,12 -- <application_path>

    # Profile a range of ranks (10 through 15)
    srun -n 32 rocprofv3 --kernel-trace --profile-mpi-ranks 10-15 -- <application_path>

The rank specification syntax supports:

- **Individual ranks**: Comma-separated integers (e.g., ``0,1,2,8``)
- **Ranges**: Hyphen-separated start and end values (e.g., ``0-7`` for ranks 0 through 7)
- **Combined**: Mix of individual ranks and ranges (e.g., ``0-3,8,10-15``)

Behavior
---------

When using ``--profile-mpi-ranks``:

- The ``rocprofv3`` tool runs on **all** MPI ranks to avoid disrupting the application's execution
- Only the specified ranks collect and output profiling/trace data
- Non-selected ranks execute the application without profiling overhead or output generation
- This reduces output file count and storage requirements for large-scale runs

Default behavior
-----------------

If ``--profile-mpi-ranks`` is not specified, all ranks provide output (default behavior), which is equivalent to running without rank filtering.

Supported MPI implementations
-------------------------------

The rank and world size detection automatically supports the following MPI implementations and job launchers. When multiple MPI-related environment variables are present, the tool uses the following priority order (from highest to lowest priority):

1. **PBS** (``PBS_NODENUM``, ``PBS_O_TASKNUM``)
2. **SLURM** (``SLURM_PROCID``, ``SLURM_NTASKS``)
3. **PMI** (``PMI_RANK``, ``PMI_SIZE``)
4. **MVAPICH2** (``MV2_COMM_WORLD_RANK``, ``MV2_COMM_WORLD_SIZE``)
5. **OpenMPI** (``OMPI_COMM_WORLD_RANK``, ``OMPI_COMM_WORLD_SIZE``)
6. **Generic MPI** variables:

   - ``MPI_RANKID``, ``MPI_NRANKS``
   - ``MPI_LOCALRANKID``, ``MPI_LOCALNRANKS``
   - ``MPI_RANK``, ``MPI_SIZE``

The tool checks these environment variables in the order listed above and uses the first matching pair found. This priority order ensures that runtime-specific variables (PBS, SLURM) take precedence over implementation-specific variables (MVAPICH2, OpenMPI), which in turn take precedence over generic MPI variables

Custom MPI environment variables
----------------------------------

For mixed environments or non-standard MPI configurations (such as interactive SLURM jobs using mpirun), you can specify custom environment variables for rank and world size detection:

.. code-block:: bash

    # Use custom environment variables for rank and world size detection
    mpirun -n 16 rocprofv3 --hip-trace --profile-mpi-ranks 0-3 \
        --mpi-world-rank-variable MY_CUSTOM_RANK \
        --mpi-world-size-variable MY_CUSTOM_SIZE -- <application_path>

When ``--mpi-world-rank-variable`` and ``--mpi-world-size-variable`` are specified, they take precedence over automatic detection. These options are useful when:

- The MPI implementation uses non-standard environment variable names
- You want to ensure specific environment variables are used instead of relying on auto-detection
- Working in mixed MPI environments where multiple MPI-related variables might be present

The ``--mpi-world-size-variable`` option is particularly useful when using ``--profile-mpi-ranks`` to ensure that the specified rank ranges are validated against the actual MPI world size, preventing out-of-range errors.
