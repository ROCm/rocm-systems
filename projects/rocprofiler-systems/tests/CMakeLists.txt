#
#   omnitrace tests
#
include_guard(GLOBAL)

include(${CMAKE_CURRENT_LIST_DIR}/omnitrace-testing.cmake)

# -------------------------------------------------------------------------------------- #
#
# general config file tests
#
# -------------------------------------------------------------------------------------- #

file(
    WRITE ${CMAKE_CURRENT_BINARY_DIR}/invalid.cfg
    "
OMNITRACE_CONFIG_FILE =
FOOBAR = ON
")

if(TARGET parallel-overhead)
    set(_CONFIG_TEST_EXE $<TARGET_FILE:parallel-overhead>)
else()
    set(_CONFIG_TEST_EXE ls)
endif()

add_test(
    NAME omnitrace-invalid-config
    COMMAND $<TARGET_FILE:omnitrace-exe> -- ${_CONFIG_TEST_EXE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    omnitrace-invalid-config
    PROPERTIES ENVIRONMENT
               "OMNITRACE_CONFIG_FILE=${CMAKE_CURRENT_BINARY_DIR}/invalid.cfg" TIMEOUT
               120 LABELS "config" WILL_FAIL ON)

add_test(
    NAME omnitrace-missing-config
    COMMAND $<TARGET_FILE:omnitrace-exe> -- ${_CONFIG_TEST_EXE}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set_tests_properties(
    omnitrace-missing-config
    PROPERTIES ENVIRONMENT
               "OMNITRACE_CONFIG_FILE=${CMAKE_CURRENT_BINARY_DIR}/missing.cfg" TIMEOUT
               120 LABELS "config" WILL_FAIL ON)

# -------------------------------------------------------------------------------------- #
#
# binary-rewrite and runtime-instrumentation tests
#
# -------------------------------------------------------------------------------------- #

omnitrace_add_test(
    NAME transpose
    TARGET transpose
    MPI ${TRANSPOSE_USE_MPI}
    GPU ON
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS -e -v 2 --print-instructions -E uniform_int_distribution
    RUNTIME_ARGS
        -e
        -v
        1
        --label
        file
        line
        return
        args
        -E
        uniform_int_distribution
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=ON")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING SKIP_RUNTIME
    NAME transpose-loops
    TARGET transpose
    LABELS "loops"
    MPI ${TRANSPOSE_USE_MPI}
    GPU ON
    NUM_PROCS ${NUM_PROCS}
    REWRITE_ARGS
        -e
        -v
        2
        --label
        return
        args
        -l
        -i
        8
        -E
        uniform_int_distribution
    RUN_ARGS 2 100 50
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure transpose")

omnitrace_add_test(
    SKIP_PRELOAD SKIP_RUNTIME SKIP_SAMPLING
    NAME rewrite-caller
    TARGET rewrite-caller
    LABELS "caller-include"
    REWRITE_ARGS
        -e
        -i
        256
        --caller-include
        "^inner"
        -v
        2
        --print-instrumented
        functions
    RUN_ARGS 17
    ENVIRONMENT "${_base_environment};OMNITRACE_COUT_OUTPUT=ON"
    BASELINE_PASS_REGEX "number of calls made = 17"
    REWRITE_PASS_REGEX "\\[function\\]\\[Forcing\\] caller-include-regex :: 'outer'"
    REWRITE_RUN_PASS_REGEX ">>> ._outer ([ \\|]+) 17")

set(OMNITRACE_ROCM_EVENTS_TEST
    "GRBM_COUNT,GPUBusy,SQ_WAVES,SQ_INSTS_VALU,VALUInsts,TCC_HIT_sum,TA_TA_BUSY[0]:device=0,TA_TA_BUSY[11]:device=0"
    )

if(OMNITRACE_USE_ROCPROFILER)
    omnitrace_add_test(
        SKIP_BASELINE SKIP_SAMPLING SKIP_RUNTIME
        NAME transpose-rocprofiler
        TARGET transpose
        LABELS "rocprofiler"
        MPI ${TRANSPOSE_USE_MPI}
        GPU ON
        NUM_PROCS ${NUM_PROCS}
        REWRITE_ARGS -e -v 2 -E uniform_int_distribution
        ENVIRONMENT
            "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_ROCM_EVENTS=${OMNITRACE_ROCM_EVENTS_TEST}"
        REWRITE_RUN_PASS_REGEX
            "rocprof-device-0-GRBM_COUNT.txt(.*)rocprof-device-0-GPUBusy.txt(.*)rocprof-device-0-SQ_WAVES.txt(.*)rocprof-device-0-SQ_INSTS_VALU.txt(.*)rocprof-device-0-VALUInsts.txt(.*)rocprof-device-0-TCC_HIT_sum.txt(.*)rocprof-device-0-TA_TA_BUSY_0.txt(.*)rocprof-device-0-TA_TA_BUSY_11.txt"
        )

    omnitrace_add_test(
        SKIP_BASELINE SKIP_SAMPLING SKIP_RUNTIME
        NAME transpose-rocprofiler-no-roctracer
        TARGET transpose
        LABELS "rocprofiler"
        MPI ${TRANSPOSE_USE_MPI}
        GPU ON
        NUM_PROCS ${NUM_PROCS}
        REWRITE_ARGS -e -v 2 -E uniform_int_distribution
        ENVIRONMENT
            "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_ROCTRACER=OFF;OMNITRACE_ROCM_EVENTS=${OMNITRACE_ROCM_EVENTS_TEST}"
        REWRITE_RUN_PASS_REGEX
            "rocprof-device-0-GRBM_COUNT.txt(.*)rocprof-device-0-GPUBusy.txt(.*)rocprof-device-0-SQ_WAVES.txt(.*)rocprof-device-0-SQ_INSTS_VALU.txt(.*)rocprof-device-0-VALUInsts.txt(.*)rocprof-device-0-TCC_HIT_sum.txt(.*)rocprof-device-0-TA_TA_BUSY_0.txt(.*)rocprof-device-0-TA_TA_BUSY_11.txt"
        REWRITE_RUN_FAIL_REGEX "roctracer.txt")
endif()

omnitrace_add_test(
    NAME parallel-overhead
    TARGET parallel-overhead
    REWRITE_ARGS -e -v 2 --min-instructions=8
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=8
        --label
        file
        line
        return
        args
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF")

omnitrace_add_test(
    NAME parallel-overhead-locks
    TARGET parallel-overhead-locks
    LABELS "locks"
    REWRITE_ARGS -e -i 256
    RUNTIME_ARGS -e -i 256
    RUN_ARGS 30 4 1000
    ENVIRONMENT
        "${_lock_environment};OMNITRACE_USE_TIMEMORY=ON;OMNITRACE_USE_PERFETTO=ON;OMNITRACE_COLLAPSE_THREADS=OFF;OMNITRACE_SAMPLING_REALTIME=ON;OMNITRACE_SAMPLING_REALTIME_FREQ=10;OMNITRACE_SAMPLING_REALTIME_TIDS=0"
    REWRITE_RUN_PASS_REGEX
        "wall_clock .*\\|_pthread_create .* 4 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000"
    RUNTIME_PASS_REGEX
        "wall_clock .*\\|_pthread_create .* 4 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000 .*\\|_pthread_mutex_lock .* 1000 .*\\|_pthread_mutex_unlock .* 1000"
    )

omnitrace_add_test(
    SKIP_RUNTIME SKIP_SAMPLING
    NAME parallel-overhead-locks-timemory
    TARGET parallel-overhead-locks
    LABELS "locks"
    REWRITE_ARGS -e -v 2 --min-instructions=4
    RUN_ARGS 10 4 1000
    ENVIRONMENT
        "${_lock_environment};OMNITRACE_FLAT_PROFILE=ON;OMNITRACE_USE_TIMEMORY=ON;OMNITRACE_USE_PERFETTO=OFF"
    REWRITE_RUN_PASS_REGEX
        "start_thread (.*) 4 (.*) pthread_mutex_lock (.*) 4000 (.*) pthread_mutex_unlock (.*) 4000"
    )

omnitrace_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_SAMPLING
    NAME parallel-overhead-locks-perfetto
    TARGET parallel-overhead-locks
    LABELS "locks"
    REWRITE_ARGS -e -v 2 --min-instructions=8
    RUN_ARGS 10 4 1000
    ENVIRONMENT
        "${_lock_environment};OMNITRACE_FLAT_PROFILE=ON;OMNITRACE_USE_TIMEMORY=OFF;OMNITRACE_USE_PERFETTO=ON"
    )

omnitrace_add_test(
    NAME user-api
    TARGET user-api
    LABELS "loops"
    REWRITE_ARGS -e -v 2 -l --min-instructions=8 -E custom_push_region
    RUNTIME_ARGS
        -e
        -v
        1
        -l
        --min-instructions=8
        -E
        custom_push_region
        --label
        file
        line
        return
        args
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF"
    REWRITE_RUN_PASS_REGEX "Pushing custom region :: run.10. x 1000"
    RUNTIME_PASS_REGEX "Pushing custom region :: run.10. x 1000"
    PRELOAD_PASS_REGEX "Pushing custom region :: run.10. x 1000"
    BASELINE_FAIL_REGEX "Pushing custom region"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure")

if(OMNITRACE_USE_MPI OR OMNITRACE_USE_MPI_HEADERS)
    omnitrace_add_test(
        SKIP_RUNTIME
        NAME "mpi"
        TARGET mpi-example
        MPI ON
        NUM_PROCS 4
        REWRITE_ARGS
            -e
            -v
            2
            --label
            file
            line
            return
            args
            --min-instructions
            0
        ENVIRONMENT "${_base_environment}"
        REWRITE_RUN_PASS_REGEX
            "(/[A-Za-z-]+/perfetto-trace-0.proto).*(/[A-Za-z-]+/wall_clock-0.txt')"
        REWRITE_RUN_FAIL_REGEX
            "(perfetto-trace|trip_count|sampling_percent|sampling_cpu_clock|sampling_wall_clock|wall_clock)-[0-9][0-9]+.(json|txt|proto)"
        )

    omnitrace_add_test(
        SKIP_RUNTIME SKIP_SAMPLING
        NAME "mpi-flat-mpip"
        TARGET mpi-example
        MPI ON
        NUM_PROCS 4
        LABELS "mpip"
        REWRITE_ARGS
            -e
            -v
            2
            --label
            file
            line
            args
            --min-instructions
            0
        ENVIRONMENT
            "${_flat_environment};OMNITRACE_USE_SAMPLING=OFF;OMNITRACE_STRICT_CONFIG=OFF;OMNITRACE_USE_MPIP=ON"
        REWRITE_RUN_PASS_REGEX
            ">>> main(.*\n.*)>>> MPI_Init_thread(.*\n.*)>>> pthread_create(.*\n.*)>>> MPI_Comm_size(.*\n.*)>>> MPI_Comm_rank(.*\n.*)>>> MPI_Barrier(.*\n.*)>>> MPI_Alltoall"
        )

    omnitrace_add_test(
        SKIP_RUNTIME SKIP_SAMPLING
        NAME "mpi-flat"
        TARGET mpi-example
        MPI ON
        NUM_PROCS 4
        LABELS "mpip"
        REWRITE_ARGS
            -e
            -v
            2
            --label
            file
            line
            args
            --min-instructions
            0
        ENVIRONMENT "${_flat_environment};OMNITRACE_USE_SAMPLING=OFF"
        REWRITE_RUN_PASS_REGEX
            ">>> main(.*\n.*)>>> MPI_Init_thread(.*\n.*)>>> pthread_create(.*\n.*)>>> MPI_Comm_size(.*\n.*)>>> MPI_Comm_rank(.*\n.*)>>> MPI_Barrier(.*\n.*)>>> MPI_Alltoall"
        )

    set(_mpip_environment
        "OMNITRACE_USE_PERFETTO=ON"
        "OMNITRACE_USE_TIMEMORY=ON"
        "OMNITRACE_USE_SAMPLING=OFF"
        "OMNITRACE_USE_PROCESS_SAMPLING=OFF"
        "OMNITRACE_TIME_OUTPUT=OFF"
        "OMNITRACE_FILE_OUTPUT=ON"
        "OMNITRACE_USE_MPIP=ON"
        "OMNITRACE_DEBUG=OFF"
        "OMNITRACE_VERBOSE=2"
        "OMNITRACE_DL_VERBOSE=2"
        "${_test_openmp_env}"
        "${_test_library_path}")

    set(_mpip_all2all_environment
        "OMNITRACE_USE_PERFETTO=ON"
        "OMNITRACE_USE_TIMEMORY=ON"
        "OMNITRACE_USE_SAMPLING=OFF"
        "OMNITRACE_USE_PROCESS_SAMPLING=OFF"
        "OMNITRACE_TIME_OUTPUT=OFF"
        "OMNITRACE_FILE_OUTPUT=ON"
        "OMNITRACE_USE_MPIP=ON"
        "OMNITRACE_DEBUG=ON"
        "OMNITRACE_VERBOSE=3"
        "OMNITRACE_DL_VERBOSE=3"
        "${_test_openmp_env}"
        "${_test_library_path}")

    foreach(_EXAMPLE all2all allgather allreduce bcast reduce scatter-gather send-recv)
        if("${_mpip_${_EXAMPLE}_environment}" STREQUAL "")
            set(_mpip_${_EXAMPLE}_environment "${_mpip_environment}")
        endif()
        omnitrace_add_test(
            SKIP_RUNTIME SKIP_SAMPLING
            NAME "mpi-${_EXAMPLE}"
            TARGET mpi-${_EXAMPLE}
            MPI ON
            NUM_PROCS 2
            LABELS "mpip"
            REWRITE_ARGS -e -v 2 --label file line --min-instructions 0
            RUN_ARGS 30
            ENVIRONMENT "${_mpip_${_EXAMPLE}_environment}")
    endforeach()
endif()

omnitrace_add_test(
    NAME lulesh
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    GPU ${LULESH_USE_GPU}
    NUM_PROCS 8
    LABELS "kokkos"
    REWRITE_ARGS -e -v 2 --label file line return args
    RUNTIME_ARGS
        -e
        -v
        1
        --label
        file
        line
        return
        args
        -ME
        [==[lib(gomp|m-)]==]
    LABELS "kokkos;kokkos-profile-library"
    RUN_ARGS -i 25 -s 20 -p
    ENVIRONMENT
        "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=ON;OMNITRACE_COUT_OUTPUT=ON;OMNITRACE_SAMPLING_FREQ=50;OMNITRACE_KOKKOSP_PREFIX=[kokkos];KOKKOS_PROFILE_LIBRARY=libomnitrace-dl.so"
    REWRITE_RUN_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]"
    RUNTIME_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]")

omnitrace_add_test(
    SKIP_RUNTIME SKIP_REWRITE
    NAME lulesh-baseline-kokkosp-libomnitrace
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    GPU ${LULESH_USE_GPU}
    NUM_PROCS 8
    LABELS "kokkos;kokkos-profile-library"
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=ON;OMNITRACE_COUT_OUTPUT=ON;OMNITRACE_SAMPLING_FREQ=50;OMNITRACE_KOKKOSP_PREFIX=[kokkos];KOKKOS_PROFILE_LIBRARY=libomnitrace.so"
    BASELINE_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]")

omnitrace_add_test(
    SKIP_RUNTIME SKIP_REWRITE
    NAME lulesh-baseline-kokkosp-libomnitrace-dl
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    GPU ${LULESH_USE_GPU}
    NUM_PROCS 8
    LABELS "kokkos;kokkos-profile-library"
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=ON;OMNITRACE_COUT_OUTPUT=ON;OMNITRACE_SAMPLING_FREQ=50;OMNITRACE_KOKKOSP_PREFIX=[kokkos];KOKKOS_PROFILE_LIBRARY=libomnitrace-dl.so"
    BASELINE_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]")

omnitrace_add_test(
    SKIP_BASELINE
    NAME lulesh-kokkosp
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    GPU ${LULESH_USE_GPU}
    NUM_PROCS 8
    LABELS "kokkos"
    REWRITE_ARGS -e -v 2
    RUNTIME_ARGS
        -e
        -v
        1
        --label
        file
        line
        return
        args
        -ME
        [==[lib(gomp|m-)]==]
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_base_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=ON")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING
    NAME lulesh-perfetto
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    GPU ${LULESH_USE_GPU}
    NUM_PROCS 8
    LABELS "kokkos;loops"
    REWRITE_ARGS -e -v 2
    RUNTIME_ARGS
        -e
        -v
        1
        -l
        --dynamic-callsites
        --traps
        --allow-overlapping
        -ME
        [==[libgomp]==]
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_perfetto_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=OFF")

omnitrace_add_test(
    SKIP_SAMPLING
    NAME lulesh-timemory
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    GPU ${LULESH_USE_GPU}
    NUM_PROCS 8
    LABELS "kokkos;loops"
    REWRITE_ARGS -e -v 2 -l --dynamic-callsites --traps --allow-overlapping
    RUNTIME_ARGS
        -e
        -v
        1
        -l
        --dynamic-callsites
        -ME
        [==[libgomp]==]
        -d
        wall_clock
        peak_rss
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_timemory_environment};OMNITRACE_CRITICAL_TRACE=OFF;OMNITRACE_USE_KOKKOSP=OFF"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure")

if(OMNITRACE_OPENMP_USING_LIBOMP_LIBRARY AND OMNITRACE_USE_OMPT)
    set(_OMPT_PASS_REGEX "\\|_ompt_")
else()
    set(_OMPT_PASS_REGEX "")
endif()

omnitrace_add_test(
    SKIP_SAMPLING
    NAME openmp-cg
    TARGET openmp-cg
    LABELS "openmp"
    REWRITE_ARGS -e -v 2 --instrument-loops
    RUNTIME_ARGS -e -v 1 --label return args
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    ENVIRONMENT "${_ompt_environment};OMNITRACE_USE_SAMPLING=OFF;OMNITRACE_COUT_OUTPUT=ON"
    REWRITE_RUN_PASS_REGEX "${_OMPT_PASS_REGEX}"
    RUNTIME_PASS_REGEX "${_OMPT_PASS_REGEX}"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure")

omnitrace_add_test(
    SKIP_RUNTIME
    NAME openmp-lu
    TARGET openmp-lu
    LABELS "openmp"
    REWRITE_ARGS -e -v 2 --instrument-loops
    RUNTIME_ARGS -e -v 1 --label return args -E ^GOMP
    REWRITE_TIMEOUT 180
    RUNTIME_TIMEOUT 360
    ENVIRONMENT
        "${_ompt_environment};OMNITRACE_USE_SAMPLING=ON;OMNITRACE_SAMPLING_FREQ=50;OMNITRACE_COUT_OUTPUT=ON"
    REWRITE_RUN_PASS_REGEX "${_OMPT_PASS_REGEX}"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure")

set(_ompt_preload_environ
    "${_ompt_environment}"
    "OMNITRACE_VERBOSE=2"
    "OMNITRACE_USE_OMPT=OFF"
    "OMNITRACE_USE_SAMPLING=ON"
    "OMNITRACE_USE_PROCESS_SAMPLING=OFF"
    "OMNITRACE_SAMPLING_FREQ=100"
    "OMNITRACE_SAMPLING_DELAY=0.1"
    "OMNITRACE_SAMPLING_DURATION=0.25"
    "OMNITRACE_SAMPLING_CPUTIME=ON"
    "OMNITRACE_SAMPLING_REALTIME=ON"
    "OMNITRACE_SAMPLING_CPUTIME_FREQ=1000"
    "OMNITRACE_SAMPLING_REALTIME_FREQ=500"
    "OMNITRACE_MONOCHROME=ON")

set(_ompt_sample_no_tmpfiles_environ
    "${_ompt_environment}"
    "OMNITRACE_VERBOSE=2"
    "OMNITRACE_USE_OMPT=OFF"
    "OMNITRACE_USE_SAMPLING=ON"
    "OMNITRACE_USE_PROCESS_SAMPLING=OFF"
    "OMNITRACE_SAMPLING_CPUTIME=ON"
    "OMNITRACE_SAMPLING_REALTIME=OFF"
    "OMNITRACE_SAMPLING_CPUTIME_FREQ=700"
    "OMNITRACE_USE_TEMPORARY_FILES=OFF"
    "OMNITRACE_MONOCHROME=ON")

set(_ompt_preload_samp_regex
    "Sampler for thread 0 will be triggered 1000.0x per second of CPU-time(.*)Sampler for thread 0 will be triggered 500.0x per second of wall-time(.*)Sampling will be disabled after 0.250000 seconds(.*)Sampling duration of 0.250000 seconds has elapsed. Shutting down sampling"
    )
set(_ompt_preload_file_regex
    "sampling-duration-preload/sampling_percent.(json|txt)(.*)sampling-duration-preload/sampling_cpu_clock.(json|txt)(.*)sampling-duration-preload/sampling_wall_clock.(json|txt)"
    )
set(_notmp_preload_file_regex
    "sampling-no-tmp-files-preload/sampling_percent.(json|txt)(.*)sampling-no-tmp-files-preload/sampling_cpu_clock.(json|txt)(.*)sampling-no-tmp-files-preload/sampling_wall_clock.(json|txt)"
    )

omnitrace_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE SKIP_SAMPLING
    NAME openmp-cg-sampling-duration
    TARGET openmp-cg
    LABELS "openmp;sampling-duration"
    ENVIRONMENT "${_ompt_preload_environ}"
    PRELOAD_PASS_REGEX "${_ompt_preload_samp_regex}(.*)${_ompt_preload_file_regex}")

omnitrace_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE SKIP_SAMPLING
    NAME openmp-lu-sampling-duration
    TARGET openmp-lu
    LABELS "openmp;sampling-duration"
    ENVIRONMENT "${_ompt_preload_environ}"
    PRELOAD_PASS_REGEX "${_ompt_preload_samp_regex}(.*)${_ompt_preload_file_regex}")

omnitrace_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_REWRITE SKIP_SAMPLING
    NAME openmp-cg-sampling-no-tmp-files
    TARGET openmp-cg
    LABELS "openmp;no-tmp-files"
    ENVIRONMENT "${_ompt_sample_no_tmpfiles_environ}"
    PRELOAD_PASS_REGEX "${_notmp_preload_file_regex}")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING SKIP_PRELOAD
    NAME code-coverage
    TARGET code-coverage
    REWRITE_ARGS
        -e
        -v
        2
        --min-instructions=4
        -E
        ^std::
        -M
        coverage
        --coverage
        function
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        -M
        coverage
        --coverage
        function
        --module-restrict
        code.coverage
    LABELS "coverage;function-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING SKIP_PRELOAD
    NAME code-coverage-hybrid
    TARGET code-coverage
    REWRITE_ARGS -e -v 2 --min-instructions=4 -E ^std:: --coverage function
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        --coverage
        function
        --module-restrict
        code.coverage
    LABELS "coverage;function-coverage;hybrid-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) code coverage     ::  66.67%")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING SKIP_PRELOAD
    NAME code-coverage-basic-blocks
    TARGET code-coverage
    REWRITE_ARGS
        -e
        -v
        2
        --min-instructions=4
        -E
        ^std::
        -M
        coverage
        --coverage
        basic_block
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        -M
        coverage
        --coverage
        basic_block
        --module-restrict
        code.coverage
    LABELS "coverage;bb-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%")

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING SKIP_PRELOAD
    NAME code-coverage-basic-blocks-hybrid
    TARGET code-coverage
    REWRITE_ARGS -e -v 2 --min-instructions=4 -E ^std:: --coverage basic_block
    RUNTIME_ARGS
        -e
        -v
        1
        --min-instructions=4
        -E
        ^std::
        --label
        file
        line
        return
        args
        --coverage
        basic_block
        --module-restrict
        code.coverage
    LABELS "coverage;bb-coverage;hybrid-coverage"
    RUN_ARGS 10 ${NUM_THREADS} 1000
    ENVIRONMENT "${_base_environment}"
    RUNTIME_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%"
    REWRITE_RUN_PASS_REGEX "(\\\[[0-9]+\\\]) function coverage ::  66.67%")

if(_OS_RELEASE STREQUAL "ubuntu-18.04")
    set(_TRACE_WINDOW_SKIP SKIP_RUNTIME)
endif()

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING SKIP_PRELOAD ${_TRACE_WINDOW_SKIP}
    NAME trace-time-window
    TARGET trace-time-window
    REWRITE_ARGS -e -v 2 --caller-include inner -i 4096
    RUNTIME_ARGS -e -v 1 --caller-include inner -i 4096
    LABELS "time-window"
    ENVIRONMENT "${_window_environment};OMNITRACE_TRACE_DURATION=1.25")

omnitrace_add_validation_test(
    NAME trace-time-window-binary-rewrite
    TIMEMORY_METRIC "wall_clock"
    TIMEMORY_FILE "wall_clock.json"
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "time-window"
    FAIL_REGEX "outer_d"
    ARGS -l
         main
         outer_a
         outer_b
         outer_c
         -c
         1
         1
         1
         1
         -d
         0
         1
         1
         1
         -p)

omnitrace_add_validation_test(
    NAME trace-time-window-runtime-instrument
    TIMEMORY_METRIC "wall_clock"
    TIMEMORY_FILE "wall_clock.json"
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "time-window"
    FAIL_REGEX "outer_d"
    ARGS -l
         main
         outer_a
         outer_b
         outer_c
         -c
         1
         1
         1
         1
         -d
         0
         1
         1
         1
         -p)

omnitrace_add_test(
    SKIP_BASELINE SKIP_SAMPLING SKIP_PRELOAD ${_TRACE_WINDOW_SKIP}
    NAME trace-time-window-delay
    TARGET trace-time-window
    REWRITE_ARGS -e -v 2 --caller-include inner -i 4096
    RUNTIME_ARGS -e -v 1 --caller-include inner -i 4096
    LABELS "time-window"
    ENVIRONMENT
        "${_window_environment};OMNITRACE_TRACE_DELAY=0.75;OMNITRACE_TRACE_DURATION=0.75")

omnitrace_add_validation_test(
    NAME trace-time-window-delay-binary-rewrite
    TIMEMORY_METRIC "wall_clock"
    TIMEMORY_FILE "wall_clock.json"
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "time-window"
    ARGS -l
         outer_c
         outer_d
         -c
         1
         1
         -d
         0
         0
         -p)

omnitrace_add_validation_test(
    NAME trace-time-window-delay-runtime-instrument
    TIMEMORY_METRIC "wall_clock"
    TIMEMORY_FILE "wall_clock.json"
    PERFETTO_METRIC "host"
    PERFETTO_FILE "perfetto-trace.proto"
    LABELS "time-window"
    ARGS -l
         outer_c
         outer_d
         -c
         1
         1
         -d
         0
         0
         -p)

omnitrace_add_test(
    NAME fork
    TARGET fork-example
    REWRITE_ARGS -e -v 2 --print-instrumented modules -i 16
    RUNTIME_ARGS -e -v 1 --label file -i 16
    ENVIRONMENT "${_base_environment};OMNITRACE_CRITICAL_TRACE=ON")

# -------------------------------------------------------------------------------------- #
#
# critical-trace tests
#
# -------------------------------------------------------------------------------------- #

omnitrace_add_test(
    SKIP_BASELINE SKIP_RUNTIME SKIP_SAMPLING SKIP_PRELOAD
    NAME parallel-overhead-critical-trace
    TARGET parallel-overhead
    LABELS "critical-trace"
    REWRITE_ARGS
        -e
        -i
        8
        -E
        "^fib"
        -v
        2
        --print-instrumented
        functions
    RUN_ARGS 10 4 100
    ENVIRONMENT "${_critical_trace_environment}")

add_test(
    NAME parallel-overhead-process-critical-trace
    COMMAND
        $<TARGET_FILE:omnitrace-critical-trace>
        ${PROJECT_BINARY_DIR}/omnitrace-tests-output/parallel-overhead-critical-trace-binary-rewrite/call-chain.json
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

set(_parallel_overhead_critical_trace_environ
    "OMNITRACE_OUTPUT_PATH=omnitrace-tests-output"
    "OMNITRACE_OUTPUT_PREFIX=parallel-overhead-critical-trace/"
    "OMNITRACE_CRITICAL_TRACE_DEBUG=ON"
    "OMNITRACE_VERBOSE=4"
    "OMNITRACE_USE_PID=OFF"
    "OMNITRACE_TIME_OUTPUT=OFF")

set_tests_properties(
    parallel-overhead-process-critical-trace
    PROPERTIES
        ENVIRONMENT
        "${_parallel_overhead_critical_trace_environ}"
        TIMEOUT
        300
        LABELS
        "parallel-overhead;critical-trace"
        PASS_REGULAR_EXPRESSION
        "Outputting.*(critical-trace-cpu.json).*Outputting.*(critical-trace-any.json)"
        DEPENDS
        parallel-overhead-critical-trace-binary-rewrite-run)

# -------------------------------------------------------------------------------------- #
#
# attach tests
#
# -------------------------------------------------------------------------------------- #

set(_VALID_PTRACE_SCOPE OFF)
if(EXISTS "/proc/sys/kernel/yama/ptrace_scope")
    file(READ "/proc/sys/kernel/yama/ptrace_scope" _PTRACE_SCOPE LIMIT 1)
    if("${_PTRACE_SCOPE}" EQUAL 0)
        set(_VALID_PTRACE_SCOPE ON)
    endif()
else()
    omnitrace_message(
        AUTHOR_WARNING
        "Disabling attach tests. Run 'echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope' to enable attaching to process"
        )
endif()

if(TARGET parallel-overhead AND _VALID_PTRACE_SCOPE)
    add_test(
        NAME parallel-overhead-attach
        COMMAND
            ${CMAKE_CURRENT_LIST_DIR}/run-omnitrace-pid.sh $<TARGET_FILE:omnitrace-exe>
            -ME "\.c$" -E fib -e -v 1 --label return args file -l --
            $<TARGET_FILE:parallel-overhead> 30 8 1000
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})

    set(_parallel_overhead_attach_environ
        "${_attach_environment}" "OMNITRACE_OUTPUT_PATH=omnitrace-tests-output"
        "OMNITRACE_OUTPUT_PREFIX=parallel-overhead-attach/")

    set_tests_properties(
        parallel-overhead-attach
        PROPERTIES ENVIRONMENT
                   "${_parallel_overhead_attach_environ}"
                   TIMEOUT
                   300
                   LABELS
                   "parallel-overhead;attach"
                   PASS_REGULAR_EXPRESSION
                   "Outputting.*(perfetto-trace.proto).*Outputting.*(wall_clock.txt)"
                   FAIL_REGULAR_EXPRESSION
                   "Dyninst was unable to attach to the specified process")
endif()

# -------------------------------------------------------------------------------------- #
#
# rccl tests
#
# -------------------------------------------------------------------------------------- #

foreach(_TARGET ${RCCL_TEST_TARGETS})
    string(REPLACE "rccl-tests::" "" _NAME "${_TARGET}")
    string(REPLACE "_" "-" _NAME "${_NAME}")
    omnitrace_add_test(
        SKIP_SAMPLING
        NAME rccl-test-${_NAME}
        TARGET ${_TARGET}
        LABELS "rccl-tests;rcclp"
        MPI ON
        GPU ON
        NUM_PROCS 1
        REWRITE_ARGS
            -e
            -v
            2
            -i
            8
            --label
            file
            line
            return
            args
        RUNTIME_ARGS
            -e
            -v
            1
            -i
            8
            --label
            file
            line
            return
            args
            -ME
            sysdeps
            --log-file
            rccl-test-${_NAME}.log
        RUN_ARGS -t
                 1
                 -g
                 1
                 -i
                 10
                 -w
                 2
                 -m
                 2
                 -p
                 -c
                 1
                 -z
                 -s
                 1
        ENVIRONMENT "${_rccl_environment}")
endforeach()

# -------------------------------------------------------------------------------------- #
#
# causal profiling tests
#
# -------------------------------------------------------------------------------------- #

omnitrace_add_causal_test(
    NAME cpu-omni-func
    TARGET causal-cpu-omni
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "function"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

omnitrace_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-func-ndebug
    TARGET causal-cpu-omni-ndebug
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "function"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

omnitrace_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-line
    TARGET causal-cpu-omni
    RUN_ARGS 70 10 432525 1000000000
    CAUSAL_MODE "line"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

omnitrace_add_causal_test(
    NAME both-omni-func
    TARGET causal-both-omni
    RUN_ARGS 70 10 432525 400000000
    CAUSAL_MODE "function"
    CAUSAL_ARGS
        -w
        1
        -d
        3
        --monochrome
        -g
        ${CMAKE_BINARY_DIR}/omnitrace-tests-config/causal-both-omni-func
        -l
        causal-both-omni
        -v
        3
    ENVIRONMENT "OMNITRACE_STRICT_CONFIG=OFF"
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

omnitrace_add_causal_test(
    NAME lulesh-func
    TARGET lulesh-omni
    RUN_ARGS -i 35 -s 50 -p
    CAUSAL_MODE "function"
    CAUSAL_ARGS -s 0,10,25,50,75
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

omnitrace_add_causal_test(
    SKIP_BASELINE
    NAME lulesh-func-ndebug
    TARGET lulesh-omni-ndebug
    RUN_ARGS -i 35 -s 50 -p
    CAUSAL_MODE "function"
    CAUSAL_ARGS -s 0,10,25,50,75
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

omnitrace_add_causal_test(
    SKIP_BASELINE
    NAME lulesh-line
    TARGET lulesh-omni
    RUN_ARGS -i 35 -s 50 -p
    CAUSAL_MODE "line"
    CAUSAL_ARGS -s 0,10,25,50,75 -S lulesh.cc
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

set(_causal_common_args
    "-n 12 -e -s 0 10 20 30 -B $<TARGET_FILE_BASE_NAME:causal-cpu-omni>")

macro(
    causal_e2e_args_and_validation
    _NAME
    _TEST
    _MODE
    _EXPER
    _V10 # expected value for virtual speedup of 10
    _V20
    _V30
    _TOL # tolerance for virtual speedup
    )
    # arguments to omnitrace-causal
    set(${_NAME}_args "${_causal_common_args} ${_MODE} ${_EXPER}")

    # arguments to validate-causal-json.py
    set(${_NAME}_valid
        "-n 0 -i omnitrace-tests-output/causal-cpu-omni-${_TEST}-e2e/causal/experiments.json -v ${_EXPER} $<TARGET_FILE_BASE_NAME:causal-cpu-omni> 10 ${_V10} ${_TOL} ${_EXPER} $<TARGET_FILE_BASE_NAME:causal-cpu-omni> 20 ${_V20} ${_TOL} ${_EXPER} $<TARGET_FILE_BASE_NAME:causal-cpu-omni> 30 ${_V30} ${_TOL}"
        )
    # patch string for command-line
    string(REPLACE " " ";" ${_NAME}_args "${${_NAME}_args}")
    string(REPLACE " " ";" ${_NAME}_valid "${${_NAME}_valid}")
endmacro()

causal_e2e_args_and_validation(_causal_slow_func slow-func "-F" "cpu_slow_func" 10 20 20
                               5)
causal_e2e_args_and_validation(_causal_fast_func fast-func "-F" "cpu_fast_func" 0 0 0 5)
causal_e2e_args_and_validation(_causal_line_155 line-155 "-S" "causal.cpp:155" 10 20 20 5)
causal_e2e_args_and_validation(_causal_line_165 line-165 "-S" "causal.cpp:165" 0 0 0 5)

omnitrace_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-slow-func-e2e
    TARGET causal-cpu-omni
    RUN_ARGS 80 12 432525 250000000
    CAUSAL_MODE "func"
    CAUSAL_ARGS ${_causal_slow_func_args}
    CAUSAL_VALIDATE_ARGS ${_causal_slow_func_valid}
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

omnitrace_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-fast-func-e2e
    TARGET causal-cpu-omni
    RUN_ARGS 80 12 432525 250000000
    CAUSAL_MODE "func"
    CAUSAL_ARGS ${_causal_fast_func_args}
    CAUSAL_VALIDATE_ARGS ${_causal_fast_func_valid}
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

omnitrace_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-line-155-e2e
    TARGET causal-cpu-omni
    RUN_ARGS 80 12 432525 250000000
    CAUSAL_MODE "line"
    CAUSAL_ARGS ${_causal_line_155_args}
    CAUSAL_VALIDATE_ARGS ${_causal_line_155_valid}
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

omnitrace_add_causal_test(
    SKIP_BASELINE
    NAME cpu-omni-line-165-e2e
    TARGET causal-cpu-omni
    RUN_ARGS 80 12 432525 250000000
    CAUSAL_MODE "line"
    CAUSAL_ARGS ${_causal_line_165_args}
    CAUSAL_VALIDATE_ARGS ${_causal_line_165_valid}
    CAUSAL_PASS_REGEX
        "Starting causal experiment #1(.*)causal/experiments.json(.*)causal/experiments.coz"
    )

# -------------------------------------------------------------------------------------- #
#
# python tests
#
# -------------------------------------------------------------------------------------- #

set(_INDEX 0)
foreach(_VERSION ${OMNITRACE_PYTHON_VERSIONS})
    if(NOT OMNITRACE_USE_PYTHON)
        continue()
    endif()

    list(GET OMNITRACE_PYTHON_ROOT_DIRS ${_INDEX} _PYTHON_ROOT_DIR)

    omnitrace_find_python(
        _PYTHON
        ROOT_DIR "${_PYTHON_ROOT_DIR}"
        COMPONENTS Interpreter)

    # ---------------------------------------------------------------------------------- #
    # python tests
    # ---------------------------------------------------------------------------------- #
    omnitrace_add_python_test(
        NAME python-external
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/external.py
        PROFILE_ARGS "--label" "file"
        RUN_ARGS -v 10 -n 5
        ENVIRONMENT "${_python_environment}")

    omnitrace_add_python_test(
        NAME python-external-exclude-inefficient
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/external.py
        PROFILE_ARGS -E "^inefficient$"
        RUN_ARGS -v 10 -n 5
        ENVIRONMENT "${_python_environment}")

    omnitrace_add_python_test(
        NAME python-builtin
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/builtin.py
        PROFILE_ARGS "-b" "--label" "file" "line"
        RUN_ARGS -v 10 -n 5
        ENVIRONMENT "${_python_environment}")

    omnitrace_add_python_test(
        NAME python-builtin-noprofile
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/noprofile.py
        PROFILE_ARGS "-b" "--label" "file"
        RUN_ARGS -v 15 -n 5
        ENVIRONMENT "${_python_environment}")

    omnitrace_add_python_test(
        STANDALONE
        NAME python-source
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/python/source.py
        RUN_ARGS -v 5 -n 5 -s 3
        ENVIRONMENT "${_python_environment}")

    omnitrace_add_python_test(
        STANDALONE
        NAME python-code-coverage
        PYTHON_EXECUTABLE ${_PYTHON_EXECUTABLE}
        PYTHON_VERSION ${_VERSION}
        FILE ${CMAKE_SOURCE_DIR}/examples/code-coverage/code-coverage.py
        RUN_ARGS
            -i
            ${PROJECT_BINARY_DIR}/omnitrace-tests-output/code-coverage-basic-blocks-binary-rewrite/coverage.json
            ${PROJECT_BINARY_DIR}/omnitrace-tests-output/code-coverage-basic-blocks-hybrid-runtime-instrument/coverage.json
            -o
            ${PROJECT_BINARY_DIR}/omnitrace-tests-output/code-coverage-basic-blocks-summary/coverage.json
        DEPENDS code-coverage-basic-blocks-binary-rewrite
                code-coverage-basic-blocks-binary-rewrite-run
                code-coverage-basic-blocks-hybrid-runtime-instrument
        LABELS "code-coverage"
        ENVIRONMENT "${_python_environment}")

    # ---------------------------------------------------------------------------------- #
    # python output tests
    # ---------------------------------------------------------------------------------- #

    if(CMAKE_VERSION VERSION_LESS "3.18.0")
        find_program(
            OMNITRACE_CAT_EXE
            NAMES cat
            PATH_SUFFIXES bin)
        if(OMNITRACE_CAT_EXE)
            set(OMNITRACE_CAT_COMMAND ${OMNITRACE_CAT_EXE})
        endif()
    else()
        set(OMNITRACE_CAT_COMMAND ${CMAKE_COMMAND} -E cat)
    endif()

    if(OMNITRACE_CAT_COMMAND)
        omnitrace_add_python_test(
            NAME python-external-check
            COMMAND ${OMNITRACE_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/python-external/${_VERSION}/trip_count.txt
            PASS_REGEX
                "(\\\[compile\\\]).*(\\\| \\\|0>>> \\\[run\\\]\\\[external.py\\\]).*(\\\| \\\|0>>> \\\|_\\\[fib\\\]\\\[external.py\\\]).*(\\\| \\\|0>>> \\\|_\\\[inefficient\\\]\\\[external.py\\\])"
            DEPENDS python-external-${_VERSION}
            ENVIRONMENT "${_python_environment}")

        omnitrace_add_python_test(
            NAME python-external-exclude-inefficient-check
            COMMAND ${OMNITRACE_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/python-external-exclude-inefficient/${_VERSION}/trip_count.txt
            FAIL_REGEX "(\\\|_inefficient).*(\\\|_sum)"
            DEPENDS python-external-exclude-inefficient-${_VERSION}
            ENVIRONMENT "${_python_environment}")

        omnitrace_add_python_test(
            NAME python-builtin-check
            COMMAND ${OMNITRACE_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/python-builtin/${_VERSION}/trip_count.txt
            PASS_REGEX "\\\[inefficient\\\]\\\[builtin.py:14\\\]"
            DEPENDS python-builtin-${_VERSION}
            ENVIRONMENT "${_python_environment}")

        omnitrace_add_python_test(
            NAME python-builtin-noprofile-check
            COMMAND ${OMNITRACE_CAT_COMMAND}
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/python-builtin-noprofile/${_VERSION}/trip_count.txt
            PASS_REGEX ".(run)..(noprofile.py)."
            FAIL_REGEX ".(fib|inefficient)..(noprofile.py)."
            DEPENDS python-builtin-noprofile-${_VERSION}
            ENVIRONMENT "${_python_environment}")
    else()
        omnitrace_message(
            WARNING
            "Neither 'cat' nor 'cmake -E cat' are available. Python source checks are disabled"
            )
    endif()

    function(OMNITRACE_ADD_PYTHON_VALIDATION_TEST)
        cmake_parse_arguments(
            TEST "" "NAME;TIMEMORY_METRIC;TIMEMORY_FILE;PERFETTO_METRIC;PERFETTO_FILE"
            "ARGS" ${ARGN})

        omnitrace_add_python_test(
            NAME ${TEST_NAME}-validate-timemory
            COMMAND
                ${_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/validate-timemory-json.py
                -m ${TEST_TIMEMORY_METRIC} ${TEST_ARGS} -i
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/${TEST_NAME}/${_VERSION}/${TEST_TIMEMORY_FILE}
            DEPENDS ${TEST_NAME}-${_VERSION}
            PASS_REGEX
                "omnitrace-tests-output/${TEST_NAME}/${_VERSION}/${TEST_TIMEMORY_FILE} validated"
            ENVIRONMENT "${_python_environment}")

        omnitrace_add_python_test(
            NAME ${TEST_NAME}-validate-perfetto
            COMMAND
                ${_PYTHON_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/validate-perfetto-proto.py
                -m ${TEST_PERFETTO_METRIC} ${TEST_ARGS} -p -i
            PYTHON_VERSION ${_VERSION}
            FILE omnitrace-tests-output/${TEST_NAME}/${_VERSION}/${TEST_PERFETTO_FILE}
            DEPENDS ${TEST_NAME}-${_VERSION}
            PASS_REGEX
                "omnitrace-tests-output/${TEST_NAME}/${_VERSION}/${TEST_PERFETTO_FILE} validated"
            ENVIRONMENT "${_python_environment}")
    endfunction()

    set(python_source_labels
        main_loop
        run
        fib
        fib
        fib
        fib
        fib
        inefficient
        _sum)
    set(python_source_count
        5
        3
        3
        6
        12
        18
        6
        3
        3)
    set(python_source_depth
        0
        1
        2
        3
        4
        5
        6
        2
        3)

    omnitrace_add_python_validation_test(
        NAME python-source
        TIMEMORY_METRIC "trip_count"
        TIMEMORY_FILE "trip_count.json"
        PERFETTO_METRIC "host;user"
        PERFETTO_FILE "perfetto-trace.proto"
        ARGS -l ${python_source_labels} -c ${python_source_count} -d
             ${python_source_depth})

    set(python_builtin_labels
        [run][builtin.py:28]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [fib][builtin.py:10]
        [inefficient][builtin.py:14])
    set(python_builtin_count
        5
        5
        10
        20
        40
        80
        160
        260
        220
        80
        10
        5)
    set(python_builtin_depth
        0
        1
        2
        3
        4
        5
        6
        7
        8
        9
        10
        1)

    omnitrace_add_python_validation_test(
        NAME python-builtin
        TIMEMORY_METRIC "trip_count"
        TIMEMORY_FILE "trip_count.json"
        PERFETTO_METRIC "host;user"
        PERFETTO_FILE "perfetto-trace.proto"
        ARGS -l ${python_builtin_labels} -c ${python_builtin_count} -d
             ${python_builtin_depth})
    math(EXPR _INDEX "${_INDEX} + 1")
endforeach()
