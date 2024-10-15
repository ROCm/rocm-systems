# -------------------------------------------------------------------------------------- #
#
# kokkos (lulesh) tests
#
# -------------------------------------------------------------------------------------- #

rocprofiler_systems_add_test(
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
        "${_base_environment};ROCPROFSYS_USE_KOKKOSP=ON;ROCPROFSYS_COUT_OUTPUT=ON;ROCPROFSYS_SAMPLING_FREQ=50;ROCPROFSYS_KOKKOSP_PREFIX=[kokkos];KOKKOS_PROFILE_LIBRARY=librocprof-sys-dl.so"
    REWRITE_RUN_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]"
    RUNTIME_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]")

rocprofiler_systems_add_test(
    SKIP_RUNTIME SKIP_REWRITE
    NAME lulesh-baseline-kokkosp-librocprofiler-systems
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    GPU ${LULESH_USE_GPU}
    NUM_PROCS 8
    LABELS "kokkos;kokkos-profile-library"
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_base_environment};ROCPROFSYS_USE_KOKKOSP=ON;ROCPROFSYS_COUT_OUTPUT=ON;ROCPROFSYS_SAMPLING_FREQ=50;ROCPROFSYS_KOKKOSP_PREFIX=[kokkos];KOKKOS_PROFILE_LIBRARY=librocprof-sys.so"
    BASELINE_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]")

rocprofiler_systems_add_test(
    SKIP_RUNTIME SKIP_REWRITE
    NAME lulesh-baseline-kokkosp-librocprofiler-systems-dl
    TARGET lulesh
    MPI ${LULESH_USE_MPI}
    GPU ${LULESH_USE_GPU}
    NUM_PROCS 8
    LABELS "kokkos;kokkos-profile-library"
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT
        "${_base_environment};ROCPROFSYS_USE_KOKKOSP=ON;ROCPROFSYS_COUT_OUTPUT=ON;ROCPROFSYS_SAMPLING_FREQ=50;ROCPROFSYS_KOKKOSP_PREFIX=[kokkos];KOKKOS_PROFILE_LIBRARY=librocprof-sys-dl.so"
    BASELINE_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]")

rocprofiler_systems_add_test(
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
    ENVIRONMENT "${_base_environment};ROCPROFSYS_USE_KOKKOSP=ON")

rocprofiler_systems_add_test(
    SKIP_BASELINE
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
    ENVIRONMENT "${_perfetto_environment};ROCPROFSYS_USE_KOKKOSP=OFF")

rocprofiler_systems_add_test(
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
        --env
        ROCPROFSYS_TIMEMORY_COMPONENTS="wall_clock peak_rss"
    RUN_ARGS -i 10 -s 20 -p
    ENVIRONMENT "${_timemory_environment};ROCPROFSYS_USE_KOKKOSP=OFF"
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure")
