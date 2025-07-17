# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
        "${_base_environment};ROCPROFSYS_USE_KOKKOSP=ON;ROCPROFSYS_COUT_OUTPUT=ON;ROCPROFSYS_SAMPLING_FREQ=50;ROCPROFSYS_KOKKOSP_PREFIX=[kokkos];KOKKOS_TOOLS_LIBS=librocprof-sys-dl.so"
    REWRITE_RUN_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]"
    RUNTIME_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]"
)

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
        "${_base_environment};ROCPROFSYS_USE_KOKKOSP=ON;ROCPROFSYS_COUT_OUTPUT=ON;ROCPROFSYS_SAMPLING_FREQ=50;ROCPROFSYS_KOKKOSP_PREFIX=[kokkos];KOKKOS_TOOLS_LIBS=librocprof-sys.so"
    BASELINE_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]"
)

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
        "${_base_environment};ROCPROFSYS_USE_KOKKOSP=ON;ROCPROFSYS_COUT_OUTPUT=ON;ROCPROFSYS_SAMPLING_FREQ=50;ROCPROFSYS_KOKKOSP_PREFIX=[kokkos];KOKKOS_TOOLS_LIBS=librocprof-sys-dl.so"
    BASELINE_PASS_REGEX "\\|_\\[kokkos\\] [a-zA-Z]"
)

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
    ENVIRONMENT "${_base_environment};ROCPROFSYS_USE_KOKKOSP=ON"
)

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
    ENVIRONMENT "${_perfetto_environment};ROCPROFSYS_USE_KOKKOSP=OFF"
)

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
    REWRITE_FAIL_REGEX "0 instrumented loops in procedure"
)
