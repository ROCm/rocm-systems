#
# Integration test applications
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-bin LANGUAGES C CXX)

# The roctx shared-library target is not always available when configuring the tests
# standalone against an installed SDK (e.g. TheRock package builds), so only add its
# directory to the build RPATH when the target exists.
if(TARGET rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library)
    set(CMAKE_BUILD_RPATH
        "\$ORIGIN:\$ORIGIN/../lib:$<TARGET_FILE_DIR:rocprofiler-sdk-roctx::rocprofiler-sdk-roctx-shared-library>"
        )
else()
    set(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/../lib")
endif()

# Find rocdecode and rocjpeg packages for testing
find_package(rocdecode)
find_package(rocjpeg)

# ==================== TEMP ASAN DIAGNOSTIC — remove before merge ====================
# Probes where the "ASan runtime does not come first" failure originates on RHEL/SLES.
# All lines are prefixed with ASAN-PROBE so they can be grepped from the CI configure log.
message(STATUS "ASAN-PROBE ROCM_PATH               = ${ROCM_PATH}")
message(STATUS "ASAN-PROBE CMAKE_CXX_COMPILER      = ${CMAKE_CXX_COMPILER}")
message(STATUS "ASAN-PROBE CMAKE_CXX_FLAGS         = ${CMAKE_CXX_FLAGS}")
message(STATUS "ASAN-PROBE CMAKE_EXE_LINKER_FLAGS  = ${CMAKE_EXE_LINKER_FLAGS}")
message(STATUS "ASAN-PROBE ROCPROFILER_MEMCHECK    = ${ROCPROFILER_MEMCHECK}")
message(STATUS "ASAN-PROBE ENV{ASAN_OPTIONS}       = $ENV{ASAN_OPTIONS}")
message(STATUS "ASAN-PROBE ENV{LD_PRELOAD}         = $ENV{LD_PRELOAD}")
message(STATUS "ASAN-PROBE ENV{LD_LIBRARY_PATH}    = $ENV{LD_LIBRARY_PATH}")
message(STATUS "ASAN-PROBE rocdecode_VERSION       = ${rocdecode_VERSION}")
message(STATUS "ASAN-PROBE rocdecode_LIBRARY       = ${rocdecode_LIBRARY}")
message(STATUS "ASAN-PROBE rocjpeg_VERSION         = ${rocjpeg_VERSION}")
message(STATUS "ASAN-PROBE rocjpeg_LIBRARY         = ${rocjpeg_LIBRARY}")

# Inspect the dynamic section (NEEDED + RPATH/RUNPATH) and exported symbols of each
# candidate library to see whether it is asan-instrumented or resolves an asan runtime.
set(_asan_probe_libs
    "${rocdecode_LIBRARY}"
    "${rocjpeg_LIBRARY}"
    "${ROCM_PATH}/lib/librocdecode.so"
    "${ROCM_PATH}/lib/librocjpeg.so"
    "${ROCM_PATH}/lib/libamdhip64.so"
    "${ROCM_PATH}/lib/libhsa-runtime64.so"
    "${ROCM_PATH}/lib/libamdocl64.so")
find_program(_asan_probe_readelf NAMES readelf)
find_program(_asan_probe_nm NAMES nm)
message(STATUS "ASAN-PROBE readelf                 = ${_asan_probe_readelf}")
foreach(_lib ${_asan_probe_libs})
    if(_lib AND EXISTS "${_lib}")
        set(_dyn "")
        if(_asan_probe_readelf)
            execute_process(COMMAND ${_asan_probe_readelf} -d "${_lib}"
                            OUTPUT_VARIABLE _dyn ERROR_QUIET)
        endif()
        set(_syms "")
        if(_asan_probe_nm)
            execute_process(COMMAND ${_asan_probe_nm} -D "${_lib}"
                            OUTPUT_VARIABLE _syms ERROR_QUIET)
        endif()
        set(_verdict "clean")
        if(_dyn MATCHES "asan" OR _syms MATCHES "__asan")
            set(_verdict "ASAN-INSTRUMENTED")
        endif()
        message(STATUS "ASAN-PROBE lib ${_verdict}: ${_lib}")
        string(REGEX MATCHALL "[^\n]*R[UN]*PATH[^\n]*" _rpaths "${_dyn}")
        foreach(_rp ${_rpaths})
            message(STATUS "ASAN-PROBE   rpath: ${_rp}")
        endforeach()
        string(REGEX MATCHALL "[^\n]*asan[^\n]*" _asanlines "${_dyn}")
        foreach(_al ${_asanlines})
            message(STATUS "ASAN-PROBE   dyn-asan: ${_al}")
        endforeach()
    else()
        message(STATUS "ASAN-PROBE lib MISSING: ${_lib}")
    endif()
endforeach()

# Probe #2: resolve the asan runtime the way the eventual fix will, to confirm the
# glob finds a real file in the CI image (RUNPATH referenced clang/23/lib/linux).
set(_asan_rt_glob "${ROCM_PATH}/lib/llvm/lib/clang/*/lib/linux/libclang_rt.asan-x86_64.so")
file(GLOB _asan_rt_candidates "${_asan_rt_glob}")
message(STATUS "ASAN-PROBE asan_rt_glob            = ${_asan_rt_glob}")
message(STATUS "ASAN-PROBE asan_rt_candidates      = ${_asan_rt_candidates}")
if(_asan_rt_candidates)
    list(SORT _asan_rt_candidates ORDER DESCENDING)
    list(GET _asan_rt_candidates 0 _asan_rt_resolved)
    message(STATUS "ASAN-PROBE asan_rt RESOLVED        = ${_asan_rt_resolved}")
    if(EXISTS "${_asan_rt_resolved}")
        message(STATUS "ASAN-PROBE asan_rt EXISTS          = YES")
    else()
        message(STATUS "ASAN-PROBE asan_rt EXISTS          = NO")
    endif()
else()
    message(STATUS "ASAN-PROBE asan_rt RESOLVED        = <none found by glob>")
    # fallback: list what actually lives under the clang dir
    file(GLOB _clang_dirs "${ROCM_PATH}/lib/llvm/lib/clang/*")
    message(STATUS "ASAN-PROBE clang dirs             = ${_clang_dirs}")
    file(GLOB_RECURSE _any_asan "${ROCM_PATH}/lib/llvm/lib/clang/*/libclang_rt.asan*")
    message(STATUS "ASAN-PROBE any asan under clang   = ${_any_asan}")
endif()

# ================== END TEMP ASAN DIAGNOSTIC ==================

# rocSHMEM's installed config references both MPI::MPI_CXX and
# rocprofiler-register::rocprofiler-register in the roc::rocshmem link interface, so both
# must be located first or the find_package call will fail to define the imported target.
find_package(MPI COMPONENTS CXX)
find_package(rocprofiler-register CONFIG PATHS ${ROCM_PATH} /opt/rocm)
if(MPI_CXX_FOUND AND rocprofiler-register_FOUND)
    find_package(rocshmem CONFIG PATHS ${ROCM_PATH} /opt/rocm)
endif()

# applications used by integration tests which DO link to rocprofiler-sdk-roctx
add_subdirectory(reproducible-runtime)
add_subdirectory(reproducible-dispatch-count)
add_subdirectory(transpose)
add_subdirectory(openmp)

set(CMAKE_BUILD_RPATH "\$ORIGIN:\$ORIGIN/../lib")

# applications used by integration tests which DO NOT link to rocprofiler-sdk-roctx
add_subdirectory(simple-transpose)
add_subdirectory(multistream)
add_subdirectory(vector-operations)
add_subdirectory(hip-in-libraries)
add_subdirectory(scratch-memory)
add_subdirectory(hsa-queue-dependency)
add_subdirectory(hip-graph)
add_subdirectory(hip-graph-attribution)
add_subdirectory(roctx-pause-resume)
add_subdirectory(att-marker-trace)
add_subdirectory(hsa-memory-allocation)
add_subdirectory(pc-sampling)
if(rocdecode_FOUND AND rocdecode_VERSION VERSION_GREATER 0.8.0)
    add_subdirectory(rocdecode)
endif()
if(rocjpeg_FOUND AND rocjpeg_VERSION VERSION_GREATER 0.6.0)
    add_subdirectory(rocjpeg)
endif()
if(rocshmem_FOUND AND rocshmem_VERSION VERSION_GREATER_EQUAL 3.3.0)
    # rocshmem >= 3.3.0 introduced the full host-stream API set that rocshmem-demo
    # exercises (quiet/sync_all/alltoallmem/broadcastmem/getmem/putmem/putmem_signal/
    # signal_wait_until on_stream). Skip the subdir on older installs so the build does
    # not break.
    add_subdirectory(rocshmem)
endif()
add_subdirectory(hsa-code-object)
add_subdirectory(hip-streams)
add_subdirectory(hip-stream-reuse)
add_subdirectory(hip-streams-per-thread)
add_subdirectory(attachment-test)
add_subdirectory(hip-host)
add_subdirectory(module-loading-test)
add_subdirectory(late-start-tracing)
add_subdirectory(thread-trace)
add_subdirectory(hip-graph-bubbles)
add_subdirectory(rocpd-api-test)
