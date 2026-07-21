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
add_subdirectory(hipfile)
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
