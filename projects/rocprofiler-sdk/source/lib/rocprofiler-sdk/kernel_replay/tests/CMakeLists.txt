rocprofiler_deactivate_clang_tidy()

if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

# C is enabled for the ABI test's C translation unit: the public headers are consumed by C
# tools, and nothing else in the tree compiles them as C.
project(rocprofiler-sdk-kernel-replay-tests LANGUAGES C CXX HIP)

# Inherit the CXX per-config flags for HIP when unset.
foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

include(GoogleTest)

# The GPU kernels are the only HIP code in this test. They are built into their own static
# library so the test executable itself has no HIP sources and therefore links as CXX.
# That matters because ensure_live_tracking() -> rocprofiler_force_configure() runs a
# register scan that dlopens this statically-linked executable for
# rocprofiler_set_api_table; the SDK exports that symbol via the rocprofiler-sdk-dl
# interface's -rdynamic, which is gated on a C/CXX link language. A HIP-linked executable
# misses it (and would need a manual -rdynamic); a CXX-linked one inherits it.
set(KERNEL_REPLAY_SNAP_KERNEL_SOURCES snap_kernels.cpp)
set_source_files_properties(${KERNEL_REPLAY_SNAP_KERNEL_SOURCES} PROPERTIES LANGUAGE HIP)

add_library(kernel-replay-snapshot-test-kernels STATIC
            ${KERNEL_REPLAY_SNAP_KERNEL_SOURCES})
target_link_libraries(kernel-replay-snapshot-test-kernels
                      PRIVATE rocprofiler-sdk::rocprofiler-sdk-hip)

set(KERNEL_REPLAY_SNAP_RESTORE_SOURCES snap_restore.cpp snap_bandwidth.cpp)

add_executable(kernel-replay-snapshot-test)
target_sources(kernel-replay-snapshot-test PRIVATE ${KERNEL_REPLAY_SNAP_RESTORE_SOURCES})

target_link_libraries(
    kernel-replay-snapshot-test
    PRIVATE kernel-replay-snapshot-test-kernels
            rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-hip
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::counter-test-constants
            GTest::gtest
            GTest::gtest_main)

# Linking the HIP kernel library would otherwise make CMake pick the HIP linker for this
# executable, which drops it back into the HIP link language and misses the C/CXX
# -rdynamic described above. The kernel objects are non-rdc (each carries a
# self-contained, self-registering fatbin), so a plain CXX link resolves and registers the
# device code correctly. Force the CXX linker so the dl interface's
# $<LINK_LANGUAGE:CXX>:-rdynamic applies in every build configuration (not just Debug).
set_target_properties(kernel-replay-snapshot-test PROPERTIES LINKER_LANGUAGE CXX)

# RESOURCE_LOCK serializes these snapshot tests against each other: gtest_add_tests
# registers each TEST() as its own ctest process, and each allocates several device
# buffers plus matching host snapshot copies. Under `ctest --parallel`, running them
# concurrently stacked their GPU/host footprint and intermittently OOM'd/partial-copied
# the largest (multi-buffer) test on shared CI GPUs. The lock keeps them sequential
# without forcing the whole suite serial.
rocprofiler_add_unit_test(
    TARGET kernel-replay-snapshot-test
    SOURCES ${KERNEL_REPLAY_SNAP_RESTORE_SOURCES}
    TIMEOUT 240
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
        RESOURCE_LOCK
        "rocprofiler_kernel_replay"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# snap/restore bandwidth tests allocate up to 128 MB ballast plus host snapshot copies.
# (thread-local override state) with no GPU or rocprofiler runtime dependency, so it is a
# plain CXX unit test that runs unconditionally.
add_executable(kernel-replay-local-context-test)
target_sources(kernel-replay-local-context-test PRIVATE local_context.cpp)

target_link_libraries(
    kernel-replay-local-context-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET kernel-replay-local-context-test
    SOURCES local_context.cpp
    TIMEOUT 60
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Layout and C/C++ agreement checks for the public kernel-replay header. These are
# properties of the header alone: no GPU, no HSA, and no link against the SDK, so the test
# runs anywhere.
#
# The C source is half the point of the target. The public headers are consumed by C tools
# but nothing else in the tree compiles them as C, so a C++-only construct reaching one of
# them would otherwise go unnoticed until a downstream tool failed to build. Compiling
# this file is that check, and it also lets the test compare C's view of the record layout
# against C++'s.
set(KERNEL_REPLAY_ABI_SOURCES replay_abi.cpp replay_abi_c.c)

add_executable(kernel-replay-abi-test)
target_sources(kernel-replay-abi-test PRIVATE ${KERNEL_REPLAY_ABI_SOURCES})

set_source_files_properties(replay_abi_c.c PROPERTIES LANGUAGE C)

# Pin the C dialect rather than inheriting whatever the toolchain defaults to, so the
# check means the same thing everywhere. aqlprofile-core-tests mixes a C translation unit
# into a gtest executable the same way (source/lib/aqlprofile/core/tests,
# aql_profile_v2_c_test.c).
set_target_properties(kernel-replay-abi-test PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED
                                                                      ON)

target_link_libraries(
    kernel-replay-abi-test PRIVATE rocprofiler-sdk::rocprofiler-sdk-headers GTest::gtest
                                   GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET kernel-replay-abi-test
    SOURCES ${KERNEL_REPLAY_ABI_SOURCES}
    TIMEOUT 60
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Exercises only the configuration path via rocprofiler_force_configure (no GPU / HSA), so
# it is a plain CXX unit test that runs unconditionally.
add_executable(kernel-replay-configure-test)
target_sources(kernel-replay-configure-test PRIVATE replay_configure.cpp)

target_link_libraries(
    kernel-replay-configure-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET kernel-replay-configure-test
    SOURCES replay_configure.cpp
    TIMEOUT 60
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
