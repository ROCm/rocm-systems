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

project(rocprofiler-sdk-kernel-replay-tests LANGUAGES CXX HIP)

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

set(KERNEL_REPLAY_SNAP_RESTORE_SOURCES snap_restore.cpp)

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

rocprofiler_add_unit_test(
    TARGET kernel-replay-snapshot-test
    SOURCES ${KERNEL_REPLAY_SNAP_RESTORE_SOURCES}
    TIMEOUT 60
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
