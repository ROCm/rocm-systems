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

set(KERNEL_REPLAY_SNAP_RESTORE_SOURCES snap_restore.cpp)
set_source_files_properties(${KERNEL_REPLAY_SNAP_RESTORE_SOURCES} PROPERTIES LANGUAGE HIP)

add_executable(kernel-replay-snapshot-test)
target_sources(kernel-replay-snapshot-test PRIVATE ${KERNEL_REPLAY_SNAP_RESTORE_SOURCES})

target_link_libraries(
    kernel-replay-snapshot-test
    PRIVATE rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::counter-test-constants
            GTest::gtest
            GTest::gtest_main)

rocprofiler_add_unit_test(
    TARGET kernel-replay-snapshot-test
    SOURCES ${KERNEL_REPLAY_SNAP_RESTORE_SOURCES}
    TIMEOUT 60
    ENVIRONMENT
        "LD_LIBRARY_PATH=${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:$ENV{LD_LIBRARY_PATH}"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
