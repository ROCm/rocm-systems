#
# aqlprofile v2 integration test (testv2), migrated from
# projects/aqlprofile/test/integration.
#
# Builds a HIP application that exercises the in-tree aqlprofile v2 API
# (aqlprofile_pmc_*) plus an LD_PRELOAD'd HSA-tool shared library (intercept)
# that enables profiling on created queues. Both link the in-tree aqlprofile
# statically so the migrated implementation is what gets tested.
#
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

enable_language(HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(CMAKE_HIP_EXTENSIONS OFF)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

set(_v2_common_libs
    -Wl,--start-group
    rocprofiler-sdk::rocprofiler-sdk-aqlprofile
    rocprofiler-sdk::rocprofiler-sdk-common-library
    -Wl,--end-group
    rocprofiler-sdk::rocprofiler-sdk-hsa-runtime
    rocprofiler-sdk::rocprofiler-sdk-amd-comgr
    rocprofiler-sdk::rocprofiler-sdk-drm)

# HSA-tool shared library: loaded via LD_PRELOAD, hooks hsa_queue_create to enable
# aqlprofile profiling on the queue (uses the in-tree v1 get_info).
add_library(aqlprofile-v2-intercept SHARED)
target_sources(aqlprofile-v2-intercept PRIVATE intercept.cpp)
target_include_directories(aqlprofile-v2-intercept PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(aqlprofile-v2-intercept PRIVATE AMD_INTERNAL_BUILD)
target_compile_options(aqlprofile-v2-intercept PRIVATE -Wno-shadow -Wno-unused-parameter)
target_link_libraries(aqlprofile-v2-intercept PRIVATE ${_v2_common_libs})
target_link_options(aqlprofile-v2-intercept PRIVATE
                    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exportmap)
set_target_properties(
    aqlprofile-v2-intercept PROPERTIES OUTPUT_NAME intercept LIBRARY_OUTPUT_DIRECTORY
                                                             ${CMAKE_CURRENT_BINARY_DIR})

# HIP application exercising the in-tree aqlprofile v2 API.
add_executable(aqlprofile-v2-test)
target_sources(aqlprofile-v2-test PRIVATE main.cpp workload.cpp counter.cpp agent.cpp)
target_include_directories(aqlprofile-v2-test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(aqlprofile-v2-test PRIVATE AMD_INTERNAL_BUILD)
target_compile_options(aqlprofile-v2-test PRIVATE -Wno-shadow -Wno-unused-parameter
                                                  -Wno-unused-variable)
set_target_properties(
    aqlprofile-v2-test PROPERTIES OUTPUT_NAME testv2 RUNTIME_OUTPUT_DIRECTORY
                                                     ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(aqlprofile-v2-test PRIVATE ${_v2_common_libs}
                                                 rocprofiler-sdk::rocprofiler-sdk-hip)

set(_v2_ld_library_path
    "${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}:${CMAKE_LIBRARY_OUTPUT_DIRECTORY}:${CMAKE_INSTALL_PREFIX}/lib:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:$ENV{LD_LIBRARY_PATH}"
    )

add_test(NAME tests.integration.aqlprofile.v2 COMMAND $<TARGET_FILE:aqlprofile-v2-test>)
set_tests_properties(
    tests.integration.aqlprofile.v2
    PROPERTIES
        ENVIRONMENT
        "LD_PRELOAD=$<TARGET_FILE:aqlprofile-v2-intercept>;LD_LIBRARY_PATH=${_v2_ld_library_path}"
        WORKING_DIRECTORY
        ${CMAKE_CURRENT_BINARY_DIR}
        LABELS
        "integration-tests;aqlprofile"
        TIMEOUT
        30
        FAIL_REGULAR_EXPRESSION
        "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
