#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

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

project(rocprofiler-sdk-samples-advanced-thread-trace LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

find_package(rocprofiler-sdk REQUIRED)
find_package(
    amd_comgr
    REQUIRED
    CONFIG
    HINTS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATHS
    ${rocm_version_DIR}
    ${ROCM_PATH}
    PATH_SUFFIXES
    lib/cmake/amd_comgr)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(main.cpp PROPERTIES COMPILE_FLAGS "-g")

add_executable(advanced-thread-trace)
target_sources(advanced-thread-trace PRIVATE main.cpp client.cpp)
target_link_libraries(
    advanced-thread-trace
    PRIVATE rocprofiler-sdk::rocprofiler-sdk amd_comgr
            rocprofiler-sdk::samples-common-library rocprofiler-sdk::samples-build-flags)

rocprofiler_samples_get_preload_env(PRELOAD_ENV advanced-thread-trace)

add_test(NAME advanced-thread-trace COMMAND $<TARGET_FILE:advanced-thread-trace>)

set_tests_properties(
    advanced-thread-trace
    PROPERTIES TIMEOUT 45 LABELS "samples" ENVIRONMENT "${PRELOAD_ENV}"
               FAIL_REGULAR_EXPRESSION "threw an exception")
