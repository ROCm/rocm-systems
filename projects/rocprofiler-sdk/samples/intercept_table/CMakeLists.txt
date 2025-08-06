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

project(rocprofiler-sdk-samples-intercept-table LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

#
# below is equivalent to: ``find_package(rocprofiler-sdk REQUIRED)``. Use COMPONENTS below
# to demonstrate/test support for COMPONENTS
find_package(rocprofiler-sdk REQUIRED COMPONENTS headers shared-library)

add_library(intercept-table-client SHARED)
target_sources(intercept-table-client PRIVATE client.cpp client.hpp)
target_link_libraries(
    intercept-table-client
    PRIVATE rocprofiler-sdk::rocprofiler-sdk rocprofiler-sdk::samples-build-flags
            rocprofiler-sdk::samples-common-library)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
find_package(Threads REQUIRED)

add_executable(intercept-table)
target_sources(intercept-table PRIVATE main.cpp)
target_link_libraries(intercept-table PRIVATE intercept-table-client Threads::Threads
                                              rocprofiler-sdk::samples-build-flags)

rocprofiler_samples_get_preload_env(PRELOAD_ENV intercept-table-client)
rocprofiler_samples_get_ld_library_path_env(LIBRARY_PATH_ENV)

set(intercept-table-env ${PRELOAD_ENV} ${LIBRARY_PATH_ENV})

add_test(NAME intercept-table COMMAND $<TARGET_FILE:intercept-table>)

set_tests_properties(
    intercept-table
    PROPERTIES TIMEOUT 45 LABELS "samples" ENVIRONMENT "${intercept-table-env}"
               FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
