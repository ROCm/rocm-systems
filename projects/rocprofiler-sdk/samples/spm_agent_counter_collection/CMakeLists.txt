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

project(
    rocprofiler-tests-spm
    LANGUAGES CXX HIP
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Threads REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
set(IS_DISABLED TRUE)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set_source_files_properties(kernel_branch.cpp PROPERTIES COMPILE_FLAGS "-g -O2")
set_source_files_properties(kernel_branch.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(kernel_lds.cpp PROPERTIES COMPILE_FLAGS "-g -O2")
set_source_files_properties(kernel_lds.cpp PROPERTIES LANGUAGE HIP)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

# Agent profiling test
add_executable(spm-api-agent-test)
target_sources(spm-api-agent-test PRIVATE main.cpp common.cpp agent.cpp kernel_branch.cpp
                                          kernel_lds.cpp)
target_link_libraries(spm-api-agent-test PRIVATE rocprofiler-sdk::rocprofiler-sdk
                                                 Threads::Threads)

add_test(NAME spm-api-agent-test COMMAND $<TARGET_FILE:spm-api-agent-test>)

set_tests_properties(
    spm-api-agent-test
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${IS_DISABLED})
