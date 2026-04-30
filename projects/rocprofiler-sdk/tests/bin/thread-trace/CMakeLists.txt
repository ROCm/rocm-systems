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

project(rocprofiler-sdk-tests-bin-thread-trace LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set_source_files_properties(
    kernel_branch.cpp kernel_lds.cpp
    PROPERTIES LANGUAGE HIP COMPILE_FLAGS "${CMAKE_HIP_FLAGS_RELWITHDEBINFO}")
set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)

add_executable(thread-trace-test-app)
target_sources(thread-trace-test-app PRIVATE main.cpp kernel_branch.cpp kernel_lds.cpp)

find_package(rocprofiler-sdk-roctx REQUIRED)
target_link_libraries(
    thread-trace-test-app PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx
                                  rocprofiler-sdk::tests-build-flags)
