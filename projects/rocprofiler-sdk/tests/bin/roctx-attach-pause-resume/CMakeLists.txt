#
# roctx-attach-pause-resume application for testing ROCTx pause/resume with attach
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

project(rocprofiler-tests-roctx-attach-pause-resume LANGUAGES CXX HIP)

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

find_package(Threads REQUIRED)
find_package(rocprofiler-sdk-roctx REQUIRED)

set_source_files_properties(roctx-attach-pause-resume.cpp PROPERTIES LANGUAGE HIP)
add_executable(roctx-attach-pause-resume)
target_sources(roctx-attach-pause-resume PRIVATE roctx-attach-pause-resume.cpp)
target_compile_options(roctx-attach-pause-resume PRIVATE -W -Wall -Wextra -Wpedantic
                                                         -Wshadow -Werror)

target_link_libraries(
    roctx-attach-pause-resume PRIVATE Threads::Threads
                                      rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
