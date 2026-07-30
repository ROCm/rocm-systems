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

project(rocprofiler-sdk-tests-bin-pc-sampling-exec-mask-manipulation LANGUAGES CXX HIP)

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

set_source_files_properties(exec_mask_manipulation.cpp PROPERTIES LANGUAGE HIP)
add_executable(exec-mask-manipulation)
target_sources(exec-mask-manipulation PRIVATE exec_mask_manipulation.cpp)
# The test uses inline assembly with scalar instructions (s_mov_b32). In debug builds (no
# optimization), the compiler may classify operands used in s_mov_b32 instructions as
# vector rather than scalar, causing assembly errors. -O2 ensures the compiler correctly
# resolves these operands as scalar registers. Debug symbols (-g) are still required for
# PC sampling decoding validation.
target_compile_options(exec-mask-manipulation PRIVATE -O2 -W -Wall -Wextra -Wpedantic
                                                      -Wshadow -Werror -g)

find_package(Threads REQUIRED)
target_link_libraries(exec-mask-manipulation PRIVATE Threads::Threads)
