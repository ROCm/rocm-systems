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

project(rocprofiler-test-app-transpose LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

option(TRANSPOSE_USE_MPI "Enable MPI support in transpose exe" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set_source_files_properties(transpose.cpp PROPERTIES LANGUAGE HIP)
add_executable(transpose)
target_sources(transpose PRIVATE transpose.cpp)
target_compile_options(transpose PRIVATE -W -Wall -Wextra -Wpedantic -Wshadow -Werror)

find_package(Threads REQUIRED)
target_link_libraries(transpose PRIVATE Threads::Threads)

if(TRANSPOSE_USE_MPI)
    find_package(MPI REQUIRED)
    target_compile_definitions(transpose PRIVATE USE_MPI)
    target_link_libraries(transpose PRIVATE MPI::MPI_C)
endif()

install(
    TARGETS transpose
    DESTINATION bin
    COMPONENT tests)
