cmake_minimum_required(VERSION 3.21 FATAL_ERROR)

find_program(
    HIPCC_EXECUTABLE
    NAMES hipcc
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
    PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm NO_CACHE)
mark_as_advanced(HIPCC_EXECUTABLE)

if(HIPCC_EXECUTABLE)
    set(CMAKE_CXX_COMPILER ${HIPCC_EXECUTABLE})
endif()

project(rocprofiler-transpose-sample LANGUAGES CXX)

option(TRANSPOSE_USE_MPI "Enable MPI support in transpose exe" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
    COMPONENT rocprofiler-samples)
