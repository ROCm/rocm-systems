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

project(rocprofiler-sdk-tests-lib-transpose-shared-library LANGUAGES CXX HIP)

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

option(TRANSPOSE_USE_MPI "Enable MPI support in transpose-shared-library exe" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set_source_files_properties(transpose.cpp PROPERTIES LANGUAGE HIP)
add_library(transpose-shared-library SHARED)
target_sources(transpose-shared-library PRIVATE transpose.cpp)
target_compile_options(transpose-shared-library PRIVATE -W -Wall -Wextra -Wpedantic
                                                        -Wshadow -Werror)
target_include_directories(transpose-shared-library PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
set_target_properties(transpose-shared-library PROPERTIES OUTPUT_NAME transpose)

find_package(Threads REQUIRED)
target_link_libraries(transpose-shared-library PRIVATE Threads::Threads)

find_package(rocprofiler-sdk-roctx REQUIRED)
target_link_libraries(transpose-shared-library
                      PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)

if(TRANSPOSE_USE_MPI)
    find_package(MPI REQUIRED)
    target_compile_definitions(transpose-shared-library PRIVATE USE_MPI)
    target_link_libraries(transpose-shared-library PRIVATE MPI::MPI_C)
endif()
