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

project(rocprofiler-sdk-tests-bin-simple-transpose LANGUAGES CXX HIP)

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

set_source_files_properties(simple-transpose.cpp PROPERTIES LANGUAGE HIP)
add_executable(simple-transpose)
target_sources(simple-transpose PRIVATE simple-transpose.cpp)
target_compile_options(simple-transpose PRIVATE -W -Wall -Wextra -Wpedantic -Wshadow
                                                -Werror)

find_package(Threads REQUIRED)
target_link_libraries(simple-transpose PRIVATE Threads::Threads)

find_package(rocprofiler-sdk-roctx REQUIRED)
target_link_libraries(simple-transpose
                      PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)

find_package(MPI COMPONENTS C)
if(MPI_FOUND AND TARGET MPI::MPI_C)
    set_source_files_properties(mpi-simple-attach.cpp PROPERTIES LANGUAGE HIP)
    add_executable(mpi-simple-attach)
    target_sources(mpi-simple-attach PRIVATE mpi-simple-attach.cpp)
    target_compile_options(mpi-simple-attach PRIVATE -W -Wall -Wextra -Wpedantic -Wshadow
                                                     -Werror)
    target_link_libraries(mpi-simple-attach PRIVATE Threads::Threads MPI::MPI_C
                                                    rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
endif()
