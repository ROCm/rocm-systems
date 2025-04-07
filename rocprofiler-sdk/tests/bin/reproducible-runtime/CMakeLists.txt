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

project(rocprofiler-sdk-tests-bin-reproducible-runtime LANGUAGES CXX HIP)

if(NOT CMAKE_BUILD_TYPE MATCHES "(Release|RelWithDebInfo)")
    set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

foreach(_TYPE DEBUG MINSIZEREL RELEASE RELWITHDEBINFO)
    if("${CMAKE_HIP_FLAGS_${_TYPE}}" STREQUAL "")
        set(CMAKE_HIP_FLAGS_${_TYPE} "${CMAKE_CXX_FLAGS_${_TYPE}}")
    endif()
endforeach()

option(REPRODUCIBLE_RUNTIME_USE_MPI "Enable MPI support in reproducible-runtime exe" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

set_source_files_properties(reproducible-runtime.cpp PROPERTIES LANGUAGE HIP)
add_executable(reproducible-runtime)
target_sources(reproducible-runtime PRIVATE reproducible-runtime.cpp)
target_compile_options(reproducible-runtime PRIVATE -W -Wall -Wextra -Wpedantic -Wshadow
                                                    -Werror)

find_package(Threads REQUIRED)
target_link_libraries(reproducible-runtime PRIVATE Threads::Threads)

find_package(rocprofiler-sdk-roctx REQUIRED)
target_link_libraries(reproducible-runtime
                      PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)

if(REPRODUCIBLE_RUNTIME_USE_MPI)
    find_package(MPI REQUIRED)
    target_compile_definitions(reproducible-runtime PRIVATE USE_MPI)
    target_link_libraries(reproducible-runtime PRIVATE MPI::MPI_C)
endif()
