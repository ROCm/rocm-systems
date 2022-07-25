cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-mpi-example LANGUAGES CXX)

set(CMAKE_BUILD_TYPE "Release")

find_package(MPI)
if(NOT MPI_FOUND)
    message(AUTHOR_WARNING "MPI could not be found. Cannot build omnitrace-mpi target")
    return()
endif()

find_package(Threads REQUIRED)

add_executable(mpi-example mpi.cpp)

target_link_libraries(
    mpi-example PRIVATE MPI::MPI_CXX Threads::Threads
                        $<TARGET_NAME_IF_EXISTS:omnitrace::omnitrace-compile-options>)

if(OMNITRACE_INSTALL_EXAMPLES)
    install(
        TARGETS mpi-example
        DESTINATION bin
        COMPONENT omnitrace-examples)
endif()
