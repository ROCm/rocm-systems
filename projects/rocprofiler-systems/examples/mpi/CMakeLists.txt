cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(omnitrace-mpi-example LANGUAGES CXX)

find_package(MPI)
if(NOT MPI_FOUND)
    message(AUTHOR_WARNING "MPI could not be found. Cannot build omnitrace-mpi target")
    return()
endif()

add_executable(mpi-example mpi.cpp)

if(TARGET omnitrace::omnitrace-compile-options)
    target_link_libraries(mpi-example PRIVATE omnitrace::omnitrace-compile-options)
endif()

target_link_libraries(mpi-example PRIVATE MPI::MPI_CXX)

if(NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_target_properties(mpi-example PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                 ${CMAKE_BINARY_DIR})
endif()
