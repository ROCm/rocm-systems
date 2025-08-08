cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-mpi-examples LANGUAGES C CXX)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

find_package(MPI)
if(NOT MPI_FOUND)
    if("${CMAKE_PROJECT_NAME}" STREQUAL "rocprofiler-systems" AND "$ENV{ROCPROFSYS_CI}")
        set(_MSG_TYPE STATUS) # don't generate warnings during CI
    else()
        set(_MSG_TYPE AUTHOR_WARNING)
    endif()
    message(
        ${_MSG_TYPE}
        "MPI could not be found. Cannot build rocprofiler-systems-mpi target"
    )
    return()
endif()

find_package(Threads REQUIRED)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

add_library(mpi-c-interface-library INTERFACE)
target_link_libraries(
    mpi-c-interface-library
    INTERFACE
        Threads::Threads
        MPI::MPI_C
        $<TARGET_NAME_IF_EXISTS:rocprofiler-systems::rocprofiler-systems-compile-options>
)
target_compile_options(mpi-c-interface-library INTERFACE -Wno-double-promotion)

add_executable(mpi-allgather allgather.c)
target_link_libraries(mpi-allgather PRIVATE mpi-c-interface-library)

add_executable(mpi-bcast bcast.c)
target_link_libraries(mpi-bcast PRIVATE mpi-c-interface-library)

add_executable(mpi-all2all all2all.c)
target_link_libraries(mpi-all2all PRIVATE mpi-c-interface-library)

add_executable(mpi-reduce reduce.c)
target_link_libraries(mpi-reduce PRIVATE mpi-c-interface-library)

add_executable(mpi-scatter-gather scatter-gather.c)
target_link_libraries(mpi-scatter-gather PRIVATE mpi-c-interface-library)

add_executable(mpi-send-recv send-recv.c)
target_link_libraries(mpi-send-recv PRIVATE mpi-c-interface-library)

add_executable(mpi-allreduce allreduce.c)
target_link_libraries(mpi-allreduce PRIVATE mpi-c-interface-library m)

set(CMAKE_BUILD_TYPE "Release")

add_library(mpi-cxx-interface-library INTERFACE)
target_link_libraries(
    mpi-cxx-interface-library
    INTERFACE
        Threads::Threads
        MPI::MPI_CXX
        $<TARGET_NAME_IF_EXISTS:rocprofiler-systems::rocprofiler-systems-compile-options>
)

add_executable(mpi-example mpi.cpp)
target_link_libraries(mpi-example PRIVATE mpi-cxx-interface-library)

if(ROCPROFSYS_INSTALL_EXAMPLES)
    install(
        TARGETS
            mpi-example
            mpi-allgather
            mpi-bcast
            mpi-all2all
            mpi-reduce
            mpi-scatter-gather
            mpi-send-recv
        DESTINATION bin
        COMPONENT rocprofiler-systems-examples
    )
endif()
