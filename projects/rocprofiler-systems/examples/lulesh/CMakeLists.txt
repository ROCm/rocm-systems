cmake_minimum_required(VERSION 3.18.4 FATAL_ERROR)

project(rocprofiler-systems-lulesh-example LANGUAGES C CXX)

if(ROCPROFSYS_DISABLE_EXAMPLES)
    get_filename_component(_DIR ${CMAKE_CURRENT_LIST_DIR} NAME)

    if(
        ${PROJECT_NAME} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
        OR ${_DIR} IN_LIST ROCPROFSYS_DISABLE_EXAMPLES
    )
        return()
    endif()
endif()

set(CMAKE_BUILD_TYPE "RelWithDebInfo")
string(
    REGEX REPLACE
    " -g(|[0-2]) "
    " -g3 "
    CMAKE_CXX_FLAGS_RELWITHDEBINFO
    "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}"
)

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake/Modules)

option(LULESH_BUILD_KOKKOS "Build Kokkos from submodule" ON)
if(LULESH_BUILD_KOKKOS)
    add_subdirectory(external)
    if(LULESH_USE_CUDA)
        kokkos_compilation(PROJECT COMPILER ${Kokkos_NVCC_WRAPPER})
    elseif(LULESH_USE_HIP AND NOT "${CMAKE_CXX_COMPILER}" MATCHES "hipcc")
        if(NOT HIPCC_EXECUTABLE)
            find_package(hip QUIET HINTS ${ROCmVersion_DIR} PATHS ${ROCmVersion_DIR})

            find_program(
                HIPCC_EXECUTABLE
                NAMES hipcc
                HINTS ${ROCmVersion_DIR} ${ROCM_PATH}
                ENV ROCM_PATH
                /opt/rocm
                PATHS ${ROCmVersion_DIR} ${ROCM_PATH}
                ENV ROCM_PATH
                /opt/rocm
            )
            mark_as_advanced(HIPCC_EXECUTABLE)
        endif()
        kokkos_compilation(PROJECT COMPILER ${HIPCC_EXECUTABLE})
    endif()
else()
    find_package(Kokkos REQUIRED COMPONENTS separable_compilation)
    kokkos_compilation(PROJECT)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "CMake build type" FORCE)
endif()

option(LULESH_USE_MPI "Enable MPI" OFF)
add_library(lulesh-mpi INTERFACE)
if(LULESH_USE_MPI)
    find_package(MPI REQUIRED)
    target_compile_definitions(lulesh-mpi INTERFACE USE_MPI=1)
    target_link_libraries(lulesh-mpi INTERFACE MPI::MPI_C MPI::MPI_CXX)
else()
    target_compile_definitions(lulesh-mpi INTERFACE USE_MPI=0)
endif()

if(NOT TARGET Kokkos::kokkos)
    find_package(Kokkos REQUIRED)
endif()

file(GLOB headers ${PROJECT_SOURCE_DIR}/*.h ${PROJECT_SOURCE_DIR}/*.hxx)
file(GLOB sources ${PROJECT_SOURCE_DIR}/*.cc)

rocprofiler_systems_causal_example_executable(
    "lulesh"
    SOURCES ${sources} ${headers}
    LINK_LIBRARIES Kokkos::kokkos lulesh-mpi
    INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/includes
)

if(ROCPROFSYS_INSTALL_EXAMPLES)
    if(LULESH_BUILD_KOKKOS)
        install(
            TARGETS kokkoscore kokkoscontainers
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT rocprofiler-systems-examples
        )
        set_target_properties(
            lulesh
            PROPERTIES INSTALL_RPATH "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}"
        )
    endif()
endif()
