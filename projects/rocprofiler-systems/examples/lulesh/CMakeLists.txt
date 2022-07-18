cmake_minimum_required(VERSION 3.16 FATAL_ERROR)

project(lulesh LANGUAGES C CXX)

set(CMAKE_BUILD_TYPE "RelWithDebInfo")

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake/Modules)

option(LULESH_BUILD_KOKKOS "Build Kokkos from submodule" ON)
if(LULESH_BUILD_KOKKOS)
    add_subdirectory(external)
    if(LULESH_USE_CUDA)
        kokkos_compilation(PROJECT)
    elseif(LULESH_USE_HIP AND NOT "${CMAKE_CXX_COMPILER}" MATCHES "hipcc")
        if(NOT HIPCC_EXECUTABLE)
            find_package(hip QUIET)

            find_program(
                HIPCC_EXECUTABLE
                NAMES hipcc
                HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
                PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm)
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
    set(CMAKE_BUILD_TYPE
        "Release"
        CACHE STRING "CMake build type" FORCE)
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

add_executable(${PROJECT_NAME} ${sources} ${headers})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/includes)
target_link_libraries(${PROJECT_NAME} PRIVATE Kokkos::kokkos lulesh-mpi)

if(OMNITRACE_INSTALL_EXAMPLES)
    if(LULESH_BUILD_KOKKOS)
        install(
            TARGETS kokkoscore kokkoscontainers
            DESTINATION ${CMAKE_INSTALL_LIBDIR}
            COMPONENT omnitrace-examples)
        set_target_properties(
            ${PROJECT_NAME} PROPERTIES INSTALL_RPATH
                                       "\$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
    endif()
    install(
        TARGETS ${PROJECT_NAME}
        DESTINATION bin
        COMPONENT omnitrace-examples)
endif()
