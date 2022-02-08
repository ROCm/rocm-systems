cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(lulesh LANGUAGES C CXX)

list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake/Modules)

option(LULESH_BUILD_KOKKOS "Build Kokkos from submodule" ON)
if(LULESH_BUILD_KOKKOS)
    add_subdirectory(external)
    if(LULESH_USE_CUDA OR LULESH_USE_HIP)
        kokkos_compilation(PROJECT)
    endif()
else()
    find_package(Kokkos REQUIRED COMPONENTS separable_compilation)
    kokkos_compilation(PROJECT)
endif()

set(CMAKE_CXX_EXTENSIONS OFF)

if("${CMAKE_BUILD_TYPE}" STREQUAL "")
    set(CMAKE_BUILD_TYPE
        "RelWithDebInfo"
        CACHE STRING "CMake build type" FORCE)
endif()

if(DEFINED OMNITRACE_USE_MPI
   AND NOT LULESH_USE_CUDA
   AND NOT LULESH_USE_HIP)
    option(LULESH_USE_MPI "Enable MPI" ${OMNITRACE_USE_MPI})
else()
    option(LULESH_USE_MPI "Enable MPI" OFF)
endif()

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

if(NOT CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    set_target_properties(${PROJECT_NAME} PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                     ${CMAKE_BINARY_DIR})
endif()

enable_testing()
if(LULESH_USE_MPI)
    add_test(
        NAME lulesh
        COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 8
                $<TARGET_FILE:${PROJECT_NAME}> -i 100 -s 20 -p
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
else()
    add_test(
        NAME lulesh
        COMMAND $<TARGET_FILE:${PROJECT_NAME}> -i 100 -s 20 -p
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR})
endif()
