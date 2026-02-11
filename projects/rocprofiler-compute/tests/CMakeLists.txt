find_package(MPI COMPONENTS CXX)

set(CMAKE_HIP_COMPILER "amdclang++" CACHE STRING "desired c++ compiler" FORCE)
if(CMAKE_HIP_COMPILER_ID STREQUAL "Clang" OR CMAKE_HIP_COMPILER_ID STREQUAL "ROCMClang")
    message(STATUS "Using ${CMAKE_HIP_COMPILER} to build for amdgpu backend")
else()
    message(
        FATAL_ERROR
        "'amdclang++' compiler required to compile test binaries for ROCm platform. Found: ${CMAKE_HIP_COMPILER_ID}"
    )
endif()

set(VCOPY_SOURCES ../sample/vcopy.cpp)
set_source_files_properties(${VCOPY_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(vcopy ${VCOPY_SOURCES})
set_target_properties(vcopy PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

set(VMEM_SOURCES ../sample/vmem.hip)
set_source_files_properties(${VMEM_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(vmem ${VMEM_SOURCES})
set_target_properties(vmem PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests)

set(VSEQ_SOURCES ../sample/vsequential_access.cpp)
set_source_files_properties(${VSEQ_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(vsequential_access ${VSEQ_SOURCES})
set_target_properties(
    vsequential_access
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

set(VRAND_SOURCES ../sample/vrandom_access.cpp)
set_source_files_properties(${VRAND_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(vrandom_access ${VRAND_SOURCES})
set_target_properties(
    vrandom_access
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

set(OCCUPANCY ../sample/occupancy.hip)
set_source_files_properties(${OCCUPANCY} PROPERTIES LANGUAGE HIP)
add_executable(occupancy ${OCCUPANCY})
set_target_properties(
    occupancy
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

set(MAT_MUL_MAX ../sample/mat_mul_max.hip)
set_source_files_properties(${MAT_MUL_MAX} PROPERTIES LANGUAGE HIP)
add_executable(mat_mul_max ${MAT_MUL_MAX})
target_compile_options(mat_mul_max PRIVATE -g -O0)
set_target_properties(
    mat_mul_max
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

set(DYNAMIC_SHARED_SOURCES ../sample/dynamic_shared/dynamic_shared.hip)
set_source_files_properties(${DYNAMIC_SHARED_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(hip_dynamic_shared ${DYNAMIC_SHARED_SOURCES})
target_include_directories(hip_dynamic_shared PRIVATE ../sample/dynamic_shared)
set_target_properties(
    hip_dynamic_shared
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

set(LAPLACE_EQN_SOURCES ../sample/laplace_eqn.hip)
set_source_files_properties(${LAPLACE_EQN_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(laplace_eqn ${LAPLACE_EQN_SOURCES})
set_target_properties(
    laplace_eqn
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

if(MPI_CXX_COMPILER)
    message(STATUS "Building laplace_eqn with MPI support")
    add_executable(mpi_aware_laplace_eqn ${LAPLACE_EQN_SOURCES})
    set_target_properties(
        mpi_aware_laplace_eqn
        PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
    )
    target_compile_definitions(mpi_aware_laplace_eqn PRIVATE USE_MPI)
    # Link MPI libraries and include directories
    target_include_directories(mpi_aware_laplace_eqn PRIVATE ${MPI_INCLUDE_PATH})
    target_link_libraries(mpi_aware_laplace_eqn PRIVATE MPI::MPI_CXX)
endif()

set(ROCFLOP_SOURCES ../sample/rocflop.cpp)
set_source_files_properties(${ROCFLOP_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(rocflop ${ROCFLOP_SOURCES})
set_target_properties(
    rocflop
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)

set(MEMCOPY_SOURCES ../sample/memcopy.cpp)
set_source_files_properties(${MEMCOPY_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(memcopy ${MEMCOPY_SOURCES})
set_target_properties(
    memcopy
    PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/tests
)
