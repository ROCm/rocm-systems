set(CMAKE_HIP_COMPILER
    "/opt/rocm/llvm/bin/amdclang++"
    CACHE STRING "desired c++ compiler" FORCE)
if(CMAKE_HIP_COMPILER_ID STREQUAL "Clang")
    message(STATUS "Using ${CMAKE_HIP_COMPILER} to build for amdgpu backend")
else()
    message(
        FATAL_ERROR
            "'amdclang++' compiler required to compile test binaries for ROCm platform.")
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
set_target_properties(vsequential_access PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                    ${CMAKE_SOURCE_DIR}/tests)

set(VRAND_SOURCES ../sample/vrandom_access.cpp)
set_source_files_properties(${VRAND_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(vrandom_access ${VRAND_SOURCES})
set_target_properties(vrandom_access PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                                ${CMAKE_SOURCE_DIR}/tests)

set(OCCUPANCY ../sample/occupancy.hip)
set_source_files_properties(${OCCUPANCY} PROPERTIES LANGUAGE HIP)
add_executable(occupancy ${OCCUPANCY})
set_target_properties(occupancy PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                           ${CMAKE_SOURCE_DIR}/tests)

set(MAT_MUL_MAX ../sample/mat_mul_max.hip)
set_source_files_properties(${MAT_MUL_MAX} PROPERTIES LANGUAGE HIP)
add_executable(mat_mul_max ${MAT_MUL_MAX})
set_target_properties(mat_mul_max PROPERTIES RUNTIME_OUTPUT_DIRECTORY
                                             ${CMAKE_SOURCE_DIR}/tests)


message(STATUS "GPU_NAME: ${GPU_NAME}")

if(KOKKOS_ARCH)
    message(STATUS "Using KOKKOS_ARCH: ${GPU_NAME}. Building mock_kokkos test.")
    # set(CMAKE_HIP_ARCHITECTURES ${GPU_NAME})
    set(CMAKE_HIP_ARCHITECTURES ${GPU_NAME} CACHE STRING "HIP GPU architecture" FORCE)
    set(KOKKOS_BUILD_DIR "${CMAKE_SOURCE_DIR}/tests/kokkos-build" CACHE PATH "Kokkos build dir" FORCE)
    set(Kokkos_DIR "${KOKKOS_BUILD_DIR}/lib/cmake/Kokkos")
    find_package(Kokkos REQUIRED)

    message(STATUS "Using KOKKOS_BUILD_DIR: ${KOKKOS_BUILD_DIR}. Building mock_kokkos test.")
    message(STATUS "Using Kokkos_DIR: ${Kokkos_DIR}. Building mock_kokkos test.")
    message(STATUS "Found Kokkos: ${Kokkos_DIR}")

    add_subdirectory(
        ${CMAKE_SOURCE_DIR}/sample/kokkos_mock_app
        ${CMAKE_BINARY_DIR}/kokkos_mock_app_build

        )

    
    # set(Kokkos_DIR "${CMAKE_BINARY_DIR}/kokkos-build/lib/cmake/Kokkos")
    # find_package(Kokkos REQUIRED)

    # set(MOCK_KOKKOS ../sample/mock_kokkos_minimal.cpp)
    # set_source_files_properties(${MOCK_KOKKOS} PROPERTIES LANGUAGE HIP)
    # add_executable(mock_kokkos ${MOCK_KOKKOS})
    # target_link_libraries(mock_kokkos PRIVATE Kokkos::kokkos)
    # set_target_properties(mock_kokkos PROPERTIES RUNTIME_OUTPUT_DIRECTORY
    #                                         ${CMAKE_SOURCE_DIR}/tests)
    message(STATUS "Built mock_kokkos test.")

else()
    message(ERROR
            "KOKKOS_ARCH not set. Skipping mock_kokkos test build.")
endif()
