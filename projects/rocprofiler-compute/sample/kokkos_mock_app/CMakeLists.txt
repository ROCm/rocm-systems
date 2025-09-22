cmake_minimum_required(VERSION 3.16)
project(mock_kokkos_minimal CXX HIP)

# set(KOKKOS_BUILD_DIR "${CMAKE_SOURCE_DIR}/../../tests/kokkos-build")
# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

# set(CMAKE_HIP_ARCHITECTURES "gfx942")
message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "KOKKOS_BUILD_DIR: ${KOKKOS_BUILD_DIR}")

set_source_files_properties(mock_kokkos_minimal.cpp PROPERTIES LANGUAGE HIP)
add_executable(mock_kokkos_minimal mock_kokkos_minimal.cpp)
target_include_directories(
    mock_kokkos_minimal
    PRIVATE
        ${KOKKOS_BUILD_DIR}/include
        ${KOKKOS_BUILD_DIR}/core/src
        ${KOKKOS_BUILD_DIR}/containers/src
        ${KOKKOS_BUILD_DIR}/algorithms/src
        /opt/rocm/include
)
target_link_directories(mock_kokkos_minimal PRIVATE ${KOKKOS_BUILD_DIR}/lib)
target_link_libraries(
    mock_kokkos_minimal
    PRIVATE kokkoscore kokkoscontainers kokkosalgorithms
)
