project(mock_kokkos_minimal CXX HIP)

message(STATUS "CMAKE_HIP_ARCHITECTURES: ${CMAKE_HIP_ARCHITECTURES}")
message(STATUS "KOKKOS_BUILD_DIR: ${KOKKOS_BUILD_DIR}")

if(NOT USE_MOCK_KOKKOS)
    set(Kokkos_DIR "${KOKKOS_BUILD_DIR}/lib/cmake/Kokkos")
    find_package(Kokkos QUIET)
    message(STATUS "Found Kokkos: ${Kokkos_DIR}")
    if(NOT Kokkos_FOUND)
        message(WARNING "Kokkos not found at ${Kokkos_DIR}, falling back to mock")
        set(USE_MOCK_KOKKOS ON PARENT_SCOPE)
        set(USE_MOCK_KOKKOS ON)
    endif()
endif()

if(NOT USE_MOCK_KOKKOS)
    set_source_files_properties(real_kokkos_minimal.cpp PROPERTIES LANGUAGE HIP)
    add_executable(real_kokkos_minimal real_kokkos_minimal.cpp)

    target_include_directories(
        real_kokkos_minimal
        PRIVATE
            ${KOKKOS_BUILD_DIR}/include
            ${KOKKOS_BUILD_DIR}/core/src
            ${KOKKOS_BUILD_DIR}/containers/src
            ${KOKKOS_BUILD_DIR}/algorithms/src
            /opt/rocm/include
    )

    target_link_directories(real_kokkos_minimal PRIVATE ${KOKKOS_BUILD_DIR}/lib)
    target_link_libraries(
        real_kokkos_minimal
        PRIVATE kokkoscore kokkoscontainers kokkosalgorithms
    )
else()
    # Mock Kokkos - minimal dependencies
    set_source_files_properties(mock_kokkos_minimal.cpp PROPERTIES LANGUAGE HIP)
    add_executable(mock_kokkos_minimal mock_kokkos_minimal.cpp)

    # Only need HIP runtime for GPU kernels

    target_include_directories(mock_kokkos_minimal PRIVATE /opt/rocm/include)
    target_link_directories(mock_kokkos_minimal PRIVATE /opt/rocm/lib)
    # target_link_libraries(mock_kokkos_minimal PRIVATE hip)
endif()
