include(Utilities)

option(LULESH_USE_CUDA "Enable Kokkos CUDA backend for lulesh" OFF)
option(LULESH_USE_HIP "Enable Kokkos HIP backend for lulesh" OFF)

set(Kokkos_ENABLE_SERIAL
    ON
    CACHE BOOL "Enable Serial")

if(LULESH_USE_CUDA)
    set(Kokkos_ENABLE_CUDA
        ON
        CACHE BOOL "Enable CUDA" FORCE)
    set(Kokkos_ENABLE_CUDA_LAMBDA
        ON
        CACHE BOOL "Enable CUDA lambda support" FORCE)
elseif(LULESH_USE_HIP)
    set(Kokkos_ENABLE_HIP
        ON
        CACHE BOOL "Enable HIP" FORCE)
else()
    set(Kokkos_ENABLE_OPENMP
        ON
        CACHE BOOL "Enable OpenMP")
endif()

# kokkos now lives at the root of the examples tree rather than inside this
# example, so that vendored sources are separable from first-party ones by
# directory (ROCm/rocm-systems#10227). ROCPROFSYS_EXAMPLE_ROOT_DIR is set by
# examples/CMakeLists.txt; fall back to a relative path when this example is
# configured standalone.
set(_lulesh_examples_root "${ROCPROFSYS_EXAMPLE_ROOT_DIR}")
if(NOT _lulesh_examples_root)
    get_filename_component(
        _lulesh_examples_root
        "${CMAKE_CURRENT_LIST_DIR}/../.."
        ABSOLUTE
    )
endif()

checkout_git_submodule(
    RELATIVE_PATH external/kokkos WORKING_DIRECTORY ${_lulesh_examples_root} REPO_URL
    https://github.com/kokkos/kokkos.git REPO_BRANCH develop
)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)

add_subdirectory(
    ${_lulesh_examples_root}/external/kokkos
    ${CMAKE_CURRENT_BINARY_DIR}/kokkos
    EXCLUDE_FROM_ALL
)
