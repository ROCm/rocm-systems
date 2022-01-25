set(Kokkos_ENABLE_SERIAL
    ON
    CACHE BOOL "Enable Serial")
set(Kokkos_ENABLE_OPENMP
    ON
    CACHE BOOL "Enable OpenMP")
if(USE_CUDA)
    set(Kokkos_ENABLE_CUDA
        ON
        CACHE BOOL "Enable CUDA")
    set(Kokkos_ENABLE_CUDA_UVM
        ON
        CACHE BOOL "Enable CUDA UVM")
    set(Kokkos_ENABLE_CUDA_LAMBDA
        ON
        CACHE BOOL "Enable CUDA UVM")
    set(Kokkos_ENABLE_CUDA_CONSTEXPR
        ON
        CACHE BOOL "Enable CUDA UVM")
endif()

checkout_git_submodule(
    RELATIVE_PATH external/kokkos WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} REPO_URL
    https://github.com/kokkos/kokkos.git REPO_BRANCH develop)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)

add_subdirectory(kokkos)
