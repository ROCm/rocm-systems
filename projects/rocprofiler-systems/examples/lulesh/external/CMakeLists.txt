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

checkout_git_submodule(
    RELATIVE_PATH external/kokkos WORKING_DIRECTORY ${PROJECT_SOURCE_DIR} REPO_URL
    https://github.com/kokkos/kokkos.git REPO_BRANCH develop)

set(CMAKE_SKIP_INSTALL_ALL_DEPENDENCY ON)

add_subdirectory(kokkos EXCLUDE_FROM_ALL)
