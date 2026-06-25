# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include(ExternalProject)

# Pinned mono-repo commit for rocshmem source checkout.
# Used by both install.sh (setup_rocshmem_worktree) and cmake (auto-detect below).
set(ROCSHMEM_MONO_HASH "0e2998b11f99e8302c72f1ac2ce9f2b8c1816587" CACHE STRING
    "Pinned rocm-systems commit hash for rocshmem source checkout")

function(add_rocshmem_targets)

    # Common dependency: libibverbs is required for all rocSHMEM paths
    find_library(_IBVERBS ibverbs)
    if(NOT _IBVERBS)
        message(FATAL_ERROR "libibverbs not found (install rdma-core/libibverbs-dev)")
    endif()
    set(IBVERBS ${_IBVERBS} PARENT_SCOPE)

    # -----------------------------------------------------------------
    # Auto-detect ROCSHMEM_SOURCE_DIR if not provided.
    # Runs first so source headers are available regardless of whether
    # we use a pre-built install or build from source.
    # -----------------------------------------------------------------
    if(NOT ROCSHMEM_SOURCE_DIR)
        # Try mono-repo relative path (projects/rocshmem alongside projects/rccl)
        get_filename_component(_mono_root "${CMAKE_SOURCE_DIR}/../.." ABSOLUTE)
        if(EXISTS "${_mono_root}/projects/rocshmem/CMakeLists.txt")
            set(ROCSHMEM_SOURCE_DIR "${_mono_root}/projects/rocshmem")
            message(STATUS "rocSHMEM: found source at ${ROCSHMEM_SOURCE_DIR}")
        else()
            # Sparse checkout: create git worktree for rocshmem sources
            set(_worktree_dir "${CMAKE_BINARY_DIR}/_rocshmem_src")
            if(NOT EXISTS "${_worktree_dir}/projects/rocshmem/CMakeLists.txt")
                message(STATUS "rocSHMEM: setting up sparse worktree at ${_worktree_dir}")
                find_package(Git REQUIRED)
                # Find the mono-repo .git dir (may be above CMAKE_SOURCE_DIR)
                execute_process(
                    COMMAND ${GIT_EXECUTABLE} rev-parse --show-toplevel
                    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                    OUTPUT_VARIABLE _git_root
                    OUTPUT_STRIP_TRAILING_WHITESPACE
                    RESULT_VARIABLE _git_result)
                if(NOT _git_result EQUAL 0)
                    message(FATAL_ERROR "rocSHMEM: not in a git repo, cannot create worktree. "
                        "Pass -DROCSHMEM_SOURCE_DIR=<path> manually.")
                endif()
                execute_process(
                    COMMAND ${GIT_EXECUTABLE} worktree add --no-checkout
                            "${_worktree_dir}" "${ROCSHMEM_MONO_HASH}"
                    WORKING_DIRECTORY "${_git_root}"
                    RESULT_VARIABLE _wt_result)
                if(NOT _wt_result EQUAL 0)
                    message(FATAL_ERROR "rocSHMEM: failed to create git worktree at ${_worktree_dir}")
                endif()
                execute_process(
                    COMMAND ${GIT_EXECUTABLE} sparse-checkout init --cone
                    WORKING_DIRECTORY "${_worktree_dir}")
                execute_process(
                    COMMAND ${GIT_EXECUTABLE} sparse-checkout set projects/rocshmem
                    WORKING_DIRECTORY "${_worktree_dir}")
                execute_process(
                    COMMAND ${GIT_EXECUTABLE} checkout
                    WORKING_DIRECTORY "${_worktree_dir}")
            else()
                message(STATUS "rocSHMEM: reusing existing worktree at ${_worktree_dir}")
            endif()
            if(NOT EXISTS "${_worktree_dir}/projects/rocshmem/CMakeLists.txt")
                message(FATAL_ERROR "rocSHMEM: worktree checkout failed — "
                    "projects/rocshmem not found at ${_worktree_dir}")
            endif()
            set(ROCSHMEM_SOURCE_DIR "${_worktree_dir}/projects/rocshmem")
            message(STATUS "rocSHMEM: source from worktree at ${ROCSHMEM_SOURCE_DIR}")
        endif()
    endif()

    # -----------------------------------------------------------------
    # Path 1: Pre-built rocSHMEM installation (ROCSHMEM_INSTALL_DIR)
    # -----------------------------------------------------------------
    if(ROCSHMEM_INSTALL_DIR)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
        find_package(rocshmem_static)
        if(rocshmem_static_FOUND)
            set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_INCLUDE_DIR}" PARENT_SCOPE)
            set(ROCSHMEM_LIBRARY     "${ROCSHMEM_LIBRARY}"      PARENT_SCOPE)
            set(ROCSHMEM_SOURCE_DIR  "${ROCSHMEM_SOURCE_DIR}"   PARENT_SCOPE)
            return()
        endif()
    endif()

    # -----------------------------------------------------------------
    # Path 2: Build from source (ENABLE_ROCSHMEM only)
    # ENABLE_ROCSHMEM_GIN only needs source headers, not a built library.
    # -----------------------------------------------------------------
    if(ENABLE_ROCSHMEM)
        set(_rccl_root           "${CMAKE_SOURCE_DIR}")
        set(ROCSHMEM_INSTALL_DIR "${_rccl_root}/ext/rocshmem")
        message(STATUS "rocSHMEM: building from ${ROCSHMEM_SOURCE_DIR}")

        ExternalProject_Add(rocshmem_ext
            SOURCE_DIR          "${ROCSHMEM_SOURCE_DIR}"
            INSTALL_DIR         "${ROCSHMEM_INSTALL_DIR}"
            UPDATE_DISCONNECTED TRUE
            LOG_DOWNLOAD        FALSE
            LOG_CONFIGURE       FALSE
            LOG_BUILD           FALSE
            LOG_INSTALL         FALSE
            BUILD_IN_SOURCE     TRUE
            DOWNLOAD_COMMAND    ""
            TEST_COMMAND        ""

            CONFIGURE_COMMAND   ""
            BUILD_COMMAND
                ${CMAKE_COMMAND} -E make_directory build
                && ${CMAKE_COMMAND} -E chdir build bash -lc "../scripts/build_configs/gda_bnxt -DUSE_EXTERNAL_MPI=OFF -DUSE_IPC=ON -DBUILD_EXAMPLES=OFF "
                && ${CMAKE_COMMAND} -E chdir build ${CMAKE_COMMAND}
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
                    -DBUILD_EXAMPLES=OFF ..
                && ${CMAKE_COMMAND} -E chdir build ${CMAKE_MAKE_PROGRAM} -j
            INSTALL_COMMAND
                ${CMAKE_COMMAND} -E chdir build ${CMAKE_MAKE_PROGRAM} install
        )

        set(ROCSHMEM_INSTALL_DIR "${ROCSHMEM_INSTALL_DIR}"          PARENT_SCOPE)
        set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_INSTALL_DIR}/include"  PARENT_SCOPE)
        set(ROCSHMEM_LIBRARY     "${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a" PARENT_SCOPE)

        add_custom_target(rocshmem_static ALL DEPENDS rocshmem_ext)
    endif()

    set(ROCSHMEM_SOURCE_DIR "${ROCSHMEM_SOURCE_DIR}" PARENT_SCOPE)

endfunction()
