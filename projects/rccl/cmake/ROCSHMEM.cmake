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

function(add_rocshmem_targets)

    # Common dependency: libibverbs is required for all rocSHMEM paths
    find_library(_IBVERBS ibverbs)
    if(NOT _IBVERBS)
        message(FATAL_ERROR "libibverbs not found (install rdma-core/libibverbs-dev)")
    endif()
    set(IBVERBS ${_IBVERBS} PARENT_SCOPE)

    # -----------------------------------------------------------------
    # Path 1: Pre-built rocSHMEM installation (ROCSHMEM_INSTALL_DIR)
    # -----------------------------------------------------------------
    if(ROCSHMEM_INSTALL_DIR)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
        find_package(rocshmem_static)
        if(rocshmem_static_FOUND)
            set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_INCLUDE_DIR}" PARENT_SCOPE)
            set(ROCSHMEM_LIBRARY     "${ROCSHMEM_LIBRARY}"      PARENT_SCOPE)
            return()
        endif()
    endif()

    # -----------------------------------------------------------------
    # Path 2: Build from mono-repo worktree source (ROCSHMEM_SOURCE_DIR)
    # -----------------------------------------------------------------
    if(NOT ROCSHMEM_SOURCE_DIR)
        message(FATAL_ERROR "ROCSHMEM_SOURCE_DIR is not set. "
            "Use --rocshmem with install.sh or pass -DROCSHMEM_SOURCE_DIR=<path>.")
    endif()

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

    set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_INSTALL_DIR}/include" PARENT_SCOPE)
    set(ROCSHMEM_LIBRARY     "${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a" PARENT_SCOPE)

    add_custom_target(rocshmem_static ALL DEPENDS rocshmem_ext)

endfunction()
