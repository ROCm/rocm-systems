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

    # Check for an existing installation via the user-provided prefix ROCSHMEM_INSTALL DIR
    if(ROCSHMEM_INSTALL_DIR)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
        find_package(rocshmem_static)
        if(NOT IBVERBS)
            find_library(IBVERBS ibverbs)
            if(IBVERBS)
                set(IBVERBS ${IBVERBS} PARENT_SCOPE)
            endif()
        endif()
    endif()

    # If no pre-existing installation, build from submodule into ext/rocshmem
    if(NOT rocshmem_static_FOUND)
        set(_rccl_root            "${CMAKE_SOURCE_DIR}")
        set(ROCSHMEM_SOURCE       "${_rccl_root}/ext-src/rocSHMEM")
        set(ROCSHMEM_INSTALL_DIR  "${_rccl_root}/ext/rocshmem")

        # Make sure submodule exists (same style as MSCCL++: custom rule + target)
        add_custom_command(
            OUTPUT "${ROCSHMEM_SOURCE}/CMakeLists.txt"
            COMMAND git submodule update --init --recursive ext-src/rocSHMEM
            WORKING_DIRECTORY "${_rccl_root}"
            COMMENT "Checking out submodule: ext-src/rocSHMEM"
            VERBATIM
        )

        add_custom_target(rocshmem_checkout_submodule
            DEPENDS "${ROCSHMEM_SOURCE}/CMakeLists.txt")

        # Where our patch files live (like MSCCL++)
        set(EXT_SOURCE "${_rccl_root}/ext-src")

            # Build and install rocSHMEM. We run `../build_scripts/gdx_bxnt`
        # from a 'build' dir just like the README shows.
        ExternalProject_Add(rocshmem_ext
            SOURCE_DIR          "${ROCSHMEM_SOURCE}"
            INSTALL_DIR         "${ROCSHMEM_INSTALL_DIR}"
            UPDATE_DISCONNECTED TRUE
            LOG_DOWNLOAD        FALSE
            LOG_CONFIGURE       FALSE
            LOG_BUILD           FALSE
            LOG_INSTALL         FALSE
            BUILD_IN_SOURCE     TRUE
            DOWNLOAD_COMMAND    ""   # using the submodule checkout above
            TEST_COMMAND        ""
            DEPENDS             rocshmem_checkout_submodule   

            # Rocshmem submodule commit hash -> commit b28a56bd54ccc581d05a439ffa466c3dacb3385
            # The project has its own scripts; we replicate the README sequence:
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

         # After build, define the variables RCCL expects
        set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_INSTALL_DIR}/include" PARENT_SCOPE)
        set(ROCSHMEM_LIBRARY      "${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a" PARENT_SCOPE)
        find_library(_IBVERBS ibverbs)
        if(NOT _IBVERBS)
            message(FATAL_ERROR "libibverbs not found (install rdma-core/libibverbs-dev)")
        endif()
        set(IBVERBS ${_IBVERBS} PARENT_SCOPE)

        # Provide a dummy target other code can depend on
        add_custom_target(rocshmem_static ALL DEPENDS rocshmem_ext)
    else()
    # We found a prebuilt rocSHMEM; export variables upward as-is
    set(ROCSHMEM_INCLUDE_DIR  "${ROCSHMEM_INCLUDE_DIR}" PARENT_SCOPE)
    set(ROCSHMEM_LIBRARY      "${ROCSHMEM_LIBRARY}"      PARENT_SCOPE)

    find_library(_IBVERBS ibverbs)
    if(NOT _IBVERBS)
        message(FATAL_ERROR "libibverbs not found")
    endif()
    set(IBVERBS ${_IBVERBS} PARENT_SCOPE)
    endif()

endfunction()
