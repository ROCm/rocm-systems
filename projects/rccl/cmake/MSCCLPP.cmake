# MIT License
#
# Copyright (c) 2020 Advanced Micro Devices, Inc. All rights reserved.
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

# Dependencies

# HIP dependency is handled earlier in the project cmake file
# when VerifyCompiler.cmake is included.

# GIT

# Test dependencies

# For downloading, building, and installing required dependencies
include(ExternalProject)

function(add_mscclpp_targets)

    if(MSCCLPP_INSTALL_DIR)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
        find_package(mscclpp_nccl_static)
    endif()

    if(NOT mscclpp_nccl_static_FOUND)
        set(MSCCLPP_INSTALL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ext/mscclpp)
        set(MSCCLPP_INCLUDE_DIRS "${MSCCLPP_INSTALL_DIR}/include")
        set(MSCCLPP_NCCL_STATIC_LIB "${MSCCLPP_INSTALL_DIR}/lib/libmscclpp_nccl_static.a")
        execute_process(
            COMMAND mkdir -p ${MSCCLPP_INSTALL_DIR}
        )

        set(EXT_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ext-src)

        if ((NOT JSON_SOURCE) OR (NOT MSCCLPP_SOURCE))
            add_custom_command(
                OUTPUT
                    ${EXT_SOURCE}/mscclpp/CMakeLists.txt
                    ${EXT_SOURCE}/json/CMakeLists.txt
                COMMAND git submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                COMMENT "Checking out submodules for mscclpp and json"
            )
            add_custom_target(
                checkout_submodules
                DEPENDS
                    ${EXT_SOURCE}/mscclpp/CMakeLists.txt
                    ${EXT_SOURCE}/json/CMakeLists.txt
            )
        endif()

        # json source path defaults to <ext-source>/json but can be overridden via JSON_SOURCE
        add_custom_target(json_source)
        if (NOT JSON_SOURCE)
            set(JSON_SOURCE ${EXT_SOURCE}/json)
            add_dependencies(json_source checkout_submodules)
        endif()

        # mscclpp source path defaults to <ext-source>/mscclpp but can be overridden via MSCCLPP_SOURCE
        add_custom_target(mscclpp_source)
        if (NOT MSCCLPP_SOURCE)
            #GIT_REPOSITORY      https://github.com/microsoft/mscclpp.git
            #GIT_TAG             4ee15b7ad085daaf74349d4c49c9b8480d28f0dc
            set(MSCCLPP_SOURCE ${EXT_SOURCE}/mscclpp)

            add_custom_target(
                mscclpp_patches
                DEPENDS checkout_submodules
            )
            add_dependencies(mscclpp_source mscclpp_patches)
            if (MSCCLPP_APPLY_PATCHES)
                
                set(MSCCLPP_PATCHED ${CMAKE_CURRENT_BINARY_DIR}/mscclpp-patched)
                set(MSCCLPP_PATCHED_STAMP ${MSCCLPP_PATCHED}/.patched_stamp)
                set(MSCCLPP_COPIED_STAMP ${MSCCLPP_PATCHED}/.copied_stamp)
                set(MSCCLPP_PATCH_FILES
                    cpx.patch
                    read-allred.patch
                    mscclpp_ibv_access_relaxed_ordering.patch
                    mem-reg.patch
                    non-multiple-128-fix.patch
                    bf16-tuning.patch
                    reg-fix.patch
                    no-cache.patch
                    device-flag.patch
                    remove-clip.patch
                    disable-executor.patch
                    disable-format-checks.patch
                )

                set(MSCCLPP_PATCH_COMMANDS "")
                set(MSCCLPP_PATCH_DEPENDS "")
                foreach(PATCH_FILE ${MSCCLPP_PATCH_FILES})
                    list(APPEND MSCCLPP_PATCH_DEPENDS ${EXT_SOURCE}/${PATCH_FILE})
                    list(APPEND MSCCLPP_PATCH_COMMANDS
                        COMMAND patch -p1 --no-backup-if-mismatch < ${EXT_SOURCE}/${PATCH_FILE}
                    )
                endforeach()

                add_custom_command(
                    OUTPUT ${MSCCLPP_COPIED_STAMP}
                    COMMAND ${CMAKE_COMMAND} -E remove_directory ${MSCCLPP_PATCHED}
                    COMMAND ${CMAKE_COMMAND} -E copy_directory ${MSCCLPP_SOURCE} ${MSCCLPP_PATCHED}
                    COMMAND ${CMAKE_COMMAND} -E touch ${MSCCLPP_COPIED_STAMP}
                    DEPENDS
                        checkout_submodules
                        ${MSCCLPP_PATCH_DEPENDS}
                    WORKING_DIRECTORY ${MSCCLPP_SOURCE}
                    COMMENT "Copying mscclpp source to patch directory"
                )

                add_custom_command(
                    OUTPUT ${MSCCLPP_PATCHED_STAMP}
                    ${MSCCLPP_PATCH_COMMANDS}
                    COMMAND ${CMAKE_COMMAND} -E touch ${MSCCLPP_PATCHED_STAMP}
                    DEPENDS
                        ${MSCCLPP_COPIED_STAMP}
                        ${MSCCLPP_PATCH_DEPENDS}
                    WORKING_DIRECTORY ${MSCCLPP_PATCHED}
                    COMMENT "Applying patches to mscclpp"
                )
                add_custom_target(
                    mscclpp_patch_step
                    DEPENDS ${MSCCLPP_PATCHED_STAMP}
                )

                set(MSCCLPP_SOURCE ${MSCCLPP_PATCHED})
                add_dependencies(mscclpp_patches mscclpp_patch_step)
            endif()
        endif()

        set(CMAKE_INHERITED_ARGS "")
        set(CMAKE_ARGS_LIST "CMAKE_PREFIX_PATH;CMAKE_INSTALL_RPATH_USE_LINK_PATH;HIP_COMPILER")
        foreach(arg IN LISTS CMAKE_ARGS_LIST)
            if(DEFINED ${arg})
                string(REPLACE ";" "%" ARG_VALUE "${${arg}}") # Replace ; with new list separator symbol % to avoid CMake errors
                string(STRIP "${ARG_VALUE}" ARG_VALUE) # Eliminate whitespace, reducing to empty string if necessary

                # Only add a cmake argument if it has a value
                if("${ARG_VALUE}" STREQUAL "")
                    continue()
                endif()
                string(APPEND CMAKE_INHERITED_ARGS "-D${arg}=\"${ARG_VALUE}\" ")
            endif()
        endforeach()

        if(NOT DEFINED CACHE{MSCCLPP_GPU_TARGETS})
            message(STATUS "Building MSCCL++ only for supported variants: gfx942;gfx950")
            set(MSCCLPP_GPU_TARGETS "gfx942;gfx950")
            if(BUILD_ADDRESS_SANITIZER)
                set(MSCCLPP_GPU_TARGETS "gfx942:xnack+;gfx950:xnack+")
            endif()
        else()
            message(STATUS "Building MSCCL++ for ${MSCCLPP_GPU_TARGETS}")
        endif()

        string(REPLACE ";" "%" MSCCLPP_GPU_TARGETS "${MSCCLPP_GPU_TARGETS}")

        ExternalProject_Add(
            mscclpp
            INSTALL_DIR         ${MSCCLPP_INSTALL_DIR}
            LIST_SEPARATOR      %
            CMAKE_ARGS          "-DGPU_TARGETS=${MSCCLPP_GPU_TARGETS}" -DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_ROCM=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DMSCCLPP_BUILD_APPS_NCCL=ON -DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF -DMSCCLPP_BUILD_TESTS=OFF -DMSCCLPP_ENABLE_CLIP=${ENABLE_MSCCLPP_CLIP} -DMSCCLPP_ENABLE_EXECUTOR=${ENABLE_MSCCLPP_EXECUTOR} -DMSCCLPP_ENABLE_FORMAT_CHECKS=${ENABLE_MSCCLPP_FORMAT_CHECKS} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_VERBOSE_MAKEFILE=1 "${CMAKE_INHERITED_ARGS}" -DFETCHCONTENT_SOURCE_DIR_JSON=${JSON_SOURCE}
            LOG_DOWNLOAD        FALSE
            LOG_CONFIGURE       FALSE
            LOG_BUILD           FALSE
            LOG_INSTALL         FALSE
            UPDATE_DISCONNECTED TRUE
            SOURCE_DIR          ${MSCCLPP_SOURCE}
            BUILD_IN_SOURCE     TRUE
            TEST_COMMAND        ""
            DOWNLOAD_COMMAND    ""
            DEPENDS
                mscclpp_source
                json_source
        )
        add_custom_target(
            mscclpp_nccl_static
            DEPENDS mscclpp
        )
    endif()

    # LIBRARY TARGET mscclpp_nccl
    set(MSCCLPP_NCCL_LIB "${PROJECT_BINARY_DIR}/libmscclpp_nccl.a")
    add_custom_command(
        OUTPUT ${MSCCLPP_NCCL_LIB}
        COMMAND objcopy
            --redefine-syms=${CMAKE_CURRENT_SOURCE_DIR}/src/misc/mscclpp/mscclpp_nccl_syms.txt
            "${MSCCLPP_NCCL_STATIC_LIB}"
            "${MSCCLPP_NCCL_LIB}"
        DEPENDS mscclpp_nccl_static
        COMMENT "Renaming mscclpp NCCL API functions"
        VERBATIM
    )
    add_custom_target(
        mscclpp_nccl_redefine_syms
        DEPENDS ${MSCCLPP_NCCL_LIB}
    )
    add_library(mscclpp_nccl STATIC IMPORTED)
    set_target_properties(mscclpp_nccl PROPERTIES IMPORTED_LOCATION ${MSCCLPP_NCCL_LIB})
    add_dependencies(mscclpp_nccl mscclpp_nccl_redefine_syms)

endfunction()
