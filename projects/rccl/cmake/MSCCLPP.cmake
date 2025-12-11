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
include(cmake/DownloadProject.cmake)

if(ENABLE_MSCCLPP)
    # Try to find the mscclpp install
    set(MSCCLPP_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/ext/mscclpp CACHE PATH "")
    execute_process(
        COMMAND mkdir -p ${MSCCLPP_ROOT}
    )
    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
    find_package(mscclpp_nccl)

    #if(NOT mscclpp_nccl_FOUND)
        # Ensure the source code is checked out
        set(MSCCLPP_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/mscclpp CACHE PATH "")
        set(JSON_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/json CACHE PATH "")
        if((NOT EXISTS ${MSCCLPP_SOURCE}/CMakeLists.txt) OR (NOT EXISTS ${JSON_SOURCE}/CMakeLists.txt))
            message(STATUS "Checking out external code")
            execute_process(
                COMMAND git submodule update --init --recursive
                WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            )
        endif()

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/cpx.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/read-allred.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/mscclpp_ibv_access_relaxed_ordering.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/mem-reg.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/non-multiple-128-fix.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/bf16-tuning.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/reg-fix.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/no-cache.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/device-flag.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/remove-clip.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/disable-executor.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/disable-format-checks.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

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

        download_project(PROJ                mscclpp_nccl
                         #GIT_REPOSITORY      https://github.com/microsoft/mscclpp.git
                         #GIT_TAG             4ee15b7ad085daaf74349d4c49c9b8480d28f0dc
                         INSTALL_DIR         ${MSCCLPP_ROOT}
                         LIST_SEPARATOR      %
			 CMAKE_ARGS          "-DGPU_TARGETS=${MSCCLPP_GPU_TARGETS}" -DMSCCLPP_BYPASS_GPU_CHECK=ON -DMSCCLPP_USE_ROCM=ON -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DMSCCLPP_BUILD_APPS_NCCL=ON -DMSCCLPP_BUILD_PYTHON_BINDINGS=OFF -DMSCCLPP_BUILD_TESTS=OFF -DMSCCLPP_CLIP_ENABLED=${ENABLE_MSCCLPP_CLIP} -DMSCCLPP_ENABLE_EXECUTOR=${ENABLE_MSCCLPP_EXECUTOR} -DMSCCLPP_ENABLE_FORMAT_CHECKS=${ENABLE_MSCCLPP_FORMAT_CHECKS} -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR> -DCMAKE_VERBOSE_MAKEFILE=1 "${CMAKE_INHERITED_ARGS}" -DFETCHCONTENT_SOURCE_DIR_JSON=${JSON_SOURCE}
                         LOG_DOWNLOAD        FALSE
                         LOG_CONFIGURE       FALSE
                         LOG_BUILD           FALSE
                         LOG_INSTALL         FALSE
                         UPDATE_DISCONNECTED TRUE
                         SOURCE_DIR          ${MSCCLPP_SOURCE}
        )

        find_package(mscclpp_nccl REQUIRED)

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/disable-format-checks.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/disable-executor.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/remove-clip.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/device-flag.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/no-cache.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/reg-fix.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/bf16-tuning.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/non-multiple-128-fix.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/mem-reg.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/mscclpp_ibv_access_relaxed_ordering.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/read-allred.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

        execute_process(
            COMMAND git apply --reverse ${CMAKE_CURRENT_SOURCE_DIR}/ext-src/cpx.patch
            WORKING_DIRECTORY ${MSCCLPP_SOURCE}
        )

    #endif()

    execute_process(COMMAND objcopy
                    --redefine-syms=${CMAKE_CURRENT_SOURCE_DIR}/src/misc/mscclpp/mscclpp_nccl_syms.txt
                    "${MSCCLPP_ROOT}/lib/libmscclpp_nccl_static.a"
                    "${PROJECT_BINARY_DIR}/libmscclpp_nccl.a"
    )
    add_library(mscclpp_nccl STATIC IMPORTED)
    set_target_properties(mscclpp_nccl PROPERTIES IMPORTED_LOCATION ${PROJECT_BINARY_DIR}/libmscclpp_nccl.a)

endif()
