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
set(ROCSHMEM_MONO_HASH "723d405cc7d0739f96a47feaa72f64bf2bad1e5c" CACHE STRING
    "Pinned rocm-systems commit hash for rocshmem source checkout")

function(add_rocshmem_targets)

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

    # GIN-only: the gate matters solely for the copied GIN headers (which no longer
    # ship constmem.hpp). An ENABLE_ROCSHMEM build compiles librocshmem itself and
    # does not use the copied headers, so it must not be blocked by a pre-gate source.
    set(_rocshmem_config_in "${ROCSHMEM_SOURCE_DIR}/cmake/rocshmem_config.h.in")
    if(ENABLE_ROCSHMEM_GIN AND EXISTS "${_rocshmem_config_in}")
        file(STRINGS "${_rocshmem_config_in}" _constmem_gate REGEX "cmakedefine[ \t]+USE_CONSTMEM")
        if(NOT _constmem_gate)
            message(FATAL_ERROR
                "rocSHMEM source at ${ROCSHMEM_SOURCE_DIR} predates the USE_CONSTMEM "
                "gate (cmake/rocshmem_config.h.in lacks '#cmakedefine USE_CONSTMEM'). "
                "rccl's GIN build no longer copies constmem.hpp and requires the gate. "
                "Bump ROCSHMEM_MONO_HASH (cmake/ROCSHMEM.cmake:27) to a commit that "
                "includes it, or point -DROCSHMEM_SOURCE_DIR at such a tree.")
        endif()
    endif()

    # Acquiring the rocSHMEM *library* (pre-built find or build-from-source) is
    # only for ENABLE_ROCSHMEM, which links librocshmem.a into librccl.so. A GIN
    # build needs nothing from that library: its host code is compiled in from
    # ROCSHMEM_SOURCE_DIR (set above) and routed through rccl's own shims, and its
    # device code is header-only (no rocshmem device bitcode). So GIN skips both
    # paths below — no find_package, no ExternalProject, no librocshmem build.
    if(NOT ENABLE_ROCSHMEM)
        set(ROCSHMEM_SOURCE_DIR "${ROCSHMEM_SOURCE_DIR}" CACHE INTERNAL "rocSHMEM source directory")
        return()
    endif()

    # -----------------------------------------------------------------
    # Path 1: Pre-built rocSHMEM installation (ROCSHMEM_INSTALL_DIR)
    # -----------------------------------------------------------------
    if(ROCSHMEM_INSTALL_DIR)
        list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
        # Set rocshmem_static_ROOT; variable read on enter, so must be set BEFORE find_package()
        if(NOT rocshmem_static_ROOT)
          set(rocshmem_static_ROOT "${ROCSHMEM_INSTALL_DIR}")
        endif()
        find_package(rocshmem_static)
        if(rocshmem_static_FOUND)
            # This path returns before the -DASAN=ON propagation below, so an
            # ASAN rccl links whatever the pre-built tree happens to contain.
            if(BUILD_ADDRESS_SANITIZER)
                message(WARNING
                    "ASAN build using pre-built rocSHMEM at ${ROCSHMEM_INSTALL_DIR}. "
                    "It is linked into librccl.so as-is; if it was not itself built "
                    "with -DASAN=ON, rccl gets an uninstrumented rocSHMEM inside an "
                    "instrumented library. Unset ROCSHMEM_INSTALL_DIR to build it "
                    "from source with ASAN propagated.")
            endif()
            set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_INCLUDE_DIR}" PARENT_SCOPE)
            set(ROCSHMEM_LIBRARY     "${ROCSHMEM_LIBRARY}"      PARENT_SCOPE)
            set(ROCSHMEM_SOURCE_DIR  "${ROCSHMEM_SOURCE_DIR}"   CACHE INTERNAL "rocSHMEM source directory")
            return()
        endif()
    endif()

    # -----------------------------------------------------------------
    # Path 2: Build from source when no pre-built install is available.
    # -----------------------------------------------------------------
    if(ENABLE_ROCSHMEM)
        set(_rccl_root           "${CMAKE_SOURCE_DIR}")
        set(ROCSHMEM_INSTALL_DIR "${_rccl_root}/ext/rocshmem")
        # Convert cmake list separators for shell command safety.
        # GPU_TARGETS semicolons would be treated as bash command separators
        # inside the bash -lc "..." string. Use commas instead — rocshmem's
        # cmake converts commas back to semicolons.
        string(REPLACE ";" " " _rocshmem_cmake_opts "${ROCSHMEM_CMAKE_OPTIONS}")
        string(REPLACE ";" "," _rocshmem_gpu_targets "${GPU_TARGETS}")
        # librocshmem.a is linked into librccl.so, so without this an ASAN
        # build gets an uninstrumented rocSHMEM inside an instrumented library.
        set(_rocshmem_asan_opt "")
        if(BUILD_ADDRESS_SANITIZER)
            set(_rocshmem_asan_opt "-DASAN=ON")
        endif()
        message(STATUS "rocSHMEM: building from ${ROCSHMEM_SOURCE_DIR}")

        # Both of these are named by absolute path as graph inputs -- the
        # archive by target_link_libraries, the header by the copy_files
        # custom command on the GIN path -- so ninja needs an edge producing
        # them; add_dependencies() gives only an order-only edge.
        ExternalProject_Add(rocshmem_ext
            SOURCE_DIR          "${ROCSHMEM_SOURCE_DIR}"
            INSTALL_DIR         "${ROCSHMEM_INSTALL_DIR}"
            BUILD_BYPRODUCTS    "${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a"
                                "${ROCSHMEM_INSTALL_DIR}/include/rocshmem/rocshmem_config.h"
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
                && ${CMAKE_COMMAND} -E chdir build bash -lc "INSTALL_PREFIX=${ROCSHMEM_INSTALL_DIR} ../scripts/build_configs/gda -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DUSE_EXTERNAL_MPI=OFF -DGDA_MLX5=ON -DGDA_BNXT=ON -DGDA_IONIC=ON -DBUILD_EXAMPLES=OFF -DBUILD_FUNCTIONAL_TESTS=OFF -DBUILD_UNIT_TESTS=OFF -DBUILD_CTESTS=OFF -DBUILD_TOOLS=OFF -DGPU_TARGETS=${_rocshmem_gpu_targets} ${_rocshmem_asan_opt} ${_rocshmem_cmake_opts} "
            INSTALL_COMMAND ""
        )

        # Rebuild when the sub-build's options change (e.g. ASAN): its build
        # stamp ignores BUILD_COMMAND. The stamp sits beside the shared install,
        # so the other config tree overwriting librocshmem.a invalidates it too.
        set(_rocshmem_opts_stamp "${ROCSHMEM_INSTALL_DIR}/rccl_build_options.txt")
        file(GENERATE OUTPUT "${_rocshmem_opts_stamp}"
             CONTENT "${CMAKE_BUILD_TYPE} ${_rocshmem_gpu_targets} ${_rocshmem_asan_opt} ${_rocshmem_cmake_opts}\n")
        ExternalProject_Add_StepDependencies(rocshmem_ext build "${_rocshmem_opts_stamp}")

        set(ROCSHMEM_INSTALL_DIR "${ROCSHMEM_INSTALL_DIR}"          PARENT_SCOPE)
        set(ROCSHMEM_INCLUDE_DIR "${ROCSHMEM_INSTALL_DIR}/include"  PARENT_SCOPE)
        set(ROCSHMEM_LIBRARY     "${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a" PARENT_SCOPE)

        add_custom_target(rocshmem_static ALL DEPENDS rocshmem_ext)
    endif()

    # CACHE INTERNAL (not PARENT_SCOPE): SOURCE_DIR is consumed by the
    # top-level CMakeLists.txt install rules, outside add_subdirectory(src).
    set(ROCSHMEM_SOURCE_DIR "${ROCSHMEM_SOURCE_DIR}" CACHE INTERNAL "rocSHMEM source directory")

endfunction()

# copy files in the list-valued variable ${FILE_LIST_VAR} from ${SRC_DIR} to ${DST_DIR}
# appends the destination file location to the list-valued variable ${OUTPUT_LIST_VAR}
function(copy_files FILE_LIST_VAR SRC_DIR DST_DIR OUTPUT_LIST_VAR)
  foreach(file_name ${${FILE_LIST_VAR}})
    set(src_file "${SRC_DIR}/${file_name}")
    set(dst_file "${DST_DIR}/${file_name}")
    get_filename_component(dst_file_dir "${dst_file}" DIRECTORY)
    add_custom_command(
      OUTPUT "${dst_file}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${dst_file_dir}"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${src_file}" "${dst_file}"
      # Re-root the rocSHMEM headers' own includes to the nccl_device/ base so the
      # device-API path resolves from -I<prefix>/include alone (same convention as
      # the other GIN backends), needing no extra -I<prefix>/.../rocshmem_gda on a
      # consumer. These headers are copied verbatim and assume rocshmem_gda/ as the
      # include root; rewrite the rooted families (rocshmem/, gda/, containers/,
      # and the top-level *.hpp) while leaving relative-to-self includes
      # (queue_pair_mux.hpp, queue_pair/...) alone. Each rule anchors on the
      # #include token, so re-running on an already-rewritten file is a no-op.
      COMMAND sed -i
        -e "s|#include \"log\\.hpp\"|#include \"nccl_device/gin/rocshmem_gda/log.hpp\"|g"
        -e "s|#include \"rocshmem/rocshmem_config\\.h\"|#include \"nccl_device/gin/rocshmem_gda/gda_config.h\"|g"
        -e "s|#include \"gda/|#include \"nccl_device/gin/rocshmem_gda/gda/|g"
        -e "s|#include \"containers/|#include \"nccl_device/gin/rocshmem_gda/containers/|g"
        -e "s|#include \"bit\\.hpp\"|#include \"nccl_device/gin/rocshmem_gda/bit.hpp\"|g"
        -e "s|#include \"constants\\.hpp\"|#include \"nccl_device/gin/rocshmem_gda/constants.hpp\"|g"
        -e "s|#include \"gda_enums\\.hpp\"|#include \"nccl_device/gin/rocshmem_gda/gda/gda_enums.hpp\"|g"
        -e "s|#include \"util\\.hpp\"|#include \"nccl_device/gin/rocshmem_gda/util.hpp\"|g"
        "${dst_file}"
      DEPENDS "${src_file}"
      COMMENT "Copying ${src_file} -> ${dst_file}"
      VERBATIM
    )
    list(APPEND ${OUTPUT_LIST_VAR} "${dst_file}")
  endforeach()
  set(${OUTPUT_LIST_VAR} ${${OUTPUT_LIST_VAR}} PARENT_SCOPE)
endfunction()
