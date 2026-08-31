# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Resolve HRR_CLR_LIB: directory containing capture-enabled libamdhip64 (hipamd/lib).
#
# Capture lives in projects/clr and is compiled into libamdhip64. Playback and the
# archive reader live in projects/hrr. Integration tests must load the same-commit
# capture library for both capture subprocesses and hrr-playback replay — a prebuilt
# ROCm SDK libamdhip64 often writes older wire-format payloads and fails replay.
#
# Set explicitly:
#   cmake ... -DHRR_CLR_LIB=/path/to/clr/build/hipamd/lib
# or export HRR_CLR_LIB before configuring.

function(hrr_resolve_capture_lib out_var)
  set(_lib_dir "")
  if(DEFINED HRR_CLR_LIB AND HRR_CLR_LIB)
    set(_lib_dir "${HRR_CLR_LIB}")
  elseif(DEFINED ENV{HRR_CLR_LIB} AND NOT "$ENV{HRR_CLR_LIB}" STREQUAL "")
    set(_lib_dir "$ENV{HRR_CLR_LIB}")
  else()
    # Common in-tree CLR build locations (rocm-systems superbuild / sibling trees).
    get_filename_component(_hrr_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
    get_filename_component(_repo_root "${_hrr_root}/.." ABSOLUTE)
    foreach(_candidate
        "${_repo_root}/build/clr/hipamd/lib"
        "${_repo_root}/build-rocr/clr/hipamd/lib"
        "${_repo_root}/../build/clr/hipamd/lib"
        "${_repo_root}/../hrr-testing/build/clr/hipamd/lib")
      if(EXISTS "${_candidate}/libamdhip64.so"
          OR EXISTS "${_candidate}/libamdhip64.so.7"
          OR EXISTS "${_candidate}/amdhip64.dll")
        set(_lib_dir "${_candidate}")
        break()
      endif()
    endforeach()
  endif()

  if(_lib_dir)
    get_filename_component(_lib_dir "${_lib_dir}" ABSOLUTE)
    if(NOT EXISTS "${_lib_dir}/libamdhip64.so"
        AND NOT EXISTS "${_lib_dir}/libamdhip64.so.7"
        AND NOT EXISTS "${_lib_dir}/amdhip64.dll")
      message(FATAL_ERROR
        "HRR_CLR_LIB=${_lib_dir} does not contain libamdhip64. "
        "Build capture-enabled amdhip64 first:\n"
        "  cmake -S projects/clr -B build/clr -GNinja "
        "-DHIP_COMMON_DIR=projects/hip -DROCM_PATH=\$ROCM_PATH "
        "-DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd\n"
        "  ninja -C build/clr amdhip64")
    endif()
  endif()

  set(${out_var} "${_lib_dir}" PARENT_SCOPE)
endfunction()

# Apply capture-lib RUNPATH to HRR executables (capture subprocesses inherit via
# LD_LIBRARY_PATH set in the test driver; RUNPATH covers the top-level binary).
function(hrr_apply_capture_lib_rpath lib_dir)
  if(NOT lib_dir)
    return()
  endif()
  foreach(_target ${ARGN})
    if(TARGET ${_target})
      set_target_properties(${_target} PROPERTIES
        BUILD_RPATH "${lib_dir}"
        INSTALL_RPATH "${lib_dir}")
    endif()
  endforeach()
endfunction()
