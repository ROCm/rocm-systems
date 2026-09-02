# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Resolve HRR_CLR_LIB: the directory holding a capture-enabled libamdhip64
# (typically <clr-build>/hipamd/lib).
#
# Capture is compiled into libamdhip64 in projects/clr; playback and the archive
# reader live here in projects/hrr. Both sides share generated wire-format structs,
# so integration tests must load a libamdhip64 built from the same source commit as
# hrr-playback. A prebuilt ROCm SDK libamdhip64 generally writes older payload
# layouts, which surfaces as "payload too small" or missing kernel/code-object
# errors during replay rather than as a build failure.
#
# Set explicitly at configure time:
#   cmake ... -DHRR_CLR_LIB=/path/to/clr-build/hipamd/lib
# or export HRR_CLR_LIB in the environment before configuring.

function(hrr_resolve_capture_lib out_var)
  if(HRR_CLR_LIB)
    set(_lib_dir "${HRR_CLR_LIB}")
  elseif(NOT "$ENV{HRR_CLR_LIB}" STREQUAL "")
    set(_lib_dir "$ENV{HRR_CLR_LIB}")
  else()
    set(${out_var} "" PARENT_SCOPE)
    return()
  endif()

  get_filename_component(_lib_dir "${_lib_dir}" ABSOLUTE)
  if(NOT EXISTS "${_lib_dir}/libamdhip64.so"
      AND NOT EXISTS "${_lib_dir}/libamdhip64.so.7"
      AND NOT EXISTS "${_lib_dir}/amdhip64.dll")
    message(FATAL_ERROR
      "HRR_CLR_LIB=${_lib_dir} does not contain libamdhip64.\n"
      "Build capture-enabled amdhip64 first, for example:\n"
      "  cmake -S projects/clr -B build/clr -GNinja -DHIP_PLATFORM=amd \\\n"
      "        -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF \\\n"
      "        -DHIP_COMMON_DIR=<repo>/projects/hip -DROCM_PATH=$ROCM_PATH\n"
      "  cmake --build build/clr --target amdhip64")
  endif()

  set(${out_var} "${_lib_dir}" PARENT_SCOPE)
endfunction()

# Point a target's RUNPATH at the capture library. Capture subprocesses spawned by
# the tests pick it up via LD_LIBRARY_PATH instead; this covers the test binary and
# hrr-playback themselves.
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
