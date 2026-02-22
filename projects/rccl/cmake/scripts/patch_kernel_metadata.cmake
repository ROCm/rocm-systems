# cmake/scripts/patch_kernel_metadata.cmake
#
# Scan callee metadata sidecar files and patch the kernel assembly's
# amdgpu.max_num_{vgpr,agpr,sgpr} values so the kernel descriptor
# correctly accounts for indirectly-called functions' register usage.
#
# Each callee TU's bc→asm step produces a small .meta sidecar containing
# lines like:
#     .set amdgpu.max_num_vgpr, 128
#     .set amdgpu.max_num_agpr, 64
#     .set amdgpu.max_num_sgpr, 102
#
# This script reads all sidecars, finds the global maximum for each
# register class, and rewrites the corresponding .set directives in
# the kernel assembly file.
#
# Additionally, it resolves forward references to amdgpu.max_num_*
# symbols inside max() expressions by replacing them with literal
# values.  Some compiler versions (e.g. ROCm 7.2) crash when the
# assembler encounters forward or undefined references to these symbols
# inside .amdhsa_kernel blocks.  The kernel TU's assembly may reference
# amdgpu.max_num_named_barrier without ever defining it (it is normally
# provided at link time in a monolithic build).  Resolving these
# references to literals makes the assembly self-contained and
# assembler-version-agnostic.
#
# Usage:
#   cmake -DASM_FILE=<kernel.s> -DMANIFEST=<file-listing-meta-paths>
#         -P patch_kernel_metadata.cmake

cmake_minimum_required(VERSION 3.18)

if(NOT ASM_FILE)
  message(FATAL_ERROR "ASM_FILE is required")
endif()
if(NOT MANIFEST)
  message(FATAL_ERROR "MANIFEST is required")
endif()

file(STRINGS "${MANIFEST}" _meta_files)

set(_max_vgpr 0)
set(_max_agpr 0)
set(_max_sgpr 0)
set(_max_named_barrier 0)

foreach(_mf ${_meta_files})
  if(NOT EXISTS "${_mf}")
    message(WARNING "Sidecar not found: ${_mf}")
    continue()
  endif()
  file(STRINGS "${_mf}" _lines)
  foreach(_line ${_lines})
    if(_line MATCHES "max_num_vgpr,[ \t]*([0-9]+)")
      if(CMAKE_MATCH_1 GREATER _max_vgpr)
        set(_max_vgpr ${CMAKE_MATCH_1})
      endif()
    elseif(_line MATCHES "max_num_agpr,[ \t]*([0-9]+)")
      if(CMAKE_MATCH_1 GREATER _max_agpr)
        set(_max_agpr ${CMAKE_MATCH_1})
      endif()
    elseif(_line MATCHES "max_num_sgpr,[ \t]*([0-9]+)")
      if(CMAKE_MATCH_1 GREATER _max_sgpr)
        set(_max_sgpr ${CMAKE_MATCH_1})
      endif()
    elseif(_line MATCHES "max_num_named_barrier,[ \t]*([0-9]+)")
      if(CMAKE_MATCH_1 GREATER _max_named_barrier)
        set(_max_named_barrier ${CMAKE_MATCH_1})
      endif()
    endif()
  endforeach()
endforeach()

message(STATUS "Callee register maximums: VGPR=${_max_vgpr} AGPR=${_max_agpr} SGPR=${_max_sgpr} NAMED_BARRIER=${_max_named_barrier}")

file(READ "${ASM_FILE}" _asm)

# -- Step 1: Patch the .set definitions for amdgpu.max_num_{vgpr,agpr,sgpr}.
#    These definitions (at the bottom of the file) set the module-wide maximums
#    that the kernel descriptor expressions reference.
string(REGEX REPLACE
  "([\t ]+\\.set[\t ]+amdgpu\\.max_num_vgpr,[\t ]*)[0-9]+"
  "\\1${_max_vgpr}" _asm "${_asm}")
string(REGEX REPLACE
  "([\t ]+\\.set[\t ]+amdgpu\\.max_num_agpr,[\t ]*)[0-9]+"
  "\\1${_max_agpr}" _asm "${_asm}")
string(REGEX REPLACE
  "([\t ]+\\.set[\t ]+amdgpu\\.max_num_sgpr,[\t ]*)[0-9]+"
  "\\1${_max_sgpr}" _asm "${_asm}")

# -- Step 2: Resolve forward/undefined references to amdgpu.max_num_* symbols.
#    Per-kernel .set directives use expressions like:
#        .set KERNEL.num_vgpr, max(62, amdgpu.max_num_vgpr)
#    The amdgpu.max_num_* symbols are defined AFTER the .amdhsa_kernel blocks
#    (or not at all, for named_barrier).  Some assembler versions cannot handle
#    these forward/undefined references.  We replace the symbol name with its
#    literal value inside max() call-sites.  The trailing ')' ensures we only
#    match operand positions, not the .set definition lines (which end with
#    a comma-separated value, not a closing paren).
string(REPLACE "amdgpu.max_num_vgpr)" "${_max_vgpr})" _asm "${_asm}")
string(REPLACE "amdgpu.max_num_agpr)" "${_max_agpr})" _asm "${_asm}")
string(REPLACE "amdgpu.max_num_sgpr)" "${_max_sgpr})" _asm "${_asm}")
string(REPLACE "amdgpu.max_num_named_barrier)" "${_max_named_barrier})" _asm "${_asm}")

file(WRITE "${ASM_FILE}" "${_asm}")
message(STATUS "Patched ${ASM_FILE}")
