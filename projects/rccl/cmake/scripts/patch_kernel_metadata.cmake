# cmake/scripts/patch_kernel_metadata.cmake
#
# Scan callee metadata sidecar files and patch the dispatcher kernel
# assembly so the kernel descriptors and YAML .note metadata correctly
# account for indirectly-called callee resource requirements.
#
# Each callee TU's strip_kernel.py produces a .meta sidecar with lines:
#     .set kd.next_free_vgpr, <N>
#     .set kd.accum_offset, <N>
#     .set kd.next_free_sgpr, <N>
#     .set kd.private_segment_fixed_size, <N>
#
# These are the actual values from the callee's .amdhsa_kernel block.
# On gfx9+, VGPRs and AGPRs share the same file:
#     num_vgpr  = accum_offset
#     num_agpr  = next_free_vgpr - accum_offset
#     total     = next_free_vgpr
#
# This script:
#   1. Collects maximums across all callee sidecars.
#   2. Patches the .set amdgpu.max_num_{vgpr,agpr,sgpr} definitions.
#   3. Resolves forward references to those symbols in max() expressions.
#   4. Patches .amdhsa_private_segment_fixed_size for IFC kernels.
#   5. Patches YAML .note metadata for IFC kernels.
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
set(_max_private_seg 0)

foreach(_mf ${_meta_files})
  if(NOT EXISTS "${_mf}")
    message(WARNING "Sidecar not found: ${_mf}")
    continue()
  endif()
  file(STRINGS "${_mf}" _lines)
  # Parse per-file KD values
  set(_nfv 0)
  set(_ao 0)
  set(_nfs 0)
  set(_ps 0)
  foreach(_line ${_lines})
    if(_line MATCHES "kd\\.next_free_vgpr,[ \t]*([0-9]+)")
      set(_nfv ${CMAKE_MATCH_1})
    elseif(_line MATCHES "kd\\.accum_offset,[ \t]*([0-9]+)")
      set(_ao ${CMAKE_MATCH_1})
    elseif(_line MATCHES "kd\\.next_free_sgpr,[ \t]*([0-9]+)")
      set(_nfs ${CMAKE_MATCH_1})
    elseif(_line MATCHES "kd\\.private_segment_fixed_size,[ \t]*([0-9]+)")
      set(_ps ${CMAKE_MATCH_1})
    endif()
  endforeach()
  # Derive per-file VGPR/AGPR: accum_offset = num VGPRs, remainder = AGPRs
  set(_file_vgpr ${_ao})
  if(_nfv GREATER _ao)
    math(EXPR _file_agpr "${_nfv} - ${_ao}")
  else()
    set(_file_agpr 0)
  endif()
  # Track independent maximums
  if(_file_vgpr GREATER _max_vgpr)
    set(_max_vgpr ${_file_vgpr})
  endif()
  if(_file_agpr GREATER _max_agpr)
    set(_max_agpr ${_file_agpr})
  endif()
  if(_nfs GREATER _max_sgpr)
    set(_max_sgpr ${_nfs})
  endif()
  if(_ps GREATER _max_private_seg)
    set(_max_private_seg ${_ps})
  endif()
endforeach()

# Compute combined next_free_vgpr = max_vgpr + max_agpr (accum_offset + agprs)
math(EXPR _max_next_free_vgpr "${_max_vgpr} + ${_max_agpr}")

message(STATUS "Callee KD maximums: VGPR=${_max_vgpr} AGPR=${_max_agpr} next_free_vgpr=${_max_next_free_vgpr} SGPR=${_max_sgpr} PRIVATE_SEG=${_max_private_seg}")

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

# -- Step 1b: Patch .amdhsa_private_segment_fixed_size for IFC kernels.
if(_max_private_seg GREATER 0)
  string(REGEX MATCHALL
    "\\.amdhsa_private_segment_fixed_size[ \t]+[0-9]+([\r\n].*\\.amdhsa_uses_dynamic_stack[ \t]+1)"
    _matches "${_asm}")
  foreach(_m ${_matches})
    string(REGEX MATCH "\\.amdhsa_private_segment_fixed_size[ \t]+([0-9]+)" _sub "${_m}")
    set(_old_ps ${CMAKE_MATCH_1})
    if(_max_private_seg GREATER _old_ps)
      string(REGEX REPLACE
        "(\\.amdhsa_private_segment_fixed_size[ \t]+)[0-9]+"
        "\\1${_max_private_seg}" _new_m "${_m}")
      string(REPLACE "${_m}" "${_new_m}" _asm "${_asm}")
    endif()
  endforeach()
endif()

# -- Step 2: Resolve forward/undefined references to amdgpu.max_num_* symbols.
#    Replace the symbol name with its literal value inside max() call-sites.
string(REPLACE "amdgpu.max_num_vgpr)" "${_max_vgpr})" _asm "${_asm}")
string(REPLACE "amdgpu.max_num_agpr)" "${_max_agpr})" _asm "${_asm}")
string(REPLACE "amdgpu.max_num_sgpr)" "${_max_sgpr})" _asm "${_asm}")
# named_barrier is not in the KD; default to 0
string(REPLACE "amdgpu.max_num_named_barrier)" "0)" _asm "${_asm}")

# -- Step 3: Patch .note YAML metadata for IFC kernels.
#    Kernel entries with .uses_dynamic_stack: true have indirect calls whose
#    register usage is not reflected in the YAML .vgpr_count / .agpr_count /
#    .sgpr_count fields.  Update to max(kernel_value, callee_max).

set(_search_pos 0)
string(LENGTH "${_asm}" _asm_len)
set(_yaml_patch_count 0)

while(1)
  string(SUBSTRING "${_asm}" ${_search_pos} -1 _tail)
  string(FIND "${_tail}" ".uses_dynamic_stack: true" _rel_uds_pos)
  if(_rel_uds_pos EQUAL -1)
    break()
  endif()
  math(EXPR _uds_pos "${_search_pos} + ${_rel_uds_pos}")

  # Find the start of this kernel entry
  string(SUBSTRING "${_asm}" 0 ${_uds_pos} _before_uds)
  string(FIND "${_before_uds}" "  - .agpr_count:" _entry_start REVERSE)
  if(_entry_start EQUAL -1)
    math(EXPR _search_pos "${_uds_pos} + 25")
    continue()
  endif()

  # Find the end of this entry
  math(EXPR _after_uds "${_uds_pos} + 25")
  string(SUBSTRING "${_asm}" ${_after_uds} -1 _after_tail)
  string(FIND "${_after_tail}" "  - .agpr_count:" _rel_next)
  if(NOT _rel_next EQUAL -1)
    math(EXPR _next_entry "${_after_uds} + ${_rel_next}")
  else()
    string(FIND "${_after_tail}" ".end_amdgpu_metadata" _rel_next)
    if(NOT _rel_next EQUAL -1)
      math(EXPR _next_entry "${_after_uds} + ${_rel_next}")
    else()
      set(_next_entry ${_asm_len})
    endif()
  endif()

  # Extract this kernel entry
  math(EXPR _entry_len "${_next_entry} - ${_entry_start}")
  string(SUBSTRING "${_asm}" ${_entry_start} ${_entry_len} _entry)

  # Extract current values and compute max(current, callee_max) for each
  string(REGEX MATCH "\\.agpr_count:[ \t]+([0-9]+)" _match "${_entry}")
  set(_cur_agpr ${CMAKE_MATCH_1})
  if(_max_agpr GREATER _cur_agpr)
    set(_new_agpr ${_max_agpr})
  else()
    set(_new_agpr ${_cur_agpr})
  endif()

  string(REGEX MATCH "\\.sgpr_count:[ \t]+([0-9]+)" _match "${_entry}")
  set(_cur_sgpr ${CMAKE_MATCH_1})
  if(_max_sgpr GREATER _cur_sgpr)
    set(_new_sgpr ${_max_sgpr})
  else()
    set(_new_sgpr ${_cur_sgpr})
  endif()

  string(REGEX MATCH "\\.vgpr_count:[ \t]+([0-9]+)" _match "${_entry}")
  set(_cur_vgpr ${CMAKE_MATCH_1})
  if(_max_next_free_vgpr GREATER _cur_vgpr)
    set(_new_vgpr ${_max_next_free_vgpr})
  else()
    set(_new_vgpr ${_cur_vgpr})
  endif()

  # Patch private_segment_fixed_size in YAML
  string(REGEX MATCH "\\.private_segment_fixed_size:[ \t]+([0-9]+)" _match "${_entry}")
  set(_cur_ps ${CMAKE_MATCH_1})
  if(_max_private_seg GREATER _cur_ps)
    set(_new_ps ${_max_private_seg})
  else()
    set(_new_ps ${_cur_ps})
  endif()

  # Patch the entry
  string(REGEX REPLACE
    "(\\.agpr_count:[ \t]+)[0-9]+" "\\1${_new_agpr}" _entry "${_entry}")
  string(REGEX REPLACE
    "(\\.sgpr_count:[ \t]+)[0-9]+" "\\1${_new_sgpr}" _entry "${_entry}")
  string(REGEX REPLACE
    "(\\.vgpr_count:[ \t]+)[0-9]+" "\\1${_new_vgpr}" _entry "${_entry}")
  string(REGEX REPLACE
    "(\\.private_segment_fixed_size:[ \t]+)[0-9]+" "\\1${_new_ps}" _entry "${_entry}")

  # Splice the patched entry back into the assembly
  string(SUBSTRING "${_asm}" 0 ${_entry_start} _prefix)
  math(EXPR _suffix_start "${_entry_start} + ${_entry_len}")
  string(SUBSTRING "${_asm}" ${_suffix_start} -1 _suffix)
  set(_asm "${_prefix}${_entry}${_suffix}")

  # Advance past this entry
  string(LENGTH "${_prefix}${_entry}" _search_pos)
  math(EXPR _yaml_patch_count "${_yaml_patch_count} + 1")

  # Log what we patched
  string(REGEX MATCH "\\.name:[ \t]+([^\n]+)" _match "${_entry}")
  message(STATUS "  YAML patched ${CMAKE_MATCH_1}: vgpr ${_cur_vgpr}->${_new_vgpr} agpr ${_cur_agpr}->${_new_agpr} sgpr ${_cur_sgpr}->${_new_sgpr} private_seg ${_cur_ps}->${_new_ps}")
endwhile()

message(STATUS "Patched ${_yaml_patch_count} IFC kernel YAML entries")

file(WRITE "${ASM_FILE}" "${_asm}")
message(STATUS "Patched ${ASM_FILE}")
