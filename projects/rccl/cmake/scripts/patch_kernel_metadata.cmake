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
# Finally, it patches the .note YAML metadata (.vgpr_count, .agpr_count,
# .sgpr_count) for kernels with indirect calls (.uses_dynamic_stack: true)
# so that profiling and diagnostic tools see correct values.
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

# -- Step 3: Patch .note YAML metadata for IFC kernels.
#    Kernel entries with .uses_dynamic_stack: true have indirect calls whose
#    register usage is not reflected in the YAML .vgpr_count / .agpr_count /
#    .sgpr_count fields (the compiler only knows the kernel's own usage).
#    We update these fields to max(kernel_value, callee_max) so that profiling
#    and diagnostic tools report correct values.
#
#    YAML field order within each kernel entry is alphabetical:
#      .agpr_count  (near entry start "  - .agpr_count:")
#      ...
#      .sgpr_count
#      ...
#      .uses_dynamic_stack
#      .vgpr_count
#      ...
#    We locate each ".uses_dynamic_stack: true", then scan backward for
#    .agpr_count and .sgpr_count, and forward for .vgpr_count.

set(_search_pos 0)
string(LENGTH "${_asm}" _asm_len)
set(_yaml_patch_count 0)

while(1)
  # string(FIND) has no start-position parameter, so search within a tail substring
  string(SUBSTRING "${_asm}" ${_search_pos} -1 _tail)
  string(FIND "${_tail}" ".uses_dynamic_stack: true" _rel_uds_pos)
  if(_rel_uds_pos EQUAL -1)
    break()
  endif()
  math(EXPR _uds_pos "${_search_pos} + ${_rel_uds_pos}")

  # Find the start of this kernel entry: nearest preceding "  - .agpr_count:"
  string(SUBSTRING "${_asm}" 0 ${_uds_pos} _before_uds)
  string(FIND "${_before_uds}" "  - .agpr_count:" _entry_start REVERSE)
  if(_entry_start EQUAL -1)
    math(EXPR _search_pos "${_uds_pos} + 25")
    continue()
  endif()

  # Find the end of this entry: next "  - .agpr_count:" or ".end_amdgpu_metadata"
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
  if(_max_vgpr GREATER _cur_vgpr)
    set(_new_vgpr ${_max_vgpr})
  else()
    set(_new_vgpr ${_cur_vgpr})
  endif()

  # Patch the entry
  string(REGEX REPLACE
    "(\\.agpr_count:[ \t]+)[0-9]+" "\\1${_new_agpr}" _entry "${_entry}")
  string(REGEX REPLACE
    "(\\.sgpr_count:[ \t]+)[0-9]+" "\\1${_new_sgpr}" _entry "${_entry}")
  string(REGEX REPLACE
    "(\\.vgpr_count:[ \t]+)[0-9]+" "\\1${_new_vgpr}" _entry "${_entry}")

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
  message(STATUS "  YAML patched ${CMAKE_MATCH_1}: vgpr ${_cur_vgpr}->${_new_vgpr} agpr ${_cur_agpr}->${_new_agpr} sgpr ${_cur_sgpr}->${_new_sgpr}")
endwhile()

message(STATUS "Patched ${_yaml_patch_count} IFC kernel YAML entries")

file(WRITE "${ASM_FILE}" "${_asm}")
message(STATUS "Patched ${ASM_FILE}")
