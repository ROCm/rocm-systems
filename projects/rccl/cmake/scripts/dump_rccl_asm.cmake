# Invoked with: cmake -DRCCL_SO=<path-to-librccl.so> -DLLVM_OBJDUMP=<llvm-objdump> -P dump_rccl_asm.cmake
# RCCL_SO must be an absolute path to the linked shared library (or a symlink to it).
if(NOT DEFINED RCCL_SO OR NOT DEFINED LLVM_OBJDUMP)
  message(FATAL_ERROR "dump_rccl_asm.cmake: set RCCL_SO and LLVM_OBJDUMP")
endif()
if(NOT EXISTS "${RCCL_SO}")
  message(FATAL_ERROR "dump_rccl_asm.cmake: RCCL_SO does not exist: ${RCCL_SO}")
endif()

get_filename_component(_bindir "${RCCL_SO}" DIRECTORY)
get_filename_component(_realso "${RCCL_SO}" REALPATH)
get_filename_component(_basename "${_realso}" NAME)

file(GLOB _oldasm "${_bindir}/librccl.gfx*.s" "${_bindir}/librccl.host*.s")
if(_oldasm)
  file(REMOVE ${_oldasm})
endif()

execute_process(
  COMMAND "${LLVM_OBJDUMP}" --offloading "${_realso}"
  WORKING_DIRECTORY "${_bindir}"
  RESULT_VARIABLE _ex
  OUTPUT_VARIABLE _out
  ERROR_VARIABLE _err
)
if(NOT _ex EQUAL 0)
  message(FATAL_ERROR "llvm-objdump --offloading failed (${_ex}): ${_err}\n${_out}")
endif()

# Collect AMDGPU bundle paths with a stable group key (e.g. gfx942 and gfx942:sramecc+ -> gfx942).
set(_pairs "")
file(GLOB _cand "${_bindir}/${_basename}.*")
list(SORT _cand)
foreach(_full ${_cand})
  get_filename_component(_fn "${_full}" NAME)
  if(_fn STREQUAL _basename)
    continue()
  endif()
  if(_fn MATCHES "\\.host-")
    file(REMOVE "${_full}")
    continue()
  endif()
  if(NOT _fn MATCHES "hipv4-amdgcn")
    continue()
  endif()
  file(SIZE "${_full}" _sz)
  if(_sz EQUAL 0)
    file(REMOVE "${_full}")
    continue()
  endif()
  if(NOT _fn MATCHES "hipv4-amdgcn-amd-amdhsa--(.+)$")
    continue()
  endif()
  set(_tail "${CMAKE_MATCH_1}")
  string(REGEX REPLACE ":.*" "" _gkey "${_tail}")
  string(REGEX REPLACE "-+$" "" _gkey "${_gkey}")
  if(_gkey STREQUAL "")
    continue()
  endif()
  list(APPEND _pairs "${_gkey}@@${_full}")
endforeach()

# Unique group keys (preserve sorted order of first occurrence).
set(_keys "")
foreach(_p ${_pairs})
  string(REGEX MATCH "^([^@]+)@@" _m "${_p}")
  set(_k "${CMAKE_MATCH_1}")
  list(FIND _keys "${_k}" _fi)
  if(_fi EQUAL -1)
    list(APPEND _keys "${_k}")
  endif()
endforeach()

foreach(_k ${_keys})
  set(_out "${_bindir}/librccl.${_k}.s")
  file(WRITE "${_out}" "; librccl AMDGPU disassembly (llvm-objdump), combined slices for ${_k}\n")
  foreach(_p ${_pairs})
    string(REGEX MATCH "^([^@]+)@@(.*)$" _m "${_p}")
    set(_pk "${CMAKE_MATCH_1}")
    set(_full "${CMAKE_MATCH_2}")
    if(NOT _pk STREQUAL _k)
      continue()
    endif()
    get_filename_component(_slice "${_full}" NAME)
    message(STATUS "DUMP_ASM: librccl.${_k}.s <- ${_slice}")
    execute_process(
      COMMAND bash -c "printf '\n\n;;; offload slice: %s\n\n' \"${_slice}\" >> \"${_out}\""
      RESULT_VARIABLE _bret
    )
    if(NOT _bret EQUAL 0)
      message(FATAL_ERROR "append banner failed (${_bret})")
    endif()
    execute_process(
      COMMAND bash -c "\"${LLVM_OBJDUMP}\" -d -l --source --symbolize-operands \"${_full}\" >> \"${_out}\""
      RESULT_VARIABLE _dret
    )
    if(NOT _dret EQUAL 0)
      message(FATAL_ERROR "llvm-objdump disassembly failed for ${_slice} (${_dret})")
    endif()
    file(REMOVE "${_full}")
  endforeach()
endforeach()

execute_process(COMMAND uname -m OUTPUT_VARIABLE _hostm OUTPUT_STRIP_TRAILING_WHITESPACE)
set(_hostasm "librccl.host-${_hostm}.s")
message(STATUS "DUMP_ASM: ${_hostasm} <- ${_basename} (host ELF)")
execute_process(
  COMMAND bash -c "\"${LLVM_OBJDUMP}\" -d -l --source --symbolize-operands \"${_realso}\" >\"${_bindir}/${_hostasm}\""
  RESULT_VARIABLE _hret
)
if(NOT _hret EQUAL 0)
  message(FATAL_ERROR "llvm-objdump host disassembly failed (${_hret})")
endif()
