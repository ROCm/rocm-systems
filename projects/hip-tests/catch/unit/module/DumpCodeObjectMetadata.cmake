# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Dump AMDGPU code-object metadata from a HIP fat binary into a text file.
# cmake -DHIP_COMPILER=<hipcc> -DCODE_OBJECT=<x.code> -DOUTPUT=<x.metadata> -P <this>

if(NOT HIP_COMPILER OR NOT CODE_OBJECT OR NOT OUTPUT)
  message(FATAL_ERROR
    "DumpCodeObjectMetadata: HIP_COMPILER, CODE_OBJECT and OUTPUT are required")
endif()

get_filename_component(_bindir "${HIP_COMPILER}" DIRECTORY)
find_program(OFFLOAD_BUNDLER clang-offload-bundler
  HINTS "${_bindir}" "${_bindir}/../llvm/bin" "${_bindir}/../lib/llvm/bin")
find_program(LLVM_READELF NAMES llvm-readelf
  HINTS "${_bindir}" "${_bindir}/../llvm/bin" "${_bindir}/../lib/llvm/bin")
if(NOT OFFLOAD_BUNDLER OR NOT LLVM_READELF)
  message(FATAL_ERROR
    "DumpCodeObjectMetadata: could not locate clang-offload-bundler / "
    "llvm-readelf near ${_bindir}")
endif()

# List bundle targets; unbundle each amdgcn one and dump its ELF notes.
execute_process(
  COMMAND "${OFFLOAD_BUNDLER}" --type=o "--input=${CODE_OBJECT}" --list
  OUTPUT_VARIABLE _targets
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "DumpCodeObjectMetadata: failed to list ${CODE_OBJECT}")
endif()
string(REPLACE "\n" ";" _targets "${_targets}")

file(WRITE "${OUTPUT}" "")
foreach(_t IN LISTS _targets)
  if(NOT _t MATCHES "amdgcn")
    continue()
  endif()
  string(MAKE_C_IDENTIFIER "${_t}" _safe)
  set(_elf "${OUTPUT}.${_safe}.elf")
  execute_process(
    COMMAND "${OFFLOAD_BUNDLER}" --type=o "--targets=${_t}"
            "--input=${CODE_OBJECT}" "--output=${_elf}" --unbundle
    RESULT_VARIABLE _urc)
  if(_urc EQUAL 0)
    execute_process(COMMAND "${LLVM_READELF}" --notes "${_elf}"
                    OUTPUT_VARIABLE _notes)
    file(APPEND "${OUTPUT}" "; ===== ${_t} =====\n${_notes}")
    file(REMOVE "${_elf}")
  endif()
endforeach()
