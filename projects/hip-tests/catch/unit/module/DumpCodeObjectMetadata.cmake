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
# Some LLVM installs ship only the llvm-readobj binary and alias llvm-readelf to
# it via a symlink that is not always materialized.
find_program(LLVM_READELF NAMES llvm-readelf llvm-readobj
  HINTS "${_bindir}" "${_bindir}/../llvm/bin" "${_bindir}/../lib/llvm/bin")
if(NOT OFFLOAD_BUNDLER OR NOT LLVM_READELF)
  message(FATAL_ERROR
    "DumpCodeObjectMetadata: could not locate clang-offload-bundler / "
    "llvm-readelf (or llvm-readobj) near ${_bindir}")
endif()

execute_process(
  COMMAND "${OFFLOAD_BUNDLER}" --type=o "--input=${CODE_OBJECT}" --list
  OUTPUT_VARIABLE _targets
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE _rc)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "DumpCodeObjectMetadata: failed to list ${CODE_OBJECT}")
endif()
# --list on Windows terminates entries with CRLF, and a carriage return carried
# into --targets makes the unbundle step reject the target name.
string(REGEX REPLACE "\r?\n" ";" _targets "${_targets}")

# Build the dump under a temporary name so an aborted run cannot leave a
# truncated file that is newer than CODE_OBJECT and therefore looks up to date.
set(_tmp "${OUTPUT}.tmp")
file(WRITE "${_tmp}" "")
set(_dumped 0)
foreach(_t IN LISTS _targets)
  if(NOT _t MATCHES "amdgcn")
    continue()
  endif()
  string(MAKE_C_IDENTIFIER "${_t}" _safe)
  set(_elf "${_tmp}.${_safe}.elf")
  execute_process(
    COMMAND "${OFFLOAD_BUNDLER}" --type=o "--targets=${_t}"
            "--input=${CODE_OBJECT}" "--output=${_elf}" --unbundle
    RESULT_VARIABLE _urc)
  if(NOT _urc EQUAL 0)
    file(REMOVE "${_elf}" "${_tmp}")
    message(FATAL_ERROR
      "DumpCodeObjectMetadata: failed to unbundle ${_t} from ${CODE_OBJECT}")
  endif()
  execute_process(COMMAND "${LLVM_READELF}" --notes "${_elf}"
                  OUTPUT_VARIABLE _notes
                  RESULT_VARIABLE _rrc)
  file(REMOVE "${_elf}")
  if(NOT _rrc EQUAL 0 OR NOT _notes MATCHES "AMDGPU Metadata")
    file(REMOVE "${_tmp}")
    message(FATAL_ERROR
      "DumpCodeObjectMetadata: no AMDGPU metadata note in ${_t}")
  endif()
  file(APPEND "${_tmp}" "# ===== ${_t} =====\n${_notes}")
  math(EXPR _dumped "${_dumped} + 1")
endforeach()

# --list reports success with no output when CODE_OBJECT is not a bundle, so an
# empty dump has to be caught here rather than left for the test to trip over.
if(_dumped EQUAL 0)
  file(REMOVE "${_tmp}")
  message(FATAL_ERROR
    "DumpCodeObjectMetadata: ${CODE_OBJECT} has no amdgcn bundle entries")
endif()
file(RENAME "${_tmp}" "${OUTPUT}")
