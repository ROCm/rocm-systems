# Flags used to probe whether the selected compiler can emit LLVM source-based
# coverage instrumentation for the amdgcn device target.
set(RCCL_DEVICE_COVERAGE_PROBE_FLAGS
    --target=amdgcn-amd-amdhsa -fprofile-instr-generate -fcoverage-mapping)

# Relative paths (under the compiler resource dir) where the amdgcn LLVM profile
# runtime may live, in priority order. Hoisted here so the search set lives in
# one place instead of inline string literals in the lookup loop.
set(RCCL_DEVICE_PROFILE_RUNTIME_RELPATHS
    "lib/amdgcn-amd-amdhsa/libclang_rt.profile.a"
    "lib/linux/libclang_rt.profile-amdgcn.a")

set(RCCL_HOST_ROCM_PROFILE_RUNTIME_NAME
    "libclang_rt.profile_rocm.a")


# Report whether <compiler> accepts the device coverage instrumentation flags.
# Sets <output_supported> to TRUE/FALSE and, when FALSE, <output_reason> to a
# human-readable diagnostic.
function(rccl_compiler_supports_device_coverage compiler output_supported output_reason)
  execute_process(
    COMMAND "${compiler}" ${RCCL_DEVICE_COVERAGE_PROBE_FLAGS}
            "-###" -x c++ /dev/null
    OUTPUT_QUIET
    ERROR_VARIABLE coverage_error
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE coverage_result)

  if(NOT coverage_result EQUAL 0)
    set(${output_supported} FALSE PARENT_SCOPE)
    set(${output_reason}
      "the selected compiler '${compiler}' rejected device coverage flags: ${coverage_error}"
      PARENT_SCOPE)
    return()
  endif()

  set(${output_supported} TRUE PARENT_SCOPE)
  set(${output_reason} "" PARENT_SCOPE)
endfunction()


# Locate the amdgcn LLVM profile runtime archive for <compiler>. Sets
# <output_runtime> to the archive path (empty when unavailable) and, when it is
# empty, <output_reason> to a human-readable diagnostic.
function(rccl_find_device_profile_runtime compiler output_runtime output_reason)
  rccl_compiler_supports_device_coverage("${compiler}" supported reason)
  if(NOT supported)
    set(${output_runtime} "" PARENT_SCOPE)
    set(${output_reason} "${reason}" PARENT_SCOPE)
    return()
  endif()

  execute_process(
    COMMAND "${compiler}" --target=amdgcn-amd-amdhsa -print-resource-dir
    OUTPUT_VARIABLE resource_dir
    ERROR_VARIABLE resource_error
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE resource_result)

  if(NOT resource_result EQUAL 0 OR resource_dir STREQUAL "")
    set(${output_runtime} "" PARENT_SCOPE)
    set(${output_reason}
      "'${compiler} --target=amdgcn-amd-amdhsa -print-resource-dir' failed: ${resource_error}"
      PARENT_SCOPE)
    return()
  endif()

  set(runtime_candidates "")
  foreach(relpath IN LISTS RCCL_DEVICE_PROFILE_RUNTIME_RELPATHS)
    list(APPEND runtime_candidates "${resource_dir}/${relpath}")
  endforeach()

  foreach(candidate IN LISTS runtime_candidates)
    if(EXISTS "${candidate}")
      set(${output_runtime} "${candidate}" PARENT_SCOPE)
      set(${output_reason} "" PARENT_SCOPE)
      return()
    endif()
  endforeach()

  string(REPLACE ";" ", " searched "${runtime_candidates}")
  set(${output_runtime} "" PARENT_SCOPE)
  set(${output_reason}
    "the selected compiler '${compiler}' has no amdgcn profile runtime; searched: ${searched}"
    PARENT_SCOPE)
endfunction()


# Find the optional self-contained host ROCm profile runtime used by newer
# compiler-rt packages. Older packages include the ROCm collection objects in
# the generic profile runtime, so an absent archive is not an error.
function(rccl_find_host_rocm_profile_runtime compiler output_runtime)
  execute_process(
    COMMAND "${compiler}" "-print-file-name=${RCCL_HOST_ROCM_PROFILE_RUNTIME_NAME}"
    OUTPUT_VARIABLE runtime
    ERROR_QUIET
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE runtime_result)

  if(runtime_result EQUAL 0 AND IS_ABSOLUTE "${runtime}" AND EXISTS "${runtime}")
    set(${output_runtime} "${runtime}" PARENT_SCOPE)
  else()
    set(${output_runtime} "" PARENT_SCOPE)
  endif()
endfunction()


# Resolve ENABLE_FULL_COVERAGE=AUTO|ON|OFF against the availability blockers.
# Sets <output_resolved> to ON or OFF and <output_blocker> to the reason the
# request cannot be honoured (empty when it can). ON plus a blocker is a
# FATAL_ERROR; AUTO plus a blocker resolves to OFF so the caller can fall back.
# The caller FORCEs AUTO's answer into the ENABLE_FULL_COVERAGE cache entry so
# cmake -LAH matches the build; this function does not write the cache.
function(rccl_resolve_full_coverage
    request
    device_linker_enabled
    rocm_version_ok
    rocm_version
    rocm_pretty
    capability_error
    output_resolved
    output_blocker)
  set(_blocker "")
  if(NOT device_linker_enabled)
    set(_blocker "the device linker is disabled")
  elseif(NOT rocm_version_ok)
    set(_blocker "the ROCm version could not be parsed")
  elseif(rocm_version VERSION_LESS "71500")
    set(_blocker
      "it requires ROCm 7.15 or newer, and this is ${rocm_pretty}")
  elseif(capability_error)
    set(_blocker
      "the selected compiler does not support it: ${capability_error}")
  endif()

  if(request STREQUAL "AUTO")
    if(_blocker)
      set(_resolved OFF)
    else()
      set(_resolved ON)
    endif()
  elseif(request STREQUAL "ON" AND _blocker)
    message(FATAL_ERROR
      "ENABLE_FULL_COVERAGE=ON is unavailable because ${_blocker}.")
  else()
    set(_resolved "${request}")
  endif()

  set(${output_resolved} "${_resolved}" PARENT_SCOPE)
  set(${output_blocker} "${_blocker}" PARENT_SCOPE)
endfunction()
