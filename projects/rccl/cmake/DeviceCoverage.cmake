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
