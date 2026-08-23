function(rccl_find_device_profile_runtime compiler output_runtime output_reason)
  execute_process(
    COMMAND "${compiler}" --target=amdgcn-amd-amdhsa
            -fprofile-instr-generate -fcoverage-mapping
            "-###" -x c++ /dev/null
    OUTPUT_QUIET
    ERROR_VARIABLE coverage_error
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE coverage_result)
  if(NOT coverage_result EQUAL 0)
    set(${output_runtime} "" PARENT_SCOPE)
    set(${output_reason}
      "the selected compiler '${compiler}' rejected device coverage flags: ${coverage_error}"
      PARENT_SCOPE)
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

  set(runtime_candidates
    "${resource_dir}/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a"
    "${resource_dir}/lib/linux/libclang_rt.profile-amdgcn.a")
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
