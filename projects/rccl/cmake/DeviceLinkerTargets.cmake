# cmake/DeviceLinkerTargets.cmake
#
# Splits GPU_TARGETS into the two forms the device-linker pipeline needs: the
# bare processor name, and the target ID including the feature string.
#
# Kept separate from DeviceLinker.cmake so that the split is unit-testable.

# dl_parse_gpu_targets(TARGETS <list>
#                      BARE_VAR <var> FLAGS_VAR <var> ID_PREFIX <prefix>)
#
# Sets, in the caller's scope:
#   <BARE_VAR>          bare processor names, in input order
#   <FLAGS_VAR>         one --offload-arch=<target id> per entry
#   <ID_PREFIX><bare>   the full target ID for that processor
function(dl_parse_gpu_targets)
  cmake_parse_arguments(P "" "BARE_VAR;FLAGS_VAR;ID_PREFIX" "TARGETS" ${ARGN})

  set(_bare "")
  set(_flags "")
  foreach(_gpu_raw ${P_TARGETS})
    string(REGEX REPLACE ":.*" "" _gpu "${_gpu_raw}")
    list(APPEND _bare "${_gpu}")
    list(APPEND _flags "--offload-arch=${_gpu_raw}")
    set(${P_ID_PREFIX}${_gpu} "${_gpu_raw}" PARENT_SCOPE)
  endforeach()

  # We allow only one target ID for a given processor.
  set(_unique "${_bare}")
  list(REMOVE_DUPLICATES _unique)
  list(LENGTH _bare _n_all)
  list(LENGTH _unique _n_unique)
  if(NOT _n_all EQUAL _n_unique)
    message(FATAL_ERROR
      "GPU_TARGETS names more than one target ID for the same processor: "
      "${P_TARGETS}. The device linker keys its CMake targets and output "
      "directories on the bare processor name, so it can build only one "
      "variant of each. Choose one per processor.")
  endif()

  set(${P_BARE_VAR} "${_bare}" PARENT_SCOPE)
  set(${P_FLAGS_VAR} "${_flags}" PARENT_SCOPE)
endfunction()
