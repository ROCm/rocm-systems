# Device ASAN needs xnack: gfx9 must request it explicitly, gfx1250 has it
# always on and rejects the term, no other family has it. Mirrors
# projects/rccl/cmake/AsanGpuTargets.cmake, repeated because rccl builds
# rocSHMEM as a separate cmake process. Runs before rocm_check_target_ids,
# which rejects gfx1250:xnack+ before the policy could normalise it.
set(_ROCSHMEM_ASAN_XNACK_ALWAYS_ON gfx1250 gfx1250-strict)

function(rocshmem_asan_adjust_gpu_targets OUT_VAR)
  set(_result "")
  set(_dropped "")

  foreach(_target IN LISTS ARGN)
    if(_target STREQUAL "")
      continue()
    endif()

    # A requested suffix says what the caller wants, not what the part
    # supports, so classify on the bare name.
    string(REGEX REPLACE ":.*" "" _base "${_target}")
    list(FIND _ROCSHMEM_ASAN_XNACK_ALWAYS_ON "${_base}" _always_on)

    # Drop any xnack term the caller wrote before deciding what to do with it:
    # it may be ":xnack-", which contradicts the build. Other feature terms
    # (e.g. ":sramecc+") are part of the requested target and are kept; xnack
    # sorts last, so re-appending it keeps the target ID canonical.
    string(REGEX REPLACE ":xnack[+-]" "" _without_xnack "${_target}")

    if(_base MATCHES "^gfx9")
      list(APPEND _result "${_without_xnack}:xnack+")
    elseif(NOT _always_on EQUAL -1)
      # Always on here, so naming xnack at all is an invalid target ID.
      list(APPEND _result "${_without_xnack}")
    else()
      list(APPEND _dropped "${_target}")
    endif()
  endforeach()

  # Both xnack spellings of one arch normalise to the same string.
  if(_result)
    list(REMOVE_DUPLICATES _result)
  endif()

  if(_dropped)
    message(STATUS
      "ASAN: dropping GPU target(s) with no xnack support: ${_dropped}")
  endif()
  if(NOT _result)
    message(FATAL_ERROR
      "ASAN requires a GPU target that supports xnack (gfx9* with :xnack+, "
      "or gfx1250 where it is always on). None of the requested GPU targets "
      "do: ${ARGN}")
  endif()

  message(STATUS "ASAN: GPU targets = ${_result}")
  set(${OUT_VAR} "${_result}" PARENT_SCOPE)
endfunction()
