# GPU target selection for device AddressSanitizer builds. Device ASAN needs
# xnack, which gfx9 must request explicitly, gfx1250 has always on (and rejects
# the suffix), and no other family has. Cases are pinned in
# tools/scripts/test_runner/tests/test_asan_gpu_targets.py;
# projects/rocshmem/CMakeLists.txt repeats the policy inline, being a separate
# cmake process.

# Parts where xnack is always on, so naming it is an error.
set(_RCCL_ASAN_XNACK_ALWAYS_ON gfx1250 gfx1250-strict)

# rccl_asan_adjust_gpu_targets(OUT_VAR gfx942 gfx1100 gfx1250)
#   -> OUT_VAR = gfx942:xnack+;gfx1250
# Uninstrumentable targets are dropped rather than left to produce an
# uninstrumented image inside an ASAN build.
function(rccl_asan_adjust_gpu_targets OUT_VAR)
  set(_result "")
  set(_dropped "")

  foreach(_target IN LISTS ARGN)
    if(_target STREQUAL "")
      continue()
    endif()

    # A requested suffix says what the caller wants, not what the part
    # supports, so classify on the bare name.
    string(REGEX REPLACE ":.*" "" _base "${_target}")
    # list(FIND) not IN_LIST: the test loads this via cmake -P, where CMP0057
    # is unset.
    list(FIND _RCCL_ASAN_XNACK_ALWAYS_ON "${_base}" _always_on)

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

  # Both xnack spellings of one arch normalise together, and a duplicate
  # becomes a second add_library() in DeviceLinker.cmake.
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

  set(${OUT_VAR} "${_result}" PARENT_SCOPE)
endfunction()
