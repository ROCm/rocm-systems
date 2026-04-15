# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Compile-time log group configuration.
#
# Groups can be specified by name or raw bitmask:
#   cmake -DRJ_LOG_GROUPS=VM        → group 0 only
#   cmake -DRJ_LOG_GROUPS=ALL       → all groups
#   cmake -DRJ_LOG_GROUPS=0x3       → raw bitmask (groups 0 and 1)
#   cmake -DRJ_LOG_GROUPS=OFF       → all logging off (default)
#
# The resolved bitmask is passed as -DRJ_LOG_GROUPS=<value> to the
# compiler. When OFF/0, no -D is emitted and the header defaults to
# groups=0 via constexpr initialization — disabled groups compile to nothing.
#
# Group IDs (must match util/log.h Logger::Group enum):
#   VM    (bit 0)  Kernel dispatch, instruction execution, memory access.
#
# Usage: include(rj_log) after project().

set(RJ_LOG_GROUPS "OFF" CACHE STRING
  "Log groups: OFF, ALL, or comma-separated names (VM). Raw bitmask also accepted.")

# --- Group name → bit mapping (keep in sync with Logger::Group in log.h) ---
set(_RJ_GROUP_VM   1)   # bit 0
# Future groups:
# set(_RJ_GROUP_CODE  2)  # bit 1
# set(_RJ_GROUP_KFD   4)  # bit 2

# All known group bits OR'd together.
math(EXPR _RJ_GROUP_ALL "${_RJ_GROUP_VM}")

# --- Resolve the user value to a numeric bitmask ---
set(_rj_groups_resolved 0)

if(RJ_LOG_GROUPS STREQUAL "OFF" OR RJ_LOG_GROUPS STREQUAL "0")
  # Nothing to do — _rj_groups_resolved stays 0.
elseif(RJ_LOG_GROUPS STREQUAL "ALL")
  set(_rj_groups_resolved ${_RJ_GROUP_ALL})
elseif(RJ_LOG_GROUPS MATCHES "^0x[0-9a-fA-F]+$|^[0-9]+$")
  # Raw numeric bitmask — pass through.
  set(_rj_groups_resolved ${RJ_LOG_GROUPS})
else()
  # Comma-separated group names: "VM", "VM,KFD", etc.
  string(REPLACE "," ";" _group_list "${RJ_LOG_GROUPS}")
  foreach(_group IN LISTS _group_list)
    string(STRIP "${_group}" _group)
    string(TOUPPER "${_group}" _group)
    if(DEFINED _RJ_GROUP_${_group})
      math(EXPR _rj_groups_resolved "${_rj_groups_resolved} | ${_RJ_GROUP_${_group}}")
    else()
      message(FATAL_ERROR
        "Unknown log group '${_group}'. Known groups: VM. Use OFF or ALL for all.")
    endif()
  endforeach()
endif()

if(NOT _rj_groups_resolved EQUAL 0)
  add_compile_definitions(RJ_LOG_GROUPS=${_rj_groups_resolved})
  message(STATUS "rocjitsu: log groups enabled: ${RJ_LOG_GROUPS} (bitmask=${_rj_groups_resolved})")
endif()
