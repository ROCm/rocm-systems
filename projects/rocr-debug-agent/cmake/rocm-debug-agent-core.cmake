# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Defines the rocm-debug-agent-core object library and the compile options it
# shares with the rocm-debug-agent library.
#
# The core holds the modules that are linked into both the rocm-debug-agent
# library and the unit tests.  It is included by the top-level CMakeLists.txt,
# so a full build compiles the modules once and the unit tests exercise the
# same objects as the shipping library.  test/unit includes it as well when
# test/ is configured on its own, so the unit tests can be built without the
# library sources being configured.
#
# The modules must not depend on libamd-dbgapi or the HSA runtime, only on the
# amd-dbgapi headers.  The amd-dbgapi package must be found before including
# this file.

set(AGENT_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/../src)

# Check for memfd_create headers and libs
include(CheckCXXSourceCompiles)
check_cxx_source_compiles(
    "
  #define _GNU_SOURCE
  #include <sys/mman.h>
  int main() { return memfd_create (\"cmake_test\", 0); }
  "
    HAVE_MEMFD_CREATE
)

# Shared by rocm-debug-agent and rocm-debug-agent-core so the two cannot drift.
# The unit test sources reuse only AGENT_WARNING_OPTIONS: they include the
# GoogleTest headers, which must see the same RTTI setting as the prebuilt
# GoogleTest library.
set(AGENT_WARNING_OPTIONS -Werror -Wall -Wno-attributes)
set(AGENT_COMPILE_OPTIONS
    ${AGENT_WARNING_OPTIONS}
    -fno-rtti
    -fvisibility=hidden
)
set(AGENT_COMPILE_DEFINITIONS
    AMD_INTERNAL_BUILD
    _GNU_SOURCE
    __STDC_LIMIT_MACROS
    __STDC_CONSTANT_MACROS
    $<$<BOOL:${HAVE_MEMFD_CREATE}>:HAVE_MEMFD_CREATE>
)

add_library(rocm-debug-agent-core OBJECT)
target_sources(
    rocm-debug-agent-core
    PRIVATE
        ${AGENT_SOURCE_DIR}/agent_config.cpp
        ${AGENT_SOURCE_DIR}/agent_dbgapi_utils.cpp
        ${AGENT_SOURCE_DIR}/agent_utils.cpp
        ${AGENT_SOURCE_DIR}/logging.cpp
)

set_target_properties(
    rocm-debug-agent-core
    PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        POSITION_INDEPENDENT_CODE ON
        NO_SYSTEM_FROM_IMPORTED ON
)

# Only the amd-dbgapi headers are used; libamd-dbgapi is not linked.  PUBLIC
# so that targets linking the core objects also see the headers.
target_include_directories(
    rocm-debug-agent-core
    PUBLIC $<TARGET_PROPERTY:amd-dbgapi,INTERFACE_INCLUDE_DIRECTORIES>
)

target_compile_options(rocm-debug-agent-core PRIVATE ${AGENT_COMPILE_OPTIONS})
target_compile_definitions(
    rocm-debug-agent-core
    PRIVATE ${AGENT_COMPILE_DEFINITIONS}
)
