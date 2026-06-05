## Copyright (c) 2026 Advanced Micro Devices, Inc.
##
## Permission is hereby granted, free of charge, to any person obtaining a copy
## of this software and associated documentation files (the "Software"), to
## deal in the Software without restriction, including without limitation the
## rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
## sell copies of the Software, and to permit persons to whom the Software is
## furnished to do so, subject to the following conditions:
##
## The above copyright notice and this permission notice shall be included in
## all copies or substantial portions of the Software.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
## IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
## FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
## AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
## LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
## FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
## IN THE SOFTWARE.

# amd-dbgapi-internal: test-only seam target.
#
# The public library (amd-dbgapi) is built with hidden visibility and
# a linker version script that exports only the ~680 public C API
# symbols.  None of the internal C++ classes (process, agent, queue,
# wave, ...) are reachable from outside.  Unit tests need to instantiate
# and probe those classes directly without going through the public
# API, so we build a parallel STATIC archive from the same sources but
# with default visibility and no version script.
#
# This module is included by the top-level CMakeLists.txt when
# AMD_DBGAPI_BUILD_TESTS is ON (or one of the per-tier sub-options).
# It assumes the following variables are already in scope:
#
#   _dbgapi_lib_sources       OS-agnostic source list (set near
#                             add_library(amd-dbgapi ...))
#   _dbgapi_lib_os_sources    OS-specific source list (set in the
#                             Linux/Windows branch)
#
# Anything else (libbacktrace, amd_comgr, generated headers) is picked
# up from the public target's properties so the two targets stay in
# sync without duplicating discovery logic.

include_guard(GLOBAL)

if(NOT TARGET amd-dbgapi)
  message(FATAL_ERROR
    "amd-dbgapi-internal: the public amd-dbgapi target must be defined "
    "before including AmdDbgapiInternal.cmake")
endif()

# STATIC so we don't ship a second .so.  EXCLUDE_FROM_ALL keeps it out
# of the default build; it's only built when a test target depends on
# it.
add_library(amd-dbgapi-internal STATIC EXCLUDE_FROM_ALL
  ${_dbgapi_lib_sources}
  ${_dbgapi_lib_os_sources})

set_target_properties(amd-dbgapi-internal PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS ON
  # Position-independent code so it can be linked into shared test
  # libraries on platforms that require it (Linux x86_64 with a shared
  # gtest, for example).  Cheap to leave on unconditionally.
  POSITION_INDEPENDENT_CODE ON)
# Deliberately NOT set:
#   CXX_VISIBILITY_PRESET hidden  - tests need to reach internal symbols
#   OUTPUT_NAME / VERSION / SOVERSION - this is never installed

if(MSVC)
  string(REPLACE "/GR" "" _dbgapi_internal_cxx_flags "${CMAKE_CXX_FLAGS}")
  # Note: no /GR-.  Tests link this archive into binaries that need RTTI
  # for GMock; leaving RTTI off here would force it off at link time.
  target_compile_options(amd-dbgapi-internal PRIVATE /permissive-)
else()
  # Same warning set as the public library, but drop -Werror and
  # -fno-rtti.  -Werror is intentionally off for the seam so a future
  # test-only refactor isn't blocked by a single warning, and -fno-rtti
  # is incompatible with GMock matchers in linked tests.
  target_compile_options(amd-dbgapi-internal PRIVATE
    -Wall -Wextra -Wshadow -Wno-attributes -Wconversion)
endif()

# Mirror the public library's compile definitions.  These come from
# system header expectations (STDC_*_MACROS) and the generated version
# header, so the same TUs need them to compile identically here.
target_compile_definitions(amd-dbgapi-internal PRIVATE
  __STDC_LIMIT_MACROS __STDC_CONSTANT_MACROS __STDC_FORMAT_MACROS)

# Pull in HAVE_BACKTRACE / __SANE_USERSPACE_TYPES__ / WITH_API_TRACING
# etc. that were applied to the public target after dependency probes.
# Copying via $<TARGET_PROPERTY:...> means we never have to re-run those
# probes here.
get_target_property(_dbgapi_pub_defs amd-dbgapi COMPILE_DEFINITIONS)
if(_dbgapi_pub_defs)
  target_compile_definitions(amd-dbgapi-internal PRIVATE ${_dbgapi_pub_defs})
endif()

# Same for include directories the probes added (libbacktrace, libdxg).
get_target_property(_dbgapi_pub_incs amd-dbgapi INCLUDE_DIRECTORIES)
if(_dbgapi_pub_incs)
  target_include_directories(amd-dbgapi-internal PRIVATE ${_dbgapi_pub_incs})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  # The libdxg headers are added SYSTEM PRIVATE on the public target,
  # which doesn't carry over via the COMPILE_DEFINITIONS / INCLUDE_DIRECTORIES
  # generic copy above.  Re-add them here.
  target_include_directories(amd-dbgapi-internal SYSTEM PRIVATE
    third_party/libdxg/include)
endif()

# Public headers — tests #include "process.h" etc.  Expose the src
# directory PUBLIC so consumers (test executables) pick it up
# automatically, and the generated amd-dbgapi/amd-dbgapi.h header so
# the public C types are also visible.
target_include_directories(amd-dbgapi-internal
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/amd-dbgapi>)

# Link dependencies of the public library.  amd_comgr + libdl are always
# needed; libbacktrace is conditional - debug.cpp calls backtrace_*
# symbols when HAVE_BACKTRACE is defined, and HAVE_BACKTRACE is inherited
# from the public target's COMPILE_DEFINITIONS above, so we must also
# inherit the link target to resolve those symbols.
#
# amd_comgr is PUBLIC because src/architecture.h #includes
# <amd_comgr/amd_comgr.h>; test TUs that include architecture.h (or
# anything that transitively pulls it in) need amd_comgr's include path
# AND the imported target must propagate to the test executable so the
# archive's references to amd_comgr_* resolve at final link.
target_link_libraries(amd-dbgapi-internal PUBLIC amd_comgr)
target_link_libraries(amd-dbgapi-internal PRIVATE ${CMAKE_DL_LIBS})

if(TARGET libbacktrace::libbacktrace)
  target_link_libraries(amd-dbgapi-internal PRIVATE libbacktrace::libbacktrace)
elseif(BACKTRACE_LIB)
  target_link_libraries(amd-dbgapi-internal PRIVATE ${BACKTRACE_LIB})
endif()

# Match the per-source compile definitions applied to versioning.cpp
# and initialization.cpp on the public target so version queries work
# the same way in tests.
set_source_files_properties(src/versioning.cpp src/initialization.cpp
  TARGET_DIRECTORY amd-dbgapi-internal
  PROPERTIES
    COMPILE_DEFINITIONS
      "AMD_DBGAPI_VERSION_PATCH=${PROJECT_VERSION_PATCH};AMD_DBGAPI_BUILD_INFO=\"${PROJECT_VERSION}-${build_info}\"")

# Make sure the generated amd-dbgapi.h header is produced before any
# TU in the internal target tries to include it.  Touching the public
# target's dependency chain isn't enough because EXCLUDE_FROM_ALL
# breaks the implicit ordering on some generators (Ninja in particular).
add_dependencies(amd-dbgapi-internal amd-dbgapi)
