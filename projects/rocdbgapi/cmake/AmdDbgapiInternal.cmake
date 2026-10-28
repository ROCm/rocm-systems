## SPDX-License-Identifier: MIT
## SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

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
# ENABLE_<TIER>_TESTS is ON.  It assumes the following variables are
# already in scope:
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
  CXX_EXTENSIONS ON)
# Deliberately NOT set:
#   CXX_VISIBILITY_PRESET hidden  - tests need to reach internal symbols
#   OUTPUT_NAME / VERSION / SOVERSION - this is never installed

if(MSVC)
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

# This is a static archive: public API declarations must carry neither
# dllimport nor dllexport.  PUBLIC so test executables that include
# amd-dbgapi.h and link this archive see the same plain declarations.
target_compile_definitions(amd-dbgapi-internal PUBLIC
  AMD_DBGAPI=AMD_DBGAPI_CALL)

# Pull in HAVE_BACKTRACE / __SANE_USERSPACE_TYPES__ / WITH_API_TRACING
# etc. that were applied to the public target after dependency probes.
# Copying via $<TARGET_PROPERTY:...> means we never have to re-run those
# probes here.
get_target_property(_dbgapi_pub_defs amd-dbgapi COMPILE_DEFINITIONS)
if(_dbgapi_pub_defs)
  target_compile_definitions(amd-dbgapi-internal PRIVATE ${_dbgapi_pub_defs})
endif()

# Same for include directories the probes added (libbacktrace, ...).
# libdxg is excluded from the copy: it is a SYSTEM include on the public
# target, and reading INCLUDE_DIRECTORIES drops that attribute, so the copy
# would turn its (unsilenceable) warnings back on.  It is re-added below.
get_target_property(_dbgapi_pub_incs amd-dbgapi INCLUDE_DIRECTORIES)
if(_dbgapi_pub_incs)
  list(FILTER _dbgapi_pub_incs EXCLUDE REGEX "third_party/libdxg/include$")
  target_include_directories(amd-dbgapi-internal PRIVATE ${_dbgapi_pub_incs})
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  target_include_directories(amd-dbgapi-internal SYSTEM PRIVATE
    third_party/libdxg/include)
  # src/runtime_rdebug.h includes <link.h>, which Windows only gets from
  # the imported stub.  Test TUs that include it need it too.
  target_include_directories(amd-dbgapi-internal PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/windows/imported>)
endif()

# Public headers — tests #include "process.h" etc.  Expose the src
# directory PUBLIC so consumers (test executables) pick it up
# automatically, and the generated amd-dbgapi/amd-dbgapi.h header so
# the public C types are also visible.
#
# The src directory is added with -iquote rather than -I where possible: it
# contains headers whose names clash with system headers (process.h, ...).
# With -I MinGW's <pthread.h> would pick up src/process.h for its own
# #include <process.h>.  -iquote only affects #include "..." lookups.
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  # cl has no -iquote equivalent.
  target_include_directories(amd-dbgapi-internal
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
else()
  target_compile_options(amd-dbgapi-internal
    PUBLIC
      "$<BUILD_INTERFACE:SHELL:-iquote ${CMAKE_CURRENT_SOURCE_DIR}/src>")
endif()
target_include_directories(amd-dbgapi-internal
  PUBLIC
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
