# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Defines `clr::warnings`, the single source of truth for strict-warning flags.
# First-party CLR targets opt in with:
#
#   target_link_libraries(<target> PRIVATE clr::warnings)

include_guard(GLOBAL)

option(CLR_ENABLE_WERROR
  "Build first-party CLR targets with -Wall -Werror (/W4 /WX on Windows)" OFF)

add_library(clr_warnings INTERFACE)
add_library(clr::warnings ALIAS clr_warnings)

if(NOT CLR_ENABLE_WERROR)
  return()
endif()

# Windows: MSVC's cl, or clang-cl (any Clang driving the MSVC frontend).
# /W4 is the strict-but-sane level; /Wall additionally turns on a large body of
# off-by-default noise from the system headers themselves, so it is not used.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"
    OR CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  # CMake seeds CMAKE_CXX_FLAGS with /W3. CLR's top-level CMakeLists.txt already
  # strips it, and in any case target options are appended last, so /W4 wins.
  target_compile_options(clr_warnings INTERFACE /W4 /WX)

  # ---------------------------------------------------------------------------
  # TEMPORARY. Unlike every other suppression below, this one is scaffolding and
  # is meant to be deleted.
  #
  # C4100 ("unreferenced formal parameter") is the last outstanding /W4 code in
  # CLR: 641 diagnostics over 445 lines in 67 files. It has no GCC/Clang
  # equivalent under -Wall -- it lives in -Wextra -- so no Linux build catches
  # it and there is no existing cleanup to copy.
  #
  # Clearing it is a long sequence of mechanical commits. Suppressing it here
  # lets /WX be enforced *now*, so that the codes already fixed cannot silently
  # regress while that work is in flight. Remove this line, not the others, once
  # the C4100 cleanup lands.
  # ---------------------------------------------------------------------------
  target_compile_options(clr_warnings INTERFACE /wd4100)

  # C4714: "function marked as __forceinline not inlined". Whether the compiler
  # honours __forceinline is its own decision and not something the source can
  # act on, so this reports a non-defect. It also dominates the Windows warning
  # count: every entry of the HIP dispatch table in hip_table_interface.cpp
  # trips it, for 542 of the 1517 /W4 sites in CLR. No GCC/Clang analogue.
  target_compile_options(clr_warnings INTERFACE /wd4714)

  # C4201: "nonstandard extension used: nameless struct/union". The anonymous
  # union is load-bearing wherever it appears: it is how HIP_vector_type lets
  # `float4::x` and `float4::data[0]` name the same storage, and how the PAL
  # capture-manager markers overlay a bitfield struct on a raw dword. Naming
  # the members would be an API break, not a fix.
  target_compile_options(clr_warnings INTERFACE /wd4201)

  # C4324: "structure was padded due to alignment specifier". Reporting that
  # alignas() aligned something is reporting that the code worked. The sites
  # are HSA packets in rocvirtual.hpp that must sit on 64-byte boundaries.
  target_compile_options(clr_warnings INTERFACE /wd4324)

  # C4702: "unreachable code". Every site is a defensive trailing return after
  # a construct MSVC can prove already returns on all paths -- `return false;`
  # after an #if/#else that both return, `return nullptr;` after a switch whose
  # default returns. Deleting them is the obvious move and the wrong one: they
  # exist because other compilers warn "not all control paths return a value"
  # on exactly these shapes, so removing them trades C4702 on Windows for
  # -Wreturn-type on GCC/Clang -- the regression the Linux -Werror work exists
  # to prevent.
  target_compile_options(clr_warnings INTERFACE /wd4702)

  # C4200: "nonstandard extension used: zero-sized array in struct/union".
  # rocglinterop.hpp's metadata_amd_ci_vi_t ends in a flexible array member
  # (per-mip-level offsets) overlaid on a hardware descriptor. The trailing
  # array is the layout, so it cannot be given a fixed size.
  target_compile_options(clr_warnings INTERFACE /wd4200)

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # clang-cl understands the GNU-style spellings, so keep the Linux policy of
    # letting deprecations warn without failing the build.
    target_compile_options(clr_warnings INTERFACE
      -Wno-error=deprecated-declarations)
  endif()

  message(STATUS "CLR warnings: /W4 /WX enabled")
  return()
endif()

# Everything else: GCC/Clang on Linux.
if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux"
    OR NOT CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang)$")
  message(WARNING
    "CLR_ENABLE_WERROR is supported for GCC/Clang on Linux and MSVC/clang-cl "
    "on Windows only; ignoring it for ${CMAKE_CXX_COMPILER_ID} on "
    "${CMAKE_SYSTEM_NAME}.")
  return()
endif()

# Deprecations are a warning on purpose: CLR must keep compiling against ROCm
# headers that deprecate an API ahead of CLR dropping its uses of it.
target_compile_options(clr_warnings INTERFACE
  -Wall -Werror -Wno-error=deprecated-declarations)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 12)
  target_compile_options(clr_warnings INTERFACE -Wno-attributes)
endif()

message(STATUS "CLR warnings: -Wall -Werror enabled")
