# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# The vocabulary matches TheRock's THEROCK_SANITIZER so a standalone build and a superbuild
# are configured the same way. HOST_ASAN is a distinct value because CLR compiles and JITs
# device code.
set(ENABLE_SANITIZER
    "OFF"
    CACHE STRING "Sanitizer mode: OFF, ASAN (host and device), or HOST_ASAN (host only)")
set_property(CACHE ENABLE_SANITIZER PROPERTY STRINGS OFF ASAN HOST_ASAN)

if(THEROCK_SANITIZER AND NOT ENABLE_SANITIZER STREQUAL "OFF")
  message(FATAL_ERROR
    "THEROCK_SANITIZER='${THEROCK_SANITIZER}' already selects the sanitizer for this build, "
    "so ENABLE_SANITIZER must be left at OFF, found '${ENABLE_SANITIZER}'. Pass "
    "-DENABLE_SANITIZER=OFF, or configure without THEROCK_SANITIZER for a standalone build.")
endif()

# TheRock injects THEROCK_SANITIZER variable. It takes precedence over ENABLE_SANITIZER.
if(THEROCK_SANITIZER STREQUAL "ASAN" OR THEROCK_SANITIZER STREQUAL "HOST_ASAN")
  # The enclosing toolchain must already be selecting a sanitizer. CLR still
  # appends the host flags below; this only rejects a THEROCK_SANITIZER value
  # that never reached the toolchain file.
  if(NOT (CMAKE_CXX_FLAGS_INIT MATCHES "-fsanitize=address"
          OR CMAKE_CXX_FLAGS MATCHES "-fsanitize=address"))
    message(FATAL_ERROR
      "THEROCK_SANITIZER='${THEROCK_SANITIZER}' but no -fsanitize=address reaches the C++ "
      "compile line. For a standalone build pass -DENABLE_SANITIZER=${THEROCK_SANITIZER} "
      "instead.")
  endif()
  set(ENABLE_SANITIZER "${THEROCK_SANITIZER}")
endif()

set(_clr_sanitizer_valid OFF ASAN HOST_ASAN)
if(NOT ENABLE_SANITIZER IN_LIST _clr_sanitizer_valid)
  list(JOIN _clr_sanitizer_valid ", " _clr_sanitizer_valid)
  message(FATAL_ERROR
    "ENABLE_SANITIZER='${ENABLE_SANITIZER}' is not one of: ${_clr_sanitizer_valid}")
endif()
unset(_clr_sanitizer_valid)

# ADDRESS_SANITIZER (hipamd, amdocl) and ENABLE_ADDRESS_SANITIZER (opencl tests) were the
# previous switches and neither could express a host-only build.
if(ADDRESS_SANITIZER OR ENABLE_ADDRESS_SANITIZER)
  message(FATAL_ERROR
    "ADDRESS_SANITIZER and ENABLE_ADDRESS_SANITIZER have been replaced by ENABLE_SANITIZER. "
    "Pass -DENABLE_SANITIZER=ASAN for the previous behaviour, or -DENABLE_SANITIZER=HOST_ASAN "
    "to instrument only host code.")
endif()

# Device instrumentation depends on xnack+, a KFD feature, and the host-only mode has not
# been brought up anywhere else, so both modes are Linux only for now.
if(NOT ENABLE_SANITIZER STREQUAL "OFF" AND NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR
    "ENABLE_SANITIZER='${ENABLE_SANITIZER}' is not supported on '${CMAKE_SYSTEM_NAME}'. "
    "Sanitizer builds are currently Linux only.")
endif()

if(ENABLE_SANITIZER STREQUAL "ASAN" AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR
    "ENABLE_SANITIZER=ASAN requires a Clang compiler, found '${CMAKE_CXX_COMPILER_ID}'. "
    "Use -DENABLE_SANITIZER=HOST_ASAN to instrument only host code.")
endif()

if(ENABLE_SANITIZER STREQUAL "ASAN")
  add_compile_definitions(DEVICE_ADDRESS_SANITIZER=1)
else()
  add_compile_definitions(DEVICE_ADDRESS_SANITIZER=0)
endif()

# Host instrumentation. Always applied, including under TheRock.
# TheRock's HOST_ASAN path puts -Xarch_host -fsanitize=address in
# CMAKE_*_FLAGS_INIT and links with -fsanitize=address, but FLAGS_INIT is not
# re-copied into CMAKE_*_FLAGS on a reconfigure. The shipped libamdhip64 then
# NEEDs libclang_rt.asan.so and only has the weak __asan_report_* stubs from
# the link line, so hipFree aborts with "free on address which was not
# malloc()-ed". Appending the flags here is what the compiler actually uses.
# Repeating -fsanitize=address is harmless when the enclosing build already
# added it. These are host C/C++ flags; device bitcode is compiled by hiprtc
# custom commands that do not inherit CMAKE_CXX_FLAGS.
if(NOT ENABLE_SANITIZER STREQUAL "OFF")
  set(_linker_flags "-fsanitize=address")
  set(_compiler_flags "-fno-omit-frame-pointer -fsanitize=address")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(BUILD_SHARED_LIBS)
      string(APPEND _compiler_flags " -shared-libsan")
      string(APPEND _linker_flags " -shared-libsan")
    else()
      string(APPEND _linker_flags " -static-libsan")
    endif()
  endif()

  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_compiler_flags}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_compiler_flags}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${_linker_flags} -Wl,--build-id=sha1")
  set(CMAKE_SHARED_LINKER_FLAGS
      "${CMAKE_SHARED_LINKER_FLAGS} ${_linker_flags} -Wl,--build-id=sha1")

  # Directory and target options, not only CMAKE_*_FLAGS. ccache drops
  # -Xarch_* unless the compile also passes -arch, so TheRock's
  # -Xarch_host -fsanitize=address never reaches clang and the cached objects
  # are uninstrumented. A plain -fsanitize=address is what the compiler and
  # ccache both keep. These sources are host C/C++; hiprtc device commands
  # do not inherit them.
  add_compile_options(
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address>")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND BUILD_SHARED_LIBS)
    add_compile_options("$<$<COMPILE_LANGUAGE:C,CXX>:-shared-libsan>")
    add_link_options(
      "$<$<LINK_LANGUAGE:C,CXX>:-fsanitize=address>"
      "$<$<LINK_LANGUAGE:C,CXX>:-shared-libsan>")
  endif()
  unset(_compiler_flags)
  unset(_linker_flags)
  message(STATUS "CLR sanitizer: ${ENABLE_SANITIZER} (host flags applied by CLR)")
endif()

# Attach the host sanitizer to one target. add_compile_options above covers
# targets created later in this directory tree; call this for targets that
# must stay instrumented even if a subdirectory replaces the directory flags.
function(clr_asan_instrument_target target)
  if(ENABLE_SANITIZER STREQUAL "OFF")
    return()
  endif()
  target_compile_options(${target} PRIVATE
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fno-omit-frame-pointer>"
    "$<$<COMPILE_LANGUAGE:C,CXX>:-fsanitize=address>")
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND BUILD_SHARED_LIBS)
    target_compile_options(${target} PRIVATE
      "$<$<COMPILE_LANGUAGE:C,CXX>:-shared-libsan>")
    target_link_options(${target} PRIVATE -fsanitize=address -shared-libsan)
  endif()
endfunction()

# Fail the build if a shared library was only linked with ASan. Link-only
# -fsanitize=address emits weak __asan_report_* symbols and no __asan_init.
function(clr_require_asan_instrumentation target)
  if(ENABLE_SANITIZER STREQUAL "OFF" OR NOT BUILD_SHARED_LIBS OR NOT UNIX)
    return()
  endif()
  add_custom_command(
    TARGET ${target} POST_BUILD
    COMMAND ${CMAKE_COMMAND}
            -DFILE=$<TARGET_FILE:${target}>
            -P ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/check_asan_instrumentation.cmake
    COMMENT "Verifying ${target} is instrumented with ASan"
    VERBATIM)
endfunction()
