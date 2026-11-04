# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

include(CheckLinkerFlag)
include(CheckPIESupported)

option(SDXI_WERROR "Treat compiler warnings as errors" OFF)

check_linker_flag(C "-Wl,-z,noexecstack" SDXI_LINKER_NOEXECSTACK)
check_linker_flag(C "-Wl,-z,relro"       SDXI_LINKER_RELRO)
check_linker_flag(C "-Wl,-z,now"         SDXI_LINKER_NOW)
check_linker_flag(C "-Wl,--as-needed"    SDXI_LINKER_AS_NEEDED)

check_pie_supported(OUTPUT_VARIABLE sdxi_pie_output LANGUAGES C)

function(sdxi_set_compiler_flags target)
    target_compile_definitions(${target} PRIVATE
      _POSIX_C_SOURCE=200809L
      _DEFAULT_SOURCE
    )

    if(CMAKE_C_COMPILER_ID STREQUAL "GNU" OR CMAKE_C_COMPILER_ID STREQUAL "Clang")
        target_compile_options(${target} PRIVATE
          -Wall -Wextra -Wshadow -Wstrict-prototypes
          -Wmissing-prototypes -Wconversion -Wformat=2
        )
    endif()

    if(SDXI_WERROR)
        set_target_properties(${target} PROPERTIES COMPILE_WARNING_AS_ERROR ON)
    endif()

    target_compile_options(${target} PRIVATE -pthread)
    target_link_options(${target} PRIVATE -pthread)

    if(CMAKE_C_LINK_PIE_SUPPORTED)
        set_target_properties(${target} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    endif()
    if(SDXI_LINKER_NOEXECSTACK)
        target_link_options(${target} PRIVATE "-Wl,-z,noexecstack")
    endif()
    if(SDXI_LINKER_RELRO)
        target_link_options(${target} PRIVATE "-Wl,-z,relro")
    endif()
    if(SDXI_LINKER_NOW)
        target_link_options(${target} PRIVATE "-Wl,-z,now")
    endif()
    if(SDXI_LINKER_AS_NEEDED)
        target_link_options(${target} PRIVATE "-Wl,--as-needed")
    endif()
endfunction()
