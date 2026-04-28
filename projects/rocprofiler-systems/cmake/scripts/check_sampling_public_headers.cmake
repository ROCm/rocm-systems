# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# NFR-PORT-1: Verify public sampling headers contain no OS-specific includes.
# Called from the sampling CMakeLists.txt add_test().

if(NOT DEFINED SAMPLING_INCLUDE)
    message(FATAL_ERROR "SAMPLING_INCLUDE not set")
endif()

file(GLOB_RECURSE _headers "${SAMPLING_INCLUDE}/*.hpp")

# No exceptions: thread_context is now an opaque forward-declaration only (NFR-PORT-3).
# The complete definition lives in src/linux/platform_thread_context.hpp (not public).

set(_forbidden_patterns
    "#include <signal.h>"
    "#include <ucontext.h>"
    "#include <pthread.h>"
    "#include <linux/perf_event.h>"
)

foreach(_h IN LISTS _headers)
    file(READ "${_h}" _content)
    foreach(_pat IN LISTS _forbidden_patterns)
        if(_content MATCHES "${_pat}")
            message(
                FATAL_ERROR
                "NFR-PORT-1 violation: forbidden include '${_pat}' found in public header:\n  ${_h}"
            )
        endif()
    endforeach()
endforeach()

message(STATUS "NFR-PORT-1 PASS: no forbidden OS includes in ${SAMPLING_INCLUDE}")
