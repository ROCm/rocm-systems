# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include_guard(GLOBAL)

#-----------------------------------------------------------------------------
# Option to use the include-what-you-use tool
#
# When this option is enabled, compilation will emit IWYU suggestions.
#-----------------------------------------------------------------------------
option(AIS_USE_IWYU "Run include-what-you-use when compiling" OFF)

if(AIS_USE_IWYU)
    find_program(IWYU_EXE NAMES include-what-you-use REQUIRED)

    # GoogleTest/GoogleMock expose their public API through umbrella headers
    # (gtest/gtest.h, gmock/gmock.h) but define symbols in private sub-headers.
    # Without a mapping, IWYU suggests including those private headers and
    # confuses the quoted vs. angle-bracket spellings. This mapping file
    # redirects all gtest/gmock private headers to the public umbrella headers.
    # libstdc++/glibc declare some symbols in private bits/* headers that IWYU
    # would otherwise suggest including directly (e.g. <bits/chrono.h> for
    # std::chrono, <bits/statx-generic.h> for statx). These are not part of the
    # bundled IWYU mappings, so redirect them to their public headers.
    set(IWYU_MAPPINGS
        "${CMAKE_CURRENT_LIST_DIR}/iwyu-gtest.imp"
        "${CMAKE_CURRENT_LIST_DIR}/iwyu-libc.imp")

    set(IWYU_COMMAND "${IWYU_EXE}")
    foreach(mapping IN LISTS IWYU_MAPPINGS)
        if(NOT EXISTS "${mapping}")
            message(FATAL_ERROR "IWYU mapping file not found: ${mapping}")
        endif()
        list(APPEND IWYU_COMMAND "-Xiwyu" "--mapping_file=${mapping}")
    endforeach()

    set(CMAKE_C_INCLUDE_WHAT_YOU_USE ${IWYU_COMMAND})
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU_COMMAND})
endif()
