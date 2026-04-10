# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

#-----------------------------------------------------------------------------
# Option to use the include-what-you-use tool
#
# When this option is enabled, compilation will emit IWYU suggestions.
#-----------------------------------------------------------------------------
option(AIS_USE_IWYU "Run include-what-you-use when compiling" OFF)

if(AIS_USE_IWYU)
    find_program(IWYU_EXE NAMES include-what-you-use REQUIRED)
    set(CMAKE_C_INCLUDE_WHAT_YOU_USE ${IWYU_EXE})
    set(CMAKE_CXX_INCLUDE_WHAT_YOU_USE ${IWYU_EXE})
endif()
