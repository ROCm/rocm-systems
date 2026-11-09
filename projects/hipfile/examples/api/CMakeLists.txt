# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(HipFileAddExecutable)

set(API_EXAMPLES
    get-version
)

foreach(example ${API_EXAMPLES})
    hipfile_add_executable(
        NAME ${example}
        DEPS hipfile
        SRCS "${example}.cpp"
        SYSINCLS ${HIPFILE_INCLUDE_PATH}
    )
endforeach()
