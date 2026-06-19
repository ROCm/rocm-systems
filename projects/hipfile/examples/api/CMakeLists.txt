# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

include(AISAddExecutable)

set(API_EXAMPLES
    get-version
)

foreach(example ${API_EXAMPLES})
    ais_add_executable(
        NAME ${example}
        DEPS hipfile
        SRCS "${example}.cpp"
        SYSINCLS ${HIPFILE_INCLUDE_PATH}
    )
endforeach()
