# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

set(TEST_SRC
  channel_descriptor.cc
)

hip_add_exe_to_target(NAME ChannelDescriptorTest
                      TEST_SRC ${TEST_SRC}
                      TEST_TARGET_NAME build_tests)
