# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

###############################################################################
# ADD ROCSHMEM TARGET FOR FILES IN CURRENT DIRECTORY
###############################################################################
target_sources(
  ${PROJECT_NAME}
  PRIVATE
    queue_pair_host.cpp
)
