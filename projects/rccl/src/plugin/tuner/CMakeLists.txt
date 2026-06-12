#
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# See LICENSE.txt for more license information
#

# Tuner plugin sources
set(PLUGIN_TUNER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/tuner_v2.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tuner_v3.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tuner_v4.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/tuner_v5.cc
)

# >>> RCCL-DELTA: csv_tuner.cc is an RCCL-only tuner backend.
list(APPEND PLUGIN_TUNER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/csv_tuner.cc
)
# <<< RCCL-DELTA

# Add tuner plugin sources to parent scope
set(PLUGIN_TUNER_SOURCES ${PLUGIN_TUNER_SOURCES} PARENT_SCOPE)
