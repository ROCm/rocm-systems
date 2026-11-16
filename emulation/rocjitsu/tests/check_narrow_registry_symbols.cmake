# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

set(EXTRA_REQUIRED_SYMBOLS rj_test_narrow_target_count)
set(EXTRA_FORBIDDEN_SYMBOLS
    "rocjitsu::cdna1::"
    "rocjitsu::cdna2::"
    "rocjitsu::cdna3::"
    "rocjitsu::cdna4::"
    "rocjitsu::rdna1::"
    "rocjitsu::rdna2::"
    "rocjitsu::rdna3::"
    "rocjitsu::rdna3_5::"
    "rocjitsu::rdna4::"
)
include(${CMAKE_CURRENT_LIST_DIR}/check_model_only_symbols.cmake)
