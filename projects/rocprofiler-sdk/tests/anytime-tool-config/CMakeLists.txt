# MIT License
#
# Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

#
# Anytime tool configuration integration tests
#
# Each subdirectory is a self-contained execute + validate test pair that exercises
# registering a rocprofiler-sdk tool at a controlled point during a running process via
# rocprofiler_force_configure (the "anytime" initialization path).
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

add_subdirectory(early-hip-late-dispatch)
add_subdirectory(early-hip-late-memcpy)
add_subdirectory(early-hip-late-memalloc)
add_subdirectory(early-multi-late-overlap)
add_subdirectory(early-hsa-late-hsa)
add_subdirectory(early-hip-two-late-distinct)
add_subdirectory(early-hip-late-marker)
add_subdirectory(early-callback-late-buffered)
add_subdirectory(early-hip-late-stopstart)
add_subdirectory(early-hip-late-double-configure)

# RCCL variations are registered unconditionally so they remain visible (as DISABLED) in
# CI when RCCL is absent. Each test's CMakeLists sets DISABLED based on rccl availability;
# the python-rccl workload target only exists when RCCL is installed (see
# tests/lib/python/CMakeLists.txt).
add_subdirectory(early-hiprccl-late-hipmemcpy)
add_subdirectory(early-hiprccl-late-hiprcclmemcpy)

# Codec variations (rocDecode / rocJPEG), same unconditional-registration + DISABLED
# pattern as the RCCL and the existing tests/rocdecode + tests/rocjpeg tests. The
# python-rocdecode / python-rocjpeg workload targets only exist when the codec packages
# are installed; the tests are additionally skipped at runtime if the codec VA-API backend
# is unavailable.
add_subdirectory(early-hip-late-rocdecode)
add_subdirectory(early-hip-late-rocjpeg)
