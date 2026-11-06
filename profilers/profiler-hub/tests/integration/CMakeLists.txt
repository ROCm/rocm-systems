# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# This directory is on the ctest walk (subdirs(integration) from tests/).
# generate-profiler-hub-pytest-ctests overwrites this build directory's
# CTestTestfile.cmake at build time, the same way rocprofiler-systems
# overwrites its tests/CTestTestfile.cmake. Unit tests stay in tests/unit.
