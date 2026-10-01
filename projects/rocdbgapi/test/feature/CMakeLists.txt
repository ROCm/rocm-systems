## SPDX-License-Identifier: MIT
## SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

# Feature tests for rocdbgapi.
#
# Tests added here may require the kernel driver (KFD on Linux, KMD on
# Windows) and, for the GPU sub-tier, a real GPU plus a HIP runtime.
# Each test must skip cleanly (GTEST_SKIP) when its prerequisites are
# absent so the same binary works on hosts without hardware.

