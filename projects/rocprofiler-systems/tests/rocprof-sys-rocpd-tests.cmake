# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

# -------------------------------------------------------------------------------------- #
#
# ROCPD tests
#
# -------------------------------------------------------------------------------------- #

set(_rocpd_environment
    "${_base_environment}"
    "ROCPROFSYS_USE_ROCPD=true"
)

rocprofiler_systems_add_test(
    SKIP_REWRITE SKIP_RUNTIME
    NAME transpose-rocpd
    TARGET transpose
    MPI OFF
    GPU ON
    NUM_PROCS 1
    ENVIRONMENT "${_rocpd_environment}")


rocprofiler_systems_add_validation_test(
    NAME transpose-rocpd-sampling
    ROCPD_FILE "rocpd.db"
    ARGS --validation-rules "${CMAKE_CURRENT_LIST_DIR}/rocpd_validation_rules/transpose/validation_rules.json"
    DEPENDS transpose-rocpd
    LABELS "rocprofiler"
)
