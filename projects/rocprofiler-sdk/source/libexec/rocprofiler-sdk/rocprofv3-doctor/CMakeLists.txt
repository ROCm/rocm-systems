# MIT License
#
# Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# Not a user-facing command: users run it as `rocprofv3 --doctor`, which execs this
# script. It lives in libexec so that it is not on PATH as a separate tool.
#
# @ONLY (not COPYONLY): rocprofv3-doctor.py carries @FULL_VERSION_STRING@ and friends in
# CONST_VERSION_INFO. The version-consistency check and the JSON report depend on those
# being substituted, exactly as for rocprofv3.py.
configure_file(
    rocprofv3-doctor.py
    ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler-sdk/rocprofv3-doctor
    @ONLY)

install(
    FILES
        ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler-sdk/rocprofv3-doctor
    DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/rocprofiler-sdk
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ
                WORLD_EXECUTE
    COMPONENT tools)
