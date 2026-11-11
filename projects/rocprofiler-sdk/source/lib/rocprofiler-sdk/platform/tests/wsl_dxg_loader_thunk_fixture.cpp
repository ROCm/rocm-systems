// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// A stand-in librocdxg for wsl_dxg_loader.cpp: the five KMT entry points the
// topology read needs, reporting one GPU node. Every copy in a test layout is
// this same object, told apart by the path it was loaded from.

#include <hsakmt/hsakmttypes.h>

#define FIXTURE_EXPORT extern "C" __attribute__((visibility("default")))

FIXTURE_EXPORT HSAKMT_STATUS
hsaKmtOpenKFD()
{
    return HSAKMT_STATUS_SUCCESS;
}

FIXTURE_EXPORT HSAKMT_STATUS
hsaKmtCloseKFD()
{
    return HSAKMT_STATUS_SUCCESS;
}

FIXTURE_EXPORT HSAKMT_STATUS
hsaKmtAcquireSystemProperties(HsaSystemProperties* props)
{
    if(props == nullptr) return HSAKMT_STATUS_INVALID_PARAMETER;
    *props          = HsaSystemProperties{};
    props->NumNodes = 1;
    return HSAKMT_STATUS_SUCCESS;
}

FIXTURE_EXPORT HSAKMT_STATUS
hsaKmtReleaseSystemProperties()
{
    return HSAKMT_STATUS_SUCCESS;
}

FIXTURE_EXPORT HSAKMT_STATUS
hsaKmtGetNodeProperties(HSAuint32 node_id, HsaNodeProperties* props)
{
    if(node_id != 0 || props == nullptr) return HSAKMT_STATUS_INVALID_PARAMETER;
    *props                  = HsaNodeProperties{};
    props->NumFComputeCores = 1;
    return HSAKMT_STATUS_SUCCESS;
}
