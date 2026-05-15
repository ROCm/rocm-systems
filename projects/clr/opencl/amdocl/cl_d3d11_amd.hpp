/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once

#include "cl_d3d10_amd.hpp"

#include "platform/context.hpp"
#include "platform/memory.hpp"
#include "platform/interop_d3d11.hpp"

#include <utility>

extern CL_API_ENTRY cl_mem CL_API_CALL clGetPlaneFromImageAMD(cl_context /* context */,
                                                              cl_mem /* mem */, cl_uint /* plane */,
                                                              cl_int* /* errcode_ret */);

namespace amd {

//! Functions for executing the D3D11 related stuff
cl_mem clCreateBufferFromD3D11ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                          ID3D11Resource* pD3DResource, int* errcode_ret);
cl_mem clCreateImage1DFromD3D11ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                           ID3D11Resource* pD3DResource, UINT subresource,
                                           int* errcode_ret);
cl_mem clCreateImage2DFromD3D11ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                           ID3D11Resource* pD3DResource, UINT subresource,
                                           int* errcode_ret);
cl_mem clCreateImage3DFromD3D11ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                           ID3D11Resource* pD3DResource, UINT subresource,
                                           int* errcode_ret);
void SyncD3D11Objects(std::vector<amd::Memory*>& memObjects);

}  // namespace amd
