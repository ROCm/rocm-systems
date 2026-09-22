/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

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
