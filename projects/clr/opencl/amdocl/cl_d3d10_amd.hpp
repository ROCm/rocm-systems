/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "cl_common.hpp"

#include "platform/context.hpp"
#include "platform/memory.hpp"
#include "platform/interop_d3d10.hpp"

#include <utility>

namespace amd {

//! Functions for executing the D3D10 related stuff
cl_mem clCreateBufferFromD3D10ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                          ID3D10Resource* pD3DResource, int* errcode_ret);
cl_mem clCreateImage1DFromD3D10ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                           ID3D10Resource* pD3DResource, UINT subresource,
                                           int* errcode_ret);
cl_mem clCreateImage2DFromD3D10ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                           ID3D10Resource* pD3DResource, UINT subresource,
                                           int* errcode_ret);
cl_mem clCreateImage3DFromD3D10ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                           ID3D10Resource* pD3DResource, UINT subresource,
                                           int* errcode_ret);
void SyncD3D10Objects(std::vector<amd::Memory*>& memObjects);

}  // namespace amd
