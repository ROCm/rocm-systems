/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* $Revision$ on $Date$ */

#pragma once

#include "cl_common.hpp"
#include "platform/context.hpp"
#include "platform/memory.hpp"
#include "platform/interop_d3d9.hpp"

#include <utility>

namespace amd {

cl_mem clCreateImage2DFromD3D9ResourceAMD(Context& amdContext, cl_mem_flags flags,
                                          cl_dx9_media_adapter_type_khr adapter_type,
                                          cl_dx9_surface_info_khr* surface_info, cl_uint plane,
                                          int* errcode_ret);

void SyncD3D9Objects(std::vector<amd::Memory*>& memObjects);

}  // namespace amd
