/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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
