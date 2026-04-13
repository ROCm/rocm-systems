// MIT License
//
// Copyright (c) 2017-2025 Advanced Micro Devices, Inc.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef SRC_HSA_INCLUDES_H_
#define SRC_HSA_INCLUDES_H_

#ifdef _WIN32
#    include <hsa.h>
#    include <hsa_ext_amd.h>
#    include <hsa_ext_finalize.h>
#    include <hsa_ven_amd_aqlprofile.h>
#    include <hsa_ven_amd_loader.h>
#else
#    include <hsa/hsa.h>
#    include <hsa/hsa_ext_amd.h>
#    include <hsa/hsa_ext_finalize.h>
#    include <hsa/hsa_ven_amd_aqlprofile.h>
#    include <hsa/hsa_ven_amd_loader.h>
#endif

#endif  // SRC_HSA_INCLUDES_H_
