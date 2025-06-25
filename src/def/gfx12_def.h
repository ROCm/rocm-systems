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

#ifndef _GFX12_DEF_H_
#define _GFX12_DEF_H_

#include "linux/soc24_enum.h"
#include "util/soc15_common.h"
#include "util/reg_offsets.h"
#include "linux/registers/gc/gc_12_0_0_offset.h"
#include "linux/registers/gc/gc_12_0_0_sh_mask.h"
// Rename CP_PERFMON_CNTL_1 to CP_PERFMON_CNTL for better compatibility
// CP_PERFMON_CNTL_1
#define regCP_PERFMON_CNTL_BASE_IDX regCP_PERFMON_CNTL_1_BASE_IDX
#define regCP_PERFMON_CNTL regCP_PERFMON_CNTL_1
#define CP_PERFMON_CNTL__PERFMON_STATE__SHIFT CP_PERFMON_CNTL_1__PERFMON_STATE__SHIFT
#define CP_PERFMON_CNTL__SPM_PERFMON_STATE__SHIFT CP_PERFMON_CNTL_1__SPM_PERFMON_STATE__SHIFT
#define CP_PERFMON_CNTL__PERFMON_ENABLE_MODE__SHIFT CP_PERFMON_CNTL_1__PERFMON_ENABLE_MODE__SHIFT
#define CP_PERFMON_CNTL__PERFMON_SAMPLE_ENABLE__SHIFT CP_PERFMON_CNTL_1__PERFMON_SAMPLE_ENABLE__SHIFT
#define CP_PERFMON_CNTL__PERFMON_STATE_MASK CP_PERFMON_CNTL_1__PERFMON_STATE_MASK
#define CP_PERFMON_CNTL__SPM_PERFMON_STATE_MASK CP_PERFMON_CNTL_1__SPM_PERFMON_STATE_MASK
#define CP_PERFMON_CNTL__PERFMON_ENABLE_MODE_MASK CP_PERFMON_CNTL_1__PERFMON_ENABLE_MODE_MASK
#define CP_PERFMON_CNTL__PERFMON_SAMPLE_ENABLE_MASK CP_PERFMON_CNTL_1__PERFMON_SAMPLE_ENABLE_MASK
#include "linux/packets/nvd.h"
#include "gfxip/gfx12/gfx12_block_info.h"
using namespace gfxip::gfx12;
using namespace gfxip::gfx12::gfx1200;
#include "gfxip/gfx12/gfx12_primitives.h"
#include "gfxip/gfx12/gfx12_block_table.h"

#endif  // _GFX12_DEF_H_

