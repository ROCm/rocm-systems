// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#pragma once

class GFX11
{
public:
    enum inst_type_issued
    {
        TYPE_VALU = 0,
        TYPE_SCALAR,
        TYPE_TEX,
        TYPE_LDS,
        TYPE_LDS_DIRECT,
        TYPE_EXPORT,
        TYPE_MESSAGE,
        TYPE_BARRIER,
        TYPE_BRANCH_NOT_TAKEN,
        TYPE_BRANCH_TAKEN,
        TYPE_JUMP,
        TYPE_OTHER,
        TYPE_NO_INST,
        TYPE_DUAL_VALU = 31,
        TYPE_MATRIX    = 31,
        TYPE_FLAT      = 31,
    };

    enum reason_not_issued
    {
        REASON_NO_INSTRUCTION_AVAILABLE = 0,
        REASON_ALU_DEPENDENCY,
        REASON_WAITCNT,
        REASON_ARBITER_NOT_WIN,
        REASON_SLEEP_WAIT,
        REASON_BARRIER_WAIT,
        REASON_OTHER_WAIT,
        REASON_INTERNAL_INSTRUCTION = 31,
        REASON_ARBITER_WIN_EX_STALL = 31,
    };

    enum arb_state
    {
        ISSUE_MISC = 0,
        ISSUE_EXP,
        ISSUE_LDS_DIRECT,
        ISSUE_LDS,
        ISSUE_VMEM_TEX,
        ISSUE_SCALAR,
        ISSUE_VALU,
        ISSUE_MATRIX = 31,
        ISSUE_FLAT   = 31,
        ISSUE_BRMSG  = 31,
    };
};
