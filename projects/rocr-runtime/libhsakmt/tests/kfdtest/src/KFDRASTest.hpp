/*
 * Copyright (C) 2019 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "KFDBaseComponentTest.hpp"

#ifndef __KFD_RAS_TEST__H__
#define __KFD_RAS_TEST__H__

// To be removed when amdgpu_drm.h updated with those definitions
#ifndef AMDGPU_INFO_RAS_ENABLED_FEATURES
#define AMDGPU_INFO_RAS_ENABLED_FEATURES    0x20

#define AMDGPU_INFO_RAS_ENABLED_UMC         (1 << 0)
#define AMDGPU_INFO_RAS_ENABLED_SDMA        (1 << 1)
#define AMDGPU_INFO_RAS_ENABLED_GFX         (1 << 2)
#endif

class KFDRASTest :  public KFDBaseComponentTest {
 public:
    KFDRASTest(void) {}
    ~KFDRASTest(void) {}

    // @brief Executed before every test in KFDRASTest.
    virtual void SetUp();
    // @brief Executed after every test in KFDRASTest.
    virtual void TearDown();

 protected:
    static const unsigned int EVENT_TIMEOUT = 5000;  // 5 seconds
    HsaEvent* m_pRasEvent;
    HSAint32 m_defaultGPUNode;
    FILE* m_pFile;
    bool m_setupStatus;
};

#endif  // __KFD_RAS_TEST__H__
