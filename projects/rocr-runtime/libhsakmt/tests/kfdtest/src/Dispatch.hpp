/*
 * Copyright (C) 2014-2018 Advanced Micro Devices, Inc. All Rights Reserved.
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

#ifndef __KFD_DISPATCH__H__
#define __KFD_DISPATCH__H__
#include "KFDTestUtil.hpp"
#include "IndirectBuffer.hpp"
#include "BaseQueue.hpp"

class Dispatch {
 public:
    Dispatch(const HsaMemoryBuffer& isaBuf, const bool eventAutoReset = false);
    ~Dispatch();

    void SetArgs(void* pArg1, void* pArg2);

    void SetDim(unsigned int x, unsigned int y, unsigned int z);

    void Submit(BaseQueue& queue);

    void Sync(unsigned int timeout = HSA_EVENTTIMEOUT_INFINITE);

    int  SyncWithStatus(unsigned int timeout);

    void SetScratch(int numWaves, int waveSize, HSAuint64 scratch_base);

    void SetSpiPriority(unsigned int priority);
    
    void SetPriv(bool priv);

    HsaEvent *GetHsaEvent() { return m_pEop; }

 private:
    void BuildIb();

 private:
    const HsaMemoryBuffer& m_IsaBuf;

    IndirectBuffer m_IndirectBuf;

    unsigned int m_DimX;
    unsigned int m_DimY;
    unsigned int m_DimZ;

    void* m_pArg1;
    void* m_pArg2;

    HsaEvent* m_pEop;

    bool            m_ScratchEn;
    unsigned int    m_ComputeTmpringSize;

    HSAuint64  m_scratch_base;
    unsigned int m_SpiPriority;
    unsigned int  m_FamilyId;
    bool  m_NeedCwsrWA;
};

#endif  // __KFD_DISPATCH__H__
