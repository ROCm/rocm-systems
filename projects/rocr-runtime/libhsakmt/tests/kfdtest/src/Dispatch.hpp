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

    // Threads per workgroup. Defaults to 1x1x1, which runs one lane per wave.
    // Only worth changing for workloads that need the full wave width, e.g. a
    // blit copy. Note SetDim() counts threads, not workgroups, so callers do
    // not need to rescale it.
    void SetWorkgroupSize(unsigned int x, unsigned int y, unsigned int z);

    /* VGPRs to give each wave, as the granule count COMPUTE_PGM_RSRC1.VGPRS
     * wants: the wave gets (granules + 1) * 4 registers on gfx9 and earlier,
     * and (granules + 1) * 8 on gfx10+ in wave32. Raise it only for a shader
     * that needs the registers - a wider allocation means fewer waves resident
     * per SIMD, which on gfx8 (256 VGPRs per SIMD) costs occupancy well before
     * it does on CDNA (512).
     */
    void SetVgprGranules(unsigned int granules);

    void Submit(BaseQueue& queue);

    // Build the IB and hand it back instead of submitting it, so the caller can
    // place the packet itself. Lets several dispatches be batched into one queue
    // (see gpu_multicopy()). Call at most once per Dispatch, and never on an
    // instance that is also passed to Submit(): the IB is append-only, so a
    // second build would append a duplicate packet stream rather than replace
    // it, and re-running a built IB would skip its completion fence.
    //
    // Note this is a caller contract, not something BuildIb() can enforce for
    // itself: Submit() deliberately rebuilds, which is what lets a caller
    // re-issue one Dispatch with fresh SetArgs() (see KFDLocalMemoryTest).
    IndirectBuffer *PrepareIb();

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

    unsigned int m_ThreadsX;
    unsigned int m_ThreadsY;
    unsigned int m_ThreadsZ;

    void* m_pArg1;
    void* m_pArg2;

    HsaEvent* m_pEop;

    bool            m_ScratchEn;
    unsigned int    m_ComputeTmpringSize;

    HSAuint64  m_scratch_base;
    unsigned int m_SpiPriority;
    unsigned int  m_FamilyId;
    bool  m_NeedCwsrWA;

    /* What every shader here got before SetVgprGranules() existed: 20 VGPRs on
     * gfx9. Shaders that predate the knob are written to fit in that.
     */
    static const unsigned int DEFAULT_VGPR_GRANULES = 4;
    unsigned int m_VgprGranules;
};

#endif  // __KFD_DISPATCH__H__
