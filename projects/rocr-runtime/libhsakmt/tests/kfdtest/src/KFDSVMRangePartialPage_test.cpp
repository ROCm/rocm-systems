/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
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
#include "KFDSVMRangeTest.hpp"
#include <sys/mman.h>
#include <string.h>

/*
 * Registration rounds out to page boundaries, so a small buffer drags in bytes
 * an allocator handed to something else. Deregister must not revoke those: on a
 * kernel that unmaps on no-access the GPU faults on the co-tenant data.
 */
TEST_P(KFDSVMRangeTest, SVMApiPartialPageNotRevoked) {
    TEST_REQUIRE_ENV_CAPABILITIES(ENVCAPS_64BITLINUX);
    TEST_START(TESTPROFILE_RUNALL);

    ASSERT_SUCCESS(KFDTestLaunch([this](int gpuNode) {
        if (!SVMAPISupported_GPU(gpuNode)) {
            LOG() << "Skipping test: SVM API not supported on gpuNode " << gpuNode << std::endl;
            return;
        }
        if (GetFamilyIdFromNodeId(gpuNode) < FAMILY_AI) {
            LOG() << "Skipping test: no svm range support on gpuNode " << gpuNode << std::endl;
            return;
        }
        const HsaNodeProperties *pNodeProperties =
            Get_NodeInfo()->GetNodeProperties(gpuNode);
        if (pNodeProperties->Integrated || Get_NodeInfo()->IsAppAPU(gpuNode)) {
            LOG() << "Skipping test on (App)APU gpuNode " << gpuNode << std::endl;
            return;
        }

        const HSAuint64 PG = PAGE_SIZE;

        auto access = [&](void *addr) -> HSAuint32 {
            HSA_SVM_ATTRIBUTE attr = { HSA_SVM_ATTR_ACCESS, (HSAuint32)gpuNode };
            EXPECT_SUCCESS_GPU(HSAKMT_CALL(hsaKmtSVMGetAttr, m_hsakmt_current_ctx,
                                           addr, PG, 1, &attr), gpuNode);
            return attr.type;
        };

        void *page = mmap(0, 2 * PG, PROT_READ | PROT_WRITE,
                          MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        ASSERT_NE_GPU(MAP_FAILED, page, gpuNode);
        memset(page, 0, 2 * PG);

        /* Only 16 bytes are ever registered. The rest of that page stands in
         * for the neighbours an allocator packs alongside it.
         */
        EXPECT_SUCCESS_GPU(HSAKMT_CALL(hsaKmtRegisterMemory, m_hsakmt_current_ctx,
                                       page, 16), gpuNode);
        EXPECT_SUCCESS_GPU(HSAKMT_CALL(hsaKmtDeregisterMemory, m_hsakmt_current_ctx,
                                       page), gpuNode);
        EXPECT_NE_GPU(HSA_SVM_ATTR_NO_ACCESS, access(page), gpuNode);

        /* A whole page is ours, so that one does get revoked. */
        EXPECT_SUCCESS_GPU(HSAKMT_CALL(hsaKmtRegisterMemory, m_hsakmt_current_ctx,
                                       page, PG), gpuNode);
        EXPECT_SUCCESS_GPU(HSAKMT_CALL(hsaKmtDeregisterMemory, m_hsakmt_current_ctx,
                                       page), gpuNode);
        EXPECT_EQ_GPU(HSA_SVM_ATTR_NO_ACCESS, access(page), gpuNode);

        munmap(page, 2 * PG);
    }));

    TEST_END
}
