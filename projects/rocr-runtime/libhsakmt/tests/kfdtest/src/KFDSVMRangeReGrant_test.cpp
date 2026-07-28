// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

#include "KFDSVMRangeTest.hpp"
#include <sys/mman.h>
#include <string.h>

/*
 * Deregister leaves a range NO_ACCESS, and allocators like jemalloc reuse the
 * VA without munmap so that sticks. Registering the same VA again has to
 * restore access or the GPU faults on it. Read access back with hsaKmtSVMGetAttr
 * instead of relying on a fault.
 */
TEST_P(KFDSVMRangeTest, SVMApiRegisterReGrant) {
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

        void *buf = mmap(0, PG, PROT_READ | PROT_WRITE,
                         MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        ASSERT_NE_GPU(MAP_FAILED, buf, gpuNode);
        memset(buf, 0, PG);

        /* Register then deregister: the range is left NO_ACCESS. */
        EXPECT_SUCCESS_GPU(HSAKMT_CALL(hsaKmtRegisterMemory, m_hsakmt_current_ctx,
                                       buf, PG), gpuNode);
        EXPECT_SUCCESS_GPU(HSAKMT_CALL(hsaKmtDeregisterMemory, m_hsakmt_current_ctx,
                                       buf), gpuNode);
        EXPECT_EQ_GPU(HSA_SVM_ATTR_NO_ACCESS, access(buf), gpuNode);

        /* Re-register the same VA (allocator reuse without munmap): access must
         * be restored, not left at NO_ACCESS.
         */
        EXPECT_SUCCESS_GPU(HSAKMT_CALL(hsaKmtRegisterMemory, m_hsakmt_current_ctx,
                                       buf, PG), gpuNode);
        EXPECT_NE_GPU(HSA_SVM_ATTR_NO_ACCESS, access(buf), gpuNode);

        EXPECT_SUCCESS_GPU(HSAKMT_CALL(hsaKmtDeregisterMemory, m_hsakmt_current_ctx,
                                       buf), gpuNode);
        munmap(buf, PG);
    }));

    TEST_END
}
