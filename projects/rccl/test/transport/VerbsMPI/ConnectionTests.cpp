/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file ConnectionTests.cpp
 * @brief Verbs acceptance tests for QP creation and connection establishment.
 */

#include "VerbsMPITestBase.hpp"

#ifdef MPI_TESTS_ENABLED

using namespace VerbsAcceptance;

// ===========================================================================
// UD_RC_Support -- baseline. Can the NIC create both RC and UD QPs and bring
// each to RTS? If this does not work, nothing else is meaningful.
// ===========================================================================
TEST_F(VerbsAcceptanceMPITest, UD_RC_Support)
{
    if(!requireEvenPairs())
        GTEST_SKIP() << "Requires an even number of MPI ranks (>= 2)";

    resolveOptionalSymbols();
    if(!allRanksAgree(openIbDevice()))
        GTEST_SKIP() << "No active IB device on all ranks";

    // RC -- reduce the create result before any OOB exchange so paired ranks
    // never block in MPI_Sendrecv when one side fails to create a QP.
    bool rc = createQp(IBV_QPT_RC);
    if(!allRanksAgree(rc))
    {
        destroyQp();
        GTEST_SKIP() << "RC QP creation not supported on all ranks";
    }
    QpInfo remote = exchangeQpInfo();
    rc            = rcToRtrRts(remote);
    destroyQp();
    ASSERT_MPI_TRUE(rc);

    // UD
    bool ud = createQp(IBV_QPT_UD);
    if(!allRanksAgree(ud))
    {
        destroyQp();
        GTEST_SKIP() << "UD QP not supported on all ranks";
    }
    (void)exchangeQpInfo(); // keep OOB symmetric across ranks
    ud = udToRts();
    destroyQp();
    ASSERT_MPI_TRUE(ud);
}

#endif // MPI_TESTS_ENABLED
