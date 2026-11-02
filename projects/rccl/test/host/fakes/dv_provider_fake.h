/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Stand-in for an ibverbs provider dv library (libionic, libmlx5) that exports none of the dv symbols.
// Built as one shared object per provider, each carrying that provider's SONAME. The
// ionic-no-gda variant (DV_PROVIDER_FAKE_IONIC_NO_GDA) also exports ionic_dv_pd_set_udma_mask.

#ifndef RCCL_TEST_HOST_FAKES_DV_PROVIDER_FAKE_H_
#define RCCL_TEST_HOST_FAKES_DV_PROVIDER_FAKE_H_

constexpr int kDvProviderFakeOpsResult = 0x1b0e;

extern "C" int rcclDvProviderFakeOps();

#endif  // RCCL_TEST_HOST_FAKES_DV_PROVIDER_FAKE_H_
