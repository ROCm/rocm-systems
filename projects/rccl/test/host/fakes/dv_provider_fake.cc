/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "dv_provider_fake.h"

// Plays the driver-ops entry libibverbs keeps a pointer to after the provider registers.
extern "C" __attribute__((visibility("default"))) int rcclDvProviderFakeOps() {
  return kDvProviderFakeOpsResult;
}

#ifdef DV_PROVIDER_FAKE_IONIC_NO_GDA
// Older libionic release: exports pd_set_udma_mask (versioned IONIC_1.0 by
// dv_provider_fake_ionic_no_gda.map) but not qp_set_gda.
extern "C" __attribute__((visibility("default"))) int ionic_dv_pd_set_udma_mask(void*, unsigned char) {
  return 0;
}
#endif
