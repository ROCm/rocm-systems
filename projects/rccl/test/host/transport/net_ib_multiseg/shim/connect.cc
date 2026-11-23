/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// NET_IB_CONNECT_CC_PATH for IB-MSEG: route the fork's spellings through connect-test.cc's seam window.
#define IbMsegGdrSupport ncclIbGdrSupport
#define IbMsegDmaBufSupport ncclIbDmaBufSupport
#include NET_IB_MSEG_CONNECT_CC_PATH
#undef IbMsegDmaBufSupport
#undef IbMsegGdrSupport
