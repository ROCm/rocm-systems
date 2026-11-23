/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib_multiseg/p2p_multiseg.cc, for the IB-MSEG binaries that do not compile it.

#include "transport/net_ib_multiseg/p2p_multiseg_mseg.h"

// IbMsegCloseSend frees the sender's layout cache.
ncclResult_t rcclIbSendLayoutCacheFree(struct IbMsegSendComm*) { return ncclSuccess; }
