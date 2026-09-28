/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_CAPABILITY_H_
#define NET_IB_CAPABILITY_H_

#include "common_cast.h"

// Device features that ibv_query_device does not report, found by trying them.
// Results are cached in struct ncclIbDev.

// Probes every device not probed yet. Call once devices and QP globals are initialized.
ncclResult_t IbCastCapProbeDevices();

// Probes on first use if IbCastCapProbeDevices did not.
ncclResult_t IbCastCapHasUd(struct ncclIbDev* dev, bool* hasUd);

#endif // NET_IB_CAPABILITY_H_
