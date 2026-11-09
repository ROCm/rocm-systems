/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NET_IB_CAPABILITY_H_
#define NET_IB_CAPABILITY_H_

#include "common_cast.h"

// Device features that ibv_query_device does not report, or reports wrongly,
// found by trying them. Results are cached in struct ncclIbDev.

// Probes every device not probed yet, only for the features the enabled
// resiliency modes need: UD for port recovery, RDMA READ for port failover.
// Call once devices and QP globals are initialized.
ncclResult_t IbCastCapProbeDevices();

// Cached results only; false for a device IbCastCapProbeDevices has not probed.
bool IbCastCapUdSupported(const struct ncclIbDev* dev);
bool IbCastCapRdmaReadSupported(const struct ncclIbDev* dev);

#endif // NET_IB_CAPABILITY_H_
