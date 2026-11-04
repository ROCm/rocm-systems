/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef _NCCL_DEVICE_GIN_ABCE_SDMA_PUT_POLICY_H_
#define _NCCL_DEVICE_GIN_ABCE_SDMA_PUT_POLICY_H_

// The put-segmentation policy (namespace gin_sdma: kGinPutSegBytes,
// ginPutSegmentCount/At, PutSegment) is backend-agnostic SDMA math shared with
// the anvil-sdma backend. Both the anvil and abce device headers can be pulled
// into one TU via gin_device_api.h, so the abce backend must REUSE the single
// gin_sdma definition rather than redefine it (that would be an ODR violation /
// redefinition error). Include the anvil copy as the one owner of gin_sdma.
#include "../anvil_sdma/gin_anvil_sdma_put_policy.h"

#endif  // _NCCL_DEVICE_GIN_ABCE_SDMA_PUT_POLICY_H_
