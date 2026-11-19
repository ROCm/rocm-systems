/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

#ifndef RCCL_TRANSPORT_MSEG_H_
#define RCCL_TRANSPORT_MSEG_H_

#include <stddef.h>
#include <stdint.h>

// RCCL: segment limits for multi-segment registrations. Upstream registers one MR for the whole range.

// RCCL: most segments one registration takes.
#define RCCL_IB_REG_SEG_DEFAULT 1024

// RCCL: the one count check. A HIP DMA-BUF export covers one segment, so each segment gets its own MR.
static inline bool rcclIbSegCountSupported(int nSeg) { return nSeg >= 1 && nSeg <= RCCL_IB_REG_SEG_DEFAULT; }

#endif // RCCL_TRANSPORT_MSEG_H_
