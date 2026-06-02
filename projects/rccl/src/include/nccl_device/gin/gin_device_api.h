/*************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
#ifndef _NCCL_GIN_DEVICE_API_H_
#define _NCCL_GIN_DEVICE_API_H_

#include "gin_device_common.h"

#if NCCL_GIN_GDAKI_ENABLE
#include "gdaki/gin_gdaki.h"
#endif
#if NCCL_GIN_PROXY_ENABLE
#include "proxy/gin_proxy.h"
#endif
#if NCCL_GIN_ROCSHMEM_ENABLE
#include "rocshmem/gin_rocshmem.h"
#endif
#if NCCL_GIN_ANVIL_ENABLE
#include "anvil/gin_anvil.h"
#endif

#endif
