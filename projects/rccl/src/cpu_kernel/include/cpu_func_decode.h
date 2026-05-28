/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 ************************************************************************/

#ifndef RCCL_CPU_FUNC_DECODE_H_
#define RCCL_CPU_FUNC_DECODE_H_

#include "device.h"
#include "nccl_common.h"
#include "plugin/nccl_tuner.h"

#include <stdint.h>

struct rcclCpuFuncDesc {
  ncclFunc_t coll;
  int algo;
  int proto;
  ncclDevRedOp_t devRedOp;
  ncclDataType_t datatype;
  int acc;
  int pipeline;
  bool valid;
};

void rcclCpuFuncDecodeInit();
bool rcclCpuDecodeFuncId(unsigned funcId, struct rcclCpuFuncDesc* out);

#endif
