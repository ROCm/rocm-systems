/*************************************************************************
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT License.
 ************************************************************************/

#ifndef MSCCL_KERNEL_H_
#define MSCCL_KERNEL_H_

#define MSCCL_KERNEL_ENTRY_NAME(devredop, type, proto, fullOps) mscclKernel_##devredop##_##type##_##proto##_##fullOps

#define MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE_PROTO(devredop, type, proto, fullOps) \
__global__ void MSCCL_KERNEL_ENTRY_NAME(devredop, type, proto, fullOps)(struct ncclDevComm* comm, struct mscclAlgo* algo, struct mscclWork* work);

#define MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, type, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE_PROTO(devredop, type, LL, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE_PROTO(devredop, type, LL128, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE_PROTO(devredop, type, Simple, fullOps)

#define MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP(devredop, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, half, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, float, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, double, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, hip_bfloat16, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, rccl_float8, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, rccl_bfloat8, fullOps)

#define MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_NOFLOAT(devredop, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int8_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint8_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int32_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint32_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, int64_t, fullOps) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP_TYPE(devredop, uint64_t, fullOps)

#define MSCCL_DECL_KERNEL_ENTRY_FUNC() \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP(Sum, false) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP(Prod, false) \
  MSCCL_DECL_KERNEL_ENTRY_FUNC_DEVREDOP(MinMax, false)

MSCCL_DECL_KERNEL_ENTRY_FUNC()

#endif
