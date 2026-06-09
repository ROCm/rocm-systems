/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef HIP_LOADER_LOADER_V6_COMPAT_H
#define HIP_LOADER_LOADER_V6_COMPAT_H

#include <stddef.h>

typedef struct HIP_MEMSET_NODE_PARAMS {
  void* dst;
  size_t pitch;
  unsigned int value;
  unsigned int elementSize;
  size_t width;
  size_t height;
} HIP_MEMSET_NODE_PARAMS;

typedef hipError_t (*hip_loader_backend_GetDevicePropertiesR0600_fn)(hipDeviceProp_tR0600* prop,
                                                                     int device);
typedef hipError_t (*hip_loader_backend_ChooseDeviceR0600_fn)(int* device,
                                                              const hipDeviceProp_tR0600* prop);
typedef hipError_t (*hip_loader_backend_DrvGraphAddMemsetNode_fn)(
    hipGraphNode_t* phGraphNode, hipGraph_t hGraph, const hipGraphNode_t* dependencies,
    size_t numDependencies, const hipMemsetParams* memsetParams, hipCtx_t ctx);
typedef hipError_t (*hip_loader_backend_DrvGraphExecMemsetNodeSetParams_fn)(
    hipGraphExec_t hGraphExec, hipGraphNode_t hNode, const hipMemsetParams* memsetParams,
    hipCtx_t ctx);

hipError_t hip_loader_v6_compat_GetDevicePropertiesR0000(
    hip_loader_backend_GetDevicePropertiesR0600_fn backend_fn, hipDeviceProp_tR0000* prop,
    int device);
hipError_t hip_loader_v6_compat_ChooseDeviceR0000(
    hip_loader_backend_ChooseDeviceR0600_fn backend_fn, int* device,
    const hipDeviceProp_tR0000* prop);
hipError_t hip_loader_v6_compat_DrvGraphAddMemsetNode(
    hip_loader_backend_DrvGraphAddMemsetNode_fn backend_fn, hipGraphNode_t* phGraphNode,
    hipGraph_t hGraph, const hipGraphNode_t* dependencies, size_t numDependencies,
    const HIP_MEMSET_NODE_PARAMS* memsetParams, hipCtx_t ctx);
hipError_t hip_loader_v6_compat_DrvGraphExecMemsetNodeSetParams(
    hip_loader_backend_DrvGraphExecMemsetNodeSetParams_fn backend_fn, hipGraphExec_t hGraphExec,
    hipGraphNode_t hNode, const HIP_MEMSET_NODE_PARAMS* memsetParams, hipCtx_t ctx);

#endif
