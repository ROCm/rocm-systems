/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 ************************************************************************/

/*
 * IR-only GIN API declarations - moved from src/include/nccl_device/gin.h
 * and nccl_device_wrapper__impl.h
 */

#ifndef _NCCL_IR_IMPL_GIN_H_
#define _NCCL_IR_IMPL_GIN_H_

#include <cstdint>

/* ncclGin_C wrapper initialization */
extern "C" __device__ void ncclGin_C_init(ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm,
                                           int contextIndex);
extern "C" __device__ void ncclGin_C_initWithResourceSharingMode(
    ncclGin_C* net, unsigned backendMask, ncclDevComm const& comm, int contextIndex,
    ncclGinResourceSharingMode resourceSharingMode);

/* GIN Put operations */
extern "C" __device__ void ncclGinPut(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                       ncclWindow_t srcWin, size_t srcOffset, size_t bytes, bool isSignal,
                                       ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                       bool isCounter, ncclGinCounter_t counterId, ncclCoopAny coop,
                                       bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                       cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);
extern "C" __device__ void ncclGinPut_v2(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                          ncclWindow_t srcWin, size_t srcOffset, size_t bytes, bool isSignal,
                                          ncclGinSignal_t signalId, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                          bool isCounter, ncclGinCounter_t counterId, ncclCoopAny coop,
                                          bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                          cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                          uint32_t optFlags);
extern "C" __device__ void ncclGinPut_v3(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                          ncclWindow_t srcWin, size_t srcOffset, size_t bytes, ncclGinSignalType signalType,
                                          ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId, bool isStrong,
                                          ncclGinSignalOp_t signalOp, uint64_t signalOpArg, bool isCounter, ncclGinCounter_t counterId,
                                          ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                          cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                          uint32_t optFlags, ncclGinSegmentType_t segmentType);
extern "C" __device__ void ncclGinPutValue(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                            uint64_t value, size_t size, bool isSignal, ncclGinSignal_t signalId,
                                            ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclCoopAny coop,
                                            bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                            cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);
extern "C" __device__ void ncclGinPutValue_v2(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                               uint64_t value, size_t size, bool isSignal, ncclGinSignal_t signalId,
                                               ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclCoopAny coop,
                                               bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                               cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                               uint32_t optFlags);
extern "C" __device__ void ncclGinPutValue_v3(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t dstWin, size_t dstOffset,
                                               uint64_t value, size_t size, ncclGinSignalType signalType,
                                               ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId,
                                               bool isStrong, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                               ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                               cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                               uint32_t optFlags);

/* GIN Get operations */
extern "C" __device__ void ncclGinGet(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t remoteWnd,
                                       size_t remoteOffset, ncclWindow_t localWnd, size_t localOffset, size_t bytes,
                                       ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                       uint32_t optFlags);
extern "C" __device__ void ncclGinGet_v2(ncclGin_C* net, ncclTeam team, int peer, ncclWindow_t remoteWnd,
                                          size_t remoteOffset, ncclWindow_t localWnd, size_t localOffset, size_t bytes,
                                          ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                          uint32_t optFlags, ncclGinSegmentType_t segmentType);

/* GIN Signal operations */
extern "C" __device__ void ncclGinSignal(ncclGin_C* net, ncclTeam team, int peer, bool isSignal, ncclGinSignal_t signalId,
                                          ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclCoopAny coop,
                                          bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                          cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease);
extern "C" __device__ void ncclGinSignal_v2(ncclGin_C* net, ncclTeam team, int peer, bool isSignal, ncclGinSignal_t signalId,
                                             ncclGinSignalOp_t signalOp, uint64_t signalOpArg, ncclCoopAny coop,
                                             bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                             cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                             uint32_t optFlags);
extern "C" __device__ void ncclGinSignal_v3(ncclGin_C* net, ncclTeam team, int peer, ncclGinSignalType signalType,
                                             ncclWindow_t signalWin, size_t signalOffset, ncclGinSignal_t signalId,
                                             bool isStrong, ncclGinSignalOp_t signalOp, uint64_t signalOpArg,
                                             ncclCoopAny coop, bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                             cuda::thread_scope givenRelease, cuda::thread_scope requiredRelease,
                                             uint32_t optFlags);

/* GIN Flush operations */
extern "C" __device__ void ncclGinFlush(ncclGin_C* net, ncclCoopAny coop, cuda::memory_order ord);
extern "C" __device__ void ncclGinFlush_v2(ncclGin_C* net, ncclCoopAny coop, cuda::memory_order ord,
                                            bool isDescriptor, ncclGinDescriptorSmem* descriptor);
extern "C" __device__ ncclResult_t ncclGinFlushTimeout(ncclGin_C* net, ncclCoopAny coop, cuda::memory_order ord,
                                                        bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                                        uint64_t timeoutCycles);
extern "C" __device__ void ncclGinFlushAsync(ncclGin_C* net, ncclTeam team, uint32_t peer, ncclGinRequest_t* outRequest,
                                              ncclCoopAny coop, uint32_t optFlags, bool isDescriptor,
                                              ncclGinDescriptorSmem* descriptor);

/* GIN Wait operations */
extern "C" __device__ void ncclGinWait(ncclGin_C* net, ncclGinRequest_t* request, ncclCoopAny coop,
                                        bool isDescriptor, ncclGinDescriptorSmem* descriptor, cuda::memory_order ord);
extern "C" __device__ ncclResult_t ncclGinWaitTimeout(ncclGin_C* net, ncclGinRequest_t* request, ncclCoopAny coop,
                                                       bool isDescriptor, ncclGinDescriptorSmem* descriptor,
                                                       cuda::memory_order ord, uint64_t timeoutCycles);

/* GIN Counter operations */
extern "C" __device__ uint64_t ncclGinReadCounter(ncclGin_C* net, ncclGinCounter_t counter, int bits,
                                                   cuda::memory_order ord);
extern "C" __device__ void ncclGinWaitCounter(ncclGin_C* net, ncclCoopAny coop, ncclGinCounter_t counter,
                                               uint64_t least, int bits, cuda::memory_order ord);
extern "C" __device__ ncclResult_t ncclGinWaitCounterTimeout(ncclGin_C* net, ncclCoopAny coop, ncclGinCounter_t counter,
                                                              uint64_t least, int bits, cuda::memory_order ord,
                                                              uint64_t timeoutCycles);
extern "C" __device__ void ncclGinResetCounter(ncclGin_C* net, ncclGinCounter_t counter);

/* GIN Signal reading */
extern "C" __device__ uint64_t* ncclGinGetSignalShadowPtr(ncclGin_C* net, ncclGinSignal_t signal);
extern "C" __device__ void ncclGinIncreaseSignalShadow(ncclGin_C* net, ncclGinSignal_t signal, uint64_t delta);
extern "C" __device__ uint64_t ncclGinReadSignal(ncclGin_C* net, ncclGinSignal_t signal, int bits,
                                                  cuda::memory_order ord);
extern "C" __device__ uint64_t ncclGinReadSignalVA(ncclGin_C* net, ncclWindow_t signalWindow, size_t signalOffset,
                                                    int bits, cuda::memory_order ord);

/* GIN Signal wait operations */
extern "C" __device__ void ncclGinWaitSignal(ncclGin_C* net, ncclCoopAny coop, ncclGinSignal_t signal,
                                              uint64_t least, int bits, cuda::memory_order ord);
extern "C" __device__ ncclResult_t ncclGinWaitSignalTimeout(ncclGin_C* net, ncclCoopAny coop, ncclGinSignal_t signal,
                                                             uint64_t least, int bits, cuda::memory_order ord,
                                                             uint64_t timeoutCycles);
extern "C" __device__ void ncclGinWaitSignalVA(ncclGin_C* net, ncclCoopAny coop, ncclWindow_t signalWindow,
                                                size_t signalOffset, uint64_t least, int bits, cuda::memory_order ord);
extern "C" __device__ ncclResult_t ncclGinWaitSignalTimeoutVA(ncclGin_C* net, ncclCoopAny coop,
                                                                ncclWindow_t signalWindow, size_t signalOffset,
                                                                uint64_t least, int bits, cuda::memory_order ord,
                                                                uint64_t timeoutCycles);
extern "C" __device__ void ncclGinWaitSignalMeetShadow(ncclGin_C* net, ncclCoopAny coop, ncclGinSignal_t signal,
                                                        int bits, cuda::memory_order ord);
extern "C" __device__ void ncclGinWaitSignalFollowShadow(ncclGin_C* net, ncclCoopAny coop, ncclGinSignal_t signal,
                                                          uint64_t leastDelta, uint64_t* before, uint64_t* delta,
                                                          int bits, cuda::memory_order ord);
extern "C" __device__ void ncclGinResetSignal(ncclGin_C* net, ncclGinSignal_t signal);
extern "C" __device__ void ncclGinResetSignalVA(ncclGin_C* net, ncclWindow_t signalWindow, size_t signalOffset);

/* GIN barrier session */
extern "C" __device__ void ncclGinBarrierSessionInit(ncclGinBarrierSession_C* session,
                                                       ncclCoopAny coop,
                                                       ncclGin_C const* net,
                                                       ncclTeam team,
                                                       ncclGinBarrierHandle handle,
                                                       uint32_t index);
extern "C" __device__ void ncclGinBarrierSessionSync(ncclGinBarrierSession_C* session,
                                                       ncclCoopAny coop,
                                                       cuda::memory_order order,
                                                       ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
extern "C" __device__ void ncclGinBarrierSessionDestroy(ncclGinBarrierSession_C* session);

/* Composite barrier session */
extern "C" __device__ void ncclBarrierSessionInit(ncclBarrierSession_C* session,
                                                    ncclCoopAny coop,
                                                    ncclTeam innerTeam,
                                                    ncclTeam outerTeam,
                                                    ncclGin_C const* net,
                                                    ncclLsaBarrierHandle const innerBarHandle,
                                                    ncclGinBarrierHandle const outerBarHandle,
                                                    uint32_t index,
                                                    bool multimem = false,
                                                    ncclMultimemHandle const innerMmHandle = {});
extern "C" __device__ void ncclBarrierSessionSync(ncclBarrierSession_C* session,
                                                    ncclCoopAny coop,
                                                    cuda::memory_order order,
                                                    ncclGinFenceLevel fence = ncclGinFenceLevel::Put | ncclGinFenceLevel::Get);
extern "C" __device__ void ncclBarrierSessionDestroy(ncclBarrierSession_C* session);

#endif // _NCCL_IR_IMPL_GIN_H_
