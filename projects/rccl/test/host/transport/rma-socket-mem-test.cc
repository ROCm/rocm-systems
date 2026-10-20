/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/transport/rma_socket_mem.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <future>

#include "ScopedHook.h"
#include "fakes/hip_fakes.h"
#include "fakes/nccl_fakes.h"
#include "transport/RmaSocketTest.h"

namespace {

static ncclResult_t DefaultGdrMapPointer(void*, size_t, gdr_mh_t*, void**, size_t*, gdr_info_t*, uint32_t) {
  return ncclInvalidUsage;
}

static ncclResult_t DefaultGdrUnmapPointer(gdr_mh_t, void*, size_t) { return ncclSuccess; }
static ncclResult_t DefaultGdrCopyToMapping(gdr_mh_t, void*, const void*, size_t) {
  return ncclInternalError;
}
static ncclResult_t DefaultGdrCopyFromMapping(gdr_mh_t, void*, const void*, size_t) {
  return ncclInternalError;
}

std::function<ncclResult_t(void*, size_t, gdr_mh_t*, void**, size_t*, gdr_info_t*, uint32_t)> g_testGdrMapPointer =
    DefaultGdrMapPointer;
std::function<ncclResult_t(gdr_mh_t, void*, size_t)> g_testGdrUnmapPointer = DefaultGdrUnmapPointer;
std::function<ncclResult_t(gdr_mh_t, void*, const void*, size_t)> g_testGdrCopyToMapping =
    DefaultGdrCopyToMapping;
std::function<ncclResult_t(gdr_mh_t, void*, const void*, size_t)> g_testGdrCopyFromMapping =
    DefaultGdrCopyFromMapping;

ncclResult_t TestGdrMapPointer(void* data, size_t size, gdr_mh_t* mh, void** map, size_t* mapSize, gdr_info_t* info,
                               uint32_t pinFlags = 0) {
  return g_testGdrMapPointer(data, size, mh, map, mapSize, info, pinFlags);
}

ncclResult_t TestGdrUnmapPointer(gdr_mh_t mh, void* map, size_t mapSize) {
  return g_testGdrUnmapPointer(mh, map, mapSize);
}

ncclResult_t TestGdrCopyToMapping(gdr_mh_t mh, void* map, const void* host, size_t size) {
  return g_testGdrCopyToMapping(mh, map, host, size);
}

ncclResult_t TestGdrCopyFromMapping(gdr_mh_t mh, void* host, const void* map, size_t size) {
  return g_testGdrCopyFromMapping(mh, host, map, size);
}

}  // namespace

#define ncclGdrCudaMapPointer TestGdrMapPointer
#define ncclGdrCudaUnmapPointer TestGdrUnmapPointer
#define wrap_gdr_copy_to_mapping TestGdrCopyToMapping
#define wrap_gdr_copy_from_mapping TestGdrCopyFromMapping
#include RMA_SOCKET_MEM_CC_PATH
#undef wrap_gdr_copy_from_mapping
#undef wrap_gdr_copy_to_mapping
#undef ncclGdrCudaUnmapPointer
#undef ncclGdrCudaMapPointer

// Owned by src/init.cc. This binary does not compile init.cc; the socket-RMA
// units use the real discriminator exposed by gdrwrap.h.
gdr_t ncclGdrCopy = nullptr;

namespace {

using rma_socket_test::Scene;

class RmaSocketMemMicrotest : public ::testing::Test {
 protected:
  void SetUp() override { g_hipAsyncOpsResult = hipSuccess; }

  void TearDown() override {
    ncclGdrCopy = nullptr;
    g_testGdrMapPointer = DefaultGdrMapPointer;
    g_testGdrUnmapPointer = DefaultGdrUnmapPointer;
    g_testGdrCopyToMapping = DefaultGdrCopyToMapping;
    g_testGdrCopyFromMapping = DefaultGdrCopyFromMapping;
    ResetHipFakes();
    ResetNcclFakes();
  }
};

TEST_F(RmaSocketMemMicrotest, SupportedPtrType_HostAndCudaAreSupported) {
  EXPECT_TRUE(ncclRmaSocketProxySupportedPtrType(NCCL_PTR_HOST));
  EXPECT_TRUE(ncclRmaSocketProxySupportedPtrType(NCCL_PTR_CUDA));
  EXPECT_FALSE(ncclRmaSocketProxySupportedPtrType(NCCL_PTR_DMABUF));
  EXPECT_FALSE(ncclRmaSocketProxySupportedPtrType(0));
}

TEST_F(RmaSocketMemMicrotest, ValidateRange_ContainedAndEmptyBoundaryRangesSucceed) {
  EXPECT_EQ(ncclRmaSocketProxyValidateRange(16, 4, 12, "object"), ncclSuccess);
  EXPECT_EQ(ncclRmaSocketProxyValidateRange(16, 16, 0, "object"), ncclSuccess);
}

TEST_F(RmaSocketMemMicrotest, ValidateRange_OffsetOrExtentPastObjectIsRejected) {
  EXPECT_EQ(ncclRmaSocketProxyValidateRange(16, 17, 0, "object"), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidateRange(16, 15, 2, "object"), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyValidateRange(0, UINT64_MAX, 1, "object"), ncclInvalidArgument);
}

TEST_F(RmaSocketMemMicrotest, GdrPtr_MappedRangeReturnsBarPointerAtMatchingOffset) {
  std::array<unsigned char, 32> data{};
  std::array<unsigned char, 32> bar{};
  ncclRmaSocketProxyMrHandle handle{};
  handle.data = data.data();
  handle.size = data.size();
  handle.gdr.mapped = 1;
  handle.gdr.barPtr = bar.data();
  void* result = nullptr;

  EXPECT_EQ(ncclRmaSocketProxyGdrPtr(&handle, data.data() + 7, 5, &result), ncclSuccess);
  EXPECT_EQ(result, bar.data() + 7);
}

TEST_F(RmaSocketMemMicrotest, GdrPtr_InvalidHandleMappingOrRangeIsRejectedAndClearsOutput) {
  std::array<unsigned char, 16> data{};
  std::array<unsigned char, 16> bar{};
  ncclRmaSocketProxyMrHandle handle{};
  handle.data = data.data();
  handle.size = data.size();
  void* result = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyGdrPtr(nullptr, data.data(), 1, &result), ncclInternalError);
  EXPECT_EQ(result, nullptr);
  handle.gdr.mapped = 1;
  handle.gdr.barPtr = bar.data();
  const void* beforeBase = reinterpret_cast<const void*>(reinterpret_cast<uintptr_t>(data.data()) - 1);
  EXPECT_EQ(ncclRmaSocketProxyGdrPtr(&handle, beforeBase, 1, &result), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyGdrPtr(&handle, data.data() + 15, 2, &result), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyGdrPtr(&handle, data.data(), 1, nullptr), ncclInvalidArgument);
}

TEST_F(RmaSocketMemMicrotest, Copy_EmptyCopySucceedsWithoutInspectingPointersOrTypes) {
  EXPECT_EQ(ncclRmaSocketProxyCopy(nullptr, -1, nullptr, nullptr, -1, nullptr, 0), ncclSuccess);
}

TEST_F(RmaSocketMemMicrotest, Copy_HostToHostCopiesRequestedBytes) {
  const std::array<unsigned char, 5> source{{1, 2, 3, 4, 5}};
  std::array<unsigned char, 5> destination{};

  EXPECT_EQ(ncclRmaSocketProxyCopy(destination.data(), NCCL_PTR_HOST, nullptr, source.data(), NCCL_PTR_HOST, nullptr,
                                  source.size()),
            ncclSuccess);
  EXPECT_EQ(destination, source);
}

TEST_F(RmaSocketMemMicrotest, Copy_CudaDirectionWithoutMatchingHandleReturnsError) {
  std::array<unsigned char, 8> bytes{};

  EXPECT_EQ(ncclRmaSocketProxyCopy(bytes.data(), NCCL_PTR_HOST, nullptr, bytes.data(), NCCL_PTR_CUDA, nullptr, 1),
            ncclInternalError);
  EXPECT_EQ(ncclRmaSocketProxyCopy(bytes.data(), NCCL_PTR_CUDA, nullptr, bytes.data(), NCCL_PTR_HOST, nullptr, 1),
            ncclInternalError);
}

TEST_F(RmaSocketMemMicrotest, Copy_CudaDirectionsPropagateCopyErrorAfterResolvingMapping) {
  alignas(uint64_t) std::array<unsigned char, 8> data{};
  alignas(uint64_t) std::array<unsigned char, 8> bar{};
  std::array<unsigned char, 8> host{};
  ncclRmaSocketProxyMrHandle handle{};
  handle.data = data.data();
  handle.size = data.size();
  handle.type = NCCL_PTR_CUDA;
  handle.gdr.mapped = 1;
  handle.gdr.barPtr = bar.data();
  ScopedHook copyFrom(g_testGdrCopyFromMapping, [](gdr_mh_t, void*, const void*, size_t) {
    return ncclInternalError;
  });
  ScopedHook copyTo(g_testGdrCopyToMapping, [](gdr_mh_t, void*, const void*, size_t) {
    return ncclInternalError;
  });

  EXPECT_EQ(ncclRmaSocketProxyCopy(host.data(), NCCL_PTR_HOST, nullptr, data.data(), NCCL_PTR_CUDA, &handle, 1),
            ncclInternalError);
  EXPECT_EQ(ncclRmaSocketProxyCopy(data.data(), NCCL_PTR_CUDA, &handle, host.data(), NCCL_PTR_HOST, nullptr, 1),
            ncclInternalError);
  EXPECT_EQ(copyFrom.calls, 1);
  EXPECT_EQ(copyTo.calls, 1);
}

TEST_F(RmaSocketMemMicrotest, Copy_CudaToHostUsesFromMapping) {
  std::array<unsigned char, 4> data{};
  const std::array<unsigned char, 4> bar{{1, 2, 3, 4}};
  std::array<unsigned char, 4> host{};
  ncclRmaSocketProxyMrHandle handle{};
  handle.data = data.data();
  handle.size = data.size();
  handle.type = NCCL_PTR_CUDA;
  handle.gdr.mapped = 1;
  handle.gdr.barPtr = const_cast<unsigned char*>(bar.data());
  ScopedHook copyFrom(g_testGdrCopyFromMapping, [&bar](gdr_mh_t, void* dst, const void* src, size_t size) {
    EXPECT_EQ(src, bar.data());
    std::memcpy(dst, src, size);
    return ncclSuccess;
  });
  ScopedHook copyTo(g_testGdrCopyToMapping, [](gdr_mh_t, void*, const void*, size_t) {
    ADD_FAILURE() << "GPU-to-host must not use copy-to-mapping";
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyCopy(host.data(), NCCL_PTR_HOST, nullptr, data.data(), NCCL_PTR_CUDA, &handle,
                                  data.size()),
            ncclSuccess);
  EXPECT_EQ(host, bar);
  EXPECT_EQ(copyFrom.calls, 1);
  EXPECT_EQ(copyTo.calls, 0);
}

TEST_F(RmaSocketMemMicrotest, Copy_HostToCudaUsesToMapping) {
  std::array<unsigned char, 4> data{};
  std::array<unsigned char, 4> bar{};
  const std::array<unsigned char, 4> host{{5, 6, 7, 8}};
  ncclRmaSocketProxyMrHandle handle{};
  handle.data = data.data();
  handle.size = data.size();
  handle.type = NCCL_PTR_CUDA;
  handle.gdr.mapped = 1;
  handle.gdr.barPtr = bar.data();
  ScopedHook copyFrom(g_testGdrCopyFromMapping, [](gdr_mh_t, void*, const void*, size_t) {
    ADD_FAILURE() << "host-to-GPU must not use copy-from-mapping";
    return ncclSuccess;
  });
  ScopedHook copyTo(g_testGdrCopyToMapping, [](gdr_mh_t, void* dst, const void* src, size_t size) {
    std::memcpy(dst, src, size);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyCopy(data.data(), NCCL_PTR_CUDA, &handle, host.data(), NCCL_PTR_HOST, nullptr,
                                  host.size()),
            ncclSuccess);
  EXPECT_EQ(bar, host);
  EXPECT_EQ(copyFrom.calls, 0);
  EXPECT_EQ(copyTo.calls, 1);
}

TEST_F(RmaSocketMemMicrotest, Copy_UnsupportedDirectionIsRejected) {
  std::array<unsigned char, 8> bytes{};
  ncclRmaSocketProxyMrHandle handle{};

  EXPECT_EQ(ncclRmaSocketProxyCopy(bytes.data(), NCCL_PTR_CUDA, &handle, bytes.data(), NCCL_PTR_CUDA, &handle, 1),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMemMicrotest, AllocAndFreeChunkBuffer_UseHostAllocationLifecycle) {
  void* ptr = nullptr;
  ScopedHook alloc(g_hipHostMalloc, [](void** out, size_t size, unsigned flags) {
    EXPECT_EQ(size, 64u);
    unsigned expectedFlags = hipHostMallocMapped;
#if defined(HIP_HOST_UNCACHED_MEMORY)
    expectedFlags |= hipHostMallocUncached;
#endif
    EXPECT_EQ(flags, expectedFlags);
    *out = std::calloc(1, size);
    return *out == nullptr ? hipErrorOutOfMemory : hipSuccess;
  });
  ScopedHook release(g_hipHostFree, [](void* p) {
    std::free(p);
    return hipSuccess;
  });

  ASSERT_EQ(ncclRmaSocketProxyAllocChunkBuffer(&ptr, 64), ncclSuccess);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(ncclRmaSocketProxyFreeChunkBuffer(ptr), ncclSuccess);
  EXPECT_EQ(ncclRmaSocketProxyFreeChunkBuffer(nullptr), ncclSuccess);
  EXPECT_EQ(alloc.calls, 1);
  EXPECT_EQ(release.calls, 1);
}

TEST_F(RmaSocketMemMicrotest, RegMrSym_InvalidArgumentsOrPointerTypeAreRejected) {
  Scene scene;
  int data = 0;
  void* handle = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyRegMrSym(nullptr, &data, sizeof(data), NCCL_PTR_HOST, 0, &handle),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyRegMrSym(&scene.comm, &data, sizeof(data), NCCL_PTR_HOST, 0, nullptr),
            ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyRegMrSym(&scene.comm, &data, sizeof(data), NCCL_PTR_DMABUF, 0, &handle),
            ncclInvalidArgument);
}

TEST_F(RmaSocketMemMicrotest, FreeMr_MappedHandleIsUnmappedWhenGdrCopyIsActive) {
  auto* handle = static_cast<ncclRmaSocketProxyMrHandle*>(std::calloc(1, sizeof(ncclRmaSocketProxyMrHandle)));
  ASSERT_NE(handle, nullptr);
  handle->gdr.mapped = 1;
  handle->gdr.map = reinterpret_cast<void*>(0x1000);
  handle->gdr.mapSize = 4096;
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  ScopedHook unmap(g_testGdrUnmapPointer, [](gdr_mh_t, void* map, size_t size) {
    EXPECT_EQ(map, reinterpret_cast<void*>(0x1000));
    EXPECT_EQ(size, 4096u);
    return ncclSuccess;
  });

  EXPECT_EQ(ncclRmaSocketProxyFreeMr(handle), ncclSuccess);
  EXPECT_EQ(ncclRmaSocketProxyFreeMr(nullptr), ncclSuccess);
  EXPECT_EQ(unmap.calls, 1);
}

TEST_F(RmaSocketMemMicrotest, RegMrSym_HostRegistrationPublishesOrderedHandleAndDeregisterRemovesIt) {
  Scene scene;
  int firstData = 0;
  int secondData = 0;
  void* firstOpaque = nullptr;
  void* secondOpaque = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyRegMrSym(&scene.comm, &firstData, sizeof(firstData), NCCL_PTR_HOST, 0, &firstOpaque),
            ncclSuccess);
  ASSERT_EQ(ncclRmaSocketProxyRegMrSym(&scene.comm, &secondData, sizeof(secondData), NCCL_PTR_HOST, 0, &secondOpaque),
            ncclSuccess);
  auto* first = static_cast<ncclRmaSocketProxyMrHandle*>(firstOpaque);
  auto* second = static_cast<ncclRmaSocketProxyMrHandle*>(secondOpaque);

  EXPECT_EQ(first->data, &firstData);
  EXPECT_EQ(first->size, sizeof(firstData));
  EXPECT_EQ(first->type, NCCL_PTR_HOST);
  EXPECT_EQ(first->mrId, 0u);
  EXPECT_EQ(second->mrId, 1u);
  EXPECT_EQ(scene.comm.mrHead, second);
  EXPECT_EQ(second->next, first);
  EXPECT_EQ(scene.comm.nextMrId, 2u);

  EXPECT_EQ(ncclRmaSocketProxyDeregMrSym(&scene.comm, first), ncclSuccess);
  EXPECT_EQ(scene.comm.mrHead, second);
  EXPECT_EQ(second->next, nullptr);
  EXPECT_EQ(ncclRmaSocketProxyDeregMrSym(&scene.comm, second), ncclSuccess);
  EXPECT_EQ(scene.comm.mrHead, nullptr);
}

TEST_F(RmaSocketMemMicrotest, DeregMrSym_ActiveReceiveKeepsMrRegisteredUntilReleased) {
  Scene scene;
  int data = 0;
  void* opaque = nullptr;
  ASSERT_EQ(ncclRmaSocketProxyRegMrSym(&scene.comm, &data, sizeof(data), NCCL_PTR_HOST, 0, &opaque), ncclSuccess);
  auto* handle = static_cast<ncclRmaSocketProxyMrHandle*>(opaque);
  ASSERT_NE(handle, nullptr);
  ncclAtomicRefCountIncrement(&handle->recvActiveOps);
  std::promise<void> entered;
  auto enteredFuture = entered.get_future();
  auto deregister = std::async(std::launch::async, [&] {
    entered.set_value();
    return ncclRmaSocketProxyDeregMrSym(&scene.comm, handle);
  });
  enteredFuture.wait();

  EXPECT_EQ(deregister.wait_for(std::chrono::milliseconds(10)), std::future_status::timeout);
  EXPECT_EQ(scene.comm.mrHead, handle);

  ncclAtomicRefCountDecrement(&handle->recvActiveOps);
  EXPECT_EQ(deregister.get(), ncclSuccess);
  EXPECT_EQ(scene.comm.mrHead, nullptr);
}

TEST_F(RmaSocketMemMicrotest, RegMrSym_EmptyCudaRegistrationDoesNotAttemptUnsupportedMapping) {
  Scene scene;
  void* opaque = nullptr;

  ASSERT_EQ(ncclRmaSocketProxyRegMrSym(&scene.comm, nullptr, 0, NCCL_PTR_CUDA, 0, &opaque), ncclSuccess);
  auto* handle = static_cast<ncclRmaSocketProxyMrHandle*>(opaque);
  EXPECT_EQ(handle->type, NCCL_PTR_CUDA);
  EXPECT_EQ(handle->gdr.mapped, 0);

  EXPECT_EQ(ncclRmaSocketProxyDeregMrSym(&scene.comm, handle), ncclSuccess);
}

TEST_F(RmaSocketMemMicrotest, RegMrSym_NonEmptyCudaRegistrationPropagatesMappingFailure) {
  Scene scene;
  int data = 0;
  void* opaque = reinterpret_cast<void*>(0x1);

  EXPECT_EQ(ncclRmaSocketProxyRegMrSym(&scene.comm, &data, sizeof(data), NCCL_PTR_CUDA, 0, &opaque),
            ncclInvalidUsage);
  EXPECT_EQ(opaque, nullptr);
  EXPECT_EQ(scene.comm.mrHead, nullptr);
  EXPECT_EQ(scene.comm.nextMrId, 0u);
}

TEST_F(RmaSocketMemMicrotest, RegMrSym_CudaMappingSuccessPublishesMappedHandle) {
  Scene scene;
  alignas(uint64_t) std::array<unsigned char, 16> data{};
  alignas(uint64_t) std::array<unsigned char, 32> bar{};
  void* opaque = nullptr;
  ncclGdrCopy = reinterpret_cast<gdr_t>(0x1);
  ScopedHook map(g_testGdrMapPointer,
                 [&data, &bar](void* ptr, size_t size, gdr_mh_t* mh, void** mapped, size_t* mappedSize,
                               gdr_info_t* info, uint32_t pinFlags) {
                   EXPECT_EQ(ptr, data.data());
                   EXPECT_EQ(size, data.size());
                   EXPECT_EQ(pinFlags, 0u);
                   mh->h = 17;
                   *mapped = bar.data();
                   *mappedSize = bar.size();
                   *info = gdr_info_t{};
                   info->va = reinterpret_cast<uint64_t>(data.data()) - 4;
                   return ncclSuccess;
                 });
  ScopedHook unmap(g_testGdrUnmapPointer, [](gdr_mh_t, void*, size_t) { return ncclSuccess; });

  ASSERT_EQ(ncclRmaSocketProxyRegMrSym(&scene.comm, data.data(), data.size(), NCCL_PTR_CUDA, 0, &opaque),
            ncclSuccess);
  auto* handle = static_cast<ncclRmaSocketProxyMrHandle*>(opaque);
  ASSERT_NE(handle, nullptr);
  EXPECT_EQ(handle->gdr.mapped, 1);
  EXPECT_EQ(handle->gdr.map, bar.data());
  EXPECT_EQ(handle->gdr.barPtr, bar.data() + 4);
  EXPECT_EQ(handle->gdr.mapSize, bar.size());
  EXPECT_EQ(map.calls, 1);

  EXPECT_EQ(ncclRmaSocketProxyDeregMrSym(&scene.comm, handle), ncclSuccess);
  EXPECT_EQ(unmap.calls, 1);
}

TEST_F(RmaSocketMemMicrotest, DeregMrSym_InvalidOrUnknownHandleIsRejected) {
  Scene scene;
  ncclRmaSocketProxyMrHandle unknown{};

  EXPECT_EQ(ncclRmaSocketProxyDeregMrSym(nullptr, &unknown), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyDeregMrSym(&scene.comm, nullptr), ncclInvalidArgument);
  EXPECT_EQ(ncclRmaSocketProxyDeregMrSym(&scene.comm, &unknown), ncclInvalidArgument);
}

TEST_F(RmaSocketMemMicrotest, RegMrSymDmaBuf_AlwaysReportsUnsupported) {
  EXPECT_EQ(ncclRmaSocketProxyRegMrSymDmaBuf(nullptr, nullptr, 0, NCCL_PTR_HOST, 0, -1, 0, nullptr),
            ncclInvalidUsage);
}

}  // namespace
