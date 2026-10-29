/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 ************************************************************************/
#include "nccl_m2n.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int worldRank;
static void fail(const char* what) {
  std::fprintf(stderr, "rank %d FAIL %s; m2n=%s\n", worldRank, what, ncclM2nGetLastError());
  MPI_Abort(MPI_COMM_WORLD, 1); std::abort();
}
#define CUDA_OK(call) do { if ((call) != cudaSuccess) fail(#call); } while (0)
#define NCCL_OK(call) do { ncclResult_t result = (call); if (result != ncclSuccess && result != ncclInProgress) { std::fprintf(stderr, "%s: %s\n", #call, ncclGetErrorString(result)); fail(#call); } } while (0)
#define MPI_OK(call) do { if ((call) != MPI_SUCCESS) fail(#call); } while (0)

static size_t elements(const ncclDistTensor_t& t) {
  size_t n = 1; for (int d = 0; d < t.ndims; d++) n *= t.localShape[d]; return n;
}
static ncclDistTensor_t tensor(ncclMesh_t* mesh, int ndims, std::array<int, 2> placement,
                                const size_t global[], void* data, ncclDataType_t type,
                                size_t localShape[], int placements[]) {
  ncclDistTensor_t t = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  t.localShape = localShape; t.placements = placements; t.mesh = mesh; t.ndims = ndims; t.dtype = type;
  t.dataPtr = worldRank >= mesh->startRank && worldRank < mesh->startRank + mesh->dims[0]*mesh->dims[1] ? data : nullptr;
  for (int d = 0; d < ndims; d++) t.localShape[d] = global[d];
  for (int axis = 0; axis < 2; axis++) {
    t.placements[axis] = placement[axis];
    if (placement[axis] >= 0) t.localShape[placement[axis]] /= mesh->dims[axis];
  }
  return t;
}
// Oracle: recursively partition the global tensor in placement order; generate
// each byte from its global element index, byte lane, and iteration salt.
static std::vector<unsigned char> expected(const ncclDistTensor_t& t, const size_t global[], size_t elementBytes,
                                           unsigned salt) {
  size_t begin[3]{}, extent[3]{};
  std::copy(global, global + t.ndims, extent);
  int localRank = worldRank - t.mesh->startRank;
  int coordinate[2] = {localRank / t.mesh->dims[1], localRank % t.mesh->dims[1]};
  for (int a = 0; a < 2; a++) if (t.placements[a] >= 0) {
    int d = t.placements[a]; extent[d] /= t.mesh->dims[a]; begin[d] += coordinate[a]*extent[d];
  }
  size_t n = elements(t);
  std::vector<unsigned char> bytes(n * elementBytes);
  for (size_t i = 0; i < n; i++) {
    size_t remainder = i, index = 0, stride = 1;
    for (int d = t.ndims - 1; d >= 0; d--) {
      index += (begin[d] + remainder % t.localShape[d]) * stride;
      remainder /= t.localShape[d]; stride *= global[d];
    }
    for (size_t lane = 0; lane < elementBytes; lane++) {
      uint64_t hash = index + 0x9e3779b97f4a7c15ULL * (salt + 1) + lane * 0xd6e8feb86659fd93ULL;
      hash ^= hash >> 30; hash *= 0xbf58476d1ce4e5b9ULL; hash ^= hash >> 27;
      hash *= 0x94d049bb133111ebULL; hash ^= hash >> 31;
      bytes[i * elementBytes + lane] = static_cast<unsigned char>(hash);
    }
  }
  return bytes;
}

int main(int argc, char** argv) {
  MPI_OK(MPI_Init(&argc, &argv));
  int worldSize = 0; MPI_OK(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank)); MPI_OK(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
  // LCM of supported mesh products permits asymmetric, non-square meshes.
  const size_t small[3] = {24, 24, 24};
  const size_t large[3] = {48, 96, 192};
  bool windowApi = false, grouped = false, nonblocking = false;
  int srcRanks = worldSize / 2, rows = 2, dstRows = 2;
  for (int i = 1; i < argc; i++) {
    if (!std::strcmp(argv[i], "--window")) windowApi = true;
    else if (!std::strcmp(argv[i], "--grouped")) grouped = true;
    else if (!std::strcmp(argv[i], "--nonblocking")) nonblocking = true;
    else if (!std::strcmp(argv[i], "--src-ranks") && i + 1 < argc) srcRanks = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--src-rows") && i + 1 < argc) rows = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--dst-rows") && i + 1 < argc) dstRows = std::atoi(argv[++i]);
    else fail("unknown argument");
  }
  if (worldSize < 4 || srcRanks < 1 || srcRanks >= worldSize || rows < 1 || dstRows < 1 ||
      srcRanks % rows || (worldSize - srcRanks) % dstRows) fail("invalid mesh sizes");
  const int dstRanks = worldSize - srcRanks;
  for (int d = 0; d < 3; d++) {
    if (small[d] % srcRanks || small[d] % dstRanks || large[d] % srcRanks || large[d] % dstRanks) {
      if (worldRank == 0) {
        std::fprintf(stderr, "srcRanks=%d dstRanks=%d tensorDim=%d small=%zu large=%zu\n",
                     srcRanks, dstRanks, d, small[d], large[d]);
      }
      fail("mesh rank counts must evenly divide built-in test shapes");
    }
  }
  MPI_Comm local; MPI_OK(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local));
  int localRank = 0; MPI_OK(MPI_Comm_rank(local, &localRank));
  int devices = 0; CUDA_OK(cudaGetDeviceCount(&devices)); if (localRank >= devices) fail("too few GPUs");
  CUDA_OK(cudaSetDevice(localRank));
  ncclUniqueId id{}; if (worldRank == 0) NCCL_OK(ncclGetUniqueId(&id));
  MPI_OK(MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));
  ncclComm_t comm{}; ncclConfig_t config = NCCL_CONFIG_INITIALIZER; config.blocking = nonblocking ? 0 : 1;
  ncclResult_t init = ncclCommInitRankConfig(&comm, worldSize, id, worldRank, &config);
  if (init != ncclSuccess && init != ncclInProgress) fail("comm initialization");
  auto ready = [&]() { ncclResult_t state; do { NCCL_OK(ncclCommGetAsyncError(comm, &state)); } while (state == ncclInProgress);
                       if (state != ncclSuccess) fail("communicator async error"); };
  ready();
  cudaStream_t stream{}; CUDA_OK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  const size_t region = 8 * 1024 * 1024, capacity = 2 * region;
  void* buffer{}; NCCL_OK(ncclMemAlloc(&buffer, capacity));
  ncclWindow_t window{};
  if (windowApi) { NCCL_OK(ncclCommWindowRegister(comm, buffer, capacity, &window, NCCL_WIN_COLL_SYMMETRIC)); ready(); }
  ncclM2nHandle_t handle{}; NCCL_OK(ncclM2nInit(&handle, nullptr));
  int srcMeshDims[2]{rows, srcRanks / rows}, dstMeshDims[2]{dstRows, (worldSize - srcRanks) / dstRows};
  ncclMesh_t sm = NCCL_M2N_MESH_INITIALIZER, dm = NCCL_M2N_MESH_INITIALIZER;
  sm.ndims = dm.ndims = 2; sm.dims = srcMeshDims; dm.dims = dstMeshDims; dm.startRank = srcRanks;
  unsigned cases = 0;
  auto run = [&](int ndims, std::array<int, 2> sp, std::array<int, 2> dp, const size_t global[], size_t itemBytes) {
    ncclDataType_t type = itemBytes == 1 ? ncclUint8 : ncclInt32;
    size_t srcShape[3]{}, dstShape[3]{};
    int srcPlacements[2]{}, dstPlacements[2]{};
    ncclDistTensor_t src = tensor(&sm, ndims, sp, global, buffer, type, srcShape, srcPlacements);
    ncclDistTensor_t dst = tensor(&dm, ndims, dp, global, buffer, type, dstShape, dstPlacements);
    const ncclDistTensor_t& own = src.dataPtr ? src : dst;
    size_t bytes = elements(own) * itemBytes;
    if (bytes + 64 > region) fail("test buffer too small");
    unsigned iterations = grouped ? 2 : 1;
    for (unsigned j = 0; j < iterations; j++) {
      auto* ptr = static_cast<unsigned char*>(buffer) + j * region;
      CUDA_OK(cudaMemsetAsync(ptr, 0xcd, bytes + 64, stream));
      if (src.dataPtr) {
        auto input = expected(src, global, itemBytes, cases * 3 + j);
        CUDA_OK(cudaMemcpyAsync(ptr, input.data(), bytes, cudaMemcpyHostToDevice, stream));
        CUDA_OK(cudaStreamSynchronize(stream));
      }
    }
    CUDA_OK(cudaStreamSynchronize(stream)); MPI_OK(MPI_Barrier(MPI_COMM_WORLD));
    if (grouped) NCCL_OK(ncclM2nGroupStart());
    for (unsigned j = 0; j < iterations; j++) {
      auto s = src, d = dst;
      if (s.dataPtr) s.dataPtr = static_cast<unsigned char*>(buffer) + j * region;
      if (d.dataPtr) d.dataPtr = static_cast<unsigned char*>(buffer) + j * region;
      if (windowApi) NCCL_OK(ncclReshardWithWindow(handle, comm, window, &s, &d, stream));
      else NCCL_OK(ncclReshard(handle, comm, &s, &d, stream));
    }
    if (grouped) NCCL_OK(ncclM2nGroupEnd());
    ready(); CUDA_OK(cudaStreamSynchronize(stream));
    for (unsigned j = 0; j < iterations; j++) {
      auto* ptr = static_cast<unsigned char*>(buffer) + j * region;
      std::vector<unsigned char> actual(bytes + 64);
      CUDA_OK(cudaMemcpy(actual.data(), ptr, actual.size(), cudaMemcpyDeviceToHost));
      auto wanted = expected(own, global, itemBytes, cases * 3 + j);
      if (!std::equal(wanted.begin(), wanted.end(), actual.begin()) ||
          !std::all_of(actual.begin() + bytes, actual.end(), [](unsigned char b) { return b == 0xcd; })) {
        std::fprintf(stderr, "case=%u ndims=%d src=[%d,%d] dst=[%d,%d] item=%zu grouped=%d\n",
                     cases, ndims, sp[0], sp[1], dp[0], dp[1], itemBytes, grouped);
        fail("tensor contents or guard bytes mismatch");
      }
    }
    MPI_OK(MPI_Barrier(MPI_COMM_WORLD)); cases++;
    if (worldRank == 0 && cases % 64 == 0) { std::printf("PROGRESS cases=%u\n", cases); std::fflush(stdout); }
  };
  for (int n = 1; n <= 3; n++)
    for (int s0 = -1; s0 < n; s0++) for (int s1 = -1; s1 < n; s1++)
    for (int d0 = -1; d0 < n; d0++) for (int d1 = -1; d1 < n; d1++)
      run(n, {s0, s1}, {d0, d1}, small, cases % 2 == 0 ? 1 : 4);
  run(3, {0, 1}, {1, 2}, large, 4);
  run(3, {2, 2}, {0, 1}, large, 4);
  run(3, {0, 2}, {2, 0}, large, 1);
  NCCL_OK(ncclM2nFinalize(handle));
  if (windowApi) { NCCL_OK(ncclCommWindowDeregister(comm, window)); ready(); }
  NCCL_OK(ncclCommDestroy(comm)); NCCL_OK(ncclMemFree(buffer)); CUDA_OK(cudaStreamDestroy(stream));
  MPI_OK(MPI_Comm_free(&local));
  if (worldRank == 0) std::printf("MULTI_DIM_GPU_PASS cases=%u api=%s grouped=%d nonblocking=%d src=%dx%d dst=%dx%d\n",
                                cases, windowApi ? "window" : "default", grouped, nonblocking,
                                sm.dims[0], sm.dims[1], dm.dims[0], dm.dims[1]);
  MPI_OK(MPI_Finalize());
}
