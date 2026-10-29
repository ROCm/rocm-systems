/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 ************************************************************************/
#include "reshard_internal.h"
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <vector>

static void require(bool ok, const char* what) {
  if (!ok) { std::fprintf(stderr, "FAIL: %s\n", what); std::exit(1); }
}

// Independent oracle: partition the global box successively in placement
// order, rather than decoding the library's flattened shard index.
static void box(const ncclDistTensor_t& t, int rank, const size_t global[], size_t start[]) {
  size_t extent[3];
  std::copy(global, global + t.ndims, extent);
  std::fill(start, start + t.ndims, 0);
  int coordinate[2] = {(rank - t.mesh->startRank) / t.mesh->dims[1],
                       (rank - t.mesh->startRank) % t.mesh->dims[1]};
  for (int a = 0; a < 2; a++) if (t.placements[a] >= 0) {
    const int d = t.placements[a];
    extent[d] /= t.mesh->dims[a];
    start[d] += coordinate[a] * extent[d];
  }
}

int main() {
  size_t pairs = 0;
  const size_t global[3] = {12, 12, 12};
  // Unequal axis sizes distinguish axis order; unit axes exercise degeneracy.
  const std::array<std::array<int, 2>, 4> meshes = {{{2, 3}, {3, 2}, {1, 6}, {6, 1}}};
  for (int ndims = 1; ndims <= 3; ndims++) for (auto sm : meshes) for (auto dm : meshes)
  for (int s0 = -1; s0 < ndims; s0++) for (int s1 = -1; s1 < ndims; s1++)
  for (int d0 = -1; d0 < ndims; d0++) for (int d1 = -1; d1 < ndims; d1++) {
    ncclMesh_t srcMesh = NCCL_M2N_MESH_INITIALIZER, dstMesh = NCCL_M2N_MESH_INITIALIZER;
    srcMesh.ndims = dstMesh.ndims = 2;
    srcMesh.dims = sm.data(); dstMesh.dims = dm.data();
    srcMesh.startRank = 3; dstMesh.startRank = 11;
    size_t srcShape[3]{}, dstShape[3]{};
    int srcPlacements[2]{s0, s1}, dstPlacements[2]{d0, d1};
    ncclDistTensor_t src = NCCL_M2N_DIST_TENSOR_INITIALIZER, dst = NCCL_M2N_DIST_TENSOR_INITIALIZER;
    src.localShape = srcShape; dst.localShape = dstShape;
    src.placements = srcPlacements; dst.placements = dstPlacements;
    src.mesh = &srcMesh; dst.mesh = &dstMesh;
    src.ndims = dst.ndims = ndims;
    src.placements[0] = s0; src.placements[1] = s1;
    dst.placements[0] = d0; dst.placements[1] = d1;
    require(validateReshardPlacement(&src, "test", "src") == ncclSuccess, "valid source placements");
    require(validateReshardPlacement(&dst, "test", "dst") == ncclSuccess, "valid destination placements");
    for (int d = 0; d < ndims; d++) {
      src.localShape[d] = dst.localShape[d] = global[d];
      for (int a = 0; a < 2; a++) {
        if (src.placements[a] == d) src.localShape[d] /= sm[a];
        if (dst.placements[a] == d) dst.localShape[d] /= dm[a];
      }
    }
    ncclReshardMeshGroupInfo si{}, di{};
    computeMeshGroupInfo(&src, srcMesh.startRank, &si);
    computeMeshGroupInfo(&dst, dstMesh.startRank, &di);
    size_t srcStrides[3]{}, dstStrides[3]{}, rs[3]{}, rd[3]{};
    require(resolveReshardDims(0, src.localShape, si.shardLayout, dst.localShape, di.shardLayout,
                               ndims, rs, rd) == ncclSuccess, "matching global shapes");
    require(computeStridesChecked(rs, ndims, srcStrides) == ncclSuccess, "source strides");
    require(computeStridesChecked(rd, ndims, dstStrides) == ncclSuccess, "destination strides");
    size_t dstElements = rd[0] * dstStrides[0];
    for (int rank = srcMesh.startRank; rank < srcMesh.startRank + 6; rank++) {
      ncclReshardMeshGroupInfo rankInfo{};
      computeMeshGroupInfo(&src, rank, &rankInfo);
      require(getMeshRank(&src, &rankInfo, rankInfo.shardIdx, rankInfo.repIdx) == rank, "source rank round trip");
    }
    for (int rank = dstMesh.startRank; rank < dstMesh.startRank + 6; rank++) {
      ncclReshardMeshGroupInfo rankInfo{};
      computeMeshGroupInfo(&dst, rank, &rankInfo);
      require(getMeshRank(&dst, &rankInfo, rankInfo.shardIdx, rankInfo.repIdx) == rank, "destination rank round trip");
    }
    for (int ds = 0; ds < di.shardCount; ds++) {
      std::vector<unsigned char> written(dstElements, 0);
      size_t dstStart[3]{};
      box(dst, getMeshRank(&dst, &di, ds, 0), global, dstStart);
      for (int ss = 0; ss < si.shardCount; ss++) {
        ncclReshardTransferPlan plan{};
        require(computeTransferPlanChecked(rs, srcStrides, si.shardLayout, ss, rd, dstStrides,
                                           di.shardLayout, ds, ndims, 1, &plan) == ncclSuccess, "transfer plan");
        size_t srcStart[3]{};
        box(src, getMeshRank(&src, &si, ss, 0), global, srcStart);
        size_t expected = 1;
        for (int d = 0; d < ndims; d++) {
          size_t lo = std::max(srcStart[d], dstStart[d]);
          size_t hi = std::min(srcStart[d] + rs[d], dstStart[d] + rd[d]);
          expected *= hi > lo ? hi - lo : 0;
        }
        require(expected == plan.innerSize * plan.totalInnerTransfers, "overlap volume");
        for (size_t outer = 0; outer < plan.totalInnerTransfers; outer++) {
          size_t so = plan.srcBaseOffset, dest = plan.dstBaseOffset, remaining = outer;
          for (int d = plan.numOuterLoops - 1; d >= 0; d--) {
            size_t c = remaining % plan.outerCounts[d]; remaining /= plan.outerCounts[d];
            so += c * plan.outerSrcStrides[d]; dest += c * plan.outerDstStrides[d];
          }
          for (size_t inner = 0; inner < plan.innerSize; inner++) {
            size_t a = so + inner, b = dest + inner;
            require(a < rs[0] * srcStrides[0] && b < dstElements, "local buffer bounds");
            require(written[b]++ == 0, "destination written exactly once");
            for (int d = 0; d < ndims; d++) {
              size_t sc = a / srcStrides[d]; a %= srcStrides[d];
              size_t dc = b / dstStrides[d]; b %= dstStrides[d];
              require(srcStart[d] + sc == dstStart[d] + dc, "copied element has same global coordinate");
            }
          }
        }
        pairs++;
      }
      require(std::all_of(written.begin(), written.end(), [](unsigned char n) { return n == 1; }),
              "complete destination coverage");
    }
  }
  int meshDims[2]{2, 3}, placements[2]{0, 0};
  ncclMesh_t mesh = NCCL_M2N_MESH_INITIALIZER;
  mesh.ndims = 2; mesh.dims = meshDims;
  ncclDistTensor_t t = NCCL_M2N_DIST_TENSOR_INITIALIZER;
  t.mesh = &mesh; t.ndims = 2; t.placements = placements;
  t.placements[0] = 0; t.placements[1] = 0;
  ncclReshardMeshGroupInfo info{}; computeMeshGroupInfo(&t, 0, &info);
  size_t good[3] = {6, 12, 1}, bad[3] = {7, 12, 1}, out[3]{};
  require(resolveReshardDims(0, good, info.shardLayout, bad, info.shardLayout, 2, out, out) == ncclInvalidArgument,
          "reject global shape mismatch");
  good[0] = std::numeric_limits<size_t>::max() / 6 + 1;
  require(resolveReshardDims(0, good, info.shardLayout, good, info.shardLayout, 2, out, out) == ncclInvalidArgument,
          "reject product-of-axes overflow");
  t.placements[1] = 2;
  require(validateReshardPlacement(&t, "test", "tensor") == ncclInvalidArgument, "reject dimension equal to ndims");
  t.placements[1] = -2;
  require(validateReshardPlacement(&t, "test", "tensor") == ncclInvalidArgument, "reject invalid placement");
  std::printf("MULTI_DIM_GEOMETRY_PASS pairs=%zu\n", pairs);
}
