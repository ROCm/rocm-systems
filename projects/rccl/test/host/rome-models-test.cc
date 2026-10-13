/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "nccl.h"
#include "comm.h"
#include "graph.h"
#include "topo.h"

#include "fakes/nccl_fakes.h"

#include ROME_MODELS_CC_PATH

namespace {

class RomeModelsParseScene {
 public:
  explicit RomeModelsParseScene(int nGpus) : system_(new ncclTopoSystem{}) {
    system_->nodes[GPU].count = nGpus;
    for (int i = 0; i < nGpus; i++) {
      system_->nodes[GPU].nodes[i].gpu.dev = i;
    }
  }

  ncclTopoSystem* system() { return system_.get(); }

 private:
  std::unique_ptr<ncclTopoSystem> system_;
};

class RomeModelsParseGraphLightMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = std::make_unique<RomeModelsParseScene>(8);
    graph_ = std::make_unique<ncclTopoGraph>();
    std::memset(graph_.get(), 0, sizeof(*graph_));
  }

  void TearDown() override { ResetNcclFakes(); }

  ncclTopoSystem* system() { return scene_->system(); }
  ncclTopoGraph* graph() { return graph_.get(); }

 private:
  std::unique_ptr<RomeModelsParseScene> scene_;
  std::unique_ptr<ncclTopoGraph> graph_;
};

// ROCM-31973: every index and accumulator in parseGraphLight is driven straight from
// RCCL_TREES with no clamping. One test per unchecked quantity,
// each named for the behaviour the hardened parser should have; all are expected to
// reject the input rather than write out of bounds.
//
//   gpus[] index `gpu`          MoreNumbersInSegmentThanMaxNodes_IsRejected
//   `gpus[gpu]*10+digit`        NumberWiderThanIntRange_IsRejected
//   treeBase column `y`         SegmentLongerThanTreeBaseRow_IsRejected
//   treeBase row `x`            MoreSegmentsThanTreeBaseRows_IsRejected
//   gpus[] read index `r`       MoreTokensThanNumbersParsed_IsRejected
//   `start_offset` paren scan   TrailingNumberWithNoClosingParen_IsRejected
//   `gpu_map[g]`                GpuNumberOutsideGpuMap_IsRejected
//   `number_str[10]`            RemappedGpuNumberTooWideForBuffer_IsRejected
//   `str` itself                NullString_IsRejected
//
// Accepted-input contracts, so the above cannot be satisfied by rejecting everything:
//
//   EmptyString_ProducesEmptyTreeBase            (below)
//   WellFormedTree_IsTranscribedToTreeBase
//   MultipleSegments_ProduceOneTreeBaseRowEach
//   GpuNumbersAreTranslatedThroughGpuMap

TEST_F(RomeModelsParseGraphLightMicrotest, EmptyString_ProducesEmptyTreeBase) {
  graph()->treeBase[0][0] = 'x';

  ASSERT_EQ(parseGraphLight("", system(), graph(), nullptr), ncclSuccess);

  EXPECT_EQ(graph()->treeBase[0][0], 0);
}

// The numbers of one segment are accumulated into a fixed NCCL_TOPO_MAX_NODES-entry
// array before being transcribed, so a segment holding more of them than the topology
// can ever contain must be refused rather than written past the end of that array.
TEST_F(RomeModelsParseGraphLightMicrotest, MoreNumbersInSegmentThanMaxNodes_IsRejected) {
  std::string tooManyNumbers;
  for (int i = 0; i <= NCCL_TOPO_MAX_NODES; i++) tooManyNumbers += "(0)";

  EXPECT_EQ(parseGraphLight(tooManyNumbers.c_str(), system(), graph(), nullptr),
            ncclInvalidUsage);
}

}  // namespace
