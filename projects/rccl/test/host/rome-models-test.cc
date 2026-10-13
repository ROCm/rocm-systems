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

TEST_F(RomeModelsParseGraphLightMicrotest, EmptyString_ProducesEmptyTreeBase) {
  graph()->treeBase[0][0] = 'x';

  ASSERT_EQ(parseGraphLight("", system(), graph(), nullptr), ncclSuccess);

  EXPECT_EQ(graph()->treeBase[0][0], 0);
}

}  // namespace
