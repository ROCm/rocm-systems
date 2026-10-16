/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <climits>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "nccl.h"
#include "comm.h"
#include "graph.h"
#include "graph/topo.h"

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

constexpr int kGpus = 16;

class RomeModelsParseGraphLightMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = std::make_unique<RomeModelsParseScene>(kGpus);
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

TEST_F(RomeModelsParseGraphLightMicrotest, NullString_IsRejected) {
  EXPECT_EQ(parseGraphLight(nullptr, system(), graph(), nullptr), ncclInvalidArgument);
}

TEST_F(RomeModelsParseGraphLightMicrotest, WellFormedTree_IsTranscribedToTreeBase) {
  const char* tree = "(4(2(14)(1))(6(5)))";
  graph()->treeBase[1][0] = 'x';  // the fixture zeroes the graph, so dirty the terminator

  ASSERT_EQ(parseGraphLight(tree, system(), graph(), nullptr), ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], tree);
  EXPECT_EQ(graph()->treeBase[1][0], 0);
}

TEST_F(RomeModelsParseGraphLightMicrotest, MultipleSegments_ProduceOneTreeBaseRowEach) {
  graph()->treeBase[3][0] = 'Z';  // the fixture zeroes the graph, so dirty the terminator

  ASSERT_EQ(parseGraphLight("(1(2))|(3(4))|(5(6))", system(), graph(), nullptr), ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], "(1(2))");
  EXPECT_STREQ(graph()->treeBase[1], "(3(4))");
  EXPECT_STREQ(graph()->treeBase[2], "(5(6))");
  EXPECT_EQ(graph()->treeBase[3][0], 0);
}

TEST_F(RomeModelsParseGraphLightMicrotest, EmptyString_ProducesEmptyTreeBase) {
  graph()->treeBase[0][0] = 'x';

  ASSERT_EQ(parseGraphLight("", system(), graph(), nullptr), ncclSuccess);

  EXPECT_EQ(graph()->treeBase[0][0], 0);
}

TEST_F(RomeModelsParseGraphLightMicrotest, MoreNumbersInSegmentThanMaxNodes_IsRejected) {
  std::string tooManyNumbers;
  for (int i = 0; i <= NCCL_TOPO_MAX_NODES; i++) tooManyNumbers += "(0)";

  EXPECT_EQ(parseGraphLight(tooManyNumbers.c_str(), system(), graph(), nullptr),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, GpuNumberAsWideAsAnInt_IsTranscribed) {
  // The largest accepted number, so the guard's boundary itself is pinned. It is also ten
  // digits: one more than the buffer originally held (it has been expanded to accomodate ints)
  constexpr int kWideDev = INT_MAX;
  system()->nodes[GPU].nodes[kGpus - 1].gpu.dev = kWideDev;
  const std::string tree = "(" + std::to_string(kWideDev) + ")";

  ASSERT_EQ(parseGraphLight(tree.c_str(), system(), graph(), nullptr), ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], tree.c_str());
}

TEST_F(RomeModelsParseGraphLightMicrotest, MostNegativeGpuNumberFromGpuMap_IsTranscribed) {
  system()->nodes[GPU].nodes[kGpus - 1].gpu.dev = INT_MIN;
  std::vector<int> mapToMostNegative(kGpus, 0);
  mapToMostNegative[0] = INT_MIN;

  ASSERT_EQ(parseGraphLight("(0)", system(), graph(), mapToMostNegative.data()), ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], "(-2147483648)");
}

TEST_F(RomeModelsParseGraphLightMicrotest, GpuNumbersAreTranslatedThroughGpuMap) {
  std::vector<int> reversingMap(kGpus);
  for (int i = 0; i < kGpus; i++) reversingMap[i] = kGpus - 1 - i;

  const std::string highestAndLowest =
      "(" + std::to_string(kGpus - 1) + "(0))";  // "(15(0))" for 16 GPUs

  ASSERT_EQ(parseGraphLight(highestAndLowest.c_str(), system(), graph(), reversingMap.data()),
            ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], "(0(15))");
}

TEST_F(RomeModelsParseGraphLightMicrotest, GpuNotInTheSystem_IsRejected) {
  const std::string absentGpu = "(" + std::to_string(kGpus + 4) + ")";

  EXPECT_EQ(parseGraphLight(absentGpu.c_str(), system(), graph(), nullptr), ncclInternalError);
}

TEST_F(RomeModelsParseGraphLightMicrotest, GpuNumberOutsideGpuMap_IsRejected) {
  // Exactly as long as the system has GPUs, matching the shortest table any caller
  // passes, and heap-allocated so a read past it is one ASan can see.
  std::vector<int> gpuMap(kGpus);
  for (int i = 0; i < kGpus; i++) gpuMap[i] = i;

  // The smallest number the table cannot hold, so the limit itself is pinned.
  const std::string justPastEnd = "(" + std::to_string(kGpus) + ")";

  EXPECT_EQ(parseGraphLight(justPastEnd.c_str(), system(), graph(), gpuMap.data()),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, TrailingNumberWithNoClosingParen_IsRejected) {
  // Long enough that std::string puts it on the heap, so the overread is a heap
  // overflow ASan can see rather than an unremarkable read of adjacent stack.
  const std::string truncatedTree = "(1)(2)(3)(4)(5)(6";

  EXPECT_EQ(parseGraphLight(truncatedTree.c_str(), system(), graph(), nullptr),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, StrayCharacterInsideATree_IsRejected) {
  // Each of these parses two GPU numbers but transcribes only the first: the scan for the
  // end of a number swallows the stray character and everything after it up to the next
  // parenthesis, so the second number is silently dropped from treeBase.
  for (const char* strayCharacter : {"(1x2)", "(0(1)(2,3))", "(-1)"}) {
    SCOPED_TRACE(strayCharacter);

    EXPECT_EQ(parseGraphLight(strayCharacter, system(), graph(), nullptr), ncclInvalidUsage);
  }
}

TEST_F(RomeModelsParseGraphLightMicrotest, MoreTokensThanNumbersParsed_IsRejected) {
  EXPECT_EQ(parseGraphLight("(5)(6)|(0)(x)", system(), graph(), nullptr), ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, MoreSegmentsThanTreeBaseRows_IsRejected) {
  // Exactly one segment too many: with one segment per row there is no row left to
  // terminate the list, so this is the smallest input that must be refused and it pins
  // the limit itself rather than merely being somewhere beyond it.
  const size_t rows = sizeof(graph()->treeBase) / sizeof(graph()->treeBase[0]);
  std::string tooManySegments = "(0)";
  for (size_t i = 1; i < rows; i++) tooManySegments += "|(0)";

  EXPECT_EQ(parseGraphLight(tooManySegments.c_str(), system(), graph(), nullptr),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, NumberWiderThanIntRange_IsRejected) {
  // INT_MAX + 1: the smallest rejected number, so a guard that drifted by one would fail here.
  EXPECT_EQ(parseGraphLight("(2147483648)", system(), graph(), nullptr), ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphLightMicrotest, SegmentLongerThanTreeBaseRow_IsRejected) {
  const size_t rowLen = sizeof(graph()->treeBase[0]);

  // Overruns while copying a parenthesis.
  const std::string parens(rowLen, '(');
  // Overruns while copying a digit of a GPU number: the parentheses fill the row, so the
  // digits of "14" land beyond it. The trailing ')' keeps the number's end-of-token scan
  // inside the string.
  // Note the row can never run out with a digit as its last written character: a number
  // is only transcribed once its closing parenthesis has been found inside the segment,
  // and that parenthesis is then written too. So the digit check cannot be the one that
  // catches the final overrun, and only its presence is pinned here, not its exact
  // boundary.
  const std::string endsInNumber = std::string(rowLen - 1, '(') + "14)";

  for (const std::string& tooLong : {parens, endsInNumber}) {
    SCOPED_TRACE("segment length " + std::to_string(tooLong.size()));
    const char sentinel = 'Z';
    graph()->treeBase[1][0] = sentinel;

    EXPECT_EQ(parseGraphLight(tooLong.c_str(), system(), graph(), nullptr), ncclInvalidUsage);
    // Returning an error is not enough: the row after this one must be untouched, or the
    // overrun happened and was merely reported afterwards.
    EXPECT_EQ(graph()->treeBase[1][0], sentinel);
  }
}

// The shipping preset topologies go through the same guarded path as RCCL_TREES does
// (rome_models.cc passes romeTopoModels[i].treeBase with a non-NULL gpu_map), under
// NCCLCHECK, so a guard that rejected one of them would fail ncclCommInitRank outright.
TEST(RomeModelsPresetTopologiesMicrotest, EveryPresetTreeBase_IsAccepted) {
  constexpr size_t kModels = sizeof(romeTopoModels) / sizeof(romeTopoModels[0]);

  for (size_t i = 0; i < kModels; i++) {
    const rcclRomeModel& model = romeTopoModels[i];
    if (model.treeBase == nullptr) continue;
    SCOPED_TRACE("romeTopoModels[" + std::to_string(i) + "], treeBase " + model.treeBase);

    RomeModelsParseScene scene(model.nGpus);
    ncclTopoGraph graph;
    std::memset(&graph, 0, sizeof(graph));
    // The production map is a permutation of the GPU numbers; identity is the
    // representative one, and its length is what bounds the guarded lookup.
    std::vector<int> identityMap(model.nGpus);
    for (int g = 0; g < model.nGpus; g++) identityMap[g] = g;

    EXPECT_EQ(parseGraphLight(model.treeBase, scene.system(), &graph, identityMap.data()),
              ncclSuccess);
  }

  ResetNcclFakes();
}

}  // namespace
