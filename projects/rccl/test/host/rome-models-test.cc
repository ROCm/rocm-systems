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
      // Ranks are deliberately not equal to the dev numbers, so a test that reads
      // graph->intra cannot pass by accident if the translation step is skipped.
      system_->nodes[GPU].nodes[i].gpu.rank = kRankBase + i;
    }
  }

  static constexpr int kRankBase = 100;

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

// ---------------------------------------------------------------------------
// parseGraph: accepted-input contracts.
//
// These pin what a well-formed NCCL_RINGS must keep doing. Every bound added for
// ROCM-31973 is a chance to reject a valid topology by accident, and only a test
// that asserts on accepted input can catch that.
// ---------------------------------------------------------------------------

class RomeModelsParseGraphMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = std::make_unique<RomeModelsParseScene>(kRingGpus);
    scene_->system()->totalBw = kTotalBw;
    graph_ = std::make_unique<ncclTopoGraph>();
    std::memset(graph_.get(), 0, sizeof(*graph_));
  }

  void TearDown() override { ResetNcclFakes(); }

  ncclTopoSystem* system() { return scene_->system(); }
  ncclTopoGraph* graph() { return graph_.get(); }

  // The ranks the scene gives devs 0..3, in dev order.
  static std::vector<int> RanksOf(std::vector<int> devs) {
    for (int& d : devs) d += RomeModelsParseScene::kRankBase;
    return devs;
  }

  std::vector<int> IntraOfChannel(int channel) const {
    return std::vector<int>(graph_->intra + channel * kRingGpus,
                            graph_->intra + (channel + 1) * kRingGpus);
  }

  static constexpr int kRingGpus = 4;
  static constexpr float kTotalBw = 120.0f;

 private:
  std::unique_ptr<RomeModelsParseScene> scene_;
  std::unique_ptr<ncclTopoGraph> graph_;
};

TEST_F(RomeModelsParseGraphMicrotest, WellFormedRings_AreTranscribedToIntra) {
  ASSERT_EQ(parseGraph("0 1 2 3|3 2 1 0", system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 2);
  EXPECT_EQ(IntraOfChannel(0), RanksOf({0, 1, 2, 3}));
  EXPECT_EQ(IntraOfChannel(1), RanksOf({3, 2, 1, 0}));
  // Bandwidth is split across the channels that were accepted.
  EXPECT_FLOAT_EQ(graph()->bwIntra, kTotalBw / 2);
  EXPECT_FLOAT_EQ(graph()->bwInter, kTotalBw / 2);
}

TEST_F(RomeModelsParseGraphMicrotest, ReversedRing_IsTranscribedBackwards) {
  ASSERT_EQ(parseGraph("0 1 2 3", system(), graph(), nullptr, nullptr, 1), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(IntraOfChannel(0), RanksOf({3, 2, 1, 0}));
}

TEST_F(RomeModelsParseGraphMicrotest, GpuNumbersAreTranslatedThroughGpuMap) {
  // A reversing map, so a result equal to the input cannot pass by accident.
  std::vector<int> reversingMap(kRingGpus);
  for (int i = 0; i < kRingGpus; i++) reversingMap[i] = kRingGpus - 1 - i;

  ASSERT_EQ(parseGraph("0 1 2 3", system(), graph(), reversingMap.data(), nullptr, 0), ncclSuccess);

  EXPECT_EQ(IntraOfChannel(0), RanksOf({3, 2, 1, 0}));
}

TEST_F(RomeModelsParseGraphMicrotest, ChannelWithWrongGpuCount_IsIgnored) {
  // Documented behaviour: rings are supplied for several GPU counts at once and
  // the ones that do not fit this communicator are skipped, not rejected.
  //
  // The short segment comes second, and is a prefix of the first, on purpose. The
  // gpus[] slot it does not write still holds 3 from the full segment parsed just
  // before it, so dropping the count check leaves four valid non-duplicate numbers
  // and the segment is accepted as a second channel. A short segment that stood
  // alone, or that collided with the carried-over value, would instead be refused
  // by the bounds or duplicate check next door and the count check would look
  // tested while being free to delete.
  ASSERT_EQ(parseGraph("0 1 2 3|0 1 2", system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(IntraOfChannel(0), RanksOf({0, 1, 2, 3}));
}

TEST_F(RomeModelsParseGraphMicrotest, DuplicateGpuInChannel_IsIgnored) {
  ASSERT_EQ(parseGraph("0 0 1 2|0 1 2 3", system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(IntraOfChannel(0), RanksOf({0, 1, 2, 3}));
}

TEST_F(RomeModelsParseGraphMicrotest, MultiDigitGpuNumbers_AreParsed) {
  RomeModelsParseScene wide(12);
  ncclTopoGraph graph;
  std::memset(&graph, 0, sizeof(graph));

  ASSERT_EQ(parseGraph("0 1 2 3 4 5 6 7 8 9 10 11", wide.system(), &graph, nullptr, nullptr, 0),
            ncclSuccess);

  EXPECT_EQ(graph.nChannels, 1);
  EXPECT_EQ(graph.intra[10], RomeModelsParseScene::kRankBase + 10);
  EXPECT_EQ(graph.intra[11], RomeModelsParseScene::kRankBase + 11);
}

// A ring list of `channels` identical well-formed segments for the 4-GPU scene.
std::string RingsWithChannels(int channels) {
  std::string rings = "0 1 2 3";
  for (int i = 1; i < channels; i++) rings += "|0 1 2 3";
  return rings;
}

// nIntraChannels sits immediately after inter[MAXCHANNELS * 2], so the first
// channel past the end writes exactly onto it. The overrun therefore stays inside
// the same ncclTopoGraph and no sanitizer can see it -- this sentinel is the only
// thing that can catch it.
constexpr int kIntraChannelsSentinel = 4242;

TEST_F(RomeModelsParseGraphMicrotest, MoreChannelsThanMaxChannels_IsRejected) {
  // One NET, so the inter[] writes that run off the end actually happen.
  system()->nodes[NET].count = 1;
  system()->nodes[NET].nodes[0].id = 77;
  graph()->nIntraChannels = kIntraChannelsSentinel;

  // Exactly one channel too many, so an off-by-one in the new bound fails here
  // rather than only something far past the limit.
  const std::string tooMany = RingsWithChannels(MAXCHANNELS + 1);

  EXPECT_EQ(parseGraph(tooMany.c_str(), system(), graph(), nullptr, nullptr, 0),
            ncclInvalidUsage);
  // Returning an error is not enough on its own: before the bound existed this
  // returned ncclSuccess and left nIntraChannels holding the NET id.
  EXPECT_EQ(graph()->nIntraChannels, kIntraChannelsSentinel);
  EXPECT_LE(graph()->nChannels, MAXCHANNELS);
}

// A single channel naming `count` GPU numbers, for a scene with that many GPUs.
std::string ChannelWithGpuCount(int count) {
  std::string rings;
  for (int i = 0; i < count; i++) {
    if (i) rings += " ";
    rings += std::to_string(i);
  }
  return rings;
}

TEST_F(RomeModelsParseGraphMicrotest, MoreGpusInOneChannelThanMaxNodes_IsRejected) {
  RomeModelsParseScene wide(NCCL_TOPO_MAX_NODES);
  wide.system()->totalBw = kTotalBw;
  ncclTopoGraph graph;
  std::memset(&graph, 0, sizeof(graph));

  // Exactly one GPU number too many, so an off-by-one in the bound fails here.
  // Before the bound, the 145th number was written to gpus[144] -- one int past
  // the array -- which ASan reports as a stack-buffer-overflow at offset 608.
  const std::string tooManyGpus = ChannelWithGpuCount(NCCL_TOPO_MAX_NODES + 1);

  EXPECT_EQ(parseGraph(tooManyGpus.c_str(), wide.system(), &graph, nullptr, nullptr, 0),
            ncclInvalidUsage);

  ResetNcclFakes();
}

TEST_F(RomeModelsParseGraphMicrotest, ExactlyMaxNodesGpusInOneChannel_IsAccepted) {
  RomeModelsParseScene wide(NCCL_TOPO_MAX_NODES);
  wide.system()->totalBw = kTotalBw;
  ncclTopoGraph graph;
  std::memset(&graph, 0, sizeof(graph));

  // The widest channel that still fits, so a bound tightened by one breaks here.
  const std::string exactlyFull = ChannelWithGpuCount(NCCL_TOPO_MAX_NODES);

  ASSERT_EQ(parseGraph(exactlyFull.c_str(), wide.system(), &graph, nullptr, nullptr, 0),
            ncclSuccess);

  EXPECT_EQ(graph.nChannels, 1);
  EXPECT_EQ(graph.intra[NCCL_TOPO_MAX_NODES - 1],
            RomeModelsParseScene::kRankBase + NCCL_TOPO_MAX_NODES - 1);

  ResetNcclFakes();
}

TEST_F(RomeModelsParseGraphMicrotest, NoMatchingChannel_LeavesBandwidthAtZero) {
  // Callers test graph->nChannels after calling, so a ring list that matches
  // nothing is not an error -- but it must not leave a bandwidth behind either.
  // Seeded with a sentinel because the fixture zeroes the graph, and "bw is 0"
  // would otherwise be true without parseGraph doing anything at all.
  graph()->bwIntra = 99.0f;
  graph()->bwInter = 99.0f;

  // Two GPUs on a four-GPU system: no segment matches, so nChannels stays 0 and
  // totalBw / nChannels used to divide by zero.
  ASSERT_EQ(parseGraph("0 1", system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 0);
  EXPECT_FLOAT_EQ(graph()->bwIntra, 0.0f);
  EXPECT_FLOAT_EQ(graph()->bwInter, 0.0f);
}

TEST_F(RomeModelsParseGraphMicrotest, ExactlyMaxChannels_IsAccepted) {
  system()->nodes[NET].count = 1;
  system()->nodes[NET].nodes[0].id = 77;
  graph()->nIntraChannels = kIntraChannelsSentinel;

  // The largest list that still fits, so a bound tightened by one breaks here.
  const std::string exactlyFull = RingsWithChannels(MAXCHANNELS);

  ASSERT_EQ(parseGraph(exactlyFull.c_str(), system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  EXPECT_EQ(graph()->nChannels, MAXCHANNELS);
  EXPECT_EQ(graph()->nIntraChannels, kIntraChannelsSentinel);
}

// The shipping preset ring literals reach parseGraph under NCCLCHECK, so a bound
// that rejected one of them would fail ncclCommInitRank on hardware that matches
// that model. This is what stops the bounds added for ROCM-31973 from being
// tightened past what the presets themselves need.
TEST(RomeModelsPresetTopologiesMicrotest, EveryPresetRingBase_IsAccepted) {
  constexpr size_t kModels = sizeof(romeTopoModels) / sizeof(romeTopoModels[0]);

  for (size_t i = 0; i < kModels; i++) {
    const rcclRomeModel& model = romeTopoModels[i];

    for (const char* rings : {model.ringBase, model.ringTail1, model.ringTail2, model.treeRail}) {
      if (rings == nullptr) continue;
      SCOPED_TRACE("romeTopoModels[" + std::to_string(i) + "], rings " + rings);

      RomeModelsParseScene scene(model.nGpus);
      scene.system()->totalBw = 120.0f;
      scene.system()->nodes[NET].count = model.nNics;
      for (int n = 0; n < model.nNics; n++) scene.system()->nodes[NET].nodes[n].id = n;

      ncclTopoGraph graph;
      std::memset(&graph, 0, sizeof(graph));
      // The production map is a permutation of the GPU numbers; identity is the
      // representative one, and its length is what bounds the guarded lookup.
      std::vector<int> identityMap(model.nGpus);
      for (int g = 0; g < model.nGpus; g++) identityMap[g] = g;
      std::vector<int> identityNets(model.nNics > 0 ? model.nNics : 1);
      for (int n = 0; n < model.nNics; n++) identityNets[n] = n;

      EXPECT_EQ(parseGraph(rings, scene.system(), &graph, identityMap.data(), identityNets.data(),
                           /*reverse=*/0),
                ncclSuccess);
    }
  }

  ResetNcclFakes();
}

// ---------------------------------------------------------------------------
// parseGraph (NCCL_RINGS) -- tests to be written, see ROCM-31973.
//
// parseGraph is the NCCL_RINGS counterpart of parseGraphLight above: search.cc
// hands it getenv("NCCL_RINGS") directly, with gpu_map and net_map both NULL and
// reverse set from the host index. Every quantity the string drives is listed
// here, named for the behaviour wanted rather than the bug's mechanism. One
// test, one fix, one commit each.
//
// Rejections and silently-wrong output:
//
//   NullRingString_IsRejected
//     str is dereferenced before it is checked. Not reachable from search.cc,
//     which guarantees non-NULL, so this is defensive only -- but parseGraphLight
//     already rejects NULL and the two should not disagree.
//
//   MoreGpusInOneChannelThanMaxNodes_IsRejected
//     gpus[gpu] is written before gpu is bounded, and the bound that follows
//     uses > rather than >=, so a 145th number writes one int past gpus[144].
//     The same off-by-one lets net_offset reach 289 against nets[288].
//
//   GpuNumberWiderThanIntRange_IsRejected
//     gpus[gpu]*10 + digit accumulates with no range check; signed overflow is
//     undefined, and the wrapped negative is then quietly swallowed by the
//     g < 0 test, so the channel vanishes instead of being reported.
//
//   NetNumberWiderThanIntRange_IsRejected
//     the same unchecked accumulation for NET numbers.
//
//   MoreNetsInOneChannelThanNetsArrayHolds_IsRejected
//     net_offset is bounded by the runtime ngpus*2, not by the size of nets[],
//     and again with > rather than >=, so a system reporting 144 GPUs writes one
//     int past nets[288].
//
//   UnparsedNetSlots_AreNotTranslated
//     the net_map translation loop runs the full ngpus*2 slots however few NET
//     tokens were actually parsed, reading uninitialised nets[] entries and
//     writing the result back.
//
//   ChannelWithNetsButNoGpus_IsRejected
//     with ngpus == 0 the reverse swap indexes nets[-1], writing before the
//     array.
//
//   MoreChannelsThanMaxChannels_IsRejected
//     nChannels is never bounded by MAXCHANNELS, so enough '|' segments walk
//     graph->intra, graph->intraNets and graph->inter off their ends with an
//     attacker-chosen offset. This is the headline defect. Surplus channels are
//     refused rather than truncated, matching parseGraphLight's handling of too
//     many trees; EveryPresetRingBase_IsAccepted below is what proves no
//     shipping topology literal is caught by the new bound.
//
//   NoMatchingChannel_LeavesBandwidthAtZero
//     when no segment matches, nChannels is 0 and totalBw / nChannels makes
//     bwIntra and bwInter inf while still returning ncclSuccess. Callers already
//     test graph->nChannels afterwards, so success with no channels stays the
//     contract and only the division is skipped.
//
//   TreeGraphWithNoGpus_DoesNotReadIntra
//     the graph->id == 1 tail indexes intra[i*ngpus + 1] with ngpus == 0.
//
//   GioToposWithMoreGpusThanTheModel_IsIgnored
//     not parseGraph itself but its caller: parseGIOTopos guards ngpus < NGPUS
//     rather than ngpus == NGPUS while sizing rdm, g_hives and devids at 16, so
//     a 17-GPU system overruns rdm and then parseGraph's g < ngpus bound reads
//     gpu_map past its end. The only call site whose map is shorter than ngpus.
//
// Accepted-input contracts. Without these nothing stops a bound above being
// tightened into rejecting a valid configuration, and no rejection test can
// ever catch that:
//
//   WellFormedRings_AreTranscribedToIntra
//   ReversedRing_IsTranscribedBackwards
//   GpuNumbersAreTranslatedThroughGpuMap
//   NetTokens_PopulateInterAndIntraNets
//   NetNumbersAreTranslatedThroughNetMap
//   ChannelWithWrongGpuCount_IsIgnored
//   DuplicateGpuInChannel_IsIgnored
//   ExactlyMaxChannels_IsAccepted
//   EveryPresetRingBase_IsAccepted
// ---------------------------------------------------------------------------




}  // namespace
