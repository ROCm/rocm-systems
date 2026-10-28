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
#include "fakes/topo_stubs.h"

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
  void SetUp() override { UseGpus(kRingGpus); }

  void TearDown() override { ResetNcclFakes(); }

  // Rebuild the scene and the graph for a GPU count other than the default, so a
  // test that needs a wider system still reads it through system() and graph().
  void UseGpus(int nGpus) {
    scene_ = std::make_unique<RomeModelsParseScene>(nGpus);
    scene_->system()->totalBw = kTotalBw;
    graph_ = std::make_unique<ncclTopoGraph>();
    std::memset(graph_.get(), 0, sizeof(*graph_));
  }

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

TEST_F(RomeModelsParseGraphMicrotest, NullRingString_IsRejected) {
  // search.cc only reaches parseGraph when NCCL_RINGS is set, so this is
  // defensive rather than reachable from the environment -- but parseGraphLight
  // already rejects a null string and the two parsers should not disagree.
  EXPECT_EQ(parseGraph(nullptr, system(), graph(), nullptr, nullptr, 0), ncclInvalidArgument);
}

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
  // Rings for other GPU counts are skipped, not rejected. The short segment follows a
  // full one, so its stale gpus[3] keeps it otherwise valid and only the count check
  // can drop it.
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
  UseGpus(12);

  ASSERT_EQ(parseGraph("0 1 2 3 4 5 6 7 8 9 10 11", system(), graph(), nullptr, nullptr, 0),
            ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(graph()->intra[10], RomeModelsParseScene::kRankBase + 10);
  EXPECT_EQ(graph()->intra[11], RomeModelsParseScene::kRankBase + 11);
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
  UseGpus(NCCL_TOPO_MAX_NODES);

  // Exactly one GPU number too many, so an off-by-one in the bound fails here.
  // Before the bound, the 145th number was written to gpus[144] -- one int past
  // the array -- which ASan reports as a stack-buffer-overflow at offset 608.
  const std::string tooManyGpus = ChannelWithGpuCount(NCCL_TOPO_MAX_NODES + 1);

  EXPECT_EQ(parseGraph(tooManyGpus.c_str(), system(), graph(), nullptr, nullptr, 0),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphMicrotest, ExactlyMaxNodesGpusInOneChannel_IsAccepted) {
  UseGpus(NCCL_TOPO_MAX_NODES);

  // The widest channel that still fits, so a bound tightened by one breaks here.
  const std::string exactlyFull = ChannelWithGpuCount(NCCL_TOPO_MAX_NODES);

  ASSERT_EQ(parseGraph(exactlyFull.c_str(), system(), graph(), nullptr, nullptr, 0),
            ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(graph()->intra[NCCL_TOPO_MAX_NODES - 1],
            RomeModelsParseScene::kRankBase + NCCL_TOPO_MAX_NODES - 1);
}

TEST_F(RomeModelsParseGraphMicrotest, GpuNumberWiderThanIntRange_IsRejected) {
  // INT_MAX + 1: the smallest number that cannot be accumulated, so a bound that
  // drifted by one would fail here. Accumulating it overflowed a signed int,
  // which is undefined, and the wrapped negative was then quietly swallowed by
  // the g < 0 test -- the channel vanished instead of being reported.
  EXPECT_EQ(parseGraph("2147483648", system(), graph(), nullptr, nullptr, 0), ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphMicrotest, GpuNumberAtIntMax_IsNotRejected) {
  // The largest number that still fits. It names no GPU this system has, so the
  // channel is ignored -- but ignoring is not rejecting, and a bound tightened by
  // one would turn this into an error.
  EXPECT_EQ(parseGraph("2147483647", system(), graph(), nullptr, nullptr, 0), ncclSuccess);
  EXPECT_EQ(graph()->nChannels, 0);
}

// A NET slot the ring did not name. nets[] holds a NET number biased by 'N', so an
// absent slot must translate to a negative NET id, which is what the translation
// and the inter[] checks both read as "no NET here".
bool IsAbsentNetSlot(int slot) { return slot - 'N' < 0; }

TEST_F(RomeModelsParseGraphMicrotest, NetSlotsTheRingDidNotName_AreNotCarriedOver) {
  system()->nodes[NET].count = 2;
  system()->nodes[NET].nodes[0].id = 50;
  system()->nodes[NET].nodes[1].id = 51;

  // The first segment names five NETs, which is odd, so it is ignored as a
  // channel -- but it still writes them into nets[]. The second segment names
  // only its two end NETs, leaving the slots between them holding the first
  // segment's numbers. Seeding the stale state from an earlier iteration of the
  // same call keeps this deterministic; the raw uninitialised values this used to
  // expose varied run to run.
  ASSERT_EQ(parseGraph("N5 0 N6 1 N7 2 N8 3 N9|N0 0 1 2 3 N1", system(), graph(), nullptr,
                       nullptr, 0),
            ncclSuccess);

  ASSERT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(graph()->inter[0], 0);
  EXPECT_EQ(graph()->inter[1], 1);

  // Slots 1, 3 and 5 are where the ignored segment's N6, N7 and N8 landed.
  for (int slot : {1, 2, 3, 4, 5, 6}) {
    SCOPED_TRACE("intraNets slot " + std::to_string(slot));
    EXPECT_TRUE(IsAbsentNetSlot(graph()->intraNets[slot]))
        << "slot holds NET " << graph()->intraNets[slot] - 'N';
  }
}

TEST_F(RomeModelsParseGraphMicrotest, ChannelWithUnnamedEndNets_IsIgnored) {
  system()->nodes[NET].count = 2;
  system()->nodes[NET].nodes[0].id = 50;
  system()->nodes[NET].nodes[1].id = 51;

  // The second segment names its NETs between the GPUs rather than at the ends,
  // so both end slots stay absent. An absent slot reads as a negative NET id, and
  // only a check for that can stop it reaching inter[] -- the limit on its own
  // sees a number below nnets and lets it through.
  ASSERT_EQ(parseGraph("N0 0 1 2 3 N1|0 N0 1 2 N1 3", system(), graph(), nullptr, nullptr, 0),
            ncclSuccess);

  ASSERT_EQ(graph()->nChannels, 1);
  EXPECT_EQ(graph()->inter[0], 0);
  EXPECT_EQ(graph()->inter[1], 1);
  // The ignored channel must not have left a NET id behind. ncclTopoIdToNetDev
  // fails multi-node init on one of these.
  EXPECT_GE(graph()->inter[2], 0);
  EXPECT_GE(graph()->inter[3], 0);
}

// A channel naming every GPU the array can hold, so the NET slot index derived
// from the GPU count (2 * gpu - 1) sits at the very end of nets[].
std::string WidestChannelThenNets(const char* trailingNets) {
  return ChannelWithGpuCount(NCCL_TOPO_MAX_NODES) + trailingNets;
}

TEST_F(RomeModelsParseGraphMicrotest, MoreNetsInOneChannelThanNetsArrayHolds_IsRejected) {
  UseGpus(NCCL_TOPO_MAX_NODES);
  system()->nodes[NET].count = 2;

  // The existing per-channel limit is the runtime ngpus * 2, not the size of
  // nets[], and it used > where the array needs >=. At the widest GPU count the
  // two coincide, so the second NET here wrote nets[288] -- one int past the end,
  // which ASan reports at offset 1888. This is the smallest input that reaches it.
  const std::string tooManyNets = WidestChannelThenNets(" N5 N6");

  EXPECT_EQ(parseGraph(tooManyNets.c_str(), system(), graph(), nullptr, nullptr, 0),
            ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphMicrotest, NetInTheLastSlotOfTheNetsArray_IsAccepted) {
  UseGpus(NCCL_TOPO_MAX_NODES);
  system()->nodes[NET].count = 2;

  // Writes nets[287], the last slot that exists, so a bound tightened by one
  // breaks here.
  const std::string exactlyFull = WidestChannelThenNets(" N5");

  EXPECT_EQ(parseGraph(exactlyFull.c_str(), system(), graph(), nullptr, nullptr, 0),
            ncclSuccess);
}

// NET numbers are held biased by 'N', so the widest one that still fits an int is
// INT_MAX - 'N' rather than INT_MAX.
constexpr int kWidestNetNumber = INT_MAX - 'N';

TEST_F(RomeModelsParseGraphMicrotest, NetNumberWiderThanIntRange_IsRejected) {
  system()->nodes[NET].count = 2;

  // One past the widest NET number that can be held. Accumulating it overflowed
  // a signed int, which is undefined; the result was then quietly discarded by
  // the "not a NET this system has" test and the ring silently lost a channel.
  const std::string tooWide = "N" + std::to_string(kWidestNetNumber + 1) + " 0 1 2 3 N1";

  EXPECT_EQ(parseGraph(tooWide.c_str(), system(), graph(), nullptr, nullptr, 0), ncclInvalidUsage);
}

TEST_F(RomeModelsParseGraphMicrotest, NetNumberAtTheWidestThatFits_IsNotRejected) {
  system()->nodes[NET].count = 2;

  // Exactly the widest NET number that fits. It names no NET this system has, so
  // the channel is ignored -- but a bound tightened by one would make it an error.
  const std::string widest = "N" + std::to_string(kWidestNetNumber) + " 0 1 2 3 N1";

  EXPECT_EQ(parseGraph(widest.c_str(), system(), graph(), nullptr, nullptr, 0), ncclSuccess);
}

TEST_F(RomeModelsParseGraphMicrotest, TreeGraph_ResolvesOutgoingNetFromTheSecondGpu) {
  // graph->id == 1 is the tree graph, whose outgoing NET is looked up per channel
  // from the rank in intra[i * ngpus + 1] rather than taken from the ring string.
  // That index is bounded only by nChannels, which is why it is safe now.
  graph()->id = 1;
  system()->nodes[NET].count = 1;

  std::vector<int> ranksAskedFor;
  g_ncclTopoGetLocalNet = [&ranksAskedFor](struct ncclTopoSystem*, int rank, int channelId,
                                           int64_t* id, int*) {
    ranksAskedFor.push_back(rank);
    *id = 900 + channelId;
    return ncclSuccess;
  };

  ASSERT_EQ(parseGraph("0 1 2 3|3 2 1 0", system(), graph(), nullptr, nullptr, 0), ncclSuccess);

  ASSERT_EQ(graph()->nChannels, 2);
  // The second GPU of each channel, which differs per channel, so a lookup that
  // ignored the channel would be visible here.
  EXPECT_EQ(ranksAskedFor, RanksOf({1, 2}));
  EXPECT_EQ(graph()->inter[1], 900);
  EXPECT_EQ(graph()->inter[3], 901);

  ResetTopoStubs();
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

// parseGIOTopos sizes devids, g_hives and rdm at its own NGPUS (16) but admits
// any system with at least that many GPUs, so a wider one overruns all three --
// and parseGraph's gpu_map bound, which trusts ngpus, then reads past rdm. The
// model describes exactly 16 GPUs, so admitting more was never intended.
//
// No GPU->CPU paths are set here, so a gate that admits this system does not just
// overrun -- it dereferences a null paths[CPU] in parseRomeSystem
// (rome_models.cc:4357) and crashes in the ordinary build, not only under ASan.
TEST(RomeModelsGioToposMicrotest, SystemWithMoreGpusThanTheModel_IsIgnored) {
  constexpr int kModelGpus = 16;
  constexpr int kCpus = 2;
  RomeModelsParseScene wider(kModelGpus + 1);
  wider.system()->nodes[CPU].count = kCpus;  // the model admits at most two
  wider.system()->nodes[NET].count = 0;      // and no NICs
  wider.system()->totalBw = 120.0f;

  for (int c = 0; c < kCpus; c++) wider.system()->nodes[CPU].nodes[c].id = c;

  ncclTopoGraph graph;
  std::memset(&graph, 0, sizeof(graph));

  EXPECT_EQ(parseGIOTopos(wider.system(), &graph), ncclSuccess);
  EXPECT_EQ(graph.nChannels, 0);

  ResetNcclFakes();
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
}  // namespace
