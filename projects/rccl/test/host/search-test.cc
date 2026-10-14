/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/search.cc's XML-graph import path:
// ncclTopoGetChannelFromXml and its caller ncclTopoGetGraphFromXmlSub.
//
// The input is attacker-controllable: NCCL_GRAPH_FILE names an XML file that
// ncclTopoGetXmlGraphFromFile parses into an ncclXml before these functions
// transcribe it into an ncclTopoGraph.

#include <gtest/gtest.h>

#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#include "nccl.h"
#include "comm.h"
#include "graph.h"
#include "graph/topo.h"
#include "graph/xml.h"

#include "fakes/nccl_fakes.h"
#include "fakes/topo_stubs.h"

#include SEARCH_CC_PATH

namespace {

// A topology system with `nGpus` GPUs, each parented to a CPU node so that
// NCCL_TOPO_ID_SYSTEM_ID(gpu.parent->id) is well defined. Ranks are
// deliberately not equal to the dev numbers so a test reading graph->intra
// cannot pass by accident if the dev->rank translation is skipped.
class SearchXmlScene {
 public:
  explicit SearchXmlScene(int nGpus) : system_(new ncclTopoSystem{}) {
    system_->nodes[CPU].count = 1;
    system_->nodes[CPU].nodes[0].id = 0;
    system_->nodes[GPU].count = nGpus;
    for (int i = 0; i < nGpus; i++) {
      system_->nodes[GPU].nodes[i].gpu.dev = i;
      system_->nodes[GPU].nodes[i].gpu.rank = kRankBase + i;
      system_->nodes[GPU].nodes[i].gpu.parent = &system_->nodes[CPU].nodes[0];
    }
  }

  static constexpr int kRankBase = 100;

  ncclTopoSystem* system() { return system_.get(); }

 private:
  std::unique_ptr<ncclTopoSystem> system_;
};

constexpr int kGpus = 8;

class SearchXmlChannelMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    scene_ = std::make_unique<SearchXmlScene>(kGpus);
    graph_ = std::make_unique<ncclTopoGraph>();
    std::memset(graph_.get(), 0, sizeof(*graph_));
    // ncclXml's `nodes[1]` is a flexible array sized by xmlAlloc, so the nodes
    // are held in a deque instead: it never reallocates, keeping every
    // previously handed-out ncclXmlNode* stable as more nodes are added.
    nodes_.clear();
  }

  void TearDown() override { ResetNcclFakes(); }

  // Allocate a zeroed node from the fixture's arena.
  ncclXmlNode* NewNode(const char* name) {
    nodes_.emplace_back();
    ncclXmlNode* node = &nodes_.back();
    std::memset(node, 0, sizeof(*node));
    std::snprintf(node->name, sizeof(node->name), "%s", name);
    return node;
  }

  // Append a <name dev="..."> child to `parent`.
  ncclXmlNode* AddChild(ncclXmlNode* parent, const char* name, const char* dev) {
    ncclXmlNode* node = NewNode(name);
    node->parent = parent;
    parent->subs[parent->nSubs++] = node;
    SetAttr(node, "dev", dev);
    return node;
  }

  static void SetAttr(ncclXmlNode* node, const char* key, const char* value) {
    int index = node->nAttrs++;
    std::snprintf(node->attrs[index].key, sizeof(node->attrs[index].key), "%s", key);
    std::snprintf(node->attrs[index].value, sizeof(node->attrs[index].value), "%s", value);
  }

  ncclTopoSystem* system() { return scene_->system(); }
  ncclTopoGraph* graph() { return graph_.get(); }

 private:
  std::unique_ptr<SearchXmlScene> scene_;
  std::unique_ptr<ncclTopoGraph> graph_;
  std::deque<ncclXmlNode> nodes_;
};

// ---------------------------------------------------------------------------
// Test list (the agreed plan -- ROCM-31973).
//
// Accepted-input contracts (landed first; they are what stops a later guard
// being tightened into rejecting valid topology files):
//
//   [C1] WellFormedChannel_TranscribesGpuDevsToRanksInOrder
//   [C2] WellFormedChannel_TranscribesNetDevsToInter
//   [C3] ExplicitRankAttribute_IsUsedInsteadOfDevLookup
//   [C4] ChannelIndex_SelectsItsOwnSliceOfIntraAndInter
//   [C5] UnknownGpuDev_IsRejected
//   [C6] GraphGeneratedByNcclTopoGetXmlFromGraph_RoundTripsBackToTheSameGraph
//        (RCCL ships no graph XML of its own, so the format its own writer
//         emits is the "our own input" contract that licenses a reject fix)
//
// Rejection tests, one per row of the Phase 1 table:
//
//   [A] ChannelIndexBeyondMaxChannels_IsRejected
//       graph->inter + 2*c and graph->intra + ngpus*c both run past the end of
//       their arrays once c >= MAXCHANNELS; MAX_SUBS (512) admits c up to 511
//       against MAXCHANNELS 256.
//   [B] MoreNetSubsThanTheChannelHolds_IsRejected
//       inter[n++] has only 2 slots per channel; nothing bounds n.
//   [C] MoreGpuSubsThanGpus_IsRejected
//       intra[g++] has only ngpus slots per channel; nothing bounds g. The
//       overrun lands inside the same ncclTopoGraph, so it is invisible to
//       ASan -- the assertion has to be on the clobbered next channel.
//   [D] RankAttributeOutsideTheRankRange_IsRejected
//       strtol into an int: "4294967296" truncates to 0 silently.
//   [E] DevAttributeTooWideForTheIdSpace_IsRejected  -- expected to be
//       reclassified (probe first); likely subsumed by the dev-not-found arm.
//
// Caller (ncclTopoGetGraphFromXmlSub), its own commit:
//   [F] MoreChannelSubsThanMaxChannels_IsRejectedByTheCaller
//       the loop over xmlGraph->nSubs and *nChannels = xmlGraph->nSubs are
//       both unclamped, so the caller admits more than the buffers hold.
//
// TO BE WRITTEN -- this block is replaced at the end of the pass with what the
// pass actually found, including reclassified rows.
// ---------------------------------------------------------------------------

// --- Accepted-input contracts ---------------------------------------------
//
// A valid channel is still accepted and still produces the right graph. These
// are what stop any of the bounds below being tightened into rejecting a
// topology file RCCL itself would write.

TEST_F(SearchXmlChannelMicrotest, WellFormedChannel_TranscribesGpuDevsToRanksInOrder) {
  // Devs deliberately out of order, and ranks are kRankBase+dev, so neither a
  // skipped dev->rank translation nor a sorted write can pass by accident.
  const int devs[kGpus] = {3, 0, 5, 1, 7, 2, 6, 4};
  ncclXmlNode* channel = NewNode("channel");
  for (int dev : devs) AddChild(channel, "gpu", std::to_string(dev).c_str());

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  for (int i = 0; i < kGpus; i++) {
    EXPECT_EQ(graph()->intra[i], SearchXmlScene::kRankBase + devs[i]) << "slot " << i;
  }
}

TEST_F(SearchXmlChannelMicrotest, WellFormedChannel_TranscribesNetDevsToInter) {
  // The <dev> attribute of a net is parsed base 16 (see strtol(str, NULL, 16)).
  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "a");
  AddChild(channel, "gpu", "0");
  AddChild(channel, "net", "b");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->inter[0], 0xa);
  EXPECT_EQ(graph()->inter[1], 0xb);
}

TEST_F(SearchXmlChannelMicrotest, ExplicitRankAttribute_IsUsedInsteadOfDevLookup) {
  // dev=0 would look up kRankBase+0; the explicit rank attribute must win.
  ncclXmlNode* channel = NewNode("channel");
  ncclXmlNode* gpu = AddChild(channel, "gpu", "0");
  SetAttr(gpu, "rank", "5");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->intra[0], 5);
}

TEST_F(SearchXmlChannelMicrotest, ChannelIndex_SelectsItsOwnSliceOfIntraAndInter) {
  constexpr int kChannel = 1;
  constexpr int kSentinel = -42;
  // Dirty channel 0's slots: the fixture zeroes the graph, so "channel 0 is
  // untouched" would otherwise be true without the offset arithmetic running.
  for (int i = 0; i < kGpus; i++) graph()->intra[i] = kSentinel;
  graph()->inter[0] = kSentinel;
  graph()->inter[1] = kSentinel;

  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "1");
  AddChild(channel, "gpu", "6");
  AddChild(channel, "net", "2");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, kChannel, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->intra[kGpus * kChannel], SearchXmlScene::kRankBase + 6);
  EXPECT_EQ(graph()->inter[2 * kChannel], 1);
  EXPECT_EQ(graph()->inter[2 * kChannel + 1], 2);
  // Channel 0 is left alone.
  for (int i = 0; i < kGpus; i++) EXPECT_EQ(graph()->intra[i], kSentinel) << "slot " << i;
  EXPECT_EQ(graph()->inter[0], kSentinel);
  EXPECT_EQ(graph()->inter[1], kSentinel);
}

TEST_F(SearchXmlChannelMicrotest, UnknownGpuDev_IsRejected) {
  // kGpus devs exist (0..7), so dev 9 names no GPU in this system.
  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "gpu", "9");

  EXPECT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSystemError);
}

TEST_F(SearchXmlChannelMicrotest,
       GraphGeneratedByNcclTopoGetXmlFromGraph_RoundTripsBackToTheSameGraph) {
  // RCCL ships no graph XML of its own, so the format its own writer emits is
  // the "our own input still parses" contract. Anything these guards reject
  // that ncclTopoGetXmlFromGraph can produce is a regression.
  const int ranks[kGpus] = {103, 100, 105, 101, 107, 102, 106, 104};
  system()->inter = 1;  // make the writer emit the leading and trailing <net>

  auto written = std::make_unique<ncclTopoGraph>();
  std::memset(written.get(), 0, sizeof(*written));
  written->id = 0;
  written->nChannels = 1;
  for (int i = 0; i < kGpus; i++) written->intra[i] = ranks[i];
  written->inter[0] = 0xa;
  written->inter[1] = 0xb;

  struct ncclXml* xml = nullptr;
  ASSERT_EQ(xmlAlloc(&xml, 64), ncclSuccess);
  std::unique_ptr<ncclXml, decltype(&std::free)> xmlOwner(xml, &std::free);
  struct ncclXmlNode* root = nullptr;
  ASSERT_EQ(xmlAddNode(xml, nullptr, "graph", &root), ncclSuccess);
  ASSERT_EQ(ncclTopoGetXmlFromChannel(written.get(), 0, system(), xml, root), ncclSuccess);
  ASSERT_EQ(root->nSubs, 1);

  ASSERT_EQ(ncclTopoGetChannelFromXml(root->subs[0], 0, system(), graph()), ncclSuccess);

  for (int i = 0; i < kGpus; i++) EXPECT_EQ(graph()->intra[i], ranks[i]) << "slot " << i;
  EXPECT_EQ(graph()->inter[0], 0xa);
  EXPECT_EQ(graph()->inter[1], 0xb);
}

// --- Rejection tests ------------------------------------------------------

TEST_F(SearchXmlChannelMicrotest, ChannelIndexBeyondMaxChannels_IsRejected) {
  // The caller indexes channels by XML sub position, bounded only by MAX_SUBS
  // (512), against graph->inter[MAXCHANNELS*2] (256 channels). At c ==
  // MAXCHANNELS the inter slice starts one element past the end of the array,
  // and the next fields of the same ncclTopoGraph are what get written --
  // so this is invisible to AddressSanitizer and must be asserted on the
  // clobbered fields.
  constexpr int kChannel = MAXCHANNELS;
  graph()->nIntraChannels = 0;
  graph()->intraNets[0] = 0;

  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "1122334455667788");

  EXPECT_EQ(ncclTopoGetChannelFromXml(channel, kChannel, system(), graph()), ncclInvalidUsage);
  EXPECT_EQ(graph()->nIntraChannels, 0);
  EXPECT_EQ(graph()->intraNets[0], 0);
}

TEST_F(SearchXmlChannelMicrotest, LastValidChannelIndex_IsStillAccepted) {
  // The tightest accepted input: one below the limit. Tightening the bound by
  // one has to break this, or the guard is free to creep inwards.
  constexpr int kChannel = MAXCHANNELS - 1;
  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "1");
  AddChild(channel, "gpu", "2");
  AddChild(channel, "net", "3");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, kChannel, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->inter[2 * kChannel], 1);
  EXPECT_EQ(graph()->inter[2 * kChannel + 1], 3);
  EXPECT_EQ(graph()->intra[kGpus * kChannel], SearchXmlScene::kRankBase + 2);
}

TEST_F(SearchXmlChannelMicrotest, MoreNetSubsThanTheChannelHolds_IsRejected) {
  // A channel's inter slice is two slots (ncclTopoGetXmlFromChannel emits a
  // leading and a trailing <net>). A third one runs into channel 1's slots.
  constexpr int kSentinel = -7;
  graph()->inter[2] = kSentinel;
  graph()->inter[3] = kSentinel;

  ncclXmlNode* channel = NewNode("channel");
  for (const char* dev : {"1", "2", "3", "4"}) AddChild(channel, "net", dev);

  EXPECT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclInvalidUsage);
  EXPECT_EQ(graph()->inter[2], kSentinel);
  EXPECT_EQ(graph()->inter[3], kSentinel);
}

TEST_F(SearchXmlChannelMicrotest, ExactlyTwoNetSubs_AreStillAccepted) {
  // The tightest accepted input for the net bound.
  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "1");
  AddChild(channel, "gpu", "0");
  AddChild(channel, "net", "2");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->inter[0], 1);
  EXPECT_EQ(graph()->inter[1], 2);
}

}  // namespace
