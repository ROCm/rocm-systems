/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Host-only microtests for src/graph/search.cc

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

TEST_F(SearchXmlChannelMicrotest, NegativeChannelIndex_IsRejected) {
  // Defensive: the one caller cannot pass a negative c today, but the function
  // is externally linked and a negative index would write *before* both
  // arrays. Without this the c < 0 half of the guard is unpinned.
  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "1");

  EXPECT_EQ(ncclTopoGetChannelFromXml(channel, -1, system(), graph()), ncclInvalidUsage);
}

TEST_F(SearchXmlChannelMicrotest, SubnodesThatAreNeitherNetNorGpu_AreIgnored) {
  // The loop requires a dev attribute on every sub but only acts on net and
  // gpu, so an unrecognised element must consume no slot in either array.
  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "net", "1");
  AddChild(channel, "nvswitch", "9");
  AddChild(channel, "gpu", "2");
  AddChild(channel, "net", "3");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->inter[0], 1);
  EXPECT_EQ(graph()->inter[1], 3);
  EXPECT_EQ(graph()->intra[0], SearchXmlScene::kRankBase + 2);
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

TEST_F(SearchXmlChannelMicrotest, MoreGpuSubsThanTheSystemHasGpus_IsRejected) {
  // A channel's intra slice is ngpus slots wide. Extra <gpu> subs run into
  // channel 1's slots -- still inside the same ncclTopoGraph, so ASan cannot
  // see it and the assertion has to be on the clobbered slots themselves.
  constexpr int kSentinel = -7;
  for (int i = 0; i < kGpus + 4; i++) graph()->intra[i] = kSentinel;

  ncclXmlNode* channel = NewNode("channel");
  for (int i = 0; i < kGpus + 2; i++) {
    AddChild(channel, "gpu", std::to_string(i % kGpus).c_str());
  }

  EXPECT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclInvalidUsage);
  EXPECT_EQ(graph()->intra[kGpus], kSentinel);
  EXPECT_EQ(graph()->intra[kGpus + 1], kSentinel);
}

TEST_F(SearchXmlChannelMicrotest, ExactlyNgpusGpuSubs_AreStillAccepted) {
  // The tightest accepted input for the gpu bound: a full channel, which is
  // exactly what ncclTopoGetXmlFromChannel emits.
  ncclXmlNode* channel = NewNode("channel");
  for (int i = 0; i < kGpus; i++) AddChild(channel, "gpu", std::to_string(i).c_str());

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  for (int i = 0; i < kGpus; i++) {
    EXPECT_EQ(graph()->intra[i], SearchXmlScene::kRankBase + i) << "slot " << i;
  }
}

TEST_F(SearchXmlChannelMicrotest, RankAttributeTooWideForAnInt_IsRejected) {
  // strtol yields a long; the assignment to int truncates. 4294967296 is the
  // interesting case: it truncates to 0, a perfectly plausible rank, so the
  // graph silently gains the wrong one rather than failing a later check.
  ncclXmlNode* channel = NewNode("channel");
  ncclXmlNode* gpu = AddChild(channel, "gpu", "0");
  SetAttr(gpu, "rank", "4294967296");

  EXPECT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclInvalidUsage);
  EXPECT_EQ(graph()->intra[0], 0);  // nothing was written
}

TEST_F(SearchXmlChannelMicrotest, NegativeRankAttribute_IsRejected) {
  ncclXmlNode* channel = NewNode("channel");
  ncclXmlNode* gpu = AddChild(channel, "gpu", "0");
  SetAttr(gpu, "rank", "-5");

  EXPECT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclInvalidUsage);
}

TEST_F(SearchXmlChannelMicrotest, LargestRepresentableRankAttribute_IsStillAccepted) {
  // The tightest accepted input at the top of the range. The rank= attribute
  // exists so a file can name ranks the local dev lookup cannot resolve, so
  // the bound must stay at INT_MAX rather than the local system's rank set.
  ncclXmlNode* channel = NewNode("channel");
  ncclXmlNode* gpu = AddChild(channel, "gpu", "0");
  SetAttr(gpu, "rank", std::to_string(INT_MAX).c_str());

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->intra[0], INT_MAX);
}

TEST_F(SearchXmlChannelMicrotest, ZeroRankAttribute_IsStillAccepted) {
  // The tightest accepted input at the bottom of the range: rank 0 must not
  // be swept up by a bound tightened from "< 0" to "<= 0".
  ncclXmlNode* channel = NewNode("channel");
  ncclXmlNode* gpu = AddChild(channel, "gpu", "1");
  SetAttr(gpu, "rank", "0");
  graph()->intra[0] = -7;  // dirty it: the fixture zeroes the graph

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);

  EXPECT_EQ(graph()->intra[0], 0);
}

// --- The caller: ncclTopoGetGraphFromXmlSub -------------------------------

class SearchXmlGraphMicrotest : public SearchXmlChannelMicrotest {
 protected:
  // A <graph> carrying every attribute the importer requires, with `nSubs`
  // single-GPU <channel> children and the given nchannels= attribute.
  ncclXmlNode* MakeGraph(int nchannels, int nSubs, int id = 0) {
    ncclXmlNode* g = NewNode("graph");
    SetAttr(g, "id", std::to_string(id).c_str());
    SetAttr(g, "crossnic", "0");
    SetAttr(g, "pattern", "4");
    SetAttr(g, "nchannels", std::to_string(nchannels).c_str());
    SetAttr(g, "speedintra", "10");
    SetAttr(g, "speedinter", "10");
    SetAttr(g, "latencyinter", "0");
    SetAttr(g, "typeintra", "NVL");
    SetAttr(g, "typeinter", "PHB");
    SetAttr(g, "samechannels", "1");
    for (int i = 0; i < nSubs; i++) {
      ncclXmlNode* channel = NewNode("channel");
      channel->parent = g;
      g->subs[g->nSubs++] = channel;
      AddChild(channel, "gpu", std::to_string(i % kGpus).c_str());
    }
    return g;
  }
};

TEST_F(SearchXmlGraphMicrotest, NchannelsClaimingMoreChannelsThanArePresent_IsRejected) {
  // nchannels is copied straight into graph->nChannels, which downstream code
  // uses as the loop bound over the MAXCHANNELS-sized intra/inter arrays.
  int nChannels = -1;

  EXPECT_EQ(ncclTopoGetGraphFromXmlSub(MakeGraph(999999, 1), system(), graph(), &nChannels),
            ncclInvalidUsage);
}

TEST_F(SearchXmlGraphMicrotest, NchannelsOneMoreThanThePresentChannels_IsRejected) {
  // The tightest failing input from above. The 999999 case above catches the
  // guard being removed, but only this one catches it being loosened to admit
  // an off-by-one overclaim.
  int nChannels = -1;

  EXPECT_EQ(ncclTopoGetGraphFromXmlSub(MakeGraph(2, 1), system(), graph(), &nChannels),
            ncclInvalidUsage);
}

TEST_F(SearchXmlGraphMicrotest, NchannelsOneFewerThanThePresentChannels_IsRejected) {
  // The tightest failing input from below: a file supplying more channels than
  // it declares would leave the trailing ones imported but uncounted.
  int nChannels = -1;

  EXPECT_EQ(ncclTopoGetGraphFromXmlSub(MakeGraph(1, 2), system(), graph(), &nChannels),
            ncclInvalidUsage);
}

TEST_F(SearchXmlGraphMicrotest, NchannelsMatchingTheChannelsPresent_IsStillAccepted) {
  int nChannels = -1;

  ASSERT_EQ(ncclTopoGetGraphFromXmlSub(MakeGraph(3, 3), system(), graph(), &nChannels),
            ncclSuccess);

  EXPECT_EQ(graph()->nChannels, 3);
  EXPECT_EQ(nChannels, 3);
  // The channels really were imported, not merely counted.
  for (int c = 0; c < 3; c++) {
    EXPECT_EQ(graph()->intra[kGpus * c], SearchXmlScene::kRankBase + c) << "channel " << c;
  }
}

TEST_F(SearchXmlGraphMicrotest, TheLargestPermittedNchannels_IsStillAccepted) {
  // The tightest accepted input: exactly MAXCHANNELS channels. Tightening the
  // bound by one has to break this.
  int nChannels = -1;

  ASSERT_EQ(ncclTopoGetGraphFromXmlSub(MakeGraph(MAXCHANNELS, MAXCHANNELS), system(), graph(),
                                       &nChannels),
            ncclSuccess);

  EXPECT_EQ(graph()->nChannels, MAXCHANNELS);
}

TEST_F(SearchXmlGraphMicrotest, GraphForAnotherIdIsIgnoredWithoutBeingValidated) {
  // The id mismatch returns before any attribute is read, so a <graph> for a
  // different algorithm must not be rejected on this graph's behalf.
  int nChannels = -1;
  graph()->id = 0;

  EXPECT_EQ(ncclTopoGetGraphFromXmlSub(MakeGraph(999999, 1, /*id=*/2), system(), graph(),
                                       &nChannels),
            ncclSuccess);
  EXPECT_EQ(graph()->nChannels, 0);
}

}  // namespace
