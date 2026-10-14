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

// Phase-2 placeholder: proves the link closure of ncclTopoGetChannelFromXml,
// i.e. that the whole of search.cc links against the fakes. Replaced by the
// real tests above as they land.
TEST_F(SearchXmlChannelMicrotest, PlaceholderLinkClosure) {
  ncclXmlNode* channel = NewNode("channel");
  AddChild(channel, "gpu", "3");

  ASSERT_EQ(ncclTopoGetChannelFromXml(channel, 0, system(), graph()), ncclSuccess);
  EXPECT_EQ(graph()->intra[0], SearchXmlScene::kRankBase + 3);
}

}  // namespace
