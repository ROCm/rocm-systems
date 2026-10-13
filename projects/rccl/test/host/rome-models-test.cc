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

// The happy path the bounds guards above must not regress: a tree naming GPUs the
// system has is accepted and transcribed into the first treeBase row, with the row
// after it terminated so the row count reads back as one. GPU 14 is two digits, so
// the multi-digit arms of both the accumulator and the transcription are exercised.
// The tree is dereferenced before anything else is checked. Every caller in this file
// happens to test for null first, so this does not fix a reachable crash; it makes the
// function's own contract total, so a future caller need not know to repeat that test.
// Reported as a bad argument rather than invalid usage, matching how the rest of the
// codebase reports a null pointer.
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

// Trees are separated by '|' and each claims its own row, in order, with an empty row
// closing the list -- that empty row is how every reader of treeBase finds the end, so
// it is part of the contract rather than an implementation detail.
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

// The numbers of one segment are accumulated into a fixed NCCL_TOPO_MAX_NODES-entry
// array before being transcribed, so a segment holding more of them than the topology
// can ever contain must be refused rather than written past the end of that array.
TEST_F(RomeModelsParseGraphLightMicrotest, MoreNumbersInSegmentThanMaxNodes_IsRejected) {
  std::string tooManyNumbers;
  for (int i = 0; i <= NCCL_TOPO_MAX_NODES; i++) tooManyNumbers += "(0)";

  EXPECT_EQ(parseGraphLight(tooManyNumbers.c_str(), system(), graph(), nullptr),
            ncclInvalidUsage);
}

// A GPU number is rendered back into text through a fixed 10-byte buffer, which is one
// byte short of the widest int. The number is only rendered once it has matched a real
// device, so this needs a system whose device number is that wide -- but the function is
// handed those numbers and does not get to assume they are small.
TEST_F(RomeModelsParseGraphLightMicrotest, GpuNumberAsWideAsAnInt_IsTranscribed) {
  // Ten digits: one more than the buffer holds.
  constexpr int kWideDev = 2000000000;
  system()->nodes[GPU].nodes[kGpus - 1].gpu.dev = kWideDev;
  const std::string tree = "(" + std::to_string(kWideDev) + ")";

  ASSERT_EQ(parseGraphLight(tree.c_str(), system(), graph(), nullptr), ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], tree.c_str());
}

// The widest an int can render is the most negative one, whose sign costs a further
// byte. A tree cannot spell a negative number, so it can only arrive through the remap
// table -- which is just an array of ints and is not constrained to hold GPU numbers.
TEST_F(RomeModelsParseGraphLightMicrotest, MostNegativeGpuNumberFromGpuMap_IsTranscribed) {
  system()->nodes[GPU].nodes[kGpus - 1].gpu.dev = INT_MIN;
  std::vector<int> mapToMostNegative(kGpus, 0);
  mapToMostNegative[0] = INT_MIN;

  ASSERT_EQ(parseGraphLight("(0)", system(), graph(), mapToMostNegative.data()), ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], "(-2147483648)");
}

// The counterpart to the range check below: every number the table can hold is accepted
// and comes out translated through it, including the highest, so the range check cannot
// be tightened into rejecting a legitimate topology. A reversing table is used rather
// than an identity one so the translation itself is visible in the output.
TEST_F(RomeModelsParseGraphLightMicrotest, GpuNumbersAreTranslatedThroughGpuMap) {
  std::vector<int> reversingMap(kGpus);
  for (int i = 0; i < kGpus; i++) reversingMap[i] = kGpus - 1 - i;

  const std::string highestAndLowest =
      "(" + std::to_string(kGpus - 1) + "(0))";  // "(15(0))" for 16 GPUs

  ASSERT_EQ(parseGraphLight(highestAndLowest.c_str(), system(), graph(), reversingMap.data()),
            ncclSuccess);

  EXPECT_STREQ(graph()->treeBase[0], "(0(15))");
}

// A GPU number is matched against the device numbers the system actually reports, and a
// tree naming one that is absent is refused. Reported as an internal error, unlike the
// invalid-usage results above; that predates this work and is left as it was.
TEST_F(RomeModelsParseGraphLightMicrotest, GpuNotInTheSystem_IsRejected) {
  const std::string absentGpu = "(" + std::to_string(kGpus + 4) + ")";

  EXPECT_EQ(parseGraphLight(absentGpu.c_str(), system(), graph(), nullptr), ncclInternalError);
}

// A GPU number is used directly as an index into the caller's remap table, which is only
// as long as the system has GPUs, so a number naming a GPU the system does not have reads
// past the end of that table. Callers inside this file pass built-in trees, but the
// function cannot tell those from a tree that came out of the environment.
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

// After consuming a number, transcription scans forward for the parenthesis that ends
// it. A segment whose last token is a number has no such parenthesis, and the scan does
// not stop at the end of the string, so it reads on through whatever follows the
// environment variable in memory. Such a tree must be refused.
TEST_F(RomeModelsParseGraphLightMicrotest, TrailingNumberWithNoClosingParen_IsRejected) {
  // Long enough that std::string puts it on the heap, so the overread is a heap
  // overflow ASan can see rather than an unremarkable read of adjacent stack.
  const std::string truncatedTree = "(1)(2)(3)(4)(5)(6";

  EXPECT_EQ(parseGraphLight(truncatedTree.c_str(), system(), graph(), nullptr),
            ncclInvalidUsage);
}

// Transcription consumes one parsed number per token, but only digits are ever parsed
// into numbers, so any other character is a token the parse never counted and its read
// runs past the numbers this segment produced. gpus[] is not cleared between segments,
// so such a read silently resolves to the previous segment's GPU rather than failing:
// here the 'x' would become GPU 6. A segment with more tokens than numbers must be
// refused.
TEST_F(RomeModelsParseGraphLightMicrotest, MoreTokensThanNumbersParsed_IsRejected) {
  EXPECT_EQ(parseGraphLight("(5)(6)|(0)(x)", system(), graph(), nullptr), ncclInvalidUsage);
}

// Each '|'-separated segment claims the next treeBase row, and one more row is used to
// terminate the list, so a string carrying more segments than there are rows must be
// refused. treeBase is the last member of ncclTopoGraph, so this one runs off the end
// of the graph itself rather than into a neighbouring row.
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

// Each number is accumulated a digit at a time into an int. A number too wide for an
// int must be refused, not wrapped: 4294967299 wraps to 3, which would otherwise be
// silently accepted as a well-formed tree naming a GPU the string never asked for.
TEST_F(RomeModelsParseGraphLightMicrotest, NumberWiderThanIntRange_IsRejected) {
  EXPECT_EQ(parseGraphLight("(4294967299)", system(), graph(), nullptr), ncclInvalidUsage);
}

// A segment is transcribed into one fixed-width treeBase row. Parentheses are copied
// through one-for-one and are not counted against any limit, so a segment can outrun
// the row; it must be refused rather than run on into the row after it (and, from the
// last row, past treeBase altogether). The row can run out under either of the two
// things written to it, a parenthesis or a digit of a GPU number, so both are checked.
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

}  // namespace
