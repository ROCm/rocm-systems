/* Copyright © Advanced Micro Devices, Inc., or its affiliates. */

// Host-only check of LastUseTracker; build and run with "make test".

#include <cstdio>
#include <unordered_map>

#include "replay_last_use.hpp"

namespace {

int failures = 0;

// Fake log addresses; the tracker only compares them.
void* const kBufA = reinterpret_cast<void*>(0x1000);
void* const kBufB = reinterpret_cast<void*>(0x2000);
void* const kStream = reinterpret_cast<void*>(0x3000);

void ExpectLine(const char* what, const std::unordered_map<void*, int>& lastUse, void* key, int want) {
  auto it = lastUse.find(key);
  const int got = it == lastUse.end() ? -1 : it->second;
  if (got != want) {
    printf("FAIL: %s: last use on line %d, want %d\n", what, got, want);
    failures++;
  }
}

void ExpectDepth(const char* what, const LastUseTracker& t, int want) {
  if (t.depth() != want) {
    printf("FAIL: %s: depth %d, want %d\n", what, t.depth(), want);
    failures++;
  }
}

void Ungrouped() {
  LastUseTracker t;
  t.UseBuffer(kBufA, 0);
  t.UseStream(kStream, 0);
  t.UseBuffer(kBufA, 3);
  ExpectLine("ungrouped buffer", t.buffers(), kBufA, 3);
  ExpectLine("ungrouped stream", t.streams(), kStream, 0);
  ExpectDepth("ungrouped", t, 0);
}

void SingleGroup() {
  LastUseTracker t;
  t.GroupStart();  // line 0
  t.UseBuffer(kBufA, 1);
  t.UseStream(kStream, 1);
  t.GroupEnd(2);
  t.UseBuffer(kBufB, 3);
  ExpectLine("grouped buffer", t.buffers(), kBufA, 2);
  ExpectLine("grouped stream", t.streams(), kStream, 2);
  ExpectLine("buffer after group", t.buffers(), kBufB, 3);
  ExpectDepth("single group", t, 0);
}

void NestedGroups() {
  LastUseTracker t;
  t.GroupStart();  // line 0
  t.UseBuffer(kBufA, 1);
  t.GroupStart();  // line 2
  t.UseBuffer(kBufB, 3);
  t.UseStream(kStream, 3);
  t.GroupEnd(4);
  ExpectLine("inner GroupEnd does not launch", t.buffers(), kBufB, 3);
  t.GroupEnd(5);
  ExpectLine("outer group buffer", t.buffers(), kBufA, 5);
  ExpectLine("inner group buffer", t.buffers(), kBufB, 5);
  ExpectLine("inner group stream", t.streams(), kStream, 5);
  ExpectDepth("nested groups", t, 0);
}

void UngroupedUseNotDeferred() {
  LastUseTracker t;
  t.UseBuffer(kBufA, 0);
  t.UseStream(kStream, 0);
  t.GroupStart();  // line 1
  t.UseBuffer(kBufB, 2);
  t.GroupEnd(3);
  ExpectLine("ungrouped buffer before a group", t.buffers(), kBufA, 0);
  ExpectLine("ungrouped stream before a group", t.streams(), kStream, 0);
  ExpectLine("grouped buffer", t.buffers(), kBufB, 3);
}

void UseAfterGroupWins() {
  LastUseTracker t;
  t.GroupStart();  // line 0
  t.UseBuffer(kBufA, 1);
  t.GroupEnd(2);
  t.UseBuffer(kBufA, 7);
  ExpectLine("ungrouped use after group", t.buffers(), kBufA, 7);
}

void GroupStateResets() {
  LastUseTracker t;
  t.GroupStart();  // line 0
  t.UseBuffer(kBufA, 1);
  t.UseStream(kStream, 1);
  t.GroupEnd(2);
  t.GroupStart();  // line 3
  t.UseBuffer(kBufB, 4);
  t.GroupEnd(5);
  ExpectLine("first group keeps its GroupEnd", t.buffers(), kBufA, 2);
  ExpectLine("first group stream keeps its GroupEnd", t.streams(), kStream, 2);
  ExpectLine("second group", t.buffers(), kBufB, 5);
}

void UnmatchedGroupEnd() {
  LastUseTracker t;
  t.UseBuffer(kBufA, 0);
  t.GroupEnd(1);
  ExpectLine("unmatched GroupEnd", t.buffers(), kBufA, 0);
  ExpectDepth("unmatched GroupEnd", t, 0);
}

// The trailing group never launches in replay, so its uses keep their own lines and depth() reports it.
void EndOfLogInsideGroup() {
  LastUseTracker t;
  t.UseBuffer(kBufA, 0);
  t.GroupStart();  // line 1
  t.UseBuffer(kBufA, 2);
  t.UseStream(kStream, 2);
  ExpectLine("open group buffer", t.buffers(), kBufA, 2);
  ExpectLine("open group stream", t.streams(), kStream, 2);
  ExpectDepth("end of log inside group", t, 1);
}

}  // namespace

int main() {
  Ungrouped();
  SingleGroup();
  NestedGroups();
  UngroupedUseNotDeferred();
  UseAfterGroupWins();
  GroupStateResets();
  UnmatchedGroupEnd();
  EndOfLogInsideGroup();
  if (failures != 0) {
    printf("%d LastUseTracker check(s) failed\n", failures);
    return 1;
  }
  printf("LastUseTracker checks passed\n");
  return 0;
}
