/*
Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// Policy names and text form, for the trace and the policy snapshot test.

#include "rccl_scheduler.h"

const char* rcclFactName(rcclFact fact) {
  static const char* const kNames[] = {"algo-env",      "in-group",      "must-enqueue",    "cta-zero",
                                       "fabric",        "in-place",      "capturing",       "ce-ar-graph-ok",
                                       "sysmem",        "symk-requested", "symk-eligible",  "symk-veto",
                                       "ce-ar-2shot-ok", "ce-rs-ok",     "reroute-ring",    "reroute-ce-reg"};
  static_assert(sizeof(kNames) / sizeof(kNames[0]) == RCCL_FACT_COUNT, "name every fact");
  return fact < RCCL_FACT_COUNT ? kNames[fact] : "?";
}

const char* rcclLimitName(rcclLimit limit) {
  static const char* const kNames[] = {"none",        "dda-entry",  "dda-ll",      "dda-ll128",
                                       "dda-vmm",     "dda-ar-vmm-first", "dda-scratch", "ce-reg",
                                       "ce-nonreg",   "ce-reg-ag",  "symk"};
  static_assert(sizeof(kNames) / sizeof(kNames[0]) == RCCL_LIMIT_COUNT, "name every limit");
  return limit < RCCL_LIMIT_COUNT ? kNames[limit] : "?";
}

namespace {

void appendFacts(std::string& out, const char* label, rcclFactMask mask) {
  if (mask == 0) return;
  out += " ";
  out += label;
  const char* sep = "=";
  for (int f = 0; f < RCCL_FACT_COUNT; f++) {
    if ((mask & rcclFactBit((rcclFact)f)) == 0) continue;
    out += sep;
    out += rcclFactName((rcclFact)f);
    sep = ",";
  }
}

void appendRule(std::string& out, const char* row, const rcclRule& r) {
  out += row;
  out += " rule ";
  out += r.backend->name;
  if (r.window != RCCL_LIMIT_NONE) {
    out += " window=";
    out += rcclLimitName(r.window);
  }
  appendFacts(out, "if", r.need);
  appendFacts(out, "not", r.avoid);
  appendFacts(out, "yield-to", r.yield);
  appendFacts(out, "any-size-when", r.anySize);
  out += "\n";
}

void appendRow(std::string& out, const rcclRow& row) {
  for (size_t i = 0; i < row.nPrologue; i++) {
    out += row.name;
    out += row.prologue[i].skipRules ? " skip-rules-if " : " eval ";
    out += rcclFactName(row.prologue[i].fact);
    out += "\n";
  }
  if (row.preferred != nullptr) {
    out += row.name;
    out += " preferred-backends\n";
  }
  for (size_t i = 0; i < row.nRules; i++) {
    const rcclRule& r = row.rules[i];
    if (r.group == nullptr) {
      appendRule(out, row.name, r);
      continue;
    }
    for (size_t j = 0; j < r.group->nRules; j++) appendRule(out, row.name, rcclRuleInGroup(r, r.group->rules[j]));
  }
  for (size_t i = 0; i < row.nFallbacks; i++) {
    out += row.name;
    out += " fallback ";
    out += row.fallbacks[i]->name;
    out += "\n";
  }
}

}  // namespace

std::string rcclPolicyDescribe(const rcclPolicy& policy) {
  std::string out = "policy ";
  out += policy.arch != nullptr ? policy.arch : "default";
  out += "\n";
  for (const rcclRow* row : {policy.allReduce, policy.allGather, policy.reduceScatter, policy.alltoAll, policy.broadcast})
    appendRow(out, *row);
  return out;
}

