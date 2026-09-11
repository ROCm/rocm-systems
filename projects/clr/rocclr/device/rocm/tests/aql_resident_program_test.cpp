// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "aql_resident_program.hpp"
#include <cassert>
#include <cstdio>
#include <limits>
#include <cstring>
using namespace amd::roc::aql_resident;
int main() {
  BindingPlan plan;
  assert(plan.validateBindings({}) == PlanStatus::Binding);
  const std::vector<MutableRange> ranges{{128, 32}, {56, 8}};
  assert(plan.compile(192, 2, ranges, {{144, 1, 16}, {56, 0, 0}}) == PlanStatus::Ok);
  assert(plan.entries().size() == 2 && plan.entries()[0].targetOffset == 56);
  assert(plan.validateBindings({0x1000, 0x2000}) == PlanStatus::Ok);
  assert(plan.validateBindings({0x1000}) == PlanStatus::Binding);
  assert(plan.validateBindings({0, std::numeric_limits<uint64_t>::max()}) == PlanStatus::Overflow);
  const auto reject = [&](uint64_t bytes, uint32_t slots, const std::vector<MutableRange>& r,
                          const std::vector<BindingFixup>& e, PlanStatus expected) {
    assert(plan.compile(bytes, slots, r, e) == expected);
    assert(plan.imageBytes() == 192 && plan.bindingCount() == 2 && plan.entries().size() == 2);
    assert(plan.validateBindings({0x1000, 0x2000}) == PlanStatus::Ok);
  };
  reject(192, 2, ranges, {{0, 0, 0}}, PlanStatus::Bounds); // immutable packet header
  reject(192, 2, ranges, {{60, 0, 0}}, PlanStatus::Alignment);
  reject(192, 2, ranges, {{56, 2, 0}}, PlanStatus::Binding);
  reject(192, 2, ranges, {{56, 0, 0}, {56, 1, 0}}, PlanStatus::Overlap);
  reject(192, 2, {{56, 7}}, {{56, 0, 0}}, PlanStatus::Bounds);
  reject(192, 2, {{56, 16}, {64, 16}}, {}, PlanStatus::Overlap);
  reject(192, 2, {{190, 8}}, {}, PlanStatus::Bounds);
  reject(192, 2, {{56, 0}}, {}, PlanStatus::Bounds);
  reject(0, 2, {}, {}, PlanStatus::Bounds);
  reject(uint64_t(1) << 32, 2, {}, {}, PlanStatus::Bounds);
  reject(192, 2, ranges, {{192, 0, 0}}, PlanStatus::Bounds);
  assert(plan.compile(192, 0, {}, {}) == PlanStatus::Ok);
  assert(plan.validateBindings({}) == PlanStatus::Ok);
  Packet set{};
  set[0] = (10u << 16) | (2u << 28); // three user words: odd payload start
  set[13] = 10; set[14] = 20; set[15] = 30;
  Packet dispatch{}; dispatch[0] = 9u << 16; dispatch[1] = 64; dispatch[3] = 128;
  Packet barrier{}; barrier[0] = (8u << 16) | (1u << 8);
  Packet terminal{}; terminal[0] = 7u << 16;
  std::vector<Packet> packets{set, dispatch, barrier, terminal};
  PacketTemplate program;
  assert(program.compile(packets) == PlanStatus::Ok);
  assert(program.packetCount() == 4 && program.fixupOffset() == 256);
  assert(program.plan().entries().size() == 8);
  assert(program.image().size() % 64 == 0);
  std::vector<uint64_t> bindings;
  assert(program.bind(packets, bindings) == PlanStatus::Ok);
  auto changed = packets;
  changed[0][13] = 99;
  changed[1][1] = 32; // geometry shares an aligned pair with a constant header
  changed[1][3] = 256;
  assert(program.bind(changed, bindings) == PlanStatus::Ok);
  auto simulated = program.image();
  for (const auto& entry : program.plan().entries()) {
    auto value = bindings[entry.bindingSlot] + entry.bindingOffset;
    std::memcpy(simulated.data()+entry.targetOffset, &value, sizeof(value));
  }
  assert(std::memcmp(simulated.data(), changed.data(), changed.size()*64) == 0);
  const auto oldBindings = bindings;
  changed[1][0] ^= 1u << 8;
  assert(program.bind(changed, bindings) == PlanStatus::Binding);
  assert(bindings == oldBindings);
  changed = packets; changed[0][12] = 1; // constant companion word cannot change
  assert(program.bind(changed, bindings) == PlanStatus::Binding);
  changed = packets; changed[3][2] = 1;
  assert(program.bind(changed, bindings) == PlanStatus::Binding);
  assert(program.compile(changed) == PlanStatus::Binding);
  assert(program.packetCount() == 4); // failed compilation preserves the program
  changed = packets; changed[0][0] = (10u << 16) | (15u << 28);
  assert(program.compile(changed) == PlanStatus::Bounds);
  assert(program.bind({}, bindings) == PlanStatus::Bounds);
  auto scratchPackets = packets;
  scratchPackets[1][9] |= 1;
  scratchPackets[1][12] = 248;
  assert(program.compile(scratchPackets) == PlanStatus::Ok);
  scratchPackets[1][12] = 512;
  assert(program.bind(scratchPackets, bindings) == PlanStatus::Ok);
  scratchPackets[1][13] = 1; // immutable companion of the requirement word
  assert(program.bind(scratchPackets, bindings) == PlanStatus::Binding);
  // Exhaust every SET payload length and both dispatch layouts. A compiled
  // immutable span may cross packet boundaries, but must never skip a control
  // word or reject a declared mutable word. Failed binds preserve the output.
  for (unsigned count = 1; count <= 14; ++count) {
    for (unsigned scratch = 0; scratch <= 1; ++scratch) {
      auto layout = packets;
      layout[0] = {};
      layout[0][0] = (10u << 16) | ((count-1) << 28);
      layout[1][9] = scratch;
      layout[1][12] = scratch ? 248 : 0;
      assert(program.compile(layout) == PlanStatus::Ok);
      for (size_t packet = 0; packet < layout.size(); ++packet) {
        for (unsigned word = 0; word < 16; ++word) {
          auto candidate = layout;
          candidate[packet][word] ^= 0x100;
          const bool mutableWord = (packet == 0 && word >= 16-count) ||
              (packet == 1 && word >= 1 && word <= (scratch ? 12u : 11u));
          const auto before = bindings;
          const auto result = program.bind(candidate, bindings);
          assert(result == (mutableWord ? PlanStatus::Ok : PlanStatus::Binding));
          if (!mutableWord) assert(bindings == before);
        }
      }
    }
  }
  std::puts("PASS: resident binding-plan ABI, ranges, overlap, overflow and transactional rejection");
  std::puts("PASS: packet template, odd SET payload, geometry fixup and immutable control flow");
}
