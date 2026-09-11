// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "aql_argument_bindings.hpp"
#include "aql_binding_update.hpp"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <memory_resource>
using namespace amd::roc::aql_resident;

static void apply(const BindingPlan& plan, const BindingUpdate& update,
                  const std::vector<uint64_t>& values, std::vector<uint8_t>& image,
                  const std::vector<uint32_t>& argumentSlots = {}) {
  const auto* data = static_cast<const uint8_t*>(update.data(values));
  for (uint32_t i = 0; i < update.count(); ++i) {
    BindingFixup entry;
    if (update.sparse()) std::memcpy(&entry, data+update.entriesOffset()+i*sizeof(entry), sizeof(entry));
    else if (update.arguments()) {
      entry = plan.entries()[argumentSlots[i]];
      entry.bindingSlot = i;
    } else entry = plan.entries()[i];
    uint64_t value;
    std::memcpy(&value, data+entry.bindingSlot*8, 8);
    value += entry.bindingOffset;
    assert(uint64_t(entry.targetOffset)+8 <= image.size());
    std::memcpy(image.data()+entry.targetOffset, &value, 8);
  }
}

int main() {
  for (unsigned count = 2; count <= 14; ++count) {
    std::vector<Packet> packets;
    std::vector<uint8_t> ordinary(8*64);
    std::vector<uint32_t> targets;
    for (unsigned i = 0; i < 8; ++i) {
      Packet args{}; args[0] = (10u << 16) | ((count-1) << 28);
      uint64_t address = 0x100000000ull + i*256;
      std::memcpy(ordinary.data()+i*64+40, &address, 8);
      std::memcpy(args.data()+16-count, &address, 8);
      targets.push_back(packets.size()*64+(16-count)*4);
      packets.push_back(args);
      Packet dispatch{}; dispatch[0] = 9u << 16;
      packets.push_back(dispatch);
    }
    Packet terminal{}; terminal[0] = 7u << 16; packets.push_back(terminal);
    PacketTemplate program;
    assert(program.compile(packets) == PlanStatus::Ok);
    std::vector<uint64_t> original;
    assert(program.bind(packets, original) == PlanStatus::Ok);
    ArgumentBindingPlan projection;
    assert(projection.compile(ordinary, targets, program.plan()));
    // Graph capture uses an aligned allocator. Binding must not depend on the
    // allocator type, nor introduce a full packet copy just to adapt inputs.
    std::pmr::monotonic_buffer_resource storage;
    std::pmr::vector<uint8_t> alternate(ordinary.begin(), ordinary.end(), &storage);
    ArgumentBindingPlan alternateProjection;
    assert(alternateProjection.compile(alternate, targets, program.plan()));
    auto alternateValues = original;
    bool alternateChanged = true;
    assert(alternateProjection.bind(alternate, alternateValues, alternateChanged));
    assert(!alternateChanged && alternateValues == original);
    assert(alternateProjection.rebaseValidated(alternate));
    auto values = original;
    bool changed = true;
    assert(projection.bind(ordinary, values, changed) && !changed);
    // Each queue misses some intervening graph generations. Compare against
    // its OWN applied values, including reversions and unrelated slot changes.
    std::vector<uint64_t> queueValues[2] = {original, original};
    std::vector<uint8_t> queueImages[2] = {program.image(), program.image()};
    for (unsigned generation = 1; generation <= 100; ++generation) {
      const unsigned kernel = generation%8;
      uint64_t address = generation%3 ? (uint64_t(generation)<<32) + kernel*256 : 0x100000000ull+kernel*256;
      std::memcpy(ordinary.data()+kernel*64+40, &address, 8);
      std::memcpy(packets[2*kernel].data()+16-count, &address, 8);
      assert(projection.bind(ordinary, values, changed));
      std::vector<uint64_t> reference;
      assert(program.bind(packets, reference) == PlanStatus::Ok);
      assert(values == reference);
      // Exercise the precompiled argument-only table as well as arbitrary
      // sparse entries: make all pointer bindings differ from the initial image.
      auto allArguments = values;
      for (auto slot : projection.slots()) allArguments[slot] = original[slot] ^ 0x10001000ull;
      BindingUpdate arguments;
      assert(arguments.prepare(program.plan(), original, allArguments, projection.slots()) == PlanStatus::Ok);
      assert(arguments.arguments());
      auto argumentImage = program.image(), argumentExpected = program.image();
      apply(program.plan(), arguments, allArguments, argumentImage, projection.slots());
      for (auto slot : projection.slots()) {
        std::memcpy(argumentExpected.data()+program.plan().entries()[slot].targetOffset, &allArguments[slot], 8);
      }
      assert(argumentImage == argumentExpected);
      const unsigned queue = generation%2;
      BindingUpdate upload;
      assert(upload.prepare(program.plan(), queueValues[queue], values) == PlanStatus::Ok);
      assert(upload.count() == 0 || upload.sparse());
      apply(program.plan(), upload, values, queueImages[queue]);
      assert(std::memcmp(queueImages[queue].data(), packets.data(), packets.size()*64) == 0);
      queueValues[queue] = values;
    }
    const auto before = values;
    auto invalid = ordinary;
    invalid[12] ^= 1; // Geometry change requires full validation, not projection.
    assert(!projection.bind(invalid, values, changed) && values == before);
    std::memset(invalid.data()+40, 0, 8);
    assert(!projection.bind(invalid, values, changed) && values == before);
    invalid = ordinary; std::memset(invalid.data()+40, 0, 8);
    assert(!projection.bind(invalid, values, changed) && values == before);
    // Once the full encoder accepted a geometry update, a subsequent argument
    // update can project against that reference without restoring old geometry.
    invalid = ordinary; invalid[12] ^= 1;
    assert(projection.rebaseValidated(invalid));
    assert(projection.bind(invalid, values, changed) && !changed);
    assert(!projection.bind(ordinary, values, changed));
    BindingUpdate unchanged;
    assert(unchanged.prepare(program.plan(), values, values) == PlanStatus::Ok);
    assert(unchanged.count() == 0 && unchanged.bytes() == 0);
    auto dense = values;
    for (auto& value : dense) value ^= 0x10000;
    BindingUpdate full;
    assert(full.prepare(program.plan(), values, dense) == PlanStatus::Ok && !full.sparse());
    assert(full.count() == program.plan().entries().size());
    auto actual = program.image(), expected = program.image();
    apply(program.plan(), full, dense, actual);
    for (const auto& entry : program.plan().entries()) {
      std::memcpy(expected.data()+entry.targetOffset, &dense[entry.bindingSlot], 8);
    }
    assert(actual == expected);
  }
  std::puts("PASS: direct bindings match encoder bindings, odd payloads, rejection, rebase");
  std::puts("PASS: sparse/dense GPU-fixup simulation, queue-local skipped generations and reversions");
}
