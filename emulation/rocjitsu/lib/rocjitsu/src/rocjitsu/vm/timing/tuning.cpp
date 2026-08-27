// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/timing/tuning.h"

#include "util/log.h"

#include <algorithm>
#include <cstdlib>
#include <flatbuffers/flexbuffers.h>
#include <flatbuffers/idl.h>
#include <map>

namespace rocjitsu::timing {
namespace {

/// @brief A latency nobody named, in cycles.
///
/// @details Roughly five microseconds at any plausible shader clock: longer
/// than most of the kernels this plane is asked about, so a run resting on it
/// is self-evidently not a measurement. That is the intent. An unconfigured
/// parameter has to be visible in the numbers as well as in the report, because
/// a plausible default is indistinguishable from a value someone chose.
constexpr std::uint64_t kUnnamedLatencyCycles = 10000;

/// @brief Re-read the config document with no schema.
///
/// @details Done here rather than through the plugin loader's copy of the same
/// two lines, so that the timing plane does not drag a dynamic-library loader
/// into every image that links it. The DBT translation tool and the HSA hook
/// library link the simulator and have no business gaining a dlopen dependency
/// because a config parser wanted a JSON reader.
bool flexbuffer_from_json(const std::string &json, flexbuffers::Builder &out) {
  flatbuffers::Parser parser;
  return parser.ParseFlexBuffer(json.c_str(), nullptr, &out);
}

/// @brief Flatten a nested map into dotted paths, so `l2: { ways: 16 }` and
///        `"l2.ways": 16` are the same key.
///
/// @details Both spellings occur in practice: a person writes the nested form
/// and a generator writes the flat one. Accepting only one of them turns a
/// perfectly clear config into silent fallbacks.
void flatten(const flexbuffers::Reference &value, const std::string &prefix,
             std::map<std::string, std::string> &out) {
  if (value.IsMap()) {
    const flexbuffers::Map map = value.AsMap();
    const flexbuffers::TypedVector keys = map.Keys();
    for (std::size_t index = 0; index < keys.size(); ++index) {
      const std::string key = keys[index].AsString().str();
      // A key beginning with two slashes is a comment. The config format has no
      // comment syntax of its own and the alternative is a config nobody can
      // read, so they are carried as data and skipped here.
      if (key.rfind("//", 0) == 0)
        continue;
      flatten(map[key.c_str()], prefix.empty() ? key : prefix + "." + key, out);
    }
    return;
  }
  if (value.IsVector() || value.IsNull())
    return;
  out[prefix] = value.ToString();
}

} // namespace

/// @brief Reads one key, recording whether the config named it.
class Resolver {
public:
  Resolver(const std::map<std::string, std::string> &machine, Tuning &tuning)
      : machine_(machine), tuning_(tuning) {}

  std::uint64_t integer(std::string_view key, std::uint64_t pessimistic) {
    const std::string name(key);
    const auto found = machine_.find(name);
    if (found == machine_.end()) {
      tuning_.fell_back.push_back(name);
      return pessimistic;
    }
    char *end = nullptr;
    const double parsed = std::strtod(found->second.c_str(), &end);
    if (end == found->second.c_str() || !(parsed >= 0.0)) {
      util::Logger::warn("timing: machine." + name + " is not a number, using the slow default");
      tuning_.fell_back.push_back(name);
      return pessimistic;
    }
    tuning_.resolved.push_back(name + " = " + found->second);
    return static_cast<std::uint64_t>(parsed);
  }

  double real(std::string_view key, double pessimistic) {
    const std::string name(key);
    const auto found = machine_.find(name);
    if (found == machine_.end()) {
      tuning_.fell_back.push_back(name);
      return pessimistic;
    }
    char *end = nullptr;
    const double parsed = std::strtod(found->second.c_str(), &end);
    if (end == found->second.c_str() || !(parsed > 0.0)) {
      util::Logger::warn("timing: machine." + name +
                         " is not a positive rate, using the slow "
                         "default");
      tuning_.fell_back.push_back(name);
      return pessimistic;
    }
    tuning_.resolved.push_back(name + " = " + found->second);
    return parsed;
  }

  /// @brief Cycles to charge one instruction of @p cls on its issue port.
  ///
  /// @details A class the config does not name resolves to the largest issue
  /// cost it gives any class. That is what makes InstClass::Unknown expensive
  /// without the config having to anticipate it: an opcode nobody classified is
  /// charged as much as the most expensive thing the part can do, so it makes a
  /// run read slow and look suspicious rather than fast and look accurate.
  std::uint64_t issue_cycles(InstClass cls) {
    const std::string key = std::string(inst_class_name(cls)) + ".issue_cycles";
    const auto found = machine_.find(key);
    if (found != machine_.end()) {
      tuning_.resolved.push_back(key + " = " + found->second);
      return std::max<std::uint64_t>(
          1, static_cast<std::uint64_t>(std::strtod(found->second.c_str(), nullptr)));
    }
    std::uint64_t widest = 1;
    for (const auto &entry : machine_) {
      if (entry.first.size() < 13 ||
          entry.first.compare(entry.first.size() - 13, 13, ".issue_cycles") != 0)
        continue;
      widest =
          std::max(widest, static_cast<std::uint64_t>(std::strtod(entry.second.c_str(), nullptr)));
    }
    tuning_.fell_back.push_back(
        key + " (resolved to the widest issue cost named: " + std::to_string(widest) + ")");
    return widest;
  }

private:
  const std::map<std::string, std::string> &machine_;
  Tuning &tuning_;
};

Tuning Tuning::parse(const std::string &config_json) {
  Tuning tuning;

  flexbuffers::Builder builder;
  if (!flexbuffer_from_json(config_json, builder))
    return tuning;
  const flexbuffers::Reference root = flexbuffers::GetRoot(builder.GetBuffer());
  if (!root.IsMap())
    return tuning;
  const flexbuffers::Reference block = root.AsMap()["timing"];
  if (!block.IsMap())
    return tuning;

  const flexbuffers::Map timing = block.AsMap();
  const flexbuffers::Reference enabled = timing["enabled"];
  // A block that is present but not enabled is a config that describes the part
  // and asks for it not to be modelled, which is a useful thing to be able to
  // say without deleting the numbers.
  tuning.enabled = !enabled.IsNull() ? enabled.AsBool() : true;
  if (!tuning.enabled)
    return tuning;

  const flexbuffers::Reference clock = timing["clock_mhz"];
  if (clock.IsNull()) {
    util::Logger::warn("timing: no clock_mhz, taking 1 GHz so cycles and nanoseconds coincide");
    tuning.fell_back.emplace_back("clock_mhz");
  } else {
    tuning.clock_mhz = clock.AsDouble();
  }

  std::map<std::string, std::string> machine;
  flatten(timing["machine"], "", machine);
  Resolver read(machine, tuning);

  tuning.compute_units = read.integer("compute_units", 1);
  tuning.xcds = read.integer("xcds", 1);
  tuning.simd_lanes = read.integer("simd_lanes", 1);
  tuning.wave_slots_per_cu = read.integer("wave_slots_per_cu", 1);
  // A register file with one register and a compute unit with one byte of group
  // memory: absurd, and deliberately so. Both resolve occupancy to a single
  // wavefront, which is the slowest the part can run and the only honest
  // reading of a config that does not describe it.
  tuning.vector_registers_per_cu = read.integer("vector_registers_per_cu", 1);
  tuning.scalar_registers_per_cu = read.integer("scalar_registers_per_cu", 1);
  tuning.lds_bytes_per_cu = read.integer("lds_bytes_per_cu", 1);

  for (std::size_t index = 0; index < kNumInstClasses; ++index)
    tuning.issue_cycles[index] = read.issue_cycles(static_cast<InstClass>(index));
  for (std::size_t index = 0; index < kNumFunctionalUnits; ++index) {
    tuning.ports[index] = read.real(
        std::string(functional_unit_name(static_cast<FunctionalUnit>(index))) + ".ports", 1.0);
  }

  // A result latency nobody named resolves the same way every other latency
  // here does: to a figure long enough that a run resting on it is self-
  // evidently not a measurement. Resolving it to a plausible few cycles instead
  // would make a forgotten key indistinguishable from a chosen one.
  tuning.vector_alu_result_cycles = read.integer("vector_alu.result_cycles", kUnnamedLatencyCycles);
  tuning.scalar_alu_result_cycles = read.integer("scalar_alu.result_cycles", kUnnamedLatencyCycles);
  tuning.transcendental_result_cycles =
      read.integer("transcendental.result_cycles", kUnnamedLatencyCycles);
  tuning.matrix_multiply_result_cycles =
      read.integer("matrix_multiply.result_cycles", kUnnamedLatencyCycles);

  const std::uint64_t matrix_default = read.integer("matrix_multiply.macs_per_cycle", 1);
  static constexpr const char *kMatrixKeys[] = {
      "matrix_multiply.macs_per_cycle",        "matrix_multiply.macs_per_cycle.f64",
      "matrix_multiply.macs_per_cycle.f32",    "matrix_multiply.macs_per_cycle.f16",
      "matrix_multiply.macs_per_cycle.bf16",   "matrix_multiply.macs_per_cycle.narrow",
      "matrix_multiply.macs_per_cycle.integer"};
  for (std::size_t index = 0; index < kNumMatrixTypes; ++index)
    tuning.matrix_macs_per_cycle[index] = read.integer(kMatrixKeys[index], matrix_default);
  tuning.lane_addresses_per_cycle = read.real("vector_memory.lane_addresses_per_cycle", 1.0);
  tuning.wavefront_issue_cycles = read.integer("wavefront_issue_cycles", 0);
  // Falls back to the widest issue cost the config names, like any other
  // unnamed class cost: a front end nobody described should read slow.
  tuning.front_end_cycles = read.integer(
      "front_end.issue_cycles", tuning.issue_cycles[static_cast<std::size_t>(InstClass::Unknown)]);
  // Per class, defaulting to the one shared figure so that a config naming
  // only the shared one behaves exactly as it did.
  for (std::size_t index = 0; index < kNumInstClasses; ++index) {
    const std::string key = std::string("front_end.") +
                            inst_class_name(static_cast<InstClass>(index)) + ".issue_cycles";
    const double cycles = read.real(key, static_cast<double>(tuning.front_end_cycles));
    tuning.front_end_sixteenths[index] =
        static_cast<std::uint64_t>(cycles * static_cast<double>(kFrontEndScale) + 0.5);
  }
  tuning.straggler_cycles = read.integer("straggler_cycles", 0);
  tuning.barrier_cycles = read.integer("barrier_cycles", 0);
  tuning.barrier_lockstep = read.real("barrier_lockstep", 0.0);
  tuning.issue_occupancy_exponent = read.real("issue_occupancy_exponent", 0.0);
  tuning.stall_exposed_fraction = read.real("stall_exposed_fraction", 1.0);
  tuning.latency_exposure_scale = read.real("latency_exposure_scale", 1.0);
  tuning.fill_exposure_scale = read.real("fill_exposure_scale", 1.0);
  tuning.fill_ramp_scale = read.real("fill_ramp_scale", 1.0);

  const auto cache = [&read](const std::string &prefix) {
    CacheTuning out;
    // Every pessimistic value here describes a cache that cannot help: one
    // line, four bytes wide, answering at the latency of a full miss. A level
    // the config forgot therefore turns every access into a divergent miss and
    // the run reads slow, rather than inventing a plausible cache.
    out.line_bytes = read.integer(prefix + ".line_bytes", 4);
    out.sets = read.integer(prefix + ".sets", 1);
    out.ways = read.integer(prefix + ".ways", 1);
    out.hit_cycles = read.integer(prefix + ".hit_cycles", kUnnamedLatencyCycles);
    out.lines_per_cycle = read.real(prefix + ".lines_per_cycle", 1.0);
    return out;
  };
  tuning.l1_vector = cache("l1_vector");
  tuning.l1_scalar = cache("l1_scalar");
  tuning.l1_instruction = cache("l1_instruction");
  tuning.l2 = cache("l2");
  tuning.mall = cache("mall");

  tuning.dram_latency_cycles = read.integer("dram.latency_cycles", kUnnamedLatencyCycles);
  tuning.dram_bytes_per_cycle = read.real("dram.bytes_per_cycle", 1.0);
  tuning.mall_bytes_per_cycle = read.real("mall.bytes_per_cycle", 1.0);
  tuning.fabric_bytes_per_cycle = read.real("fabric.bytes_per_cycle", 1.0);
  tuning.memory_channels = read.integer("memory_channels", 1);
  tuning.fabric_request_bytes = read.integer("fabric_request_bytes", 64);
  tuning.dram_row_bytes = read.integer("dram.row_bytes", 1024);
  tuning.dram_row_miss_cycles = read.integer("dram.row_miss_cycles", 0);
  tuning.miss_status_registers_per_cu = read.integer("miss_status_registers_per_cu", 1);
  tuning.stall_overlap_wavefronts = read.integer("stall_overlap_wavefronts", 1);

  tuning.lds_banks = read.integer("lds.banks", 1);
  tuning.lds_bank_bytes = read.integer("lds.bank_bytes", 4);
  tuning.lds_latency_cycles = read.integer("lds.latency_cycles", kUnnamedLatencyCycles);
  tuning.lds_lanes_per_phase = read.integer("lds.lanes_per_phase", 1);

  tuning.dispatch_start_cycles = read.integer("dispatch.start_cycles", kUnnamedLatencyCycles);
  tuning.dispatch_end_cycles = read.integer("dispatch.end_cycles", kUnnamedLatencyCycles);
  tuning.launch_invalidate_cycles =
      read.integer("dispatch.invalidate_cycles", kUnnamedLatencyCycles);
  tuning.workgroups_per_cycle = read.real("dispatch.workgroups_per_cycle", 0.001);

  std::sort(tuning.resolved.begin(), tuning.resolved.end());
  std::sort(tuning.fell_back.begin(), tuning.fell_back.end());
  return tuning;
}

void Tuning::write_report(std::string &out) const {
  out += "timing plane: ";
  out += enabled ? "enabled" : "disabled";
  out += "\nshader clock: " + std::to_string(clock_mhz) + " MHz\n";
  out += "parameters in effect (" + std::to_string(resolved.size()) + "):\n";
  for (const std::string &entry : resolved)
    out += "  machine." + entry + "\n";
  if (fell_back.empty())
    return;
  // Named, not counted. A parameter the config forgot changes what the run
  // computed, and a reader who is going to trust a number needs to see which
  // numbers were nobody's decision.
  out += "parameters the config did not name, resolved to their slowest value (" +
         std::to_string(fell_back.size()) + "):\n";
  for (const std::string &entry : fell_back)
    out += "  machine." + entry + "\n";
}

} // namespace rocjitsu::timing
