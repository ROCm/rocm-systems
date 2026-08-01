// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/trampoline_builder.h"

#include "rocjitsu/code/patch/error_report.h"
#include "rocjitsu/code/patch/instruction_builder.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cstring>
#include <iterator>
#include <limits>
#include <numeric>
#include <queue>

namespace rocjitsu {

namespace {

[[nodiscard]] bool check_size_and_words(const TrampolinePlan &plan, std::string *err) {
  if (plan.arch == ROCJITSU_CODE_ARCH_INVALID) {
    report(err, "trampoline plan: arch was not set");
    return false;
  }
  if (plan.original_size != 4 && plan.original_size != 8) {
    report(err, "trampoline plan: original_size must be 4 or 8");
    return false;
  }
  const size_t expected_words = plan.original_size / sizeof(uint32_t);
  if (plan.original_words.size() != expected_words) {
    report(err, "trampoline plan: original_words count does not match original_size");
    return false;
  }
  return true;
}

// TODO: the following functions are very similar to those in LivenessAnalysis
// but they take a RegisterSet instead of an Instruction. These functions
// probably belong there and with some refactoring, we can probably reduce the
// duplicated code. Would like another opinion before making that call though.
// `any_sgpr_in_range` is similar to a test used by `find_free_*`
// `find_free_sgpr_pair` is similar to `find_free_sgpr_pair`
// `find_free_sgpr` is similar to `find_free_sgpr`
[[nodiscard]] bool any_sgpr_in_range(const RegisterSet &set, uint16_t base, uint16_t count) {
  for (uint16_t i = 0; i < count; ++i) {
    if (set.contains(RegisterRef{RegClass::SGPR, static_cast<uint16_t>(base + i), 1}))
      return true;
  }
  return false;
}

// First even-aligned SGPR pair with both lanes free of @p unavailable, within the
// conservative cross-family allocatable bound. nullopt if none.
[[nodiscard]] std::optional<uint16_t> find_free_sgpr_pair(const RegisterSet &unavailable) {
  for (uint16_t base = 0; static_cast<size_t>(base) + 1 < REGISTER_SET_ALLOCATABLE_SGPRS;
       base += 2) {
    if (!any_sgpr_in_range(unavailable, base, 2))
      return base;
  }
  return std::nullopt;
}

// First single SGPR free of @p unavailable, within the allocatable bound.
[[nodiscard]] std::optional<uint16_t> find_free_sgpr(const RegisterSet &unavailable) {
  for (uint16_t base = 0; base < REGISTER_SET_ALLOCATABLE_SGPRS; ++base) {
    if (!unavailable.contains(RegisterRef{RegClass::SGPR, base, 1}))
      return base;
  }
  return std::nullopt;
}

// Appends @p w to @p dst in host byte order. AMDGPU code objects are little-
// endian and rocjitsu only supports little-endian hosts (matches DBT's
// memcpy convention in binary_translator.cpp); if either invariant ever
// changes, this helper needs an explicit byte-swap.
void append_word(std::vector<uint8_t> &dst, uint32_t w) {
  uint8_t buf[sizeof(w)];
  std::memcpy(buf, &w, sizeof(w));
  dst.insert(dst.end(), buf, buf + sizeof(w));
}

} // namespace

std::optional<TrampolineBytes> TrampolineBuilder::build(const TrampolinePlan &plan,
                                                        std::string *error_out) {
  if (!check_size_and_words(plan, error_out))
    return std::nullopt;

  // Forward branch: from the anchor to the trampoline.
  const auto fwd = compute_sopp_branch_simm16(plan.anchor_offset, plan.trampoline_offset);
  if (!fwd) {
    report(error_out, "relocation trampoline forward branch exceeds s_branch simm16");
    return std::nullopt;
  }

  // Lay out trampoline body so we can compute the return branch offset. The
  // generic loops below handle any multi-item inline-asm shape; no reserve
  // hint because the per-item word counts aren't known up front and
  // vector::insert handles growth.
  std::vector<uint32_t> body;
  for (const InlineAsmItem &item : plan.before_items)
    body.insert(body.end(), item.words.begin(), item.words.end());
  if (plan.emit_original)
    body.insert(body.end(), plan.original_words.begin(), plan.original_words.end());
  for (const InlineAsmItem &item : plan.after_items)
    body.insert(body.end(), item.words.begin(), item.words.end());

  const uint64_t return_branch_pc = plan.trampoline_offset + body.size() * sizeof(uint32_t);
  const auto ret = compute_sopp_branch_simm16(return_branch_pc, plan.return_target);
  if (!ret) {
    report(error_out, "relocation trampoline return branch exceeds s_branch simm16");
    return std::nullopt;
  }

  TrampolineBytes out;
  out.patched_anchor_bytes.reserve(plan.original_size);
  append_word(out.patched_anchor_bytes, build_s_branch(*fwd, plan.arch));
  if (plan.original_size == 8)
    append_word(out.patched_anchor_bytes, build_s_nop(0, plan.arch));

  out.trampoline_words = std::move(body);
  out.trampoline_words.push_back(build_s_branch(*ret, plan.arch));
  return out;
}

bool TrampolineBuilder::plan_probe_call(TrampolinePlan &plan, ProbeCallingConvention cc,
                                        const RegisterSet &live_at_anchor,
                                        const RegisterSet &probe_body_clobbers,
                                        std::string *error_out) {
  // The link pair is whatever the probe's calling convention returns through,
  // so the call site and the probe body agree on one pair. An unknown
  // convention cannot be called.
  const std::optional<uint16_t> link_base = link_pair_for(cc);
  if (!link_base) {
    report(error_out, "probe-call resource planning: unknown probe calling convention; cannot "
                      "derive the return-link pair");
    return false;
  }
  const uint16_t kLinkPairBase = *link_base;

  // Reject if either lane of the link pair is live at the anchor; saving a live
  // link pair is deferred.
  if (any_sgpr_in_range(live_at_anchor, kLinkPairBase, 2)) {
    report(error_out, "probe-call resource planning: return-link pair s[30:31] is live at the "
                      "anchor; cannot yet save a live link pair");
    return false;
  }

  RegisterSet link_pair;
  link_pair.expand(RegisterRef{RegClass::SGPR, kLinkPairBase, 2});

  // NOTE: target/scc selection scans the conservative cross-ISA allocatable
  // bound (REGISTER_SET_ALLOCATABLE_SGPRS), not the patched kernel's actual
  // .sgpr_count. This is safe today only because the scan returns the lowest
  // dead registers, and we currently require the s30/31 regs. Handling this
  // is deferred.
  // Target-address pair: dead, even-aligned, and not the link pair. It is
  // read by s_swappc before the probe body runs, so it may overlap
  // probe_body_clobbers.
  const RegisterSet target_unavail = live_at_anchor | link_pair;
  const std::optional<uint16_t> target_pair = find_free_sgpr_pair(target_unavail);
  if (!target_pair) {
    report(error_out, "probe-call resource planning: no dead SGPR pair available for the probe "
                      "target address");
    return false;
  }

  RegisterSet target_pair_set;
  target_pair_set.expand(RegisterRef{RegClass::SGPR, *target_pair, 2});

  // SCC temp: only needed when we preserve SCC across the call. It lives across
  // the call (saved before materialization, restored after), so it must avoid
  // the live set, the link/target pairs, AND the probe body clobbers. When SCC
  // is not preserved we reserve nothing
  // TODO: allow for reuse of target_pair if unavailable
  std::optional<uint16_t> scc_temp;
  if (plan.preserve_scc) {
    const RegisterSet scc_unavail = target_unavail | target_pair_set | probe_body_clobbers;
    scc_temp = find_free_sgpr(scc_unavail);
    if (!scc_temp) {
      report(error_out, "probe-call resource planning: no dead SGPR available for the SCC "
                        "preservation temp");
      return false;
    }
  }

  // Word count is derived from the resource decisions, not a fixed envelope size.
  // Each add/addc uses the 32-bit literal form (instruction + literal word) so the
  // count is independent of the (layout-dependent) addend values.
  uint32_t before_words = 0;
  before_words += 1;     // s_getpc_b64
  before_words += 2 + 2; // s_add_u32 + literal, s_addc_u32 + literal
  before_words += 1;     // s_swappc_b64
  if (plan.preserve_scc)
    before_words += 2; // s_cselect_b32 (save) + s_cmp_lg_u32 (restore)

  plan.is_probe_call = true;
  plan.link_pair_base = kLinkPairBase;
  plan.target_pair_base = *target_pair;
  if (scc_temp)
    plan.scc_temp = *scc_temp;
  plan.before_word_count = before_words;

  plan.builder_clobbers = link_pair | target_pair_set;
  if (scc_temp)
    plan.builder_clobbers.expand(RegisterRef{RegClass::SGPR, *scc_temp, 1});
  return true;
}

std::optional<TrampolineBytes> TrampolineBuilder::emit_probe_call(const TrampolinePlan &plan,
                                                                  std::string *error_out) {
  if (!plan.is_probe_call) {
    report(error_out, "emit_probe_call: plan is not a probe call (run plan_probe_call first)");
    return std::nullopt;
  }

  const uint16_t link = plan.link_pair_base;
  const uint16_t target_lo = plan.target_pair_base;
  const uint16_t target_hi = static_cast<uint16_t>(plan.target_pair_base + 1);
  // Literal-constant scalar source code; the 32-bit literal follows the word.
  constexpr uint16_t kLiteralConstant = 0xFF;

  std::vector<uint32_t> env;

  // SCC save (prologue): capture SCC into the temp without disturbing it. The
  // matching restore is emitted after the call but still before the relocated
  // original.
  if (plan.preserve_scc)
    env.push_back(build_s_cselect_b32(plan.scc_temp, scalar_positive_inline_u32(1),
                                      scalar_positive_inline_u32(0), plan.arch));

  // Target-address materialization. s_getpc_b64 writes the runtime VA of the
  // *next* instruction (the s_add_u32 below) into the target pair; the
  // build-time delta to the probe body is then folded in via the 64-bit add
  // chain (s_add_u32 sets carry -> SCC, s_addc_u32 consumes it). Both sides are
  // .text-relative and share the load base, so the delta is a pure layout
  // distance. The adds always use the literal form so the word count is
  // independent of the (layout-dependent) delta value (see before_word_count).
  const size_t getpc_index = env.size();
  env.push_back(build_s_getpc_b64(target_lo, plan.arch));
  const uint64_t va_after_getpc =
      plan.trampoline_offset + static_cast<uint64_t>(getpc_index + 1) * sizeof(uint32_t);
  const uint64_t delta = static_cast<uint64_t>(static_cast<int64_t>(plan.probe_target_offset) -
                                               static_cast<int64_t>(va_after_getpc));
  env.push_back(build_s_add_u32(target_lo, target_lo, kLiteralConstant, plan.arch));
  env.push_back(static_cast<uint32_t>(delta & 0xFFFFFFFFu));
  env.push_back(build_s_addc_u32(target_hi, target_hi, kLiteralConstant, plan.arch));
  env.push_back(static_cast<uint32_t>(delta >> 32));

  // The call: writes the return PC into the cc-derived link pair, jumps to the
  // materialized target. The probe returns here via s_setpc_b64 of the same pair.
  env.push_back(build_s_swappc_b64(link, target_lo, plan.arch));

  // SCC restore (epilogue): set SCC from the saved temp before the relocated
  // original runs.
  if (plan.preserve_scc)
    env.push_back(build_s_cmp_lg_u32(plan.scc_temp, scalar_positive_inline_u32(0), plan.arch));

  // Plan/emit drift guard: the planner committed to this many envelope words and
  // the orchestrator sized the layout around it. A mismatch means the two
  // disagree about the envelope shape.
  if (env.size() != plan.before_word_count) {
    report(error_out, "emit_probe_call: synthesized envelope word count does not match the planned "
                      "before_word_count");
    return std::nullopt;
  }

  // Hand the synthesized envelope to build() for layout and branch math so the
  // SOPP range checks are shared with the inline path.
  TrampolinePlan emit_plan = plan;
  emit_plan.before_items.assign(1, InlineAsmItem{std::move(env)});
  emit_plan.after_items.clear();
  emit_plan.emit_original = true;
  return build(emit_plan, error_out);
}

class SoppRelayPlanningWorkContext {
public:
  SoppRelayPlanningWorkContext(size_t input_count, const SoppBranchRelayPlanningWorkLimits &limits,
                               SoppBranchRelayPlanningWorkTelemetry *telemetry)
      : meter_(limits.total.for_inputs(input_count),
               telemetry == nullptr ? nullptr : &telemetry->work_count,
               telemetry == nullptr ? nullptr : &telemetry->exhaustion_count) {}

  [[nodiscard]] bool consume(size_t amount = 1u) { return meter_.consume(amount); }
  [[nodiscard]] bool exhausted() const { return meter_.exhausted(); }

private:
  MeteredPlanningWork meter_;
};

static std::optional<SoppBranchRelayPlan> plan_monotonic_sopp_branch_relays(
    std::span<const uint64_t> source_offsets, std::span<const uint64_t> relay_offsets,
    std::span<const uint64_t> island_offsets, uint64_t maximum_hop_bytes,
    SoppRelayPlanningWorkContext &work, std::string *error_out) {
  enum class CoordinateKind : uint8_t {
    Source,
    Relay,
    Island,
  };
  const auto coordinate_kind_name = [](CoordinateKind kind) {
    switch (kind) {
    case CoordinateKind::Source:
      return "source";
    case CoordinateKind::Relay:
      return "relay";
    case CoordinateKind::Island:
      return "island";
    }
    return "unknown";
  };
  struct Coordinate {
    uint64_t offset = 0;
    CoordinateKind kind = CoordinateKind::Source;
    size_t input_index = 0;
  };
  std::vector<Coordinate> coordinates;
  coordinates.reserve(source_offsets.size() + relay_offsets.size() + island_offsets.size());
  const auto append_coordinates = [&](std::span<const uint64_t> offsets, CoordinateKind kind) {
    for (size_t i = 0; i < offsets.size(); ++i) {
      if (!work.consume())
        return false;
      coordinates.push_back({offsets[i], kind, i});
    }
    return true;
  };
  if (!append_coordinates(source_offsets, CoordinateKind::Source) ||
      !append_coordinates(relay_offsets, CoordinateKind::Relay) ||
      !append_coordinates(island_offsets, CoordinateKind::Island)) {
    report(error_out, "SOPP relay planner exhausted its work allowance while reading inputs");
    return std::nullopt;
  }
  const size_t coordinate_sort_work = saturated_multiply(
      coordinates.size(), std::max<size_t>(std::bit_width(coordinates.size()), 1u));
  if (!work.consume(coordinate_sort_work)) {
    report(error_out, "SOPP relay planner exhausted its work allowance while ordering inputs");
    return std::nullopt;
  }
  std::ranges::sort(coordinates, [](const Coordinate &lhs, const Coordinate &rhs) {
    if (lhs.offset != rhs.offset)
      return lhs.offset < rhs.offset;
    if (lhs.kind != rhs.kind)
      return lhs.kind < rhs.kind;
    return lhs.input_index < rhs.input_index;
  });
  for (size_t i = 0; i < coordinates.size(); ++i) {
    if (!work.consume()) {
      report(error_out, "SOPP relay planner exhausted its work allowance while validating inputs");
      return std::nullopt;
    }
    if (coordinates[i].offset % sizeof(uint32_t) != 0) {
      report(error_out, "SOPP relay planner: coordinates must be dword aligned");
      return std::nullopt;
    }
    if (i != 0 && coordinates[i - 1].offset == coordinates[i].offset) {
      if (error_out != nullptr) {
        const Coordinate &first = coordinates[i - 1];
        const Coordinate &second = coordinates[i];
        *error_out = "SOPP relay planner: coordinates must be globally unique; offset=" +
                     std::to_string(first.offset) + " first=" + coordinate_kind_name(first.kind) +
                     "[" + std::to_string(first.input_index) +
                     "] second=" + coordinate_kind_name(second.kind) + "[" +
                     std::to_string(second.input_index) + "]";
      }
      return std::nullopt;
    }
  }

  // In this one-dimensional forward interval graph, earliest-endpoint-first
  // routing is maximum-cardinality. At a relay or island, suppose a solution
  // serves a route B while route A has an earlier endpoint. A can take B's
  // suffix, and B can take A's suffix (every point reachable from A is also
  // reachable from the later B), so exchanging them never reduces the number
  // of completed routes. Repeating that exchange proves the greedy order.
  //
  // Keep the residual-flow implementation below as a small-instance oracle
  // with precise min-cut diagnostics. Large generated code objects need this
  // O(N log N) path: even an interval-compressed exact-flow graph can require
  // many expensive blocking-flow phases for tens of thousands of sites.
  const auto plan_greedily = [&]() -> std::optional<SoppBranchRelayPlan> {
    const size_t heap_levels = std::max<size_t>(std::bit_width(coordinates.size()), 1u);
    const auto charge_heap_operation = [&]() { return work.consume(heap_levels); };
    struct GreedyRouteState {
      std::vector<uint64_t> relay_offsets;
      std::optional<uint64_t> island_offset;
    };
    using ActiveRoute = std::pair<uint64_t, size_t>; // endpoint, source input index.
    std::priority_queue<ActiveRoute, std::vector<ActiveRoute>, std::greater<>> active;
    std::vector<GreedyRouteState> route_states(source_offsets.size());
    std::vector<std::pair<uint64_t, uint64_t>> cut_hops;
    for (const Coordinate &coordinate : coordinates) {
      if (!work.consume()) {
        report(error_out, "SOPP relay planner exhausted its work allowance in greedy routing");
        return std::nullopt;
      }
      if (coordinate.kind == CoordinateKind::Source) {
        if (!charge_heap_operation()) {
          report(error_out, "SOPP relay planner exhausted its work allowance in greedy routing");
          return std::nullopt;
        }
        active.emplace(coordinate.offset, coordinate.input_index);
        continue;
      }
      while (!active.empty() && coordinate.offset - active.top().first > maximum_hop_bytes) {
        if (!charge_heap_operation()) {
          report(error_out, "SOPP relay planner exhausted its work allowance in greedy routing");
          return std::nullopt;
        }
        cut_hops.emplace_back(active.top().first, coordinate.offset);
        active.pop();
      }
      if (active.empty())
        continue;
      if (!charge_heap_operation()) {
        report(error_out, "SOPP relay planner exhausted its work allowance in greedy routing");
        return std::nullopt;
      }
      const auto [endpoint, source_index] = active.top();
      (void)endpoint;
      active.pop();
      if (coordinate.kind == CoordinateKind::Relay) {
        route_states[source_index].relay_offsets.push_back(coordinate.offset);
        if (!charge_heap_operation()) {
          report(error_out, "SOPP relay planner exhausted its work allowance in greedy routing");
          return std::nullopt;
        }
        active.emplace(coordinate.offset, source_index);
      } else {
        route_states[source_index].island_offset = coordinate.offset;
      }
    }

    SoppBranchRelayPlan plan;
    plan.min_cut_hops = std::move(cut_hops);
    plan.routes.reserve(std::min(source_offsets.size(), island_offsets.size()));
    for (size_t source_index = 0; source_index < route_states.size(); ++source_index) {
      if (!work.consume()) {
        report(error_out, "SOPP relay planner exhausted its work allowance in greedy routing");
        return std::nullopt;
      }
      GreedyRouteState &state = route_states[source_index];
      if (!state.island_offset) {
        plan.rejected_source_indices.push_back(source_index);
        continue;
      }
      plan.routes.push_back({.source_index = source_index,
                             .island_offset = *state.island_offset,
                             .relay_offsets = std::move(state.relay_offsets)});
    }
    return plan;
  };

  constexpr size_t kGreedyRelayPlannerThreshold = 4096u;
  if (coordinates.size() > kGreedyRelayPlannerThreshold)
    return plan_greedily();

  struct Edge {
    size_t to = 0;
    size_t reverse = 0;
    uint32_t capacity = 0;
    uint32_t initial_capacity = 0;
    bool original = false;
  };
  // A relay is split into in/out nodes so its instruction word has capacity
  // one. Sources and islands already have capacity-one boundary edges.
  const size_t super_source = 0;
  const size_t source_base = 1;
  const size_t relay_in_base = source_base + source_offsets.size();
  const size_t relay_out_base = relay_in_base + relay_offsets.size();
  const size_t island_base = relay_out_base + relay_offsets.size();
  const size_t super_sink = island_base + island_offsets.size();
  if (!work.consume(super_sink + 1u)) {
    report(error_out, "SOPP relay planner exhausted its work allowance building exact graph");
    return std::nullopt;
  }
  std::vector<std::vector<Edge>> graph(super_sink + 1);
  const auto add_edge = [&](size_t from, size_t to, uint32_t capacity = 1u) {
    if (!work.consume(2u))
      return false;
    const size_t forward_index = graph[from].size();
    const size_t reverse_index = graph[to].size();
    graph[from].push_back({to, reverse_index, capacity, capacity, true});
    graph[to].push_back({from, forward_index, 0, 0, false});
    return true;
  };

  const auto order_work = [](size_t count) {
    return saturated_add(count,
                         saturated_multiply(count, std::max<size_t>(std::bit_width(count), 1u)));
  };
  size_t exact_order_work = order_work(source_offsets.size());
  accumulate_saturated(exact_order_work, order_work(relay_offsets.size()));
  accumulate_saturated(exact_order_work, order_work(island_offsets.size()));
  if (!work.consume(exact_order_work)) {
    report(error_out, "SOPP relay planner exhausted its work allowance ordering exact inputs");
    return std::nullopt;
  }

  std::vector<size_t> source_order(source_offsets.size());
  std::iota(source_order.begin(), source_order.end(), 0);
  std::ranges::sort(source_order, [&](size_t lhs, size_t rhs) {
    if (source_offsets[lhs] != source_offsets[rhs])
      return source_offsets[lhs] < source_offsets[rhs];
    return lhs < rhs;
  });
  std::vector<size_t> relay_order(relay_offsets.size());
  std::iota(relay_order.begin(), relay_order.end(), 0);
  std::ranges::sort(relay_order, [&](size_t lhs, size_t rhs) {
    if (relay_offsets[lhs] != relay_offsets[rhs])
      return relay_offsets[lhs] < relay_offsets[rhs];
    return lhs < rhs;
  });
  std::vector<size_t> island_order(island_offsets.size());
  std::iota(island_order.begin(), island_order.end(), 0);
  std::ranges::sort(island_order, [&](size_t lhs, size_t rhs) {
    if (island_offsets[lhs] != island_offsets[rhs])
      return island_offsets[lhs] < island_offsets[rhs];
    return lhs < rhs;
  });

  for (size_t source : source_order) {
    if (!add_edge(super_source, source_base + source)) {
      report(error_out, "SOPP relay planner exhausted its work allowance building exact graph");
      return std::nullopt;
    }
  }
  for (size_t relay : relay_order) {
    if (!add_edge(relay_in_base + relay, relay_out_base + relay)) {
      report(error_out, "SOPP relay planner exhausted its work allowance building exact graph");
      return std::nullopt;
    }
  }
  for (size_t island : island_order) {
    if (!add_edge(island_base + island, super_sink)) {
      report(error_out, "SOPP relay planner exhausted its work allowance building exact graph");
      return std::nullopt;
    }
  }

  struct HopTarget {
    uint64_t offset = 0;
    size_t node = 0;
  };
  std::vector<HopTarget> hop_targets;
  hop_targets.reserve(relay_offsets.size() + island_offsets.size());
  for (size_t relay : relay_order) {
    if (!work.consume()) {
      report(error_out, "SOPP relay planner exhausted its work allowance building exact graph");
      return std::nullopt;
    }
    hop_targets.push_back({relay_offsets[relay], relay_in_base + relay});
  }
  for (size_t island : island_order) {
    if (!work.consume()) {
      report(error_out, "SOPP relay planner exhausted its work allowance building exact graph");
      return std::nullopt;
    }
    hop_targets.push_back({island_offsets[island], island_base + island});
  }
  if (!work.consume(order_work(hop_targets.size()))) {
    report(error_out, "SOPP relay planner exhausted its work allowance ordering exact targets");
    return std::nullopt;
  }
  std::ranges::sort(hop_targets, [](const HopTarget &lhs, const HopTarget &rhs) {
    return std::tie(lhs.offset, lhs.node) < std::tie(rhs.offset, rhs.node);
  });

  // Every origin reaches one contiguous interval of later coordinates. A
  // segment-tree range graph represents all such edges in O(N log N) space,
  // rather than materializing the quadratic source/relay cross product seen
  // in large generated code objects.
  struct TargetTreeNode {
    size_t graph_node = 0;
    size_t begin = 0;
    size_t end = 0;
    size_t left = std::numeric_limits<size_t>::max();
    size_t right = std::numeric_limits<size_t>::max();
  };
  std::vector<TargetTreeNode> target_tree;
  const auto build_target_tree = [&](auto &&self, size_t begin,
                                     size_t end) -> std::optional<size_t> {
    if (!work.consume())
      return std::nullopt;
    const size_t tree_index = target_tree.size();
    const size_t graph_node = graph.size();
    graph.emplace_back();
    target_tree.push_back({.graph_node = graph_node, .begin = begin, .end = end});
    if (end - begin == 1u) {
      if (!add_edge(graph_node, hop_targets[begin].node,
                    static_cast<uint32_t>(source_offsets.size())))
        return std::nullopt;
      return tree_index;
    }
    const size_t middle = begin + (end - begin) / 2u;
    const auto left = self(self, begin, middle);
    const auto right = self(self, middle, end);
    if (!left || !right)
      return std::nullopt;
    target_tree[tree_index].left = *left;
    target_tree[tree_index].right = *right;
    if (!add_edge(graph_node, target_tree[*left].graph_node,
                  static_cast<uint32_t>(source_offsets.size())) ||
        !add_edge(graph_node, target_tree[*right].graph_node,
                  static_cast<uint32_t>(source_offsets.size())))
      return std::nullopt;
    return tree_index;
  };
  const std::optional<size_t> target_tree_root =
      hop_targets.empty() ? std::nullopt
                          : build_target_tree(build_target_tree, 0, hop_targets.size());
  if (!hop_targets.empty() && !target_tree_root) {
    report(error_out, "SOPP relay planner exhausted its work allowance building exact graph");
    return std::nullopt;
  }
  const auto add_target_interval = [&](auto &&self, size_t origin_node, size_t query_begin,
                                       size_t query_end, size_t tree_index) -> bool {
    if (!work.consume())
      return false;
    const TargetTreeNode &tree_node = target_tree[tree_index];
    if (query_end <= tree_node.begin || tree_node.end <= query_begin)
      return true;
    if (query_begin <= tree_node.begin && tree_node.end <= query_end) {
      return add_edge(origin_node, tree_node.graph_node,
                      static_cast<uint32_t>(source_offsets.size()));
    }
    return self(self, origin_node, query_begin, query_end, tree_node.left) &&
           self(self, origin_node, query_begin, query_end, tree_node.right);
  };
  const auto connect_origin = [&](size_t origin_node, uint64_t offset) {
    if (!target_tree_root)
      return true;
    const size_t lookup_work =
        saturated_multiply(2u, std::max<size_t>(std::bit_width(hop_targets.size()), 1u));
    if (!work.consume(lookup_work))
      return false;
    const uint64_t maximum_target =
        offset > std::numeric_limits<uint64_t>::max() - maximum_hop_bytes
            ? std::numeric_limits<uint64_t>::max()
            : offset + maximum_hop_bytes;
    const size_t begin =
        static_cast<size_t>(std::ranges::upper_bound(hop_targets, offset, {}, &HopTarget::offset) -
                            hop_targets.begin());
    const size_t end = static_cast<size_t>(
        std::ranges::upper_bound(hop_targets, maximum_target, {}, &HopTarget::offset) -
        hop_targets.begin());
    if (begin != end)
      return add_target_interval(add_target_interval, origin_node, begin, end, *target_tree_root);
    return true;
  };
  for (size_t source : source_order) {
    if (!connect_origin(source_base + source, source_offsets[source])) {
      report(error_out, "SOPP relay planner exhausted its work allowance building exact graph");
      return std::nullopt;
    }
  }
  for (size_t relay : relay_order) {
    if (!connect_origin(relay_out_base + relay, relay_offsets[relay])) {
      report(error_out, "SOPP relay planner exhausted its work allowance building exact graph");
      return std::nullopt;
    }
  }

  // Dinic blocking flow preserves the exact maximum-cardinality result while
  // augmenting many routes per residual traversal.
  bool exact_work_exhausted = false;
  for (;;) {
    if (!work.consume(graph.size())) {
      report(error_out, "SOPP relay planner exhausted its work allowance in exact routing");
      return std::nullopt;
    }
    std::vector<int32_t> level(graph.size(), -1);
    std::queue<size_t> queue;
    level[super_source] = 0;
    queue.push(super_source);
    while (!queue.empty()) {
      if (!work.consume()) {
        exact_work_exhausted = true;
        break;
      }
      const size_t from = queue.front();
      queue.pop();
      for (const Edge &edge : graph[from]) {
        if (!work.consume()) {
          exact_work_exhausted = true;
          break;
        }
        if (edge.capacity == 0 || level[edge.to] != -1)
          continue;
        level[edge.to] = level[from] + 1;
        queue.push(edge.to);
      }
      if (exact_work_exhausted)
        break;
    }
    if (exact_work_exhausted) {
      report(error_out, "SOPP relay planner exhausted its work allowance in exact routing");
      return std::nullopt;
    }
    if (level[super_sink] == -1)
      break;
    std::vector<size_t> next_edge(graph.size(), 0u);
    const auto push_flow = [&](auto &&self, size_t from, uint32_t limit) -> uint32_t {
      if (from == super_sink)
        return limit;
      for (size_t &edge_index = next_edge[from]; edge_index < graph[from].size(); ++edge_index) {
        if (!work.consume()) {
          exact_work_exhausted = true;
          return 0u;
        }
        Edge &edge = graph[from][edge_index];
        if (edge.capacity == 0 || level[edge.to] != level[from] + 1)
          continue;
        const uint32_t pushed = self(self, edge.to, std::min(limit, edge.capacity));
        if (pushed == 0)
          continue;
        edge.capacity -= pushed;
        graph[edge.to][edge.reverse].capacity += pushed;
        return pushed;
      }
      return 0u;
    };
    while (push_flow(push_flow, super_source, static_cast<uint32_t>(source_offsets.size())) != 0u) {
    }
    if (exact_work_exhausted) {
      report(error_out, "SOPP relay planner exhausted its work allowance in exact routing");
      return std::nullopt;
    }
  }
  const auto take_used_original_edge_to = [&](size_t from) -> std::optional<size_t> {
    for (Edge &edge : graph[from]) {
      if (!work.consume()) {
        exact_work_exhausted = true;
        return std::nullopt;
      }
      if (!edge.original || edge.initial_capacity == edge.capacity)
        continue;
      ++edge.capacity;
      Edge &reverse = graph[edge.to][edge.reverse];
      if (reverse.capacity == 0) {
        report(error_out, "SOPP relay planner: inconsistent residual flow");
        return std::nullopt;
      }
      --reverse.capacity;
      return edge.to;
    }
    return std::nullopt;
  };
  SoppBranchRelayPlan plan;
  {
    if (!work.consume(graph.size())) {
      report(error_out, "SOPP relay planner exhausted its work allowance finding exact cut");
      return std::nullopt;
    }
    std::vector<bool> reachable(graph.size(), false);
    std::queue<size_t> queue;
    reachable[super_source] = true;
    queue.push(super_source);
    while (!queue.empty()) {
      if (!work.consume()) {
        report(error_out, "SOPP relay planner exhausted its work allowance finding exact cut");
        return std::nullopt;
      }
      const size_t from = queue.front();
      queue.pop();
      for (const Edge &edge : graph[from]) {
        if (!work.consume()) {
          report(error_out, "SOPP relay planner exhausted its work allowance finding exact cut");
          return std::nullopt;
        }
        if (edge.capacity == 0 || reachable[edge.to])
          continue;
        reachable[edge.to] = true;
        queue.push(edge.to);
      }
    }
    const auto node_offset = [&](size_t node) -> std::optional<uint64_t> {
      if (node >= source_base && node < relay_in_base)
        return source_offsets[node - source_base];
      if (node >= relay_in_base && node < relay_out_base)
        return relay_offsets[node - relay_in_base];
      if (node >= relay_out_base && node < island_base)
        return relay_offsets[node - relay_out_base];
      if (node >= island_base && node < super_sink)
        return island_offsets[node - island_base];
      return std::nullopt;
    };
    for (size_t relay = 0; relay < relay_offsets.size(); ++relay) {
      if (!work.consume()) {
        report(error_out, "SOPP relay planner exhausted its work allowance finding exact cut");
        return std::nullopt;
      }
      if (reachable[relay_in_base + relay] && !reachable[relay_out_base + relay])
        plan.min_cut_relay_offsets.push_back(relay_offsets[relay]);
    }
    if (!work.consume(order_work(plan.min_cut_relay_offsets.size()))) {
      report(error_out, "SOPP relay planner exhausted its work allowance ordering exact cut");
      return std::nullopt;
    }
    std::ranges::sort(plan.min_cut_relay_offsets);
    for (size_t from = 0; from < graph.size(); ++from) {
      if (!work.consume()) {
        report(error_out, "SOPP relay planner exhausted its work allowance finding exact cut");
        return std::nullopt;
      }
      if (!reachable[from])
        continue;
      for (const Edge &edge : graph[from]) {
        if (!work.consume()) {
          report(error_out, "SOPP relay planner exhausted its work allowance finding exact cut");
          return std::nullopt;
        }
        if (!edge.original || edge.capacity != 0 || reachable[edge.to])
          continue;
        const auto from_offset = node_offset(from);
        const auto to_offset = node_offset(edge.to);
        if (from_offset && to_offset && *from_offset != *to_offset)
          plan.min_cut_hops.emplace_back(*from_offset, *to_offset);
      }
    }
    if (!work.consume(order_work(plan.min_cut_hops.size()))) {
      report(error_out, "SOPP relay planner exhausted its work allowance ordering exact cut");
      return std::nullopt;
    }
    std::ranges::sort(plan.min_cut_hops);
    plan.min_cut_hops.erase(std::ranges::unique(plan.min_cut_hops).begin(),
                            plan.min_cut_hops.end());
  }
  plan.routes.reserve(std::min(source_offsets.size(), island_offsets.size()));
  // Decompose shared range-tree flow in coordinate order to retain the
  // planner's documented deterministic tie-breaking, then restore input order
  // in the returned vectors.
  for (size_t source_index : source_order) {
    if (!work.consume()) {
      report(error_out, "SOPP relay planner exhausted its work allowance reconstructing routes");
      return std::nullopt;
    }
    bool admitted = false;
    for (const Edge &edge : graph[super_source]) {
      if (!work.consume()) {
        report(error_out, "SOPP relay planner exhausted its work allowance reconstructing routes");
        return std::nullopt;
      }
      if (edge.to != source_base + source_index || edge.capacity != 0)
        continue;
      SoppBranchRelayRoute route;
      route.source_index = source_index;
      size_t node = source_base + source_index;
      for (;;) {
        const auto next = take_used_original_edge_to(node);
        if (!next) {
          if (exact_work_exhausted) {
            report(error_out,
                   "SOPP relay planner exhausted its work allowance reconstructing routes");
            return std::nullopt;
          }
          report(error_out, "SOPP relay planner: internal flow decomposition failure");
          return std::nullopt;
        }
        node = *next;
        if (node >= relay_in_base && node < relay_out_base) {
          const size_t relay = node - relay_in_base;
          route.relay_offsets.push_back(relay_offsets[relay]);
          const auto relay_out = take_used_original_edge_to(node);
          if (exact_work_exhausted) {
            report(error_out,
                   "SOPP relay planner exhausted its work allowance reconstructing routes");
            return std::nullopt;
          }
          if (!relay_out || *relay_out != relay_out_base + relay) {
            report(error_out, "SOPP relay planner: relay capacity flow is inconsistent");
            return std::nullopt;
          }
          node = *relay_out;
          continue;
        }
        if (node >= island_base && node < super_sink) {
          route.island_offset = island_offsets[node - island_base];
          break;
        }
        if (node <= super_sink) {
          report(error_out, "SOPP relay planner: internal flow path has an invalid node");
          return std::nullopt;
        }
      }
      plan.routes.push_back(std::move(route));
      admitted = true;
      break;
    }
    if (!admitted)
      plan.rejected_source_indices.push_back(source_index);
  }
  if (!work.consume(saturated_add(order_work(plan.routes.size()),
                                  order_work(plan.rejected_source_indices.size())))) {
    report(error_out, "SOPP relay planner exhausted its work allowance ordering exact result");
    return std::nullopt;
  }
  std::ranges::sort(plan.routes, {}, &SoppBranchRelayRoute::source_index);
  std::ranges::sort(plan.rejected_source_indices);
  return plan;
}

std::optional<SoppBranchRelayPlan>
plan_forward_sopp_branch_relays(std::span<const uint64_t> source_offsets,
                                std::span<const uint64_t> relay_offsets,
                                std::span<const uint64_t> island_offsets, std::string *error_out,
                                SoppBranchRelayPlanningWorkTelemetry *work_telemetry,
                                const SoppBranchRelayPlanningWorkLimits &work_limits) {
  const size_t input_count = saturated_add(
      saturated_add(source_offsets.size(), relay_offsets.size()), island_offsets.size());
  SoppRelayPlanningWorkContext work(input_count, work_limits, work_telemetry);
  return plan_monotonic_sopp_branch_relays(source_offsets, relay_offsets, island_offsets,
                                           kSoppBranchMaximumForwardReachBytes, work, error_out);
}

std::optional<SoppBranchRelayPlan>
plan_backward_sopp_branch_relays(std::span<const uint64_t> source_offsets,
                                 std::span<const uint64_t> relay_offsets,
                                 std::span<const uint64_t> island_offsets, std::string *error_out,
                                 SoppBranchRelayPlanningWorkTelemetry *work_telemetry,
                                 const SoppBranchRelayPlanningWorkLimits &work_limits) {
  const size_t input_count = saturated_add(
      saturated_add(source_offsets.size(), relay_offsets.size()), island_offsets.size());
  SoppRelayPlanningWorkContext work(input_count, work_limits, work_telemetry);
  if (!work.consume(saturated_multiply(3u, input_count))) {
    report(error_out, "SOPP backward relay planner exhausted its work allowance while mirroring");
    return std::nullopt;
  }
  uint64_t mirror = 0;
  const auto include_offsets = [&](std::span<const uint64_t> offsets) {
    if (!offsets.empty())
      mirror = std::max(mirror, *std::ranges::max_element(offsets));
  };
  include_offsets(source_offsets);
  include_offsets(relay_offsets);
  include_offsets(island_offsets);
  const auto mirror_offsets = [&](std::span<const uint64_t> offsets) {
    std::vector<uint64_t> mirrored;
    mirrored.reserve(offsets.size());
    for (uint64_t offset : offsets)
      mirrored.push_back(mirror - offset);
    return mirrored;
  };
  const std::vector<uint64_t> mirrored_sources = mirror_offsets(source_offsets);
  const std::vector<uint64_t> mirrored_relays = mirror_offsets(relay_offsets);
  const std::vector<uint64_t> mirrored_islands = mirror_offsets(island_offsets);
  auto plan =
      plan_monotonic_sopp_branch_relays(mirrored_sources, mirrored_relays, mirrored_islands,
                                        kSoppBranchMaximumBackwardReachBytes, work, error_out);
  if (!plan)
    return std::nullopt;
  size_t output_work =
      saturated_add(plan->routes.size(),
                    saturated_add(plan->min_cut_relay_offsets.size(), plan->min_cut_hops.size()));
  for (const SoppBranchRelayRoute &route : plan->routes) {
    accumulate_saturated(output_work, saturated_multiply(route.relay_offsets.size(), 3u));
  }
  const size_t cut_count =
      saturated_add(plan->min_cut_relay_offsets.size(), plan->min_cut_hops.size());
  accumulate_saturated(
      output_work, saturated_multiply(cut_count, std::max<size_t>(std::bit_width(cut_count), 1u)));
  if (!work.consume(output_work)) {
    report(error_out,
           "SOPP backward relay planner exhausted its work allowance while restoring coordinates");
    return std::nullopt;
  }
  for (SoppBranchRelayRoute &route : plan->routes) {
    route.island_offset = mirror - route.island_offset;
    for (uint64_t &relay_offset : route.relay_offsets)
      relay_offset = mirror - relay_offset;
    uint64_t from = source_offsets[route.source_index];
    for (uint64_t relay_offset : route.relay_offsets) {
      if (!compute_sopp_branch_simm16(from, relay_offset)) {
        report(error_out, "SOPP backward relay planner produced an unreachable relay hop");
        return std::nullopt;
      }
      from = relay_offset;
    }
    if (!compute_sopp_branch_simm16(from, route.island_offset)) {
      report(error_out, "SOPP backward relay planner produced an unreachable island hop");
      return std::nullopt;
    }
  }
  for (uint64_t &offset : plan->min_cut_relay_offsets)
    offset = mirror - offset;
  for (auto &[from, to] : plan->min_cut_hops) {
    from = mirror - from;
    to = mirror - to;
  }
  std::ranges::sort(plan->min_cut_relay_offsets);
  std::ranges::sort(plan->min_cut_hops);
  return plan;
}

DbiPatchPlacementPlanner::DbiPatchPlacementPlanner(rj_code_arch_t arch, uint64_t original_text_size)
    : arch_(arch), original_text_size_(original_text_size), appended_cursor_(original_text_size) {}

bool DbiPatchPlacementPlanner::range_is_free(uint64_t begin, uint64_t end) const {
  if (begin >= end)
    return false;
  return !overlaps_reserved_range(begin, end);
}

bool DbiPatchPlacementPlanner::overlaps_reserved_range(uint64_t begin, uint64_t end) const {
  if (begin >= end)
    return true;
  const auto next =
      std::ranges::lower_bound(occupied_ranges_, begin, {}, &std::pair<uint64_t, uint64_t>::first);
  if (next != occupied_ranges_.end() && next->first < end)
    return true;
  return next != occupied_ranges_.begin() && std::prev(next)->second > begin;
}

bool DbiPatchPlacementPlanner::highest_appended_reservation_within_cursor() const {
  return occupied_ranges_.empty() || occupied_ranges_.back().first < original_text_size_ ||
         occupied_ranges_.back().second <= appended_cursor_;
}

void DbiPatchPlacementPlanner::insert_validated_range(uint64_t begin, uint64_t end) {
  assert(!overlaps_reserved_range(begin, end) &&
         "validated reservations must remain nonempty, sorted, and disjoint");
  const auto insertion =
      std::ranges::lower_bound(occupied_ranges_, begin, {}, &std::pair<uint64_t, uint64_t>::first);
  occupied_ranges_.emplace(insertion, begin, end);
}

bool DbiPatchPlacementPlanner::reserve_ranges(
    std::span<const std::pair<uint64_t, uint64_t>> ranges) {
  for (size_t range = 0u; range < ranges.size(); ++range) {
    const auto [begin, end] = ranges[range];
    if (!range_is_free(begin, end))
      return false;
    for (size_t prior = 0u; prior < range; ++prior) {
      const auto [prior_begin, prior_end] = ranges[prior];
      if (begin < prior_end && prior_begin < end)
        return false;
    }
  }
  for (const auto &[begin, end] : ranges)
    insert_validated_range(begin, end);
  return true;
}

bool DbiPatchPlacementPlanner::reserve_range(uint64_t begin, uint64_t end) {
  const std::array ranges = {std::pair{begin, end}};
  return reserve_ranges(ranges);
}

bool DbiPatchPlacementPlanner::reserve_existing_range(uint64_t begin, uint64_t size,
                                                      std::string *error_out) {
  if (!can_reserve_existing_range(begin, size)) {
    report(error_out,
           "DBI patch placement: existing range is empty, out of bounds, or overlapping");
    return false;
  }
  insert_validated_range(begin, begin + size);
  return true;
}

bool DbiPatchPlacementPlanner::can_reserve_existing_range(uint64_t begin, uint64_t size) const {
  return size != 0u && begin <= original_text_size_ && size <= original_text_size_ - begin &&
         range_is_free(begin, begin + size);
}

bool DbiPatchPlacementPlanner::reserve_appended_prefix(uint64_t size, std::string *error_out) {
  assert(highest_appended_reservation_within_cursor());
  if (size == 0)
    return true;
  if (size > std::numeric_limits<uint64_t>::max() - appended_cursor_) {
    report(error_out, "DBI patch placement: appended prefix overflows text coordinates");
    return false;
  }
  const uint64_t end = appended_cursor_ + size;
  if (!reserve_range(appended_cursor_, end)) {
    report(error_out, "DBI patch placement: appended prefix overlaps an existing reservation");
    return false;
  }
  appended_cursor_ = end;
  assert(highest_appended_reservation_within_cursor());
  return true;
}

std::optional<DbiPatchPlacement>
DbiPatchPlacementPlanner::plan(const DbiPatchPlacementRequest &request, std::string *error_out) {
  assert(highest_appended_reservation_within_cursor());
  const auto checked_end = [](uint64_t begin, uint64_t size) -> std::optional<uint64_t> {
    if (size > std::numeric_limits<uint64_t>::max() - begin)
      return std::nullopt;
    return begin + size;
  };
  if (arch_ == ROCJITSU_CODE_ARCH_INVALID) {
    report(error_out, "DBI patch placement: architecture was not set");
    return std::nullopt;
  }
  if (request.original_size < sizeof(uint32_t) || request.original_size % sizeof(uint32_t) != 0 ||
      request.body_size == 0 || request.body_size % sizeof(uint32_t) != 0) {
    report(error_out, "DBI patch placement: sizes must be nonzero instruction multiples");
    return std::nullopt;
  }
  const auto anchor_end = checked_end(request.anchor_offset, request.original_size);
  if (!anchor_end || *anchor_end > original_text_size_) {
    report(error_out, "DBI patch placement: anchor exceeds original .text");
    return std::nullopt;
  }

  if (request.body_size <= request.inline_capacity) {
    const auto body_end = checked_end(request.anchor_offset, request.body_size);
    if (body_end && *body_end <= original_text_size_ &&
        range_is_free(request.anchor_offset, *body_end)) {
      if (!reserve_range(request.anchor_offset, *body_end)) {
        report(error_out, "DBI patch placement: inline reservation conflicted");
        return std::nullopt;
      }
      return DbiPatchPlacement{
          .kind = DbiPatchPlacementKind::Inline,
          .anchor_offset = request.anchor_offset,
          .original_size = request.original_size,
          .body_offset = request.anchor_offset,
          .body_size = request.body_size,
          .return_branch_offset = 0,
          .return_target = *anchor_end,
      };
    }
  }

  const auto try_trampoline = [&](DbiPatchPlacementKind kind, uint64_t body_offset,
                                  uint64_t capacity) -> std::optional<DbiPatchPlacement> {
    const auto body_end = checked_end(body_offset, request.body_size);
    const auto reservation_end = body_end ? checked_end(*body_end, sizeof(uint32_t)) : std::nullopt;
    const bool local_body_overlaps_anchor =
        kind == DbiPatchPlacementKind::LocalCave && reservation_end &&
        request.anchor_offset < *reservation_end && body_offset < *anchor_end;
    if (!body_end || !reservation_end || request.body_size + sizeof(uint32_t) > capacity ||
        !range_is_free(request.anchor_offset, *anchor_end) ||
        !range_is_free(body_offset, *reservation_end) ||
        (kind == DbiPatchPlacementKind::LocalCave &&
         (local_body_overlaps_anchor || *reservation_end > original_text_size_)) ||
        !compute_sopp_branch_simm16(request.anchor_offset, body_offset) ||
        !compute_sopp_branch_simm16(*body_end, *anchor_end)) {
      return std::nullopt;
    }
    return DbiPatchPlacement{
        .kind = kind,
        .anchor_offset = request.anchor_offset,
        .original_size = request.original_size,
        .body_offset = body_offset,
        .body_size = request.body_size,
        .return_branch_offset = *body_end,
        .return_target = *anchor_end,
    };
  };

  if (request.local_cave) {
    if (auto placement = try_trampoline(DbiPatchPlacementKind::LocalCave,
                                        request.local_cave->offset, request.local_cave->capacity)) {
      const std::array reservations = {
          std::pair{request.anchor_offset, *anchor_end},
          std::pair{placement->body_offset, placement->return_branch_offset + sizeof(uint32_t)},
      };
      if (!reserve_ranges(reservations)) {
        // try_trampoline preflighted both ranges. A conflict here is an
        // internal planner-state inconsistency, not an unavailable local tier;
        // do not hide it by falling through to appended placement.
        report(error_out, "DBI patch placement: local-cave reservations conflicted");
        return std::nullopt;
      }
      return placement;
    }
  }

  if (request.allow_appended_cave) {
    if (auto placement = try_trampoline(DbiPatchPlacementKind::AppendedCave, appended_cursor_,
                                        std::numeric_limits<uint64_t>::max())) {
      const std::array reservations = {
          std::pair{request.anchor_offset, *anchor_end},
          std::pair{placement->body_offset, placement->return_branch_offset + sizeof(uint32_t)},
      };
      if (!reserve_ranges(reservations)) {
        report(error_out, "DBI patch placement: appended-cave reservations conflicted");
        return std::nullopt;
      }
      appended_cursor_ = placement->return_branch_offset + sizeof(uint32_t);
      assert(highest_appended_reservation_within_cursor());
      return placement;
    }
  }

  report(error_out,
         "DBI patch placement: no nonoverlapping reachable inline, local, or appended placement");
  return std::nullopt;
}

std::optional<DbiPatchPlacement>
DbiPatchPlacementPlanner::plan_indirect_appended(uint64_t anchor_offset, uint32_t original_size,
                                                 uint64_t body_size, std::string *error_out) {
  assert(highest_appended_reservation_within_cursor());
  if (arch_ == ROCJITSU_CODE_ARCH_INVALID || original_size < sizeof(uint32_t) ||
      original_size % sizeof(uint32_t) != 0 || body_size == 0 ||
      body_size % sizeof(uint32_t) != 0 || anchor_offset > original_text_size_ ||
      original_size > original_text_size_ - anchor_offset ||
      body_size > std::numeric_limits<uint64_t>::max() - appended_cursor_ ||
      !range_is_free(anchor_offset, anchor_offset + original_size) ||
      !range_is_free(appended_cursor_, appended_cursor_ + body_size)) {
    report(error_out, "DBI patch placement: invalid or overlapping indirect appended reservation");
    return std::nullopt;
  }

  const uint64_t body_offset = appended_cursor_;
  const uint64_t body_end = body_offset + body_size;
  const std::array reservations = {
      std::pair{anchor_offset, anchor_offset + original_size},
      std::pair{body_offset, body_end},
  };
  if (!reserve_ranges(reservations)) {
    report(error_out, "DBI patch placement: indirect appended reservations conflicted");
    return std::nullopt;
  }
  appended_cursor_ = body_end;
  assert(highest_appended_reservation_within_cursor());
  return DbiPatchPlacement{
      .kind = DbiPatchPlacementKind::AppendedCave,
      .anchor_offset = anchor_offset,
      .original_size = original_size,
      .body_offset = body_offset,
      .body_size = body_size,
      .return_branch_offset = 0u,
      .return_target = anchor_offset + original_size,
  };
}

} // namespace rocjitsu
