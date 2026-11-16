/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup hipGraphInstantiate hipGraphInstantiate
 * @{
 * @ingroup GraphTest
 * Graph instantiation constructs AQL packets without executing them. Packet
 * construction must not update queue bookkeeping as though those packets had
 * run: doing so causes unrelated later launches to receive redundant
 * system-scope barriers. This is only observable in the AQL trace, so the
 * workload runs in a child process with tracing enabled. AMD + Linux only.
 */

#include <hip_test_common.hh>

#if HT_AMD && HT_LINUX

#include <hip_test_aqllog.hh>

#include <algorithm>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace aqllog = hip::aqllog;

constexpr int kFenceScopeSystem = 2;

size_t countBarriersBetweenDispatches(const std::string& log) {
  const std::optional<std::string> region = aqllog::ExtractRegion(log, "workload");
  INFO("child trace must contain the workload region");
  REQUIRE(region.has_value());

  const std::vector<aqllog::Packet> packets = aqllog::ParsePackets(*region);
  INFO("workload region must contain AQL packets");
  REQUIRE_FALSE(packets.empty());

  const size_t dispatches = std::count_if(
      packets.begin(), packets.end(),
      [](const aqllog::Packet& packet) { return packet.kind == aqllog::PacketKind::kDispatch; });
  REQUIRE(dispatches == 2);
  return aqllog::BarriersBetweenDispatches(*region).size();
}

}  // namespace

/**
 * Test Description
 * ------------------------
 *  - Runs the child workload with and without graph instantiation and requires
 *    the same number of barrier packets between two subsequent eager kernel
 *    dispatches.
 * Test source
 * ------------------------
 *  - catch/unit/graph/hipGraphInstantiateNullStreamFence.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD backend
 */
HIP_TEST_CASE(Unit_hipGraphInstantiate_NullStreamFenceUnchanged) {
  const std::string with_graph =
      aqllog::RunExecutable("hipGraphInstantiateNullStreamFence_exe", "graph");
  const std::string without_graph =
      aqllog::RunExecutable("hipGraphInstantiateNullStreamFence_exe", "nograph");

  const size_t with_graph_barriers = countBarriersBetweenDispatches(with_graph);
  const size_t without_graph_barriers = countBarriersBetweenDispatches(without_graph);

  INFO("repeated barriers: with graph " << with_graph_barriers << ", without graph "
                                        << without_graph_barriers);
  REQUIRE(with_graph_barriers == without_graph_barriers);
}

/**
 * Test Description
 * ------------------------
 *  - Captures and launches a blit from non-coherent pinned host memory. The
 *    captured dispatch must retain its intrinsic system scope while graph
 *    construction skips real-queue bookkeeping.
 * Test source
 * ------------------------
 *  - catch/unit/graph/hipGraphInstantiateNullStreamFence.cc
 * Test requirements
 * ------------------------
 *  - Linux, AMD backend
 */
HIP_TEST_CASE(Unit_hipGraphInstantiate_CapturedPacketSystemScopePreserved) {
  const std::string log =
      aqllog::RunExecutable("hipGraphInstantiateNullStreamFence_exe", "captured_system_scope");

  const std::optional<std::string> region = aqllog::ExtractRegion(log, "workload");
  INFO("child trace must contain the workload region");
  REQUIRE(region.has_value());

  size_t system_dispatches = 0;
  for (const aqllog::Packet& packet : aqllog::ParsePackets(*region)) {
    if (packet.kind != aqllog::PacketKind::kDispatch) {
      continue;
    }
    REQUIRE(packet.acquire.has_value());
    REQUIRE(packet.release.has_value());
    INFO("captured dispatch line: " << packet.line);
    if (*packet.acquire == kFenceScopeSystem && *packet.release == kFenceScopeSystem) {
      ++system_dispatches;
    }
  }
  REQUIRE(system_dispatches > 0);
}

/**
 * End doxygen group GraphTest.
 * @}
 */

#endif  // HT_AMD && HT_LINUX
