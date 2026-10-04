// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "topology_nearest.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <vector>

namespace {

constexpr uint32_t kCapacity = AMDSMI_MAX_DEVICES * AMDSMI_MAX_NUM_XCP;

struct Peer {
  uint64_t hops = 0;
  uint64_t weight = 0;
  amdsmi_link_type_t type = AMDSMI_LINK_TYPE_XGMI;
  bool accessible = true;
  amdsmi_status_t access_status = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t type_status = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t weight_status = AMDSMI_STATUS_SUCCESS;
};

struct Topology {
  explicit Topology(uint32_t socket_count = 1) : sockets(socket_count) {}

  Peer& AddPeer(uint32_t socket, uint64_t hops, uint64_t weight) {
    peers.push_back({hops, weight});
    sockets[socket].push_back(&peers.back());
    return peers.back();
  }

  amdsmi_status_t Query(amdsmi_topology_nearest_t* result,
                        amdsmi_link_type_t type = AMDSMI_LINK_TYPE_XGMI) {
    return amd::smi::get_link_topology_nearest(
        &source, type, result,
        [this](uint32_t* count, amdsmi_socket_handle* handles) {
          if (!handles) {
            *count = static_cast<uint32_t>(sockets.size());
            return socket_count_status;
          }
          *count = std::min(*count, static_cast<uint32_t>(sockets.size()));
          for (uint32_t i = 0; i < *count; ++i) handles[i] = &sockets[i];
          return socket_fill_status;
        },
        [this](amdsmi_socket_handle socket, uint32_t* count, amdsmi_processor_handle* handles) {
          const auto& devices = *static_cast<std::vector<amdsmi_processor_handle>*>(socket);
          *count = std::min(*count, static_cast<uint32_t>(devices.size()));
          std::copy_n(devices.begin(), *count, handles);
          return processor_status;
        },
        [](amdsmi_processor_handle, amdsmi_processor_handle target, bool* accessible) {
          const auto& peer = *static_cast<Peer*>(target);
          *accessible = peer.accessible;
          return peer.access_status;
        },
        [](amdsmi_processor_handle, amdsmi_processor_handle target, uint64_t* hops,
           amdsmi_link_type_t* type) {
          const auto& peer = *static_cast<Peer*>(target);
          *hops = peer.hops;
          *type = peer.type;
          return peer.type_status;
        },
        [](amdsmi_processor_handle, amdsmi_processor_handle target, uint64_t* weight) {
          const auto& peer = *static_cast<Peer*>(target);
          *weight = peer.weight;
          return peer.weight_status;
        });
  }

  Peer source;
  std::deque<Peer> peers;
  std::vector<std::vector<amdsmi_processor_handle>> sockets;
  amdsmi_status_t socket_count_status = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t socket_fill_status = AMDSMI_STATUS_SUCCESS;
  amdsmi_status_t processor_status = AMDSMI_STATUS_SUCCESS;
};

}  // namespace

TEST(SystemUnit, TopologyNearestBoundsCountToStoredPeers) {
  Topology topology(2);
  std::vector<amdsmi_processor_handle> peers;
  for (uint32_t i = 0; i < 2 * kCapacity; ++i) {
    peers.push_back(&topology.AddPeer(i / kCapacity, 1, 2 * kCapacity - i));
  }
  amdsmi_topology_nearest_t result{};
  result.count = 1;

  ASSERT_EQ(topology.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.count, kCapacity);
  for (uint32_t i = 0; i < result.count; ++i) {
    EXPECT_EQ(result.processor_list[i], peers[peers.size() - 1 - i]) << i;
  }
}

TEST(SystemUnit, TopologyNearestResetsCapacityForEachSocket) {
  Topology topology(2);
  auto* farthest = &topology.AddPeer(0, 3, 10);
  auto* third = &topology.AddPeer(1, 2, 10);
  auto* second = &topology.AddPeer(1, 1, 20);
  auto* nearest = &topology.AddPeer(1, 1, 10);
  amdsmi_topology_nearest_t result{};

  ASSERT_EQ(topology.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.count, 4u);
  EXPECT_EQ(result.processor_list[0], nearest);
  EXPECT_EQ(result.processor_list[1], second);
  EXPECT_EQ(result.processor_list[2], third);
  EXPECT_EQ(result.processor_list[3], farthest);
}

TEST(SystemUnit, TopologyNearestEqualHopsUseAscendingWeight) {
  Topology topology;
  auto* heaviest = &topology.AddPeer(0, 1, UINT64_MAX);
  auto* lightest = &topology.AddPeer(0, 1, 0);
  auto* middle = &topology.AddPeer(0, 1, 10);
  amdsmi_topology_nearest_t result{};

  ASSERT_EQ(topology.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.count, 3u);
  EXPECT_EQ(result.processor_list[0], lightest);
  EXPECT_EQ(result.processor_list[1], middle);
  EXPECT_EQ(result.processor_list[2], heaviest);
}

TEST(SystemUnit, TopologyNearestAntiCorrelatedHopsTakePriority) {
  Topology topology;
  auto* farthest = &topology.AddPeer(0, UINT64_MAX, 0);
  auto* nearest = &topology.AddPeer(0, 0, UINT64_MAX);
  auto* middle = &topology.AddPeer(0, 2, 10);
  amdsmi_topology_nearest_t result{};

  ASSERT_EQ(topology.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.count, 3u);
  EXPECT_EQ(result.processor_list[0], nearest);
  EXPECT_EQ(result.processor_list[1], middle);
  EXPECT_EQ(result.processor_list[2], farthest);
}

TEST(SystemUnit, TopologyNearestTiesRetainAllPeers) {
  Topology topology;
  auto* farthest = &topology.AddPeer(0, 2, 10);
  auto* first = &topology.AddPeer(0, 1, 10);
  auto* second = &topology.AddPeer(0, 1, 10);
  amdsmi_topology_nearest_t result{};

  ASSERT_EQ(topology.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.count, 3u);
  EXPECT_EQ(std::count(result.processor_list, result.processor_list + 2, first), 1);
  EXPECT_EQ(std::count(result.processor_list, result.processor_list + 2, second), 1);
  EXPECT_EQ(result.processor_list[2], farthest);
}

TEST(SystemUnit, TopologyNearestEmptySocketDoesNotHideLaterPeers) {
  Topology topology(3);
  auto* peer = &topology.AddPeer(1, 1, 10);
  amdsmi_topology_nearest_t result{};
  result.count = kCapacity;
  std::fill_n(result.processor_list, kCapacity, &topology.source);

  ASSERT_EQ(topology.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.count, 1u);
  EXPECT_EQ(result.processor_list[0], peer);
  for (uint32_t i = 1; i < kCapacity; ++i) EXPECT_EQ(result.processor_list[i], nullptr);
}

TEST(SystemUnit, TopologyNearestNoSocketsClearsOutput) {
  Topology topology(0);
  amdsmi_topology_nearest_t result{};
  result.count = kCapacity;
  std::fill_n(result.processor_list, kCapacity, &topology.source);

  ASSERT_EQ(topology.Query(&result), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(result.count, 0u);
  for (auto peer : result.processor_list) EXPECT_EQ(peer, nullptr);
}

TEST(SystemUnit, TopologyNearestSkipsSourceAndUnavailablePeers) {
  Topology topology;
  topology.sockets[0].push_back(&topology.source);
  topology.AddPeer(0, 0, 0).accessible = false;
  topology.AddPeer(0, 0, 0).access_status = AMDSMI_STATUS_API_FAILED;
  topology.AddPeer(0, 0, 0).type = AMDSMI_LINK_TYPE_PCIE;
  topology.AddPeer(0, 0, 0).type_status = AMDSMI_STATUS_NOT_SUPPORTED;
  topology.AddPeer(0, 0, 0).weight_status = AMDSMI_STATUS_API_FAILED;
  auto* valid = &topology.AddPeer(0, 1, 10);
  amdsmi_topology_nearest_t result{};

  ASSERT_EQ(topology.Query(&result), AMDSMI_STATUS_SUCCESS);
  ASSERT_EQ(result.count, 1u);
  EXPECT_EQ(result.processor_list[0], valid);
}

TEST(SystemUnit, TopologyNearestNoMatchingPeersClearsOutput) {
  Topology topology;
  topology.AddPeer(0, 1, 10).type = AMDSMI_LINK_TYPE_PCIE;
  amdsmi_topology_nearest_t result{};
  result.count = kCapacity;
  std::fill_n(result.processor_list, kCapacity, &topology.source);

  ASSERT_EQ(topology.Query(&result), AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(result.count, 0u);
  for (auto peer : result.processor_list) EXPECT_EQ(peer, nullptr);
}

TEST(SystemUnit, TopologyNearestPropagatesDiscoveryErrors) {
  Topology topology;
  topology.AddPeer(0, 1, 10);
  amdsmi_topology_nearest_t result{};

  topology.socket_count_status = AMDSMI_STATUS_NOT_SUPPORTED;
  EXPECT_EQ(topology.Query(&result), AMDSMI_STATUS_NOT_SUPPORTED);
  topology.socket_count_status = AMDSMI_STATUS_SUCCESS;
  topology.socket_fill_status = AMDSMI_STATUS_API_FAILED;
  EXPECT_EQ(topology.Query(&result), AMDSMI_STATUS_API_FAILED);
  topology.socket_fill_status = AMDSMI_STATUS_SUCCESS;
  topology.processor_status = AMDSMI_STATUS_INVAL;
  EXPECT_EQ(topology.Query(&result), AMDSMI_STATUS_INVAL);
}
