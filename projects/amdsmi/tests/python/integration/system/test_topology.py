#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Topology, link and affinity APIs."""

import unittest

import common.api_test as api
import common.common as common


def _same_device(labels):
    """A device is not its own peer, so that pair is a rejection case only."""
    return labels[0] == labels[1]


class TestSystemTopology(api.ApiTestCase):
    def _peer(self):
        return api.Handle("gpu_dst", self.common.processors)

    def test_topo_get_numa_node_number(self):
        self.both("amdsmi_topo_get_numa_node_number", self.handle)

    def test_topo_get_link_weight(self):
        self.both(
            "amdsmi_topo_get_link_weight",
            self.handle,
            self._peer(),
            needs_peer=True,
            skip_when=_same_device,
        )

    def test_topo_get_link_type(self):
        self.both(
            "amdsmi_topo_get_link_type",
            self.handle,
            self._peer(),
            needs_peer=True,
            skip_when=_same_device,
        )

    def test_topo_get_p2p_status(self):
        self.both(
            "amdsmi_topo_get_p2p_status",
            self.handle,
            self._peer(),
            needs_peer=True,
            skip_when=_same_device,
        )

    def test_is_P2P_accessible(self):
        self.both(
            "amdsmi_is_P2P_accessible",
            self.handle,
            self._peer(),
            needs_peer=True,
            skip_when=_same_device,
        )

    def test_get_minmax_bandwidth_between_processors(self):
        self.both(
            "amdsmi_get_minmax_bandwidth_between_processors",
            self.handle,
            self._peer(),
            needs_peer=True,
            skip_when=_same_device,
        )

    def test_get_link_metrics(self):
        self.both("amdsmi_get_link_metrics", self.handle)

    def test_get_link_topology_nearest(self):
        self.both(
            "amdsmi_get_link_topology_nearest",
            self.handle,
            api.enum("link_type", common.LINK_TYPES),
        )

    def test_get_gpu_topo_numa_affinity(self):
        self.both("amdsmi_get_gpu_topo_numa_affinity", self.handle)

    def test_get_gpu_topo_cpu_affinity(self):
        self.both("amdsmi_get_gpu_topo_cpu_affinity", self.handle)


if __name__ == "__main__":
    unittest.main()
