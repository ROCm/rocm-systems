/*************************************************************************
 * Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/*************************************************************************
 * Network Counter Collector
 *
 * Self-contained library for collecting Thor2 NIC counters before/after
 * an operation and printing a summary table.  No dependency on NCCL,
 * RCCL, or any GPU runtime -- can be integrated into any application.
 *
 * Environment variables:
 *   RCCL_TESTS_NET_COUNTER_ENABLE=1   – enable collection
 *   RCCL_TESTS_NIC_COUNTER_LIST=a,b   – comma-separated counter names
 *                                        (default: all known counters)
 *   NCCL_IB_HCA=ib0,ib1,...           – IB device list (primary)
 *   RCCL_TESTS_NET_COUNTER_NIC_PREFIX – NIC prefix filter for auto-discovery
 *
 * Counter sources (looked up automatically from registry):
 *   COUNTER_SRC_ETHTOOL  – ethtool -S <ethernet device>
 *   COUNTER_SRC_IB_HW    – /sys/class/infiniband/<dev>/ports/1/hw_counters/
 *   COUNTER_SRC_DEBUGFS  – /sys/kernel/debug/bnxt_re/<dev>/info
 *************************************************************************/
#ifndef __COLLECTOR_H__
#define __COLLECTOR_H__

#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// ---- types --------------------------------------------------------------

enum CounterSource {
  COUNTER_SRC_ETHTOOL,
  COUNTER_SRC_IB_HW,
  COUNTER_SRC_DEBUGFS
};

struct CounterDescriptor {
  std::string name;
  CounterSource source;
  bool is_prefix;   // true  → prefix match (expands to name0..name7)
};

struct NetworkCounterSnapshot {
  std::map<std::string, uint64_t> counters;
  char nic_name[256];
  char ib_device[256];
  long timestamp;
};

struct NetworkCounterContext {
  std::vector<NetworkCounterSnapshot> snapshots_before;
  std::vector<std::string> nic_names;   // ethernet NIC names (for ethtool)
  std::vector<std::string> ib_names;    // IB device names (for hw_counters / debugfs)
  std::vector<CounterDescriptor> selected_counters;
  int  base_rank;
  int  nranks;
  int  nGpus;
  bool enabled;
};

// ---- public API ---------------------------------------------------------

// Check RCCL_TESTS_NET_COUNTER_ENABLE=1
bool NetCounterIsEnabled();

// Parse RCCL_TESTS_NIC_COUNTER_LIST (or return full registry)
std::vector<CounterDescriptor> NetCounterParseCounterList();

// Parse NCCL_IB_HCA into IB device names (empty if unset)
std::vector<std::string> NetCounterParseIbHcaList();

// Device lookup helpers
std::string NetCounterFindIbDeviceForNic(const std::string& nic);
std::string NetCounterFindNicForIbDevice(const std::string& ib_device);

// Discover ethernet NICs on the local node (filtered by NIC prefix env vars)
void NetCounterGetNetworkInterfaces(std::vector<std::string>& interfaces);

// Snapshot selected counters for one device (NIC + IB name)
NetworkCounterSnapshot NetCounterCollectSnapshot(
    const std::string& nic,
    const std::string& ib_dev,
    const std::vector<CounterDescriptor>& selected);

// Convenience: auto-resolve IB device from NIC name
NetworkCounterSnapshot NetCounterCollectSnapshot(
    const std::string& nic,
    const std::vector<CounterDescriptor>& selected);

// Compute delta for a single counter between two snapshots
uint64_t NetCounterComputeDelta(
    const NetworkCounterSnapshot& before,
    const NetworkCounterSnapshot& after,
    const CounterDescriptor& desc);

// Print one table per node to stdout
void NetCounterPrintTable(
    const std::vector<std::string>& nic_names,
    const std::vector<NetworkCounterSnapshot>& before,
    const std::vector<NetworkCounterSnapshot>& after,
    int rank,
    const std::vector<CounterDescriptor>& selected);

#endif // __COLLECTOR_H__
