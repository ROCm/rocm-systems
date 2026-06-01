/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <array>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <vector>

#include "amd_smi/amdsmi.h"

// ---------------------------------------------------------------------------
// NOTE: Partition changes alter device topology -- AMD SMI must re-initialize
//
// Changing either memory partition (NPS mode) or accelerator (compute)
// partition changes the number of logical devices visible to the OS and to
// AMD SMI. AMD SMI builds its internal device table at amdsmi_init(); it does
// not update live. After any partition change, callers must call
// amdsmi_shut_down() followed by amdsmi_init() to pull in the updated
// topology; any amdsmi_processor_handle obtained before re-initialization is
// invalid and must not be used.
//
//   Memory partition (NPS mode):
//     Requires amdsmi_gpu_driver_reload() to take effect. The driver reload
//     tears down and rebuilds all kernel device objects, which also destroys
//     any existing handles. amdsmi_shut_down() + amdsmi_init() is therefore
//     mandatory before querying or changing anything further.
//
//   Accelerator (compute) partition:
//     Takes effect immediately without a driver reload, but still changes the
//     logical device count (e.g. SPX: 1 handle per physical GPU vs. DPX: 2).
//     amdsmi_shut_down() + amdsmi_init() is required to get the correct
//     post-change handle list before issuing any further queries or sets.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Partition name converters
// ---------------------------------------------------------------------------

[[nodiscard]] static std::string_view resource_type_str(
    amdsmi_accelerator_partition_resource_type_t t) {
  switch (t) {
    case AMDSMI_ACCELERATOR_XCC:
      return "XCC";
    case AMDSMI_ACCELERATOR_ENCODER:
      return "ENCODER";
    case AMDSMI_ACCELERATOR_DECODER:
      return "DECODER";
    case AMDSMI_ACCELERATOR_DMA:
      return "DMA";
    case AMDSMI_ACCELERATOR_JPEG:
      return "JPEG";
    default:
      return "N/A";
  }
}

[[nodiscard]] static std::string_view accel_partition_str(amdsmi_accelerator_partition_type_t t) {
  switch (t) {
    case AMDSMI_ACCELERATOR_PARTITION_SPX:
      return "SPX";
    case AMDSMI_ACCELERATOR_PARTITION_DPX:
      return "DPX";
    case AMDSMI_ACCELERATOR_PARTITION_TPX:
      return "TPX";
    case AMDSMI_ACCELERATOR_PARTITION_QPX:
      return "QPX";
    case AMDSMI_ACCELERATOR_PARTITION_CPX:
      return "CPX";
    default:
      return "N/A";
  }
}

[[nodiscard]] static std::string_view mem_partition_type_str(amdsmi_memory_partition_type_t t) {
  switch (t) {
    case AMDSMI_MEMORY_PARTITION_NPS1:
      return "NPS1";
    case AMDSMI_MEMORY_PARTITION_NPS2:
      return "NPS2";
    case AMDSMI_MEMORY_PARTITION_NPS4:
      return "NPS4";
    case AMDSMI_MEMORY_PARTITION_NPS8:
      return "NPS8";
    default:
      return "unknown";
  }
}

// Returns a comma-separated list of supported NPS modes (e.g. "NPS1,NPS2,NPS4").
[[nodiscard]] static std::string nps_caps_str(const amdsmi_nps_caps_t& caps) {
  const auto& f = caps.nps_flags;
  std::string s;
  auto append = [&](uint32_t cap, std::string_view name) {
    if (cap) {
      if (!s.empty()) s += ',';
      s += name;
    }
  };
  append(f.nps1_cap, "NPS1");
  append(f.nps2_cap, "NPS2");
  append(f.nps4_cap, "NPS4");
  append(f.nps8_cap, "NPS8");
  return s.empty() ? "none" : s;
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

// Returns a short human-readable status name (strips the detail after ':').
[[nodiscard]] static std::string status_str(amdsmi_status_t ret) {
  const char* s = nullptr;
  amdsmi_status_code_to_string(ret, &s);
  if (!s) return "unknown";
  std::string full{s};
  if (auto pos = full.find(":"); pos != std::string::npos) full.erase(pos);
  return full;
}

static void print_separator(std::string_view title = "") {
  constexpr std::string_view kFill{
      "----------------------------------------------------------------------"};
  if (title.empty()) {
    std::cout << kFill << '\n';
    return;
  }
  std::cout << "\n--- " << title << ' ';
  const auto prefix_len = 5 + title.size();
  if (prefix_len < kFill.size()) std::cout << kFill.substr(prefix_len);
  std::cout << '\n';
}

// ---------------------------------------------------------------------------
// RAII amdsmi lifecycle
// ---------------------------------------------------------------------------

// Scoped amdsmi session. Calls amdsmi_init on construction and amdsmi_shut_down
// on destruction. Re-enumerate (new instance) after any partition change that
// alters device topology:
//   - Memory partition change requires a driver reload, which destroys all
//     kernel device objects and invalidates every amdsmi_processor_handle.
//   - Accelerator partition change changes the number of visible logical
//     devices (e.g. SPX: 1 per GPU -> DPX: 2 per GPU), so the handle list
//     built at amdsmi_init is stale even without a driver reload.
// Treat every amdsmi_processor_handle as valid only within the session that
// obtained it.
class AmdsmiSession {
 public:
  explicit AmdsmiSession(uint64_t init_flags = AMDSMI_INIT_AMD_GPUS) {
    auto ret = amdsmi_init(init_flags);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      std::cout << "amdsmi_init failed: " << status_str(ret) << '\n';
      return;
    }
    ok_ = true;
    std::cout << "AMDSMI session started (amdsmi_init called)\n";
  }
  ~AmdsmiSession() {
    if (!ok_) return;
    amdsmi_shut_down();
    std::cout << "AMDSMI session ended (amdsmi_shut_down called)\n";
  }

  bool is_ok() const { return ok_; }

  AmdsmiSession(const AmdsmiSession&) = delete;
  AmdsmiSession& operator=(const AmdsmiSession&) = delete;
  AmdsmiSession(AmdsmiSession&&) = delete;
  AmdsmiSession& operator=(AmdsmiSession&&) = delete;

 private:
  bool ok_ = false;
};

// ---------------------------------------------------------------------------
// GPU enumeration
// ---------------------------------------------------------------------------

// Enumerate all AMD GPU handles across all sockets into a flat vector.
[[nodiscard]] static std::vector<amdsmi_processor_handle> get_all_gpu_handles() {
  uint32_t socket_count = 0;
  if (auto ret = amdsmi_get_socket_handles(&socket_count, nullptr); ret != AMDSMI_STATUS_SUCCESS) {
    std::cout << "amdsmi_get_socket_handles failed: " << status_str(ret) << '\n';
    return {};
  }

  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (auto ret = amdsmi_get_socket_handles(&socket_count, sockets.data());
      ret != AMDSMI_STATUS_SUCCESS) {
    std::cout << "amdsmi_get_socket_handles failed: " << status_str(ret) << '\n';
    return {};
  }

  std::vector<amdsmi_processor_handle> handles;
  for (auto socket : sockets) {
    uint32_t dev_count = 0;
    if (amdsmi_get_processor_handles(socket, &dev_count, nullptr) != AMDSMI_STATUS_SUCCESS)
      continue;
    std::vector<amdsmi_processor_handle> devs(dev_count);
    if (amdsmi_get_processor_handles(socket, &dev_count, devs.data()) != AMDSMI_STATUS_SUCCESS)
      continue;
    handles.insert(handles.end(), devs.begin(), devs.end());
  }
  return handles;
}

// Returns only the "primary" handles from a full handle list -- i.e., those
// that successfully respond to amdsmi_get_gpu_accelerator_partition_profile_config.
// In a partitioned layout (e.g. CPX/NPS4) amdsmi enumerates one handle per
// *logical* partition. Only the root handle of each physical GPU owns the
// partition config; sub-partition handles return NOT_SUPPORTED. Calling set on
// a sub-partition handle is a no-op (NOT_SUPPORTED), so filter them out first.
// Each entry is {amd-smi GPU index, handle}.
using PrimaryGpuList = std::vector<std::pair<uint32_t, amdsmi_processor_handle>>;

[[nodiscard]] static PrimaryGpuList get_primary_gpu_handles(
    const std::vector<amdsmi_processor_handle>& all_handles) {
  PrimaryGpuList primary;
  for (size_t i = 0; i < all_handles.size(); ++i) {
    amdsmi_accelerator_partition_profile_config_t cfg{};
    if (amdsmi_get_gpu_accelerator_partition_profile_config(all_handles[i], &cfg) ==
        AMDSMI_STATUS_SUCCESS)
      primary.emplace_back(static_cast<uint32_t>(i), all_handles[i]);
  }
  return primary;
}

// ---------------------------------------------------------------------------
// Per-step display helpers
// ---------------------------------------------------------------------------

static void print_current_partition(uint32_t idx, amdsmi_processor_handle gpu) {
  std::cout << "  GPU " << idx << ":\n";

  amdsmi_accelerator_partition_profile_t profile{};
  std::array<uint32_t, AMDSMI_MAX_ACCELERATOR_PARTITIONS> partition_ids{};
  if (auto ret = amdsmi_get_gpu_accelerator_partition_profile(gpu, &profile, partition_ids.data());
      ret == AMDSMI_STATUS_SUCCESS) {
    std::cout << "    Accelerator profile type : " << accel_partition_str(profile.profile_type)
              << '\n'
              << "    Profile index            : " << profile.profile_index << '\n'
              << "    Num partitions           : " << profile.num_partitions << '\n'
              << "    Partition ID             : " << partition_ids[0] << '\n'
              << "    Compatible NPS           : " << nps_caps_str(profile.memory_caps) << '\n';
  } else {
    std::cout << "    amdsmi_get_gpu_accelerator_partition_profile: " << status_str(ret) << '\n';
  }

  amdsmi_memory_partition_config_t mem_cfg{};
  if (auto ret = amdsmi_get_gpu_memory_partition_config(gpu, &mem_cfg);
      ret == AMDSMI_STATUS_SUCCESS) {
    std::cout << "    Memory partition : " << mem_partition_type_str(mem_cfg.mp_mode) << '\n';
  } else {
    std::cout << "    amdsmi_get_gpu_memory_partition_config: " << status_str(ret) << '\n';
  }
}

static void print_available_modes(uint32_t idx, amdsmi_processor_handle gpu) {
  std::cout << "  GPU " << idx << ":\n";
  amdsmi_memory_partition_config_t cfg{};
  amdsmi_status_t ret = amdsmi_get_gpu_memory_partition_config(gpu, &cfg);

  if (ret == AMDSMI_STATUS_SUCCESS) {
    std::cout << "    Supported NPS modes     : " << nps_caps_str(cfg.partition_caps) << '\n';
  } else {
    std::cout << "    amdsmi_get_gpu_memory_partition_config: " << status_str(ret) << '\n';
  }

  amdsmi_accelerator_partition_profile_config_t acc_cfg{};
  ret = amdsmi_get_gpu_accelerator_partition_profile_config(gpu, &acc_cfg);
  if (ret == AMDSMI_STATUS_SUCCESS) {
    std::cout << "    Available accelerator profiles (" << acc_cfg.num_profiles << "):\n";
    uint32_t res_idx = 0;
    for (uint32_t p = 0; p < acc_cfg.num_profiles; ++p) {
      const auto& prof = acc_cfg.profiles[p];
      std::cout << "      Index " << prof.profile_index << " ("
                << accel_partition_str(prof.profile_type) << ", " << prof.num_partitions
                << " partition(s), compatible NPS: " << nps_caps_str(prof.memory_caps) << ")\n";
      for (uint32_t r = 0; r < prof.num_resources; ++r, ++res_idx) {
        const auto& res = acc_cfg.resource_profiles[res_idx];
        std::cout << "        Resource " << r << ": " << resource_type_str(res.resource_type)
                  << ", per_partition=" << res.partition_resource
                  << ", shared_by=" << res.num_partitions_share_resource << "\n";
      }
    }
  } else {
    std::cout << "    amdsmi_get_gpu_accelerator_partition_profile_config: " << status_str(ret)
              << '\n';
  }
}

// ---------------------------------------------------------------------------
// NPS → accelerator-profile map
// ---------------------------------------------------------------------------

// Maps each NPS mode to the accelerator profiles that are compatible with it.
// Built by inverting the memory_caps bitmask on every profile returned by
// amdsmi_get_gpu_accelerator_partition_profile_config.
using NpsProfileMap =
    std::map<amdsmi_memory_partition_type_t, std::vector<amdsmi_accelerator_partition_profile_t>>;

[[nodiscard]] static NpsProfileMap build_nps_to_profiles_map(
    const amdsmi_accelerator_partition_profile_config_t& cfg) {
  NpsProfileMap m;
  for (uint32_t p = 0; p < cfg.num_profiles; ++p) {
    const auto& prof = cfg.profiles[p];
    const auto& f = prof.memory_caps.nps_flags;
    if (f.nps1_cap) m[AMDSMI_MEMORY_PARTITION_NPS1].push_back(prof);
    if (f.nps2_cap) m[AMDSMI_MEMORY_PARTITION_NPS2].push_back(prof);
    if (f.nps4_cap) m[AMDSMI_MEMORY_PARTITION_NPS4].push_back(prof);
    if (f.nps8_cap) m[AMDSMI_MEMORY_PARTITION_NPS8].push_back(prof);
  }
  return m;
}

// ---------------------------------------------------------------------------
// Workflow helpers
// ---------------------------------------------------------------------------

static void print_current_partition_info(const std::vector<amdsmi_processor_handle>& gpus) {
  print_separator("Current partition settings");
  uint32_t idx = 0;
  for (auto gpu : gpus) print_current_partition(idx++, gpu);
}

static void print_available_partition_modes(const std::vector<amdsmi_processor_handle>& gpus) {
  print_separator("Available partition modes");
  uint32_t idx = 0;
  for (auto gpu : gpus) print_available_modes(idx++, gpu);
}

// Returns true if the memory partition was set successfully.
[[nodiscard]] static bool set_memory_partition(amdsmi_processor_handle gpu0,
                                               amdsmi_memory_partition_type_t target) {
  print_separator("Set memory partition");
  // Memory partition is hive-wide -- setting it on one device affects all.
  // Change the target mode to another supported mode as needed.
  std::cout << "  Attempting: " << mem_partition_type_str(target) << '\n';
  auto ret = amdsmi_set_gpu_memory_partition_mode(gpu0, target);
  std::cout << "  amdsmi_set_gpu_memory_partition_mode(GPU 0, " << mem_partition_type_str(target)
            << "): " << status_str(ret) << '\n';
  return ret == AMDSMI_STATUS_SUCCESS;
}

// Returns true if the driver was reloaded successfully.
[[nodiscard]] static bool reload_driver() {
  print_separator("Reload driver");
  // Mandatory to apply memory partition change. All GPU activity must be
  // stopped first. The reload may reset the accelerator partition to default.
  // Root privileges (sudo) are required for driver reload.
  std::cout << "  Reloading driver, this may take some time...\n";
  auto ret = amdsmi_gpu_driver_reload();
  std::cout << "  amdsmi_gpu_driver_reload: " << status_str(ret) << '\n';
  return ret == AMDSMI_STATUS_SUCCESS;
}

// For each GPU: build the NPS→profiles map, print every entry, then demonstrate
// iterating over only the profiles compatible with the currently active NPS mode.
static void print_profiles_by_nps(const std::vector<amdsmi_processor_handle>& gpus) {
  print_separator("Accelerator profiles grouped by NPS mode");

  uint32_t gpu_idx = 0;
  for (auto gpu : gpus) {
    std::cout << "  GPU " << gpu_idx << ":\n";

    // --- query available accelerator profiles ---
    amdsmi_accelerator_partition_profile_config_t acc_cfg{};
    amdsmi_status_t ret = amdsmi_get_gpu_accelerator_partition_profile_config(gpu, &acc_cfg);
    if (ret != AMDSMI_STATUS_SUCCESS) {
      std::cout << "    amdsmi_get_gpu_accelerator_partition_profile_config: " << status_str(ret)
                << '\n';
      ++gpu_idx;
      continue;
    }

    // --- query current NPS mode ---
    amdsmi_memory_partition_config_t mem_cfg{};
    amdsmi_memory_partition_type_t current_nps{};
    bool has_nps = (amdsmi_get_gpu_memory_partition_config(gpu, &mem_cfg) == AMDSMI_STATUS_SUCCESS);
    if (has_nps) current_nps = mem_cfg.mp_mode;

    // --- build map ---
    const NpsProfileMap nps_map = build_nps_to_profiles_map(acc_cfg);

    // Print every NPS mode and its compatible profiles
    std::cout << "    All profiles by NPS mode:\n";
    for (const auto& [nps, profiles] : nps_map) {
      bool is_current = has_nps && (nps == current_nps);
      std::cout << "      " << mem_partition_type_str(nps) << (is_current ? " [current]" : "")
                << " (" << profiles.size() << " profile(s)):\n";
      for (const auto& prof : profiles) {
        std::cout << "        index=" << prof.profile_index
                  << "  type=" << accel_partition_str(prof.profile_type)
                  << "  partitions=" << prof.num_partitions << '\n';
      }
    }

    // Iterate over profiles for the *current* NPS mode
    if (has_nps) {
      std::cout << "    Profiles available for current NPS (" << mem_partition_type_str(current_nps)
                << "):\n";
      auto it = nps_map.find(current_nps);
      if (it != nps_map.end()) {
        for (const auto& prof : it->second) {
          std::cout << "      -> index=" << prof.profile_index
                    << "  type=" << accel_partition_str(prof.profile_type)
                    << "  partitions=" << prof.num_partitions << '\n';
        }
      } else {
        std::cout << "      (none)\n";
      }
    }

    ++gpu_idx;
  }
}

// Set the accelerator partition on every *primary* handle.
//
// Why primary-only:
//   In a partitioned layout (e.g. CPX+NPS4) amdsmi enumerates one handle per
//   logical partition (e.g. 64 handles for 8 physical GPUs in CPX). Only the
//   root handle of each physical GPU owns the partition config; the rest return
//   NOT_SUPPORTED. Calling set on them is noise, so we filter first.
static void set_accelerator_partition_all_devices(
    const std::vector<amdsmi_processor_handle>& all_gpus, uint32_t profile_index,
    std::string_view profile_type_name) {
  print_separator(std::string("Set accelerator partition -> index=") +
                  std::to_string(profile_index) + " (" + std::string(profile_type_name) + ")");

  const auto primary = get_primary_gpu_handles(all_gpus);
  std::cout << "  Primary handles: " << primary.size() << " of " << all_gpus.size() << " total\n";

  uint32_t phy_idx = 0;
  for (const auto& [gpu_num, gpu] : primary) {
    amdsmi_status_t ret = amdsmi_set_gpu_accelerator_partition_profile(gpu, profile_index);
    std::cout << "  Physical GPU " << phy_idx << " (amd-smi GPU " << gpu_num << ")"
              << ": amdsmi_set_gpu_accelerator_partition_profile -> " << status_str(ret) << '\n';
    ++phy_idx;
  }
}

int main() {
  bool mem_changed = false;

  // -----------------------------------------------------------------------
  // Phase 1: Query state, set memory partition, reload driver if changed.
  //
  // If memory partition is successfully changed, a driver reload follows and
  // the session goes out of scope (amdsmi_shut_down()) before Phase 2
  // re-initializes with fresh handles. If memory partition cannot be changed,
  // the driver reload is skipped and accelerator partition changes are
  // attempted immediately within this same session.
  // -----------------------------------------------------------------------
  {
    AmdsmiSession session;
    auto gpus = get_all_gpu_handles();
    if (gpus.empty()) {
      std::cout << "No GPUs found.\n";
      return 0;
    }
    std::cout << "Found " << gpus.size() << " GPU(s).\n";

    print_current_partition_info(gpus);
    print_available_partition_modes(gpus);

    // In real-world usage, the target mode may be dictated by workload requirements.
    // For this demo, we pick the highest supported NPS mode to show the most
    // significant partition change.

    // Pick the highest supported NPS mode, preferring NPS4 > NPS2 > NPS1.
    // Query the caps from GPU 0 (memory partition is hive-wide, so any GPU works).
    amdsmi_memory_partition_config_t mem_caps_cfg{};
    amdsmi_memory_partition_type_t target_memory_partition = AMDSMI_MEMORY_PARTITION_NPS1;
    amdsmi_memory_partition_type_t current_nps = AMDSMI_MEMORY_PARTITION_UNKNOWN;
    if (amdsmi_get_gpu_memory_partition_config(gpus.front(), &mem_caps_cfg) ==
        AMDSMI_STATUS_SUCCESS) {
      const auto& f = mem_caps_cfg.partition_caps.nps_flags;
      current_nps = mem_caps_cfg.mp_mode;
      if (f.nps4_cap) {
        target_memory_partition = AMDSMI_MEMORY_PARTITION_NPS4;
      } else if (f.nps2_cap) {
        target_memory_partition = AMDSMI_MEMORY_PARTITION_NPS2;
      } else {
        target_memory_partition = AMDSMI_MEMORY_PARTITION_NPS1;
      }
    }
    std::cout << "  Selected NPS target: " << mem_partition_type_str(target_memory_partition)
              << '\n';
    if (current_nps == target_memory_partition) {
      std::cout << "\n[info] Current NPS mode is already " << mem_partition_type_str(current_nps)
                << "; skipping set and driver reload.\n";
    } else {
      mem_changed = set_memory_partition(gpus.front(), target_memory_partition);
      if (mem_changed) {
        if (!reload_driver())
          std::cout << "[warn] Driver reload failed; memory partition change may not have "
                       "taken effect.\n";
      } else {
        std::cout << "\n[info] Memory partition unchanged; skipping driver reload.\n";
      }
    }
  }  // ~AmdsmiSession() -> amdsmi_shut_down()

  // -----------------------------------------------------------------------
  // Phase 2: Re-enumerate to get fresh handles and accurate device count after
  // the driver reload (triggered by the memory partition change). Display the
  // current state and capture the accelerator profiles compatible with the
  // active NPS mode.
  //
  // Accelerator-partition structs (amdsmi_accelerator_partition_profile_t) are
  // plain data with no embedded handles, so they remain valid across
  // amdsmi_shut_down / amdsmi_init boundaries and can be saved here for use
  // in Phase 3.
  // -----------------------------------------------------------------------
  std::vector<amdsmi_accelerator_partition_profile_t> profiles_to_test;
  {
    AmdsmiSession session;
    auto gpus = get_all_gpu_handles();
    if (gpus.empty()) {
      std::cout << "No GPUs found after driver reload.\n";
      return 0;
    }
    std::cout << "Found " << gpus.size() << " GPU(s) after driver reload.\n";

    print_current_partition_info(gpus);
    print_available_partition_modes(gpus);
    print_profiles_by_nps(gpus);

    // Build NPS→profiles map and save the profiles for the active NPS mode.
    // These are plain structs and remain valid after the session is torn down.
    amdsmi_memory_partition_config_t mem_cfg{};
    amdsmi_accelerator_partition_profile_config_t acc_cfg{};
    if (amdsmi_get_gpu_memory_partition_config(gpus.front(), &mem_cfg) == AMDSMI_STATUS_SUCCESS &&
        amdsmi_get_gpu_accelerator_partition_profile_config(gpus.front(), &acc_cfg) ==
            AMDSMI_STATUS_SUCCESS) {
      const NpsProfileMap nps_map = build_nps_to_profiles_map(acc_cfg);
      if (auto it = nps_map.find(mem_cfg.mp_mode); it != nps_map.end()) {
        profiles_to_test = it->second;
        std::cout << "\n[info] " << profiles_to_test.size()
                  << " accelerator profile(s) to iterate for current NPS mode ("
                  << mem_partition_type_str(mem_cfg.mp_mode) << ").\n";
      }
    }
  }  // ~AmdsmiSession() -> amdsmi_shut_down()

  // -----------------------------------------------------------------------
  // Phase 3: For each accelerator profile compatible with the current NPS mode:
  //   a) Open a session, set the profile on every device, close the session.
  //   b) Open a NEW session to re-enumerate before reading anything.
  //      Accelerator partition changes alter the number of logical devices
  //      visible to the OS (e.g. SPX: N handles -> DPX: 2*N handles), so the
  //      handle list from step (a) is immediately stale. A new amdsmi_init
  //      is required to get the correct post-change device count and handles
  //      before displaying topology or issuing further queries.
  // -----------------------------------------------------------------------
  for (const auto& target : profiles_to_test) {
    // a) Set accelerator partition on all devices.
    {
      AmdsmiSession session;
      auto gpus = get_all_gpu_handles();
      if (gpus.empty()) {
        std::cout << "\n[warn] No GPUs found; skipping profile index=" << target.profile_index
                  << " (" << accel_partition_str(target.profile_type) << ").\n";
        continue;
      }
      std::cout << "Found " << gpus.size()
                << " GPU(s) BEFORE re-initialization for accelerator profile change to index="
                << target.profile_index << " (" << accel_partition_str(target.profile_type)
                << ").\n";
      set_accelerator_partition_all_devices(gpus, target.profile_index,
                                            accel_partition_str(target.profile_type));
    }  // ~AmdsmiSession() -> amdsmi_shut_down()

    // b) Re-enumerate and show resulting device topology.
    {
      AmdsmiSession session;
      auto gpus = get_all_gpu_handles();
      if (gpus.empty()) {
        std::cout << "\n[warn] No GPUs found; skipping profile index=" << target.profile_index
                  << " (" << accel_partition_str(target.profile_type) << ").\n";
        continue;
      }
      std::cout << "Found " << gpus.size()
                << " GPU(s) AFTER re-initialization for accelerator profile change to index="
                << target.profile_index << " (" << accel_partition_str(target.profile_type)
                << ").\n";
      print_current_partition_info(gpus);
      print_available_partition_modes(gpus);
      print_profiles_by_nps(gpus);
    }  // ~AmdsmiSession() -> amdsmi_shut_down()
  }

  return 0;
}
