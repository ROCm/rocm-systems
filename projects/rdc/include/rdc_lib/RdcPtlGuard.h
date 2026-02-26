/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RDC_MODULES_RDC_ROCP_RDCPTLGUARD_H_
#define RDC_MODULES_RDC_ROCP_RDCPTLGUARD_H_

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "rdc_lib/RdcLogger.h"

namespace amd {
namespace rdc {

/**
 * @brief RAII guard that disables PTL via sysfs on targeted GPUs for the duration of its lifetime.
 *
 * Prevents rapid PTL toggling when counter profiles are sampled.
 * On construction, discovers /sys/class/drm/card*\/device/ptl/ptl_enable files,
 * matches them to the requested GPU device indices by PCI BDF order,
 * saves their current state, and disables PTL on any that are enabled.
 * On destruction, restores each to its original state.
 *
 * GPU device indices correspond to the amdsmi enumeration order (sorted by PCI BDF).
 */
class PtlGuard {
 public:
  /**
   * @param device_indices Set of GPU device indices to guard. GPUs are identified
   *        by their position in PCI BDF-sorted order (matching amdsmi GPU numbering).
   */
  explicit PtlGuard(const std::set<uint32_t>& device_indices) {
    namespace fs = std::filesystem;
    std::error_code ec;

    if (device_indices.empty()) {
      return;
    }

    const std::string drm_dir = "/sys/class/drm";
    if (!fs::is_directory(drm_dir, ec)) {
      return;
    }

    // Discover all physical GPUs with PTL support and their PCI BDFs
    struct PtlGpu {
      std::string bdf;
      std::string ptl_path;
    };
    std::vector<PtlGpu> ptl_gpus;

    for (const auto& entry : fs::directory_iterator(drm_dir, ec)) {
      auto name = entry.path().filename().string();
      if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) {
        continue;
      }
      auto ptl_path = entry.path() / "device" / "ptl" / "ptl_enable";
      if (!fs::is_regular_file(ptl_path, ec)) {
        continue;
      }
      // Read PCI BDF from device symlink
      auto device_link = fs::read_symlink(entry.path() / "device", ec);
      if (ec) {
        continue;
      }
      std::string bdf = device_link.filename().string();
      ptl_gpus.push_back({bdf, ptl_path.string()});
    }

    // Sort by BDF to match amdsmi GPU enumeration order
    std::sort(ptl_gpus.begin(), ptl_gpus.end(),
              [](const PtlGpu& a, const PtlGpu& b) { return a.bdf < b.bdf; });

    // Only disable PTL on the requested device indices
    for (uint32_t idx : device_indices) {
      if (idx >= ptl_gpus.size()) {
        continue;
      }
      const auto& gpu = ptl_gpus[idx];

      std::ifstream in(gpu.ptl_path);
      if (!in) {
        continue;
      }
      std::string state;
      in >> state;
      in.close();

      if (state == "disabled") {
        entries_.push_back({gpu.ptl_path, state, false});
        continue;
      }

      std::ofstream out(gpu.ptl_path);
      if (!out) {
        RDC_LOG(RDC_INFO, "Failed to open PTL sysfs for writing: " << gpu.ptl_path);
        continue;
      }
      out << "disabled" << std::endl;
      out.close();
      entries_.push_back({gpu.ptl_path, state, true});
      RDC_LOG(RDC_DEBUG, "PtlGuard: disabled PTL on GPU " << idx << " (" << gpu.ptl_path
                                                           << ", was " << state << ")");
    }

    if (!entries_.empty()) {
      int disabled_count = 0;
      int skipped_count = 0;
      for (const auto& e : entries_) {
        if (e.needs_restore)
          disabled_count++;
        else
          skipped_count++;
      }
      RDC_LOG(RDC_DEBUG, "PtlGuard: acquired — disabled " << disabled_count << " GPUs, "
                                                           << skipped_count
                                                           << " already disabled");
    }
  }

  ~PtlGuard() noexcept {
    int restored_count = 0;
    for (const auto& e : entries_) {
      if (!e.needs_restore) {
        continue;
      }
      try {
        std::ofstream out(e.path);
        if (out) {
          out << e.original_state << std::endl;
          restored_count++;
          RDC_LOG(RDC_DEBUG, "PtlGuard: restored PTL on " << e.path << " to "
                                                           << e.original_state);
        }
      } catch (...) {
      }
    }
    if (restored_count > 0) {
      RDC_LOG(RDC_DEBUG, "PtlGuard: released — restored " << restored_count << " GPUs");
    }
  }

  PtlGuard(const PtlGuard&) = delete;
  PtlGuard& operator=(const PtlGuard&) = delete;
  PtlGuard(PtlGuard&&) = delete;
  PtlGuard& operator=(PtlGuard&&) = delete;

 private:
  struct PtlEntry {
    std::string path;
    std::string original_state;
    bool needs_restore;
  };
  std::vector<PtlEntry> entries_;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCP_RDCPTLGUARD_H_
