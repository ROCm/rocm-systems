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

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "rdc_lib/RdcLogger.h"

namespace amd {
namespace rdc {

/**
 * @brief RAII guard that disables PTL via sysfs on all GPUs for the duration of its lifetime.
 *
 * Prevents rapid PTL toggling when counter profiles are sampled across GPUs.
 * On construction, discovers all /sys/class/drm/card*\/device/ptl/ptl_enable files,
 * saves their current state, and disables PTL on any that are enabled.
 * On destruction, restores each to its original state.
 */
class PtlGuard {
 public:
  PtlGuard() {
    namespace fs = std::filesystem;
    std::error_code ec;

    // Discover all PTL sysfs files across all GPUs
    const std::string drm_dir = "/sys/class/drm";
    if (!fs::is_directory(drm_dir, ec)) {
      return;
    }

    for (const auto& entry : fs::directory_iterator(drm_dir, ec)) {
      // Only use card* entries, not renderD* (both point to the same device)
      auto name = entry.path().filename().string();
      if (name.rfind("card", 0) != 0) {
        continue;
      }
      auto ptl_path = entry.path() / "device" / "ptl" / "ptl_enable";
      if (!fs::is_regular_file(ptl_path, ec)) {
        continue;
      }

      std::ifstream in(ptl_path);
      if (!in) {
        continue;
      }
      std::string state;
      in >> state;
      in.close();

      if (state == "disabled") {
        // Already disabled, record but don't touch
        entries_.push_back({ptl_path.string(), state, false});
        continue;
      }

      std::ofstream out(ptl_path);
      if (!out) {
        RDC_LOG(RDC_INFO, "Failed to open PTL sysfs for writing: " << ptl_path);
        continue;
      }
      out << "disabled" << std::endl;
      out.close();
      entries_.push_back({ptl_path.string(), state, true});
      RDC_LOG(RDC_DEBUG, "PtlGuard: disabled PTL on " << ptl_path.string()
                                                       << " (was " << state << ")");
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
