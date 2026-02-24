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

#include "rdc_lib/RdcLogger.h"

namespace amd {
namespace rdc {

/**
 * @brief RAII guard that disables PTL via sysfs for the duration of its lifetime.
 *
 * Prevents rapid PTL toggling when multiple counter profiles are sampled
 * sequentially on the same GPU. PTL is disabled once on construction and
 * restored to its original state on destruction.
 */
class PtlGuard {
 public:
  explicit PtlGuard(uint32_t drm_render_minor) {
    namespace fs = std::filesystem;
    std::error_code ec;

    // Build the sysfs path via the render device symlink
    ptl_path_ =
        "/sys/class/drm/renderD" + std::to_string(drm_render_minor) + "/device/ptl/ptl_enable";

    if (!fs::exists(ptl_path_, ec) || !fs::is_regular_file(ptl_path_, ec)) {
      // PTL not supported on this GPU, nothing to do
      return;
    }

    // Read current state
    std::ifstream in(ptl_path_);
    if (!in) {
      return;
    }
    in >> original_state_;
    in.close();

    if (original_state_ == "disabled") {
      // Already disabled, no action needed
      return;
    }

    // Disable PTL
    std::ofstream out(ptl_path_);
    if (!out) {
      RDC_LOG(RDC_WARNING, "Failed to open PTL sysfs for writing: " << ptl_path_);
      return;
    }
    out << "disabled" << std::endl;
    out.close();
    active_ = true;
  }

  ~PtlGuard() noexcept {
    if (!active_) {
      return;
    }
    try {
      std::ofstream out(ptl_path_);
      if (out) {
        out << original_state_ << std::endl;
      }
    } catch (...) {
      // Best-effort restore; swallow all exceptions
    }
  }

  PtlGuard(const PtlGuard&) = delete;
  PtlGuard& operator=(const PtlGuard&) = delete;
  PtlGuard(PtlGuard&&) = delete;
  PtlGuard& operator=(PtlGuard&&) = delete;

 private:
  std::string ptl_path_;
  std::string original_state_;
  bool active_ = false;
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCP_RDCPTLGUARD_H_
