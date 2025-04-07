/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCMETRICSUPDATERIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCMETRICSUPDATERIMPL_H_

#include <future>  // NOLINT(build/c++11)
#include <memory>

#include "rdc_lib/RdcMetricsUpdater.h"
#include "rdc_lib/RdcWatchTable.h"

namespace amd {
namespace rdc {

class RdcMetricsUpdaterImpl final : public RdcMetricsUpdater {
 public:
  void start() override;
  void stop() override;
  explicit RdcMetricsUpdaterImpl(const RdcWatchTablePtr& watch_table,
                                 const uint32_t check_frequency);
  ~RdcMetricsUpdaterImpl() = default;

 private:
  RdcWatchTablePtr watch_table_;
  std::atomic<bool> started_;
  std::future<void> updater_;        // keep the future of updater
  std::future<void> notif_updater_;  // keep the future of notif updater
  const uint32_t _check_frequency;   // Check frequency in milliseconds
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCMETRICSUPDATERIMPL_H_
