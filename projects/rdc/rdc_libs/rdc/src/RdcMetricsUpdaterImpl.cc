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
#include "rdc_lib/impl/RdcMetricsUpdaterImpl.h"
#include <sys/time.h>
#include <ctime>
#include <chrono>
#include "rdc_lib/rdc_common.h"

namespace amd {
namespace rdc {

RdcMetricsUpdaterImpl::RdcMetricsUpdaterImpl(
            const RdcWatchTablePtr& watch_table,
            const uint32_t check_frequency):
            watch_table_(watch_table)
            , started_(false)
            , _check_frequency(check_frequency) {
}

void RdcMetricsUpdaterImpl::start() {
    if (started_) {
        return;
    }
    started_ = true;
    updater_ = std::async(std::launch::async, [this](){
        while (started_) {
            watch_table_->rdc_update_all_fields();
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    });
}

void RdcMetricsUpdaterImpl::stop() {
    started_ = false;
}

}  // namespace rdc
}  // namespace amd
