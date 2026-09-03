// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include <gtest/gtest.h>
#include <hsa/amd_hsa_signal.h>
#include <hsa/hsa.h>
#include <unistd.h>
#include <cstdint>
#include "lib/rocprofiler-sdk/kfd/doorbell_map.hpp"
namespace rocprofiler
{
namespace hsa
{
// DOORBELL kind decodes hardware_doorbell_ptr's page slot (§3.2, pure derivation --
// no bind, no page_size); USER kind (.value aliases it), null signal, and no queue
// all reject.
TEST(queue_controller, capture_doorbell_key)
{
    volatile uint64_t doorbell   = 0;
    auto              user_sig   = amd_signal_t{};
    user_sig.kind                = AMD_SIGNAL_KIND_USER;
    user_sig.value               = 0x7f0000004010;
    auto db_sig                  = amd_signal_t{};
    db_sig.kind                  = AMD_SIGNAL_KIND_DOORBELL;
    db_sig.hardware_doorbell_ptr = &doorbell;
    const uint32_t want_slot =
        kfd::doorbell_ptr_to_page_slot(reinterpret_cast<uint64_t>(&doorbell));
    // has_queue=false models "no intercept queue"; sig=nullptr models a null doorbell signal.
    struct tc
    {
        const amd_signal_t* sig;
        bool                has_queue;
        bool                accept;
        const char*         label;
    };
    const tc rows[] = {{&user_sig, true, false, "non-doorbell (USER) kind rejected"},
                       {&db_sig, true, true, "doorbell kind accepted"},
                       {nullptr, false, false, "no intercept queue"},
                       {nullptr, true, false, "null doorbell signal"}};
    for(const auto& tc : rows)
    {
        auto q            = hsa_queue_t{};
        q.doorbell_signal = hsa_signal_t{.handle = reinterpret_cast<uint64_t>(tc.sig)};
        auto e            = capture_doorbell_key(tc.has_queue ? &q : nullptr);
        EXPECT_EQ(e.has_value(), tc.accept) << tc.label;
        if(!tc.accept || !e.has_value()) continue;
        EXPECT_EQ(*e, want_slot) << tc.label;
    }
}
}  // namespace hsa
}  // namespace rocprofiler
