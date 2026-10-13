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

#if defined(HSA_AMD_EXT_API_TABLE_STEP_VERSION) && HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x10
// Only descriptor properties the replacement InterceptQueue cannot honor are reported;
// a default (system-memory, normal-priority, unmasked) descriptor reports nothing.
TEST(queue_controller, classify_amd_queue_desc_loss)
{
    const uint32_t cu_mask[2] = {0xffffffffu, 0x0000ffffu};

    auto make_desc = [](uint32_t                 flags,
                        hsa_amd_queue_priority_t priority,
                        uint32_t                 cu_bits,
                        const uint32_t*          mask) {
        auto desc                         = hsa_amd_queue_create_desc_t{};
        desc.version                      = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
        desc.flags                        = static_cast<uint16_t>(flags);
        desc.engine_type                  = HSA_AMD_QUEUE_ENGINE_COMPUTE;
        desc.queue_size_bytes             = 4096 * 64;
        desc.priority                     = priority;
        desc.engine.compute.type          = HSA_QUEUE_TYPE_MULTI;
        desc.engine.compute.cu_mask_count = cu_bits;
        desc.engine.compute.cu_mask       = mask;
        return desc;
    };

    constexpr uint32_t sysm = HSA_AMD_QUEUE_CREATE_SYSTEM_MEM;
    constexpr uint32_t ring = HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF;
    constexpr uint32_t qdsc = HSA_AMD_QUEUE_CREATE_DEVICE_MEM_QUEUE_DESCRIPTOR;
    constexpr auto     norm = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    constexpr auto     high = HSA_AMD_QUEUE_PRIORITY_HIGH;
    constexpr auto     low  = HSA_AMD_QUEUE_PRIORITY_LOW;

    struct tc
    {
        hsa_amd_queue_create_desc_t desc;
        bool                        ring_lost;
        bool                        qdsc_lost;
        bool                        priority_lost;
        bool                        cu_mask_lost;
        const char*                 label;
    };
    const tc rows[] = {
        {make_desc(sysm, norm, 0, nullptr), false, false, false, false, "default descriptor"},
        {make_desc(ring, norm, 0, nullptr), true, false, false, false, "ring flag"},
        {make_desc(qdsc, norm, 0, nullptr), false, true, false, false, "queue-descriptor flag"},
        {make_desc(ring | qdsc, norm, 0, nullptr), true, true, false, false, "both flags"},
        {make_desc(sysm, high, 0, nullptr), false, false, true, false, "high priority"},
        {make_desc(sysm, low, 0, nullptr), false, false, true, false, "low priority"},
        {make_desc(sysm, norm, 64, cu_mask), false, false, false, true, "CU mask"},
        {make_desc(ring | qdsc, high, 64, cu_mask), true, true, true, true, "everything lost"},
    };

    for(const auto& row : rows)
    {
        const auto got = classify_amd_queue_desc_loss(row.desc);
        EXPECT_EQ(got.device_mem_ring_buf, row.ring_lost) << row.label;
        EXPECT_EQ(got.device_mem_queue_descriptor, row.qdsc_lost) << row.label;
        EXPECT_EQ(got.priority, row.priority_lost) << row.label;
        EXPECT_EQ(got.cu_mask, row.cu_mask_lost) << row.label;
        EXPECT_EQ(got.any(),
                  row.ring_lost || row.qdsc_lost || row.priority_lost || row.cu_mask_lost)
            << row.label;
    }

    // Queue type and private segment size are carried over, so they are not reported.
    auto other                                = make_desc(0, norm, 0, nullptr);
    other.engine.compute.private_segment_size = 1024;
    other.engine.compute.type                 = HSA_QUEUE_TYPE_SINGLE;
    EXPECT_FALSE(classify_amd_queue_desc_loss(other).any());
}
#endif
}  // namespace hsa
}  // namespace rocprofiler
