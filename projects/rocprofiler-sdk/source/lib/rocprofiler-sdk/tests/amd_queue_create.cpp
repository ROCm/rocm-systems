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
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/hsa/queue_controller.hpp"
#include "lib/rocprofiler-sdk/hsa/queue_interposition.hpp"

#include <gtest/gtest.h>

#include <cstdint>

namespace
{
uint32_t raw_destroy_count = 0;

hsa_status_t
count_raw_destroy(hsa_queue_t*)
{
    ++raw_destroy_count;
    return HSA_STATUS_SUCCESS;
}
}  // namespace

TEST(amd_queue_create, destroy_queue_reports_untracked_queue)
{
    rocprofiler::hsa::QueueController controller;
    hsa_queue_t                       queue{};
    queue.id = 1;

    EXPECT_FALSE(controller.destroy_queue(&queue));
}

TEST(amd_queue_create, destroy_queue_with_runtime_fallback_calls_raw_destroy_for_untracked_queue)
{
    rocprofiler::hsa::QueueController controller;
    hsa_queue_t                       queue{};
    queue.id          = 1;
    raw_destroy_count = 0;

    EXPECT_EQ(rocprofiler::hsa::destroy_queue_with_runtime_fallback(
                  controller, &queue, count_raw_destroy),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(raw_destroy_count, 1);
}

TEST(amd_queue_create, destroy_queue_with_runtime_fallback_clears_ignored_queue)
{
    rocprofiler::hsa::QueueController controller;
    amd_queue_v2_t                    queue{};
    queue.hsa_queue.size            = 64;
    queue.hsa_queue.doorbell_signal = hsa_signal_t{.handle = 1};
    auto* hsa_queue                 = &queue.hsa_queue;
    raw_destroy_count               = 0;

    rocprofiler::hsa::queue_interposition::ignore_queue_state(hsa_queue);
    EXPECT_EQ(rocprofiler::hsa::queue_interposition::lookup_queue_state(hsa_queue, true), nullptr);

    EXPECT_EQ(rocprofiler::hsa::destroy_queue_with_runtime_fallback(
                  controller, hsa_queue, count_raw_destroy),
              HSA_STATUS_SUCCESS);
    EXPECT_EQ(raw_destroy_count, 1);

    EXPECT_NE(rocprofiler::hsa::queue_interposition::lookup_queue_state(hsa_queue, true), nullptr);
    rocprofiler::hsa::queue_interposition::destroy_queue_state(hsa_queue);
}

#if HSA_AMD_EXT_API_TABLE_STEP_VERSION >= 0x10

namespace
{
hsa_amd_queue_create_desc_t
make_old_intercept_compatible_desc()
{
    hsa_amd_queue_create_desc_t desc{};
    desc.version                             = HSA_AMD_QUEUE_CREATE_DESC_VERSION;
    desc.flags                               = HSA_AMD_QUEUE_CREATE_SYSTEM_MEM;
    desc.engine_type                         = HSA_AMD_QUEUE_ENGINE_COMPUTE;
    desc.queue_size_bytes                    = 64 * sizeof(hsa_kernel_dispatch_packet_t);
    desc.priority                            = HSA_AMD_QUEUE_PRIORITY_NORMAL;
    desc.engine.compute.type                 = HSA_QUEUE_TYPE_MULTI;
    desc.engine.compute.private_segment_size = HSA_AMD_PRIVATE_SEGMENT_SIZE_DEFAULT;
    return desc;
}
}  // namespace

TEST(amd_queue_create, old_intercept_compatible_compute_descriptor_maps_packet_count)
{
    auto     desc               = make_old_intercept_compatible_desc();
    uint32_t queue_size_packets = 0;

    EXPECT_TRUE(rocprofiler::hsa::is_amd_queue_create_desc_old_intercept_compatible(
        desc, &queue_size_packets));
    EXPECT_EQ(queue_size_packets, 64);
}

TEST(amd_queue_create, old_intercept_compatible_rejects_unrepresentable_descriptors)
{
    auto desc        = make_old_intercept_compatible_desc();
    desc.engine_type = HSA_AMD_QUEUE_ENGINE_SDMA;
    EXPECT_FALSE(rocprofiler::hsa::is_amd_queue_create_desc_old_intercept_compatible(desc));

    desc       = make_old_intercept_compatible_desc();
    desc.flags = HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF;
    EXPECT_FALSE(rocprofiler::hsa::is_amd_queue_create_desc_old_intercept_compatible(desc));

    desc          = make_old_intercept_compatible_desc();
    desc.priority = HSA_AMD_QUEUE_PRIORITY_LOW;
    EXPECT_FALSE(rocprofiler::hsa::is_amd_queue_create_desc_old_intercept_compatible(desc));

    desc                              = make_old_intercept_compatible_desc();
    desc.engine.compute.cu_mask_count = 32;
    uint32_t cu_mask                  = 0x1;
    desc.engine.compute.cu_mask       = &cu_mask;
    EXPECT_FALSE(rocprofiler::hsa::is_amd_queue_create_desc_old_intercept_compatible(desc));

    desc                            = make_old_intercept_compatible_desc();
    desc.engine.compute.reserved[0] = 1;
    EXPECT_FALSE(rocprofiler::hsa::is_amd_queue_create_desc_old_intercept_compatible(desc));

    desc               = make_old_intercept_compatible_desc();
    desc.traffic_class = 1;
    EXPECT_FALSE(rocprofiler::hsa::is_amd_queue_create_desc_old_intercept_compatible(desc));

    desc                  = make_old_intercept_compatible_desc();
    desc.queue_size_bytes = sizeof(hsa_kernel_dispatch_packet_t) + 1;
    EXPECT_FALSE(rocprofiler::hsa::is_amd_queue_create_desc_old_intercept_compatible(desc));

    desc                  = make_old_intercept_compatible_desc();
    desc.queue_size_bytes = 2 * sizeof(hsa_kernel_dispatch_packet_t);
    EXPECT_FALSE(rocprofiler::hsa::is_amd_queue_create_desc_old_intercept_compatible(desc));
}

TEST(amd_queue_create, old_intercept_compatibility_ignores_output_queue_field)
{
    auto desc  = make_old_intercept_compatible_desc();
    desc.queue = reinterpret_cast<hsa_queue_t*>(0x1);

    EXPECT_TRUE(rocprofiler::hsa::is_amd_queue_create_desc_old_intercept_compatible(desc));
}

TEST(amd_queue_create, inline_pass_through_descriptors_should_be_ignored)
{
    auto desc        = make_old_intercept_compatible_desc();
    desc.engine_type = HSA_AMD_QUEUE_ENGINE_SDMA;
    EXPECT_TRUE(rocprofiler::hsa::should_ignore_inline_amd_queue_create_desc(desc));

    desc       = make_old_intercept_compatible_desc();
    desc.flags = HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF;
    EXPECT_TRUE(rocprofiler::hsa::should_ignore_inline_amd_queue_create_desc(desc));

    desc = make_old_intercept_compatible_desc();
    EXPECT_FALSE(rocprofiler::hsa::should_ignore_inline_amd_queue_create_desc(desc));
}

#endif
