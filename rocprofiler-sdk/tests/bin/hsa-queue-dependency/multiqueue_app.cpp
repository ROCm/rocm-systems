// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

/** ROC Profiler Multi Queue Dependency Test
 *
 * The goal of this test is to ensure ROC profiler does not go to deadlock
 * when multiple queue are created and they are dependent on each other
 *
 */

#include "multiqueue_app.h"

std::vector<hsa_agent_t> Device::all_devices;
std::vector<Device>      MQDependencyTest::cpu;
std::vector<Device>      MQDependencyTest::gpu;
Device::Memory           MQDependencyTest::kernarg;

int
main()
{
    hsa_status_t     status;
    MQDependencyTest obj;

    // Get Agent info
    obj.device_discovery();

    char agent_name[64];
    status = hsa_agent_get_info(obj.gpu[0].agent, HSA_AGENT_INFO_NAME, agent_name);
    RET_IF_HSA_ERR(status)

    // Getting hasco Path
    std::string hasco_file_path = std::string(agent_name) + std::string("_copy.hsaco");
    obj.search_hasco(fs::current_path(), hasco_file_path);

    MQDependencyTest::CodeObject code_object;
    if(!obj.load_code_object(hasco_file_path, obj.gpu[0].agent, code_object))
    {
        printf("Kernel file not found or not usable with given agent.\n");
        abort();
    }

    MQDependencyTest::Kernel copyA;
    if(!obj.get_kernel(code_object, "copyA", obj.gpu[0].agent, copyA))
    {
        printf("Test kernel A not found.\n");
        abort();
    }

    MQDependencyTest::Kernel copyB;
    if(!obj.get_kernel(code_object, "copyB", obj.gpu[0].agent, copyB))
    {
        printf("Test kernel B not found.\n");
        abort();
    }

    MQDependencyTest::Kernel copyC;
    if(!obj.get_kernel(code_object, "copyC", obj.gpu[0].agent, copyC))
    {
        printf("Test kernel C not found.\n");
        abort();
    }

    struct args_t
    {
        uint32_t*                       a      = nullptr;
        uint32_t*                       b      = nullptr;
        MQDependencyTest::OCLHiddenArgs hidden = {};
    };

    args_t* args = static_cast<args_t*>(obj.hsa_malloc(sizeof(args_t), obj.kernarg));
    *args        = {};

    uint32_t* a = static_cast<uint32_t*>(obj.hsa_malloc(64 * sizeof(uint32_t), obj.kernarg));
    uint32_t* b = static_cast<uint32_t*>(obj.hsa_malloc(64 * sizeof(uint32_t), obj.kernarg));

    memset(a, 0, 64 * sizeof(uint32_t));
    memset(b, 1, 64 * sizeof(uint32_t));

    // Create queue in gpu agent and prepare a kernel dispatch packet
    hsa_queue_t* queue1 = nullptr;
    status              = hsa_queue_create(obj.gpu[0].agent,
                              1024,
                              HSA_QUEUE_TYPE_SINGLE,
                              nullptr,
                              nullptr,
                              UINT32_MAX,
                              UINT32_MAX,
                              &queue1);
    RET_IF_HSA_ERR(status)

    // Create a signal with a value of 1 and attach it to the first kernel
    // dispatch packet
    hsa_signal_t completion_signal_1 = {};
    status                           = hsa_signal_create(1, 0, nullptr, &completion_signal_1);
    RET_IF_HSA_ERR(status)

    // First dispath packet on queue 1, Kernel A
    {
        MQDependencyTest::Aql packet{};
        packet.header.type    = HSA_PACKET_TYPE_KERNEL_DISPATCH;
        packet.header.barrier = 1;
        packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
        packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

        packet.dispatch.setup            = 1;
        packet.dispatch.workgroup_size_x = 64;
        packet.dispatch.workgroup_size_y = 1;
        packet.dispatch.workgroup_size_z = 1;
        packet.dispatch.grid_size_x      = 64;
        packet.dispatch.grid_size_y      = 1;
        packet.dispatch.grid_size_z      = 1;

        packet.dispatch.group_segment_size   = copyA.group;
        packet.dispatch.private_segment_size = copyA.scratch;
        packet.dispatch.kernel_object        = copyA.handle;

        packet.dispatch.kernarg_address   = args;
        packet.dispatch.completion_signal = completion_signal_1;

        args->a = a;
        args->b = b;
        // Tell packet processor of A to launch the first kernel dispatch packet
        obj.submit_packet(queue1, packet);
    }

    // Create a signal with a value of 1 and attach it to the second kernel
    // dispatch packet
    hsa_signal_t completion_signal_2 = {};
    status                           = hsa_signal_create(1, 0, nullptr, &completion_signal_2);
    RET_IF_HSA_ERR(status)

    hsa_signal_t completion_signal_3 = {};
    status                           = hsa_signal_create(1, 0, nullptr, &completion_signal_3);
    RET_IF_HSA_ERR(status)

    // Create barrier-AND packet that is enqueued in queue 1
    {
        MQDependencyTest::Aql packet{};
        packet.header.type    = HSA_PACKET_TYPE_BARRIER_AND;
        packet.header.barrier = 1;
        packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
        packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

        packet.barrier_and.dep_signal[0] = completion_signal_2;
        obj.submit_packet(queue1, packet);
    }

    // Second dispath packet on queue 1, Kernel C
    {
        MQDependencyTest::Aql packet{};
        packet.header.type    = HSA_PACKET_TYPE_KERNEL_DISPATCH;
        packet.header.barrier = 1;
        packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
        packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

        packet.dispatch.setup            = 1;
        packet.dispatch.workgroup_size_x = 64;
        packet.dispatch.workgroup_size_y = 1;
        packet.dispatch.workgroup_size_z = 1;
        packet.dispatch.grid_size_x      = 64;
        packet.dispatch.grid_size_y      = 1;
        packet.dispatch.grid_size_z      = 1;

        packet.dispatch.group_segment_size   = copyC.group;
        packet.dispatch.private_segment_size = copyC.scratch;
        packet.dispatch.kernel_object        = copyC.handle;
        packet.dispatch.completion_signal    = completion_signal_3;
        packet.dispatch.kernarg_address      = args;

        args->a = a;
        args->b = b;
        // Tell packet processor to launch the second kernel dispatch packet
        obj.submit_packet(queue1, packet);
    }

    // Create queue 2
    hsa_queue_t* queue2 = nullptr;
    status              = hsa_queue_create(obj.gpu[0].agent,
                              1024,
                              HSA_QUEUE_TYPE_SINGLE,
                              nullptr,
                              nullptr,
                              UINT32_MAX,
                              UINT32_MAX,
                              &queue2);
    RET_IF_HSA_ERR(status)

    // Create barrier-AND packet that is enqueued in queue 2
    {
        MQDependencyTest::Aql packet{};
        packet.header.type    = HSA_PACKET_TYPE_BARRIER_AND;
        packet.header.barrier = 1;
        packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
        packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

        packet.barrier_and.dep_signal[0] = completion_signal_1;
        obj.submit_packet(queue2, packet);
    }

    // Third dispath packet on queue 2, Kernel B
    {
        MQDependencyTest::Aql packet{};
        packet.header.type    = HSA_PACKET_TYPE_KERNEL_DISPATCH;
        packet.header.barrier = 1;
        packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
        packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

        packet.dispatch.setup            = 1;
        packet.dispatch.workgroup_size_x = 64;
        packet.dispatch.workgroup_size_y = 1;
        packet.dispatch.workgroup_size_z = 1;
        packet.dispatch.grid_size_x      = 64;
        packet.dispatch.grid_size_y      = 1;
        packet.dispatch.grid_size_z      = 1;

        packet.dispatch.group_segment_size   = copyB.group;
        packet.dispatch.private_segment_size = copyB.scratch;
        packet.dispatch.kernel_object        = copyB.handle;

        packet.dispatch.kernarg_address   = args;
        packet.dispatch.completion_signal = completion_signal_2;

        args->a = a;
        args->b = b;
        // Tell packet processor to launch the third kernel dispatch packet
        obj.submit_packet(queue2, packet);
    }

    // Wait on the completion signal
    hsa_signal_wait_relaxed(
        completion_signal_1, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Wait on the completion signal
    hsa_signal_wait_relaxed(
        completion_signal_2, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Wait on the completion signal
    hsa_signal_wait_relaxed(
        completion_signal_3, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    for(int i = 0; i < 64; i++)
    {
        if(a[i] != b[i])
        {
            printf("error at %d: expected %d, got %d\n", i, b[i], a[i]);
            abort();
        }
    }

    // Clearing data structures and memory
    status = hsa_signal_destroy(completion_signal_1);
    RET_IF_HSA_ERR(status)

    status = hsa_signal_destroy(completion_signal_2);
    RET_IF_HSA_ERR(status)

    status = hsa_signal_destroy(completion_signal_3);
    RET_IF_HSA_ERR(status)

    if(queue1 != nullptr)
    {
        status = hsa_queue_destroy(queue1);
        RET_IF_HSA_ERR(status)
    }

    if(queue2 != nullptr)
    {
        status = hsa_queue_destroy(queue2);
        RET_IF_HSA_ERR(status)
    }

    status = hsa_memory_free(a);
    RET_IF_HSA_ERR(status)

    status = hsa_memory_free(b);
    RET_IF_HSA_ERR(status)

    status = hsa_executable_destroy(code_object.executable);
    RET_IF_HSA_ERR(status)

    status = hsa_code_object_reader_destroy(code_object.code_obj_rdr);
    RET_IF_HSA_ERR(status)

    close(code_object.file);
}
