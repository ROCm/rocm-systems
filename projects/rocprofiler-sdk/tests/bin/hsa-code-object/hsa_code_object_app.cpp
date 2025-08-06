// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "hsa_code_object_app.h"

enum class storage_type
{
    CODE_OBJECT_STORAGE_FILE,
    CODE_OBJECT_STORAGE_MEMORY
};

void
code_object_load(MQDependencyTest&             obj,
                 storage_type                  type,
                 MQDependencyTest::CodeObject& code_object)
{
    hsa_status_t status;
    obj.device_discovery();
    char agent_name[64];
    status = hsa_agent_get_info(obj.gpu[0].agent, HSA_AGENT_INFO_NAME, agent_name);
    RET_IF_HSA_ERR(status)

    if(type == storage_type::CODE_OBJECT_STORAGE_FILE)
    {
        std::string hasco_file_path = std::string(agent_name) + std::string("_copy.hsaco");
        obj.search_hasco(fs::current_path(), hasco_file_path);
        if(!obj.load_code_object(hasco_file_path, obj.gpu[0].agent, code_object))
        {
            printf("Kernel file not found or not usable with given agent.\n");
            abort();
        }
    }
    else
    {
        std::string hasco_file_path = std::string(agent_name) + std::string("_copy_memory.hsaco");
        obj.search_hasco(fs::current_path(), hasco_file_path);
        if(!obj.load_code_object_memory(hasco_file_path, obj.gpu[0].agent, code_object))
        {
            abort();
        }
    }
}

MQDependencyTest::Kernel
get_kernel(MQDependencyTest::CodeObject& code_object,
           std::string                   kernel_name,
           MQDependencyTest&             obj)
{
    MQDependencyTest::Kernel copy;
    if(!obj.get_kernel(code_object, kernel_name, obj.gpu[0].agent, copy))
    {
        printf("Test %s not found.\n", kernel_name.c_str());
        abort();
    }
    return copy;
}

hsa_signal_t
create_completion_signal()
{
    hsa_signal_t signal = {};
    hsa_status_t status = hsa_signal_create(1, 0, nullptr, &signal);
    RET_IF_HSA_ERR(status)
    return signal;
}

hsa_queue_t*
create_queue(hsa_agent_t agent)
{
    hsa_queue_t* queue  = nullptr;
    hsa_status_t status = hsa_queue_create(
        agent, 1024, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX, UINT32_MAX, &queue);
    RET_IF_HSA_ERR(status)
    return queue;
}

void
submit_kernel_packet(MQDependencyTest&               obj,
                     hsa_queue_t*                    queue,
                     const MQDependencyTest::Kernel& kernel,
                     void*                           args,
                     hsa_signal_t                    completion_signal)
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

    packet.dispatch.group_segment_size   = kernel.group;
    packet.dispatch.private_segment_size = kernel.scratch;
    packet.dispatch.kernel_object        = kernel.handle;
    packet.dispatch.kernarg_address      = args;
    packet.dispatch.completion_signal    = completion_signal;

    obj.submit_packet(queue, packet);
}

void
submit_barrier_packet(MQDependencyTest& obj, hsa_queue_t* queue, hsa_signal_t dependency_signal)
{
    MQDependencyTest::Aql packet{};
    packet.header.type    = HSA_PACKET_TYPE_BARRIER_AND;
    packet.header.barrier = 1;
    packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
    packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

    packet.barrier_and.dep_signal[0] = dependency_signal;
    obj.submit_packet(queue, packet);
}

/**
 * Expected Execution Pattern with Serialization:
 *
 * This test validates that the profiler's serialization mechanism can handle
 * inter-queue dependencies without deadlock. The execution should follow this pattern:
 *
 * Phase 1:
 *   Queue1: Kernel A executes → sets signal_1 = 0
 *   Queue1: Barrier blocks (waiting for signal_2)
 *   [Serializer switches to Queue2]
 *   Queue2: Barrier proceeds (signal_1 = 0) → Kernel B executes → sets signal_2 = 0
 *   [Serializer switches back to Queue1]
 *   Queue1: Barrier proceeds (signal_2 = 0) → Kernel C executes
 *
 * Phase 2:
 *   Queue1: Kernel D executes → sets signal_4 = 0
 *   Queue1: Barrier blocks (waiting for signal_5)
 *   [Serializer switches to Queue2]
 *   Queue2: Barrier proceeds (signal_4 = 0) → Kernel E executes → sets signal_5 = 0
 *   [Serializer switches back to Queue1]
 *   Queue1: Barrier proceeds (signal_5 = 0) → Kernel F executes
 *
 * Key: Dependencies flow forward without cycles, allowing the serializer to make
 * progress by switching between queues when one blocks on a barrier.
 */
int
main()
{
    hsa_status_t                 status;
    MQDependencyTest             obj;
    MQDependencyTest             obj_memory  = {};
    MQDependencyTest::CodeObject code_object = {}, code_object_memory = {};

    code_object_load(obj, storage_type::CODE_OBJECT_STORAGE_FILE, code_object);
    code_object_load(obj_memory, storage_type::CODE_OBJECT_STORAGE_MEMORY, code_object_memory);

    MQDependencyTest::Kernel copyA = get_kernel(code_object, "copyA", obj);
    MQDependencyTest::Kernel copyB = get_kernel(code_object, "copyB", obj);
    MQDependencyTest::Kernel copyC = get_kernel(code_object, "copyC", obj);

    MQDependencyTest::Kernel copyD = get_kernel(code_object_memory, "copyD", obj_memory);
    MQDependencyTest::Kernel copyE = get_kernel(code_object_memory, "copyE", obj_memory);
    MQDependencyTest::Kernel copyF = get_kernel(code_object_memory, "copyF", obj_memory);

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

    args_t* args_memory =
        static_cast<args_t*>(obj_memory.hsa_malloc(sizeof(args_t), obj_memory.kernarg));
    *args_memory = {};

    uint32_t* c =
        static_cast<uint32_t*>(obj_memory.hsa_malloc(64 * sizeof(uint32_t), obj_memory.kernarg));
    uint32_t* d =
        static_cast<uint32_t*>(obj_memory.hsa_malloc(64 * sizeof(uint32_t), obj_memory.kernarg));

    memset(c, 0, 64 * sizeof(uint32_t));
    memset(d, 1, 64 * sizeof(uint32_t));

    // Create queues
    hsa_queue_t* queue1 = create_queue(obj.gpu[0].agent);
    hsa_queue_t* queue2 = create_queue(obj.gpu[0].agent);

    // Create completion signals
    hsa_signal_t completion_signal_1 = create_completion_signal();

    // Set up arguments for first batch
    args->a = a;
    args->b = b;

    // Create more completion signals
    hsa_signal_t completion_signal_2 = create_completion_signal();
    hsa_signal_t completion_signal_3 = create_completion_signal();

    // First dispatch packet on queue 1, Kernel A
    submit_kernel_packet(obj, queue1, copyA, args, completion_signal_1);

    // Barrier on queue 1 waiting for signal_2 (from queue2's Kernel B)
    submit_barrier_packet(obj, queue1, completion_signal_2);

    // Barrier on queue 2 waiting for signal_1 (from queue1's Kernel A)
    submit_barrier_packet(obj, queue2, completion_signal_1);

    // Kernel B on queue 2 (waits for barrier above)
    submit_kernel_packet(obj, queue2, copyB, args, completion_signal_2);

    // Second dispatch packet on queue 1, Kernel C (waits for barrier above)
    submit_kernel_packet(obj, queue1, copyC, args, completion_signal_3);

    // Set up arguments for second batch
    args_memory->a = c;
    args_memory->b = d;

    // Create signals for second batch
    hsa_signal_t completion_signal_4 = create_completion_signal();
    hsa_signal_t completion_signal_5 = create_completion_signal();
    hsa_signal_t completion_signal_6 = create_completion_signal();
    // Second batch: Kernel D on queue 1
    submit_kernel_packet(obj_memory, queue1, copyD, args_memory, completion_signal_4);

    // Barrier on queue 1 waiting for signal_5 (from queue2's Kernel E)
    submit_barrier_packet(obj_memory, queue1, completion_signal_5);

    // Barrier on queue 2 waiting for signal_4 (from queue1's Kernel D)
    submit_barrier_packet(obj_memory, queue2, completion_signal_4);

    // Kernel E on queue 2 (waits for barrier above)
    submit_kernel_packet(obj_memory, queue2, copyE, args_memory, completion_signal_5);

    // Kernel F on queue 1 (waits for barrier above)
    submit_kernel_packet(obj_memory, queue1, copyF, args_memory, completion_signal_6);

    // Wait on the completion signal
    hsa_signal_wait_relaxed(
        completion_signal_1, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Wait on the completion signal
    hsa_signal_wait_relaxed(
        completion_signal_2, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Wait on the completion signal
    hsa_signal_wait_relaxed(
        completion_signal_3, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Wait on the completion signal
    hsa_signal_wait_relaxed(
        completion_signal_4, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Wait on the completion signal
    hsa_signal_wait_relaxed(
        completion_signal_5, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    // Wait on the completion signal
    hsa_signal_wait_relaxed(
        completion_signal_6, HSA_SIGNAL_CONDITION_EQ, 0, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

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

    // Clearing data structures and memory
    status = hsa_signal_destroy(completion_signal_4);
    RET_IF_HSA_ERR(status)

    status = hsa_signal_destroy(completion_signal_5);
    RET_IF_HSA_ERR(status)

    status = hsa_signal_destroy(completion_signal_6);
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

    status = hsa_memory_free(c);
    RET_IF_HSA_ERR(status)

    status = hsa_memory_free(d);
    RET_IF_HSA_ERR(status)

    status = hsa_executable_destroy(code_object.executable);
    RET_IF_HSA_ERR(status)

    status = hsa_code_object_reader_destroy(code_object.code_obj_rdr);
    RET_IF_HSA_ERR(status)

    status = hsa_executable_destroy(code_object_memory.executable);
    RET_IF_HSA_ERR(status)

    status = hsa_code_object_reader_destroy(code_object_memory.code_obj_rdr);
    RET_IF_HSA_ERR(status)

    close(code_object.file);

    close(code_object_memory.file);
}
