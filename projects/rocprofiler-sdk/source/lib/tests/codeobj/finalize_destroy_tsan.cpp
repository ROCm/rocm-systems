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

#include "lib/rocprofiler-sdk/code_object/code_object.hpp"

#include <gtest/gtest.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <hsa/hsa_ven_amd_loader.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <thread>

namespace
{
using namespace std::chrono_literals;

std::atomic<rocprofiler_attach_code_object_cb_t> code_object_callback{nullptr};
std::atomic<bool>                                teardown_enabled{false};
std::atomic<bool>                                destroy_entered{false};
std::atomic<bool>                                finalize_started{false};
std::atomic<bool>                                first_destroy_is_writing{false};
std::atomic<bool>                                release_destroy{false};
std::atomic<uint32_t>                            teardown_calls{0};

// This is intentionally non-atomic. With the finalize mutex, the two shutdown calls are
// ordered by that mutex. If the mutex is removed, the fake loader holds the first shutdown
// open until finalize enters the second one, and TSan reports the concurrent accesses here.
volatile uint64_t shutdown_sentinel = 0;

int
add_code_object_callback(rocprofiler_attach_code_object_cb_t callback, void*)
{
    code_object_callback.store(callback, std::memory_order_release);
    return 0;
}

hsa_status_t
iterate_loaded_code_objects(hsa_executable_t,
                            hsa_status_t (*)(hsa_executable_t,
                                             hsa_loaded_code_object_t,
                                             void*),
                            void*)
{
    if(!teardown_enabled.load(std::memory_order_acquire)) return HSA_STATUS_SUCCESS;

    const auto call = teardown_calls.fetch_add(1, std::memory_order_acq_rel);
    if(call == 0)
    {
        destroy_entered.store(true, std::memory_order_release);
        while(!finalize_started.load(std::memory_order_acquire))
            std::this_thread::yield();

        first_destroy_is_writing.store(true, std::memory_order_release);
        while(!release_destroy.load(std::memory_order_acquire))
            ++shutdown_sentinel;
    }
    else
    {
        while(!first_destroy_is_writing.load(std::memory_order_acquire))
            std::this_thread::yield();

        for(uint32_t i = 0; i < 100000; ++i)
            ++shutdown_sentinel;

        release_destroy.store(true, std::memory_order_release);
    }

    return HSA_STATUS_SUCCESS;
}

hsa_status_t
get_major_extension_table(uint16_t extension,
                          uint16_t version_major,
                          size_t   table_length,
                          void*    table)
{
    if(extension != HSA_EXTENSION_AMD_LOADER || version_major != 1 || table == nullptr)
        return HSA_STATUS_ERROR_INVALID_ARGUMENT;

    auto loader_table = hsa_ven_amd_loader_1_01_pfn_t{};
    loader_table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects =
        iterate_loaded_code_objects;
    std::memcpy(table, &loader_table, std::min(table_length, sizeof(loader_table)));
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
status_string(hsa_status_t, const char** value)
{
    static constexpr auto message = "test HSA status";
    if(value) *value = message;
    return HSA_STATUS_SUCCESS;
}

bool
wait_for(const std::atomic<bool>& value, std::chrono::milliseconds timeout)
{
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    while(!value.load(std::memory_order_acquire))
    {
        if(std::chrono::steady_clock::now() >= deadline) return false;
        std::this_thread::yield();
    }
    return true;
}
}  // namespace

TEST(code_object_finalize_tsan, serializes_with_executable_destroy)
{
    auto attach_table                                      = RocAttachDispatchTable{};
    attach_table.rocprofiler_attach_add_code_object_cb     = add_code_object_callback;
    auto hsa_tables                                        = HsaApiTableContainer{};
    hsa_tables.core.hsa_status_string_fn                   = status_string;
    hsa_tables.core.hsa_system_get_major_extension_table_fn = get_major_extension_table;

    rocprofiler::code_object::initialize(&attach_table);
    rocprofiler::code_object::initialize(&hsa_tables.root);

    auto callback = code_object_callback.load(std::memory_order_acquire);
    ASSERT_NE(callback, nullptr);

    constexpr auto executable = hsa_executable_t{0x1234};
    callback(executable, ROCPROFILER_ATTACH_CODE_OBJECT_CREATED, nullptr);

    teardown_enabled.store(true, std::memory_order_release);
    auto destroy_thread = std::thread{[&]() {
        callback(executable, ROCPROFILER_ATTACH_CODE_OBJECT_DESTROYED, nullptr);
    }};

    if(!wait_for(destroy_entered, 5s))
    {
        finalize_started.store(true, std::memory_order_release);
        release_destroy.store(true, std::memory_order_release);
        destroy_thread.join();
        FAIL() << "executable destroy did not enter code-object shutdown";
    }

    auto finalize_thread = std::thread{[]() {
        finalize_started.store(true, std::memory_order_release);
        rocprofiler::code_object::finalize();
    }};

    const auto deadline = std::chrono::steady_clock::now() + 1s;
    while(teardown_calls.load(std::memory_order_acquire) == 1 &&
          std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();

    release_destroy.store(true, std::memory_order_release);
    destroy_thread.join();
    finalize_thread.join();

    // With serialization, destroy removes the executable before finalize examines the list,
    // so shutdown is entered exactly once. Without the finalize mutex, the count becomes two;
    // the overlapping sentinel accesses above also give TSan a direct regression signal.
    EXPECT_EQ(teardown_calls.load(std::memory_order_acquire), 1);
}
