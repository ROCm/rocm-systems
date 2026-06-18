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

#include "lib/rocprofiler-sdk/hsa/aql_packet.hpp"

#include <gtest/gtest.h>

#include <cstdint>

using namespace rocprofiler::hsa;

namespace
{
struct MockState
{
    void*        last_freed_ptr  = nullptr;
    size_t       last_alloc_size = 0;
    void*        alloc_out       = nullptr;
    hsa_status_t alloc_return    = HSA_STATUS_SUCCESS;
    void*        last_copy_dst   = nullptr;
    const void*  last_copy_src   = nullptr;
    size_t       last_copy_size  = 0;
};

MockState&
mock_state()
{
    static MockState s{};
    return s;
}

void
reset_mock_state()
{
    mock_state() = MockState{};
}

hsa_status_t
mock_free(void* ptr)
{
    mock_state().last_freed_ptr = ptr;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
mock_allocate(hsa_amd_memory_pool_t, size_t size, uint32_t, void** ptr)
{
    mock_state().last_alloc_size = size;
    *ptr                         = &mock_state().alloc_out;
    return mock_state().alloc_return;
}

hsa_status_t
mock_copy(void* dst, const void* src, size_t size)
{
    mock_state().last_copy_dst  = dst;
    mock_state().last_copy_src  = src;
    mock_state().last_copy_size = size;
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
mock_allow_access(uint32_t, const hsa_agent_t*, const uint32_t*, const void*)
{
    return HSA_STATUS_SUCCESS;
}

hsa_status_t
mock_fill(void*, uint32_t, size_t)
{
    return HSA_STATUS_SUCCESS;
}

void
make_pool_valid(SPMMemoryPool& pool)
{
    pool.free_fn     = &mock_free;
    pool.allocate_fn = &mock_allocate;
    pool.api_copy_fn = &mock_copy;
    pool.cpu_pool_   = {.handle = 1};
}

}  // namespace

// ---------------------------------------------------------------------------
// Free tests
// ---------------------------------------------------------------------------

// Freeing a null pointer should be a safe no-op (free_fn must not be called)
TEST(spm_memory_pool, free_nullptr)
{
    reset_mock_state();
    auto pool = SPMMemoryPool{};
    make_pool_valid(pool);
    SPMMemoryPool::Free(nullptr, &pool);
    EXPECT_EQ(mock_state().last_freed_ptr, nullptr);
}

// Freeing a valid pointer should delegate to the pool's free_fn
TEST(spm_memory_pool, free_validptr)
{
    reset_mock_state();
    auto pool = SPMMemoryPool{};
    make_pool_valid(pool);
    int dummy = 42;
    SPMMemoryPool::Free(&dummy, &pool);
    EXPECT_EQ(mock_state().last_freed_ptr, &dummy);
}

// ---------------------------------------------------------------------------
// Alloc tests
// ---------------------------------------------------------------------------

// Zero-size allocation with a valid output pointer should set the pointer to null
TEST(spm_memory_pool, alloc_zero_size)
{
    reset_mock_state();
    auto pool = SPMMemoryPool{};
    make_pool_valid(pool);
    int                            dummy = 0;
    void*                          out   = &dummy;
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = 1;

    auto status = SPMMemoryPool::Alloc(&out, 0, flags, &pool);
    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
    EXPECT_EQ(out, nullptr);
}

// Zero-size allocation with a null output pointer should succeed without side effects
TEST(spm_memory_pool, alloc_zero_size_nullptr)
{
    auto status = SPMMemoryPool::Alloc(nullptr, 0, {}, nullptr);
    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
}

// Non-zero allocation with a null output pointer is an invalid call
TEST(spm_memory_pool, alloc_nullptr)
{
    auto                           pool = SPMMemoryPool{};
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = 1;

    auto status = SPMMemoryPool::Alloc(nullptr, 1024, flags, &pool);
    EXPECT_EQ(status, HSA_STATUS_ERROR);
}

// Non-zero allocation with a null pool (data) pointer is an invalid call
TEST(spm_memory_pool, alloc_null_data)
{
    void*                          out = nullptr;
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = 1;

    auto status = SPMMemoryPool::Alloc(&out, 1024, flags, nullptr);
    EXPECT_EQ(status, HSA_STATUS_ERROR);
}

// Allocation without host_access flag set should be rejected
TEST(spm_memory_pool, alloc_nohostaccess)
{
    auto pool = SPMMemoryPool{};
    make_pool_valid(pool);
    pool.allow_access_fn = &mock_allow_access;
    pool.fill_fn         = &mock_fill;

    void*                          out = nullptr;
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = 0;

    auto status = SPMMemoryPool::Alloc(&out, 1024, flags, &pool);
    EXPECT_EQ(status, HSA_STATUS_ERROR);
}

// Allocation should fail if the pool has no allocate_fn set
TEST(spm_memory_pool, alloc_missingallocatefn)
{
    auto pool = SPMMemoryPool{};
    make_pool_valid(pool);
    pool.allocate_fn     = nullptr;
    pool.allow_access_fn = &mock_allow_access;
    pool.fill_fn         = &mock_fill;

    void*                          out = nullptr;
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = 1;

    auto status = SPMMemoryPool::Alloc(&out, 1024, flags, &pool);
    EXPECT_EQ(status, HSA_STATUS_ERROR);
}

// Allocation should fail if the pool has no free_fn set
TEST(spm_memory_pool, alloc_missingfreefn)
{
    auto pool = SPMMemoryPool{};
    make_pool_valid(pool);
    pool.free_fn         = nullptr;
    pool.allow_access_fn = &mock_allow_access;
    pool.fill_fn         = &mock_fill;

    void*                          out = nullptr;
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = 1;

    auto status = SPMMemoryPool::Alloc(&out, 1024, flags, &pool);
    EXPECT_EQ(status, HSA_STATUS_ERROR);
}

// Valid allocation should succeed and forward the requested size to allocate_fn
TEST(spm_memory_pool, alloc)
{
    reset_mock_state();
    auto pool = SPMMemoryPool{};
    make_pool_valid(pool);
    pool.allow_access_fn = &mock_allow_access;
    pool.fill_fn         = &mock_fill;

    void*                          out = nullptr;
    aqlprofile_buffer_desc_flags_t flags{};
    flags.host_access = 1;

    auto status = SPMMemoryPool::Alloc(&out, 4096, flags, &pool);
    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
    EXPECT_NE(out, nullptr);
    EXPECT_EQ(mock_state().last_alloc_size, 4096u);
}

// ---------------------------------------------------------------------------
// Copy tests
// ---------------------------------------------------------------------------

// Zero-size copy should succeed without requiring valid pointers or pool
TEST(spm_memory_pool, copy_zero_size)
{
    auto status = SPMMemoryPool::Copy(nullptr, nullptr, 0, nullptr);
    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
}

// Valid copy should delegate to the pool's api_copy_fn with correct arguments
TEST(spm_memory_pool, copy)
{
    reset_mock_state();
    auto pool = SPMMemoryPool{};
    make_pool_valid(pool);
    char dst[16] = {};
    char src[16] = "hello";

    auto status = SPMMemoryPool::Copy(dst, src, 16, &pool);
    EXPECT_EQ(status, HSA_STATUS_SUCCESS);
    EXPECT_EQ(mock_state().last_copy_dst, dst);
    EXPECT_EQ(mock_state().last_copy_src, src);
    EXPECT_EQ(mock_state().last_copy_size, 16u);
}
