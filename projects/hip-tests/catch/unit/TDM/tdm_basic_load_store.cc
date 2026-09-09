#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <hip/amd_detail/amd_gfx1250_TDM.h>

#include <string>

#if defined(__clang__) && defined(__HIP__)
typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));
__global__ void  TDM_load_store_tester([[maybe_unused]] const int* data,
                                       [[maybe_unused]] int* result,
                                       [[maybe_unused]] int sizex,
                                       [[maybe_unused]] int sizey)
{
    #if defined(__gfx1250__) || defined(__gfx1251__)
    __shared__ int shmem[10 * 10];
    auto* pShmem = static_cast<int*>(shmem);
    gfx1250_TDM_GROUP0 group0;
    group0.globalAddr((uintptr_t)data);
    group0.ldsAddr((uintptr_t)pShmem);

    gfx1250_TDM_GROUP1 group1;
    group1.dataSize(2);
    group1.tensorDim0(sizex);
    group1.tensorDim1(sizey);
    group1.tensorDim0Stride(sizex);
    group1.tensorDim1Stride(sizey);
    group1.tileDim0(sizex);
    group1.tileDim1(sizey);

    v4i v4i_zeros{0, 0, 0, 0};
    v8i v8i_zeros{0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(group0.m_bitfield, group1.m_bitfield, v4i_zeros, v4i_zeros, v8i_zeros, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __syncthreads();

    // write back to global

    group0.globalAddr((uintptr_t)result);
    __builtin_amdgcn_tensor_store_from_lds(group0.m_bitfield, group1.m_bitfield, v4i_zeros, v4i_zeros, v8i_zeros, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);

    #endif // #if defined(__gfx1250__) || defined(__gfx1251__)
}

// General N-D (rank 2..5) load-to-LDS / store-from-LDS round trip. Rank is inferred by
// hardware from the highest nonzero tile_dim field, so unused dims (e2/e3/e4 for lower
// ranks) are simply left at their default-constructed zero value.
template <int Rank, int TotalElems>
__global__ void TDM_load_store_tester_nd([[maybe_unused]] const int* data,
                                          [[maybe_unused]] int* result,
                                          [[maybe_unused]] int e0, [[maybe_unused]] int e1,
                                          [[maybe_unused]] int e2, [[maybe_unused]] int e3,
                                          [[maybe_unused]] int e4)
{
    static_assert(Rank >= 2 && Rank <= 5, "Rank must be 2..5");
    #if defined(__gfx1250__) || defined(__gfx1251__)
    __shared__ int shmem[TotalElems];
    auto* pShmem = static_cast<int*>(shmem);
    gfx1250_TDM_GROUP0 group0;
    group0.globalAddr((uintptr_t)data);
    group0.ldsAddr((uintptr_t)pShmem);

    gfx1250_TDM_GROUP1 group1;
    group1.dataSize(2);
    group1.tensorDim0(e0);
    group1.tensorDim1(e1);
    group1.tensorDim0Stride(e0);
    group1.tileDim0(e0);
    group1.tileDim1(e1);

    gfx1250_TDM_GROUP2 group2;
    gfx1250_TDM_GROUP3 group3;

    if constexpr (Rank >= 3) {
        group1.tensorDim1Stride((uint64_t)e0 * e1);
        group1.tileDim2(e2);
        group2.tensorDim2(e2);
    }
    if constexpr (Rank >= 4) {
        group2.tensorDim2Stride((uint64_t)e0 * e1 * e2);
        group2.tensorDim3(e3);
        group2.tileDim3(e3);
    }
    if constexpr (Rank >= 5) {
        group3.tensorDim3Stride((uint64_t)e0 * e1 * e2 * e3);
        group3.tensorDim4(e4);
        group3.m_tile_dim4 = e4;
    }

    v8i v8i_zeros{0, 0, 0, 0, 0, 0, 0, 0};
    __builtin_amdgcn_tensor_load_to_lds(group0.m_bitfield, group1.m_bitfield, group2.m_bitfield,
                                         group3.m_bitfield, v8i_zeros, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __syncthreads();

    // write back to global

    group0.globalAddr((uintptr_t)result);
    __builtin_amdgcn_tensor_store_from_lds(group0.m_bitfield, group1.m_bitfield, group2.m_bitfield,
                                            group3.m_bitfield, v8i_zeros, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);

    #endif // #if defined(__gfx1250__) || defined(__gfx1251__)
}

// Verifies that stride setters spanning the 48-bit lo/hi split (GROUP1's dim0/dim1
// strides, GROUP2's dim2 stride, GROUP3's dim3 stride) pack their upper bits into the
// correct SGPR without disturbing neighboring bitfields. Writes the raw SGPR words back
// to global memory instead of issuing a real TDM op, since exercising the upper stride
// bits with an actual transfer would require allocations spanning that address range.
__global__ void TDM_stride_encoding_tester([[maybe_unused]] uint32_t* out,
                                            [[maybe_unused]] uint64_t dim0_stride,
                                            [[maybe_unused]] uint64_t dim1_stride,
                                            [[maybe_unused]] uint64_t dim2_stride,
                                            [[maybe_unused]] uint64_t dim3_stride)
{
    #if defined(__gfx1250__) || defined(__gfx1251__)
    gfx1250_TDM_GROUP1 group1;
    group1.tensorDim0Stride(dim0_stride);
    group1.tensorDim1Stride(dim1_stride);

    gfx1250_TDM_GROUP2 group2;
    group2.tensorDim2Stride(dim2_stride);

    gfx1250_TDM_GROUP3 group3;
    group3.tensorDim3Stride(dim3_stride);

    for (int i = 0; i < 8; ++i) out[i] = group1.m_bitfield[i];
    for (int i = 0; i < 4; ++i) out[8 + i] = group2.m_bitfield[i];
    for (int i = 0; i < 4; ++i) out[12 + i] = group3.m_bitfield[i];
    #endif // #if defined(__gfx1250__) || defined(__gfx1251__)
}

static void SkipIfNotTDMCapable(const char* test_name) {
#if HT_AMD
    int device = 0;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props{};
    HIP_CHECK(hipGetDeviceProperties(&props, device));
    const std::string arch(props.gcnArchName);
    if (arch.find("gfx1250") == std::string::npos && arch.find("gfx1251") == std::string::npos) {
        HIP_SKIP_TEST((std::string(test_name) + " requires gfx1250 or gfx1251").c_str());
        return;
    }
#endif
}

TEST_CASE("TDM_Basic_load_2d")
{
    SkipIfNotTDMCapable("TDM_Basic_load_2d");
    constexpr int kAllocSize = 10 * 10;
    const auto alloc_size = kAllocSize * sizeof(int);

    LinearAllocGuard<int> input_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<int> input(LinearAllocs::hipHostMalloc, alloc_size);

    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<int> result(LinearAllocs::hipHostMalloc, alloc_size);
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, alloc_size));

    for(int i = 0; i < kAllocSize; ++i)
    {
        input.ptr()[i] = i;
    }

    HIP_CHECK(hipMemcpy(input_dev.ptr(), input.ptr(), alloc_size, hipMemcpyHostToDevice));
    TDM_load_store_tester<<<1, 32>>>(input_dev.ptr(), result_dev.ptr(), 10, 10);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(result.ptr(), result_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));

    for(int i = 0; i < kAllocSize; ++i)
    {
        REQUIRE(result.ptr()[i] == input.ptr()[i]);
    }
}

template <int Rank, int TotalElems>
static void RunTdmNdTest(const char* test_name, int e0, int e1, int e2 = 1, int e3 = 1,
                         int e4 = 1)
{
    SkipIfNotTDMCapable(test_name);
    const auto alloc_size = TotalElems * sizeof(int);

    LinearAllocGuard<int> input_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<int> input(LinearAllocs::hipHostMalloc, alloc_size);

    LinearAllocGuard<int> result_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<int> result(LinearAllocs::hipHostMalloc, alloc_size);
    HIP_CHECK(hipMemset(result_dev.ptr(), 0, alloc_size));

    for(int i = 0; i < TotalElems; ++i)
    {
        input.ptr()[i] = i;
    }

    HIP_CHECK(hipMemcpy(input_dev.ptr(), input.ptr(), alloc_size, hipMemcpyHostToDevice));
    TDM_load_store_tester_nd<Rank, TotalElems><<<1, 32>>>(input_dev.ptr(), result_dev.ptr(), e0,
                                                           e1, e2, e3, e4);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(result.ptr(), result_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));

    for(int i = 0; i < TotalElems; ++i)
    {
        REQUIRE(result.ptr()[i] == input.ptr()[i]);
    }
}

TEST_CASE("TDM_Basic_load_3d")
{
    RunTdmNdTest<3, 4 * 5 * 3>("TDM_Basic_load_3d", 4, 5, 3);
}

TEST_CASE("TDM_Basic_load_4d")
{
    RunTdmNdTest<4, 3 * 4 * 2 * 5>("TDM_Basic_load_4d", 3, 4, 2, 5);
}

TEST_CASE("TDM_Basic_load_5d")
{
    RunTdmNdTest<5, 2 * 3 * 2 * 2 * 3>("TDM_Basic_load_5d", 2, 3, 2, 2, 3);
}

TEST_CASE("TDM_Stride_Encoding")
{
    SkipIfNotTDMCapable("TDM_Stride_Encoding");

    // dim0/dim2/dim3 strides split as 32-bit lo + 16-bit hi: exceed UINT32_MAX so the hi
    // half is exercised. dim1 stride splits as 16-bit lo + 32-bit hi: exceed UINT16_MAX.
    constexpr uint64_t kDim0Stride = (uint64_t{0xBEEF} << 32) | 0xDEADBEEFu;
    constexpr uint64_t kDim1Stride = (uint64_t{0xDEADBEEF} << 16) | 0xCAFEu;
    constexpr uint64_t kDim2Stride = (uint64_t{0xABCD} << 32) | 0x12345678u;
    constexpr uint64_t kDim3Stride = (uint64_t{0x5678} << 32) | 0x9ABCDEF0u;

    constexpr int kOutWords = 16;
    const auto alloc_size = kOutWords * sizeof(uint32_t);
    LinearAllocGuard<uint32_t> out_dev(LinearAllocs::hipMalloc, alloc_size);
    LinearAllocGuard<uint32_t> out(LinearAllocs::hipHostMalloc, alloc_size);

    TDM_stride_encoding_tester<<<1, 1>>>(out_dev.ptr(), kDim0Stride, kDim1Stride, kDim2Stride,
                                          kDim3Stride);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(out.ptr(), out_dev.ptr(), alloc_size, hipMemcpyDeviceToHost));

    uint32_t expected[kOutWords] = {};
    // GROUP1 (D1): sgpr5 = dim0_stride_lo, sgpr6 = dim0_stride_hi | dim1_stride_lo<<16,
    // sgpr7 = dim1_stride_hi
    expected[5] = static_cast<uint32_t>(kDim0Stride & 0xFFFFFFFFu);
    expected[6] = static_cast<uint32_t>(((kDim0Stride >> 32) & 0xFFFFu) |
                                         ((kDim1Stride & 0xFFFFu) << 16));
    expected[7] = static_cast<uint32_t>(kDim1Stride >> 16);
    // GROUP2 (D2): sgpr2 = dim2_stride_lo, sgpr3 = dim2_stride_hi | tile_dim3(0)<<16
    expected[8 + 2] = static_cast<uint32_t>(kDim2Stride & 0xFFFFFFFFu);
    expected[8 + 3] = static_cast<uint32_t>((kDim2Stride >> 32) & 0xFFFFu);
    // GROUP3 (D3): sgpr0 = dim3_stride_lo, sgpr1 = dim3_stride_hi | tensor_dim4_lo(0)<<16
    expected[12 + 0] = static_cast<uint32_t>(kDim3Stride & 0xFFFFFFFFu);
    expected[12 + 1] = static_cast<uint32_t>((kDim3Stride >> 32) & 0xFFFFu);

    for (int i = 0; i < kOutWords; ++i)
    {
        REQUIRE(out.ptr()[i] == expected[i]);
    }
}
#endif // #if defined(__clang__) && defined(__HIP__)
