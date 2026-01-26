#include <hip_test_common.hh>
#include <resource_guards.hh>
#include <hip/amd_detail/amd_gfx1250_TDM.h>

#if defined(__clang__) && defined(__HIP__)
__global__ void  TDM_load_store_tester(const int* data,  int* result, int sizex, int sizey)
{
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

    __builtin_amdgcn_tensor_load_to_lds_d2(group0.m_bitfield, group1.m_bitfield, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);
    __syncthreads();

    // write back to global

    group0.globalAddr((uintptr_t)result);
    __builtin_amdgcn_tensor_store_from_lds_d2(group0.m_bitfield, group1.m_bitfield, 0);
    __builtin_amdgcn_s_wait_tensorcnt(0);

}

TEST_CASE("TDM_Basic_load_2d")
{
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

    for(int i =0; i < alloc_size;++i)
    {
        REQUIRE(result.ptr()[i] == input.ptr()[i]);
    }
}
#endif // #if defined(__clang__) && defined(__HIP__)