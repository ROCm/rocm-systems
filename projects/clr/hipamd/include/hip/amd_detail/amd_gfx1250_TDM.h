/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_INCLUDE_HIP_AMD_GFX1250_TDM_H
#define HIP_INCLUDE_HIP_AMD_GFX1250_TDM_H

#if !defined(__HIPCC_RTC__)
#include <hip/hip_common.h>
#include <hip/driver_types.h>
#include <hip/amd_detail/amd_hip_vector_types.h>
#if __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif
#endif

namespace gfx1250
{

using __hip_uint32x4 = __NATIVE_VECTOR__(4, uint32_t);
using __hip_uint32x8 = __NATIVE_VECTOR__(8, uint32_t);
using __hip_int32x4 = __NATIVE_VECTOR__(4, int32_t);
using __hip_int32x8 = __NATIVE_VECTOR__(8, int32_t);


#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wshift-count-overflow" // we're going to be using overflow on purpose


union TDM_GROUP0
{
    __device__ constexpr TDM_GROUP0() : m_bitfield{0, 0, 0, 0}
    {
        m_count             = 1;
        m_type              = 2;
    }

    __device__ TDM_GROUP0( uintptr_t lds_addr_in,
                           uintptr_t global_addr_in,
                           uint32_t  gather_idx_size_in = 0,
                           uint32_t  gather_mode_in = 0 ): m_bitfield{0, 0, 0, 0}
    {
        globalAddr(global_addr_in);
        m_count             = 1;
        m_lds_addr          = lds_addr_in;
        m_type              = 2;
        m_gather_index_size = gather_idx_size_in;
        m_gather_mode       = gather_mode_in;
    }

    struct
    {
        //reserve the first 32 bits
        union
        {
            struct
            {
                uint32_t m_count : 2;
                uint32_t m_is_restore : 1;
                uint32_t m_is_store : 1;
                uint32_t m_nv : 1;
                uint32_t m_scope_trait : 2;
                uint32_t m_th :3;
                uint32_t m_reserved_space : 20;
                uint32_t m_gather_mode : 1;
                uint32_t m_gather_index_size : 1;
            };

            uint32_t m_reserved0;
        };

        uint32_t m_lds_addr;
        uint32_t m_global_addr_lo;
        union
        {
            struct
            {
                uint32_t m_global_addr_hi : 25;
                uint32_t m_reserved2  : 5;
                uint32_t m_type       : 2;
            };
            uint32_t m_sgpr3;
        };
    };
    __hip_uint32x4 m_bitfield;

    // setters for all user configurable fields
    __device__ void count(bool value)
    {
        m_count = (int)value;
    }

    __device__ void ldsAddr(uint32_t value)
    {
        m_lds_addr = value;
    }

    __device__ void gatherMode(int enable, int value)
    {
        m_gather_mode = enable;
        m_gather_index_size = value;
    }

    __device__ void globalAddr(uintptr_t value)
    {
        m_global_addr_lo = value & 0xFFFFFFFF;
        m_global_addr_hi = (value >> 32);
    }
};

union TDM_GROUP1
{
    __device__ constexpr TDM_GROUP1() : m_bitfield{0,0,0,0,0,0,0,0} {}

    struct
    {
        union
        {
            struct
            {
                uint32_t m_workgroup_mask         : 16;
                uint32_t m_data_size              : 2;
                uint32_t m_atomic_barrier_enable  : 1;
                uint32_t m_iterate_enable         : 1;
                uint32_t m_pad_enable             : 1;
                uint32_t m_early_timeout          : 1;
                uint32_t m_pad_interval           : 3;
                uint32_t m_pad_amount             : 7;
            };
            uint32_t m_sgpr0;
        };
        union
        {
            struct
            {
                uint32_t m_atomic_barrier_address  : 16;
                uint32_t m_tensor_dim0_lo          : 16;
            };
            uint32_t m_sgpr1;
        };
        union
        {
            struct
            {
                uint32_t m_tensor_dim0_hi : 16;
                uint32_t m_tensor_dim1_lo : 16;
            };
            uint32_t m_sgpr2;
        };
        union
        {
            struct
            {
                uint32_t m_tensor_dim1_hi : 16;
                uint32_t m_tile_dim0      : 16;
            };
            uint32_t m_sgpr3;
        };
        union
        {
            struct
            {
                uint32_t m_tile_dim1 : 16;
                uint32_t m_tile_dim2 : 16;
            };
            uint32_t m_sgpr4;
        };
        union
        {
            uint32_t m_tensor_dim0_stride_lo;
            uint32_t m_sgpr5;
        };
        union
        {
            struct
            {
                uint32_t m_tensor_dim0_stride_hi : 16;
                uint32_t m_tensor_dim1_stride_lo : 16;
            };
            uint32_t m_sgpr6;
        };
        union
        {
            uint32_t m_tensor_dim1_stride_hi;
            uint32_t m_sgpr7;
        };
    };
    __hip_int32x8 m_bitfield;

    // setters for all fields
    void __device__ inline workgroupMask(uint32_t value)
    {
        m_workgroup_mask = value;
    }

    void __device__ inline dataSize(uint32_t value)
    {
        m_data_size = value;
    }

    void __device__ inline atomicBarrierEnable(bool value)
    {
        m_atomic_barrier_enable = value;
    }

    void __device__ inline iterateEnable(bool value)
    {
        m_iterate_enable = value;
    }

    void __device__ inline padEnable(bool value)
    {
        m_pad_enable = value;
    }

    void __device__ inline earlyTimeout(bool value)
    {
        m_early_timeout = value;
    }

    void __device__ inline padInterval(uint32_t value)
    {
        m_pad_interval = value;
    }

    void __device__ inline padAmount(uint32_t value)
    {
        m_pad_amount = value;
    }

    void __device__ inline atomicBarrierAddress(uint32_t value)
    {
        m_atomic_barrier_address = value;
    }

    void __device__ inline tileDim0(uint32_t value)
    {
        m_tile_dim0 = value;
    }

    void __device__ inline tileDim1(uint32_t value)
    {
        m_tile_dim1 = value;
    }
    void __device__ inline tileDim2(uint32_t value)
    {
        m_tile_dim2 = value;
    }

    void __device__ inline tensorDim0(uint32_t value)
    {
        m_tensor_dim0_lo = value & 0xFFFF;
        m_tensor_dim0_hi = (value >> 16);
    }
    void __device__ inline tensorDim1(uint32_t value)
    {
        m_tensor_dim1_lo = value & 0xFFFF;
        m_tensor_dim1_hi = (value >> 16);
    }
    void __device__ inline tensorDim0Stride(uint64_t value)
    {
        m_tensor_dim0_stride_lo = value & 0xFFFFFFFF;
        m_tensor_dim0_stride_hi = (value >> 32);
    }

    void __device__ inline tensorDim1Stride(uint64_t value)
    {
        m_tensor_dim1_stride_lo = value & 0xFFFF;
        m_tensor_dim1_stride_hi = (value >> 16);
    }
};

union TDM_GROUP2
{
    __device__ TDM_GROUP2() : m_bitfield{0,0,0,0} {}

    struct
    {
        uint32_t m_tensor_dim2; //sgpr0
        uint32_t m_tensor_dim3; //sgpr1
        uint32_t m_tensor_dim2_stride_lo; //sgpr2
        union
        {
            struct
            {
                uint32_t m_tensor_dim2_stride_hi : 16;
                uint32_t m_tile_dim3             : 16;
            };
            uint32_t m_sgpr3;
        };
    };
    __hip_int32x4 m_bitfield;

    void __device__ inline tensorDim2(uint32_t value)
    {
        m_tensor_dim2 = value;
    }

    void __device__ inline tensorDim3(uint32_t value)
    {
        m_tensor_dim3 = value;
    }

    void __device__ inline tensorDim2Stride(uint64_t value)
    {
        m_tensor_dim2_stride_lo = value & 0xFFFFFFFF;
        m_tensor_dim2_stride_hi = value >> 32;
    }

    void __device__ inline tileDim3(uint32_t value)
    {
        m_tile_dim3 = value;
    }
};

union TDM_GROUP3
{
    __device__ TDM_GROUP3() : m_bitfield{0,0,0,0} {}

    struct
    {
        uint32_t m_tensor_dim3_stride_lo; //sgpr0
        union
        {
            struct
            {
                uint32_t m_tensor_dim3_stride_hi : 16;
                uint32_t m_tensor_dim_4_lo : 16;
            };
            uint32_t m_sgpr1;
        };
        union
        {
            struct
            {
                uint32_t m_tensor_dim_4_hi : 16;
                uint32_t m_tile_dim4       : 16;
            };
            uint32_t m_sgpr2;
        };
        uint32_t m_sgpr3_reserved;
    };
    __hip_int32x4 m_bitfield;

    void __device__ inline tensorDim3Stride(uint64_t value)
    {
        m_tensor_dim3_stride_lo =  value & 0xFFFFFFFF;
        m_tensor_dim3_stride_hi = value >> 32;
    }

    void __device__ inline tensorDim4(uint32_t value)
    {
        m_tensor_dim_4_lo = value & 0xFFFF;
        m_tensor_dim_4_hi = value >> 16;
    }

    void __device__ inline tileDim4(uint32_t value)
    {
        m_tile_dim4 = value;
    }
};

struct TDM_DESCRIPTOR
{
    TDM_GROUP0 group0;
    TDM_GROUP1 group1;
    TDM_GROUP2 group2;
    TDM_GROUP3 group3;
};

union TDM_GATHER_GROUP2_32BIT
{
    __device__ TDM_GATHER_GROUP2_32BIT() : m_bitfield{0,0,0,0} {}

    __hip_int32x4 m_bitfield;

    void __device__ inline rowIndex(uint32_t slot, uint32_t value)
    {
        m_bitfield[slot] = static_cast<int32_t>(value);
    }
};

union TDM_GATHER_GROUP3_32BIT
{
    __device__ TDM_GATHER_GROUP3_32BIT() : m_bitfield{0,0,0,0} {}

    __hip_int32x4 m_bitfield;

    void __device__ inline rowIndex(uint32_t slot, uint32_t value)
    {
        m_bitfield[slot] = static_cast<int32_t>(value);
    }
};

// Descriptor layout from the hardware specification; currently untested.
union TDM_GATHER_GROUP2_16BIT
{
    __device__ TDM_GATHER_GROUP2_16BIT() : m_bitfield{0,0,0,0} {}

    __hip_int32x4 m_bitfield;

    void __device__ inline rowIndex(uint32_t slot, uint16_t value)
    {
        const uint32_t lane = slot / 2;
        const uint32_t shift = (slot % 2) * 16;
        const uint32_t packed = static_cast<uint32_t>(m_bitfield[lane]);
        m_bitfield[lane] = static_cast<int32_t>(
            (packed & ~(0xffffu << shift)) | (static_cast<uint32_t>(value) << shift));
    }
};

// Descriptor layout from the hardware specification; currently untested.
union TDM_GATHER_GROUP3_16BIT
{
    __device__ TDM_GATHER_GROUP3_16BIT() : m_bitfield{0,0,0,0} {}

    __hip_int32x4 m_bitfield;

    void __device__ inline rowIndex(uint32_t slot, uint16_t value)
    {
        const uint32_t lane = slot / 2;
        const uint32_t shift = (slot % 2) * 16;
        const uint32_t packed = static_cast<uint32_t>(m_bitfield[lane]);
        m_bitfield[lane] = static_cast<int32_t>(
            (packed & ~(0xffffu << shift)) | (static_cast<uint32_t>(value) << shift));
    }
};

template<typename Index>
using TDM_INDEX_GROUP2 = std::conditional_t<
    std::is_same_v<Index, uint16_t>,
    TDM_GATHER_GROUP2_16BIT,
    TDM_GATHER_GROUP2_32BIT>;

template<typename Index>
using TDM_INDEX_GROUP3 = std::conditional_t<
    std::is_same_v<Index, uint16_t>,
    TDM_GATHER_GROUP3_16BIT,
    TDM_GATHER_GROUP3_32BIT>;

template<typename Index>
struct TDM_INDEXED_DESCRIPTOR
{
    TDM_GROUP0 group0;
    TDM_GROUP1 group1;
    TDM_INDEX_GROUP2<Index> group2;
    TDM_INDEX_GROUP3<Index> group3;
};
} // namespace gfx1250

#pragma clang diagnostic pop
#endif // HIP_INCLUDE_HIP_AMD_GFX1250_TDM_H