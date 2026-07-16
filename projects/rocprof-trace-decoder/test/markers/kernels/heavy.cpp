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

#include <hip/hip_runtime.h>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <chrono>
#include <iostream>
#include <cstring>

#ifdef ENABLE_ROCTX
#include <rocprofiler-sdk-roctx/roctx.h>
#else
#define roctxProfilerResume(_x)
#define roctxProfilerPause(_x)
#endif

#include "markers.hpp"

#define HIP_API_CALL(CALL)                                                                         \
    {                                                                                              \
        hipError_t error_ = (CALL);                                                                \
        if(error_ != hipSuccess)                                                                   \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "%s:%d :: HIP error : %s\n",                                                   \
                    __FILE__,                                                                      \
                    __LINE__,                                                                      \
                    hipGetErrorString(error_));                                                    \
            throw std::runtime_error("hip_api_call");                                              \
        }                                                                                          \
    }

template<typename Type>
class Matrix
{
public:
    Matrix(int _rows, int _columns): rows(_rows), columns(_columns), memsize(_rows*_columns*sizeof(Type))
    {
        host = new Type[rows*columns];
        memset(host, 0, memsize);
        HIP_API_CALL(hipMalloc((void**)&dev, memsize));
    }

    ~Matrix()
    {
        if(hipDeviceSynchronize() != hipSuccess) abort();
        if(hipFree(dev) != hipSuccess) abort();
        delete[] host;
    }

    void toDevice()
    {
        HIP_API_CALL(hipMemcpy(dev, host, memsize, hipMemcpyDefault));
        HIP_API_CALL(hipDeviceSynchronize());
    }

    void toHost()
    {
        HIP_API_CALL(hipMemcpy(host, dev, memsize, hipMemcpyDefault));
        HIP_API_CALL(hipDeviceSynchronize());
    }

    const int rows;
    const int columns;
    const int memsize;

    Type* host;
    Type* dev;
};

#define SHMBLOCK 64
#define TBLOCK 16

using float16 = __hip_bfloat16;
using float8  = __hip_fp8_e4m3_fnuz;
using Vec4    = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using IVec4    = __attribute__((__vector_size__(4 * sizeof(uint32_t)))) uint32_t;
using Vec16   = __attribute__((__vector_size__(16 * sizeof(float)))) float;
using Vec8     = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using float8x8 = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;
using IVec2 = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using IVec8 = __attribute__((__vector_size__(8 * sizeof(int)))) int;

static_assert(sizeof(float8) == sizeof(uint8_t));

#if defined(__gfx942__) || defined(__gfx950__)

inline __device__ uint64_t castfrom8x8(const float8x8& f)
{
    return *reinterpret_cast<const uint64_t*>(&f);
}

inline __device__ Vec4 mfma(const float8x8& a, const float8x8& b, const Vec4& initial = Vec4{})
{
    return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(castfrom8x8(a), castfrom8x8(b), initial, 0, 0, 0);
}

inline __device__ Vec4 mfma(const float8x8& a0, const float8x8& a1, const float8x8& b0, const float8x8& b1)
{
    Vec4 ret = mfma(a0, b0);
    return mfma(a1, b1, ret);
}

#else

#if defined(__gfx1200__) || defined(__gfx1201__)
inline __device__ IVec2 castfrom8x8(const float8x8& f)
{
    return *reinterpret_cast<const IVec2*>(&f);
}
inline __device__ Vec8 mfma(const float8x8& a, const float8x8& b, const Vec8& initial = Vec8{})
{
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(castfrom8x8(a), castfrom8x8(b), initial);
}
#else
inline __device__ IVec8 castfrom8x8(const float8x8& f)
{
    return *reinterpret_cast<const IVec8*>(&f);
}
inline __device__ Vec8 mfma(const IVec8& a, const IVec8& b, const Vec8& initial = Vec8{})
{
    return __builtin_amdgcn_wmma_f32_16x16x64_fp8_fp8(a, b, 0, initial, 0, 0);
}
#endif

inline __device__ Vec4 mfma(const float8x8& a0, const float8x8& a1, const float8x8& b0, const float8x8& b1)
{
    Vec8 ret = mfma(castfrom8x8(a0), castfrom8x8(b0));
    ret = mfma(castfrom8x8(a1), castfrom8x8(b1), ret);
    return {ret[0], ret[1], ret[2], ret[3]};
}

#endif

// HEIGHT <= 4
// WIDTH > 4 causes register spill
template<int WIDTH, int HEIGHT = 4>
__global__ void __launch_bounds__(TBLOCK*TBLOCK*2, 2)
fp8_gemm_kernel(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    float16* __restrict__ c,
    const float* __restrict__ scale_a,
    const float* __restrict__ scale_b,
    int MDIM, int NDIM, int KDIM
) {
    const int X = blockIdx.x * SHMBLOCK * WIDTH;
    const int Y = blockIdx.y * TBLOCK * HEIGHT;

    const int TX = threadIdx.x%TBLOCK;
    const int TY = threadIdx.y%4;
    const int TZ = (threadIdx.y/4)%4;
    const int TYZ = threadIdx.y%16;
    const int TW = (threadIdx.y/16)%2;

    __shared__ uint8_t a_shared[WIDTH][4][4][4][TBLOCK][4];
    __shared__ uint8_t b_shared[4][HEIGHT][4][TBLOCK][4];
    __shared__ float   scalar[4][TBLOCK][WIDTH];

    if (TW != 0)
    {
        sqtt_marker_enter("Producer Thread");

        for (int k1=0; k1 < KDIM; k1 += 2*SHMBLOCK)
        for (int k2=0; k2 < 2*SHMBLOCK; k2 += SHMBLOCK)
        {
            sqtt_marker_enter("VRAM matrix load");

            int k0 = k1 + k2;
            IVec4 temp_a[WIDTH], temp_b;

            int tx3 = (TX & 3) << 2;
            int i1 = TX >> 2;

#pragma unroll
            for (int j1=0; j1<4; j1++)
            if (k0 + j1*TBLOCK < KDIM && Y + 4*TX < NDIM)
                temp_b[j1] = *reinterpret_cast<const uint32_t*>(&b[(k0 + TZ*TBLOCK + TY + 4*j1)*NDIM + Y + 4*TX]);

#pragma unroll
            for (int j1=0; j1<4; j1++)
            if (k0 + j1*TBLOCK < KDIM && X < MDIM)
#pragma unroll
            for (int n=0; n<WIDTH; n++)
                temp_a[n][j1] = (X + n*SHMBLOCK < MDIM) ? *reinterpret_cast<const uint32_t*>(&a[(k0 + j1*TBLOCK + TYZ)*MDIM + X + 4*TX + n*SHMBLOCK]) : 0;

            sqtt_marker_exit("VRAM matrix load");

            if (k0 != 0) __syncthreads();

            sqtt_marker_enter("Store matrix to DS");

            if (i1 < HEIGHT)
#pragma unroll
            for (int j1=0; j1<4; j1++)
#pragma unroll
            for (int m=0; m<4; m++)
                b_shared[TZ][i1][TY][tx3 + m][j1] = (temp_b[j1] >> (8*m)) & 0xFF;

#pragma unroll
            for (int j1=0; j1<4; j1++)
#pragma unroll
            for (int m=0; m<4; m++)
#pragma unroll
            for (int n=0; n<WIDTH; n++)
                a_shared[n][i1][TZ][TY][tx3 + m][j1] = (temp_a[n][j1] >> (8*m)) & 0xFF;

            sqtt_marker_exit("Store matrix to DS");

            __syncthreads();
        }
    
        sqtt_marker_exit("Producer Thread");
        return;
    }

    sqtt_marker_enter("Consumer Thread");
    
    Vec16 reg_res[WIDTH];
#pragma unroll
    for (int n=0; n<WIDTH; n++) reg_res[n] = {};

    float tmp_a[WIDTH];
    float tmp_b;
    if (TZ == 0)
    {
        sqtt_marker_enter("Preload scale");
#pragma unroll
        for (int n=0; n<WIDTH; n++)
            tmp_a[n] = (X + TY*TBLOCK + n*SHMBLOCK < MDIM) ? scale_a[X + TY*TBLOCK + TX + n*SHMBLOCK] : 0;
        tmp_b = scale_b[Y/128];
        sqtt_marker_exit("Preload scale");
    }

    for (int k1=0; k1 < KDIM; k1 += 2*SHMBLOCK)
    {
        if (TZ == 0)
        {
            sqtt_marker_enter("Preload scale");
            for (int n=0; n<WIDTH; n++) scalar[TY][TX][n] = tmp_b * tmp_a[n];

            if (k1 + 2*SHMBLOCK < KDIM)
            {
#pragma unroll
                for (int n=0; n<WIDTH; n++)
                    tmp_a[n] = (X + TY*TBLOCK + n*SHMBLOCK < MDIM) ? scale_a[(k1/128 + 1)*MDIM + X + TY*TBLOCK + TX + n*SHMBLOCK] : 0;
                tmp_b = scale_b[(k1/128 + 1)*((NDIM+127)/128) + (Y/128)];
            }
            sqtt_marker_exit("Preload scale");
        }
        for (int k2=0; k2 < 2*SHMBLOCK; k2 += SHMBLOCK)
        {
            int k0 = k1 + k2;

            sqtt_marker_enter("Wait for producer");
            __syncthreads();
            sqtt_marker_exit("Wait for producer");
            sqtt_marker_enter("Load matrix from DS");

            float8x8 a0_load[WIDTH], a1_load[WIDTH];
            float8x8 b0_load[HEIGHT], b1_load[HEIGHT];

#pragma unroll
            for (int m=0; m<8; m++)
#pragma unroll
            for (int n=0; n<HEIGHT; n++)
            {
                b0_load[n][m] = b_shared[m/4 + 0][n][TY][TX][m%4];
                b1_load[n][m] = b_shared[m/4 + 2][n][TY][TX][m%4];
            }

#pragma unroll
            for (int m=0; m<8; m++)
#pragma unroll
            for (int r=0; r<WIDTH; r++)
            {
                a0_load[r][m] = a_shared[r][TZ][m%4][TY][TX][m/4 + 0];
                a1_load[r][m] = a_shared[r][TZ][m%4][TY][TX][m/4 + 2];
            }

            Vec4 scal[WIDTH];
#pragma unroll
            for (int n=0; n<WIDTH; n++)
#pragma unroll
            for (int m=0; m<4; m++)
                scal[n][m] = scalar[TZ][TY*4 + m][n];

            sqtt_marker_exit("Load matrix from DS");
            sqtt_marker_enter("Wait for producer");

            __syncthreads();

            sqtt_marker_exit("Wait for producer");
            sqtt_marker_enter("MFMA Section");

#pragma unroll
            for (int n=0; n<HEIGHT; n++)
#pragma unroll
            for (int r1=0; r1<WIDTH; r1++)
            {
                Vec4 res = mfma(a0_load[r1], a1_load[r1], b0_load[n], b1_load[n]);

                for (int m=0; m<4; m++)
                    reg_res[r1][m*HEIGHT + n] += scal[r1][m] * res[m];
            }

            sqtt_marker_exit("MFMA Section");
        }
    }

    for (int j=0; j<4; j ++) for (int i=0; i<HEIGHT; i ++) for (int r1=0; r1<WIDTH; r1++)
    if (Y + i*TBLOCK < NDIM && 4*TYZ + X + r1*SHMBLOCK < MDIM)
        c[(4*TYZ + j + X + r1*SHMBLOCK)*NDIM + Y + i*TBLOCK + TX] = (float16) reg_res[r1][j*HEIGHT + i];

    sqtt_marker_exit("Consumer Thread");
}

void launchHip(
    const void* a,
    const void* b,
    void* c,
    const float* scale_a,
    const float* scale_b,
    int M, int N, int K
) {
    dim3 block(TBLOCK, 32, 1);
    dim3 grid((M + 4*SHMBLOCK - 1) / SHMBLOCK / 4, N / SHMBLOCK);

    fp8_gemm_kernel<4><<<grid, block, 0, 0>>>(
        reinterpret_cast<const uint8_t*>(a),
        reinterpret_cast<const uint8_t*>(b),
        reinterpret_cast<float16*>(c),
        scale_a,
        scale_b,
        M, N, K);

    HIP_API_CALL(hipGetLastError());
    HIP_API_CALL(hipDeviceSynchronize());
}


int main()
{
    const int device = 0;
    int M = 6144;
    int K = 7168;
    int N = 4608;

    hipDeviceProp_t devProp{};
    HIP_API_CALL(hipGetDeviceProperties(&devProp, device));
    HIP_API_CALL(hipSetDevice(device));

    if (devProp.major > 9 && devProp.minor == 0)
    {
        M /= 3;
        K /= 3;
        N /= 3;
    }
    else if (devProp.major > 9 && devProp.minor == 5)
    {
        M /= 2;
        K /= 2;
    }

    Matrix<float8> a(K, M);
    Matrix<float8> b(K, N);
    Matrix<float16> c(M, N);
    Matrix<float> scale_a(K/128, M);
    Matrix<float> scale_b(K/128, N/128);

    // warmup
    launchHip(a.dev, b.dev, c.dev, scale_a.dev, scale_b.dev, M, N, K);

    HIP_API_CALL(hipDeviceSynchronize());
    roctxProfilerResume(0);

    int runs = 10;
    for (int i=0; i<runs; i++)
        launchHip(a.dev, b.dev, c.dev, scale_a.dev, scale_b.dev, M, N, K);

    HIP_API_CALL(hipDeviceSynchronize());
    roctxProfilerPause(0);

    return 0;
}
