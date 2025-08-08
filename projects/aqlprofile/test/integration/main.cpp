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

#include <assert.h>
#include <stdlib.h>
#include <unistd.h>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <vector>
#include <map>
#include <atomic>
#include <future>
#include "counter.hpp"
#include "workload.hpp"
#include "hip/hip_runtime.h"

#define DATA_SIZE          (64*4)

#define HIP_API_CALL(CALL) do { if ((CALL) != hipSuccess) abort(); } while(0)


//#define ATTEMPT_GMI
#define DOT2_ARCH

class hipMemory
{
public:
    hipMemory(size_t size)
    {
        HIP_API_CALL(hipMalloc(&ptr, size * sizeof(float)));
        HIP_API_CALL(hipMemset(ptr, 0, size * sizeof(float)));
    }
    ~hipMemory()
    {
        if(ptr) HIP_API_CALL(hipFree(ptr));
    }

    hipMemory(hipMemory& other) = delete;
    hipMemory& operator=(hipMemory& other) = delete;

    float* ptr = nullptr;
};

class Stream
{
public:
    Stream() { HIP_API_CALL(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking)); }
    ~Stream() { HIP_API_CALL(hipStreamDestroy(stream)); }

    Stream(Stream& other) = delete;
    Stream& operator=(Stream& other) = delete;

    void synchronize() { HIP_API_CALL(hipStreamSynchronize(stream)); }

    hipStream_t stream;
};

class HIPWorkload : public IWorkload
{
public:
    HIPWorkload(AgentInfo& agent, const std::vector<std::string>& counters)
    {
        col = std::make_unique<Collection>(agent, counters);
    }
    virtual ~HIPWorkload() {};
    virtual std::string_view name() = 0;

    std::map<std::string, int64_t> collect(Queue& queue)
    {
        assert(col);
        return col->iterate(queue, *this);
    }

    void printcounters(Queue& queue)
    {
        std::cout << "Name: " << name() << std::endl;
        for (auto& [name, v] : collect(queue)) std::cout << " - " << name << ": " << v << std::endl;
    }

    std::unique_ptr<Collection> col{nullptr};
    hipMemory src{DATA_SIZE};
    hipMemory dst{DATA_SIZE};
    Stream stream{};
};


__global__ void copy_kernel(float* a, const float* b)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < DATA_SIZE)
        a[idx] = b[idx];
}

__global__ void atomic_kernel(float* a, const float* b)
{
    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    if (idx < DATA_SIZE)
        atomicAdd(a+threadIdx.x, b[idx]);
}

__global__ void iops_kernel_trans()
{
    // 3 F16 Trans OPS
    asm volatile("v_cos_f16 v0, v0; v_cos_f16 v1, v1; v_cos_f16 v2, v2;");
    // 2 F32 Trans OPS
    asm volatile("v_cos_f32 v3, v3; v_cos_f32 v4, v4");
}

__global__ void iops_kernel1()
{
    asm volatile("v_add_f16 v2, v1, v0"); // 1 F16 OPS
    asm volatile("v_fma_f32 v3, v1, v2, v3"); // 2 F32 OPs

    asm volatile("v_add_f64 v[0:1], v[2:3], v[4:5]"); // 1 F64 OP
    asm volatile("v_fma_f64 v[0:1], v[2:3], v[4:5], v[6:7]"); // 2 F64 OP
    asm volatile("v_fma_f64 v[0:1], v[2:3], v[4:5], v[6:7]"); // 2 F64 OP
}

__global__ void iops_kernel2()
{
   #if defined(__gfx940__) || defined(__gfx90a__) || defined(__gfx1030__)
    // Supported architectures
    asm volatile("v_dot2_f32_f16 v0, v1, v2, v3");
#else
    // Fallback or skip
    asm volatile("v_add_f32 v4, v5, v6"); // 1 F32 OP
    asm volatile("v_fma_f64 v[0:1], v[0:1], v[2:3], v[4:5]"); // 2 F64 OPs
#endif

}

class CopyWorkload : public HIPWorkload
{
public:
    CopyWorkload(AgentInfo& agent, const std::vector<std::string>& counters): HIPWorkload(agent, counters) {}
    virtual void run() override
    {
        copy_kernel<<<DATA_SIZE/64,64,0,stream.stream>>>(dst.ptr, src.ptr);
        stream.synchronize();
    }
    virtual std::string_view name() override { return "CopyWorkload"; };
};

class AtomicWorkload : public HIPWorkload
{
public:
    AtomicWorkload(AgentInfo& agent, const std::vector<std::string>& counters): HIPWorkload(agent, counters) {}
    virtual void run() override
    {
        atomic_kernel<<<DATA_SIZE/64,64,0,stream.stream>>>(dst.ptr, src.ptr);
        stream.synchronize();
    }
    virtual std::string_view name() override { return "AtomicWorkload"; };
};

class IOPSWorkload1 : public HIPWorkload
{
public:
    IOPSWorkload1(AgentInfo& agent, const std::vector<std::string>& counters): HIPWorkload(agent, counters) {}
    virtual void run() override
    {
        iops_kernel1<<<DATA_SIZE/64,64,0,stream.stream>>>();
        stream.synchronize();
    }
    virtual std::string_view name() override { return "IOPSWorkload1"; };
};

class IOPSWorkload2 : public HIPWorkload
{
public:
    IOPSWorkload2(AgentInfo& agent, const std::vector<std::string>& counters): HIPWorkload(agent, counters) {}
    virtual void run() override
    {
        iops_kernel2<<<DATA_SIZE/64,64,0,stream.stream>>>();
        stream.synchronize();
    }
    virtual std::string_view name() override { return "IOPSWorkload2"; };
};

class IOPSWorkload3 : public HIPWorkload
{
public:
    IOPSWorkload3(AgentInfo& agent, const std::vector<std::string>& counters): HIPWorkload(agent, counters) {}
    virtual void run() override
    {
        iops_kernel_trans<<<DATA_SIZE/64,64,0,stream.stream>>>();
        stream.synchronize();
    }
    virtual std::string_view name() override { return "Trans IOPSWorkload"; };
};

class GMIWorkload : public HIPWorkload
{
public:
    GMIWorkload(AgentInfo& agent, const std::vector<std::string>& counters): HIPWorkload(agent, counters) {}
    virtual void run() override
    {
        auto policies = std::vector<unsigned>{hipHostMallocDefault, hipHostMallocCoherent, hipHostMallocNonCoherent};
        for (auto& policy : policies)
        {
            float* srchost;
            float* dsthost;
            HIP_API_CALL(hipHostMalloc(&srchost, DATA_SIZE * sizeof(float), policy));
            HIP_API_CALL(hipHostMalloc(&dsthost, DATA_SIZE * sizeof(float), policy));

            for (size_t i=0; i<DATA_SIZE; i++)
                srchost[i] = float(i);

            copy_kernel<<<DATA_SIZE/64,64,0,stream.stream>>>(dsthost, srchost);
            stream.synchronize();
            atomic_kernel<<<DATA_SIZE/64,64,0,stream.stream>>>(srchost, dsthost);
            stream.synchronize();
            copy_kernel<<<DATA_SIZE/64,64,0,stream.stream>>>(dst.ptr, src.ptr);
            stream.synchronize();

            HIP_API_CALL(hipHostFree(srchost));
            HIP_API_CALL(hipHostFree(dsthost));
        }
    }
    virtual std::string_view name() override { return "GMIWorkload"; };
};

auto tcp1_counters(std::string_view gfxip)
{
    std::vector<std::string> counters = {"GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU"};

    if (gfxip.find("gfx95") == 0)
    {
        counters.push_back("TCP_CACHE_ACCESS");
        counters.push_back("TCP_CACHE_MISS");
        counters.push_back("TCP_READ");
        counters.push_back("TCP_WRITE");
        counters.push_back("TCC_EA0_WRREQ_DRAM");
        counters.push_back("TCC_EA0_WRREQ_WRITE_DRAM");
        counters.push_back("TCC_EA0_WRREQ_WRITE_DRAM_32B");
        counters.push_back("TCC_EA0_WRREQ_ATOMIC_DRAM");
    }
    else if (gfxip.find("gfx94") == 0)
    {
        counters.push_back("TCP_READ");
        counters.push_back("TCP_WRITE");
        counters.push_back("TCC_REQ");
        counters.push_back("TCC_EA0_RDREQ");
        counters.push_back("TCC_ATOMIC");
        counters.push_back("TCC_EA0_ATOMIC");
    }
    return counters;
}

auto tcp2_counters(std::string_view gfxip)
{
    std::vector<std::string> counters = {"GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU"};

    if (gfxip.find("gfx95") == 0)
    {
        counters.push_back("TCP_CACHE_MISS_TG0");
        counters.push_back("TCP_CACHE_MISS_TG1");
        counters.push_back("TCP_CACHE_MISS_TG2");
        counters.push_back("TCP_CACHE_MISS_TG3");
    }
    else if (gfxip.find("gfx94") == 0)
    {
        counters.push_back("TCP_READ");
        counters.push_back("TCP_WRITE");
        counters.push_back("TCC_REQ");
        counters.push_back("TCC_EA0_RDREQ");
        counters.push_back("TCC_ATOMIC");
        counters.push_back("TCC_EA0_ATOMIC");
    }
    return counters;
}

auto atomic_counters(std::string_view gfxip)
{
    std::vector<std::string> counters = {"GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU"};

    if (gfxip.find("gfx95") == 0)
    {
        counters.push_back("TCC_EA0_WRREQ_ATOMIC_DRAM");
        counters.push_back("TCC_EA0_WRREQ_ATOMIC_DRAM_32B");
        counters.push_back("TCC_EA0_WRREQ_DRAM");
        counters.push_back("TCC_EA0_WRREQ_WRITE_DRAM");
    }
    else if (gfxip.find("gfx94") == 0)
    {
        counters.push_back("TCP_READ");
        counters.push_back("TCP_WRITE");
        counters.push_back("TCC_REQ");
        counters.push_back("TCC_EA0_RDREQ");
        counters.push_back("TCC_ATOMIC");
        counters.push_back("TCC_EA0_ATOMIC");
    }
    return counters;
}

auto iops_counters(std::string_view gfxip)
{
    std::vector<std::string> counters = {"GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU"};

    if (gfxip.find("gfx95") == 0)
    {
        counters.push_back("SQ_INSTS_VALU_FLOPS_FP16");
        counters.push_back("SQ_INSTS_VALU_FLOPS_FP32");
        counters.push_back("SQ_INSTS_VALU_FLOPS_FP64");
        counters.push_back("SQ_INSTS_VALU_FLOPS_FP16_TRANS");
        counters.push_back("SQ_INSTS_VALU_FLOPS_FP32_TRANS");
        counters.push_back("SQ_INSTS_VALU_FLOPS_FP64_TRANS");
    }
    return counters;
}

auto gmi_counters(std::string_view gfxip)
{
    std::vector<std::string> counters = {"GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU"};

    if (gfxip.find("gfx95") == 0)
    {
        counters.push_back("TCC_EA0_RDREQ");
        counters.push_back("TCC_EA0_RDREQ_GMI_32B");
        counters.push_back("TCC_EA0_WRREQ_GMI_32B");
        counters.push_back("TCC_EA0_ATOMIC_GMI_32B");
    }
    else if (gfxip.find("gfx94") == 0)
    {
        counters.push_back("TCC_EA0_WRREQ_CREDIT_STALL");
        counters.push_back("TCC_EA0_WRREQ_IO_CREDIT_STALL");
        counters.push_back("TCC_EA0_WRREQ_GMI_CREDIT_STALL");
        counters.push_back("TCC_EA0_WRREQ_DRAM_CREDIT_STALL");
    }
    return counters;
}

auto io_counters(std::string_view gfxip)
{
    std::vector<std::string> counters = {"GRBM_COUNT", "SQ_WAVES", "SQ_INSTS_VALU"};

    if (gfxip.find("gfx95") == 0)
    {
        counters.push_back("TCC_EA0_RDREQ");
        counters.push_back("TCC_EA0_RDREQ_IO_32B");
        counters.push_back("TCC_EA0_WRREQ_IO_32B");
        counters.push_back("TCC_EA0_ATOMIC_IO_32B");
    }
    else if (gfxip.find("gfx94") == 0)
    {
        counters.push_back("TCC_EA0_RDREQ");
        counters.push_back("TCC_EA0_RDREQ_IO_CREDIT_STALL");
        counters.push_back("TCC_EA0_RDREQ_GMI_CREDIT_STALL");
        counters.push_back("TCC_EA0_RDREQ_DRAM_CREDIT_STALL");
    }
    return counters;
}

void printcounters(const std::map<std::string, int64_t>& map)
{
    for (auto& [name, v] : map) std::cout << " - " << name << ": " << v << std::endl;
}

int main()
{
    CHECK_HSA(hsa_init());
    AgentInfo::iterate_agents();

    auto agent = AgentInfo::gpu_agents.at(0);

    {
        Queue queue(agent);
        CopyWorkload tcp1(*agent, tcp1_counters(agent->gfxip));
        CopyWorkload tcp2(*agent, tcp2_counters(agent->gfxip));
        AtomicWorkload atomic(*agent, atomic_counters(agent->gfxip));
        IOPSWorkload1 iops1(*agent, iops_counters(agent->gfxip));
        IOPSWorkload2 iops2(*agent, iops_counters(agent->gfxip));
        IOPSWorkload3 iops3(*agent, iops_counters(agent->gfxip));

        // warmup
        tcp1.run();
        tcp2.run();
        atomic.run();

        // Test
        tcp1.printcounters(queue);
        tcp2.printcounters(queue);
        atomic.printcounters(queue);
        iops1.printcounters(queue);
        iops2.printcounters(queue);
        iops3.printcounters(queue);

#ifdef ATTEMPT_GMI
        GMIWorkload(*agent, gmi_counters(agent->gfxip)).printcounters(queue);
        GMIWorkload(*agent, io_counters(agent->gfxip)).printcounters(queue);
#endif
    }

    CHECK_HSA(hsa_shut_down());
    return 0;
}
