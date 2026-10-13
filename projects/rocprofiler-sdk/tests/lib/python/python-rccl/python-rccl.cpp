// MIT License
//
// Copyright (c) 2023-2026 Advanced Micro Devices, Inc. All rights reserved.
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

// Single-process multi-GPU RCCL workload module for the anytime-tool-config tests.
// Drives the RCCL dispatch table (the table rocprofiler interposes for RCCL_API tracing)
// via ncclCommInitAll + grouped ncclAllReduce across all visible GPUs in one process.
//
// C ABI (loaded via ctypes):
//   rccl_init()            -> pick up to MAX_DEVICES GPUs, ncclCommInitAll, per-device
//                             HIP stream + send/recv buffers. Returns device count, or -1.
//   rccl_allreduce(reps)   -> per rep: grouped ncclAllReduce across all devices + sync.
//                             Returns 0 on success, -1 on error.
//   rccl_fini()            -> destroy comms/streams/buffers. Returns 0 on success, -1.

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <cstdio>
#include <vector>

#define RCCL_WORKLOAD_PUBLIC_API __attribute__((visibility("default")))

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t err = (call);                                                                   \
        if(err != hipSuccess)                                                                      \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[python-rccl] HIP error %d (%s) at %s:%d\n",                                  \
                    (int) err,                                                                     \
                    hipGetErrorString(err),                                                        \
                    __FILE__,                                                                      \
                    __LINE__);                                                                     \
            return -1;                                                                             \
        }                                                                                          \
    } while(0)

#define NCCL_CHECK(call)                                                                           \
    do                                                                                             \
    {                                                                                              \
        ncclResult_t res = (call);                                                                 \
        if(res != ncclSuccess)                                                                     \
        {                                                                                          \
            fprintf(stderr,                                                                        \
                    "[python-rccl] RCCL error %d (%s) at %s:%d\n",                                 \
                    (int) res,                                                                     \
                    ncclGetErrorString(res),                                                       \
                    __FILE__,                                                                      \
                    __LINE__);                                                                     \
            return -1;                                                                             \
        }                                                                                          \
    } while(0)

namespace
{
constexpr int    max_devices = 8;
constexpr size_t elements    = 1024 * 1024;  // 4 MiB of float per buffer

int                      num_devices = 0;
std::vector<ncclComm_t>  comms;
std::vector<hipStream_t> streams;
std::vector<float*>      sendbuf;
std::vector<float*>      recvbuf;
}  // namespace

extern "C" {

/// Initialize RCCL across up to max_devices GPUs in this single process. Returns the
/// number of devices initialized, or -1 on error.
RCCL_WORKLOAD_PUBLIC_API int
rccl_init()
{
    int avail = 0;
    HIP_CHECK(hipGetDeviceCount(&avail));
    num_devices = (avail < max_devices) ? avail : max_devices;
    if(num_devices < 1)
    {
        fprintf(stderr, "[python-rccl] no GPU devices available\n");
        return -1;
    }

    comms.resize(num_devices);
    streams.resize(num_devices, nullptr);
    sendbuf.resize(num_devices, nullptr);
    recvbuf.resize(num_devices, nullptr);

    std::vector<int> devlist(num_devices);
    for(int i = 0; i < num_devices; ++i)
    {
        devlist[i] = i;
        HIP_CHECK(hipSetDevice(i));
        HIP_CHECK(hipStreamCreate(&streams[i]));
        HIP_CHECK(hipMalloc(&sendbuf[i], elements * sizeof(float)));
        HIP_CHECK(hipMalloc(&recvbuf[i], elements * sizeof(float)));
        HIP_CHECK(hipMemset(sendbuf[i], 1, elements * sizeof(float)));
    }

    NCCL_CHECK(ncclCommInitAll(comms.data(), num_devices, devlist.data()));
    fprintf(stderr, "[python-rccl] initialized RCCL across %d devices\n", num_devices);
    return num_devices;
}

/// Run `reps` grouped all-reduce collectives across all devices. Returns 0 / -1.
RCCL_WORKLOAD_PUBLIC_API int
rccl_allreduce(int reps)
{
    if(num_devices < 1)
    {
        fprintf(stderr, "[python-rccl] rccl_allreduce called before init\n");
        return -1;
    }

    for(int r = 0; r < reps; ++r)
    {
        NCCL_CHECK(ncclGroupStart());
        for(int i = 0; i < num_devices; ++i)
        {
            NCCL_CHECK(ncclAllReduce(
                sendbuf[i], recvbuf[i], elements, ncclFloat, ncclSum, comms[i], streams[i]));
        }
        NCCL_CHECK(ncclGroupEnd());

        for(int i = 0; i < num_devices; ++i)
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipStreamSynchronize(streams[i]));
        }
    }
    return 0;
}

/// Destroy RCCL communicators and free per-device resources. Returns 0 / -1.
RCCL_WORKLOAD_PUBLIC_API int
rccl_fini()
{
    for(int i = 0; i < num_devices; ++i)
    {
        if(comms[i]) ncclCommDestroy(comms[i]);
        if(streams[i])
        {
            HIP_CHECK(hipSetDevice(i));
            HIP_CHECK(hipStreamDestroy(streams[i]));
        }
        if(sendbuf[i]) HIP_CHECK(hipFree(sendbuf[i]));
        if(recvbuf[i]) HIP_CHECK(hipFree(recvbuf[i]));
    }
    comms.clear();
    streams.clear();
    sendbuf.clear();
    recvbuf.clear();
    num_devices = 0;
    fprintf(stderr, "[python-rccl] finalized\n");
    return 0;
}

}  // extern "C"
