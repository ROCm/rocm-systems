/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hipfile-warnings.h"
#include "util.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

#ifdef __HIP_PLATFORM_NVIDIA__
#include <cuda.h>
#endif

namespace {
struct ScopedHipStream {
    ScopedHipStream()
    {
        assert(hipStreamCreateWithFlags(&stream_, hipStreamNonBlocking) == hipSuccess);
    }
    ~ScopedHipStream()
    {
        assert(hipStreamDestroy(stream_) == hipSuccess);
    }
    hipStream_t stream() const
    {
        return stream_;
    }

private:
    hipStream_t stream_;
};

hipStream_t
staticHipStream()
{
    HIPFILE_WARN_NO_EXIT_DTOR_OFF
    static ScopedHipStream s;
    HIPFILE_WARN_NO_EXIT_DTOR_ON
    return s.stream();
}

bool
isGpuMemory(void *mem)
{
    hipPointerAttribute_t attrs;
    hipError_t            err = hipPointerGetAttributes(&attrs, mem);

#ifdef __HIP_PLATFORM_NVIDIA__
    // NVIDIA doesn't support pointers allocated outside of runtime
    if (err == hipErrorInvalidValue) {
        return false;
    }
#endif
    assert(err == hipSuccess);
    return attrs.type == hipMemoryTypeDevice;
}

std::vector<uint8_t>
copyGpuMemory(void *gpu_mem, hoff_t gpu_mem_offset, size_t region_size)
{
    std::vector<uint8_t> mem_region(region_size);
    assert(hipMemcpyAsync(mem_region.data(),
                          reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(gpu_mem) +
                                                   static_cast<size_t>(gpu_mem_offset)),
                          region_size, hipMemcpyDeviceToHost, staticHipStream()) == hipSuccess);
    assert(hipStreamSynchronize(staticHipStream()) == hipSuccess);
    return mem_region;
}
} // namespace

void
HipFileDataOps::assertMemoryRegionsMatch(void *mem1, hoff_t mem1_offset, void *mem2, hoff_t mem2_offset,
                                         size_t region_size)
{
    std::vector<uint8_t> mem1_v;
    std::vector<uint8_t> mem2_v;
    assert(mem1_offset >= 0);
    assert(mem2_offset >= 0);
    if (isGpuMemory(mem1)) {
        mem1_v = copyGpuMemory(mem1, mem1_offset, region_size);
        mem1   = mem1_v.data();
    }
    else {
        mem1 =
            reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem1) + static_cast<uintptr_t>(mem1_offset));
    }
    if (isGpuMemory(mem2)) {
        mem2_v = copyGpuMemory(mem2, mem2_offset, region_size);
        mem2   = mem2_v.data();
    }
    else {
        mem2 =
            reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem2) + static_cast<uintptr_t>(mem2_offset));
    }
    assert(std::memcmp(mem1, mem2, region_size) == 0);
}

void
HipFileDataOps::assertFileAndMemoryRegionsMatch(void *mem, hoff_t mem_offset, int fd, hoff_t fd_offset,
                                                size_t region_size)
{
    assert(fd_offset >= 0);
    auto file_region = std::vector<uint8_t>(region_size);

    ssize_t rv = pread(fd, file_region.data(), region_size, fd_offset);
    assert(rv > 0 && static_cast<size_t>(rv) == region_size);

    assertMemoryRegionsMatch(file_region.data(), 0, mem, mem_offset, region_size);
}

void
HipFileDataOps::assertZeroedMemRegion(void *mem, hoff_t mem_offset, size_t region_size)
{
    std::vector<uint8_t> mem_v;
    assert(mem_offset >= 0);
    if (isGpuMemory(mem)) {
        mem_v = copyGpuMemory(mem, mem_offset, region_size);
        mem   = mem_v.data();
    }
    else {
        mem = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem) + static_cast<uintptr_t>(mem_offset));
    }
    for (size_t i = 0; i < region_size; ++i) {
        assert(reinterpret_cast<uint8_t *>(mem)[i] == 0);
    }
}

void
HipFileDataOps::assertZeroedFileRegion(int fd, hoff_t fd_offset, size_t region_size)
{
    assert(fd_offset >= 0);
    auto    file_region = std::vector<uint8_t>(region_size);
    ssize_t rv          = pread(fd, file_region.data(), region_size, fd_offset);
    assert(rv > 0 && static_cast<size_t>(rv) == region_size);
    for (size_t i = 0; i < region_size; ++i) {
        assert(file_region.data()[i] == 0);
    }
}

void
HipFileDataOps::randomizeMemoryRegion(void *mem, hoff_t offset, size_t region_size)
{
    ssize_t rv;
    int     rand_fd = open("/dev/urandom", O_RDONLY);
    assert(rand_fd != -1);
    if (isGpuMemory(mem)) {
        std::vector<uint8_t> mem_v(region_size);
        rv = read(rand_fd, mem_v.data(), region_size);
        assert(rv > 0 && static_cast<size_t>(rv) == region_size);
        assert(hipMemcpyAsync(
                   reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem) + static_cast<size_t>(offset)),
                   mem_v.data(), region_size, hipMemcpyHostToDevice, staticHipStream()) == hipSuccess);
        assert(hipStreamSynchronize(staticHipStream()) == hipSuccess);
    }
    else {
        rv = read(rand_fd,
                  reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem) + static_cast<size_t>(offset)),
                  region_size);
        assert(rv > 0 && static_cast<size_t>(rv) == region_size);
    }
    assert(close(rand_fd) == 0);
}

void
HipFileDataOps::zeroMemoryRegion(void *mem, hoff_t offset, size_t region_size)
{
    if (isGpuMemory(mem)) {
        assert(hipMemsetAsync(
                   reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem) + static_cast<size_t>(offset)),
                   0, region_size, staticHipStream()) == hipSuccess);
        assert(hipStreamSynchronize(staticHipStream()) == hipSuccess);
    }
    else {
        memset(reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(mem) + static_cast<size_t>(offset)), 0,
               region_size);
    }
}

void
HipFileDataOps::zeroFileRegion(int fd, size_t size, hoff_t offset)
{
    auto    vec = std::vector<uint8_t>(size, 0);
    ssize_t rv  = pwrite(fd, vec.data(), size, offset);
    assert(rv > 0 && static_cast<size_t>(rv) == size);
}

void
HipFileDataOps::randomizeFileRegion(int fd, size_t size, hoff_t offset)
{
    auto vec = std::vector<uint8_t>(size, 0);
    randomizeMemoryRegion(vec.data(), 0, size);
    ssize_t rv = pwrite(fd, vec.data(), size, offset);
    assert(rv > 0 && static_cast<size_t>(rv) == size);
}
