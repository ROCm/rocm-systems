// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
//
// Minimal hipFile workload used to exercise rocprofiler-systems' hipFile I/O
// telemetry collection. Adapted from the hipFile project's basics examples
// (examples/basics/roundtrip-verify.cpp): it registers a GPU buffer and a file
// handle, then loops read/write for a fixed duration so that the profiler's
// periodic process sampler observes the cumulative hipFile stats.
//
// The file is opened WITHOUT O_DIRECT so the workload runs on any filesystem
// (it exercises hipFile's fallback backend); the goal is to validate the
// telemetry pipeline, not the GPU-direct fast path.
//
// Usage: hipfile-test [FILE] [GPUID] [SECONDS]

#include <hipfile.h>

#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

int
main(int argc, char** argv)
{
    const char*  path    = (argc > 1) ? argv[1] : "hipfile-test.bin";
    const int    gpu_id  = (argc > 2) ? atoi(argv[2]) : 0;
    const int    seconds = (argc > 3) ? atoi(argv[3]) : 5;
    const size_t bytes   = 1UL * 1024UL * 1024UL;  // 1 MiB per op

    if(hipSetDevice(gpu_id) != hipSuccess)
    {
        fprintf(stderr, "hipSetDevice(%d) failed\n", gpu_id);
        return EXIT_FAILURE;
    }

    void* devbuf = nullptr;
    if(hipMalloc(&devbuf, bytes) != hipSuccess)
    {
        fprintf(stderr, "hipMalloc failed\n");
        return EXIT_FAILURE;
    }
    (void) hipMemset(devbuf, 0xAB, bytes);

    hipFileError_t err = hipFileBufRegister(devbuf, bytes, 0);
    if(err.err != hipFileSuccess)
    {
        fprintf(stderr, "hipFileBufRegister failed (%s)\n",
                hipFileGetOpErrorString(err.err));
        return EXIT_FAILURE;
    }

    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if(fd < 0)
    {
        fprintf(stderr, "open(%s) failed (%s)\n", path, strerror(errno));
        return EXIT_FAILURE;
    }
    if(ftruncate(fd, static_cast<off_t>(bytes)) != 0)
    {
        fprintf(stderr, "ftruncate failed (%s)\n", strerror(errno));
        return EXIT_FAILURE;
    }

    hipFileHandle_t handle{};
    hipFileDescr_t  descr{};
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = fd;
    err             = hipFileHandleRegister(&handle, &descr);
    if(err.err != hipFileSuccess)
    {
        fprintf(stderr, "hipFileHandleRegister failed (%s)\n",
                hipFileGetOpErrorString(err.err));
        return EXIT_FAILURE;
    }

    printf("hipfile-test: pid=%d looping hipFile I/O for %ds\n",
           static_cast<int>(getpid()), seconds);
    fflush(stdout);

    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    std::uint64_t iters = 0;
    while(std::chrono::steady_clock::now() < deadline)
    {
        ssize_t nw = hipFileWrite(handle, devbuf, bytes, 0, 0);
        if(nw < 0)
        {
            fprintf(stderr, "hipFileWrite failed (%zd)\n", nw);
            break;
        }
        ssize_t nr = hipFileRead(handle, devbuf, bytes, 0, 0);
        if(nr < 0)
        {
            fprintf(stderr, "hipFileRead failed (%zd)\n", nr);
            break;
        }
        ++iters;
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    printf("hipfile-test: completed %llu write+read iterations\n",
           static_cast<unsigned long long>(iters));
    fflush(stdout);

    hipFileHandleDeregister(handle);
    close(fd);
    (void) hipFileBufDeregister(devbuf);
    (void) hipFree(devbuf);
    (void) unlink(path);
    return EXIT_SUCCESS;
}
