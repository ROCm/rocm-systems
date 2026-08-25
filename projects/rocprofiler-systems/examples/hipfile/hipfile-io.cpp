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
// Usage: hipfile-io [FILE] [GPUID] [SECONDS]

#include <hipfile.h>

#include <hip/hip_runtime_api.h>

#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace
{
constexpr int          k_default_gpu_id       = 0;
constexpr int          k_default_duration_sec = 5;
constexpr int          k_loop_sleep_ms        = 20;
constexpr std::uint8_t k_buffer_fill_byte     = 0xAB;

void
print_usage(const char* program)
{
    fprintf(stderr,
            "usage: %s [FILE] [GPUID] [SECONDS]\n"
            "  FILE     scratch file to write and read, removed on exit\n"
            "           (default: hipfile-io.bin in the current directory)\n"
            "  GPUID    GPU ordinal to run on (default: 0)\n"
            "  SECONDS  duration of the I/O loop, at least 1 (default: 5)\n",
            program);
}

/// Strict decimal parse: rejects trailing junk and values below @p minimum, so a
/// mistyped argument fails here instead of silently becoming a no-op run.
bool
parse_int(const char* text, int minimum, int& out)
{
    errno            = 0;
    char*      end   = nullptr;
    const long value = std::strtol(text, &end, 10);

    if(end == text || end == nullptr || *end != '\0' || errno == ERANGE ||
       value < minimum || value > std::numeric_limits<int>::max())
    {
        return false;
    }

    out = static_cast<int>(value);
    return true;
}

/// Owns the scratch file so that every exit path closes the descriptor and removes
/// the file, including the setup failures that return before the I/O loop starts.
class scratch_file
{
public:
    scratch_file(const char* path, int fd)
    : m_path{ path }
    , m_fd{ fd }
    {}

    ~scratch_file()
    {
        if(m_fd >= 0) close(m_fd);
        if(m_path != nullptr) unlink(m_path);
    }

    scratch_file(const scratch_file&)            = delete;
    scratch_file& operator=(const scratch_file&) = delete;
    scratch_file(scratch_file&&)                 = delete;
    scratch_file& operator=(scratch_file&&)      = delete;

    [[nodiscard]] int fd() const noexcept { return m_fd; }

private:
    const char* m_path;
    int         m_fd;
};
}  // namespace

int
main(int argc, char** argv)
{
    const char* path    = (argc > 1) ? argv[1] : "hipfile-io.bin";
    int         gpu_id  = k_default_gpu_id;
    int         seconds = k_default_duration_sec;

    if(argc > 2 && !parse_int(argv[2], 0, gpu_id))
    {
        fprintf(stderr, "invalid GPUID: %s\n", argv[2]);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    // A zero or negative duration would leave the loop below with an already-expired
    // deadline, so the run would exit successfully having issued no hipFile I/O at all
    // and the telemetry would look like a collector failure rather than a bad argument.
    if(argc > 3 && !parse_int(argv[3], 1, seconds))
    {
        fprintf(stderr, "invalid SECONDS: %s (must be an integer >= 1)\n", argv[3]);
        print_usage(argv[0]);
        return EXIT_FAILURE;
    }

    const size_t bytes = 1UL * 1024UL * 1024UL;  // 1 MiB per op

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
    (void) hipMemset(devbuf, k_buffer_fill_byte, bytes);

    hipFileError_t err = hipFileBufRegister(devbuf, bytes, 0);
    if(err.err != hipFileSuccess)
    {
        fprintf(stderr, "hipFileBufRegister failed (%s)\n",
                hipFileGetOpErrorString(err.err));
        return EXIT_FAILURE;
    }

    const int raw_fd = open(path, O_RDWR | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
    if(raw_fd < 0)
    {
        fprintf(stderr, "open(%s) failed (%s)\n", path, strerror(errno));
        return EXIT_FAILURE;
    }

    scratch_file file{ path, raw_fd };

    if(ftruncate(file.fd(), static_cast<off_t>(bytes)) != 0)
    {
        fprintf(stderr, "ftruncate failed (%s)\n", strerror(errno));
        return EXIT_FAILURE;
    }

    hipFileHandle_t handle{};
    hipFileDescr_t  descr{};
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = file.fd();
    err             = hipFileHandleRegister(&handle, &descr);
    if(err.err != hipFileSuccess)
    {
        fprintf(stderr, "hipFileHandleRegister failed (%s)\n",
                hipFileGetOpErrorString(err.err));
        return EXIT_FAILURE;
    }

    printf("hipfile-io: pid=%d looping hipFile I/O for %ds\n", static_cast<int>(getpid()),
           seconds);
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
        std::this_thread::sleep_for(std::chrono::milliseconds(k_loop_sleep_ms));
    }

    printf("hipfile-io: completed %llu write+read iterations\n",
           static_cast<unsigned long long>(iters));
    fflush(stdout);

    // Ordering matters: hipFile must release the handle before ~scratch_file closes the
    // descriptor it was registered against.
    hipFileHandleDeregister(handle);
    (void) hipFileBufDeregister(devbuf);
    (void) hipFree(devbuf);
    return EXIT_SUCCESS;
}
