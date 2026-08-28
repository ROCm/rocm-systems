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
#include <utility>

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
            "  FILE     scratch file to write and read, created fresh and removed on\n"
            "           exit; must not already exist\n"
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

/// Runs its cleanup callable at scope exit, so the setup failures that return before
/// the I/O loop starts still release whatever has already been acquired.
///
template <typename Cleanup>
class scope_guard
{
public:
    explicit scope_guard(Cleanup cleanup)
    : m_cleanup{ std::move(cleanup) }
    {}

    ~scope_guard() { m_cleanup(); }

    scope_guard(const scope_guard&)            = delete;
    scope_guard& operator=(const scope_guard&) = delete;
    scope_guard(scope_guard&&)                 = delete;
    scope_guard& operator=(scope_guard&&)      = delete;

private:
    Cleanup m_cleanup;
};

struct io_result
{
    /// False when an operation reported an error or moved fewer bytes than asked for;
    /// either one leaves the file and the hipFile counters in a state this workload
    /// does not claim, so neither counts as an iteration.
    bool          succeeded;
    std::uint64_t iterations;
};

/// Writes the buffer and reads it back until @p seconds have elapsed.
io_result
run_io_loop(hipFileHandle_t handle, void* devbuf, size_t bytes, int seconds)
{
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(seconds);
    std::uint64_t iterations = 0;
    while(std::chrono::steady_clock::now() < deadline)
    {
        const auto bytes_written = hipFileWrite(handle, devbuf, bytes, 0, 0);
        if(bytes_written != static_cast<ssize_t>(bytes))
        {
            fprintf(stderr, "hipFileWrite returned %zd, expected %zu\n", bytes_written,
                    bytes);
            return { false, iterations };
        }
        const auto bytes_read = hipFileRead(handle, devbuf, bytes, 0, 0);
        if(bytes_read != static_cast<ssize_t>(bytes))
        {
            fprintf(stderr, "hipFileRead returned %zd, expected %zu\n", bytes_read,
                    bytes);
            return { false, iterations };
        }
        ++iterations;
        std::this_thread::sleep_for(std::chrono::milliseconds(k_loop_sleep_ms));
    }
    return { true, iterations };
}
}  // namespace

// NOLINTBEGIN(readability-function-size)
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
    const scope_guard devbuf_guard{ [&] {
        if(hipFree(devbuf) != hipSuccess)
        {
            fprintf(stderr, "hipFree failed\n");
        }
    } };

    if(hipMemset(devbuf, k_buffer_fill_byte, bytes) != hipSuccess)
    {
        fprintf(stderr, "hipMemset failed\n");
        return EXIT_FAILURE;
    }

    hipFileError_t err = hipFileBufRegister(devbuf, bytes, 0);
    if(err.err != hipFileSuccess)
    {
        fprintf(stderr, "hipFileBufRegister failed (%s)\n",
                hipFileGetOpErrorString(err.err));
        return EXIT_FAILURE;
    }
    const scope_guard bufreg_guard{ [&] {
        const hipFileError_t status = hipFileBufDeregister(devbuf);
        if(status.err != hipFileSuccess)
        {
            fprintf(stderr, "hipFileBufDeregister failed (%s)\n",
                    hipFileGetOpErrorString(status.err));
        }
    } };

    // O_EXCL, not O_TRUNC: the file is unlinked on exit, so a mistyped FILE would
    // otherwise overwrite and then delete data the user meant to keep.
    const int raw_fd = open(path, O_RDWR | O_CREAT | O_EXCL, S_IRUSR | S_IWUSR);
    if(raw_fd < 0)
    {
        if(errno == EEXIST)
        {
            fprintf(stderr,
                    "%s already exists; hipfile-io only writes a file it creates, and "
                    "removes it on exit. Remove it or pass a different FILE.\n",
                    path);
        }
        else
        {
            fprintf(stderr, "open(%s) failed (%s)\n", path, strerror(errno));
        }
        return EXIT_FAILURE;
    }
    const scope_guard file_guard{ [&] {
        close(raw_fd);
        unlink(path);
    } };

    if(ftruncate(raw_fd, bytes) != 0)
    {
        fprintf(stderr, "ftruncate failed (%s)\n", strerror(errno));
        return EXIT_FAILURE;
    }

    hipFileHandle_t handle{};
    hipFileDescr_t  descr;
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = raw_fd;
    err             = hipFileHandleRegister(&handle, &descr);
    if(err.err != hipFileSuccess)
    {
        fprintf(stderr, "hipFileHandleRegister failed (%s)\n",
                hipFileGetOpErrorString(err.err));
        return EXIT_FAILURE;
    }
    const scope_guard handle_guard{ [&] { hipFileHandleDeregister(handle); } };

    printf("hipfile-io: pid=%d looping hipFile I/O for %ds\n", static_cast<int>(getpid()),
           seconds);
    fflush(stdout);

    const io_result result = run_io_loop(handle, devbuf, bytes, seconds);

    printf("hipfile-io: %s after %llu write+read iterations\n",
           result.succeeded ? "completed" : "failed",
           static_cast<unsigned long long>(result.iterations));
    fflush(stdout);

    return result.succeeded ? EXIT_SUCCESS : EXIT_FAILURE;
}
// NOLINTEND(readability-function-size)
