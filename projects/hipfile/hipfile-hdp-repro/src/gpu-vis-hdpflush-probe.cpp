/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Stress test that data hipFileRead writes into GPU memory is observable by
// GPU kernels once hipFileRead returns
//
// hipfileread-cpu-visibility reads the buffer from the CPU, through the GPU's
// host data path (HDP), the same path fastpath's peer-to-peer writes take into
// VRAM. Kernels read VRAM through the GPU's own caches instead, so this test
// also catches data still held in the HDP, rather than written to VRAM, when
// hipFileRead returns, and GPU caches that still hold the data from before
// hipFileRead.
//
// The test data, two patterns written once, goes in a temporary file in
// --directory. Each iteration reads whichever pattern is not in the GPU
// buffer, and a kernel then checks the buffer for it. The kernel's workgroups
// split the buffer evenly, one workgroup per compute unit by default. With
// --prime, the kernel first reads the whole buffer and checks for the previous
// pattern, which brings it into the GPU's caches. --kernel chooses when the
// kernel runs:
//
// - after: a kernel launched after hipFileRead returns checks the buffer, which
//   hipMalloc allocates. With --prime, a separate kernel reads the buffer
//   before hipFileRead. HIP makes coarse-grained memory like this up to date
//   at kernel boundaries, so the kernel must see the new data. --prime-wait
//   chooses how the host learns that the priming kernel has finished reading
//   the buffer before it calls hipFileRead. With sync, hipStreamSynchronize
//   waits for it, and HIP also enqueues a system-scope acquire, which
//   invalidates the GPU's L2 cache, so the old data the priming kernel cached
//   is gone before hipFileRead. With flag or sleep, the priming kernel instead
//   sets a flag in host memory for each workgroup, and the host either spins
//   on the flags or sleeps and then checks them. Nothing is enqueued between
//   the priming kernel and hipFileRead, so the old data can still be in L2 when
//   the checking kernel runs. --read async calls hipFileReadAsync on the
//   kernels' stream instead of hipFileRead, and the checking kernel follows it
//   on that stream.
// - during: one kernel, launched before hipFileRead, waits until hipFileRead
//   returns and then checks the buffer. A kernel that is running while another
//   agent writes coarse-grained memory may see the old data, so the buffer is
//   fine-grained memory instead, which hipExtMallocWithFlags allocates. On
//   PCIe GPUs, ROCr provides fine-grained VRAM only when
//   HSA_FORCE_FINE_GRAIN_PCIE=1, and otherwise hipExtMallocWithFlags silently
//   returns coarse-grained memory, which the GPU caches in L2, so the program
//   sets that variable for this mode. The kernel waits on a flag the host
//   stores, with release semantics, once hipFileRead returns, and it acquires
//   the flag at system scope before reading the buffer, which makes the new
//   data visible to it.
//
// The test runs on HIP device 0, the first GPU the process can see. To test a
// different GPU, make it the only one the process can see by setting
// ROCR_VISIBLE_DEVICES to that GPU's index or UUID.
//
// hipFile chooses the appropriate backend for the file in --directory, using
// fastpath when it can and fallback otherwise. To control which backend the
// test uses, set hipFile's environment variables:
// HIPFILE_FORCE_COMPAT_MODE=true makes every read use the fallback backend,
// and HIPFILE_ALLOW_COMPAT_MODE=false makes every read use fastpath. In the
// second case, a read that can't use fastpath fails, so the test fails where
// fastpath is unavailable. Run with --help for options.
//
// For a soak test, run the program directly with --iterations -1, which runs
// until a check fails or the program receives SIGTERM or SIGINT. A signal
// stops the test after the current iteration, and the program prints how many
// iterations passed. Run it for each backend and --kernel, for example, for
// fastpath with a primed kernel running during hipFileRead:
//
//   HIPFILE_ALLOW_COMPAT_MODE=false timeout 8h hipfileread-gpu-visibility --directory DIR
//     --iterations -1 --kernel during --prime

#include "args.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"
#include "hipfileread-gpu-visibility-kernel.h"
#include "test-common.h"
#include "test-options.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/types.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

using namespace hipFileTest;

namespace {

/// @brief --buffer-size must be a multiple of this, so that the second
/// pattern's file offset suits direct I/O
constexpr size_t BUFFER_SIZE_GRANULARITY{4_KiB};

/// @brief Largest piece of a pattern the program writes, and copies into the
/// GPU buffer, at once
constexpr size_t PATTERN_CHUNK_SIZE{16_MiB};

/// @brief How long a running kernel waits for hipFileRead to return, and the
/// host waits for the kernel's workgroups to start, before giving up
constexpr std::chrono::seconds READ_TIMEOUT{60};

/// @brief Index of the first ready flag in the host signals, which keeps the
/// flags the GPU writes off the cache line of the ReadState the host writes
constexpr size_t READY_FLAGS_OFFSET{64 / sizeof(uint32_t)};

/// @brief How long the host sleeps for the priming kernel with --prime-wait sleep
constexpr std::chrono::milliseconds PRIME_SLEEP{50};

/// @brief When the kernel that checks the buffer runs, relative to hipFileRead
enum class KernelTiming {
    /// @brief The kernel launches after hipFileRead returns
    After,
    /// @brief The kernel launches before hipFileRead and waits for it to return
    During,
};

/// @brief How the host waits for the priming kernel before hipFileRead, with
/// KernelTiming::After
enum class PrimeWait {
    /// @brief hipStreamSynchronize, which also invalidates the GPU's L2 cache
    Sync,
    /// @brief Spin on flags the priming kernel sets in host memory
    Flag,
    /// @brief Sleep for PRIME_SLEEP, then check the flags
    Sleep,
};

/// @brief Which hipFile call reads the file
enum class ReadApi {
    /// @brief hipFileRead
    Sync,
    /// @brief hipFileReadAsync on the kernels' stream
    Async,
};

/// @brief Command-line options that control the check
struct Options {
    /// @brief Directory for the temporary test file
    std::string directory;
    /// @brief Size of the GPU buffer each hipFileRead fills, in bytes
    size_t buffer_size;
    /// @brief Number of hipFileRead calls to check, or unset to run until a check
    /// fails or the program receives SIGTERM or SIGINT
    std::optional<size_t> iterations;
    /// @brief When the kernel that checks the buffer runs
    KernelTiming timing;
    /// @brief Whether a kernel reads the buffer before hipFileRead
    bool prime;
    /// @brief How the host waits for the priming kernel, with KernelTiming::After
    PrimeWait prime_wait;
    /// @brief Which hipFile call reads the file
    ReadApi read_api;
    /// @brief Number of workgroups in each kernel
    unsigned int workgroups;
};

/// @brief Collects failure descriptions
class Failures {
public:
    /// @brief Records a failure
    ///
    /// @tparam Parts Types of the values that make up the description
    /// @param parts Values streamed one after another to form the description
    template <typename... Parts> void add(const Parts &...parts)
    {
        std::ostringstream description;
        (description << ... << parts);
        descriptions.push_back(description.str());
    }

    /// @brief Checks whether any failure has been recorded
    ///
    /// @return true if add() has been called
    bool any() const
    {
        return !descriptions.empty();
    }

    /// @brief Hands over the recorded failures
    ///
    /// @return The failure descriptions, in the order they were recorded
    std::vector<std::string> take()
    {
        return std::move(descriptions);
    }

private:
    /// @brief Failure descriptions, in the order they were recorded
    std::vector<std::string> descriptions{};
};

/// @brief Owns the test file, its registered hipFile handle, the GPU buffer,
/// and what the kernels use to synchronize with the host and report results
struct Resources {
    /// @brief Temporary test file in Options::directory
    std::optional<Tmpfile> tmpfile;
    /// @brief Registered handle for the test file, or nullptr
    hipFileHandle_t handle{nullptr};
    /// @brief GPU buffer, or nullptr
    void *buffer{nullptr};
    /// @brief Fine-grained host memory from hipHostMalloc for the ReadState and
    /// the workgroups' ready flags, or nullptr
    void *signals{nullptr};
    /// @brief GPU memory from hipMalloc for the workgroups' results, or nullptr
    void *results{nullptr};
    /// @brief Non-blocking stream the kernels run on, so that hipFileRead's own
    /// copies on the null stream don't wait for a running kernel, or nullptr
    hipStream_t stream{nullptr};

    /// @brief Creates an object that owns nothing yet
    Resources() = default;
    /// @brief Not copyable, since the object owns the file, handle, memory, and stream
    Resources(const Resources &) = delete;
    /// @brief Not copyable, since the object owns the file, handle, memory, and stream
    Resources &operator=(const Resources &) = delete;

    /// @brief Destroys the stream, frees the memory, and deregisters the handle
    ///
    /// The tmpfile member closes the test file afterward, as members are
    /// destroyed after the destructor's body runs.
    ~Resources()
    {
        if (stream) {
            (void)hipStreamDestroy(stream);
        }
        if (results) {
            (void)hipFree(results);
        }
        if (signals) {
            (void)hipHostFree(signals);
        }
        if (buffer) {
            (void)hipFree(buffer);
        }
        if (handle) {
            hipFileHandleDeregister(handle);
        }
    }
};

/// @brief Gets a pointer into GPU memory at a byte offset
///
/// GPU memory arrives from the HIP runtime as a raw pointer, so addresses within
/// it are computed as integers rather than with pointer arithmetic.
///
/// @param buffer Start of the GPU buffer
/// @param byte_offset Offset from the start of the buffer, in bytes
///
/// @return Pointer to the byte at byte_offset
void *
bytesAt(void *buffer, size_t byte_offset)
{
    return reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(buffer) + byte_offset);
}

/// @brief Parses a size given in bytes or with a K, M, or G suffix
///
/// The suffixes are binary and either case: K is 1024 bytes, M is 1024 K, and
/// G is 1024 M.
///
/// @param text The size, such as 4096, 4K, 2M, or 1G
///
/// @return The size in bytes, or nothing if text is not a size or the size
///         does not fit in size_t
std::optional<size_t>
parseSize(std::string_view text)
{
    const size_t           digits_end{std::min(text.find_first_not_of("0123456789"), text.size())};
    const std::string_view digits{text.substr(0, digits_end)};
    const std::string_view suffix{text.substr(digits_end)};
    if (digits.empty()) {
        return std::nullopt;
    }

    size_t     value{};
    const auto parsed{std::from_chars(std::to_address(digits.begin()), std::to_address(digits.end()), value)};
    if (parsed.ec != std::errc{}) {
        return std::nullopt;
    }

    size_t multiplier{1};
    if (suffix == "K" || suffix == "k") {
        multiplier = 1_KiB;
    }
    else if (suffix == "M" || suffix == "m") {
        multiplier = 1_MiB;
    }
    else if (suffix == "G" || suffix == "g") {
        multiplier = 1_GiB;
    }
    else if (!suffix.empty()) {
        return std::nullopt;
    }
    if (value > std::numeric_limits<size_t>::max() / multiplier) {
        return std::nullopt;
    }
    return value * multiplier;
}

/// @brief Gets the most bytes one hipFileRead can read
///
/// hipFile documents the limit, which the Linux kernel sets, as 2^31 bytes
/// minus the system page size.
///
/// @return The limit in bytes
size_t
maxReadSize()
{
    const long page_size{sysconf(_SC_PAGESIZE)};
    return (size_t{1} << 31) - (page_size > 0 ? static_cast<size_t>(page_size) : size_t{4_KiB});
}

/// @brief Writes all of a buffer to a file, continuing after partial writes
///
/// @param fd The file
/// @param data The bytes to write
/// @param offset File offset to write them at
///
/// @return true if every byte was written; false, with errno set, otherwise
bool
writeAll(int fd, std::span<const std::byte> data, hoff_t offset)
{
    while (!data.empty()) {
        const ssize_t written{pwrite(fd, data.data(), data.size(), offset)};
        if (written <= 0) {
            if (written == 0) {
                errno = ENOSPC;
            }
            return false;
        }
        data = data.subspan(static_cast<size_t>(written));
        offset += written;
    }
    return true;
}

/// @brief The signal that asked the check to stop, or 0 if none has
///
/// Set from a signal handler, so it must be lock-free.
std::atomic<int> stop_signal{0};
static_assert(std::atomic<int>::is_always_lock_free);

/// @brief Signal handler that asks the check to stop after the current iteration
///
/// @param signal_number The signal received
void
requestStop(int signal_number)
{
    stop_signal.store(signal_number, std::memory_order_relaxed);
}

/// @brief Makes SIGTERM and SIGINT stop the check after the current iteration
///
/// glibc's signal() restarts the system calls a signal interrupts, so the read
/// in progress doesn't fail. The handlers stay installed, because timeout sends
/// its signal twice; SIGKILL ends a run whose read hangs.
///
/// @return true if the handlers were installed; false, with errno set, otherwise
bool
stopOnSignals()
{
    return std::signal(SIGTERM, requestStop) != SIG_ERR && std::signal(SIGINT, requestStop) != SIG_ERR;
}

/// @brief Describes when the kernel that checks the buffer runs, for messages
///
/// @param options Supplies the timing, whether and how the buffer is primed, and
///        the hipFile call
///
/// @return The description
std::string
kernelDescription(const Options &options)
{
    const std::string_view read{options.read_api == ReadApi::Async ? "hipFileReadAsync" : "hipFileRead"};
    std::string            description{options.timing == KernelTiming::After ? "a kernel launched after "
                                                                             : "a kernel running during "};
    description += read;
    if (options.prime) {
        description += ", primed";
        if (options.timing == KernelTiming::After) {
            description += options.prime_wait == PrimeWait::Sync   ? " and synchronized"
                           : options.prime_wait == PrimeWait::Flag ? " and waited on with flags"
                                                                   : " and waited on with a sleep";
        }
    }
    return description;
}

/// @brief Checks that data each hipFileRead writes into GPU memory is observable
/// by GPU kernels once hipFileRead returns
///
/// Writes two buffer-sized patterns to the start of the file once, then runs
/// options.iterations iterations, or runs without a limit if it is unset. Each
/// reads whichever pattern is not in the buffer, and a kernel, timed as
/// options.timing says, checks for it. Stops after the first iteration with a
/// failure, or after the current iteration once SIGTERM or SIGINT arrives.
///
/// @param resources The test file, its registered handle, the GPU buffer, the
///        host signals, the results, and the stream, all ready to use
/// @param options Supplies the buffer size, iteration count, and kernel options
/// @param[out] passed_iterations Set to the number of iterations that passed
///
/// @return A description of each failure; empty if the check passed
std::vector<std::string>
checkReadVisibility(const Resources &resources, const Options &options, size_t &passed_iterations)
{
    Failures failures;
    passed_iterations = 0;

    const int    fd{resources.tmpfile->fd};
    const size_t buffer_size{options.buffer_size};
    // The kernels identify a value's word by the upper 32 bits of a 64-bit
    // word, and each pattern's values must be unique 32-bit values
    const size_t word_count{buffer_size / sizeof(uint32_t)};
    if (2 * word_count > size_t{UINT32_MAX} + 1) {
        failures.add("The buffer has too many words (", word_count, ") for unique 32-bit values");
        return failures.take();
    }
    auto pattern_base{[word_count](size_t pattern) { return static_cast<uint32_t>(pattern * word_count); }};

    // Build the patterns a chunk at a time, so a large buffer doesn't need twice
    // its size in host memory. The buffer starts with pattern 0, and iteration i
    // reads pattern (i + 1) % 2.
    std::vector<uint32_t> chunk(std::min(buffer_size, PATTERN_CHUNK_SIZE) / sizeof(uint32_t));
    for (size_t pattern{0}; pattern < 2; pattern++) {
        for (size_t chunk_offset{0}; chunk_offset < buffer_size; chunk_offset += PATTERN_CHUNK_SIZE) {
            const size_t chunk_bytes{std::min(PATTERN_CHUNK_SIZE, buffer_size - chunk_offset)};
            const size_t first_word{chunk_offset / sizeof(uint32_t)};
            const auto   values{std::span(chunk).first(chunk_bytes / sizeof(uint32_t))};
            for (size_t i{0}; i < values.size(); i++) {
                values[i] = pattern_base(pattern) + static_cast<uint32_t>(first_word + i);
            }
            if (!writeAll(fd, std::as_bytes(values),
                          static_cast<hoff_t>(pattern * buffer_size + chunk_offset))) {
                failures.add("Writing the test patterns failed: ", std::strerror(errno));
                return failures.take();
            }
            if (pattern == 0 && hipMemcpy(bytesAt(resources.buffer, chunk_offset), values.data(), chunk_bytes,
                                          hipMemcpyHostToDevice) != hipSuccess) {
                failures.add("Initializing the GPU buffer failed");
                return failures.take();
            }
        }
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        failures.add("Initializing the GPU buffer failed");
        return failures.take();
    }
    // The fastpath's direct reads bypass the page cache, so the patterns must be
    // in storage before the first hipFileRead
    if (fdatasync(fd) != 0) {
        failures.add("Flushing the test patterns to storage failed: ", std::strerror(errno));
        return failures.take();
    }

    int device{};
    int wall_clock_khz{};
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&wall_clock_khz, hipDeviceAttributeWallClockRate, device) != hipSuccess ||
        wall_clock_khz <= 0) {
        failures.add("Unable to query the GPU's wall clock rate");
        return failures.take();
    }
    // PROBE (scratch only, not committed): write-only HDP flush between hipFileRead and the checker
    volatile unsigned int *hdp_flush{nullptr};
    if (std::getenv("PROBE_HDP_FLUSH")) {
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, device) != hipSuccess || prop.hdpMemFlushCntl == nullptr) {
            failures.add("PROBE: HDP flush register unavailable");
            return failures.take();
        }
        hdp_flush = prop.hdpMemFlushCntl;
        std::cerr << "PROBE: write-only HDP flush via " << static_cast<const volatile void *>(hdp_flush)
                  << " after each hipFileRead\n";
    }
    const uint64_t timeout_ticks{static_cast<uint64_t>(wall_clock_khz) * 1000 *
                                 static_cast<uint64_t>(READ_TIMEOUT.count())};

    // The host signals: the ReadState, then a ready flag for each workgroup.
    // hipHostMalloc returns a raw pointer, so a span of it can only come from a
    // pointer and a length.
    HIPFILE_WARN_UNSAFE_BUFFER_USAGE_OFF
    const std::span<uint32_t> signals{static_cast<uint32_t *>(resources.signals),
                                      READY_FLAGS_OFFSET + options.workgroups};
    HIPFILE_WARN_UNSAFE_BUFFER_USAGE_ON
    uint32_t                 &read_state{signals.front()};
    const std::span<uint32_t> ready{signals.subspan(READY_FLAGS_OFFSET)};

    // The checking kernel's results, then the priming kernel's. Both are copied
    // back only after the checking kernel, as a blocking copy between the
    // priming kernel and hipFileRead would invalidate the GPU's L2 cache.
    std::vector<VisibilityResult> results(2 * size_t{options.workgroups});
    auto                         *check_results{static_cast<VisibilityResult *>(resources.results)};
    auto                         *prime_results{static_cast<VisibilityResult *>(
        bytesAt(resources.results, options.workgroups * sizeof(VisibilityResult)))};

    // Waits until every workgroup has set its ready flag
    auto wait_for_ready{[&ready](std::chrono::steady_clock::time_point deadline) {
        auto   is_ready{[](uint32_t &flag) {
            return std::atomic_ref<uint32_t>{flag}.load(std::memory_order_acquire) != 0;
        }};
        size_t started{0};
        while ((started = static_cast<size_t>(std::ranges::count_if(ready, is_ready))) < ready.size() &&
               std::chrono::steady_clock::now() < deadline) {
            std::this_thread::yield();
        }
        return started;
    }};

    for (size_t iteration{0}; (!options.iterations || iteration < *options.iterations) && !failures.any() &&
                              stop_signal.load(std::memory_order_relaxed) == 0;
         iteration++) {
        const size_t previous_pattern{iteration % 2};
        const size_t current_pattern{(iteration + 1) % 2};

        VisibilityKernelArgs prime_args{
            .buffer         = static_cast<const uint32_t *>(resources.buffer),
            .word_count     = word_count,
            .previous_base  = pattern_base(previous_pattern),
            .current_base   = pattern_base(current_pattern),
            .check_previous = true,
            .check_current  = false,
            .read_state     = nullptr,
            .ready          = nullptr,
            .timeout_ticks  = timeout_ticks,
            .results        = prime_results,
        };
        VisibilityKernelArgs check_args{prime_args};
        check_args.check_previous = false;
        check_args.check_current  = true;
        check_args.results        = check_results;

        std::atomic_ref<uint32_t>{read_state}.store(static_cast<uint32_t>(ReadState::InProgress),
                                                    std::memory_order_relaxed);
        for (uint32_t &flag : ready) {
            std::atomic_ref<uint32_t>{flag}.store(0, std::memory_order_relaxed);
        }

        if (options.timing == KernelTiming::During) {
            check_args.check_previous = options.prime;
            check_args.read_state     = &read_state;
            check_args.ready          = ready.data();
            if (launchVisibilityKernel(check_args, options.workgroups, resources.stream) != hipSuccess) {
                failures.add("Iteration ", iteration, ": launching the kernel failed");
                break;
            }

            // Start the read only once every workgroup is running and, with
            // --prime, has read the previous pattern
            const size_t started{wait_for_ready(std::chrono::steady_clock::now() + READ_TIMEOUT)};
            if (started < ready.size()) {
                failures.add("Iteration ", iteration, ": only ", started, " of ", ready.size(),
                             " workgroups started within ", READ_TIMEOUT.count(),
                             " s; they must all fit on the GPU at once");
                std::atomic_ref<uint32_t>{read_state}.store(static_cast<uint32_t>(ReadState::Failed),
                                                            std::memory_order_release);
                (void)hipStreamSynchronize(resources.stream);
                break;
            }
        }
        else if (options.prime) {
            if (options.prime_wait != PrimeWait::Sync) {
                prime_args.ready = ready.data();
            }
            if (launchVisibilityKernel(prime_args, options.workgroups, resources.stream) != hipSuccess) {
                failures.add("Iteration ", iteration, ": launching the kernel that primes the buffer failed");
                break;
            }
            if (options.prime_wait == PrimeWait::Sync) {
                if (hipStreamSynchronize(resources.stream) != hipSuccess) {
                    failures.add("Iteration ", iteration, ": the kernel that primes the buffer failed");
                    break;
                }
            }
            else {
                // Nothing that enqueues GPU work may run between here and the read
                if (options.prime_wait == PrimeWait::Sleep) {
                    std::this_thread::sleep_for(PRIME_SLEEP);
                }
                const auto   deadline{options.prime_wait == PrimeWait::Flag
                                          ? std::chrono::steady_clock::now() + READ_TIMEOUT
                                          : std::chrono::steady_clock::now()};
                const size_t finished{wait_for_ready(deadline)};
                if (finished < ready.size()) {
                    failures.add(
                        "Iteration ", iteration, ": only ", finished, " of ", ready.size(),
                        " workgroups of the kernel that primes the buffer finished reading it within ",
                        options.prime_wait == PrimeWait::Flag ? std::to_string(READ_TIMEOUT.count()) + " s"
                                                              : std::to_string(PRIME_SLEEP.count()) + " ms");
                    (void)hipStreamSynchronize(resources.stream);
                    break;
                }
            }
        }

        // hipFileReadAsync reads these when its operation runs on the stream, so
        // they must outlive it; the stream is synchronized within the iteration
        size_t      read_size{buffer_size};
        hoff_t      file_offset{static_cast<hoff_t>(current_pattern * buffer_size)};
        hoff_t      buffer_offset{0};
        ssize_t     nbytes{0};
        int         read_errno{0};
        const bool  async_read{options.read_api == ReadApi::Async};
        const char *read_name{async_read ? "hipFileReadAsync" : "hipFileRead"};
        if (async_read) {
            const hipFileError_t err{hipFileReadAsync(resources.handle, resources.buffer, &read_size,
                                                      &file_offset, &buffer_offset, &nbytes,
                                                      resources.stream)};
            if (err.err != hipFileSuccess) {
                failures.add("Iteration ", iteration, ": hipFileReadAsync failed to enqueue the read: ",
                             hipFileGetOpErrorString(err.err));
                (void)hipStreamSynchronize(resources.stream);
                break;
            }
        }
        else {
            nbytes     = hipFileRead(resources.handle, resources.buffer, buffer_size, file_offset, 0);
            read_errno = errno;
            if (hdp_flush != nullptr) {
                // Store only; HIP's hipDeviceFlushGPUDirectRDMAWrites also reads the register back,
                // which upstream amdgpu stopped doing after it caused AER errors
                *hdp_flush = 1U;
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }
        }

        // Report how the read failed; with hipFileReadAsync, nbytes is known only
        // once the stream has been synchronized
        auto check_read{[&]() {
            const bool read_succeeded{std::cmp_equal(nbytes, buffer_size)};
            // hipFileRead sets errno only when it returns -1; other negative values are
            // negated hipFileOpError_t codes
            if (nbytes == -1 && !async_read) {
                failures.add("Iteration ", iteration, ": hipFileRead failed: ", std::strerror(read_errno));
            }
            else if (nbytes < 0) {
                failures.add("Iteration ", iteration, ": ", read_name, " failed with hipFile error ", -nbytes,
                             ": ", hipFileGetOpErrorString(static_cast<hipFileOpError_t>(-nbytes)));
            }
            else if (!read_succeeded) {
                failures.add("Iteration ", iteration, ": ", read_name, " read ", nbytes, " of ", buffer_size,
                             " bytes");
            }
            return read_succeeded;
        }};

        if (options.timing == KernelTiming::During) {
            const bool read_succeeded{check_read()};
            std::atomic_ref<uint32_t>{read_state}.store(
                static_cast<uint32_t>(read_succeeded ? ReadState::Succeeded : ReadState::Failed),
                std::memory_order_release);
        }
        else if (!async_read && !check_read()) {
            (void)hipStreamSynchronize(resources.stream);
            break;
        }
        // With hipFileReadAsync, the stream runs the checking kernel after the read
        else if (launchVisibilityKernel(check_args, options.workgroups, resources.stream) != hipSuccess) {
            failures.add("Iteration ", iteration, ": launching the kernel failed");
            (void)hipStreamSynchronize(resources.stream);
            break;
        }
        if (hipStreamSynchronize(resources.stream) != hipSuccess) {
            failures.add("Iteration ", iteration, ": the kernel that checks the buffer failed");
            break;
        }
        if (async_read && !check_read()) {
            break;
        }
        if (failures.any()) {
            break;
        }
        if (hipMemcpy(results.data(), resources.results, results.size() * sizeof(VisibilityResult),
                      hipMemcpyDeviceToHost) != hipSuccess) {
            failures.add("Iteration ", iteration, ": copying the kernels' results failed");
            break;
        }

        if (options.timing == KernelTiming::After && options.prime) {
            for (size_t w{0}; w < options.workgroups; w++) {
                const VisibilityResult &result{results[options.workgroups + w]};
                if (result.previous_mismatches != 0) {
                    failures.add("Iteration ", iteration, ", workgroup ", w, ": ", result.previous_mismatches,
                                 " values did not hold the previous pattern before ", read_name);
                }
            }
        }
        for (size_t w{0}; w < options.workgroups; w++) {
            const VisibilityResult &result{results[w]};
            if (result.previous_mismatches != 0) {
                failures.add("Iteration ", iteration, ", workgroup ", w, ": ", result.previous_mismatches,
                             " values did not hold the previous pattern before hipFileRead");
            }
            if (result.timed_out != 0) {
                failures.add("Iteration ", iteration, ", workgroup ", w, ": timed out after ",
                             READ_TIMEOUT.count(), " s waiting for hipFileRead to return");
            }
            if (result.stale != 0 || result.unexpected != 0) {
                const size_t   word{result.first_bad >> 32};
                const uint32_t observed{static_cast<uint32_t>(result.first_bad)};
                failures.add("Iteration ", iteration, ", workgroup ", w, ": ", result.stale, " stale and ",
                             result.unexpected,
                             " unexpected values after hipFileRead returned; the first, at byte offset ",
                             word * sizeof(uint32_t), ", was ", observed, " instead of ",
                             pattern_base(current_pattern) + static_cast<uint32_t>(word));
            }
        }
        if (!failures.any()) {
            passed_iterations++;
        }
    }

    return failures.take();
}

/// @brief Runs the check on a temporary file
///
/// Creates the temporary test file, registers it, and allocates the GPU
/// buffer, the host signals, the results, and the stream before running the
/// check.
///
/// @param options Command-line options
///
/// @return true if the check passes; false if it fails or cannot run
bool
runCheck(const Options &options)
{
    Resources resources;
    try {
        resources.tmpfile.emplace(options.directory);
    }
    catch (const std::runtime_error &e) {
        std::cerr << e.what() << '\n';
        return false;
    }

    hipFileDescr_t descr{};
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = resources.tmpfile->fd;

    if (hipFileHandleRegister(&resources.handle, &descr).err != hipFileSuccess) {
        resources.handle = nullptr;
        std::cerr << "hipFileHandleRegister failed\n";
        return false;
    }
    // A kernel that is already running is only guaranteed to see another agent's
    // writes to fine-grained memory
    const hipError_t buffer_err{
        options.timing == KernelTiming::During
            ? hipExtMallocWithFlags(&resources.buffer, options.buffer_size, hipDeviceMallocFinegrained)
            : hipMalloc(&resources.buffer, options.buffer_size)};
    if (buffer_err != hipSuccess) {
        resources.buffer = nullptr;
        std::cerr << (options.timing == KernelTiming::During ? "hipExtMallocWithFlags" : "hipMalloc")
                  << " failed for the GPU buffer\n";
        return false;
    }
    if (hipHostMalloc(&resources.signals, (READY_FLAGS_OFFSET + options.workgroups) * sizeof(uint32_t),
                      hipHostMallocCoherent) != hipSuccess) {
        resources.signals = nullptr;
        std::cerr << "hipHostMalloc failed\n";
        return false;
    }
    // Room for the checking kernel's results, then the priming kernel's
    if (hipMalloc(&resources.results, 2 * size_t{options.workgroups} * sizeof(VisibilityResult)) !=
        hipSuccess) {
        resources.results = nullptr;
        std::cerr << "hipMalloc failed for the kernel's results\n";
        return false;
    }
    if (hipStreamCreateWithFlags(&resources.stream, hipStreamNonBlocking) != hipSuccess) {
        resources.stream = nullptr;
        std::cerr << "hipStreamCreateWithFlags failed\n";
        return false;
    }

    size_t     passed_iterations{0};
    const auto start{std::chrono::steady_clock::now()};
    const auto failures{checkReadVisibility(resources, options, passed_iterations)};
    const auto elapsed{std::chrono::duration<double>{std::chrono::steady_clock::now() - start}};

    const int signal_number{stop_signal.load(std::memory_order_relaxed)};
    if (signal_number != 0) {
        std::cerr << "Stopped by " << (signal_number == SIGTERM ? "SIGTERM" : "SIGINT") << '\n';
    }
    if (!failures.empty()) {
        for (const auto &failure : failures) {
            std::cerr << failure << '\n';
        }
        std::cerr << passed_iterations << " iterations passed before the failure\n";
        return false;
    }

    std::cout << passed_iterations << " iterations of " << options.buffer_size << " bytes with "
              << kernelDescription(options) << ", on " << options.workgroups
              << (options.workgroups == 1 ? " workgroup" : " workgroups") << ", passed in " << elapsed.count()
              << " s\n";
    return true;
}

}

/// @brief Parses the options and runs the check
///
/// Runs the check on HIP device 0 with whichever backend hipFile chooses.
///
/// @param argc Number of command-line arguments
/// @param argv Command-line arguments
///
/// @return EXIT_SUCCESS if the check passed, EXIT_FAILURE otherwise
int
main(int argc, char **argv)
{
    args::ArgumentParser parser{
        "Checks that data hipFileRead writes into GPU memory is observable by GPU kernels once hipFileRead "
        "returns. hipFile chooses the backend; set HIPFILE_FORCE_COMPAT_MODE=true to test only fallback, "
        "or HIPFILE_ALLOW_COMPAT_MODE=false to test only fastpath. The test runs on HIP device 0; set "
        "ROCR_VISIBLE_DEVICES to test a different GPU."};
    [[maybe_unused]] args::HelpFlag help_arg(parser, "help", "Show this help message",
                                             args::Matcher{'h', "help"});

    args::ValueFlag<std::string> directory_arg(
        parser, "path",
        "Directory for the temporary test file, which is deleted on exit; fastpath needs an AIS capable "
        "file system (required)",
        args::Matcher{"directory"}, args::Options::Single);

    args::ValueFlag<std::string> buffer_size_arg(
        parser, "size",
        "Size of the GPU buffer, in bytes or with a K, M, or G suffix. It must be a multiple of 4K and at "
        "most 2G minus the page size (default: 1M)",
        args::Matcher{"buffer-size"}, "1M", args::Options::Single);

    args::ValueFlag<long long> iterations_arg(
        parser, "count", "hipFileRead calls to check, or -1 to run until SIGTERM or SIGINT (required)",
        args::Matcher{"iterations"}, args::Options::Single);

    const std::unordered_map<std::string, KernelTiming> timings{{"after", KernelTiming::After},
                                                                {"during", KernelTiming::During}};

    args::MapFlag<std::string, KernelTiming> timing_arg(
        parser, "after|during",
        "When the kernel that checks the buffer runs: launched after hipFileRead returns, with the buffer "
        "in coarse-grained memory, or launched before hipFileRead and waiting for it to return, with the "
        "buffer in fine-grained memory (default: after)",
        args::Matcher{"kernel"}, timings, KernelTiming::After, args::Options::Single);

    args::Flag prime_arg(parser, "prime",
                         "Before hipFileRead, read the buffer from the GPU and check it holds the previous "
                         "pattern, which brings it into the GPU's caches",
                         args::Matcher{"prime"});

    const std::unordered_map<std::string, PrimeWait> prime_waits{
        {"sync", PrimeWait::Sync}, {"flag", PrimeWait::Flag}, {"sleep", PrimeWait::Sleep}};

    args::MapFlag<std::string, PrimeWait> prime_wait_arg(
        parser, "sync|flag|sleep",
        "With --kernel after --prime, how the host waits for the priming kernel before reading: "
        "hipStreamSynchronize, which also invalidates the GPU's L2 cache; spinning on flags the kernel "
        "sets in host memory; or sleeping, then checking the flags. Only sync enqueues GPU work between "
        "the priming kernel and the read (default: sync)",
        args::Matcher{"prime-wait"}, prime_waits, PrimeWait::Sync, args::Options::Single);

    const std::unordered_map<std::string, ReadApi> read_apis{{"sync", ReadApi::Sync},
                                                             {"async", ReadApi::Async}};

    args::MapFlag<std::string, ReadApi> read_api_arg(
        parser, "sync|async",
        "Read with hipFileRead, or with hipFileReadAsync on the kernels' stream, which needs --kernel after "
        "(default: sync)",
        args::Matcher{"read"}, read_apis, ReadApi::Sync, args::Options::Single);

    args::ValueFlag<unsigned int> workgroups_arg(
        parser, "count",
        "Workgroups in each kernel, each checking an equal share of the buffer (default and maximum: the "
        "GPU's compute units, or the buffer's 32-bit words if fewer)",
        args::Matcher{"workgroups"}, args::Options::Single);
    parseCli(parser, argc, argv);

    // Without this, hipExtMallocWithFlags(hipDeviceMallocFinegrained) silently
    // returns coarse-grained memory on PCIe GPUs. ROCr reads the variable when
    // HIP initializes, so it must be set before the first HIP call.
    if (args::get(timing_arg) == KernelTiming::During && setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1", 1) != 0) {
        std::cerr << "Unable to set HSA_FORCE_FINE_GRAIN_PCIE: " << std::strerror(errno) << '\n';
        return EXIT_FAILURE;
    }

    int device{};
    int compute_units{};
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&compute_units, hipDeviceAttributeMultiprocessorCount, device) != hipSuccess ||
        compute_units <= 0) {
        std::cerr << "Unable to query HIP device " << device << '\n';
        return EXIT_FAILURE;
    }

    const std::optional<size_t> buffer_size{parseSize(args::get(buffer_size_arg))};
    // Every workgroup of a running kernel waits for hipFileRead, so they must
    // all fit on the GPU at once; one per compute unit always does
    const auto max_workgroups{
        static_cast<unsigned int>(std::min(static_cast<size_t>(compute_units),
                                           std::max(buffer_size.value_or(0) / sizeof(uint32_t), size_t{1})))};

    const long long iterations{args::get(iterations_arg)};

    const Options options{
        .directory   = args::get(directory_arg),
        .buffer_size = buffer_size.value_or(0),
        .iterations  = iterations >= 0 ? std::optional{static_cast<size_t>(iterations)} : std::nullopt,
        .timing      = args::get(timing_arg),
        .prime       = args::get(prime_arg),
        .prime_wait  = args::get(prime_wait_arg),
        .read_api    = args::get(read_api_arg),
        .workgroups  = workgroups_arg ? args::get(workgroups_arg) : max_workgroups,
    };

    if (!iterations_arg) {
        std::cerr << "--iterations is required\n";
        return EXIT_FAILURE;
    }
    if (iterations < -1) {
        std::cerr << "--iterations must be at least 0, or -1 to run until the program is killed\n";
        return EXIT_FAILURE;
    }
    if (options.directory.empty()) {
        std::cerr << "--directory is required\n";
        return EXIT_FAILURE;
    }
    if (!buffer_size) {
        std::cerr << "--buffer-size must be a number of bytes, optionally followed by K, M, or G: "
                  << args::get(buffer_size_arg) << '\n';
        return EXIT_FAILURE;
    }
    if (options.buffer_size == 0 || options.buffer_size % BUFFER_SIZE_GRANULARITY != 0) {
        std::cerr << "--buffer-size must be a positive multiple of 4K\n";
        return EXIT_FAILURE;
    }
    if (options.buffer_size > maxReadSize()) {
        std::cerr << "--buffer-size can be at most " << maxReadSize()
                  << " bytes, the most one hipFileRead can read\n";
        return EXIT_FAILURE;
    }
    if (options.workgroups == 0 || options.workgroups > max_workgroups) {
        std::cerr << "--workgroups must be from 1 to " << max_workgroups << '\n';
        return EXIT_FAILURE;
    }
    if (prime_wait_arg && (options.timing != KernelTiming::After || !options.prime)) {
        std::cerr << "--prime-wait only applies to --kernel after --prime\n";
        return EXIT_FAILURE;
    }
    if (options.read_api == ReadApi::Async && options.timing != KernelTiming::After) {
        std::cerr << "--read async needs --kernel after, as a kernel waiting on the stream would block the "
                     "read\n";
        return EXIT_FAILURE;
    }

    if (!stopOnSignals()) {
        std::cerr << "Unable to install the SIGTERM and SIGINT handlers: " << std::strerror(errno) << '\n';
        return EXIT_FAILURE;
    }

    if (!runCheck(options)) {
        return EXIT_FAILURE;
    }

    std::cout << "DONE!" << std::endl;
    return EXIT_SUCCESS;
}
