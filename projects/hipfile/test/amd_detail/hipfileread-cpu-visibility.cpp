/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Multithreading stress test that data hipFileRead writes into GPU memory is
// observable by the CPU once hipFileRead returns
//
// On XGMI-connected systems, the CPU may cache GPU memory, so this verifies
// that CPU caches are properly invalidated. On PCIe-connected systems, the CPU
// does not cache GPU memory, but CPU reads of GPU memory pass through the GPU's
// host data path (HDP), which caches reads on some GPUs, so this verifies that
// path does not return stale data.
//
// The test data, two patterns written once, goes in a temporary file in
// --directory, or at the start of the block device given with --device and
// --force. Each iteration reads whichever pattern is not in the GPU buffer
// while watcher threads, by default one per CPU, spin on it from the CPU. The
// watchers are spread evenly across both the CPUs and the buffer. Because they
// read GPU memory from the CPU, the test needs a GPU whose VRAM the CPU can
// map: a PCIe GPU with a large BAR, or one connected via XGMI. It fails on
// other GPUs, which ctest reports as skipped.
//
// The test runs on HIP device 0, the first GPU the process can see. To test a
// different GPU, make it the only one the process can see by setting
// ROCR_VISIBLE_DEVICES to that GPU's index or UUID.
//
// hipFile chooses the appropriate backend for the file in --directory or the
// device given with --device, using fastpath when it can and fallback
// otherwise. To control which backend the test uses, set hipFile's environment
// variables: HIPFILE_FORCE_COMPAT_MODE=true makes every read use the fallback
// backend, and HIPFILE_ALLOW_COMPAT_MODE=false makes every read use fastpath.
// In the second case, a read that can't use fastpath fails, so the test fails
// where fastpath is unavailable. Run with --help for options.
//
// For a soak test, run the program directly with a large --iterations and no
// --watchers, which gives one watcher per CPU, once for each backend:
//
//   HIPFILE_ALLOW_COMPAT_MODE=false hipfileread-cpu-visibility --directory DIR --iterations 100000
//   HIPFILE_FORCE_COMPAT_MODE=true hipfileread-cpu-visibility --directory DIR --iterations 100000

#include "args.h"
#include "file-descriptor.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"
#include "test-common.h"
#include "test-options.h"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <hip/driver_types.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
#include <optional>
#include <pthread.h>
#include <sched.h>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <system_error>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

using namespace hipFile;

namespace {

/// @brief --buffer-size must be a multiple of this, so that the second
/// pattern's file offset suits direct I/O
constexpr size_t BUFFER_SIZE_GRANULARITY{4_KiB};

/// @brief Largest piece of a pattern the program writes, and copies into the
/// GPU buffer, at once
constexpr size_t PATTERN_CHUNK_SIZE{16_MiB};

/// @brief Minimum distance between two objects that keeps them off each
/// other's cache lines, so that threads writing one don't slow threads using
/// the other
///
/// Clang 18 with libstdc++ doesn't provide
/// std::hardware_destructive_interference_size.
#if defined(__cpp_lib_hardware_interference_size)
constexpr size_t DESTRUCTIVE_INTERFERENCE_SIZE{std::hardware_destructive_interference_size};
#else
constexpr size_t DESTRUCTIVE_INTERFERENCE_SIZE{64};
#endif

/// @brief Command-line options that control the check
struct Options {
    /// @brief Directory for the temporary test file, or empty when testing a device
    std::string directory;
    /// @brief Block device to test, or empty when testing a file in directory
    std::string device;
    /// @brief Size of the GPU buffer each hipFileRead fills, in bytes
    size_t buffer_size;
    /// @brief Number of hipFileRead calls to check
    size_t iterations;
    /// @brief Number of watcher threads
    size_t watcher_count;
};

/// @brief State of the current iteration's hipFileRead, as the watchers see it
enum class ReadState {
    /// @brief hipFileRead has not returned yet
    InProgress,
    /// @brief hipFileRead returned and read the whole buffer
    Succeeded,
    /// @brief hipFileRead returned an error or read less than the whole buffer
    Failed,
};

/// @brief Collects failure descriptions from any thread
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
        const std::lock_guard<std::mutex> lock{mutex};
        descriptions.push_back(description.str());
        failed.store(true, std::memory_order_relaxed);
    }

    /// @brief Checks whether any failure has been recorded
    ///
    /// @return true if add() has been called
    bool any() const
    {
        return failed.load(std::memory_order_relaxed);
    }

    /// @brief Hands over the recorded failures
    ///
    /// @return The failure descriptions, in the order they were recorded
    std::vector<std::string> take()
    {
        const std::lock_guard<std::mutex> lock{mutex};
        return std::move(descriptions);
    }

private:
    /// @brief Guards descriptions
    std::mutex mutex{};
    /// @brief Failure descriptions, in the order they were recorded
    std::vector<std::string> descriptions{};
    /// @brief Set when the first failure is recorded; read without taking mutex
    std::atomic<bool> failed{false};
};

/// @brief Owns the test file or device, its registered hipFile handle, and the GPU buffer
struct Resources {
    /// @brief Temporary test file in Options::directory, when testing a file
    std::optional<Tmpfile> tmpfile;
    /// @brief Block device given by Options::device, when testing a device
    FileDescriptor device_fd;
    /// @brief Registered handle for the test file or device, or nullptr
    hipFileHandle_t handle{nullptr};
    /// @brief GPU buffer from hipMalloc, or nullptr
    void *buffer{nullptr};

    /// @brief Creates an object that owns nothing yet
    Resources() = default;
    /// @brief Not copyable, since the object owns the file, handle, and buffer
    Resources(const Resources &) = delete;
    /// @brief Not copyable, since the object owns the file, handle, and buffer
    Resources &operator=(const Resources &) = delete;

    /// @brief Frees the GPU buffer and deregisters the handle
    ///
    /// The tmpfile and device_fd members close the test file or device
    /// afterward, as members are destroyed after the destructor's body runs.
    ~Resources()
    {
        if (buffer) {
            (void)hipFree(buffer);
        }
        if (handle) {
            hipFileHandleDeregister(handle);
        }
    }

    /// @brief Gets the descriptor of the test file or device
    ///
    /// @return The descriptor, or -1 if neither is open
    int fd() const
    {
        return tmpfile ? tmpfile->fd : device_fd.get();
    }
};

/// @brief Gets the address a pointer holds, as an integer
///
/// GPU memory arrives from the HIP runtime as a raw pointer, so addresses within
/// it are computed as integers, here and in bytesAt(), rather than with pointer
/// arithmetic.
///
/// @param ptr The pointer
///
/// @return The address ptr holds
uintptr_t
address(const volatile void *ptr)
{
    return reinterpret_cast<uintptr_t>(ptr);
}

/// @brief Gets a pointer into GPU memory at a byte offset
///
/// @param buffer Start of the GPU buffer
/// @param byte_offset Offset from the start of the buffer, in bytes
///
/// @return Pointer to the byte at byte_offset
void *
bytesAt(void *buffer, size_t byte_offset)
{
    return reinterpret_cast<void *>(address(buffer) + byte_offset);
}

/// @brief Gets a 32-bit value in GPU memory that the CPU can read
///
/// @param buffer Start of the GPU buffer
/// @param byte_offset Offset of the value from the start of the buffer, in bytes
///
/// @return Pointer to the value
volatile uint32_t *
valueAt(void *buffer, size_t byte_offset)
{
    return static_cast<volatile uint32_t *>(bytesAt(buffer, byte_offset));
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

/// @brief Gets the size of the CPU's L1 data cache lines
///
/// @return The line size in bytes, or 64 if the system does not report it
size_t
cpuCacheLineSize()
{
    // sysconf returns 0 or -1 when it cannot determine the line size
    const auto reported{sysconf(_SC_LEVEL1_DCACHE_LINESIZE)};
    return reported > 0 ? static_cast<size_t>(reported) : size_t{64};
}

/// @brief Lists the CPUs this process may run on
///
/// @return The CPU numbers in increasing order, or an empty list if they cannot
///         be determined
std::vector<size_t>
allowedCpus()
{
    cpu_set_t set;
    HIPFILE_WARN_UNSAFE_BUFFER_USAGE_OFF
    CPU_ZERO(&set);
    HIPFILE_WARN_UNSAFE_BUFFER_USAGE_ON
    std::vector<size_t> cpus;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) {
        for (size_t cpu{0}; cpu < static_cast<size_t>(CPU_SETSIZE); cpu++) {
            HIPFILE_WARN_UNSAFE_BUFFER_USAGE_OFF
            if (CPU_ISSET(cpu, &set)) {
                cpus.push_back(cpu);
            }
            HIPFILE_WARN_UNSAFE_BUFFER_USAGE_ON
        }
    }
    return cpus;
}

/// @brief Pins the calling thread to one CPU
///
/// @param cpu The CPU number
///
/// @return true if the thread was pinned
bool
pinCurrentThread(size_t cpu)
{
    cpu_set_t set;
    HIPFILE_WARN_UNSAFE_BUFFER_USAGE_OFF
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    HIPFILE_WARN_UNSAFE_BUFFER_USAGE_ON
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

/// @brief Opens a block device for exclusive use as the test file
///
/// Opening a block device with O_EXCL fails with EBUSY while it is mounted or
/// another program has it open with O_EXCL, so the test does not overwrite a
/// device that is in use.
///
/// @param path Path of the block device
///
/// @return The open device, or an empty FileDescriptor, after printing why, if
///         the path is not a block device or cannot be opened
FileDescriptor
openDevice(const std::string &path)
{
    struct stat statbuf;
    if (stat(path.c_str(), &statbuf) != 0) {
        const int err{errno};
        std::cerr << "Unable to find " << path << ": " << std::strerror(err) << '\n';
        return {};
    }
    if (!S_ISBLK(statbuf.st_mode)) {
        std::cerr << path << " is not a block device; use --directory for a file on a file system\n";
        return {};
    }
    const int fd{open(path.c_str(), O_RDWR | O_EXCL | O_CLOEXEC)};
    if (fd == -1) {
        const int err{errno};
        if (err == EBUSY) {
            std::cerr << path << " is busy: it is mounted or in use by another program\n";
        }
        else {
            std::cerr << "Unable to open " << path << ": " << std::strerror(err) << '\n';
        }
        return {};
    }
    return FileDescriptor::make_managed(fd);
}

/// @brief Checks that data each hipFileRead writes into GPU memory is observable
/// by the CPU once hipFileRead returns
///
/// Writes two buffer-sized patterns to the start of the file once, then runs
/// options.iterations iterations. Each reads whichever pattern is not in the
/// buffer while watcher threads, each pinned to a CPU and spread evenly across
/// the buffer, check from the CPU that their value has changed by the time
/// hipFileRead returns. Stops after the first iteration with a failure.
///
/// @param fd Read-write descriptor of the test file or device, used to write
///        the patterns
/// @param handle Registered hipFile handle for fd, used to read the patterns
/// @param device_buffer GPU buffer to read into. It runs to the end of the GPU
///        allocation that contains it, must be CPU-accessible, and its address
///        and size must be multiples of the CPU cache line size.
/// @param options Supplies the iteration and watcher counts
///
/// @return A description of each failure; empty if the check passed
std::vector<std::string>
checkReadVisibility(int fd, hipFileHandle_t handle, void *device_buffer, const Options &options)
{
    Failures failures;

    hipDeviceptr_t allocation_base{};
    size_t         allocation_size{};
    if (hipMemGetAddressRange(&allocation_base, &allocation_size, device_buffer) != hipSuccess) {
        failures.add("Unable to find the GPU allocation that contains the buffer");
        return failures.take();
    }
    const size_t buffer_size{allocation_size - (address(device_buffer) - address(allocation_base))};

    // The file holds both patterns, the second starting at buffer_size
    if (buffer_size > std::numeric_limits<size_t>::max() / 2 || !std::in_range<hoff_t>(2 * buffer_size)) {
        failures.add("The buffer is too large to test (", buffer_size, " bytes)");
        return failures.take();
    }

    const size_t cache_line_size{cpuCacheLineSize()};
    if (cache_line_size % sizeof(uint32_t) != 0 || PATTERN_CHUNK_SIZE % cache_line_size != 0 ||
        buffer_size % cache_line_size != 0 || address(device_buffer) % cache_line_size != 0) {
        failures.add("The buffer must be aligned to, and a multiple of, the ", cache_line_size,
                     "-byte CPU cache line size");
        return failures.take();
    }
    const size_t cache_line_stride{cache_line_size / sizeof(uint32_t)};
    const size_t cache_line_count{buffer_size / cache_line_size};

    // Every value in a cache line is that line's value for the pattern: its line
    // index in pattern 0, and its line index plus cache_line_count in pattern 1.
    // Values must be unique across both patterns to detect misplaced data.
    if (2 * cache_line_count > size_t{UINT32_MAX} + 1) {
        failures.add("The buffer has too many cache lines (", cache_line_count, ") for unique 32-bit values");
        return failures.take();
    }
    auto pattern_value{[cache_line_count](size_t pattern, size_t line) {
        return static_cast<uint32_t>(line + pattern * cache_line_count);
    }};

    // Build the patterns a chunk at a time, so a large buffer doesn't need twice
    // its size in host memory. The buffer starts with pattern 0, and iteration i
    // reads pattern (i + 1) % 2.
    std::vector<uint32_t> chunk(std::min(buffer_size, PATTERN_CHUNK_SIZE) / sizeof(uint32_t));
    for (size_t pattern{0}; pattern < 2; pattern++) {
        for (size_t chunk_offset{0}; chunk_offset < buffer_size; chunk_offset += PATTERN_CHUNK_SIZE) {
            const size_t chunk_bytes{std::min(PATTERN_CHUNK_SIZE, buffer_size - chunk_offset)};
            const size_t first_line{chunk_offset / cache_line_size};
            const auto   values{std::span(chunk).first(chunk_bytes / sizeof(uint32_t))};
            for (size_t i{0}; i < values.size(); i++) {
                values[i] = pattern_value(pattern, first_line + i / cache_line_stride);
            }
            if (!writeAll(fd, std::as_bytes(values),
                          static_cast<hoff_t>(pattern * buffer_size + chunk_offset))) {
                failures.add("Writing the test patterns failed: ", std::strerror(errno));
                return failures.take();
            }
            if (pattern == 0 && hipMemcpy(bytesAt(device_buffer, chunk_offset), values.data(), chunk_bytes,
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

    const size_t watcher_count{std::min(options.watcher_count, cache_line_count)};
    if (watcher_count == 0) {
        failures.add("At least one watcher is required");
        return failures.take();
    }
    const std::vector<size_t> cpus{allowedCpus()};
    if (cpus.empty()) {
        failures.add("Unable to determine which CPUs this process may run on");
        return failures.take();
    }

    // On separate cache lines, so updates to watchers_ready don't disturb the
    // watchers spinning on read_state
    alignas(DESTRUCTIVE_INTERFERENCE_SIZE) std::atomic<ReadState> read_state{ReadState::InProgress};
    alignas(DESTRUCTIVE_INTERFERENCE_SIZE) std::atomic<size_t>    watchers_ready{0};
    std::atomic<bool>                                             done{false};
    // Marks the start and end of every iteration for the watchers and this thread
    std::barrier<> iteration_boundary{static_cast<std::ptrdiff_t>(watcher_count + 1)};

    {
        std::vector<std::jthread> watchers;

        for (size_t w{0}; w < watcher_count; w++) {
            // Spread the watchers evenly across the buffer, each on a unique cache line
            const size_t       line{w * cache_line_count / watcher_count};
            volatile uint32_t *value{valueAt(device_buffer, line * cache_line_size)};
            // Spread the watchers evenly across the CPUs too. Linux usually numbers
            // the CPUs of a chiplet or socket consecutively, so with fewer watchers
            // than CPUs, pinning them to the first CPUs would leave the other
            // chiplets and sockets unchecked.
            const size_t cpu{cpus[w * cpus.size() / watcher_count]};

            watchers.emplace_back([&, w, line, value, cpu]() {
                if (!pinCurrentThread(cpu)) {
                    failures.add("Watcher ", w, " could not be pinned to CPU ", cpu);
                }

                for (size_t iteration{0};; iteration++) {
                    iteration_boundary.arrive_and_wait();
                    if (done.load(std::memory_order_relaxed)) {
                        return;
                    }

                    const uint32_t old_value{pattern_value(iteration % 2, line)};
                    const uint32_t new_value{pattern_value((iteration + 1) % 2, line)};
                    if (*value != old_value) {
                        failures.add("Iteration ", iteration, ", watcher ", w, " (cache line ", line,
                                     "): did not observe the previous value");
                    }
                    watchers_ready.fetch_add(1, std::memory_order_release);

                    for (;;) {
                        const ReadState state{read_state.load(std::memory_order_acquire)};
                        // The acquire load orders cacheable memory, but PCIe GPU memory is mapped
                        // write-combined, which x86 allows to be read speculatively and out of order.
                        std::atomic_thread_fence(std::memory_order_seq_cst);
                        const uint32_t observed{*value};
                        if (observed == new_value) {
                            break;
                        }
                        if (observed != old_value) {
                            failures.add("Iteration ", iteration, ", watcher ", w, " (cache line ", line,
                                         "): observed an unexpected value: ", observed);
                            break;
                        }
                        if (state == ReadState::Succeeded) {
                            failures.add("Iteration ", iteration, ", watcher ", w, " (cache line ", line,
                                         "): observed a stale value after hipFileRead returned");
                            break;
                        }
                        if (state == ReadState::Failed) {
                            break;
                        }
                    }

                    iteration_boundary.arrive_and_wait();
                }
            });
        }

        for (size_t iteration{0}; iteration < options.iterations && !failures.any(); iteration++) {
            read_state.store(ReadState::InProgress, std::memory_order_relaxed);
            watchers_ready.store(0, std::memory_order_relaxed);
            iteration_boundary.arrive_and_wait();

            // Start the read only once every watcher has read the previous value
            while (watchers_ready.load(std::memory_order_acquire) < watcher_count) {
                std::this_thread::yield();
            }

            const auto    file_offset{static_cast<hoff_t>(((iteration + 1) % 2) * buffer_size)};
            const ssize_t nbytes{hipFileRead(handle, device_buffer, buffer_size, file_offset, 0)};
            const int     read_errno{errno};
            const bool    read_succeeded{std::cmp_equal(nbytes, buffer_size)};
            read_state.store(read_succeeded ? ReadState::Succeeded : ReadState::Failed,
                             std::memory_order_release);
            // hipFileRead sets errno only when it returns -1; other negative values are
            // negated hipFileOpError_t codes
            if (nbytes == -1) {
                failures.add("Iteration ", iteration, ": hipFileRead failed: ", std::strerror(read_errno));
            }
            else if (nbytes < 0) {
                failures.add("Iteration ", iteration, ": hipFileRead failed with hipFile error ", -nbytes,
                             ": ", hipFileGetOpErrorString(static_cast<hipFileOpError_t>(-nbytes)));
            }
            else if (!read_succeeded) {
                failures.add("Iteration ", iteration, ": hipFileRead returned ", nbytes, " of ", buffer_size,
                             " bytes");
            }

            iteration_boundary.arrive_and_wait();
        }

        done.store(true, std::memory_order_relaxed);
        iteration_boundary.arrive_and_wait();
    }

    return failures.take();
}

/// @brief Runs the check on the test file or device
///
/// Creates the temporary test file or opens the device, registers it, and
/// allocates the GPU buffer before running the check.
///
/// @param options Command-line options
///
/// @return true if the check passes; false if it fails or cannot run
bool
runCheck(const Options &options)
{
    Resources resources;
    if (!options.device.empty()) {
        resources.device_fd = openDevice(options.device);
        if (resources.device_fd.get() == -1) {
            return false;
        }
    }
    else {
        try {
            resources.tmpfile.emplace(options.directory);
        }
        catch (const std::runtime_error &e) {
            std::cerr << e.what() << '\n';
            return false;
        }
    }

    hipFileDescr_t descr{};
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = resources.fd();

    if (hipFileHandleRegister(&resources.handle, &descr).err != hipFileSuccess) {
        resources.handle = nullptr;
        std::cerr << "hipFileHandleRegister failed\n";
        return false;
    }
    if (hipMalloc(&resources.buffer, options.buffer_size) != hipSuccess) {
        resources.buffer = nullptr;
        std::cerr << "hipMalloc failed\n";
        return false;
    }

    const auto start{std::chrono::steady_clock::now()};
    const auto failures{checkReadVisibility(resources.fd(), resources.handle, resources.buffer, options)};
    const std::chrono::duration<double> elapsed{std::chrono::steady_clock::now() - start};

    if (!failures.empty()) {
        for (const auto &failure : failures) {
            std::cerr << failure << '\n';
        }
        return false;
    }

    std::cout << options.iterations << " iterations of " << options.buffer_size << " bytes with "
              << options.watcher_count << " watchers passed in " << elapsed.count() << " s\n";
    return true;
}

}

/// @brief Parses the options and runs the check
///
/// Fails on GPUs whose VRAM is not CPU-accessible. Otherwise runs the check on
/// HIP device 0 with whichever backend hipFile chooses.
///
/// @param argc Number of command-line arguments
/// @param argv Command-line arguments
///
/// @return EXIT_SUCCESS if the check passed, EXIT_FAILURE otherwise
int
main(int argc, char **argv)
{
    args::ArgumentParser parser{
        "Checks that data hipFileRead writes into GPU memory is observable by the CPU "
        "once hipFileRead returns. hipFile chooses the backend; set HIPFILE_FORCE_COMPAT_MODE=true "
        "to test only fallback, or HIPFILE_ALLOW_COMPAT_MODE=false to test only fastpath. The test runs "
        "on HIP device 0; set ROCR_VISIBLE_DEVICES to test a different GPU."};
    [[maybe_unused]] args::HelpFlag help_arg(parser, "help", "Show this help message",
                                             args::Matcher{'h', "help"});

    args::ValueFlag<std::string> directory_arg(
        parser, "path",
        "Directory for the temporary test file, which is deleted on exit; fastpath needs an AIS capable "
        "file system. Give either this or --device",
        args::Matcher{"directory"}, args::Options::Single);
    args::ValueFlag<std::string> device_arg(
        parser, "path",
        "NVMe namespace or partition to test instead of a file. The test overwrites the start of it, "
        "so this also needs --force",
        args::Matcher{"device"}, args::Options::Single);
    args::Flag force_arg(parser, "force", "Allow --device to overwrite the start of the device",
                         args::Matcher{"force"});

    args::ValueFlag<std::string> buffer_size_arg(
        parser, "size",
        "Size of the GPU buffer, in bytes or with a K, M, or G suffix. It must be a multiple of 4K and at "
        "most 2G minus the page size (default: 1M)",
        args::Matcher{"buffer-size"}, "1M", args::Options::Single);

    args::ValueFlag<size_t> iterations_arg(parser, "count", "hipFileRead calls to check (required)",
                                           args::Matcher{"iterations"}, args::Options::Single);
    args::ValueFlag<size_t> watchers_arg(
        parser, "count",
        "Watcher threads, each pinned to a CPU (default: one per CPU this process may run on)",
        args::Matcher{"watchers"}, std::max(size_t{1}, allowedCpus().size()), args::Options::Single);
    parseCli(parser, argc, argv);

    const std::optional<size_t> buffer_size{parseSize(args::get(buffer_size_arg))};

    const Options options{
        .directory     = args::get(directory_arg),
        .device        = args::get(device_arg),
        .buffer_size   = buffer_size.value_or(0),
        .iterations    = args::get(iterations_arg),
        .watcher_count = args::get(watchers_arg),
    };

    if (!iterations_arg) {
        std::cerr << "--iterations is required\n";
        return EXIT_FAILURE;
    }
    if (options.directory.empty() && options.device.empty()) {
        std::cerr << "Either --directory or --device is required\n";
        return EXIT_FAILURE;
    }
    if (directory_arg && device_arg) {
        std::cerr << "--directory and --device cannot be used together\n";
        return EXIT_FAILURE;
    }
    if (force_arg && !device_arg) {
        std::cerr << "--force only applies to --device\n";
        return EXIT_FAILURE;
    }
    if (device_arg && !force_arg) {
        std::cerr << "Warning: --device overwrites the start of " << args::get(device_arg)
                  << ", destroying any file system, partition table, or data there. Use --force to "
                     "run the test on it anyway.\n";
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

    // The watchers dereference GPU memory from the CPU, which faults unless VRAM
    // is mapped into the CPU's address space. Despite its name,
    // hipDeviceAttributeIsLargeBar is true both for PCIe GPUs with a large BAR
    // and for GPUs connected to the CPU via XGMI.
    int device{};
    int large_bar{};
    if (hipGetDevice(&device) != hipSuccess ||
        hipDeviceGetAttribute(&large_bar, hipDeviceAttributeIsLargeBar, device) != hipSuccess) {
        std::cerr << "Unable to query HIP device " << device << '\n';
        return EXIT_FAILURE;
    }
    if (!large_bar) {
        // Keep this marker synchronized with SKIP_REGULAR_EXPRESSION in test/amd_detail/CMakeLists.txt.
        std::cerr << "VRAM is not CPU-accessible on device " << device
                  << "; the test needs a PCIe GPU with a large BAR or a GPU connected via XGMI\n";
        return EXIT_FAILURE;
    }

    if (!runCheck(options)) {
        return EXIT_FAILURE;
    }

    std::cout << "DONE!" << std::endl;
    return EXIT_SUCCESS;
}
