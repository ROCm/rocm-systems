/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Stress test that data in GPU memory is in the file once hipFileWrite returns
//
// hipfileread-cpu-visibility and hipfileread-gpu-visibility check who can
// observe what hipFileRead wrote. This test checks the other direction: that
// hipFileWrite observes what was last written to the GPU buffer. Fastpath's
// peer-to-peer reads of VRAM go through the GPU's host data path (HDP), not
// through the GPU's caches, so this test catches data that is still in the
// GPU's caches, rather than in VRAM, when hipFileWrite reads it, data the HDP
// still holds from an earlier read or has not yet written to VRAM, and reads
// that overtake the writes they should follow.
//
// Each iteration, a producer stores a new generation of data in the GPU
// buffer, the host waits for the producer, hipFileWrite writes the buffer to a
// temporary file in --directory, and the host reads the file back from storage
// and compares it with the generation. Every 64-bit word of a generation holds
// the generation's number and the word's index, and no generation is stored
// twice, so the comparison tells data from an earlier store, and how many
// stores old it is, from data that belongs nowhere. Nothing uses the GPU
// between the wait and the comparison. When the comparison fails, the test
// writes the buffer again and compares again, which tells data that arrived
// late from data that is wrong in the buffer, then copies the buffer back with
// hipMemcpy and compares that, and stops.
//
// --producer chooses what stores the generation:
//
// - kernel: a kernel. --kernel chooses when it runs:
//   - after: the kernel finishes before hipFileWrite, with the buffer in
//     coarse-grained memory, which hipMalloc allocates. HIP makes
//     coarse-grained memory up to date at kernel boundaries, so hipFileWrite
//     must write the kernel's data. --wait chooses how the host learns that
//     the kernel has finished. With sync, hipStreamSynchronize waits for it,
//     and HIP also enqueues a barrier whose system-scope release writes the
//     GPU's L2 cache back to VRAM. With query, the host polls hipStreamQuery.
//     With event, it records an event after the kernel and polls
//     hipEventQuery. With callback, a host function enqueued after the kernel
//     tells the host. With sleep, the kernel sets a flag in host memory for
//     each workgroup once it has stored its share, and the host sleeps and
//     then checks the flags, so the kernel finished long before hipFileWrite,
//     but HIP was never asked. flag is the same without the sleep: the host
//     spins on the flags, so the kernel may not have finished, and a failure
//     shows only how late its data reaches VRAM.
//   - during: the kernel is still running during hipFileWrite. It stores its
//     share of the buffer, releases the stores at system scope, sets its
//     flags, and waits for hipFileWrite to return. A running kernel's stores
//     are only guaranteed to be visible to other agents in fine-grained
//     memory, which hipExtMallocWithFlags allocates. On PCIe GPUs, ROCr
//     provides fine-grained VRAM only when HSA_FORCE_FINE_GRAIN_PCIE=1, and
//     otherwise hipExtMallocWithFlags silently returns coarse-grained memory,
//     so the program sets that variable when it allocates fine-grained memory.
// - memcpy: hipMemcpy from host memory, which returns once the copy is done.
//   HIP copies small buffers with a kernel and larger ones with the GPU's
//   SDMA engine, so --buffer-size chooses which.
// - memset: hipMemsetD32, which stores the generation's number in every
//   32-bit word. It is a synchronous call, so hipFileWrite must write what it
//   stored, but HIP fills the buffer with a kernel and doesn't always wait for
//   it.
// - copy: hipMemcpy from a second GPU buffer, which the host fills with
//   hipMemcpy first. It is a synchronous call too, and HIP copies between GPU
//   buffers with a kernel it doesn't wait for.
// - read: hipFileRead from a second temporary file, in --source-directory,
//   which the host writes each generation to first. With both files on
//   fastpath, the data never leaves the GPU's HDP: one peer-to-peer transfer
//   writes VRAM and the next reads it. With the files on different drives, the
//   drive that reads VRAM is not the one that wrote it, so nothing in PCIe
//   orders its reads after the other drive's writes. With one file where only
//   fallback works, such as in /tmp, one transfer goes through the HDP and the
//   other through hipMemcpy.
//
// --memory overrides the kind of GPU memory the buffer is in. Uncached memory
// bypasses the GPU's L2 cache, so a failure that goes away with --memory
// uncached is in the GPU's caches, and one that stays is not.
//
// These options add what an application may legitimately do around the
// producer:
//
// - --prime write: before the producer, hipFileWrite writes the buffer, which
//   still holds the previous generation, to another part of the file. That
//   read of VRAM can leave the previous generation in the HDP's read cache.
// - --prime store: before a memcpy or read producer, a kernel stores another
//   generation, waited for as --wait says, which can leave it in the GPU's L2
//   cache. The producer can then write VRAM without going through L2, so the
//   test catches L2 writing its older data back over the producer's.
// - --background-writers: threads that keep writing the buffer to other parts
//   of the file with hipFileWrite, including while the producer stores. What
//   they write while the buffer changes is not checked, but their reads of
//   VRAM keep the HDP's read path busy, and can fill its cache with data the
//   producer is about to replace.
// - --threads: several threads run the test at once, each on its own stream
//   and its own part of one GPU buffer and of the file, next to the others'.
// - --write async: hipFileWriteAsync on the producer's stream instead of
//   hipFileWrite, so the stream, rather than the host, orders the write after
//   the producer.
//
// The test runs on HIP device 0, the first GPU the process can see. To test a
// different GPU, make it the only one the process can see by setting
// ROCR_VISIBLE_DEVICES to that GPU's index or UUID.
//
// hipFile chooses the appropriate backend for each file, using fastpath when
// it can and fallback otherwise. To control which backend the test uses, set
// hipFile's environment variables: HIPFILE_FORCE_COMPAT_MODE=true makes every
// transfer use the fallback backend, and HIPFILE_ALLOW_COMPAT_MODE=false makes
// every transfer use fastpath. In the second case, a transfer that can't use
// fastpath fails, so the test fails where fastpath is unavailable. Run with
// --help for options.
//
// For a soak test, run the program directly with --iterations -1, which runs
// until a check fails or the program receives SIGTERM or SIGINT. A signal
// stops the test after the current iteration, and the program prints how many
// iterations passed. Run it for each backend and producer, for example, for
// fastpath with a kernel the host waits for with an event:
//
//   HIPFILE_ALLOW_COMPAT_MODE=false timeout 8h hipfilewrite-visibility --directory DIR
//     --iterations -1 --producer kernel --wait event

#include "args.h"
#include "file-descriptor.h"
#include "hipfile-literals.h"
#include "hipfile-warnings.h"
#include "hipfile.h"
#include "hipfilewrite-visibility-kernel.h"
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
#include <fcntl.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <new>
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

using namespace hipFile;
using namespace hipFileTest;

namespace {

/// @brief --buffer-size must be a multiple of this, which suits direct I/O and
/// is the alignment the host buffers read back from storage need
constexpr size_t BUFFER_SIZE_GRANULARITY{4_KiB};

/// @brief How long the host waits for a producer, and a running kernel waits
/// for hipFileWrite to return, before giving up
constexpr std::chrono::seconds WAIT_TIMEOUT{60};

/// @brief Index of the first ready flag in a thread's host signals, which
/// keeps the flags the GPU writes off the cache line of the WriteState the
/// host writes
constexpr size_t READY_FLAGS_OFFSET{64 / sizeof(uint32_t)};

/// @brief How long the host sleeps for the kernel with --wait sleep
constexpr std::chrono::milliseconds PRODUCER_SLEEP{50};

/// @brief Most runs of wrong words a failure describes
constexpr size_t MAX_DESCRIBED_RUNS{8};

/// @brief What stores each generation in the GPU buffer
enum class Producer {
    /// @brief A kernel
    Kernel,
    /// @brief hipMemcpy from host memory
    Memcpy,
    /// @brief hipMemsetD32
    Memset,
    /// @brief hipMemcpy from a second GPU buffer
    Copy,
    /// @brief hipFileRead from the source file
    Read,
};

/// @brief When the kernel that stores a generation runs, relative to hipFileWrite
enum class KernelTiming {
    /// @brief The kernel finishes before hipFileWrite
    After,
    /// @brief The kernel is still running during hipFileWrite
    During,
};

/// @brief How the host learns that a kernel has stored its generation
enum class ProducerWait {
    /// @brief hipStreamSynchronize
    Sync,
    /// @brief Poll hipStreamQuery
    Query,
    /// @brief Record an event after the kernel and poll hipEventQuery
    Event,
    /// @brief A host function enqueued after the kernel
    Callback,
    /// @brief Spin on flags the kernel sets in host memory
    Flag,
    /// @brief Sleep for PRODUCER_SLEEP, then check the flags
    Sleep,
};

/// @brief What uses the buffer before the producer
enum class Prime {
    /// @brief Nothing
    None,
    /// @brief hipFileWrite writes the buffer to another part of the file
    Write,
    /// @brief A kernel stores another generation
    Store,
};

/// @brief Which hipFile call writes the file
enum class WriteApi {
    /// @brief hipFileWrite
    Sync,
    /// @brief hipFileWriteAsync on the producer's stream
    Async,
};

/// @brief The kind of GPU memory the buffer is in
enum class Memory {
    /// @brief Coarse-grained memory from hipMalloc
    Coarse,
    /// @brief Fine-grained memory from hipExtMallocWithFlags
    Fine,
    /// @brief Uncached memory from hipExtMallocWithFlags
    Uncached,
};

/// @brief Command-line options that control the check
struct Options {
    /// @brief Directory for the temporary file hipFileWrite writes
    std::string directory;
    /// @brief Directory for the temporary file a read producer reads
    std::string source_directory;
    /// @brief Size of each thread's part of the GPU buffer, which each
    /// hipFileWrite writes, in bytes
    size_t buffer_size;
    /// @brief Number of hipFileWrite calls each thread checks, or unset to run
    /// until a check fails or the program receives SIGTERM or SIGINT
    std::optional<size_t> iterations;
    /// @brief What stores each generation
    Producer producer;
    /// @brief When a kernel producer runs
    KernelTiming timing;
    /// @brief How the host waits for a kernel
    ProducerWait wait;
    /// @brief What uses the buffer before the producer
    Prime prime;
    /// @brief Which hipFile call writes the file
    WriteApi write_api;
    /// @brief The kind of GPU memory the buffer is in
    Memory memory;
    /// @brief Number of workgroups in each kernel
    unsigned int workgroups;
    /// @brief Number of threads running the test
    size_t threads;
    /// @brief Number of threads, for each thread running the test, that keep
    /// writing its part of the buffer to other parts of the file
    size_t background_writers;
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

/// @brief Host memory for one generation, aligned for direct I/O
class AlignedWords {
public:
    /// @brief Allocates the words
    ///
    /// @param word_count Number of 64-bit words
    explicit AlignedWords(size_t word_count)
        : words{new(std::align_val_t{BUFFER_SIZE_GRANULARITY}) uint64_t[word_count]}, count{word_count}
    {
    }

    /// @brief Gets the words
    ///
    /// @return The words, whose values are indeterminate until they are written
    std::span<uint64_t> span() const
    {
        // The aligned allocation is a raw pointer, so a span of it can only come
        // from a pointer and a length
        HIPFILE_WARN_UNSAFE_BUFFER_USAGE_OFF
        return {words.get(), count};
        HIPFILE_WARN_UNSAFE_BUFFER_USAGE_ON
    }

private:
    /// @brief Frees words allocated with the alignment
    struct Deleter {
        /// @brief Frees the words
        ///
        /// @param ptr The words
        void operator()(uint64_t *ptr) const
        {
            ::operator delete[](ptr, std::align_val_t{BUFFER_SIZE_GRANULARITY});
        }
    };

    /// @brief The words
    std::unique_ptr<uint64_t[], Deleter> words;
    /// @brief Number of words
    size_t count;
};

/// @brief Owns what one thread running the test uses to run kernels,
/// synchronize with them, and hold a generation in host memory
struct ThreadResources {
    /// @brief Non-blocking stream the thread's kernels run on, so that hipFile's
    /// own copies on the null stream don't wait for a running kernel, or nullptr
    hipStream_t stream{nullptr};
    /// @brief Event recorded after a kernel with ProducerWait::Event, or nullptr
    hipEvent_t event{nullptr};
    /// @brief Fine-grained host memory from hipHostMalloc for the WriteState,
    /// then a ready flag for each workgroup, then a timed-out flag for each
    /// workgroup, or nullptr
    void *signals{nullptr};
    /// @brief Set by the host function enqueued with ProducerWait::Callback
    std::atomic<uint32_t> callback_ran{0};
    /// @brief The generation a memcpy or read producer stores
    std::optional<AlignedWords> pattern;
    /// @brief The data read back from the file
    std::optional<AlignedWords> observed;

    /// @brief Creates an object that owns nothing yet
    ThreadResources() = default;
    /// @brief Not copyable, since the object owns the stream, event, and memory
    ThreadResources(const ThreadResources &) = delete;
    /// @brief Not copyable, since the object owns the stream, event, and memory
    ThreadResources &operator=(const ThreadResources &) = delete;

    /// @brief Destroys the event and stream and frees the memory
    ~ThreadResources()
    {
        if (event) {
            (void)hipEventDestroy(event);
        }
        if (stream) {
            (void)hipStreamDestroy(stream);
        }
        if (signals) {
            (void)hipHostFree(signals);
        }
    }
};

/// @brief Owns the files, their registered hipFile handles, the GPU buffer,
/// and each thread's resources
struct Resources {
    /// @brief Temporary file in Options::directory that hipFileWrite writes
    std::optional<Tmpfile> tmpfile;
    /// @brief Temporary file in Options::source_directory that a read producer reads
    std::optional<Tmpfile> source_tmpfile;
    /// @brief The file hipFileWrite writes, opened again to read it back from
    /// storage, without the page cache where the file system allows it
    FileDescriptor verify_fd;
    /// @brief Registered handle for tmpfile, or nullptr
    hipFileHandle_t handle{nullptr};
    /// @brief Registered handle for source_tmpfile, or nullptr
    hipFileHandle_t source_handle{nullptr};
    /// @brief GPU buffer, with a part for each thread, or nullptr
    void *buffer{nullptr};
    /// @brief Second GPU buffer, with a part for each thread, that a copy
    /// producer copies each generation from, or nullptr
    void *staging{nullptr};
    /// @brief Each thread's resources
    std::vector<std::unique_ptr<ThreadResources>> threads;

    /// @brief Creates an object that owns nothing yet
    Resources() = default;
    /// @brief Not copyable, since the object owns the files, handles, and memory
    Resources(const Resources &) = delete;
    /// @brief Not copyable, since the object owns the files, handles, and memory
    Resources &operator=(const Resources &) = delete;

    /// @brief Frees the threads' resources and the GPU buffer and deregisters the handles
    ///
    /// The file members close the files afterward, as members are destroyed
    /// after the destructor's body runs.
    ~Resources()
    {
        threads.clear();
        if (staging) {
            (void)hipFree(staging);
        }
        if (buffer) {
            (void)hipFree(buffer);
        }
        if (source_handle) {
            hipFileHandleDeregister(source_handle);
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

/// @brief Gets the most bytes one hipFileWrite can write
///
/// hipFile documents the limit, which the Linux kernel sets, as 2^31 bytes
/// minus the system page size.
///
/// @return The limit in bytes
size_t
maxWriteSize()
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

/// @brief Fills a buffer from a file, continuing after partial reads
///
/// @param fd The file
/// @param data The bytes to fill
/// @param offset File offset to read them from
///
/// @return true if every byte was read; false, with errno set, otherwise
bool
readAll(int fd, std::span<std::byte> data, hoff_t offset)
{
    while (!data.empty()) {
        const ssize_t nread{pread(fd, data.data(), data.size(), offset)};
        if (nread <= 0) {
            if (nread == 0) {
                errno = ENODATA;
            }
            return false;
        }
        data = data.subspan(static_cast<size_t>(nread));
        offset += nread;
    }
    return true;
}

/// @brief Opens an open file again to read it back from storage
///
/// The temporary files have no name, so the file is opened through /proc. The
/// file is opened for direct I/O, so that reads come from storage, where the
/// file system supports it, and through the page cache otherwise.
///
/// @param fd The open file
///
/// @return The file, opened again read-only, or an empty FileDescriptor, with
///         errno set, if it cannot be opened
FileDescriptor
reopenForVerification(int fd)
{
    const std::string path{"/proc/self/fd/" + std::to_string(fd)};
    int               verify_fd{open(path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC)};
    if (verify_fd == -1 && errno == EINVAL) {
        verify_fd = open(path.c_str(), O_RDONLY | O_CLOEXEC);
    }
    if (verify_fd == -1) {
        return {};
    }
    return FileDescriptor::make_managed(verify_fd);
}

/// @brief Stores a generation in host memory
///
/// @param words The words to fill
/// @param generation The generation
void
fillPattern(std::span<uint64_t> words, uint32_t generation)
{
    for (size_t i{0}; i < words.size(); i++) {
        words[i] = patternValue(generation, i);
    }
}

/// @brief Gets the value every word holds once a memset producer has stored a generation
///
/// hipMemsetD32 stores one value in every 32-bit word, so the words of such a
/// generation can't hold their indexes.
///
/// @param generation The generation
///
/// @return The generation in both halves
constexpr uint64_t
memsetValue(uint32_t generation)
{
    return (uint64_t{generation} << 32) | generation;
}

/// @brief A run of consecutive words that differ from the generation they should hold
struct WrongRun {
    /// @brief Index of the run's first word
    size_t first_word;
    /// @brief Number of words in the run
    size_t word_count;
    /// @brief Value of the run's first word
    uint64_t first_value;
};

/// @brief How data differs from the generation it should hold
struct Mismatches {
    /// @brief Words that hold their value from an earlier generation
    size_t stale{0};
    /// @brief Words that hold anything else
    size_t unexpected{0};
    /// @brief Earliest generation a stale word held
    uint32_t oldest_generation{UINT32_MAX};
    /// @brief Latest generation a stale word held
    uint32_t newest_generation{0};
    /// @brief Number of runs of consecutive wrong words
    size_t run_count{0};
    /// @brief Number of runs that start on a 4 KiB boundary
    size_t aligned_run_count{0};
    /// @brief Number of words in the shortest run
    size_t shortest_run{std::numeric_limits<size_t>::max()};
    /// @brief Number of words in the longest run
    size_t longest_run{0};
    /// @brief The first MAX_DESCRIBED_RUNS runs
    std::vector<WrongRun> runs{};

    /// @brief Checks whether any word differs
    ///
    /// @return true if a word is stale or unexpected
    bool any() const
    {
        return stale != 0 || unexpected != 0;
    }
};

/// @brief Compares data with the generation it should hold
///
/// A wrong word is stale if it holds what an earlier generation stored there,
/// whichever way that generation was stored.
///
/// @param observed The data
/// @param generation The generation
/// @param memset_stored Whether a memset producer stored the generation, so
///        every word holds memsetValue() rather than patternValue()
///
/// @return How the data differs
Mismatches
findMismatches(std::span<const uint64_t> observed, uint32_t generation, bool memset_stored)
{
    Mismatches mismatches;
    size_t     run_length{0};

    auto end_run{[&mismatches, &run_length]() {
        if (run_length == 0) {
            return;
        }
        mismatches.shortest_run = std::min(mismatches.shortest_run, run_length);
        mismatches.longest_run  = std::max(mismatches.longest_run, run_length);
        if (mismatches.runs.size() == mismatches.run_count) {
            mismatches.runs.back().word_count = run_length;
        }
        run_length = 0;
    }};

    for (size_t i{0}; i < observed.size(); i++) {
        const uint64_t value{observed[i]};
        if (value == (memset_stored ? memsetValue(generation) : patternValue(generation, i))) {
            end_run();
            continue;
        }

        const auto value_generation{static_cast<uint32_t>(value >> 32)};
        if (value_generation < generation &&
            (value == patternValue(value_generation, i) || value == memsetValue(value_generation))) {
            mismatches.stale++;
            mismatches.oldest_generation = std::min(mismatches.oldest_generation, value_generation);
            mismatches.newest_generation = std::max(mismatches.newest_generation, value_generation);
        }
        else {
            mismatches.unexpected++;
        }

        if (run_length == 0) {
            mismatches.run_count++;
            if (i * sizeof(uint64_t) % BUFFER_SIZE_GRANULARITY == 0) {
                mismatches.aligned_run_count++;
            }
            if (mismatches.runs.size() < MAX_DESCRIBED_RUNS) {
                mismatches.runs.push_back({.first_word = i, .word_count = 0, .first_value = value});
            }
        }
        run_length++;
    }
    end_run();
    return mismatches;
}

/// @brief Describes how data differs from the generation it should hold
///
/// @param mismatches How the data differs; at least one word must
/// @param word_count Number of words compared
/// @param generation The generation
///
/// @return The description
std::string
describeMismatches(const Mismatches &mismatches, size_t word_count, uint32_t generation)
{
    std::ostringstream description;
    description << mismatches.stale + mismatches.unexpected << " of " << word_count
                << " 64-bit words did not hold generation " << generation << ": " << mismatches.stale
                << " stale";
    if (mismatches.stale != 0) {
        description << ", from generation";
        if (mismatches.oldest_generation == mismatches.newest_generation) {
            description << ' ' << mismatches.oldest_generation;
        }
        else {
            description << "s " << mismatches.oldest_generation << " to " << mismatches.newest_generation;
        }
    }
    description << ", and " << mismatches.unexpected << " unexpected, in " << mismatches.run_count
                << (mismatches.run_count == 1 ? " run" : " runs") << " of "
                << mismatches.shortest_run * sizeof(uint64_t) << " to "
                << mismatches.longest_run * sizeof(uint64_t) << " bytes, " << mismatches.aligned_run_count
                << " of them starting on a 4 KiB boundary.";
    description << (mismatches.runs.size() < mismatches.run_count ? " The first runs:" : " The runs:");
    for (const WrongRun &run : mismatches.runs) {
        description << " byte offset " << run.first_word * sizeof(uint64_t) << ", "
                    << run.word_count * sizeof(uint64_t) << " bytes, starting with " << std::hex
                    << std::showbase << run.first_value << std::dec << std::noshowbase << " (generation "
                    << (run.first_value >> 32) << ");";
    }
    return description.str();
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
/// glibc's signal() restarts the system calls a signal interrupts, so the
/// transfer in progress doesn't fail. The handlers stay installed, because
/// timeout sends its signal twice; SIGKILL ends a run whose transfer hangs.
///
/// @return true if the handlers were installed; false, with errno set, otherwise
bool
stopOnSignals()
{
    return std::signal(SIGTERM, requestStop) != SIG_ERR && std::signal(SIGINT, requestStop) != SIG_ERR;
}

/// @brief Host function that tells the host the work before it on the stream has finished
///
/// @param callback_ran The ThreadResources::callback_ran to set
void
reportCallbackRan(void *callback_ran)
{
    static_cast<std::atomic<uint32_t> *>(callback_ran)->store(1, std::memory_order_release);
}

/// @brief Waits, yielding, until a condition holds or a deadline passes
///
/// @tparam Condition Type of the condition
/// @param condition Returns true once the wait is over
/// @param deadline When to give up
///
/// @return true if the condition held; false if the deadline passed first
template <typename Condition>
bool
waitUntil(Condition condition, std::chrono::steady_clock::time_point deadline)
{
    while (!condition()) {
        if (std::chrono::steady_clock::now() >= deadline) {
            return false;
        }
        std::this_thread::yield();
    }
    return true;
}

/// @brief Describes how a kernel's data reaches hipFileWrite, for messages
///
/// @param wait How the host waits for the kernel
///
/// @return The description
std::string_view
waitDescription(ProducerWait wait)
{
    switch (wait) {
        case ProducerWait::Sync:
            return "waited for with hipStreamSynchronize";
        case ProducerWait::Query:
            return "waited for with hipStreamQuery";
        case ProducerWait::Event:
            return "waited for with an event";
        case ProducerWait::Callback:
            return "waited for with a host function";
        case ProducerWait::Flag:
            return "waited for with flags";
        case ProducerWait::Sleep:
            return "waited for with a sleep";
        default:
            return "";
    }
}

/// @brief Describes the test the options select, for messages
///
/// @param options Supplies the producer, how the host waits for it, what primes
///        the buffer, and the hipFile call
///
/// @return The description
std::string
testDescription(const Options &options)
{
    const bool  async_write{options.write_api == WriteApi::Async};
    std::string description{async_write ? "hipFileWriteAsync" : "hipFileWrite"};
    switch (options.producer) {
        case Producer::Kernel:
            if (options.timing == KernelTiming::During) {
                description += " during a kernel";
            }
            else {
                description += " after a kernel";
                if (!async_write) {
                    description += ' ';
                    description += waitDescription(options.wait);
                }
            }
            break;
        case Producer::Memcpy:
            description += " after hipMemcpy from host memory";
            break;
        case Producer::Memset:
            description += " after hipMemsetD32";
            break;
        case Producer::Copy:
            description += " after hipMemcpy from GPU memory";
            break;
        case Producer::Read:
            description += " after hipFileRead";
            break;
        default:
            break;
    }
    switch (options.memory) {
        case Memory::Coarse:
            description += ", in coarse-grained memory";
            break;
        case Memory::Fine:
            description += ", in fine-grained memory";
            break;
        case Memory::Uncached:
            description += ", in uncached memory";
            break;
        default:
            break;
    }
    if (options.prime == Prime::Write) {
        description += ", primed with hipFileWrite";
    }
    else if (options.prime == Prime::Store) {
        description += ", primed with a kernel ";
        description += waitDescription(options.wait);
    }
    if (options.background_writers != 0) {
        description += ", with " + std::to_string(options.background_writers) + " background " +
                       (options.background_writers == 1 ? "writer" : "writers");
    }
    return description;
}

/// @brief What the threads running the test share
struct Shared {
    /// @brief The files, their registered handles, and the GPU buffer
    const Resources &resources;
    /// @brief Command-line options
    const Options &options;
    /// @brief Wall clock ticks a running kernel waits for hipFileWrite
    uint64_t timeout_ticks;
    /// @brief Set once any thread fails, which stops the others after their
    /// current iteration
    std::atomic<bool> failed{false};

    /// @brief Checks whether the threads should stop
    ///
    /// @return true if a thread failed or SIGTERM or SIGINT arrived
    bool stopping() const
    {
        return failed.load(std::memory_order_relaxed) || stop_signal.load(std::memory_order_relaxed) != 0;
    }
};

/// @brief Checks, on one thread, that data stored in the thread's part of the
/// GPU buffer is in the file once hipFileWrite returns
///
/// Runs options.iterations iterations, or runs without a limit if it is unset.
/// Each stores a new generation in the thread's part of the buffer, writes it
/// to the thread's part of the file, and reads the file back. Stops after the
/// first iteration with a failure, once another thread has failed, or after
/// the current iteration once SIGTERM or SIGINT arrives.
///
/// The file holds a part for each thread, then, with Prime::Write, a part for
/// each thread's priming hipFileWrite, then a part for each background writer.
///
/// @param shared What the threads share
/// @param thread_index Which thread this is
/// @param[out] passed_iterations Set to the number of iterations that passed
///
/// @return A description of each failure; empty if the check passed
std::vector<std::string>
checkWriteVisibility(Shared &shared, size_t thread_index, size_t &passed_iterations)
{
    Failures failures;
    passed_iterations = 0;

    const Resources &resources{shared.resources};
    const Options   &options{shared.options};
    ThreadResources &thread{*resources.threads[thread_index]};

    const size_t buffer_size{options.buffer_size};
    const size_t word_count{buffer_size / sizeof(uint64_t)};
    // The thread's part of the GPU buffer, of the file, and of the source file
    const auto  part_offset{static_cast<hoff_t>(thread_index * buffer_size)};
    const auto  prime_offset{static_cast<hoff_t>((options.threads + thread_index) * buffer_size)};
    void *const part{bytesAt(resources.buffer, thread_index * buffer_size)};

    const bool  async_write{options.write_api == WriteApi::Async};
    const char *write_name{async_write ? "hipFileWriteAsync" : "hipFileWrite"};
    const bool  kernel_during{options.producer == Producer::Kernel && options.timing == KernelTiming::During};
    const bool  flags_wait{options.wait == ProducerWait::Flag || options.wait == ProducerWait::Sleep};
    const bool  memset_stored{options.producer == Producer::Memset};

    // The host signals: the WriteState, then a ready flag and a timed-out flag
    // for each workgroup. hipHostMalloc returns a raw pointer, so a span of it
    // can only come from a pointer and a length.
    HIPFILE_WARN_UNSAFE_BUFFER_USAGE_OFF
    const std::span<uint32_t> signals{static_cast<uint32_t *>(thread.signals),
                                      READY_FLAGS_OFFSET + 2 * size_t{options.workgroups}};
    HIPFILE_WARN_UNSAFE_BUFFER_USAGE_ON
    uint32_t                 &write_state{signals.front()};
    const std::span<uint32_t> ready{signals.subspan(READY_FLAGS_OFFSET, options.workgroups)};
    const std::span<uint32_t> timed_out{signals.subspan(READY_FLAGS_OFFSET + options.workgroups)};

    auto is_set{
        [](uint32_t &flag) { return std::atomic_ref<uint32_t>{flag}.load(std::memory_order_acquire) != 0; }};
    auto all_ready{[&ready, &is_set]() { return std::ranges::all_of(ready, is_set); }};

    // Launches a kernel that stores a generation in the thread's part of the buffer
    auto launch_kernel{[&](uint32_t generation, bool report_ready, bool wait_for_write) {
        for (uint32_t &flag : ready) {
            std::atomic_ref<uint32_t>{flag}.store(0, std::memory_order_relaxed);
        }
        const ProducerKernelArgs args{
            .buffer        = static_cast<uint64_t *>(part),
            .word_count    = word_count,
            .generation    = generation,
            .ready         = report_ready ? ready.data() : nullptr,
            .write_state   = wait_for_write ? &write_state : nullptr,
            .timeout_ticks = shared.timeout_ticks,
            .timed_out     = wait_for_write ? timed_out.data() : nullptr,
        };
        return launchProducerKernel(args, options.workgroups, thread.stream) == hipSuccess;
    }};

    // Waits for a kernel that has just been launched to store its generation,
    // as options.wait says. Returns a description of the failure, or nothing.
    auto wait_for_kernel{[&]() -> std::optional<std::string> {
        const auto deadline{std::chrono::steady_clock::now() + WAIT_TIMEOUT};
        const auto timeout{" within " + std::to_string(WAIT_TIMEOUT.count()) + " s"};
        hipError_t err{hipSuccess};
        switch (options.wait) {
            case ProducerWait::Sync:
                if (hipStreamSynchronize(thread.stream) != hipSuccess) {
                    return "hipStreamSynchronize failed";
                }
                break;
            case ProducerWait::Query:
                if (!waitUntil([&]() { return (err = hipStreamQuery(thread.stream)) != hipErrorNotReady; },
                               deadline)) {
                    return "hipStreamQuery did not report that the stream had finished" + timeout;
                }
                if (err != hipSuccess) {
                    return "hipStreamQuery failed";
                }
                break;
            case ProducerWait::Event:
                if (hipEventRecord(thread.event, thread.stream) != hipSuccess) {
                    return "hipEventRecord failed";
                }
                if (!waitUntil([&]() { return (err = hipEventQuery(thread.event)) != hipErrorNotReady; },
                               deadline)) {
                    return "hipEventQuery did not report that the event had finished" + timeout;
                }
                if (err != hipSuccess) {
                    return "hipEventQuery failed";
                }
                break;
            case ProducerWait::Callback:
                thread.callback_ran.store(0, std::memory_order_relaxed);
                if (hipLaunchHostFunc(thread.stream, reportCallbackRan, &thread.callback_ran) != hipSuccess) {
                    return "hipLaunchHostFunc failed";
                }
                if (!waitUntil([&]() { return thread.callback_ran.load(std::memory_order_acquire) != 0; },
                               deadline)) {
                    return "the host function did not run" + timeout;
                }
                break;
            case ProducerWait::Flag:
                if (!waitUntil(all_ready, deadline)) {
                    return "not every workgroup stored its share of the buffer" + timeout;
                }
                break;
            case ProducerWait::Sleep:
                // Nothing that enqueues GPU work may run between here and the write
                std::this_thread::sleep_for(PRODUCER_SLEEP);
                if (!all_ready()) {
                    return "not every workgroup stored its share of the buffer within " +
                           std::to_string(PRODUCER_SLEEP.count()) + " ms";
                }
                break;
            default:
                return "the way to wait is unknown";
        }
        return std::nullopt;
    }};

    // Describes how a transfer of the thread's part failed, or nothing if it
    // transferred all of it. hipFileRead and hipFileWrite set errno only when
    // they return -1; other negative values are negated hipFileOpError_t codes.
    auto transfer_failure{
        [&](const char *name, ssize_t nbytes, int transfer_errno) -> std::optional<std::string> {
            std::ostringstream description;
            if (std::cmp_equal(nbytes, buffer_size)) {
                return std::nullopt;
            }
            if (nbytes == -1) {
                description << name << " failed";
                if (transfer_errno != 0) {
                    description << ": " << std::strerror(transfer_errno);
                }
            }
            else if (nbytes < 0) {
                description << name << " failed with hipFile error " << -nbytes << ": "
                            << hipFileGetOpErrorString(static_cast<hipFileOpError_t>(-nbytes));
            }
            else {
                description << name << " transferred " << nbytes << " of " << buffer_size << " bytes";
            }
            return description.str();
        }};

    // Reads the thread's part of the file back and compares it with a
    // generation. Returns nothing if the file could not be read.
    auto compare_file{[&](uint32_t generation) -> std::optional<Mismatches> {
        const std::span<uint64_t> observed{thread.observed->span()};
        if (!readAll(resources.verify_fd.get(), std::as_writable_bytes(observed), part_offset)) {
            return std::nullopt;
        }
        return findMismatches(observed, generation, memset_stored);
    }};

    // The buffer and the file start with generation 0
    uint32_t generation{0};

    for (size_t iteration{0};
         (!options.iterations || iteration < *options.iterations) && !failures.any() && !shared.stopping();
         iteration++) {
        // Records a failure of this iteration
        auto fail{[&](const auto &...parts) {
            failures.add("Thread ", thread_index, ", iteration ", iteration, ": ", parts...);
        }};

        std::atomic_ref<uint32_t>{write_state}.store(static_cast<uint32_t>(WriteState::InProgress),
                                                     std::memory_order_relaxed);
        for (uint32_t &flag : timed_out) {
            std::atomic_ref<uint32_t>{flag}.store(0, std::memory_order_relaxed);
        }

        if (options.prime == Prime::Write) {
            const ssize_t nbytes{
                hipFileWrite(resources.handle, resources.buffer, buffer_size, prime_offset, part_offset)};
            if (const auto failure{transfer_failure("the priming hipFileWrite", nbytes, errno)}) {
                fail(*failure);
                break;
            }
        }
        else if (options.prime == Prime::Store) {
            generation++;
            if (!launch_kernel(generation, flags_wait, false)) {
                fail("launching the kernel that primes the buffer failed");
                break;
            }
            if (const auto failure{wait_for_kernel()}) {
                fail("waiting for the kernel that primes the buffer: ", *failure);
                (void)hipStreamSynchronize(thread.stream);
                break;
            }
        }

        generation++;
        switch (options.producer) {
            case Producer::Kernel:
                if (!launch_kernel(generation, kernel_during || (flags_wait && !async_write),
                                   kernel_during)) {
                    fail("launching the kernel failed");
                }
                else if (kernel_during) {
                    // Start the write only once every workgroup has stored and
                    // released its share of the buffer
                    if (!waitUntil(all_ready, std::chrono::steady_clock::now() + WAIT_TIMEOUT)) {
                        fail("not every workgroup stored its share of the buffer within ",
                             WAIT_TIMEOUT.count(), " s; they must all fit on the GPU at once");
                    }
                }
                // With hipFileWriteAsync, the stream runs the write after the kernel
                else if (!async_write) {
                    if (const auto failure{wait_for_kernel()}) {
                        fail("waiting for the kernel: ", *failure);
                    }
                }
                break;
            case Producer::Memcpy: {
                const std::span<uint64_t> pattern{thread.pattern->span()};
                fillPattern(pattern, generation);
                if (hipMemcpy(part, pattern.data(), buffer_size, hipMemcpyHostToDevice) != hipSuccess) {
                    fail("hipMemcpy failed");
                }
                break;
            }
            case Producer::Memset:
                if (hipMemsetD32(part, static_cast<int>(generation), buffer_size / sizeof(uint32_t)) !=
                    hipSuccess) {
                    fail("hipMemsetD32 failed");
                }
                break;
            case Producer::Copy: {
                const std::span<uint64_t> pattern{thread.pattern->span()};
                void *const staging_part{bytesAt(resources.staging, thread_index * buffer_size)};
                fillPattern(pattern, generation);
                if (hipMemcpy(staging_part, pattern.data(), buffer_size, hipMemcpyHostToDevice) !=
                    hipSuccess) {
                    fail("hipMemcpy to the second GPU buffer failed");
                }
                else if (hipMemcpy(part, staging_part, buffer_size, hipMemcpyDeviceToDevice) != hipSuccess) {
                    fail("hipMemcpy from the second GPU buffer failed");
                }
                break;
            }
            case Producer::Read: {
                const std::span<uint64_t> pattern{thread.pattern->span()};
                fillPattern(pattern, generation);
                // The fastpath's direct reads bypass the page cache, so the
                // generation must be in storage before hipFileRead
                const int source_fd{resources.source_tmpfile->fd};
                if (!writeAll(source_fd, std::as_bytes(pattern), part_offset) || fdatasync(source_fd) != 0) {
                    fail("writing the generation to the source file failed: ", std::strerror(errno));
                    break;
                }
                const ssize_t nbytes{hipFileRead(resources.source_handle, resources.buffer, buffer_size,
                                                 part_offset, part_offset)};
                if (const auto failure{transfer_failure("hipFileRead", nbytes, errno)}) {
                    fail(*failure);
                }
                break;
            }
            default:
                fail("the producer is unknown");
                break;
        }
        if (failures.any()) {
            std::atomic_ref<uint32_t>{write_state}.store(static_cast<uint32_t>(WriteState::Finished),
                                                         std::memory_order_release);
            (void)hipStreamSynchronize(thread.stream);
            break;
        }

        // hipFileWriteAsync reads these when its operation runs on the stream, so
        // they must outlive it; the stream is synchronized right after
        size_t  write_size{buffer_size};
        hoff_t  file_offset{part_offset};
        hoff_t  buffer_offset{part_offset};
        ssize_t nbytes{0};
        int     write_errno{0};
        if (async_write) {
            const hipFileError_t err{hipFileWriteAsync(resources.handle, resources.buffer, &write_size,
                                                       &file_offset, &buffer_offset, &nbytes, thread.stream)};
            if (err.err != hipFileSuccess) {
                fail("hipFileWriteAsync failed to enqueue the write: ", hipFileGetOpErrorString(err.err));
                (void)hipStreamSynchronize(thread.stream);
                break;
            }
            if (hipStreamSynchronize(thread.stream) != hipSuccess) {
                fail("hipStreamSynchronize failed after hipFileWriteAsync");
                break;
            }
        }
        else {
            nbytes = hipFileWrite(resources.handle, resources.buffer, buffer_size, part_offset, part_offset);
            write_errno = errno;
        }
        // Lets a running kernel finish. The comparison below doesn't use the GPU.
        std::atomic_ref<uint32_t>{write_state}.store(static_cast<uint32_t>(WriteState::Finished),
                                                     std::memory_order_release);
        if (const auto failure{transfer_failure(write_name, nbytes, write_errno)}) {
            fail(*failure);
            (void)hipStreamSynchronize(thread.stream);
            break;
        }

        const auto                      written{std::chrono::steady_clock::now()};
        const std::optional<Mismatches> mismatches{compare_file(generation)};
        if (!mismatches) {
            fail("reading the file back failed: ", std::strerror(errno));
        }
        else if (mismatches->any()) {
            fail("after ", write_name, " returned, the file differed from the GPU buffer: ",
                 describeMismatches(*mismatches, word_count, generation));

            // Write the buffer again before anything else uses the GPU, to tell
            // data that reached the file late from data that never would
            const ssize_t again_nbytes{
                hipFileWrite(resources.handle, resources.buffer, buffer_size, part_offset, part_offset)};
            const std::chrono::duration<double, std::milli> delay{std::chrono::steady_clock::now() - written};
            const std::optional<Mismatches>                 again{
                std::cmp_equal(again_nbytes, buffer_size) ? compare_file(generation) : std::nullopt};
            if (!again) {
                fail("writing the buffer again and reading the file back failed");
            }
            else if (again->any()) {
                fail("a second hipFileWrite, ", delay.count(),
                     " ms later, still differed: ", describeMismatches(*again, word_count, generation));
            }
            else {
                fail("a second hipFileWrite, ", delay.count(), " ms later, wrote the generation");
            }
        }

        // A kernel that waited for the write must finish before the next
        // iteration resets the WriteState it waits on
        if (kernel_during || failures.any()) {
            if (hipStreamSynchronize(thread.stream) != hipSuccess) {
                fail("the kernel failed");
            }
        }
        if (kernel_during) {
            const auto gave_up{static_cast<size_t>(std::ranges::count_if(timed_out, is_set))};
            if (gave_up != 0) {
                fail(gave_up, " of ", timed_out.size(), " workgroups timed out after ", WAIT_TIMEOUT.count(),
                     " s waiting for hipFileWrite to return");
            }
        }

        if (mismatches && mismatches->any()) {
            // Now that the comparison is over, check what the GPU buffer holds
            const std::span<uint64_t> observed{thread.observed->span()};
            if (hipMemcpy(observed.data(), part, buffer_size, hipMemcpyDeviceToHost) != hipSuccess) {
                fail("copying the GPU buffer back failed");
            }
            else if (const Mismatches in_buffer{findMismatches(observed, generation, memset_stored)};
                     in_buffer.any()) {
                fail("the GPU buffer, copied back with hipMemcpy, differed too: ",
                     describeMismatches(in_buffer, word_count, generation));
            }
            else {
                fail("the GPU buffer, copied back with hipMemcpy, held the generation");
            }
        }
        if (!failures.any()) {
            passed_iterations++;
        }
    }

    if (failures.any()) {
        shared.failed.store(true, std::memory_order_relaxed);
    }
    return failures.take();
}

/// @brief Keeps writing a thread's part of the GPU buffer to the background
/// writer's own part of the file until asked to stop
///
/// @param shared What the threads share
/// @param thread_index Which thread's part of the buffer to write
/// @param writer_index Which of that thread's background writers this is
/// @param stop Set once the thread running the test has finished
///
/// @return A description of the failure; empty if every write succeeded
std::vector<std::string>
writeInBackground(Shared &shared, size_t thread_index, size_t writer_index, const std::atomic<bool> &stop)
{
    Failures         failures;
    const Resources &resources{shared.resources};
    const Options   &options{shared.options};
    const size_t     buffer_size{options.buffer_size};
    const size_t file_part{2 * options.threads + thread_index * options.background_writers + writer_index};
    const auto   file_offset{static_cast<hoff_t>(file_part * buffer_size)};
    const auto   buffer_offset{static_cast<hoff_t>(thread_index * buffer_size)};

    while (!stop.load(std::memory_order_relaxed)) {
        const ssize_t nbytes{
            hipFileWrite(resources.handle, resources.buffer, buffer_size, file_offset, buffer_offset)};
        if (!std::cmp_equal(nbytes, buffer_size)) {
            failures.add("Thread ", thread_index, ", background writer ", writer_index,
                         ": hipFileWrite returned ", nbytes);
            shared.failed.store(true, std::memory_order_relaxed);
            break;
        }
    }
    return failures.take();
}

/// @brief Prepares the files and the GPU buffer, then runs the check on every thread
///
/// Writes generation 0 to each thread's part of the file and of the GPU
/// buffer, and zeros to the rest of the file, so that no hipFileWrite extends
/// the file.
///
/// @param resources The files, their registered handles, the GPU buffer, and
///        each thread's resources, all ready to use
/// @param options Command-line options
/// @param[out] passed_iterations Set to the fewest iterations that passed on any thread
///
/// @return A description of each failure; empty if the check passed
std::vector<std::string>
checkWriteVisibility(const Resources &resources, const Options &options, size_t &passed_iterations)
{
    Failures failures;
    passed_iterations = 0;

    const size_t buffer_size{options.buffer_size};
    const size_t file_parts{options.threads * (2 + options.background_writers)};
    {
        const AlignedWords        words{buffer_size / sizeof(uint64_t)};
        const std::span<uint64_t> pattern{words.span()};
        for (size_t file_part{0}; file_part < file_parts; file_part++) {
            const bool thread_part{file_part < options.threads};
            if (thread_part) {
                fillPattern(pattern, 0);
            }
            else {
                std::ranges::fill(pattern, uint64_t{0});
            }
            const auto offset{static_cast<hoff_t>(file_part * buffer_size)};
            if (!writeAll(resources.tmpfile->fd, std::as_bytes(pattern), offset)) {
                failures.add("Initializing the file failed: ", std::strerror(errno));
                return failures.take();
            }
            if (thread_part && resources.source_tmpfile &&
                !writeAll(resources.source_tmpfile->fd, std::as_bytes(pattern), offset)) {
                failures.add("Initializing the source file failed: ", std::strerror(errno));
                return failures.take();
            }
            if (thread_part && hipMemcpy(bytesAt(resources.buffer, file_part * buffer_size), pattern.data(),
                                         buffer_size, hipMemcpyHostToDevice) != hipSuccess) {
                failures.add("Initializing the GPU buffer failed");
                return failures.take();
            }
        }
    }
    if (hipDeviceSynchronize() != hipSuccess) {
        failures.add("Initializing the GPU buffer failed");
        return failures.take();
    }
    if (fdatasync(resources.tmpfile->fd) != 0 ||
        (resources.source_tmpfile && fdatasync(resources.source_tmpfile->fd) != 0)) {
        failures.add("Flushing the files to storage failed: ", std::strerror(errno));
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
    Shared shared{
        .resources = resources,
        .options   = options,
        .timeout_ticks =
            static_cast<uint64_t>(wall_clock_khz) * 1000 * static_cast<uint64_t>(WAIT_TIMEOUT.count()),
    };

    // Each thread running the test, and each of its background writers, reports
    // its failures and, for the former, its iterations
    std::mutex               results_mutex;
    std::vector<std::string> descriptions;
    size_t                   fewest_passed{std::numeric_limits<size_t>::max()};
    auto                     report{[&](std::vector<std::string> thread_failures) {
        const std::lock_guard<std::mutex> lock{results_mutex};
        std::ranges::move(thread_failures, std::back_inserter(descriptions));
    }};
    {
        std::vector<std::jthread> threads;
        for (size_t thread_index{0}; thread_index < options.threads; thread_index++) {
            threads.emplace_back([&, thread_index]() {
                std::atomic<bool>         stop{false};
                std::vector<std::jthread> writers;
                for (size_t writer_index{0}; writer_index < options.background_writers; writer_index++) {
                    writers.emplace_back([&, writer_index]() {
                        report(writeInBackground(shared, thread_index, writer_index, stop));
                    });
                }
                size_t passed{0};
                report(checkWriteVisibility(shared, thread_index, passed));
                stop.store(true, std::memory_order_relaxed);
                writers.clear();
                const std::lock_guard<std::mutex> lock{results_mutex};
                fewest_passed = std::min(fewest_passed, passed);
            });
        }
    }

    passed_iterations = fewest_passed;
    return descriptions;
}

/// @brief Runs the check on temporary files
///
/// Creates the temporary files, registers them, and allocates the GPU buffer
/// and each thread's resources before running the check.
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
        if (options.producer == Producer::Read) {
            resources.source_tmpfile.emplace(options.source_directory);
        }
    }
    catch (const std::runtime_error &e) {
        std::cerr << e.what() << '\n';
        return false;
    }
    resources.verify_fd = reopenForVerification(resources.tmpfile->fd);
    if (resources.verify_fd.get() == -1) {
        std::cerr << "Unable to open the file again to read it back: " << std::strerror(errno) << '\n';
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
    if (resources.source_tmpfile) {
        descr.handle.fd = resources.source_tmpfile->fd;
        if (hipFileHandleRegister(&resources.source_handle, &descr).err != hipFileSuccess) {
            resources.source_handle = nullptr;
            std::cerr << "hipFileHandleRegister failed for the source file\n";
            return false;
        }
    }

    const size_t total_size{options.threads * options.buffer_size};
    hipError_t   buffer_err{hipSuccess};
    switch (options.memory) {
        case Memory::Coarse:
            buffer_err = hipMalloc(&resources.buffer, total_size);
            break;
        case Memory::Fine:
            buffer_err = hipExtMallocWithFlags(&resources.buffer, total_size, hipDeviceMallocFinegrained);
            break;
        case Memory::Uncached:
            buffer_err = hipExtMallocWithFlags(&resources.buffer, total_size, hipDeviceMallocUncached);
            break;
        default:
            buffer_err = hipErrorInvalidValue;
            break;
    }
    if (buffer_err != hipSuccess) {
        resources.buffer = nullptr;
        std::cerr << (options.memory == Memory::Coarse ? "hipMalloc" : "hipExtMallocWithFlags")
                  << " failed for the GPU buffer\n";
        return false;
    }

    if (options.producer == Producer::Copy && hipMalloc(&resources.staging, total_size) != hipSuccess) {
        resources.staging = nullptr;
        std::cerr << "hipMalloc failed for the second GPU buffer\n";
        return false;
    }

    const size_t word_count{options.buffer_size / sizeof(uint64_t)};
    for (size_t thread_index{0}; thread_index < options.threads; thread_index++) {
        ThreadResources &thread{*resources.threads.emplace_back(std::make_unique<ThreadResources>())};
        if (hipStreamCreateWithFlags(&thread.stream, hipStreamNonBlocking) != hipSuccess) {
            thread.stream = nullptr;
            std::cerr << "hipStreamCreateWithFlags failed\n";
            return false;
        }
        if (hipEventCreate(&thread.event) != hipSuccess) {
            thread.event = nullptr;
            std::cerr << "hipEventCreate failed\n";
            return false;
        }
        if (hipHostMalloc(&thread.signals,
                          (READY_FLAGS_OFFSET + 2 * size_t{options.workgroups}) * sizeof(uint32_t),
                          hipHostMallocCoherent) != hipSuccess) {
            thread.signals = nullptr;
            std::cerr << "hipHostMalloc failed\n";
            return false;
        }
        thread.pattern.emplace(word_count);
        thread.observed.emplace(word_count);
    }

    size_t     passed_iterations{0};
    const auto start{std::chrono::steady_clock::now()};
    const auto failures{checkWriteVisibility(resources, options, passed_iterations)};
    const auto elapsed{std::chrono::duration<double>{std::chrono::steady_clock::now() - start}};

    const int signal_number{stop_signal.load(std::memory_order_relaxed)};
    if (signal_number != 0) {
        std::cerr << "Stopped by " << (signal_number == SIGTERM ? "SIGTERM" : "SIGINT") << '\n';
    }
    if (!failures.empty()) {
        for (const auto &failure : failures) {
            std::cerr << failure << '\n';
        }
        std::cerr << passed_iterations << " iterations of " << testDescription(options)
                  << " passed before the failure\n";
        return false;
    }

    std::cout << passed_iterations << " iterations of " << options.buffer_size << " bytes with "
              << testDescription(options) << ", on " << options.threads
              << (options.threads == 1 ? " thread" : " threads") << " and " << options.workgroups
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
        "Checks that data in GPU memory is in the file once hipFileWrite returns. hipFile chooses the "
        "backend; set HIPFILE_FORCE_COMPAT_MODE=true to test only fallback, or "
        "HIPFILE_ALLOW_COMPAT_MODE=false to test only fastpath. The test runs on HIP device 0; set "
        "ROCR_VISIBLE_DEVICES to test a different GPU."};
    [[maybe_unused]] args::HelpFlag help_arg(parser, "help", "Show this help message",
                                             args::Matcher{'h', "help"});

    args::ValueFlag<std::string> directory_arg(
        parser, "path",
        "Directory for the temporary file hipFileWrite writes, which is deleted on exit; fastpath needs an "
        "AIS capable file system (required)",
        args::Matcher{"directory"}, args::Options::Single);

    args::ValueFlag<std::string> source_directory_arg(
        parser, "path",
        "With --producer read, directory for the temporary file hipFileRead reads, which is deleted on "
        "exit (default: the --directory path)",
        args::Matcher{"source-directory"}, args::Options::Single);

    args::ValueFlag<std::string> buffer_size_arg(
        parser, "size",
        "Size of each thread's part of the GPU buffer, in bytes or with a K, M, or G suffix. It must be a "
        "multiple of 4K and at most 2G minus the page size (default: 1M)",
        args::Matcher{"buffer-size"}, "1M", args::Options::Single);

    args::ValueFlag<long long> iterations_arg(
        parser, "count",
        "hipFileWrite calls each thread checks, or -1 to run until SIGTERM or SIGINT (required)",
        args::Matcher{"iterations"}, args::Options::Single);

    const std::unordered_map<std::string, Producer> producers{{"kernel", Producer::Kernel},
                                                              {"memcpy", Producer::Memcpy},
                                                              {"memset", Producer::Memset},
                                                              {"copy", Producer::Copy},
                                                              {"read", Producer::Read}};

    args::MapFlag<std::string, Producer> producer_arg(
        parser, "kernel|memcpy|memset|copy|read",
        "What stores each generation in the GPU buffer: a kernel, hipMemcpy from host memory, "
        "hipMemsetD32, hipMemcpy from a second GPU buffer, or hipFileRead from a second temporary file "
        "(default: kernel)",
        args::Matcher{"producer"}, producers, Producer::Kernel, args::Options::Single);

    const std::unordered_map<std::string, KernelTiming> timings{{"after", KernelTiming::After},
                                                                {"during", KernelTiming::During}};

    args::MapFlag<std::string, KernelTiming> timing_arg(
        parser, "after|during",
        "With --producer kernel, when the kernel runs: it finishes before hipFileWrite, with the buffer in "
        "coarse-grained memory, or it stores the buffer and is still running during hipFileWrite, with the "
        "buffer in fine-grained memory (default: after)",
        args::Matcher{"kernel"}, timings, KernelTiming::After, args::Options::Single);

    const std::unordered_map<std::string, ProducerWait> waits{
        {"sync", ProducerWait::Sync},         {"query", ProducerWait::Query}, {"event", ProducerWait::Event},
        {"callback", ProducerWait::Callback}, {"flag", ProducerWait::Flag},   {"sleep", ProducerWait::Sleep}};

    args::MapFlag<std::string, ProducerWait> wait_arg(
        parser, "sync|query|event|callback|flag|sleep",
        "How the host waits for a kernel that finishes before hipFileWrite, and for the kernel of --prime "
        "store: hipStreamSynchronize; polling hipStreamQuery; polling hipEventQuery on an event recorded "
        "after the kernel; a host function enqueued after the kernel; spinning on flags the kernel sets "
        "in host memory, which doesn't wait for the kernel to finish; or sleeping, then checking the flags "
        "(default: sync)",
        args::Matcher{"wait"}, waits, ProducerWait::Sync, args::Options::Single);

    const std::unordered_map<std::string, Prime> primes{
        {"none", Prime::None}, {"write", Prime::Write}, {"store", Prime::Store}};

    args::MapFlag<std::string, Prime> prime_arg(
        parser, "none|write|store",
        "What uses the buffer before the producer: nothing; hipFileWrite, which writes the previous "
        "generation to another part of the file; or, with --producer memcpy or read, a kernel that "
        "stores another generation (default: none)",
        args::Matcher{"prime"}, primes, Prime::None, args::Options::Single);

    const std::unordered_map<std::string, WriteApi> write_apis{{"sync", WriteApi::Sync},
                                                               {"async", WriteApi::Async}};

    args::MapFlag<std::string, WriteApi> write_api_arg(
        parser, "sync|async",
        "Write with hipFileWrite, or with hipFileWriteAsync on the producer's stream, which can't be used "
        "with --kernel during (default: sync)",
        args::Matcher{"write"}, write_apis, WriteApi::Sync, args::Options::Single);

    const std::unordered_map<std::string, Memory> memories{
        {"coarse", Memory::Coarse}, {"fine", Memory::Fine}, {"uncached", Memory::Uncached}};

    args::MapFlag<std::string, Memory> memory_arg(
        parser, "coarse|fine|uncached",
        "The kind of GPU memory the buffer is in (default: fine with --kernel during, coarse otherwise)",
        args::Matcher{"memory"}, memories, Memory::Coarse, args::Options::Single);

    args::ValueFlag<unsigned int> workgroups_arg(
        parser, "count",
        "Workgroups in each kernel, each storing an equal share of a thread's part of the buffer (default "
        "and maximum: the GPU's compute units divided among the threads, or the part's 64-bit words if "
        "fewer)",
        args::Matcher{"workgroups"}, args::Options::Single);

    args::ValueFlag<size_t> threads_arg(
        parser, "count",
        "Threads running the test at once, each on its own part of the GPU buffer and of the file "
        "(default: 1)",
        args::Matcher{"threads"}, size_t{1}, args::Options::Single);

    args::ValueFlag<size_t> background_writers_arg(
        parser, "count",
        "Threads, for each thread running the test, that keep writing its part of the GPU buffer to "
        "other parts of the file with hipFileWrite (default: 0)",
        args::Matcher{"background-writers"}, size_t{0}, args::Options::Single);
    parseCli(parser, argc, argv);

    const Producer     producer{args::get(producer_arg)};
    const KernelTiming timing{args::get(timing_arg)};
    const bool         kernel_during{producer == Producer::Kernel && timing == KernelTiming::During};
    const Memory memory{memory_arg ? args::get(memory_arg) : (kernel_during ? Memory::Fine : Memory::Coarse)};

    // Without this, hipExtMallocWithFlags(hipDeviceMallocFinegrained) silently
    // returns coarse-grained memory on PCIe GPUs. ROCr reads the variable when
    // HIP initializes, so it must be set before the first HIP call.
    if (memory == Memory::Fine && setenv("HSA_FORCE_FINE_GRAIN_PCIE", "1", 1) != 0) {
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
    const size_t                threads{args::get(threads_arg)};
    // Every workgroup of a kernel that is running during hipFileWrite waits for
    // it, so all the threads' workgroups must fit on the GPU at once; one per
    // compute unit always does
    const auto max_workgroups{static_cast<unsigned int>(
        std::clamp(static_cast<size_t>(compute_units) / std::max(threads, size_t{1}), size_t{1},
                   std::max(buffer_size.value_or(0) / sizeof(uint64_t), size_t{1})))};

    const long long iterations{args::get(iterations_arg)};

    const Options options{
        .directory        = args::get(directory_arg),
        .source_directory = source_directory_arg ? args::get(source_directory_arg) : args::get(directory_arg),
        .buffer_size      = buffer_size.value_or(0),
        .iterations       = iterations >= 0 ? std::optional{static_cast<size_t>(iterations)} : std::nullopt,
        .producer         = producer,
        .timing           = timing,
        .wait             = args::get(wait_arg),
        .prime            = args::get(prime_arg),
        .write_api        = args::get(write_api_arg),
        .memory           = memory,
        .workgroups       = workgroups_arg ? args::get(workgroups_arg) : max_workgroups,
        .threads          = threads,
        .background_writers = args::get(background_writers_arg),
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
    if (options.buffer_size > maxWriteSize()) {
        std::cerr << "--buffer-size can be at most " << maxWriteSize()
                  << " bytes, the most one hipFileWrite can write\n";
        return EXIT_FAILURE;
    }
    if (options.threads == 0 || options.threads > static_cast<size_t>(compute_units)) {
        std::cerr << "--threads must be from 1 to " << compute_units << ", the GPU's compute units\n";
        return EXIT_FAILURE;
    }
    if (options.workgroups == 0 || options.workgroups > max_workgroups) {
        std::cerr << "--workgroups must be from 1 to " << max_workgroups << '\n';
        return EXIT_FAILURE;
    }
    if (timing_arg && producer != Producer::Kernel) {
        std::cerr << "--kernel only applies to --producer kernel\n";
        return EXIT_FAILURE;
    }
    if (source_directory_arg && producer != Producer::Read) {
        std::cerr << "--source-directory only applies to --producer read\n";
        return EXIT_FAILURE;
    }
    if (options.prime == Prime::Store && producer != Producer::Memcpy && producer != Producer::Read) {
        std::cerr << "--prime store needs --producer memcpy or read, which can write the buffer without "
                     "a kernel\n";
        return EXIT_FAILURE;
    }
    if (options.write_api == WriteApi::Async && kernel_during) {
        std::cerr << "--write async can't be used with --kernel during, as a kernel waiting on the stream "
                     "would block the write\n";
        return EXIT_FAILURE;
    }
    const bool kernel_waited_for{options.prime == Prime::Store ||
                                 (producer == Producer::Kernel && timing == KernelTiming::After &&
                                  options.write_api == WriteApi::Sync)};
    if (wait_arg && !kernel_waited_for) {
        std::cerr << "--wait only applies to a kernel the host waits for: --producer kernel --kernel after "
                     "--write sync, or --prime store\n";
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
