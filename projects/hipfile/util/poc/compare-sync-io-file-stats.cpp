/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <hip/hip_runtime_api.h>
#include <hipfile.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <iostream>
#include <linux/stat.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

namespace {

constexpr mode_t file_mode{0640};
constexpr size_t default_io_size{4096};
constexpr size_t default_file_size{1024 * 1024};

class FileDescriptor {
public:
    explicit FileDescriptor(int fd = -1) : fd_{fd}
    {
    }
    ~FileDescriptor()
    {
        if (fd_ >= 0) {
            close(fd_);
        }
    }

    FileDescriptor(const FileDescriptor &)            = delete;
    FileDescriptor &operator=(const FileDescriptor &) = delete;

    int get() const
    {
        return fd_;
    }

    int release()
    {
        const int fd{fd_};
        fd_ = -1;
        return fd;
    }

private:
    int fd_;
};

std::runtime_error
system_error(std::string_view operation)
{
    return std::runtime_error(std::string{operation} + ": " + std::strerror(errno));
}

void
check_hip(hipError_t error, std::string_view operation)
{
    if (error != hipSuccess) {
        throw std::runtime_error(std::string{operation} + ": " + hipGetErrorString(error));
    }
}

void
check_hipfile(hipFileError_t error, std::string_view operation)
{
    if (error.err != hipFileSuccess) {
        throw std::runtime_error(std::string{operation} + ": " + hipFileGetOpErrorString(error.err));
    }
}

class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t size)
    {
        check_hip(hipMalloc(&ptr_, size), "hipMalloc");
        try {
            check_hipfile(hipFileBufRegister(ptr_, size, 0), "hipFileBufRegister");
            registered_ = true;
        }
        catch (...) {
            static_cast<void>(hipFree(ptr_));
            ptr_ = nullptr;
            throw;
        }
    }

    ~DeviceBuffer()
    {
        if (registered_) {
            static_cast<void>(hipFileBufDeregister(ptr_));
        }
        if (ptr_) {
            static_cast<void>(hipFree(ptr_));
        }
    }

    DeviceBuffer(const DeviceBuffer &)            = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;

    void *get() const
    {
        return ptr_;
    }

private:
    void *ptr_{nullptr};
    bool  registered_{false};
};

class RegisteredFile {
public:
    explicit RegisteredFile(int fd)
    {
        hipFileDescr_t descriptor{};
        descriptor.type      = hipFileHandleTypeOpaqueFD;
        descriptor.handle.fd = fd;
        check_hipfile(hipFileHandleRegister(&handle_, &descriptor), "hipFileHandleRegister");
    }

    ~RegisteredFile()
    {
        if (handle_) {
            static_cast<void>(hipFileHandleDeregister(handle_));
        }
    }

    RegisteredFile(const RegisteredFile &)            = delete;
    RegisteredFile &operator=(const RegisteredFile &) = delete;

    hipFileHandle_t get() const
    {
        return handle_;
    }

private:
    hipFileHandle_t handle_{nullptr};
};

struct MetadataSnapshot {
    struct stat  fstat_data {};
    struct statx statx_data {};
};

enum class ComparisonPolicy {
    ExactTransition,
    ChangedOnly,
};

struct FieldValue {
    std::string      name;
    std::string      value;
    ComparisonPolicy policy;
};

struct FieldTransition {
    std::string      name;
    std::string      before;
    std::string      after;
    ComparisonPolicy policy;
};

std::string
format_unsigned(uint64_t value)
{
    return std::to_string(value);
}

std::string
format_signed(int64_t value)
{
    return std::to_string(value);
}

std::string
format_octal(uint64_t value)
{
    std::ostringstream output;
    output << '0' << std::oct << value;
    return output.str();
}

std::string
format_hex(uint64_t value)
{
    std::ostringstream output;
    output << "0x" << std::hex << value;
    return output.str();
}

std::string
format_device(dev_t device)
{
    return std::to_string(major(device)) + ":" + std::to_string(minor(device));
}

std::string
format_timespec(const struct timespec &time)
{
    std::ostringstream output;
    output << time.tv_sec << '.' << std::setw(9) << std::setfill('0') << time.tv_nsec;
    return output.str();
}

std::string
format_statx_timestamp(const struct statx_timestamp &time)
{
    std::ostringstream output;
    output << time.tv_sec << '.' << std::setw(9) << std::setfill('0') << time.tv_nsec;
    return output.str();
}

MetadataSnapshot
snapshot_metadata(int fd)
{
    MetadataSnapshot snapshot;
    if (fstat(fd, &snapshot.fstat_data) == -1) {
        throw system_error("fstat");
    }

    unsigned int mask{STATX_BASIC_STATS | STATX_BTIME};
#if defined(STATX_DIOALIGN)
    mask |= STATX_DIOALIGN;
#endif
    if (statx(fd, "", AT_EMPTY_PATH | AT_STATX_SYNC_AS_STAT, mask, &snapshot.statx_data) == -1) {
        throw system_error("statx");
    }
    return snapshot;
}

MetadataSnapshot
snapshot_metadata(const std::string &path)
{
    FileDescriptor fd{open(path.c_str(), O_RDONLY | O_CLOEXEC)};
    if (fd.get() == -1) {
        throw system_error("open file for post-close metadata");
    }
    return snapshot_metadata(fd.get());
}

std::vector<FieldValue>
metadata_fields(const MetadataSnapshot &snapshot)
{
    const auto            &st{snapshot.fstat_data};
    const auto            &stx{snapshot.statx_data};
    const ComparisonPolicy exact{ComparisonPolicy::ExactTransition};
    const ComparisonPolicy changed_only{ComparisonPolicy::ChangedOnly};

    std::vector<FieldValue> fields{
        {"fstat.st_dev", format_device(st.st_dev), exact},
        {"fstat.st_ino", format_unsigned(st.st_ino), changed_only},
        {"fstat.st_mode", format_octal(st.st_mode), exact},
        {"fstat.st_nlink", format_unsigned(st.st_nlink), exact},
        {"fstat.st_uid", format_unsigned(st.st_uid), exact},
        {"fstat.st_gid", format_unsigned(st.st_gid), exact},
        {"fstat.st_rdev", format_device(st.st_rdev), exact},
        {"fstat.st_size", format_signed(st.st_size), exact},
        {"fstat.st_blksize", format_signed(st.st_blksize), exact},
        {"fstat.st_blocks", format_signed(st.st_blocks), exact},
        {"fstat.st_atim", format_timespec(st.st_atim), changed_only},
        {"fstat.st_mtim", format_timespec(st.st_mtim), changed_only},
        {"fstat.st_ctim", format_timespec(st.st_ctim), changed_only},
        {"statx.stx_mask", format_hex(stx.stx_mask), exact},
        {"statx.stx_blksize", format_unsigned(stx.stx_blksize), exact},
        {"statx.stx_attributes", format_hex(stx.stx_attributes), exact},
        {"statx.stx_nlink", format_unsigned(stx.stx_nlink), exact},
        {"statx.stx_uid", format_unsigned(stx.stx_uid), exact},
        {"statx.stx_gid", format_unsigned(stx.stx_gid), exact},
        {"statx.stx_mode", format_octal(stx.stx_mode), exact},
        {"statx.stx_ino", format_unsigned(stx.stx_ino), changed_only},
        {"statx.stx_size", format_unsigned(stx.stx_size), exact},
        {"statx.stx_blocks", format_unsigned(stx.stx_blocks), exact},
        {"statx.stx_attributes_mask", format_hex(stx.stx_attributes_mask), exact},
        {"statx.stx_atime", format_statx_timestamp(stx.stx_atime), changed_only},
        {"statx.stx_btime", format_statx_timestamp(stx.stx_btime), changed_only},
        {"statx.stx_ctime", format_statx_timestamp(stx.stx_ctime), changed_only},
        {"statx.stx_mtime", format_statx_timestamp(stx.stx_mtime), changed_only},
        {"statx.stx_rdev", std::to_string(stx.stx_rdev_major) + ":" + std::to_string(stx.stx_rdev_minor),
         exact},
        {"statx.stx_dev", std::to_string(stx.stx_dev_major) + ":" + std::to_string(stx.stx_dev_minor), exact},
    };
#if defined(STATX_DIOALIGN)
    fields.push_back({"statx.stx_dio_mem_align", format_unsigned(stx.stx_dio_mem_align), exact});
    fields.push_back({"statx.stx_dio_offset_align", format_unsigned(stx.stx_dio_offset_align), exact});
#endif
    return fields;
}

std::vector<FieldTransition>
metadata_transitions(const MetadataSnapshot &before, const MetadataSnapshot &after)
{
    const auto before_fields{metadata_fields(before)};
    const auto after_fields{metadata_fields(after)};
    if (before_fields.size() != after_fields.size()) {
        throw std::runtime_error("metadata field list changed between snapshots");
    }

    std::vector<FieldTransition> transitions;
    transitions.reserve(before_fields.size());
    for (size_t i{}; i < before_fields.size(); ++i) {
        if (before_fields[i].name != after_fields[i].name ||
            before_fields[i].policy != after_fields[i].policy) {
            throw std::runtime_error("metadata field ordering changed between snapshots");
        }
        transitions.push_back(
            {before_fields[i].name, before_fields[i].value, after_fields[i].value, before_fields[i].policy});
    }
    return transitions;
}

void
write_all(int fd, const uint8_t *data, size_t size)
{
    size_t written{};
    while (written < size) {
        const ssize_t result{write(fd, data + written, size - written)};
        if (result == -1) {
            throw system_error("write baseline");
        }
        written += static_cast<size_t>(result);
    }
}

void
seed_file(const std::string &path, const std::vector<uint8_t> &contents)
{
    FileDescriptor fd{open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY | O_CLOEXEC, file_mode)};
    if (fd.get() == -1) {
        throw system_error("open baseline file");
    }
    write_all(fd.get(), contents.data(), contents.size());
    if (fsync(fd.get()) == -1) {
        throw system_error("fsync baseline file");
    }

    const struct timespec fixed_times[2]{{946684800, 0}, {946684801, 0}};
    if (futimens(fd.get(), fixed_times) == -1) {
        throw system_error("futimens baseline file");
    }
}

uint64_t
fnv1a(const uint8_t *data, size_t size)
{
    uint64_t hash{14695981039346656037ULL};
    for (size_t i{}; i < size; ++i) {
        hash ^= data[i];
        hash *= 1099511628211ULL;
    }
    return hash;
}

uint64_t
hash_file(const std::string &path)
{
    FileDescriptor fd{open(path.c_str(), O_RDONLY | O_CLOEXEC)};
    if (fd.get() == -1) {
        throw system_error("open file for hashing");
    }

    std::vector<uint8_t> buffer(64 * 1024);
    uint64_t             hash{14695981039346656037ULL};
    while (true) {
        const ssize_t result{read(fd.get(), buffer.data(), buffer.size())};
        if (result == -1) {
            throw system_error("read file for hashing");
        }
        if (result == 0) {
            break;
        }
        for (ssize_t i{}; i < result; ++i) {
            hash ^= buffer[static_cast<size_t>(i)];
            hash *= 1099511628211ULL;
        }
    }
    return hash;
}

std::string
format_hash(uint64_t hash)
{
    std::ostringstream output;
    output << "0x" << std::hex << std::setw(16) << std::setfill('0') << hash;
    return output.str();
}

struct AlignedBuffer {
    explicit AlignedBuffer(size_t alignment, size_t size) : size{size}
    {
        const int error{posix_memalign(&data, alignment, size)};
        if (error != 0) {
            throw std::runtime_error("posix_memalign: " + std::string{std::strerror(error)});
        }
    }

    ~AlignedBuffer()
    {
        free(data);
    }

    AlignedBuffer(const AlignedBuffer &)            = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;

    void  *data{};
    size_t size{};
};

enum class Api {
    Posix,
    HipFile,
};

enum class Operation {
    Read,
    Write,
};

struct Scenario {
    std::string name;
    Operation   operation;
    size_t      size;
    off_t       offset;
};

struct RunResult {
    MetadataSnapshot before;
    MetadataSnapshot after;
    MetadataSnapshot after_close;
    ssize_t          return_value{};
    int              saved_errno{};
    uint64_t         file_hash{};
    uint64_t         read_buffer_hash{};
    bool             has_read_buffer_hash{false};
};

RunResult
run_one(const std::string &path, Api api, const Scenario &scenario, const std::vector<uint8_t> &baseline,
        const std::vector<uint8_t> &write_pattern, AlignedBuffer &host_buffer, DeviceBuffer &device_buffer)
{
    seed_file(path, baseline);

    FileDescriptor fd{open(path.c_str(), O_RDWR | O_CLOEXEC)};
    if (fd.get() == -1) {
        throw system_error("open test file");
    }

    std::unique_ptr<RegisteredFile> registered_file;
    if (api == Api::HipFile) {
        registered_file = std::make_unique<RegisteredFile>(fd.get());
    }

    if (scenario.operation == Operation::Write && scenario.size > 0) {
        std::memcpy(host_buffer.data, write_pattern.data(), scenario.size);
        check_hip(hipMemcpy(device_buffer.get(), write_pattern.data(), scenario.size, hipMemcpyHostToDevice),
                  "hipMemcpy write pattern to device");
    }
    else if (scenario.size > 0) {
        std::memset(host_buffer.data, 0, scenario.size);
        check_hip(hipMemset(device_buffer.get(), 0, scenario.size), "hipMemset read buffer");
    }
    if (scenario.size > 0) {
        check_hip(hipDeviceSynchronize(), "hipDeviceSynchronize after buffer preparation");
    }

    RunResult result;
    result.before = snapshot_metadata(fd.get());
    std::this_thread::sleep_for(std::chrono::milliseconds{2});

    errno = 0;
    if (api == Api::Posix) {
        result.return_value = scenario.operation == Operation::Read
                                  ? pread(fd.get(), host_buffer.data, scenario.size, scenario.offset)
                                  : pwrite(fd.get(), host_buffer.data, scenario.size, scenario.offset);
    }
    else {
        result.return_value =
            scenario.operation == Operation::Read
                ? hipFileRead(registered_file->get(), device_buffer.get(), scenario.size, scenario.offset, 0)
                : hipFileWrite(registered_file->get(), device_buffer.get(), scenario.size, scenario.offset,
                               0);
    }
    result.saved_errno = errno;

    if (scenario.operation == Operation::Write && result.return_value >= 0 && fsync(fd.get()) == -1) {
        throw system_error("fsync test file");
    }
    result.after = snapshot_metadata(fd.get());

    if (scenario.operation == Operation::Read && scenario.size > 0 && result.return_value > 0) {
        const size_t transferred{static_cast<size_t>(result.return_value)};
        if (api == Api::HipFile) {
            std::vector<uint8_t> host_copy(transferred);
            check_hip(hipMemcpy(host_copy.data(), device_buffer.get(), transferred, hipMemcpyDeviceToHost),
                      "hipMemcpy read result to host");
            result.read_buffer_hash = fnv1a(host_copy.data(), host_copy.size());
        }
        else {
            result.read_buffer_hash = fnv1a(static_cast<const uint8_t *>(host_buffer.data), transferred);
        }
        result.has_read_buffer_hash = true;
    }

    registered_file.reset();
    if (close(fd.release()) == -1) {
        throw system_error("close test file");
    }
    result.after_close = snapshot_metadata(path);
    result.file_hash   = hash_file(path);
    return result;
}

bool
print_comparison(const Scenario &scenario, const RunResult &posix, const RunResult &hipfile)
{
    std::cout << "\n=== " << scenario.name << " (size=" << scenario.size << ", offset=" << scenario.offset
              << ") ===\n";
    std::cout << "RETURN\tPOSIX=" << posix.return_value << " errno=" << posix.saved_errno
              << "\tHIPFILE=" << hipfile.return_value << " errno=" << hipfile.saved_errno << '\n';
    std::cout << "CONTENT\tPOSIX=" << format_hash(posix.file_hash)
              << "\tHIPFILE=" << format_hash(hipfile.file_hash) << '\t'
              << (posix.file_hash == hipfile.file_hash ? "MATCH" : "MISMATCH") << '\n';
    if (posix.has_read_buffer_hash || hipfile.has_read_buffer_hash) {
        std::cout << "READ_BUFFER\tPOSIX="
                  << (posix.has_read_buffer_hash ? format_hash(posix.read_buffer_hash) : "none")
                  << "\tHIPFILE="
                  << (hipfile.has_read_buffer_hash ? format_hash(hipfile.read_buffer_hash) : "none") << '\t'
                  << (posix.has_read_buffer_hash == hipfile.has_read_buffer_hash &&
                              posix.read_buffer_hash == hipfile.read_buffer_hash
                          ? "MATCH"
                          : "MISMATCH")
                  << '\n';
    }

    const auto posix_transitions{metadata_transitions(posix.before, posix.after)};
    const auto hipfile_transitions{metadata_transitions(hipfile.before, hipfile.after)};
    if (posix_transitions.size() != hipfile_transitions.size()) {
        throw std::runtime_error("POSIX and hipFile metadata field lists differ");
    }

    std::cout << "FIELD\tPOSIX before -> after\tHIPFILE before -> after\tCOMPARISON\n";
    bool metadata_matches{true};
    for (size_t i{}; i < posix_transitions.size(); ++i) {
        const auto &posix_field{posix_transitions[i]};
        const auto &hipfile_field{hipfile_transitions[i]};
        if (posix_field.name != hipfile_field.name || posix_field.policy != hipfile_field.policy) {
            throw std::runtime_error("POSIX and hipFile metadata field ordering differs");
        }

        const bool posix_changed{posix_field.before != posix_field.after};
        const bool hipfile_changed{hipfile_field.before != hipfile_field.after};
        const bool match{posix_field.policy == ComparisonPolicy::ChangedOnly
                             ? posix_changed == hipfile_changed
                             : posix_field.before == hipfile_field.before &&
                                   posix_field.after == hipfile_field.after};
        metadata_matches &= match;

        std::cout << posix_field.name << '\t' << posix_field.before << " -> " << posix_field.after << " ("
                  << (posix_changed ? "changed" : "unchanged") << ")\t" << hipfile_field.before << " -> "
                  << hipfile_field.after << " (" << (hipfile_changed ? "changed" : "unchanged") << ")\t"
                  << (match ? "MATCH" : "MISMATCH") << '\n';
    }

    const auto posix_post_close{metadata_transitions(posix.before, posix.after_close)};
    const auto hipfile_post_close{metadata_transitions(hipfile.before, hipfile.after_close)};
    bool       post_close_matches{true};
    std::cout << "POST_CLOSE_FIELD\tPOSIX before -> post-close\tHIPFILE before -> post-close\tCOMPARISON\n";
    for (size_t i{}; i < posix_post_close.size(); ++i) {
        const auto &posix_field{posix_post_close[i]};
        const auto &hipfile_field{hipfile_post_close[i]};
        if (posix_field.name != hipfile_field.name || posix_field.policy != hipfile_field.policy) {
            throw std::runtime_error("POSIX and hipFile post-close metadata field ordering differs");
        }

        const bool posix_changed{posix_field.before != posix_field.after};
        const bool hipfile_changed{hipfile_field.before != hipfile_field.after};
        const bool match{posix_field.policy == ComparisonPolicy::ChangedOnly
                             ? posix_changed == hipfile_changed
                             : posix_field.before == hipfile_field.before &&
                                   posix_field.after == hipfile_field.after};
        post_close_matches &= match;

        if (posix_changed || hipfile_changed || !match) {
            std::cout << "post-close." << posix_field.name << '\t' << posix_field.before << " -> "
                      << posix_field.after << " (" << (posix_changed ? "changed" : "unchanged") << ")\t"
                      << hipfile_field.before << " -> " << hipfile_field.after << " ("
                      << (hipfile_changed ? "changed" : "unchanged") << ")\t"
                      << (match ? "MATCH" : "MISMATCH") << '\n';
        }
    }
    std::cout << "POST_CLOSE_RESULT\t" << (post_close_matches ? "MATCH" : "MISMATCH") << '\n';

    const bool returns_match{posix.return_value == static_cast<ssize_t>(scenario.size) &&
                             hipfile.return_value == static_cast<ssize_t>(scenario.size)};
    const bool content_matches{posix.file_hash == hipfile.file_hash};
    const bool read_buffers_match{
        posix.has_read_buffer_hash == hipfile.has_read_buffer_hash &&
        (!posix.has_read_buffer_hash || posix.read_buffer_hash == hipfile.read_buffer_hash)};
    const bool passed{returns_match && content_matches && read_buffers_match && metadata_matches &&
                      post_close_matches};
    std::cout << "CASE_RESULT\t" << (passed ? "PASS" : "FAIL") << '\n';
    return passed;
}

bool
is_power_of_two(size_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

size_t
align_up(size_t value, size_t alignment)
{
    return (value + alignment - 1) & ~(alignment - 1);
}

struct DirectIoAlignment {
    size_t memory{default_io_size};
    size_t offset{default_io_size};
};

DirectIoAlignment
direct_io_alignment(const std::string &directory)
{
    const std::string probe_path{directory + "/alignment-probe"};
    seed_file(probe_path, std::vector<uint8_t>(default_io_size));
    FileDescriptor probe{open(probe_path.c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC)};
    if (probe.get() == -1) {
        throw system_error("open alignment probe");
    }

    const MetadataSnapshot snapshot{snapshot_metadata(probe.get())};
    DirectIoAlignment      alignment;
#if defined(STATX_DIOALIGN)
    if (snapshot.statx_data.stx_mask & STATX_DIOALIGN) {
        alignment.memory = snapshot.statx_data.stx_dio_mem_align;
        alignment.offset = snapshot.statx_data.stx_dio_offset_align;
    }
#endif
    if (!is_power_of_two(alignment.memory) || !is_power_of_two(alignment.offset)) {
        throw std::runtime_error("statx returned a non-power-of-two direct-I/O alignment");
    }
    if (unlink(probe_path.c_str()) == -1) {
        throw system_error("unlink alignment probe");
    }
    return alignment;
}

void
fill_patterns(std::vector<uint8_t> &baseline, std::vector<uint8_t> &write_pattern)
{
    for (size_t i{}; i < baseline.size(); ++i) {
        baseline[i] = static_cast<uint8_t>((i * 131U + 17U) & 0xffU);
    }
    for (size_t i{}; i < write_pattern.size(); ++i) {
        write_pattern[i] = static_cast<uint8_t>(0xa5U ^ ((i * 29U) & 0xffU));
    }
}

} // namespace

int
main(int argc, char **argv)
try {
    if (argc < 2 || argc > 3) {
        std::cerr << "Usage: " << argv[0] << " RUN_DIRECTORY [GPU_ID]\n";
        return EXIT_FAILURE;
    }

    const std::string run_directory{argv[1]};
    const int         gpu_id{argc == 3 ? std::stoi(argv[2]) : 0};

    check_hip(hipSetDevice(gpu_id), "hipSetDevice");

    const DirectIoAlignment alignment{direct_io_alignment(run_directory)};
    const size_t            io_size{std::max(default_io_size, alignment.offset)};
    const size_t            file_size{align_up(std::max(default_file_size, 2 * io_size), alignment.offset)};
    const size_t            host_alignment{std::max({default_io_size, alignment.memory, sizeof(void *)})};

    std::cout << "RUN_DIRECTORY\t" << run_directory << '\n';
    std::cout << "GPU_ID\t" << gpu_id << '\n';
    std::cout << "DIO_MEMORY_ALIGNMENT\t" << alignment.memory << '\n';
    std::cout << "DIO_OFFSET_ALIGNMENT\t" << alignment.offset << '\n';
    std::cout << "NORMAL_IO_SIZE\t" << io_size << '\n';
    std::cout << "INITIAL_FILE_SIZE\t" << file_size << '\n';
    std::cout << "NOTE\tExact transitions are compared for structural fields; timestamps and inode values "
                 "are compared by changed/unchanged state.\n";

    std::vector<uint8_t> baseline(file_size);
    std::vector<uint8_t> write_pattern(io_size);
    fill_patterns(baseline, write_pattern);

    AlignedBuffer host_buffer{host_alignment, io_size};
    DeviceBuffer  device_buffer{io_size};

    const std::vector<Scenario> scenarios{
        {"read-zero-in-range", Operation::Read, 0, 0},
        {"read-zero-beyond-eof", Operation::Read, 0, static_cast<off_t>(2 * file_size)},
        {"read-normal-in-range", Operation::Read, io_size, 0},
        {"write-zero-in-range", Operation::Write, 0, 0},
        {"write-zero-beyond-eof", Operation::Write, 0, static_cast<off_t>(2 * file_size)},
        {"write-normal-overwrite", Operation::Write, io_size, 0},
        {"write-normal-extend", Operation::Write, io_size, static_cast<off_t>(file_size)},
    };

    size_t passed{};
    for (const auto &scenario : scenarios) {
        const std::string posix_path{run_directory + "/" + scenario.name + ".posix"};
        const std::string hipfile_path{run_directory + "/" + scenario.name + ".hipfile"};

        const RunResult posix{
            run_one(posix_path, Api::Posix, scenario, baseline, write_pattern, host_buffer, device_buffer)};
        const RunResult hipfile{run_one(hipfile_path, Api::HipFile, scenario, baseline, write_pattern,
                                        host_buffer, device_buffer)};
        passed += print_comparison(scenario, posix, hipfile) ? 1 : 0;
    }

    std::cout << "\nSUMMARY\t" << passed << '/' << scenarios.size() << " cases passed\n";
    return passed == scenarios.size() ? EXIT_SUCCESS : EXIT_FAILURE;
}
catch (const std::exception &error) {
    std::cerr << "ERROR\t" << error.what() << '\n';
    return EXIT_FAILURE;
}
