// htl_writer.cpp
#include "htl_writer.hpp"

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fcntl.h>
#include <sys/uio.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

namespace htl {

namespace {
uint64_t now_ns() {
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + ts.tv_nsec;
}
}  // namespace

Writer::Writer() = default;

Writer::~Writer() { stop(); }

bool Writer::start(const std::string& path) {
    path_ = path;
    fd_ = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) {
        std::fprintf(stderr, "[hip-trace-lite] open(%s) failed: %s\n",
                     path.c_str(), std::strerror(errno));
        return false;
    }

    file_header_t hdr{};
    std::memcpy(hdr.magic, kFileMagic, 4);
    hdr.version = kFileVersion;
    hdr.start_ns = now_ns();
    hdr.pid = static_cast<uint64_t>(::getpid());
    hdr.record_size = sizeof(record_t);
    hdr.header_size = kHeaderSize;

    if (::write(fd_, &hdr, sizeof(hdr)) != static_cast<ssize_t>(sizeof(hdr))) {
        std::fprintf(stderr, "[hip-trace-lite] header write failed\n");
        ::close(fd_);
        fd_ = -1;
        return false;
    }

    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this] { run(); });
    return true;
}

bool Writer::enqueue(const slot_t& s) {
    if (!running_.load(std::memory_order_acquire)) return false;
    if (!ring_.try_push(s)) {
        dropped_.fetch_add(1, std::memory_order_relaxed);
        return false;
    }
    return true;
}

void Writer::run() {
    constexpr size_t kBatch = 64;
    std::vector<slot_t> batch;
    batch.reserve(kBatch);
    std::vector<std::string> strings;        // collected kernel_name copies
    uint64_t string_cursor = 0;              // running offset into the future string section

    auto flush = [&]() {
        if (batch.empty()) return;
        // Patch kernel_name_off into each record before writing.
        for (size_t i = 0; i < batch.size(); ++i) {
            if (batch[i].name[0] != '\0') {
                strings.emplace_back(batch[i].name);
                batch[i].rec.kernel_name_off = string_cursor;
                string_cursor += sizeof(uint32_t) + strings.back().size();
            } else {
                batch[i].rec.kernel_name_off = 0;
            }
        }
        // writev only the record_t prefix of each slot.
        std::vector<iovec> iov;
        iov.reserve(batch.size());
        for (auto& s : batch) iov.push_back({&s.rec, sizeof(record_t)});
        ssize_t want = static_cast<ssize_t>(batch.size() * sizeof(record_t));
        ssize_t got  = ::writev(fd_, iov.data(), static_cast<int>(iov.size()));
        if (got != want) {
            std::fprintf(stderr, "[hip-trace-lite] writev short/failed: %zd of %zd\n",
                         got, want);
        }
        written_.fetch_add(batch.size(), std::memory_order_relaxed);
        batch.clear();
    };

    while (!stop_.load(std::memory_order_acquire) || !ring_.empty()) {
        slot_t s;
        if (ring_.try_pop(s)) {
            batch.push_back(s);
            if (batch.size() >= kBatch) flush();
        } else {
            flush();
            timespec ts{0, 100000};  // 100us nap
            nanosleep(&ts, nullptr);
        }
    }
    flush();

    // String section.
    file_footer_t foot{};
    off_t string_offset = ::lseek(fd_, 0, SEEK_CUR);
    for (const auto& str : strings) {
        uint32_t len = static_cast<uint32_t>(str.size());
        ::write(fd_, &len, sizeof(len));
        ::write(fd_, str.data(), str.size());
    }
    off_t string_end = ::lseek(fd_, 0, SEEK_CUR);

    foot.records_written = written_.load(std::memory_order_relaxed);
    foot.records_dropped = dropped_.load(std::memory_order_relaxed);
    ::write(fd_, &foot, sizeof(foot));

    // Patch header with string-section location.
    ::lseek(fd_, offsetof(file_header_t, string_section_offset), SEEK_SET);
    uint64_t off_val = static_cast<uint64_t>(string_offset);
    uint64_t sz_val  = static_cast<uint64_t>(string_end - string_offset);
    ::write(fd_, &off_val, sizeof(off_val));
    ::write(fd_, &sz_val,  sizeof(sz_val));
}

void Writer::stop() {
    if (!running_.exchange(false)) return;
    stop_.store(true, std::memory_order_release);
    if (thread_.joinable()) thread_.join();
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

}  // namespace htl
