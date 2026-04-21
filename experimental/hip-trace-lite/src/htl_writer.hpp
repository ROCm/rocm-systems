// htl_writer.hpp — owns the output fd and the drain loop.
#pragma once

#include "htl_record.hpp"
#include "htl_ring.hpp"
#include <atomic>
#include <cstdint>
#include <string>
#include <thread>

namespace htl {

inline constexpr size_t kRingCapacity = 1u << 16;  // 65536

struct slot_t {
    record_t rec;
    char     name[128];   // inlined kernel_name copy; truncated to 127+NUL
};

class Writer {
public:
    Writer();
    ~Writer();

    // Open the output file and start the drain thread. Returns false on failure.
    bool start(const std::string& path);

    // Stop the drain thread, flush remaining slots, write the string section
    // and footer, close the fd. Idempotent.
    void stop();

    // Producer-side enqueue. Returns false (and bumps drop counter) if full.
    bool enqueue(const slot_t& s);

    uint64_t records_written() const { return written_.load(std::memory_order_relaxed); }
    uint64_t records_dropped() const { return dropped_.load(std::memory_order_relaxed); }

private:
    void run();

    SpscRing<slot_t, kRingCapacity> ring_;
    std::thread thread_;
    std::atomic<bool> stop_{false};
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> written_{0};
    std::atomic<uint64_t> dropped_{0};
    int fd_ = -1;
    std::string path_;
};

}  // namespace htl
