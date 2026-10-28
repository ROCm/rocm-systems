// Unit tests for cpu_snap::sampling::perf_ring, driven by a hand-built mmap
// ring fed with pre-canned PERF_RECORD_* bytes: plain and wrapped records,
// data_tail release semantics, and unmap on destruction.

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include <fcntl.h>
#include <sys/epoll.h>
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <unistd.h>

#include "cpu_snap/sampling/perf_ring.hpp"

namespace
{

using cpu_snap::sampling::perf_ring;
using cpu_snap::sampling::perf_ring_config;

/// 4 data pages + 1 metadata page = 5 pages.
constexpr std::size_t k_data_pages = 4;

class PerfRingTest : public ::testing::Test
{
protected:
    int         fd          = -1;
    void*       base        = nullptr;
    std::size_t total_bytes = 0;
    std::size_t data_bytes  = 0;
    long        page_sz     = 0;
    char*       data_ring   = nullptr;
    perf_event_mmap_page* meta = nullptr;

    void SetUp() override
    {
        page_sz     = ::sysconf(_SC_PAGESIZE);
        total_bytes = static_cast<std::size_t>(page_sz) * (1 + k_data_pages);
        data_bytes  = static_cast<std::size_t>(page_sz) * k_data_pages;

        fd = ::memfd_create("perf_ring_test", 0);
        ASSERT_GE(fd, 0);
        ASSERT_EQ(::ftruncate(fd, static_cast<off_t>(total_bytes)), 0);

        base = ::mmap(nullptr, total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED,
                      fd, 0);
        ASSERT_NE(base, MAP_FAILED);
        std::memset(base, 0, total_bytes);

        meta      = reinterpret_cast<perf_event_mmap_page*>(base);
        data_ring = static_cast<char*>(base) + page_sz;
    }

    void TearDown() override
    {
        // perf_ring's dtor will munmap+close on adopt; for cases where
        // we did not hand off (or the test failed early), do cleanup
        // here.
        if(base != nullptr && base != MAP_FAILED)
        {
            ::munmap(base, total_bytes);
            base = nullptr;
        }
        if(fd >= 0)
        {
            ::close(fd);
            fd = -1;
        }
    }

    perf_ring_config adopt()
    {
        perf_ring_config cfg{};
        cfg.fd         = fd;
        cfg.mmap_base  = base;
        cfg.mmap_bytes = total_bytes;
        fd   = -1;
        base = nullptr;
        return cfg;
    }
};

/// Write a synthetic PERF_RECORD with `type`, `size`, and `size` bytes
/// of body (header included). Returns new tail.
void
write_record(char* ring, std::size_t ring_bytes, std::uint64_t tail,
             std::uint32_t type, std::uint16_t size,
             const std::uint8_t* body)
{
    const std::uint64_t mask = ring_bytes - 1;
    perf_event_header   hdr{};
    hdr.type = type;
    hdr.misc = 0;
    hdr.size = size;

    const std::uint64_t pos = tail & mask;
    if(pos + size <= ring_bytes)
    {
        std::memcpy(ring + pos, &hdr, sizeof(hdr));
        if(body != nullptr && size > sizeof(hdr))
        {
            std::memcpy(ring + pos + sizeof(hdr), body, size - sizeof(hdr));
        }
    }
    else
    {
        const std::uint64_t first = ring_bytes - pos;
        std::vector<std::uint8_t> buf(size);
        std::memcpy(buf.data(), &hdr, sizeof(hdr));
        if(body != nullptr && size > sizeof(hdr))
        {
            std::memcpy(buf.data() + sizeof(hdr), body, size - sizeof(hdr));
        }
        std::memcpy(ring + pos, buf.data(), first);
        std::memcpy(ring, buf.data() + first, size - first);
    }
}

}  // namespace

TEST(PerfRing, ConstructWithEmptyConfigYieldsInvalidRing)
{
    perf_ring ring{ perf_ring_config{} };
    EXPECT_FALSE(ring.valid());
    EXPECT_EQ(ring.data_bytes(), 0U);
}

TEST_F(PerfRingTest, DrainDispatchesEveryNonWrappedRecord)
{
    perf_ring ring{ adopt() };
    ASSERT_TRUE(ring.valid());
    ASSERT_GT(ring.data_bytes(), 0U);

    // Two 16-byte records back to back at the head of the ring.
    constexpr std::uint16_t rec_size = 16;
    std::array<std::uint8_t, rec_size - sizeof(perf_event_header)> body_a{};
    std::array<std::uint8_t, rec_size - sizeof(perf_event_header)> body_b{};
    body_a.fill(0xAA);
    body_b.fill(0xBB);

    auto* ring_data_ring =
        static_cast<char*>(ring.mmap_base()) + page_sz;
    write_record(ring_data_ring, ring.data_bytes(), 0, PERF_RECORD_SAMPLE,
                 rec_size, body_a.data());
    write_record(ring_data_ring, ring.data_bytes(), rec_size,
                 PERF_RECORD_SAMPLE, rec_size, body_b.data());

    auto* ring_meta =
        reinterpret_cast<perf_event_mmap_page*>(ring.mmap_base());
    __atomic_store_n(&ring_meta->data_head, std::uint64_t{ 2 * rec_size },
                     __ATOMIC_RELEASE);

    std::vector<std::uint32_t> seen_types;
    std::vector<std::uint8_t>  seen_first_body_byte;
    const std::size_t          drained =
        ring.drain([&](const perf_event_header* hdr, const std::uint8_t* record) {
            seen_types.push_back(hdr->type);
            seen_first_body_byte.push_back(record[sizeof(perf_event_header)]);
        });

    EXPECT_EQ(drained, 2U);
    ASSERT_EQ(seen_types.size(), 2U);
    EXPECT_EQ(seen_types[0], static_cast<std::uint32_t>(PERF_RECORD_SAMPLE));
    EXPECT_EQ(seen_types[1], static_cast<std::uint32_t>(PERF_RECORD_SAMPLE));
    EXPECT_EQ(seen_first_body_byte[0], 0xAAU);
    EXPECT_EQ(seen_first_body_byte[1], 0xBBU);

    // Tail must have advanced; a subsequent drain returns 0.
    EXPECT_EQ(ring_meta->data_tail, 2U * rec_size);
    EXPECT_EQ(ring.drain([](auto*, auto*) {}), 0U);
}

TEST_F(PerfRingTest, DrainSplicesWrappedRecordIntoContiguousBuffer)
{
    perf_ring ring{ adopt() };
    ASSERT_TRUE(ring.valid());

    auto* ring_data_ring =
        static_cast<char*>(ring.mmap_base()) + page_sz;
    auto* ring_meta =
        reinterpret_cast<perf_event_mmap_page*>(ring.mmap_base());

    // Park tail one record's worth before the ring boundary so the
    // next record wraps. Use a 32-byte record so the wrap split is
    // unambiguous.
    constexpr std::uint16_t rec_size = 32;
    const std::uint64_t     start_tail =
        ring.data_bytes() - (rec_size - 8);  // 8 bytes of record sit at end
    ring_meta->data_tail = start_tail;

    std::array<std::uint8_t, rec_size - sizeof(perf_event_header)> body{};
    for(std::size_t i = 0; i < body.size(); ++i)
    {
        body[i] = static_cast<std::uint8_t>(0x10U + i);
    }
    write_record(ring_data_ring, ring.data_bytes(), start_tail,
                 PERF_RECORD_SAMPLE, rec_size, body.data());
    __atomic_store_n(&ring_meta->data_head, start_tail + rec_size,
                     __ATOMIC_RELEASE);

    bool       saw_record = false;
    const auto drained    =
        ring.drain([&](const perf_event_header* hdr, const std::uint8_t* record) {
            EXPECT_EQ(hdr->type,
                      static_cast<std::uint32_t>(PERF_RECORD_SAMPLE));
            EXPECT_EQ(hdr->size, rec_size);
            // Verify the body bytes are contiguous in `record` even though
            // the underlying ring split them.
            for(std::size_t i = 0; i < rec_size - sizeof(perf_event_header); ++i)
            {
                EXPECT_EQ(record[sizeof(perf_event_header) + i],
                          static_cast<std::uint8_t>(0x10U + i))
                    << "wrap splice corrupted body at i=" << i;
            }
            saw_record = true;
        });

    EXPECT_EQ(drained, 1U);
    EXPECT_TRUE(saw_record);
    EXPECT_EQ(ring_meta->data_tail, start_tail + rec_size);
}

TEST_F(PerfRingTest, MoveConstructTransfersOwnership)
{
    perf_ring a{ adopt() };
    ASSERT_TRUE(a.valid());

    perf_ring b{ std::move(a) };
    EXPECT_TRUE(b.valid());
    EXPECT_FALSE(a.valid());   // NOLINT(bugprone-use-after-move): post-move state intentional
    EXPECT_EQ(a.fd(), -1);     // NOLINT(bugprone-use-after-move)

    // b's dtor handles cleanup.
}

TEST_F(PerfRingTest, ThrowingDecoderDoesNotReplayConsumedRecords)
{
    perf_ring ring{ adopt() };
    auto*     ring_data = static_cast<char*>(ring.mmap_base()) + page_sz;
    auto*     ring_meta = reinterpret_cast<perf_event_mmap_page*>(ring.mmap_base());

    constexpr std::uint16_t rec_size = 16;
    std::array<std::uint8_t, rec_size - sizeof(perf_event_header)> body{};
    for(std::uint64_t i = 0; i < 3; ++i)
    {
        body.fill(static_cast<std::uint8_t>(i + 1));
        write_record(ring_data, ring.data_bytes(), i * rec_size, PERF_RECORD_SAMPLE,
                     rec_size, body.data());
    }
    __atomic_store_n(&ring_meta->data_head, std::uint64_t{ 3 * rec_size },
                     __ATOMIC_RELEASE);

    std::vector<std::uint8_t> seen;
    EXPECT_THROW(ring.drain([&](const perf_event_header*, const std::uint8_t* record) {
                     seen.push_back(record[sizeof(perf_event_header)]);
                     if(seen.size() == 2)
                     {
                         throw std::runtime_error("decoder failure");
                     }
                 }),
                 std::runtime_error);
    ASSERT_EQ(seen.size(), 2U);

    seen.clear();
    EXPECT_EQ(ring.drain([&](const perf_event_header*, const std::uint8_t* record) {
                  seen.push_back(record[sizeof(perf_event_header)]);
              }),
              1U);
    ASSERT_EQ(seen.size(), 1U);
    EXPECT_EQ(seen[0], 3U);
}

TEST_F(PerfRingTest, ZeroSizeRecordIsDroppedWithoutLooping)
{
    perf_ring ring{ adopt() };
    auto*     ring_data = static_cast<char*>(ring.mmap_base()) + page_sz;
    auto*     ring_meta = reinterpret_cast<perf_event_mmap_page*>(ring.mmap_base());

    write_record(ring_data, ring.data_bytes(), 0, PERF_RECORD_SAMPLE, 0, nullptr);
    __atomic_store_n(&ring_meta->data_head, std::uint64_t{ 64 }, __ATOMIC_RELEASE);

    std::size_t calls = 0;
    EXPECT_EQ(ring.drain([&](auto*, auto*) { ++calls; }), 0U);
    EXPECT_EQ(calls, 0U);
    EXPECT_EQ(ring_meta->data_tail, 64U);
}

TEST_F(PerfRingTest, RecordLargerThanRingIsDropped)
{
    perf_ring ring{ adopt() };
    auto*     ring_data = static_cast<char*>(ring.mmap_base()) + page_sz;
    auto*     ring_meta = reinterpret_cast<perf_event_mmap_page*>(ring.mmap_base());

    perf_event_header hdr{};
    hdr.type = PERF_RECORD_SAMPLE;
    hdr.size = static_cast<std::uint16_t>(ring.data_bytes() + 8);
    std::memcpy(ring_data, &hdr, sizeof(hdr));
    __atomic_store_n(&ring_meta->data_head, std::uint64_t{ 128 }, __ATOMIC_RELEASE);

    std::size_t calls = 0;
    EXPECT_EQ(ring.drain([&](auto*, auto*) { ++calls; }), 0U);
    EXPECT_EQ(calls, 0U);
    EXPECT_EQ(ring_meta->data_tail, 128U);
}

TEST_F(PerfRingTest, HonoursDataOffsetAndSizeFromMetadataPage)
{
    meta->data_offset = static_cast<std::uint64_t>(2 * page_sz);
    meta->data_size   = static_cast<std::uint64_t>(2 * page_sz);

    perf_ring ring{ adopt() };
    ASSERT_TRUE(ring.valid());
    EXPECT_EQ(ring.data_bytes(), static_cast<std::size_t>(2 * page_sz));

    auto* ring_data = static_cast<char*>(ring.mmap_base()) + 2 * page_sz;
    auto* ring_meta = reinterpret_cast<perf_event_mmap_page*>(ring.mmap_base());

    constexpr std::uint16_t rec_size = 16;
    std::array<std::uint8_t, rec_size - sizeof(perf_event_header)> body{};
    body.fill(0x5A);
    write_record(ring_data, ring.data_bytes(), 0, PERF_RECORD_SAMPLE, rec_size,
                 body.data());
    __atomic_store_n(&ring_meta->data_head, std::uint64_t{ rec_size }, __ATOMIC_RELEASE);

    std::uint8_t first = 0;
    EXPECT_EQ(ring.drain([&](const perf_event_header*, const std::uint8_t* record) {
                  first = record[sizeof(perf_event_header)];
              }),
              1U);
    EXPECT_EQ(first, 0x5AU);
}

namespace
{

// Registers a dup of a pipe read end in an epoll set, hands the dup to a ring,
// destroys the ring and reports whether the registration survived.
bool registration_survives_ring_destruction(bool forget)
{
    int p[2];
    EXPECT_EQ(::pipe(p), 0);
    const int ep    = ::epoll_create1(EPOLL_CLOEXEC);
    const int ring_fd = ::dup(p[0]);
    epoll_event ev{};
    ev.events = EPOLLIN;
    EXPECT_EQ(::epoll_ctl(ep, EPOLL_CTL_ADD, ring_fd, &ev), 0);
    {
        perf_ring_config cfg{};
        cfg.fd                 = ring_fd;
        cfg.epoll_fd_for_dereg = ep;
        perf_ring ring{ cfg };
        if(forget)
        {
            ring.forget_epoll_registration();
        }
    }
    const char byte = 1;
    EXPECT_EQ(::write(p[1], &byte, 1), 1);
    epoll_event out{};
    const int   n = ::epoll_wait(ep, &out, 1, 0);
    ::close(ep);
    ::close(p[0]);
    ::close(p[1]);
    return n == 1;
}

}  // namespace

TEST(PerfRingEpoll, DtorDeregistersUnlessForgotten)
{
    EXPECT_FALSE(registration_survives_ring_destruction(false));
    EXPECT_TRUE(registration_survives_ring_destruction(true));
}
