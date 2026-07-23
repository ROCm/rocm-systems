#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <string.h>

#include <cerrno>
#include <cstring>
#include <cassert>

#include <core/ipc.hpp>

#include "ringbuffer.hpp"

namespace rocm_timesync
{

namespace ipc
{

#define POLL_WAIT_MS 100

struct header_t
{
    uint32_t magic;
    uint32_t version;
    uint64_t ring_len;
};

struct ring_t
{
    alignas(64) std::atomic<uint64_t> write_idx;
    alignas(64) std::atomic<uint32_t> write_seq;

    event_t events[];
};

struct cursor_t
{
    uint64_t read_idx;
};

struct ringbuffer_shm_t
{

    header_t header;
    ring_t ring; // must be last
};

int ringbuffer_t::create(std::string name, uint8_t ring_order)
{
    const size_t ring_len = (1U << ring_order);
    const size_t rbuf_size = sizeof(ringbuffer_t) + ring_len * sizeof(event_t);
    ringbuffer_shm_t *rbuf = nullptr;

    // create shm object
    int fd = shm_open(
        name.c_str(),
        O_CREAT | O_RDWR,
        0660
    );
    if (fd < 0)
        return -errno;

    if (ftruncate(fd, rbuf_size) != 0) {
        close(fd);
        goto err_truncate;
    }

    // map shm object
    rbuf = static_cast<ringbuffer_shm_t*>(mmap(
        nullptr,
        rbuf_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        0
    ));
    close(fd);

    if (rbuf == MAP_FAILED)
        goto err_mmap;

    std::memset(rbuf, 0, rbuf_size);

    rbuf->header.magic = MAGIC;
    rbuf->header.version = VERSION;
    rbuf->header.ring_len = ring_len;

    this->name = name;
    this->rbuf = rbuf;
    return 0;

err_sock:
    munmap(rbuf, rbuf_size);
err_mmap:
err_truncate:
    shm_unlink(name.c_str());
    return -errno;
}

int ringbuffer_t::attach(std::string name)
{
    ringbuffer_shm_t *rbuf = nullptr;
    size_t ring_len, rbuf_size;

    int fd = shm_open(
        name.c_str(),
        O_RDONLY,
        0);
    if (fd < 0)
        return -errno;

    // map the header to determine the size 
    auto* hdr = static_cast<header_t*>(mmap(
        nullptr,
        sizeof(header_t),
        PROT_READ,
        MAP_SHARED,
        fd,
        0
    ));
    if (hdr == MAP_FAILED)
        goto err_hdr_map;

    if (hdr->magic != MAGIC) {
        fprintf(stderr, "Corrupt header: MAGIC %d != %d\n", hdr->magic, MAGIC);
        errno = EIO;
        goto err_hdr;
    }

    ring_len = hdr->ring_len;
    munmap(hdr, sizeof(header_t));

    if ((ring_len == 0) || ((ring_len & (ring_len - 1)) != 0)) {
        fprintf(stderr, "ring_len (%ld) is either 0 or not a power of 2", ring_len);
        errno = EIO;
        goto err_hdr;
    }

    rbuf_size = sizeof(ringbuffer_t) + ring_len * sizeof(event_t);
    rbuf = static_cast<ringbuffer_shm_t*>(mmap(
        nullptr,
        rbuf_size,
        PROT_READ,
        MAP_SHARED,
        fd,
        0
    ));
    close(fd);

    if (rbuf == MAP_FAILED)
        goto err_rbuf;

    assert(rbuf->header.ring_len == ring_len);

    this->name = name;
    this->rbuf = rbuf;
    return 0;

err_hdr:
    munmap(hdr, sizeof(header_t));
err_rbuf:
err_hdr_map:
    close(fd);
    return -errno;
}

void ringbuffer_t::destroy()
{
    this->detach();
    shm_unlink(name.c_str());
}

void ringbuffer_t::detach()
{
    munmap(rbuf, size());
}

uint64_t ringbuffer_t::size() const
{
    return sizeof(ringbuffer_t) + rbuf->header.ring_len * sizeof(event_t);
}

uint64_t ringbuffer_t::length() const
{
    return rbuf->header.ring_len;
}

void ringbuffer_t::publish(std::vector<event_t>& events)
{
    const header_t& header = rbuf->header;
    ring_t& ring = rbuf->ring;

    uint64_t idx = ring.write_idx.load(std::memory_order_relaxed);
    for (const auto& event : events)
        ring.events[idx++ & (header.ring_len - 1)] = event;

    ring.write_idx.store(idx, std::memory_order_release);

    // wake consumers
    ring.write_seq.fetch_add(1, std::memory_order_relaxed);
    atomic_notify_all(ring.write_seq);
}

static bool consume(
    const ringbuffer_shm_t* rbuf, cursor_t& cursor, const callback_t& callback
)
{
    const header_t& header = rbuf->header;
    const ring_t& ring = rbuf->ring;
    const uint64_t write_idx = ring.write_idx.load(std::memory_order_acquire);

    if (write_idx == cursor.read_idx)
        return false;

    if ((write_idx - cursor.read_idx) > header.ring_len)
        cursor.read_idx = write_idx - header.ring_len;

    while (cursor.read_idx < write_idx) {
        const event_t& event = ring.events[cursor.read_idx & (header.ring_len - 1)];
        callback(event);
        ++cursor.read_idx;
    }

    return true;
}

void ringbuffer_t::poll(const callback_t& callback) const
{
    const ring_t& ring = rbuf->ring;
    auto cursor = cursor_t();

    while (!stop_requested.load(std::memory_order_acquire)) {
        // cache last produced value
        uint32_t seq = ring.write_seq.load(std::memory_order_relaxed);

        // read data
        auto processed = consume(rbuf, cursor, callback);
        if (!processed) {
            printf("waiting ...\n");
            atomic_wait_for(ring.write_seq, seq, POLL_WAIT_MS);
        }
    }
}

void ringbuffer_t::stop()
{
    stop_requested.store(true, std::memory_order_release);
}

#undef POLL_WAIT_MS

} // namespace ipc
} // namespace rocm_timesync
