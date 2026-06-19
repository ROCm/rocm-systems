#include <core/ipc.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>

#include <cstring>
#include <cassert>

#include "atomic.hpp"

namespace rocm_timesync
{

namespace ipc
{

state_t *create(std::string name, uint8_t ring_order)
{
    const size_t ring_size = (1U << ring_order);
    const size_t state_size = sizeof(state_t) + ring_size * sizeof(event_t);

    int fd = shm_open(
        name.c_str(),
        O_CREAT | O_RDWR,
        0660
    );
    if (fd < 0)
        return nullptr;

    if (ftruncate(fd, state_size) != 0) {
        close(fd);
        return nullptr;
    }

    auto *state = static_cast<state_t*>(mmap(
        nullptr,
        state_size,
        PROT_READ | PROT_WRITE,
        MAP_SHARED,
        fd,
        0
    ));
    close(fd);

    if (state == MAP_FAILED)
        return nullptr;

    std::memset(state, 0, state_size);

    state->header.magic = MAGIC;
    state->header.version = VERSION;
    state->header.ring_size = ring_size;

    return state;
}

const state_t* attach(std::string name)
{
    int fd = shm_open(
        name.c_str(),
        O_RDONLY,
        0);
    if (fd < 0)
        return nullptr;

    // map the header to determine the size 
    auto* hdr = static_cast<header_t*>(mmap(
        nullptr,
        sizeof(header_t),
        PROT_READ,
        MAP_SHARED,
        fd,
        0
    ));
    if (hdr == MAP_FAILED) {
        fprintf(stderr, "Failed to map header: %s", strerror(errno));
        close(fd);
        return nullptr;
    }

    if (hdr->magic != MAGIC) {
        fprintf(stderr, "Corrupt header: MAGIC %d != %d\n", hdr->magic, MAGIC);
        munmap(hdr, sizeof(header_t));
        close(fd);
        return nullptr;
    }

    const size_t ring_size = hdr->ring_size;
    munmap(hdr, sizeof(header_t));

    if ((ring_size == 0) || ((ring_size & (ring_size - 1)) != 0)) {
        fprintf(stderr, "ring_size (%ld) is either 0 or not a power of 2", ring_size);
        close(fd);
        return nullptr;
    }

    const size_t state_size = sizeof(state_t) + ring_size * sizeof(event_t);
    auto* state = static_cast<state_t*>(mmap(
        nullptr,
        state_size,
        PROT_READ,
        MAP_SHARED,
        fd,
        0
    ));
    close(fd);

    if (state == MAP_FAILED)
        return nullptr;

    assert(state->header.ring_size == ring_size);
    return state;
}

void destroy(state_t* state)
{
    const size_t state_size = sizeof(state_t) + state->header.ring_size * sizeof(event_t);
    munmap(state, state_size);
}

void publish(state_t* state, event_t& event)
{
    const header_t& header = state->header;
    ringbuffer_t& rbuf = state->rbuf;

    const uint64_t idx = rbuf.write_idx.load();
    rbuf.events[idx & (header.ring_size - 1)] = event;
    rbuf.write_idx.store(idx+1, std::memory_order_release);

    // wake consumers
    rbuf.wait_seq.fetch_add(1, std::memory_order_relaxed);
    atomic_notify_all(rbuf.wait_seq);
}

template <typename Callback>
bool poll(const state_t* state, cursor_t& cursor, Callback&& callback)
{
    const header_t& header = state->header;
    const ringbuffer_t& rbuf = state->rbuf;
    const uint64_t write_idx = rbuf.write_idx.load(std::memory_order_acquire);

    if (write_idx == cursor.read_idx)
        return false;

    if ((write_idx - cursor.read_idx) > header.ring_size)
        cursor.read_idx = write_idx - header.ring_size;

    while (cursor.read_idx < write_idx) {
        const event_t& event = rbuf.events[cursor.read_idx & (header.ring_size - 1)];
        callback(event);
        ++cursor.read_idx;
    }

    return true;
}

void test(const state_t* state, cursor_t& cursor)
{
    const ringbuffer_t& rbuf = state->rbuf;

    while (true) {
        uint32_t seq = rbuf.wait_seq.load(std::memory_order_relaxed);

        auto processed = poll(state, cursor, [](const event_t& event) {
            printf("processed event with gpu_id: %u gpu_timestamp: %lu \n", event.gpu_id, event.gpu_timestamp_ns);
        });

        if (!processed) {
            printf("waiting ...\n");
            atomic_wait(rbuf.wait_seq, seq);
        }
    }
}

} // namespace ipc
} // namespace rocm_timesync
