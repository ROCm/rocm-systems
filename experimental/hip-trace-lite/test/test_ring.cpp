// test/test_ring.cpp
#include "htl_ring.hpp"
#include <cassert>
#include <cstdio>
#include <thread>
#include <vector>

int main() {
    htl::SpscRing<int, 8> ring;

    // Empty pop returns false.
    int out = -1;
    assert(!ring.try_pop(out));

    // Push then pop returns same value.
    assert(ring.try_push(42));
    assert(ring.try_pop(out));
    assert(out == 42);

    // Fill to capacity-1 (one slot reserved to disambiguate full vs empty).
    for (int i = 0; i < 7; ++i) assert(ring.try_push(i));
    assert(!ring.try_push(99));  // full

    // Drain.
    for (int i = 0; i < 7; ++i) {
        assert(ring.try_pop(out));
        assert(out == i);
    }
    assert(!ring.try_pop(out));

    // Threaded smoke: 1 producer, 1 consumer, 100k items.
    htl::SpscRing<int, 1024> r2;
    constexpr int N = 100000;
    std::thread prod([&] {
        for (int i = 0; i < N; ++i) {
            while (!r2.try_push(i)) std::this_thread::yield();
        }
    });
    int last = -1;
    for (int i = 0; i < N; ++i) {
        int v;
        while (!r2.try_pop(v)) std::this_thread::yield();
        assert(v == last + 1);
        last = v;
    }
    prod.join();

    std::printf("test_ring: ok\n");
    return 0;
}
