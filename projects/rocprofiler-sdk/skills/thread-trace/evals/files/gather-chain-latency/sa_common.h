// Frozen plumbing shared by the standalone harnesses: argument parsing, random inputs,
// output poisoning, and timing of the whole launch function.
#pragma once
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIP_CHECK(x)                                                                               \
    do                                                                                             \
    {                                                                                              \
        hipError_t e_ = (x);                                                                       \
        if(e_ != hipSuccess)                                                                       \
        {                                                                                          \
            std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, hipGetErrorString(e_));         \
            std::exit(1);                                                                          \
        }                                                                                          \
    } while(0)

namespace sa
{

struct Args
{
    bool     once  = false;  // one launch, for a profiler capture
    uint64_t seed  = 7;      // input seed
    int      iters = 20;     // timed calls
};

inline Args
parse(int argc, char** argv)
{
    Args a;
    for(int i = 1; i < argc; ++i)
    {
        if(!std::strcmp(argv[i], "--once"))
            a.once = true;
        else if(!std::strcmp(argv[i], "--seed") && i + 1 < argc)
            a.seed = std::strtoull(argv[++i], nullptr, 10);
        else if(!std::strcmp(argv[i], "--iters") && i + 1 < argc)
            a.iters = std::atoi(argv[++i]);
    }
    return a;
}

struct Rng
{
    uint64_t s;
    explicit Rng(uint64_t seed)
    : s(seed * 0x9E3779B97F4A7C15ull + 1)
    {}
    uint64_t next()
    {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return s;
    }
    float uniform(float lo, float hi)
    {
        return lo + (hi - lo) * float(next() >> 40) / float(1 << 24);
    }
    int below(int n) { return int(next() % uint64_t(n)); }
};

inline void
poison(void* dev, size_t bytes)
{
    HIP_CHECK(hipMemset(dev, 0xFF, bytes));
}

inline void
sync_all()
{
    int n = 0, cur = 0;
    HIP_CHECK(hipGetDevice(&cur));
    HIP_CHECK(hipGetDeviceCount(&n));
    for(int d = 0; d < n; ++d)
    {
        HIP_CHECK(hipSetDevice(d));
        HIP_CHECK(hipDeviceSynchronize());
    }
    HIP_CHECK(hipSetDevice(cur));
}

// Times each call of launch() from the host with every device synchronized before and
// after it, so work the launch function does on the host counts. before(i) runs untimed
// and changes the inputs, so a result cannot be reused from an earlier call.
template <class Before, class Launch>
double
median_ms(int iters, Before before, Launch launch)
{
    for(int w = 0; w < 2; ++w)
    {
        before(-1 - w);
        launch();
    }
    std::vector<double> ms;
    for(int i = 0; i < iters; ++i)
    {
        before(i);
        sync_all();
        auto t0 = std::chrono::steady_clock::now();
        launch();
        sync_all();
        auto t1 = std::chrono::steady_clock::now();
        ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

inline int
fail(const char* what)
{
    std::printf("FAIL  %s\n", what);
    return 1;
}

}  // namespace sa
