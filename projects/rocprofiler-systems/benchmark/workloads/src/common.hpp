// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace workload
{
struct args
{
    std::uint64_t size  = 0;
    std::uint64_t iters = 0;
    std::uint64_t seed  = 1;
    // Optional knobs: a program enables one by giving it a nonzero default.
    std::uint64_t depth    = 0;
    std::uint64_t threads  = 0;
    std::uint64_t streams  = 0;
    std::uint64_t gpu_size = 0;
};

/** Parses --size/--iters/--seed, plus --depth/--threads/--streams/--gpu-size when
 * defaulted nonzero (also "--size=N"); exits 0 on --help, 2 on bad usage. */
inline args
parse_args(int argc, char** argv, args defaults, const char* name, const char* help)
{
    auto usage = [&](int code) {
        std::fprintf(code ? stderr : stdout,
                     "usage: %s [--size N] [--iters N] [--seed N]\n%s\n"
                     "defaults: size=%llu iters=%llu seed=%llu\n",
                     name, help, (unsigned long long) defaults.size,
                     (unsigned long long) defaults.iters,
                     (unsigned long long) defaults.seed);
        if(defaults.depth)
            std::fprintf(code ? stderr : stdout, "         --depth=%llu\n",
                         (unsigned long long) defaults.depth);
        if(defaults.threads)
            std::fprintf(code ? stderr : stdout, "         --threads=%llu\n",
                         (unsigned long long) defaults.threads);
        if(defaults.gpu_size)
            std::fprintf(code ? stderr : stdout, "         --gpu-size=%llu\n",
                         (unsigned long long) defaults.gpu_size);
        if(defaults.streams)
            std::fprintf(code ? stderr : stdout, "         --streams=%llu\n",
                         (unsigned long long) defaults.streams);
        std::exit(code);
    };
    for(int i = 1; i < argc; ++i)
    {
        std::string key = argv[i];
        if(key == "--help" || key == "-h") usage(0);
        std::string value;
        if(auto eq = key.find('='); eq != std::string::npos)
        {
            value = key.substr(eq + 1);
            key.resize(eq);
        }
        else if(i + 1 < argc)
            value = argv[++i];
        char*         end = nullptr;
        std::uint64_t v   = std::strtoull(value.c_str(), &end, 10);
        if(value.empty() || *end != '\0') usage(2);
        if(key == "--size")
            defaults.size = v;
        else if(key == "--iters")
            defaults.iters = v;
        else if(key == "--seed")
            defaults.seed = v;
        else if(key == "--depth" && defaults.depth)
            defaults.depth = v;
        else if(key == "--threads" && defaults.threads)
            defaults.threads = v;
        else if(key == "--streams" && defaults.streams)
            defaults.streams = v;
        else if(key == "--gpu-size" && defaults.gpu_size)
            defaults.gpu_size = v;
        else
            usage(2);
    }
    return defaults;
}

constexpr std::uint64_t
splitmix64(std::uint64_t& state) noexcept
{
    std::uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
    z               = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z               = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

constexpr std::uint64_t
mix(std::uint64_t h, std::uint64_t v) noexcept
{
    return (h ^ v) * 0x100000001b3ULL + (h >> 29);
}

inline void
print_checksum(std::uint64_t sum)
{
    std::printf("checksum: %016llx\n", (unsigned long long) sum);
}
}  // namespace workload
