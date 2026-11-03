// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "bench_hooks.h"
#include "common.hpp"
#include <cstdint>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
class temp_dir
{
public:
    temp_dir()
    {
        const char* base = std::getenv("TMPDIR");
        std::string tmpl =
            std::string(base && *base ? base : "/tmp") + "/bench_io_XXXXXX";
        if(mkdtemp(tmpl.data()) != nullptr) m_path = tmpl;
    }
    ~temp_dir()
    {
        std::error_code ec;
        if(!m_path.empty()) fs::remove_all(m_path, ec);
    }
    temp_dir(const temp_dir&)                  = delete;
    temp_dir&       operator=(const temp_dir&) = delete;
    const fs::path& path() const { return m_path; }

private:
    fs::path m_path;
};

std::uint64_t
run(const workload::args& a, const fs::path& dir)
{
    std::uint64_t     rng = a.seed, sum = 0;
    std::vector<char> data(a.size * 1024);
    for(std::uint64_t f = 0; f < a.iters; ++f)
    {
        for(auto& c : data)
            c = static_cast<char>(workload::splitmix64(rng));
        const auto p = dir / ("f" + std::to_string(f));
        {
            std::ofstream out(p, std::ios::binary);
            out.write(data.data(), static_cast<std::streamsize>(data.size()));
        }
        std::ifstream           in(p, std::ios::binary);
        const std::vector<char> back((std::istreambuf_iterator<char>(in)), {});
        for(char c : back)
            sum = workload::mix(sum, static_cast<unsigned char>(c));
    }
    return sum;
}
}  // namespace

int
main(int argc, char** argv)
{
    const auto a = workload::parse_args(
        argc, argv, { 64, 6500, 1 }, "io_files",
        "size = file size in KB, iters = number of files "
        "(each written then read back in a temp dir, removed on exit).");
    const temp_dir dir;
    if(dir.path().empty()) return 1;
    bench_hook(BENCH_HOOK_WORK_BEGIN);
    const std::uint64_t sum = run(a, dir.path());
    bench_hook(BENCH_HOOK_WORK_END);
    workload::print_checksum(sum);
    return 0;
}
