// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Per-test scratch driver for the basics/ and async/ example CTests.
//
// Creates a unique scratch directory under --base via mkdtemp (the directory
// analogue of mkstemp used in test/common/test-common.h's Tmpfile helper),
// optionally seeds an input file with deterministic bytes, substitutes
// {TMPDIR} and {INPUT} tokens in the child argv, runs the example, and
// removes the directory on the way out. This replaces the shared-fixture
// model whose cleanup raced across CI runners sharing AIS_CAPABLE_DIR.
//
// Usage:
//   examples_tmpdriver --base <dir> [--seed-input <bytes>] [--seed <u32>]
//                      -- <program> [args...]
//
// Tokens recognized in <args...>:
//   {TMPDIR} -> path to the unique scratch dir
//   {INPUT}  -> path to the seeded input file (requires --seed-input)

#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <ftw.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

namespace {

// std::filesystem deliberately avoided: rocky8 (in the hipFile CI matrix)
// ships GCC 8, where libstdc++ requires linking -lstdc++fs. Sticking to
// POSIX keeps the driver buildable across rocky / rocky8 / suse / ubuntu.

std::string g_scratch;
pid_t       g_child = -1;

int
nftw_unlink(const char *path, const struct stat *, int, struct FTW *)
{
    remove(path);
    return 0;
}

void
cleanup()
{
    if (g_child > 0) {
        kill(g_child, SIGTERM);
        waitpid(g_child, nullptr, 0);
        g_child = -1;
    }
    if (!g_scratch.empty()) {
        nftw(g_scratch.c_str(), nftw_unlink, 16, FTW_DEPTH | FTW_PHYS);
        g_scratch.clear();
    }
}

void
mkdir_p(const std::string &path)
{
    std::size_t pos = 0;
    while (pos != std::string::npos) {
        pos             = path.find('/', pos + 1);
        std::string sub = path.substr(0, pos);
        if (sub.empty())
            continue;
        if (mkdir(sub.c_str(), 0755) != 0 && errno != EEXIST) {
            std::fprintf(stderr, "tmpdriver: mkdir(%s) failed: %s\n", sub.c_str(), std::strerror(errno));
            std::exit(2);
        }
    }
}

void
on_signal(int sig)
{
    cleanup();
    signal(sig, SIG_DFL);
    raise(sig);
}

void
seed_file(const std::string &path, std::size_t bytes, std::uint32_t seed)
{
    std::mt19937                            gen(seed);
    std::uniform_int_distribution<uint16_t> dist(0, 255);
    std::ofstream                           out(path, std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "tmpdriver: cannot open %s for write: %s\n", path.c_str(), std::strerror(errno));
        std::exit(2);
    }
    constexpr std::size_t      kChunk = 4096;
    std::vector<unsigned char> buf(kChunk);
    std::size_t                written = 0;
    while (written < bytes) {
        std::size_t n = std::min(kChunk, bytes - written);
        for (std::size_t i = 0; i < n; ++i)
            buf[i] = static_cast<unsigned char>(dist(gen));
        out.write(reinterpret_cast<char *>(buf.data()), static_cast<std::streamsize>(n));
        written += n;
    }
}

std::string
substitute(std::string s, const std::string &tmpdir, const std::string &input)
{
    auto replace = [&](const std::string &token, const std::string &value) {
        for (std::size_t pos = 0; (pos = s.find(token, pos)) != std::string::npos; pos += value.size()) {
            s.replace(pos, token.size(), value);
        }
    };
    replace("{TMPDIR}", tmpdir);
    if (!input.empty())
        replace("{INPUT}", input);
    return s;
}

[[noreturn]] void
usage(const char *prog)
{
    std::fprintf(stderr,
                 "Usage: %s --base <dir> [--seed-input <bytes>] [--seed <u32>] "
                 "-- <program> [args...]\n",
                 prog);
    std::exit(2);
}

} // namespace

int
main(int argc, char **argv)
{
    std::string   base;
    std::size_t   seed_bytes = 0;
    std::uint32_t seed       = 97;
    int           i          = 1;

    for (; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--base" && i + 1 < argc) {
            base = argv[++i];
        }
        else if (a == "--seed-input" && i + 1 < argc) {
            seed_bytes = std::stoull(argv[++i]);
        }
        else if (a == "--seed" && i + 1 < argc) {
            seed = static_cast<std::uint32_t>(std::stoul(argv[++i]));
        }
        else if (a == "--") {
            ++i;
            break;
        }
        else {
            usage(argv[0]);
        }
    }
    if (base.empty() || i >= argc)
        usage(argv[0]);

    mkdir_p(base); // ok if exists
    std::string tmpl = base + "/hipfile_ex.XXXXXX";
    if (mkdtemp(tmpl.data()) == nullptr) {
        std::fprintf(stderr, "tmpdriver: mkdtemp(%s) failed: %s\n", tmpl.c_str(), std::strerror(errno));
        return 2;
    }
    g_scratch = tmpl;
    std::atexit(cleanup);
    for (int s : {SIGINT, SIGTERM, SIGHUP})
        signal(s, on_signal);

    std::string input_path;
    if (seed_bytes > 0) {
        input_path = g_scratch + "/input.bin";
        seed_file(input_path, seed_bytes, seed);
    }

    std::vector<std::string> child_args;
    child_args.reserve(argc - i);
    for (; i < argc; ++i)
        child_args.push_back(substitute(argv[i], g_scratch, input_path));

    std::vector<char *> cargs;
    cargs.reserve(child_args.size() + 1);
    for (auto &s : child_args)
        cargs.push_back(s.data());
    cargs.push_back(nullptr);

    g_child = fork();
    if (g_child < 0) {
        std::fprintf(stderr, "tmpdriver: fork failed: %s\n", std::strerror(errno));
        return 2;
    }
    if (g_child == 0) {
        if (chdir(g_scratch.c_str()) != 0) {
            std::fprintf(stderr, "tmpdriver: chdir(%s) failed: %s\n", g_scratch.c_str(),
                         std::strerror(errno));
            _exit(2);
        }
        execvp(cargs[0], cargs.data());
        std::fprintf(stderr, "tmpdriver: execvp(%s) failed: %s\n", cargs[0], std::strerror(errno));
        _exit(127);
    }

    int status = 0;
    if (waitpid(g_child, &status, 0) < 0) {
        std::fprintf(stderr, "tmpdriver: waitpid failed: %s\n", std::strerror(errno));
        return 2;
    }
    g_child = -1;
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    if (WIFSIGNALED(status))
        return 128 + WTERMSIG(status);
    return 2;
}
