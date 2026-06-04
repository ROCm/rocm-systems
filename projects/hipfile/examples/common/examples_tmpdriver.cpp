// Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
//
// SPDX-License-Identifier: MIT
//
// Per-test scratch driver for the basics/ and async/ example CTests.
//
// mkdtemp's a unique scratch dir under --base, optionally seeds an input
// file with deterministic bytes, substitutes {TMPDIR} and {INPUT} tokens
// in the child argv, runs the example, and rm -rf's the dir on the way
// out. Replaces the shared-fixture model whose cleanup raced across CI
// runners sharing AIS_CAPABLE_DIR.
//
// Usage:
//   examples_tmpdriver --base <dir> [--seed-input <bytes>] -- <program> [args...]
//
// Tokens in <args...>:
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

#include <sys/wait.h>
#include <unistd.h>

namespace {

// std::filesystem deliberately avoided: rocky8 (in the hipFile CI matrix)
// ships GCC 8, where libstdc++ requires linking -lstdc++fs. Sticking to
// POSIX keeps the driver buildable across rocky / rocky8 / suse / ubuntu.

std::string g_scratch;

void
cleanup()
{
    if (!g_scratch.empty()) {
        std::string cmd = "rm -rf '" + g_scratch + "'";
        std::system(cmd.c_str());
        g_scratch.clear();
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
seed_file(const std::string &path, std::size_t bytes)
{
    std::mt19937                            gen(97);
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

[[noreturn]] void
usage(const char *prog)
{
    std::fprintf(stderr, "Usage: %s --base <dir> [--seed-input <bytes>] -- <program> [args...]\n", prog);
    std::exit(2);
}

} // namespace

int
main(int argc, char **argv)
{
    std::string base;
    std::size_t seed_bytes = 0;
    int         i          = 1;

    for (; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--base" && i + 1 < argc)
            base = argv[++i];
        else if (a == "--seed-input" && i + 1 < argc)
            seed_bytes = std::stoull(argv[++i]);
        else if (a == "--") {
            ++i;
            break;
        }
        else
            usage(argv[0]);
    }
    if (base.empty() || i >= argc)
        usage(argv[0]);

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
        seed_file(input_path, seed_bytes);
    }

    std::vector<std::string> child_args;
    child_args.reserve(argc - i);
    for (; i < argc; ++i) {
        std::string s = argv[i];
        for (std::size_t p = 0; (p = s.find("{TMPDIR}", p)) != std::string::npos; p += g_scratch.size())
            s.replace(p, 8, g_scratch);
        if (!input_path.empty())
            for (std::size_t p = 0; (p = s.find("{INPUT}", p)) != std::string::npos; p += input_path.size())
                s.replace(p, 7, input_path);
        child_args.push_back(std::move(s));
    }

    std::vector<char *> cargs;
    cargs.reserve(child_args.size() + 1);
    for (auto &s : child_args)
        cargs.push_back(s.data());
    cargs.push_back(nullptr);

    pid_t child = fork();
    if (child < 0) {
        std::fprintf(stderr, "tmpdriver: fork failed: %s\n", std::strerror(errno));
        return 2;
    }
    if (child == 0) {
        execvp(cargs[0], cargs.data());
        std::fprintf(stderr, "tmpdriver: execvp(%s) failed: %s\n", cargs[0], std::strerror(errno));
        _exit(127);
    }

    int status = 0;
    if (waitpid(child, &status, 0) < 0) {
        std::fprintf(stderr, "tmpdriver: waitpid failed: %s\n", std::strerror(errno));
        return 2;
    }
    if (WIFEXITED(status))
        return WEXITSTATUS(status);
    if (WIFSIGNALED(status))
        return 128 + WTERMSIG(status);
    return 2;
}
