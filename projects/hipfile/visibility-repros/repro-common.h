/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

// Boilerplate shared by the reproducers in this directory. Nothing in this
// header is part of what fails: it parses the command line, stops on errors,
// creates and registers temporary files, and counts wrong words.

#pragma once

#include <hip/hip_runtime.h>
#include <hipfile.h>

#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <unistd.h>
#include <vector>

namespace repro {

constexpr size_t KiB{1024};
constexpr size_t MiB{1024 * KiB};

/// @brief Prints a message and stops the program
[[noreturn]] inline void
die(const std::string &message)
{
    fprintf(stderr, "error: %s\n", message.c_str());
    exit(2);
}

/// @brief Stops the program unless a HIP call succeeded
#define HIP_OK(call)                                                                                         \
    do {                                                                                                     \
        if ((call) != hipSuccess) {                                                                          \
            repro::die(#call " failed");                                                                     \
        }                                                                                                    \
    } while (false)

/// @brief The command line: REPRODUCER DIRECTORY [FIX]
struct Args {
    /// @brief Directory for the temporary file; fastpath needs an AIS-capable file system
    std::string directory;
    /// @brief Name of the fix to apply, or "none"
    std::string fix{"none"};

    /// @brief Parses the command line, and prints the usage and stops if it is wrong
    ///
    /// @param fixes The fix names the reproducer knows
    Args(int argc, char **argv, const std::vector<std::string> &fixes)
    {
        bool known{argc == 2};
        if (argc == 3) {
            fix   = argv[2];
            known = fix == "none";
            for (const std::string &name : fixes) {
                known = known || fix == name;
            }
        }
        if (!known) {
            fprintf(stderr, "usage: %s DIRECTORY [FIX]\n  FIX is one of: none", argv[0]);
            for (const std::string &name : fixes) {
                fprintf(stderr, ", %s", name.c_str());
            }
            fprintf(stderr, "\n");
            exit(2);
        }
        directory = argv[1];
    }
};

/// @brief Makes every hipFileRead and hipFileWrite in the process a fastpath
/// transfer (NVMe <-> VRAM, peer to peer)
///
/// hipFile then fails a transfer it would otherwise do with the fallback
/// backend. Call it before the first hipFile call.
inline void
fastpathOnly()
{
    setenv("HIPFILE_ALLOW_COMPAT_MODE", "false", 1);
}

/// @brief Makes every hipFileRead and hipFileWrite in the process a fallback
/// transfer (pread or pwrite on a host buffer, and hipMemcpy)
///
/// Call it before the first hipFile call.
inline void
fallbackOnly()
{
    setenv("HIPFILE_FORCE_COMPAT_MODE", "true", 1);
}

/// @brief A temporary file registered with hipFile
struct File {
    /// @brief The file hipFile transfers to and from
    int fd;
    /// @brief The same file, opened for direct I/O where the file system
    /// supports it, so that the reproducer's own reads come from storage
    int direct_fd;
    /// @brief The file's hipFile handle
    hipFileHandle_t handle;
};

/// @brief Creates a temporary file of zeros and registers it with hipFile
///
/// @param directory Directory to create the file in; it is unlinked at once
/// @param size Size of the file, in bytes
inline File
createFile(const std::string &directory, size_t size)
{
    File        file{};
    std::string path{directory + "/visibility-repro.XXXXXX"};
    file.fd = mkstemp(path.data());
    if (file.fd == -1) {
        die("cannot create " + path + ": " + strerror(errno));
    }
    unlink(path.c_str());

    // Allocate the file's blocks, so that no hipFileWrite extends the file
    const std::vector<char> zeros(size);
    if (pwrite(file.fd, zeros.data(), size, 0) != static_cast<ssize_t>(size) || fdatasync(file.fd) != 0) {
        die("cannot write the temporary file");
    }

    const std::string proc_path{"/proc/self/fd/" + std::to_string(file.fd)};
    file.direct_fd = open(proc_path.c_str(), O_RDWR | O_DIRECT);
    if (file.direct_fd == -1) {
        file.direct_fd = open(proc_path.c_str(), O_RDWR);
    }
    if (file.direct_fd == -1) {
        die("cannot reopen the temporary file");
    }

    hipFileDescr_t descr{};
    descr.type      = hipFileHandleTypeOpaqueFD;
    descr.handle.fd = file.fd;
    if (hipFileHandleRegister(&file.handle, &descr).err != hipFileSuccess) {
        die("hipFileHandleRegister failed");
    }
    return file;
}

/// @brief Allocates host memory aligned for direct I/O
///
/// @param size Size in bytes, a multiple of 4 KiB
inline uint32_t *
hostWords(size_t size)
{
    void *words{aligned_alloc(4 * KiB, size)};
    if (!words) {
        die("out of host memory");
    }
    return static_cast<uint32_t *>(words);
}

/// @brief Stores one value in every 32-bit word of host memory
inline void
fill(uint32_t *words, size_t size, uint32_t value)
{
    for (size_t i{0}; i < size / sizeof(uint32_t); i++) {
        words[i] = value;
    }
}

/// @brief Writes host memory to the start of a file and makes sure it is on storage
inline void
writeToStorage(const File &file, const uint32_t *words, size_t size)
{
    if (pwrite(file.direct_fd, words, size, 0) != static_cast<ssize_t>(size) ||
        fdatasync(file.direct_fd) != 0) {
        die("cannot write the temporary file");
    }
}

/// @brief Reads the start of a file back from storage into host memory
inline void
readFromStorage(const File &file, uint32_t *words, size_t size)
{
    if (pread(file.direct_fd, words, size, 0) != static_cast<ssize_t>(size)) {
        die("cannot read the temporary file back");
    }
}

/// @brief Stops the program unless a hipFileRead or hipFileWrite transferred everything
inline void
transferred(ssize_t result, size_t size, const char *call)
{
    if (result != static_cast<ssize_t>(size)) {
        die(std::string{call} + " returned " + std::to_string(result) +
            (result == -1 ? std::string{": "} + strerror(errno) : std::string{}) +
            "; is the directory on a file system that supports this backend?");
    }
}

/// @brief Writes 1 to the HDP's memory flush register, which makes the GPU's
/// host data path write out what it holds and drop what it has cached
///
/// The register is only ever written: reading it back has caused PCIe errors
/// on some systems. hipDeviceFlushGPUDirectRDMAWrites does the same write in
/// the HIP versions that have it.
inline void
flushHdp()
{
    static volatile unsigned int *flush_register{nullptr};
    if (!flush_register) {
        int             device{};
        hipDeviceProp_t properties{};
        HIP_OK(hipGetDevice(&device));
        HIP_OK(hipGetDeviceProperties(&properties, device));
        flush_register = properties.hdpMemFlushCntl;
        if (!flush_register) {
            die("HIP does not report the HDP flush register");
        }
    }
    *flush_register = 1;
}

/// @brief Counts the iterations whose data was wrong
struct Tally {
    /// @brief Iterations checked
    size_t iterations{0};
    /// @brief Iterations with at least one wrong word
    size_t failed{0};

    /// @brief Checks that every 32-bit word holds the value it should
    ///
    /// @param words The data
    /// @param size Size of the data, in bytes
    /// @param expected The value every word should hold
    void check(const volatile uint32_t *words, size_t size, uint32_t expected)
    {
        const size_t count{size / sizeof(uint32_t)};
        size_t       wrong{0};
        uint32_t     first_wrong{0};
        for (size_t i{0}; i < count; i++) {
            const uint32_t value{words[i]};
            if (value != expected && wrong++ == 0) {
                first_wrong = value;
            }
        }
        record(wrong, count, first_wrong, expected);
    }

    /// @brief Records an iteration whose wrong words were counted elsewhere
    void record(size_t wrong, size_t count, uint32_t first_wrong, uint32_t expected)
    {
        iterations++;
        if (wrong == 0) {
            return;
        }
        if (failed++ < 3) {
            printf("  iteration %zu: %zu of %zu words are wrong; the first holds %#x, not %#x\n", iterations,
                   wrong, count, first_wrong, expected);
        }
    }

    /// @brief Prints the result
    ///
    /// @return The program's exit status: 0 if every iteration was right
    int finish(const Args &args) const
    {
        printf("%s (fix: %s): %zu of %zu iterations had wrong data\n", failed ? "FAIL" : "PASS",
               args.fix.c_str(), failed, iterations);
        return failed ? 1 : 0;
    }
};

}
