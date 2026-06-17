/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"
#include "magic-word.h"

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

constexpr hipFileError_t
HipFileOpError(hipFileOpError_t err)
{
    return {err, hipSuccess};
}

constexpr hipFileError_t
HipFileDriverError(hipError_t err)
{
    return {hipFileHipDriverError, err};
}

inline bool
operator==(const hipFileError_t &lhs, const hipFileError_t &rhs)
{
    return lhs.err == rhs.err && lhs.hip_drv_err == rhs.hip_drv_err;
}

static std::ostream &
operator<<(std::ostream &os, const hipFileError_t &hfe)
{
    return os << "hipFileError_t{ err: " << hfe.err << ", hip_drv_err: " << hfe.hip_drv_err << " }";
}

// Convenience "success" value
inline constexpr hipFileError_t HIPFILE_SUCCESS{hipFileSuccess, hipSuccess};

inline void
rfill(void *buffer, uint64_t len, uint32_t seed = 97)
{
    std::mt19937                            gen(seed);
    std::uniform_int_distribution<uint16_t> dist(0, 255);

    for (uint64_t i = 0; i < len; i++) {
        static_cast<uint8_t *>(buffer)[i] = static_cast<uint8_t>(dist(gen));
    }
}

struct Tmpfile {
    int         fd;
    std::string path;

    Tmpfile() : Tmpfile{"."}
    {
    }

    /*!
     * \warning The constructor is NOT thread-safe and can result in
     *          interleaved calls to `umask()`
     */
    Tmpfile(std::string directory) : path{std::move(directory)}
    {
        // Security analyzers will be sad if you don't set umask 0077
        // before creating temporary files
        mode_t old_umask = umask(S_IRWXG | S_IRWXO);

        path += "/hipFile.XXXXXX";
        if ((fd = mkstemp(path.data())) == -1) {
            int mkstemp_err{errno};
            umask(old_umask);
            throw std::runtime_error("Could not create temporary file " + path + ": " +
                                     std::string(std::strerror(mkstemp_err)));
        }

        umask(old_umask);

#ifdef __HIP_PLATFORM_AMD__
        if (unlink(path.c_str()) == -1) {
            close(fd);
            throw std::runtime_error("Could not unlink temporary file");
        }
#endif
    }

    ~Tmpfile()
    {
#ifdef __HIP_PLATFORM_NVIDIA__
        unlink(path.c_str());
#endif
        close(fd);
    }
};
