/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstdint>
#include <linux/stat.h>
#include <sys/stat.h>
#include <sys/types.h>

/* sys wraps system APIs used by hipFile which enables unit tests to mock system calls.
 *
 * The wrapper methods should
 *   - Throw a std::system_error if the wrapped function fails
 *   - Use return values instead of out arguments when possible
 */

namespace hipFile {

struct Sys {
    virtual ~Sys() = default;

    virtual int  open(const char *pathname, int flags, mode_t mode = 0) const;
    virtual void close(int fd) const;

    virtual void fdatasync(int fd) const;

    virtual void *mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) const;
    virtual void  munmap(void *addr, size_t length) const;

    virtual ssize_t pread(int fd, void *buf, size_t count, off_t offset) const;
    virtual ssize_t pwrite(int fd, void *buf, size_t count, off_t offset) const;

    virtual ssize_t readlink(const char *pathname, char *buf, size_t bufsize) const;

    virtual void syslog(int priority, const char *msg) const;

    virtual struct stat  fstat(int fd) const;
    virtual struct statx statx(int dirfd, const char *pathname, int flags, unsigned int mask) const;

    virtual int  fcntl(int fd, int op, uintptr_t arg) const;
    virtual void ftruncate(int fd, off_t offset) const;

    virtual char *getenv(const char *name) const noexcept;

    virtual int memfd_create(const char *name, unsigned int flags) const;
    virtual int eventfd(unsigned int initval, int flags) const;
    virtual int pidfd_open(pid_t pid, unsigned int flags) const;
};

}
