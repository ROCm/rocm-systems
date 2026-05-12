/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "sys.h"

#include <cerrno>
#include <fcntl.h>
#include <stdlib.h>
#include <sys/eventfd.h>
#include <sys/mman.h>
#include <sys/stat.h> // IWYU pragma: keep
#include <sys/syscall.h>
#include <sys/types.h>
#include <syslog.h>
#include <system_error>
#include <unistd.h>

namespace hipFile {

template <typename L, typename R>
static inline R
throwOn(const L throw_value, R value)
{
    if (throw_value == value) {
        throw std::system_error(errno, std::generic_category());
    }
    return value;
}

int
Sys::open(const char *pathname, int flags, mode_t mode) const
{
    if (flags & O_CREAT) {
        return throwOn(-1, ::open(pathname, flags, mode));
    }
    else {
        return throwOn(-1, ::open(pathname, flags));
    }
}

void
Sys::close(int fd) const
{
    throwOn(-1, ::close(fd));
}

void
Sys::fdatasync(int fd) const
{
    throwOn(-1, ::fdatasync(fd));
}

void *
Sys::mmap(void *addr, size_t length, int prot, int flags, int fd, off_t offset) const
{
    return throwOn(reinterpret_cast<void *>(-1), ::mmap(addr, length, prot, flags, fd, offset));
}

void
Sys::munmap(void *addr, size_t length) const
{
    throwOn(-1, ::munmap(addr, length));
}

ssize_t
Sys::pread(int fd, void *buf, size_t count, off_t offset) const
{
    return throwOn(-1, ::pread(fd, buf, count, offset));
}

ssize_t
Sys::pwrite(int fd, void *buf, size_t count, off_t offset) const
{
    return throwOn(-1, ::pwrite(fd, buf, count, offset));
}

ssize_t
Sys::readlink(const char *pathname, char *buf, size_t bufsize) const
{
    return throwOn(-1, ::readlink(pathname, buf, bufsize));
}

void
Sys::syslog(int priority, const char *msg) const
{
    ::syslog(priority, "%s", msg);
}

struct stat
Sys::fstat(int fd) const
{
    struct stat statbuf;

    throwOn(-1, ::fstat(fd, &statbuf));

    return statbuf;
}

int
Sys::fcntl(int fd, int op, uintptr_t arg) const
{
    return throwOn(-1, ::fcntl(fd, op, arg));
}

void
Sys::ftruncate(int fd, off_t offset) const
{
    throwOn(-1, ::ftruncate(fd, offset));
}

struct statx
Sys::statx(int dirfd, const char *pathname, int flags, unsigned int mask) const
{
    struct statx statxbuf;
    throwOn(-1, ::statx(dirfd, pathname, flags, mask, &statxbuf));
    return statxbuf;
}

char *
Sys::getenv(const char *name) const noexcept
{
    return ::getenv(name);
}

int
Sys::memfd_create(const char *name, unsigned int flags) const
{
    return throwOn(-1, ::memfd_create(name, flags));
}

int
Sys::eventfd(unsigned int initval, int flags) const
{
    return throwOn(-1, ::eventfd(initval, flags));
}

int
Sys::pidfd_open(pid_t pid, unsigned int flags) const
{
    return throwOn(-1, static_cast<int>(::syscall(SYS_pidfd_open, pid, flags)));
}

}
