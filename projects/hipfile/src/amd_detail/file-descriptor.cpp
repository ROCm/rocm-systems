/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "context.h"
#include "file-descriptor.h"
#include "sys.h"

using namespace hipFile;

FileDescriptor
FileDescriptor::make_managed(int fd)
{
    return FileDescriptor(fd, true);
}

FileDescriptor
FileDescriptor::make_unmanaged(int fd)
{
    return FileDescriptor(fd, false);
}

FileDescriptor::FileDescriptor() : m_managed(false), m_fd(-1)
{
}

FileDescriptor::FileDescriptor(int fd, bool managed) : m_managed(managed), m_fd(fd)
{
}

FileDescriptor::~FileDescriptor()
{
    if (m_managed && m_fd != -1) {
        Context<Sys>::get()->close(m_fd);
    }
    m_managed = false;
    m_fd      = -1;
}

FileDescriptor::FileDescriptor(FileDescriptor &&other) noexcept : m_managed(other.m_managed), m_fd(other.m_fd)
{
    other.m_fd      = -1;
    other.m_managed = false;
}

FileDescriptor &
FileDescriptor::operator=(FileDescriptor &&other) noexcept
{
    if (this != &other) {
        if (m_managed && m_fd != -1) {
            Context<Sys>::get()->close(m_fd);
        }
        m_fd            = other.m_fd;
        m_managed       = other.m_managed;
        other.m_fd      = -1;
        other.m_managed = false;
    }
    return *this;
}

int
FileDescriptor::get() const
{
    return m_fd;
}
