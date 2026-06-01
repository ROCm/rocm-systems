/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

namespace hipFile {

/// @brief Manage access to a file descriptor
class FileDescriptor {
public:
    /// @brief Create an FileDescriptor instance that will close fd on destruction
    /// @param fd A file descriptor
    /// @return A FileDescriptor instance
    static FileDescriptor make_managed(int fd);

    /// @brief Create an FileDescriptor instance that will not close fd on destruction
    /// @param fd A file descriptor
    /// @return A FileDescriptor instance
    static FileDescriptor make_unmanaged(int fd);

    /// @brief Create an empty (fd=-1), unmanaged FileDescriptor
    FileDescriptor();
    ~FileDescriptor();

    FileDescriptor(const FileDescriptor &)            = delete;
    FileDescriptor &operator=(const FileDescriptor &) = delete;
    FileDescriptor(FileDescriptor &&other) noexcept;
    FileDescriptor &operator=(FileDescriptor &&other) noexcept;

    /// @brief Get the file descriptor
    /// @return The file descriptor
    int get() const;

private:
    FileDescriptor(int fd, bool managed);

    bool m_managed;
    int  m_fd;
};
}
