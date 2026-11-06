/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef PIPE_UTILS_HPP
#define PIPE_UTILS_HPP

#include <cerrno>
#include <cstddef>
#include <sys/types.h>
#include <type_traits>
#include <unistd.h>
#include <vector>

namespace RcclUnitTesting
{
  /**
   * @brief Reads exactly 'count' bytes from a file descriptor, handling partial
   * reads and signal interruptions (EINTR).
   * @return 'count' on success, -1 on read error (errno is set), or the
   *         number of bytes read before EOF (0 if the writer closed first).
   */
  inline ssize_t safe_pipe_read(int fd, void* buf, std::size_t count)
  {
    char* ptr = static_cast<char*>(buf);
    std::size_t bytesLeft = count;

    while (bytesLeft > 0)
    {
      ssize_t const bytesRead = read(fd, ptr, bytesLeft);
      if (bytesRead < 0)
      {
        if (errno == EINTR) continue; // Interrupted by OS signal, retry
        return -1;                    // Read error
      }
      if (bytesRead == 0)
      {
        return static_cast<ssize_t>(count - bytesLeft); // EOF: pipe closed prematurely
      }
      ptr += bytesRead;
      bytesLeft -= bytesRead;
    }
    return static_cast<ssize_t>(count); // Successfully read all requested bytes
  }

  /**
   * @brief Writes exactly 'count' bytes to a file descriptor, handling partial
   * writes and signal interruptions (EINTR).
   * @return 'count' on success, or -1 on error.
   */
  inline ssize_t safe_pipe_write(int fd, const void* buf, std::size_t count)
  {
    const char* ptr = static_cast<const char*>(buf);
    std::size_t bytesLeft = count;

    while (bytesLeft > 0)
    {
      ssize_t const bytesWritten = write(fd, ptr, bytesLeft);
      if (bytesWritten < 0)
      {
        if (errno == EINTR) continue; // Interrupted by OS signal, retry
        return -1;                    // Write error
      }
      if (bytesWritten == 0)
      {
        return -1;                    // No progress: avoid an infinite loop
      }
      ptr += bytesWritten;
      bytesLeft -= bytesWritten;
    }
    return static_cast<ssize_t>(count); // Successfully wrote all requested bytes
  }

  /**
   * @brief Sends a std::vector as its int element count followed by its elements.
   * @return true if the count and every element were written.
   */
  template <typename T>
  bool pipe_write_vec(int fd, std::vector<T> const& vec)
  {
    static_assert(std::is_trivially_copyable<T>::value, "elements are sent as raw bytes");
    int const count = static_cast<int>(vec.size());
    if (safe_pipe_write(fd, &count, sizeof(count)) != static_cast<ssize_t>(sizeof(count))) return false;
    ssize_t const numBytes = static_cast<ssize_t>(vec.size() * sizeof(T));
    return count == 0 || safe_pipe_write(fd, vec.data(), numBytes) == numBytes;
  }

  /**
   * @brief Receives a std::vector sent by pipe_write_vec.
   * @return true if a non-negative count and that many elements were read.
   */
  template <typename T>
  bool pipe_read_vec(int fd, std::vector<T>& vec)
  {
    static_assert(std::is_trivially_copyable<T>::value, "elements are sent as raw bytes");
    int count = 0;
    if (safe_pipe_read(fd, &count, sizeof(count)) != static_cast<ssize_t>(sizeof(count)) || count < 0)
      return false;
    vec.resize(count);
    ssize_t const numBytes = static_cast<ssize_t>(vec.size() * sizeof(T));
    return count == 0 || safe_pipe_read(fd, vec.data(), numBytes) == numBytes;
  }
}

#endif // PIPE_UTILS_HPP
