/*************************************************************************
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cerrno>
#include <cstddef>
#include <sys/types.h>
#include <unistd.h>

namespace RcclUnitTesting
{
  namespace detail
  {
    /**
     * @brief Reads exactly 'count' bytes from a file descriptor, handling partial
     * reads and signal interruptions (EINTR).
     * @return 'count' on success, or -1 on error / premature EOF.
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
          return -1;                    // EOF: Pipe closed prematurely
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
  }
}
