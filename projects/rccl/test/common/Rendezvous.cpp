/*************************************************************************
 * Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

/**
 * @file Rendezvous.cpp
 * @brief Implementation of the init-pipeline READY/GO barrier.
 */

#include "Rendezvous.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <poll.h>
#include <unistd.h>

namespace RcclUnitTesting
{
  std::string Rendezvous::Dir()
  {
    const char* d = getenv("RCCL_TEST_RENDEZVOUS_DIR");
    return (d && d[0] != '\0') ? std::string(d) : std::string();
  }

  bool Rendezvous::Enabled()
  {
    return !Dir().empty();
  }

  bool Rendezvous::PublishReady()
  {
    if (!Enabled()) return true;

    std::string const dir = Dir();
    std::string const tmp = dir + "/ready.tmp";
    std::string const dst = dir + "/ready";

    // Write to a temp file then atomically rename into place (same directory) so
    // the runner never observes a torn/partial token.
    FILE* f = fopen(tmp.c_str(), "w");
    if (!f)
    {
      fprintf(stderr, "[RCCL_TEST_RENDEZVOUS] pid %d: failed to create %s: %s\n",
              (int)getpid(), tmp.c_str(), strerror(errno));
      fflush(stderr);
      return false;
    }
    fputs("ready\n", f);
    fflush(f);
    fclose(f);

    if (rename(tmp.c_str(), dst.c_str()) != 0)
    {
      fprintf(stderr, "[RCCL_TEST_RENDEZVOUS] pid %d: failed to rename %s -> %s: %s\n",
              (int)getpid(), tmp.c_str(), dst.c_str(), strerror(errno));
      fflush(stderr);
      return false;
    }

    fprintf(stderr, "[RCCL_TEST_RENDEZVOUS] pid %d published READY (%s)\n", (int)getpid(), dst.c_str());
    fflush(stderr);
    return true;
  }

  void Rendezvous::WaitForGo()
  {
    if (!Enabled()) return;

    std::string const go = Dir() + "/go";

    // Optional runner-liveness fd (inherited pipe read end). EOF/HUP means the
    // runner died, so tear down instead of waiting forever. Best-effort: an
    // absent or unusable fd just falls back to polling, and process-group kill
    // from the runner remains the primary teardown mechanism.
    int liveFd = -1;
    if (const char* fdEnv = getenv("RCCL_TEST_LIVENESS_FD"))
    {
      liveFd = atoi(fdEnv);
    }

    fprintf(stderr, "[RCCL_TEST_RENDEZVOUS] pid %d waiting for GO (%s)\n", (int)getpid(), go.c_str());
    fflush(stderr);

    struct timespec ts;
    ts.tv_sec  = 0;
    ts.tv_nsec = 50 * 1000 * 1000; // 50 ms poll interval

    while (true)
    {
      if (access(go.c_str(), F_OK) == 0)
      {
        fprintf(stderr, "[RCCL_TEST_RENDEZVOUS] pid %d received GO\n", (int)getpid());
        fflush(stderr);
        return;
      }

      if (liveFd >= 0)
      {
        struct pollfd pfd;
        pfd.fd      = liveFd;
        pfd.events  = 0; // only interested in HUP/ERR/NVAL, which poll always reports
        pfd.revents = 0;
        if (poll(&pfd, 1, 0) > 0)
        {
          if (pfd.revents & POLLNVAL)
          {
            // Bad fd: disable the liveness check and keep polling for GO.
            liveFd = -1;
          }
          else if (pfd.revents & (POLLHUP | POLLERR))
          {
            fprintf(stderr,
                    "[RCCL_TEST_RENDEZVOUS] pid %d: runner liveness fd closed; aborting GO wait\n",
                    (int)getpid());
            fflush(stderr);
            _exit(1);
          }
        }
      }

      nanosleep(&ts, nullptr);
    }
  }
}
