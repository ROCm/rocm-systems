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

  ReleaseStatus Rendezvous::WaitForGo(double timeout_sec)
  {
    if (!Enabled()) return RELEASE_GO;

    std::string const go     = Dir() + "/go";
    std::string const cancel = Dir() + "/cancel";

    // Optional runner-liveness fd (inherited pipe read end). EOF/HUP means the
    // runner died. Best-effort: an absent or unusable fd just falls back to
    // polling + the bounded timeout, and process-group kill from the runner
    // remains the primary teardown mechanism.
    int liveFd = -1;
    if (const char* fdEnv = getenv("RCCL_TEST_LIVENESS_FD"))
    {
      liveFd = atoi(fdEnv);
    }

    fprintf(stderr,
            "[RCCL_TEST_RENDEZVOUS] pid %d waiting for GO (%s), timeout=%.1fs\n",
            (int)getpid(), go.c_str(), timeout_sec);
    fflush(stderr);

    struct timespec ts;
    ts.tv_sec  = 0;
    ts.tv_nsec = 50 * 1000 * 1000; // 50 ms poll interval
    struct timespec t0;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    while (true)
    {
      if (access(go.c_str(), F_OK) == 0)
      {
        fprintf(stderr, "[RCCL_TEST_RENDEZVOUS] pid %d received GO\n", (int)getpid());
        fflush(stderr);
        return RELEASE_GO;
      }
      if (access(cancel.c_str(), F_OK) == 0)
      {
        fprintf(stderr, "[RCCL_TEST_RENDEZVOUS] pid %d received CANCEL\n", (int)getpid());
        fflush(stderr);
        return RELEASE_CANCEL;
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
            liveFd = -1;  // bad fd: disable the check, keep polling
          }
          else if (pfd.revents & (POLLHUP | POLLERR))
          {
            fprintf(stderr,
                    "[RCCL_TEST_RENDEZVOUS] pid %d: runner liveness fd closed (LIVENESS_LOST)\n",
                    (int)getpid());
            fflush(stderr);
            return RELEASE_LIVENESS_LOST;
          }
        }
      }

      if (timeout_sec > 0.0)
      {
        struct timespec now;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double elapsed = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
        if (elapsed > timeout_sec)
        {
          fprintf(stderr,
                  "[RCCL_TEST_RENDEZVOUS] pid %d: GO wait exceeded %.1fs (GO_TIMEOUT)\n",
                  (int)getpid(), timeout_sec);
          fflush(stderr);
          return RELEASE_GO_TIMEOUT;
        }
      }

      nanosleep(&ts, nullptr);
    }
  }
}
