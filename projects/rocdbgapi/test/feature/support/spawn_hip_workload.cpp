/* Copyright (c) 2026 Advanced Micro Devices, Inc.

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "spawn_hip_workload.h"

#include <string>

#if defined(__linux__)
#  include <libgen.h>
#  include <linux/limits.h>
#  include <signal.h>
#  include <sys/stat.h>
#  include <sys/wait.h>
#  include <unistd.h>

#  include <cstdio>
#  include <cstdlib>
#  include <cstring>
#endif

namespace amd::dbgapi::test
{

namespace
{

/* Workload binary name as built by test/feature/CMakeLists.txt.
   Both layouts the suite is expected to run from — the build tree
   (build/test/feature/) and the install tree (tests/rocdbgapi/) —
   colocate the workload with every feature test binary, so a single
   lookup strategy (look next to the running executable) works for
   both.  */
constexpr const char hip_idle_workload_name[] = "dbgapi_test_idle_kernel";

#if defined(__linux__)

/* Resolve the workload path by reading /proc/self/exe and appending
   the workload basename.  Returns an empty string if anything fails
   or the resulting path is not an executable regular file — the
   caller treats that as "no workload available" and the test skips.
   Cached in a function-local static so the readlink/stat cost is
   paid once per process.  */
const std::string &
hip_idle_workload_path ()
{
  static const std::string path = [] () -> std::string {
    char self[PATH_MAX];
    ssize_t n = ::readlink ("/proc/self/exe", self, sizeof (self) - 1);
    if (n <= 0)
      return {};
    self[n] = '\0';

    /* POSIX dirname() may modify its input; operate on a copy.  */
    char buf[PATH_MAX];
    std::strncpy (buf, self, sizeof (buf));
    buf[sizeof (buf) - 1] = '\0';
    const char *dir = ::dirname (buf);

    std::string candidate
        = std::string (dir) + "/" + hip_idle_workload_name;

    struct stat st;
    if (::stat (candidate.c_str (), &st) != 0)
      return {};
    if (!S_ISREG (st.st_mode))
      return {};
    if (::access (candidate.c_str (), X_OK) != 0)
      return {};

    return candidate;
  }();
  return path;
}

#else

const std::string &
hip_idle_workload_path ()
{
  static const std::string empty;
  return empty;
}

#endif

} /* namespace */

bool
hip_workload_available ()
{
  return !hip_idle_workload_path ().empty ();
}

#if defined(__linux__)

idle_child_t
spawn_hip_workload ()
{
  const std::string &path = hip_idle_workload_path ();
  if (path.empty ())
    return idle_child_t{};

  int pipefd[2] = { -1, -1 };
  if (::pipe (pipefd) < 0)
    return idle_child_t{};

  ::pid_t pid = ::fork ();
  if (pid < 0)
    {
      ::close (pipefd[0]);
      ::close (pipefd[1]);
      return idle_child_t{};
    }

  if (pid == 0)
    {
      /* Child: rewire stdout into the pipe so the parent can read
         "READY\n", then exec.  */
      ::close (pipefd[0]);
      if (::dup2 (pipefd[1], STDOUT_FILENO) < 0)
        ::_exit (126);
      ::close (pipefd[1]);
      /* HSA_ENABLE_DEBUG=1 lets dbgapi report full wave info for
         pre-attach dispatches; harmless if unused.  */
      ::setenv ("HSA_ENABLE_DEBUG", "1", 1);
      ::execl (path.c_str (), hip_idle_workload_name,
               static_cast<char *> (nullptr));
      ::_exit (127);
    }

  /* Parent: read until we see READY, the pipe closes, or the child
     dies.  Bounded read so a runaway workload can't hang us; if we
     don't see READY in ~30 KiB / a few seconds we kill the child and
     return invalid so the test can GTEST_SKIP.  */
  ::close (pipefd[1]);

  std::string buf;
  char chunk[256];
  bool ready = false;
  for (int i = 0; i < 120; ++i)
    {
      ssize_t n = ::read (pipefd[0], chunk, sizeof (chunk));
      if (n <= 0)
        break;
      buf.append (chunk, static_cast<size_t> (n));
      if (buf.find ("READY\n") != std::string::npos)
        {
          ready = true;
          break;
        }
      if (buf.size () > 32768)
        break;
    }
  ::close (pipefd[0]);

  if (!ready)
    {
      ::kill (pid, SIGKILL);
      int status = 0;
      ::waitpid (pid, &status, 0);
      return idle_child_t{};
    }

  return idle_child_t{ pid };
}

#else

idle_child_t
spawn_hip_workload ()
{
  return idle_child_t{};
}

#endif

} /* namespace amd::dbgapi::test */
