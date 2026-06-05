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

/* spawn_idle_child: portable helper to launch a child process that
   exists long enough for a feature test to attach to it, without
   doing any GPU work.

   The child is a fresh process (via fork+exec on Linux), so it does
   not inherit the gtest runtime state — important because some
   gtest internals are unsafe to fork-without-exec from.  On Linux
   we exec /bin/sleep with a large duration.  On platforms without
   a usable equivalent the spawn fails (pid == -1) and the test
   should GTEST_SKIP.

   The returned handle owns the child: its destructor SIGKILLs and
   reaps so a failing test never leaks the child.  Tests that want
   to confirm post-mortem behavior (attach to a dead PID) can call
   kill_and_wait() explicitly and then attach.  */

#ifndef DBGAPI_FEATURE_SUPPORT_SPAWN_IDLE_CHILD_H
#define DBGAPI_FEATURE_SUPPORT_SPAWN_IDLE_CHILD_H

#include "amd-dbgapi.h"

namespace amd::dbgapi::test
{

class idle_child_t
{
public:
  idle_child_t () = default;
  explicit idle_child_t (amd_dbgapi_os_process_id_t pid) : m_pid (pid) {}

  idle_child_t (const idle_child_t &) = delete;
  idle_child_t &operator= (const idle_child_t &) = delete;

  idle_child_t (idle_child_t &&other) noexcept : m_pid (other.m_pid)
  {
    other.m_pid = -1;
  }
  idle_child_t &operator= (idle_child_t &&other) noexcept
  {
    if (this != &other)
      {
        kill_and_wait ();
        m_pid = other.m_pid;
        other.m_pid = -1;
      }
    return *this;
  }

  ~idle_child_t () { kill_and_wait (); }

  amd_dbgapi_os_process_id_t pid () const { return m_pid; }
  bool valid () const { return m_pid > 0; }

  /* Send SIGKILL (or platform equivalent) and waitpid().  Safe to
     call multiple times; second call is a no-op.  */
  void kill_and_wait ();

private:
  amd_dbgapi_os_process_id_t m_pid = -1;
};

/* Spawn an idle child that does not touch the GPU.  Returns an
   invalid handle (pid <= 0) on platforms where this isn't supported
   or if the spawn itself fails.  */
idle_child_t spawn_idle_child ();

} /* namespace amd::dbgapi::test */

#endif /* DBGAPI_FEATURE_SUPPORT_SPAWN_IDLE_CHILD_H */
