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

#include "spawn_idle_child.h"

#if defined(__linux__)
#  include <signal.h>
#  include <sys/wait.h>
#  include <unistd.h>
#endif

namespace amd::dbgapi::test
{

#if defined(__linux__)

idle_child_t
spawn_idle_child ()
{
  ::pid_t pid = ::fork ();
  if (pid < 0)
    return idle_child_t{}; /* spawn failed */

  if (pid == 0)
    {
      /* Child: exec a long sleep so we replace the address space and
         leave behind no fork-of-gtest hazards.  Use execlp so PATH is
         consulted — /bin/sleep is missing on minimal containers, some
         BusyBox layouts, and non-FHS systems.  Fall back to /bin/sleep
         only if the PATH lookup fails.  If both fail, _exit (not exit)
         so we don't run parent atexit handlers.  */
      ::execlp ("sleep", "sleep", "9999", nullptr);
      ::execl ("/bin/sleep", "sleep", "9999", nullptr);
      ::_exit (127);
    }

  return idle_child_t{ pid };
}

void
idle_child_t::kill_and_wait ()
{
  if (m_pid <= 0)
    return;
  ::kill (m_pid, SIGKILL);
  int status = 0;
  ::waitpid (m_pid, &status, 0);
  m_pid = -1;
}

#else

idle_child_t
spawn_idle_child ()
{
  return idle_child_t{};
}

void
idle_child_t::kill_and_wait ()
{
  m_pid = -1;
}

#endif

} /* namespace amd::dbgapi::test */
