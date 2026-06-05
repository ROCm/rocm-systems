// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.

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
