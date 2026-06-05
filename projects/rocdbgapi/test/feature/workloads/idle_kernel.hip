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

/* Idle HIP workload child for feature tests.

   Launches a single-wave kernel that spins on a device-resident flag
   the host never sets, then prints "READY\n" on stdout so the parent
   knows the kernel is dispatched, and sleeps until SIGKILLed.

   We deliberately keep the launch tiny (one block, one wave) so the
   tests have a single deterministic wave to inspect.  Subsequent
   commits can add richer workloads (multi-grid, multi-stream) as
   needed.

   The READY handshake matters: an attach that races the kernel
   launch may see queue_count == 0 and report no waves.  Tests should
   wait for that line on stdout before calling
   amd_dbgapi_process_attach.  */

#include <hip/hip_runtime.h>

#include <sys/prctl.h>
#include <sys/ptrace.h>

#include <cstdio>
#include <cstdlib>
#include <unistd.h>

__global__ void
idle_kernel (volatile int *flag)
{
  while (*flag == 0)
    {
      /* spin */
    }
}

static void
die (const char *what, hipError_t err)
{
  std::fprintf (stderr, "idle_kernel: %s failed: %s\n", what,
                hipGetErrorString (err));
  std::exit (1);
}

int
main ()
{
  /* Tell the kernel our parent is our tracer.  KFD DBG_TRAP_ENABLE
     calls ptrace_may_access on the target; this guarantees that
     check passes for our parent regardless of YAMA or dumpable
     state.  Called after exec so there's no exec-time SIGTRAP, and
     we never receive any other signals before SIGKILL.  */
  ::ptrace (PTRACE_TRACEME, 0, nullptr, nullptr);

  hipError_t err;

  int *d_flag = nullptr;
  if ((err = hipMalloc (&d_flag, sizeof (int))) != hipSuccess)
    die ("hipMalloc", err);
  if ((err = hipMemset (d_flag, 0, sizeof (int))) != hipSuccess)
    die ("hipMemset", err);

  hipLaunchKernelGGL (idle_kernel, dim3 (1), dim3 (64), 0, 0, d_flag);
  if ((err = hipGetLastError ()) != hipSuccess)
    die ("hipLaunchKernelGGL", err);

  /* Defensive: ensure dumpable=1 so a same-uid parent can satisfy
     ptrace_may_access for KFD DBG_TRAP_ENABLE.  HSA does not appear
     to flip this on this host but it's cheap insurance.  */
  ::prctl (PR_SET_DUMPABLE, 1, 0, 0, 0);

  /* Make sure the dispatch has actually been pushed onto the queue
     before signalling readiness.  hipStreamQuery on the default
     stream returns hipErrorNotReady while the kernel is in flight —
     which is exactly the state we want when the parent attaches.  */
  std::printf ("READY\n");
  std::fflush (stdout);

  /* Block until the parent SIGKILLs us.  pause() unblocks on any
     signal; loop so a stray SIGCHLD or SIGWINCH doesn't make us
     exit early.  */
  for (;;)
    ::pause ();

  return 0;
}
