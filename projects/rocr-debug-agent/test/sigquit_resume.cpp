/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal with the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

    - Redistributions of source code must retain the above copyright notice,
      this list of conditions and the following disclaimers.
    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimers in
      the documentation and/or other materials provided with the distribution.
    - Neither the names of Advanced Micro Devices, Inc,
      nor the names of its contributors may be used to endorse or promote
      products derived from this Software without specific prior written
      permission.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
   OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
   ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS WITH THE SOFTWARE.  */

#include "util.h"

#include <hip/hip_runtime.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

/* Spins until the host releases it.  */
__global__ void
sigquit_resume_kern (int *fence, const int *release)
{
  atomicAdd_system (fence, 1);
  while (__hip_atomic_load (release, __ATOMIC_RELAXED,
                            __HIP_MEMORY_SCOPE_SYSTEM)
         == 0)
    __builtin_amdgcn_s_sleep (16);
}

static void
handle_alarm (int sig)
{
  fprintf (stderr, "Timeout reached. Exiting.\n");
  exit (EXIT_FAILURE);
}

/* The test driver sends SIGQUIT once the kernel runs, then writes a line to
   stdin after the agent printed the waves.  The kernel can only finish if the
   agent resumed the waves it stopped to print them.  */
void
SigquitResumeTest ()
{
  signal (SIGALRM, handle_alarm);
  alarm (60);

  int *fence, *release;
  hipError_t err = hipHostMalloc (&fence, sizeof (int), hipHostMallocCoherent);
  TEST_ASSERT (err == hipSuccess, "alloc fence");
  err = hipHostMalloc (&release, sizeof (int), hipHostMallocCoherent);
  TEST_ASSERT (err == hipSuccess, "alloc release");
  __atomic_store_n (fence, 0, __ATOMIC_RELAXED);
  __atomic_store_n (release, 0, __ATOMIC_RELAXED);

  sigquit_resume_kern<<<1, 1>>> (fence, release);
  while (__atomic_load_n (fence, __ATOMIC_RELAXED) == 0)
    ;
  printf ("Kernel started\n");
  fflush (stdout);

  char line[16];
  TEST_ASSERT (fgets (line, sizeof (line), stdin) != nullptr, "release line");
  __atomic_store_n (release, 1, __ATOMIC_RELAXED);

  err = hipDeviceSynchronize ();
  alarm (0);
  TEST_ASSERT (err == hipSuccess, "kernel error");
  printf ("Kernel finished\n");
  fflush (stdout);

  err = hipHostFree (fence);
  TEST_ASSERT (err == hipSuccess, "free fence");
  err = hipHostFree (release);
  TEST_ASSERT (err == hipSuccess, "free release");
}
