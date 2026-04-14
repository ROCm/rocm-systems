/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

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
#include <cstdio>

__global__ void
sigquit_kern (int *fence)
{
  atomicAdd_system (fence, 1);
  while (true)
    __builtin_amdgcn_s_sleep (16);
}

void
handle_alarm (int sig)
{
  fprintf (stderr, "Timeout reached. Exiting.\n");
  exit (EXIT_FAILURE);
}

void
SigquitTest ()
{
  signal (SIGALRM, handle_alarm);
  alarm (30);

  hipError_t err;
  int start = 0;
  int *fence;
  err = hipMallocManaged (&fence, sizeof (int));
  TEST_ASSERT (err == hipSuccess, "alloc");

  __atomic_store_n (fence, 0, __ATOMIC_RELAXED);

  sigquit_kern<<<1, 1>>> (fence);

  while (__atomic_load_n (fence, __ATOMIC_RELAXED)  == 0)
	 ;

  printf ("Kernel started\n");
  fflush (stdout);

  err = hipDeviceSynchronize ();

  alarm (0);

  TEST_ASSERT (err == hipSuccess, "kernel error");
}
