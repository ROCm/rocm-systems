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

#include <fstream>
#include <hip/hip_runtime.h>
#include <iostream>
#include <vector>

#define CHECK(cmd)                                                            \
  {                                                                           \
    hipError_t error = cmd;                                                   \
    if (error != hipSuccess)                                                  \
      {                                                                       \
        fprintf (stderr, "error: '%s'(%d) at %s:%d\n",                        \
                 hipGetErrorString (error), error, __FILE__, __LINE__);       \
        exit (EXIT_FAILURE);                                                  \
      }                                                                       \
  }

#ifdef __HIP_DEVICE_COMPILE__

extern "C" __global__ void
kernel_abort (volatile int *v)
{
  /* Wait for the host to clober the code object before calling abort.  */
  while (*v == 0)
    __builtin_amdgcn_s_sleep (1);

  abort ();
}

#endif

void
SnapshotCodeObjOnLoadTest ()
{
  /* Copy the content of the module file into a memory buffer.  */
  const char *modpath = "snapshot_objfile_on_load.hipfb";
  std::ifstream mod (modpath, std::ios::binary | std::ios::ate);
  size_t module_size = mod.tellg ();
  mod.seekg (0, std::ios::beg);
  std::vector<char> module_buffer (module_size);
  if (!mod.read (module_buffer.data (), module_size))
    {
      std::cerr << "Failed to load HIP module into memory\n";
      return;
    }
  mod.close ();

  hipModule_t m;
  CHECK (hipModuleLoadData (&m, module_buffer.data ()));

  hipFunction_t f;
  CHECK (hipModuleGetFunction (&f, m, "kernel_abort"));

  /* The guard is used to hold the kernel execution until we have modified the
     memory buffer.  This not used to synchronize any other data, so a
     volatile is enough.  */
  volatile int *guard;
  CHECK (hipHostMalloc (&guard, sizeof (int), hipHostMallocMapped));
  *guard = 0;

  struct
  {
    int *guard;
  } args;
  CHECK (hipHostGetDevicePointer (reinterpret_cast<void **> (&args.guard),
                                  const_cast<int *> (guard), 0));
  size_t args_size = sizeof (args);

  void *config[]
      = { HIP_LAUNCH_PARAM_BUFFER_POINTER, &args, HIP_LAUNCH_PARAM_BUFFER_SIZE,
          &args_size, HIP_LAUNCH_PARAM_END };

  CHECK (hipModuleLaunchKernel (f, 1, 1, 1, 1, 1, 1, 0, 0, nullptr,
                                static_cast<void **> (config)));

  /* Now that the module is submitted to the device, try to be the worst
     possible citizen by unloading the module and scrambling the underlying
     buffer.  */
  CHECK (hipModuleUnload (m));
  std::fill (module_buffer.begin (), module_buffer.end (), 0);
  module_buffer.resize (0);
  module_buffer.shrink_to_fit ();

  /* Notify the kernel that it can go ahead and execute "abort".  */
  *guard = 1;

  CHECK (hipDeviceSynchronize ());
  CHECK (hipFree (const_cast<int *> (guard)));
}
