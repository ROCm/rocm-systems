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

/* Use an empty kernel.  */
extern "C" __global__ void
saved_test_kernel ()
{
}

#endif

void
SaveCodeObjectTest ()
{
  /* Copy the content of the module file into a memory buffer.  */
  const char *modpath = "save_code_objects.hipfb";
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

  hipModule_t m1;
  CHECK (hipModuleLoadData (&m1, module_buffer.data ()));

  /* A first version of the code object should be saved at this point.  */

  CHECK (hipModuleUnload (m1));

  /* Load a second module from the same location, which should have the same
     uri as the previous one.  */
  hipModule_t m2;
  CHECK (hipModuleLoadData (&m2, module_buffer.data ()));

  /* A second version of the code object should be saved at this point.  */

  CHECK (hipModuleUnload (m2));

}

