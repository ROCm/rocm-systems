/* The University of Illinois/NCSA
   Open Source License (NCSA)

   Copyright (c) 2020, Advanced Micro Devices, Inc. All rights reserved.

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

extern void VectorAddNormalTest ();
extern void VectorAddDebugTrapTest ();
extern void VectorAddMemoryFaultTest ();
extern void SnapshotCodeObjOnLoadTest ();
extern void SaveCodeObjectTest ();
extern void PrintAllWavesTest ();
extern void SigquitTest ();
extern void VectorAddDebugTrapTestNoDebug ();

static void PrintTestInfo (const char *header);
static void RunVectorAddDebugTrapTest ();
static void RunVectorAddNormalTest ();
static void RunVectorAddMemoryFaultTest ();
static void RunSnapshotCodeObjOnLoadTest ();
static void RunSaveCodeObjectTest ();
static void RunPrintAllWavesTest ();
static void RunSigquitTest ();
static void RunVectorAddDebugTrapTestNoDebug ();

int
main (int argc, char *argv[])
{
  PrintTestInfo ("Debug agent tests start");
  std::vector<unsigned int> run_test_list;
  if (argc == 1)
    {
      run_test_list.push_back (0);
      run_test_list.push_back (1);
      run_test_list.push_back (2);
      run_test_list.push_back (3);
      run_test_list.push_back (4);
    }
  else
    {
      int i = 1;
      while (i <= argc - 1)
        {
          unsigned int test_id = static_cast<int> (*argv[i]) - '0';
          run_test_list.push_back (test_id);
          i++;
        }
    }

  for (unsigned int i = 0; i < run_test_list.size (); i++)
    {
      switch (run_test_list.at (i))
        {
        case 0:
          RunVectorAddNormalTest ();
          break;
        case 1:
          RunVectorAddDebugTrapTest ();
          break;
        case 2:
          RunVectorAddMemoryFaultTest ();
          break;
        case 3:
          RunSnapshotCodeObjOnLoadTest ();
          break;
        case 4:
          RunSaveCodeObjectTest ();
          break;
        case 5:
          RunPrintAllWavesTest ();
          break;
        case 6:
          RunSigquitTest ();
          break;
        case 7:
          RunVectorAddDebugTrapTestNoDebug ();
          break;
        default:
          std::cout << "  *** Invalid Test ID ***" << std::endl;
          break;
        }
    }

  PrintTestInfo ("Debug agent test finished");
  std::cout << TEST_SEPARATOR << std::endl;
}

static void
PrintTestInfo (const char *info)
{
  std::cout << "  *** Debug Agent Test: " << info << " ***" << std::endl;
}

static void
RunVectorAddNormalTest ()
{
  PrintTestInfo ("VectorAddNormalTest start");

  int deviceCount;
  hipError_t err = hipGetDeviceCount (&deviceCount);
  TEST_ASSERT (err == hipSuccess, "hipGetDeviceCount");

  for (int i = 0; i < deviceCount; ++i)
    {
      err = hipSetDevice (i);
      TEST_ASSERT (err == hipSuccess, "hipSetDevice");

      VectorAddNormalTest ();

      err = hipDeviceReset ();
      TEST_ASSERT (err == hipSuccess, "hipDeviceReset");
    }

  PrintTestInfo ("VectorAddNormalTest end");
}

static void
RunVectorAddDebugTrapTest ()
{
  PrintTestInfo ("VectorAddDebugTrapTest start");

  int deviceCount;
  hipError_t err = hipGetDeviceCount (&deviceCount);
  TEST_ASSERT (err == hipSuccess, "hipGetDeviceCount");

  for (int i = 0; i < deviceCount; ++i)
    {
      err = hipSetDevice (i);
      TEST_ASSERT (err == hipSuccess, "hipSetDevice");

      VectorAddDebugTrapTest ();

      err = hipDeviceReset ();
      TEST_ASSERT (err == hipSuccess, "hipDeviceReset");
    }

  PrintTestInfo ("VectorAddDebugTrapTest end");
}

static void
RunVectorAddMemoryFaultTest ()
{
  PrintTestInfo ("VectorAddMemoryFaultTest start");

  int deviceCount;
  hipError_t err = hipGetDeviceCount (&deviceCount);
  TEST_ASSERT (err == hipSuccess, "hipGetDeviceCount");

  for (int i = 0; i < deviceCount; ++i)
    {
      err = hipSetDevice (i);
      TEST_ASSERT (err == hipSuccess, "hipSetDevice");

      VectorAddMemoryFaultTest ();

      err = hipDeviceReset ();
      TEST_ASSERT (err == hipSuccess, "hipDeviceReset");
    }

  PrintTestInfo ("VectorAddMemoryFaultTest end");
}

static void
RunSnapshotCodeObjOnLoadTest ()
{
  PrintTestInfo ("VectorAddMemoryFaultTest start");

  int deviceCount;
  hipError_t err = hipGetDeviceCount (&deviceCount);
  TEST_ASSERT (err == hipSuccess, "hipGetDeviceCount");

  for (int i = 0; i < deviceCount; ++i)
    {
      err = hipSetDevice (i);
      TEST_ASSERT (err == hipSuccess, "hipSetDevice");

      SnapshotCodeObjOnLoadTest ();

      err = hipDeviceReset ();
      TEST_ASSERT (err == hipSuccess, "hipDeviceReset");
    }

  PrintTestInfo ("VectorAddMemoryFaultTest end");
}

static void
RunSaveCodeObjectTest ()
{
  PrintTestInfo ("SaveCodeObjectTest start");
  SaveCodeObjectTest ();
}

static void
RunPrintAllWavesTest ()
{
  PrintTestInfo ("PrintAllWaves start");
  int deviceCount;
  hipError_t err = hipGetDeviceCount (&deviceCount);
  TEST_ASSERT (err == hipSuccess, "hipGetDeviceCount");
  for (int i = 0; i < deviceCount; ++i)
    {
      err = hipSetDevice (i);
      TEST_ASSERT (err == hipSuccess, "hipSetDevice");

      PrintAllWavesTest ();

      err = hipDeviceReset ();
      TEST_ASSERT (err == hipSuccess, "hipDeviceReset");
    }
  PrintTestInfo ("PrintAllWaves end");
}

static void
RunSigquitTest ()
{
  PrintTestInfo ("Sigquit start");
  int deviceCount;
  hipError_t err = hipGetDeviceCount (&deviceCount);
  TEST_ASSERT (err == hipSuccess, "hipGetDeviceCount");
  for (int i = 0; i < deviceCount; ++i)
    {
      err = hipSetDevice (i);
      TEST_ASSERT (err == hipSuccess, "hipSetDevice");

      SigquitTest ();

      err = hipDeviceReset ();
      TEST_ASSERT (err == hipSuccess, "hipDeviceReset");
    }
  PrintTestInfo ("Sigquit end");
}

static void
RunVectorAddDebugTrapTestNoDebug ()
{
  PrintTestInfo ("No debug info start");
  int deviceCount;
  hipError_t err = hipGetDeviceCount (&deviceCount);
  TEST_ASSERT (err == hipSuccess, "hipGetDeviceCount");

  for (int i = 0; i < deviceCount; ++i)
    {
      err = hipSetDevice (i);
      TEST_ASSERT (err == hipSuccess, "hipSetDevice");

      VectorAddDebugTrapTestNoDebug ();

      err = hipDeviceReset ();
      TEST_ASSERT (err == hipSuccess, "hipDeviceReset");
    }
  PrintTestInfo ("No debug info end");
}
