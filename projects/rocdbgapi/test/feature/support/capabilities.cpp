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

#include "capabilities.h"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#if defined(__linux__)
#  include <sys/stat.h>
#  include <unistd.h>
#  include <dirent.h>
#endif

namespace amd::dbgapi::test
{

#if defined(__linux__)

bool
is_kfd_available ()
{
  /* The driver requires both that /dev/kfd exists and that the calling
     process can open it for read.  We probe with access(R_OK) rather
     than actually opening to keep the probe side-effect free.  */
  struct stat st;
  if (::stat ("/dev/kfd", &st) != 0)
    return false;
  return ::access ("/dev/kfd", R_OK) == 0;
}

bool
is_kmd_available ()
{
  return false;
}

namespace
{

/* Read a single uint32_t property from a KFD topology node's
   "properties" file.  Returns 0 if the file or key is missing.  */
uint32_t
read_node_property (const std::string &node_dir, const std::string &key)
{
  std::ifstream f (node_dir + "/properties");
  if (!f)
    return 0;
  std::string k;
  uint64_t v;
  while (f >> k >> v)
    {
      if (k == key)
        return static_cast<uint32_t> (v);
    }
  return 0;
}

} /* namespace */

std::vector<uint32_t>
enumerate_gfx_targets ()
{
  std::vector<uint32_t> out;
  if (!is_kfd_available ())
    return out;

  const char *topo = "/sys/class/kfd/kfd/topology/nodes";
  DIR *d = ::opendir (topo);
  if (d == nullptr)
    return out;

  while (auto *entry = ::readdir (d))
    {
      if (entry->d_name[0] == '.')
        continue;
      std::string node_dir = std::string (topo) + "/" + entry->d_name;

      /* Skip CPU nodes (gpu_id == 0).  */
      std::ifstream gpu_id_file (node_dir + "/gpu_id");
      uint32_t gpu_id = 0;
      if (gpu_id_file)
        gpu_id_file >> gpu_id;
      if (gpu_id == 0)
        continue;

      uint32_t ver = read_node_property (node_dir, "gfx_target_version");
      if (ver != 0)
        out.push_back (ver);
    }

  ::closedir (d);
  return out;
}

bool
is_gpu_available ()
{
  return !enumerate_gfx_targets ().empty ();
}

#else /* !__linux__ */

bool
is_kfd_available ()
{
  return false;
}

bool
is_kmd_available ()
{
  /* TODO: probe KMD via DXGKernel handle once Windows port lands.  */
  return false;
}

std::vector<uint32_t>
enumerate_gfx_targets ()
{
  return {};
}

bool
is_gpu_available ()
{
  return false;
}

#endif

bool
is_driver_available ()
{
  return is_kfd_available () || is_kmd_available ();
}

} /* namespace amd::dbgapi::test */
