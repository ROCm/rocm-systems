
/*
Copyright (c) 2019 - present Advanced Micro Devices, Inc. All rights reserved.

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
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef COMMON_RDC_UTILS_H_
#define COMMON_RDC_UTILS_H_

#include <cstdint>
#include <string>
#include <vector>

namespace amd {
namespace rdc {

#ifdef NDEBUG
#define debug_print(fmt, ...) \
  do {                        \
  } while (false)
#else
#define debug_print(fmt, ...)            \
  do {                                   \
    fprintf(stderr, fmt, ##__VA_ARGS__); \
  } while (false)
#endif

bool FileExists(char const* filename);

int ReadFile(std::string path, std::string* retStr, bool chop_newline = false);
int ReadFile(const char* path, std::string* retStr, bool chop_newline = false);

bool IsNumber(const std::string& s);
bool IsIP(const std::string& s);

// A GPU index's socket, and its partition there; -1 when the socket is not partitioned.
// RDC's partition ID g<socket>.<partition> names it.
struct GpuPartition {
  uint32_t socket;
  int32_t partition;
};

// RDC lists GPU indexes socket by socket, a partitioned socket's partitions in order, and
// reports no index's socket. `counts` holds rdc_get_num_partition() for each GPU index: its
// socket's partition count, 1 for a socket in SPX mode, UINT16_MAX for a GPU that cannot be
// partitioned. An index whose count n is above 1 starts a socket that takes the indexes after
// it while they report the same n, up to n of them, the k-th being partition k; any other
// index is a socket of its own. A socket that lists fewer indexes than its count, followed by
// a socket with the same count, reads as one socket.
std::vector<GpuPartition> GpuPartitions(const std::vector<uint16_t>& counts);

}  // namespace rdc
}  // namespace amd

#endif  // COMMON_RDC_UTILS_H_
