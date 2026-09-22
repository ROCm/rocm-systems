/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2021-2021, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#include "hsa/hsa_ext_image.h"

#include "common/helper_funcs.h"

#include "gtest/gtest.h"

#include <fcntl.h>
#include <assert.h>
#include "string.h"

#include <vector>

#define CHECK(err) [&](){                         \
    if(err != HSA_STATUS_SUCCESS) {               \
      EXPECT_EQ(HSA_STATUS_SUCCESS, err);         \
      throw std::runtime_error("CHECK failure."); \
    }                                             \
  }();

struct Device {
  struct Memory {
    hsa_amd_memory_pool_t pool;
    bool fine;
    bool kernarg;
    size_t size;
    size_t granule;
  };

  hsa_agent_t agent;
  std::vector<Memory> pools;
  uint32_t fine;
  uint32_t coarse;
};

struct Kernel {
  uint64_t handle;
  uint32_t scratch;
  uint32_t group;
  uint32_t kernarg_size;
  uint32_t kernarg_align;
};

// Assumes bitfield layout is little endian.
// Assumes std::atomic<uint16_t> is binary compatible with uint16_t and uses HW atomics.
union AqlHeader {
  struct {
    uint16_t type     : 8;
    uint16_t barrier  : 1;
    uint16_t acquire  : 2;
    uint16_t release  : 2;
    uint16_t reserved : 3;
  };
  uint16_t raw;
};

struct BarrierValue {
  AqlHeader header;
  uint8_t AmdFormat;
  uint8_t reserved;
  uint32_t reserved1;
  hsa_signal_t signal;
  hsa_signal_value_t value;
  hsa_signal_value_t mask;
  uint32_t cond;
  uint32_t reserved2;
  uint64_t reserved3;
  uint64_t reserved4;
  hsa_signal_t completion_signal;
};

union Aql {
  AqlHeader header;
  hsa_kernel_dispatch_packet_t dispatch;
  hsa_barrier_and_packet_t barrier_and;
  hsa_barrier_or_packet_t barrier_or;
  BarrierValue barrier_value;
};

struct OCLHiddenArgs {
  uint64_t offset_x;
  uint64_t offset_y;
  uint64_t offset_z;
  void* printf_buffer;
  void* enqueue;
  void* enqueue2;
  void* multi_grid;
};

struct hip_hiddens {
  uint64_t offset_x;
  uint64_t offset_y;
  uint64_t offset_z;
  uint64_t _;
  uint64_t _2;
  uint64_t _3;
  uint64_t multi_grid_sync;
};

class System {
public:
  std::vector<Device> cpu_, gpu_;
  std::vector<hsa_agent_t> all_devices_;
  Device::Memory kernarg_;

  static void Init();
  static void Shutdown();
  static std::vector<Device>& cpu() { return sys.cpu_; }
  static std::vector<Device>& gpu() { return sys.gpu_; }
  static std::vector<hsa_agent_t>& all_devices() { return sys.all_devices_; }
  static Device::Memory& kernarg() { return sys.kernarg_; }
  static System sys;
};

class CodeObject {
public:
  CodeObject(std::string filename, Device& agent);
  ~CodeObject();
  bool GetKernel(std::string name, Kernel& kernel);
private:
  hsa_file_t file;
  hsa_code_object_reader_t code_obj_rdr;
  hsa_executable_t executable;
  hsa_agent_t agent;
};

// Not for parallel insertion.
bool SubmitPacket(hsa_queue_t* queue, Aql& pkt);

void* hsaMalloc(size_t size, const Device::Memory& mem);
void* hsaMalloc(size_t size, const Device& dev, bool fine);
