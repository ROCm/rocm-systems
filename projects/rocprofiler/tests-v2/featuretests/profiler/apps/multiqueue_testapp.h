/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef TESTS_FEATURETESTS_PROFILER_DISCRETETESTS_BINARY_MULTIQUEUE_TESTAPP_H_
#define TESTS_FEATURETESTS_PROFILER_DISCRETETESTS_BINARY_MULTIQUEUE_TESTAPP_H_

#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <hsa/hsa.h>
#include <hsa/hsa_api_trace.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <cstdlib>

#include <iostream>
#include <string>
#include <vector>

#include "src/utils/exception.h"

#define ASSERT_EQ(val1, val2)                                                                      \
  do {                                                                                             \
    if ((val1) != val2) {                                                                          \
      assert(false);                                                                               \
      abort();                                                                                     \
    }                                                                                              \
  } while (false)

struct Device {
  struct Memory {
    hsa_amd_memory_pool_t pool;
    bool fine;
    bool kernarg;
    size_t size;
    size_t granule;
  };

  hsa_agent_t agent;
  char name[64];
  std::vector<Memory> pools;
  uint32_t fine;
  uint32_t coarse;
  static std::vector<hsa_agent_t> all_devices;
};

std::vector<Device> cpu, gpu;
Device::Memory kernarg;

class MQDependencyTest {
 public:
  MQDependencyTest() { hsa_init(); }
  ~MQDependencyTest() { hsa_shut_down(); }

  struct CodeObject {
    hsa_file_t file;
    hsa_code_object_reader_t code_obj_rdr;
    hsa_executable_t executable;
  };

  struct Kernel {
    uint64_t handle;
    uint32_t scratch;
    uint32_t group;
    uint32_t kernarg_size;
    uint32_t kernarg_align;
  };

  union AqlHeader {
    struct {
      uint16_t type : 8;
      uint16_t barrier : 1;
      uint16_t acquire : 2;
      uint16_t release : 2;
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

  bool LoadCodeObject(std::string filename, hsa_agent_t agent, CodeObject& code_object) {
    hsa_status_t err;
    printf("%s", filename.c_str());
    code_object.file = open(filename.c_str(), O_RDONLY);
    if (code_object.file == -1) {
      abort();
      return false;
    }

    err = hsa_code_object_reader_create_from_file(code_object.file, &code_object.code_obj_rdr);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                    nullptr, &code_object.executable);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_executable_load_agent_code_object(code_object.executable, agent,
                                                code_object.code_obj_rdr, nullptr, nullptr);
    if (err != HSA_STATUS_SUCCESS) return false;

    err = hsa_executable_freeze(code_object.executable, nullptr);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    return true;
  }

  bool GetKernel(const CodeObject& code_object, std::string kernel, hsa_agent_t agent,
                 Kernel& kern) {
    hsa_executable_symbol_t symbol;
    hsa_status_t err =
        hsa_executable_get_symbol_by_name(code_object.executable, kernel.c_str(), &agent, &symbol);
    if (err != HSA_STATUS_SUCCESS) {
      err = hsa_executable_get_symbol_by_name(code_object.executable, (kernel + ".kd").c_str(),
                                              &agent, &symbol);
      if (err != HSA_STATUS_SUCCESS) {
        return false;
      }
    }
    // printf("\nkernel-name: %s\n", kernel.c_str());
    err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                         &kern.handle);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &kern.scratch);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    // printf("Scratch: %d\n", kern.scratch);

    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &kern.group);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    // printf("LDS: %d\n", kern.group);

    // Remaining needs code object v2 or comgr.
    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kern.kernarg_size);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    // printf("Kernarg Size: %d\n", kern.kernarg_size);

    err = hsa_executable_symbol_get_info(
        symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &kern.kernarg_align);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    // printf("Kernarg Align: %d\n", kern.kernarg_align);

    return true;
  }

  // Not for parallel insertion.
  bool SubmitPacket(hsa_queue_t* queue, Aql& pkt) {
    size_t mask = queue->size - 1;
    Aql* ring = static_cast<Aql*>(queue->base_address);

    uint64_t write = hsa_queue_load_write_index_relaxed(queue);
    uint64_t read = hsa_queue_load_read_index_relaxed(queue);
    if (write - read + 1 > queue->size) return false;

    Aql& dst = ring[write & mask];

    uint16_t header = pkt.header.raw;
    pkt.header.raw = dst.header.raw;
    dst = pkt;
    __atomic_store_n(&dst.header.raw, header, __ATOMIC_RELEASE);
    pkt.header.raw = header;

    hsa_queue_store_write_index_release(queue, write + 1);
    hsa_signal_store_screlease(queue->doorbell_signal, write);

    return true;
  }

  void* hsaMalloc(size_t size, const Device::Memory& mem) {
    void* ret;
    hsa_status_t err = hsa_amd_memory_pool_allocate(mem.pool, size, HSA_AMD_MEMORY_POOL_EXECUTABLE_FLAG, &ret);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    err = hsa_amd_agents_allow_access(Device::all_devices.size(), &Device::all_devices[0], nullptr,
                                      ret);
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);
    return ret;
  }

  void* hsaMalloc(size_t size, const Device& dev, bool fine) {
    uint32_t index = fine ? dev.fine : dev.coarse;
    assert(index != -1u && "Memory type unavailable.");
    return hsaMalloc(size, dev.pools[index]);
  }

  bool DeviceDiscovery() {
    hsa_status_t err;
    err = hsa_iterate_agents(
        [](hsa_agent_t agent, void*) {
          hsa_status_t err;

          Device dev;
          dev.agent = agent;

          dev.fine = -1u;
          dev.coarse = -1u;

          err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, dev.name);
          ASSERT_EQ(err, HSA_STATUS_SUCCESS);

          hsa_device_type_t type;
          err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
          ASSERT_EQ(err, HSA_STATUS_SUCCESS);

          err = hsa_amd_agent_iterate_memory_pools(
              agent,
              [](hsa_amd_memory_pool_t pool, void* data) {
                std::vector<Device::Memory>& pools =
                    *reinterpret_cast<std::vector<Device::Memory>*>(data);
                hsa_status_t err;

                hsa_amd_segment_t segment;
                err =
                    hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
                ASSERT_EQ(err, HSA_STATUS_SUCCESS);

                if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

                uint32_t flags;
                err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS,
                                                   &flags);
                ASSERT_EQ(err, HSA_STATUS_SUCCESS);

                Device::Memory mem;
                mem.pool = pool;
                mem.fine = (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED);
                mem.kernarg = (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);

                err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &mem.size);
                ASSERT_EQ(err, HSA_STATUS_SUCCESS);

                err = hsa_amd_memory_pool_get_info(
                    pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &mem.granule);
                ASSERT_EQ(err, HSA_STATUS_SUCCESS);

                pools.push_back(mem);
                return HSA_STATUS_SUCCESS;
              },
              static_cast<void*>(&dev.pools));

          if (!dev.pools.empty()) {
            for (size_t i = 0; i < dev.pools.size(); i++) {
              if (dev.pools[i].fine && dev.pools[i].kernarg && dev.fine == -1u) dev.fine = i;
              if (dev.pools[i].fine && !dev.pools[i].kernarg) dev.fine = i;
              if (!dev.pools[i].fine) dev.coarse = i;
            }

            if (type == HSA_DEVICE_TYPE_CPU)
              cpu.push_back(dev);
            else
              gpu.push_back(dev);

            Device::all_devices.push_back(dev.agent);
          }

          return HSA_STATUS_SUCCESS;
        },
        nullptr);

    []() {
      for (auto& dev : cpu) {
        for (auto& mem : dev.pools) {
          if (mem.fine && mem.kernarg) {
            kernarg = mem;
            return;
          }
        }
      }
    }();
    ASSERT_EQ(err, HSA_STATUS_SUCCESS);

    if (cpu.empty() || gpu.empty() || kernarg.pool.handle == 0) return false;
    return true;
  }
};
#endif  // TESTS_FEATURETESTS_PROFILER_DISCRETETESTS_BINARY_MULTIQUEUE_TESTAPP_H_
