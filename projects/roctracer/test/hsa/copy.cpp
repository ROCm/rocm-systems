/* Copyright (c) 2022 Advanced Micro Devices, Inc.

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

#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <hsa/hsa_ext_image.h>

#include <fcntl.h>
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <chrono>
#include <thread>

#define CHECK(x)                                                                                   \
  do {                                                                                             \
    if ((x) != HSA_STATUS_SUCCESS) {                                                               \
      assert(false);                                                                               \
      abort();                                                                                     \
    }                                                                                              \
  } while (false);

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
std::vector<hsa_agent_t> Device::all_devices;

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

struct hip_hiddens {
  uint64_t offset_x;
  uint64_t offset_y;
  uint64_t offset_z;
  uint64_t _;
  uint64_t _2;
  uint64_t _3;
  uint64_t multi_grid_sync;
};


std::vector<Device> cpu, gpu;
Device::Memory kernarg;

struct CodeObject {
  hsa_file_t file;
  hsa_code_object_reader_t code_obj_rdr;
  hsa_executable_t executable;
};

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
        CHECK(err);

        hsa_device_type_t type;
        err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
        CHECK(err);

        err = hsa_amd_agent_iterate_memory_pools(
            agent,
            [](hsa_amd_memory_pool_t pool, void* data) {
              std::vector<Device::Memory>& pools =
                  *reinterpret_cast<std::vector<Device::Memory>*>(data);
              hsa_status_t err;

              hsa_amd_segment_t segment;
              err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
              CHECK(err);

              if (segment != HSA_AMD_SEGMENT_GLOBAL) return HSA_STATUS_SUCCESS;

              uint32_t flags;
              err =
                  hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
              CHECK(err);

              Device::Memory mem;
              mem.pool = pool;
              mem.fine = (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED);
              mem.kernarg = (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);

              err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &mem.size);
              CHECK(err);

              err = hsa_amd_memory_pool_get_info(
                  pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &mem.granule);
              CHECK(err);

              pools.push_back(mem);
              return HSA_STATUS_SUCCESS;
            },
            (void*)&dev.pools);

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
  CHECK(err);

  if (cpu.empty() || gpu.empty() || kernarg.pool.handle == 0) return false;
  return true;
}

bool LoadCodeObject(std::string filename, hsa_agent_t agent, CodeObject& code_object) {
  hsa_status_t err;

  code_object.file = open(filename.c_str(), O_RDONLY);
  if (code_object.file == -1) return false;

  err = hsa_code_object_reader_create_from_file(code_object.file, &code_object.code_obj_rdr);
  CHECK(err);

  err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                  nullptr, &code_object.executable);
  CHECK(err);

  err = hsa_executable_load_agent_code_object(code_object.executable, agent,
                                              code_object.code_obj_rdr, nullptr, nullptr);
  if (err != HSA_STATUS_SUCCESS) return false;

  err = hsa_executable_freeze(code_object.executable, nullptr);
  CHECK(err);

  return true;
}

bool GetKernel(const CodeObject& code_object, std::string kernel, hsa_agent_t agent, Kernel& kern) {
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

  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &kern.handle);
  CHECK(err);

  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &kern.scratch);
  CHECK(err);
  // printf("Scratch: %d\n", kern.scratch);

  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE,
                                       &kern.group);
  CHECK(err);
  // printf("LDS: %d\n", kern.group);

  // Remaining needs code object v2 or comgr.
  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kern.kernarg_size);
  CHECK(err);
  // printf("Kernarg Size: %d\n", kern.kernarg_size);

  err = hsa_executable_symbol_get_info(
      symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &kern.kernarg_align);
  CHECK(err);
  // printf("Kernarg Align: %d\n", kern.kernarg_align);

  return true;
}

// Not for parallel insertion.
bool SubmitPacket(hsa_queue_t* queue, Aql& pkt) {
  size_t mask = queue->size - 1;
  Aql* ring = (Aql*)queue->base_address;

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
  hsa_status_t err = hsa_amd_memory_pool_allocate(mem.pool, size, 0, &ret);
  CHECK(err);
  err = hsa_amd_agents_allow_access(Device::all_devices.size(), &Device::all_devices[0], nullptr,
                                    ret);
  CHECK(err);
  return ret;
}

void* hsaMalloc(size_t size, const Device& dev, bool fine) {
  uint32_t index = fine ? dev.fine : dev.coarse;
  assert(index != -1u && "Memory type unavailable.");
  return hsaMalloc(size, dev.pools[index]);
}

void test_func(int kiter, int diter, int agents) {
  for (int device_index = 0; device_index < agents; ++device_index) {
    hsa_status_t err;

    hsa_queue_t* queue;
    err = hsa_queue_create(gpu[device_index].agent, 1024, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr,
                           0, 0, &queue);
    CHECK(err);

    CodeObject code_object;
    if (!LoadCodeObject(std::string(gpu[device_index].name) + "_copy.hsaco",
                        gpu[device_index].agent, code_object)) {
      printf("Kernel file not found or not usable with given agent.\n");
      abort();
    }

    Kernel copy;
    if (!GetKernel(code_object, "copy", gpu[device_index].agent, copy)) {
      printf("Test kernel not found.\n");
      abort();
    }

    for (int i = 0; i < kiter; ++i) {
      struct args_t {
        uint32_t* a;
        uint32_t* b;
        OCLHiddenArgs hidden;
      };

      args_t* args;
      args = (args_t*)hsaMalloc(sizeof(args_t), kernarg);
      memset(args, 0, sizeof(args_t));

      uint32_t* a = (uint32_t*)hsaMalloc(64 * sizeof(uint32_t), kernarg);
      uint32_t* b = (uint32_t*)hsaMalloc(64 * sizeof(uint32_t), kernarg);

      memset(a, 0, 64 * sizeof(uint32_t));
      memset(b, 1, 64 * sizeof(uint32_t));

      hsa_signal_t signal;
      // Use interrupts.
      err = hsa_amd_signal_create(1, 0, nullptr, 0, &signal);
      CHECK(err);

      for (int j = 1; j <= diter; ++j) {
        Aql packet{};
        packet.header.type = HSA_PACKET_TYPE_KERNEL_DISPATCH;
        packet.header.barrier = 1;
        packet.header.acquire = HSA_FENCE_SCOPE_SYSTEM;
        packet.header.release = HSA_FENCE_SCOPE_SYSTEM;

        packet.dispatch.setup = 1;
        packet.dispatch.workgroup_size_x = 64;
        packet.dispatch.workgroup_size_y = 1;
        packet.dispatch.workgroup_size_z = 1;
        packet.dispatch.grid_size_x = 64;
        packet.dispatch.grid_size_y = 1;
        packet.dispatch.grid_size_z = 1;

        packet.dispatch.group_segment_size = copy.group;
        packet.dispatch.private_segment_size = copy.scratch;
        packet.dispatch.kernel_object = copy.handle;

        packet.dispatch.kernarg_address = args;
        if (j == diter) packet.dispatch.completion_signal = signal;

        args->a = a;
        args->b = b;
        SubmitPacket(queue, packet);
      }
      hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_EQ, 0, -1, HSA_WAIT_STATE_BLOCKED);
      err = hsa_signal_destroy(signal);
      CHECK(err);

      for (int i = 0; i < 64; i++) {
        if (a[i] != b[i]) {
          printf("error at %d: expected %d, got %d\n", i, b[i], a[i]);
          abort();
        }
      }

      err = hsa_memory_free(a);
      CHECK(err);
      err = hsa_memory_free(b);
      CHECK(err);
    }

    err = hsa_executable_destroy(code_object.executable);
    CHECK(err);
    err = hsa_code_object_reader_destroy(code_object.code_obj_rdr);
    CHECK(err);
    close(code_object.file);
  }
}

int main(int argc, char** argv) {
  const char* kiter_s = getenv("ROCP_KITER");
  const char* diter_s = getenv("ROCP_DITER");
  const char* agents_s = getenv("ROCP_AGENTS");
  const char* threads_s = getenv("ROCP_THRS");

  int kiter = (kiter_s != nullptr) ? atoi(kiter_s) : 1;
  int diter = (diter_s != nullptr) ? atoi(diter_s) : 1;
  int agents = (agents_s != nullptr) ? atoi(agents_s) : 1;
  int threads = (threads_s != nullptr) ? atoi(threads_s) : 1;

  hsa_status_t err;
  err = hsa_init();
  CHECK(err);

  if (!DeviceDiscovery()) {
    printf("Usable devices not found.\n");
    return -1;
  }

  std::vector<std::thread> t(threads);
  for (int n = 0; n < threads; ++n)
    t[n] = std::thread(test_func, kiter, diter, std::min(agents, (int)gpu.size()));
  for (int n = 0; n < threads; ++n) t[n].join();

  err = hsa_shut_down();
  CHECK(err);

  return 0;
}