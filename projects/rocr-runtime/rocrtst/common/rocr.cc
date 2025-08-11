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

#include "common/rocr.h"
System System::sys;

bool DeviceDiscovery(System& devices) {
  hsa_status_t err;

  err = hsa_iterate_agents([](hsa_agent_t agent, void* data) {
    hsa_status_t err;

    System* devices = (System*)data;

    Device dev;
    dev.agent = agent;

    dev.fine = -1u;
    dev.coarse = -1u;

    hsa_device_type_t type;
    err = hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    CHECK(err);

    err = hsa_amd_agent_iterate_memory_pools(agent, [](hsa_amd_memory_pool_t pool, void* data) {
      std::vector<Device::Memory>& pools = *reinterpret_cast<std::vector<Device::Memory>*>(data);
      hsa_status_t err;

      hsa_amd_segment_t segment;
      err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SEGMENT, &segment);
      CHECK(err);

      if(segment != HSA_AMD_SEGMENT_GLOBAL)
        return HSA_STATUS_SUCCESS;

      uint32_t flags;
      err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
      CHECK(err);

      Device::Memory mem;
      mem.pool=pool;
      mem.fine = (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED);
      mem.kernarg = (flags & HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT);

      err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_SIZE, &mem.size);
      CHECK(err);

      err = hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &mem.granule);
      CHECK(err);

      pools.push_back(mem);
      return HSA_STATUS_SUCCESS;
    }, (void*)&dev.pools);

    if(!dev.pools.empty()) {
      for(size_t i=0; i<dev.pools.size(); i++) {
        if(dev.pools[i].fine && dev.pools[i].kernarg && dev.fine==-1u)
          dev.fine = i;
        if(dev.pools[i].fine && !dev.pools[i].kernarg)
          dev.fine = i;
        if(!dev.pools[i].fine)
          dev.coarse = i;
      }

      if(type == HSA_DEVICE_TYPE_CPU)
        devices->cpu_.push_back(dev);
      else
        devices->gpu_.push_back(dev);

      devices->all_devices_.push_back(dev.agent);
    }

    return HSA_STATUS_SUCCESS;
  }, &devices);

  [&]() {
    for(auto& dev : devices.cpu_) {
      for(auto& mem : dev.pools) {
        if(mem.fine && mem.kernarg) {
          devices.kernarg_ = mem;
          return;
        }
      }
    }
  }();

  if(devices.cpu_.empty() || devices.gpu_.empty() || devices.kernarg_.pool.handle == 0)
    return false;
  return true;
}

void System::Init() {
  hsa_status_t err = hsa_init();
  CHECK(err);

  DeviceDiscovery(sys);
}

void System::Shutdown() {
  sys.~System();
  new (&sys) System();
  hsa_status_t err = hsa_shut_down();
  ASSERT_EQ(HSA_STATUS_SUCCESS, err);
  err = hsa_shut_down();
  EXPECT_EQ(HSA_STATUS_ERROR_NOT_INITIALIZED, err);
}

CodeObject::CodeObject(std::string filename, Device& agent) : agent(agent.agent) {
  hsa_status_t err;

  file = open(filename.c_str(), O_RDONLY);
  if(file == -1) {
    throw std::runtime_error("Could not open file.\n");
  }
  MAKE_NAMED_SCOPE_GUARD(fileGuard, [&](){ close(file); });

  err = hsa_code_object_reader_create_from_file(file, &code_obj_rdr);
  CHECK(err);
  MAKE_NAMED_SCOPE_GUARD(readerGuard, [&](){ hsa_code_object_reader_destroy(code_obj_rdr); });
  
  err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, nullptr, &executable);
  CHECK(err);
  MAKE_NAMED_SCOPE_GUARD(exeGuard, [&](){ hsa_executable_destroy(executable); });

  err = hsa_executable_load_agent_code_object(executable, agent.agent, code_obj_rdr, nullptr, nullptr);
  CHECK(err);

  err = hsa_executable_freeze(executable, nullptr);
  CHECK(err);

  exeGuard.Dismiss();
  readerGuard.Dismiss();
  fileGuard.Dismiss();
}

CodeObject::~CodeObject() {
  hsa_executable_destroy(executable);
  hsa_code_object_reader_destroy(code_obj_rdr);
  close(file);
}

bool CodeObject::GetKernel(std::string name, Kernel& kern) {
  hsa_executable_symbol_t symbol;
  hsa_status_t err = hsa_executable_get_symbol_by_name(executable, name.c_str(), &agent, &symbol);
  if(err != HSA_STATUS_SUCCESS) {
    err = hsa_executable_get_symbol_by_name(executable, (name+".kd").c_str(), &agent, &symbol);
    if(err != HSA_STATUS_SUCCESS) {
      return false;
    }
  }

  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT, &kern.handle);
  CHECK(err);

  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &kern.scratch);
  CHECK(err);
  //printf("Scratch: %d\n", kern.scratch);

  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &kern.group);
  CHECK(err);
  //printf("LDS: %d\n", kern.group);
  
  // Remaining needs code object v2 or comgr.
  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &kern.kernarg_size);
  CHECK(err);
  //printf("Kernarg Size: %d\n", kern.kernarg_size);

  err = hsa_executable_symbol_get_info(symbol, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &kern.kernarg_align);
  CHECK(err);
  //printf("Kernarg Align: %d\n", kern.kernarg_align);

  return true;
}

// Not for parallel insertion.
bool SubmitPacket(hsa_queue_t* queue, Aql& pkt) {
  size_t mask = queue->size - 1;
  Aql* ring = (Aql*)queue->base_address;

  uint64_t write = hsa_queue_load_write_index_relaxed(queue);
  uint64_t read = hsa_queue_load_read_index_relaxed(queue);
  //if(write - read + 1 > queue->size)
  //  return false;
  
  Aql& dst = ring[write & mask];

  uint16_t header = pkt.header.raw;
  pkt.header.raw = dst.header.raw;
  dst = pkt;
  __atomic_store_n(&dst.header.raw, header, __ATOMIC_RELEASE);
  pkt.header.raw = header;

  hsa_queue_store_write_index_release(queue, write+1);
  hsa_signal_store_screlease(queue->doorbell_signal, write);

  return true;
}

void* hsaMalloc(size_t size, const Device::Memory& mem) {
  void* ret;
  hsa_status_t err = hsa_amd_memory_pool_allocate(mem.pool, size, 0, &ret);
  CHECK(err);
  err = hsa_amd_agents_allow_access(System::all_devices().size(), &System::all_devices()[0], nullptr, ret);
  CHECK(err);
  return ret;
}

void* hsaMalloc(size_t size, const Device& dev, bool fine) {
  uint32_t index = fine ? dev.fine : dev.coarse;
  assert(index != -1u && "Memory type unavailable.");
  return hsaMalloc(size, dev.pools[index]);
}
