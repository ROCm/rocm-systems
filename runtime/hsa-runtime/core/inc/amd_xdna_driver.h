////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////
#ifndef HSA_RUNTIME_CORE_INC_AMD_XDNA_DRIVER_H_
#define HSA_RUNTIME_CORE_INC_AMD_XDNA_DRIVER_H_

#include <array>
#include <climits>
#include <map>
#include <memory>
#include <unordered_map>

#include "core/driver/xdna/uapi/amdxdna_accel.h"
#include "core/inc/amd_aie_agent.h"
#include "core/inc/driver.h"
#include "core/inc/memory_region.h"

/// @brief struct amdxdna_cmd_chain - Interpretation of data payload for
/// ERT_CMD_CHAIN
struct amdxdna_cmd_chain {
  /// Number of commands in chain
  __u32 command_count;
  /// Index of last successfully submitted command in chain
  __u32 submit_index;
  /// Index of failing command if cmd status is not completed
  __u32 error_index;
  __u32 reserved[3];
  /// Address of each command in chain
  __u64 data[] __counted_by(command_count);
};

/// @brief struct amdxdna_cmd - Exec buffer command header format
struct amdxdna_cmd {
  union {
    struct {
      /// Current state of a command
      __u32 state : 4;
      __u32 unused : 6;
      /// Extra CU masks in addition to mandatory mask
      __u32 extra_cu_masks : 2;
      /// Number of words in payload (data)
      __u32 count : 11;
      /// Opcode identifying specific command
      __u32 opcode : 5;
      __u32 reserved : 4;
    };
    __u32 header;
  };
  /// Count number of words representing packet payload
  __u32 data[] __counted_by(count);
};

namespace rocr {
namespace core {
class Queue;
}

namespace AMD {

// @brief: Used to transform an address into a device address
constexpr uint32_t DEV_ADDR_BASE = 0x04000000;
constexpr uint32_t DEV_ADDR_OFFSET_MASK = 0x02FFFFFF;

/// @brief: The driver places a structure before each command in a command chain.
/// Need to increase the size of the command by the size of this structure.
/// In the following xdna driver source can see where this is implemented:
/// Commit hash: eddd92c0f61592c576a500f16efa24eb23667c23
/// https://github.com/amd/xdna-driver/blob/main/src/driver/amdxdna/aie2_msg_priv.h#L387-L391
/// https://github.com/amd/xdna-driver/blob/main/src/driver/amdxdna/aie2_message.c#L637
constexpr uint32_t CMD_COUNT_SIZE_INCREASE = 3;

/// @brief: The size of an instruction in bytes
constexpr uint32_t INSTR_SIZE_BYTES = 4;

/// @brief: Index of command payload where the instruction sequence
/// address is located
constexpr uint32_t CMD_PKT_PAYLOAD_INSTRUCTION_SEQUENCE_IDX = 2;
constexpr uint32_t CMD_PKT_PAYLOAD_INSTRUCTION_SEQUENCE_SIZE_IDX = 4;

/// @brief Environment variable to define job submission timeout
constexpr uint32_t DEFAULT_TIMEOUT_VAL = 50;

class XdnaDriver final : public core::Driver {
  /// @brief BO handle information.
  struct BOHandle {
    /// Mapped address.
    void* vaddr = nullptr;
    /// Handle returned by xdna.
    uint32_t handle = AMDXDNA_INVALID_BO_HANDLE;
    /// Size in bytes.
    size_t size = 0;

    constexpr BOHandle() = default;
    constexpr BOHandle(void* vaddr, uint32_t handle, size_t size)
        : vaddr{vaddr}, handle{handle}, size{size} {}
    constexpr bool IsValid() const { return handle != AMDXDNA_INVALID_BO_HANDLE; }
  };

  /// @brief CU mask size.
  static constexpr size_t cu_mask_size = sizeof(uint32_t) * CHAR_BIT;

  /// @brief Per hardware context PDI cache.
  class PDICache {
    std::array<BOHandle, cu_mask_size> entries = {};
    size_t entry_count = 0;

   public:
    /// @brief Sentinel value for entries not found.
    constexpr static size_t NotFound = cu_mask_size;

    /// @brief Returns the size of the cache.
    constexpr size_t size() const { return entry_count; }

    /// @brief Returns the index of the BO handle if it is the cache, otherwise @ref NotFound.
    ///
    /// This function does a linear search because the mask is small (32 elements).
    size_t GetIndex(uint32_t pdi_handle) const {
      for (size_t i = 0; i < entry_count; ++i) {
        if (entries[i].handle == pdi_handle) {
          return i;
        }
      }
      return NotFound;
    }

    /// @brief Sets the next cache entry.
    hsa_status_t SetNext(const BOHandle& pdi_bo_handle, size_t& index) {
      if (entry_count == entries.size()) {
        // cache is full
        return HSA_STATUS_ERROR_OUT_OF_RESOURCES;
      }

      index = entry_count++;
      entries[index] = pdi_bo_handle;
      return HSA_STATUS_SUCCESS;
    }

    constexpr const BOHandle& operator[](size_t index) const { return entries[index]; }
  };

public:
  XdnaDriver(std::string devnode_name);

  static hsa_status_t DiscoverDriver(std::unique_ptr<core::Driver>& driver);

  /// @brief Returns the size of the system memory heap in bytes.
  static uint64_t GetSystemMemoryByteSize();

  /// @brief Returns the size of the dev heap in bytes.
  static uint64_t GetDevHeapByteSize();

  hsa_status_t Init() override;
  hsa_status_t ShutDown() override;
  hsa_status_t QueryKernelModeDriver(core::DriverQuery query) override;

  hsa_status_t Open() override;
  hsa_status_t Close() override;
  hsa_status_t GetSystemProperties(HsaSystemProperties& sys_props) const override;
  hsa_status_t GetNodeProperties(HsaNodeProperties& node_props, uint32_t node_id) const override;
  hsa_status_t GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                 uint32_t node_id) const override;
  hsa_status_t GetMemoryProperties(uint32_t node_id,
                                   std::vector<HsaMemoryProperties>& mem_props) const override;
  hsa_status_t GetCacheProperties(uint32_t node_id, uint32_t processor_id,
                                  std::vector<HsaCacheProperties>& cache_props) const override;
  hsa_status_t AllocateMemory(const core::MemoryRegion &mem_region,
                              core::MemoryRegion::AllocateFlags alloc_flags,
                              void **mem, size_t size,
                              uint32_t node_id) override;
  hsa_status_t FreeMemory(void *mem, size_t size) override;
  hsa_status_t CreateQueue(uint32_t node_id, HSA_QUEUE_TYPE type, uint32_t queue_pct,
                           HSA_QUEUE_PRIORITY priority, uint32_t sdma_engine_id, void* queue_addr,
                           uint64_t queue_size_bytes, HsaEvent* event,
                           HsaQueueResource& queue_resource) const override;
  hsa_status_t UpdateQueue(HSA_QUEUEID queue_id, uint32_t queue_pct, HSA_QUEUE_PRIORITY priority,
                           void* queue_addr, uint64_t queue_size, HsaEvent* event) const override;
  hsa_status_t DestroyQueue(HSA_QUEUEID queue_id) const override;
  hsa_status_t SetQueueCUMask(HSA_QUEUEID queue_id, uint32_t cu_mask_count,
                              uint32_t* queue_cu_mask) const override;
  hsa_status_t AllocQueueGWS(HSA_QUEUEID queue_id, uint32_t num_gws,
                             uint32_t* first_gws) const override;
  hsa_status_t ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                            size_t *offset) override;
  hsa_status_t ImportDMABuf(int dmabuf_fd, core::Agent &agent,
                            core::ShareableHandle &handle) override;
  hsa_status_t Map(core::ShareableHandle handle, void *mem, size_t offset,
                   size_t size, hsa_access_permission_t perms) override;
  hsa_status_t Unmap(core::ShareableHandle handle, void *mem, size_t offset,
                     size_t size) override;
  hsa_status_t ReleaseShareableHandle(core::ShareableHandle &handle) override;

  /// @brief Submits @p num_pkts packets in a command chain.
  hsa_status_t SubmitCmdChain(hsa_amd_aie_ert_packet_t* first_pkt, uint32_t num_pkts,
                              HSA_QUEUEID& queue_id, uint32_t num_core_tiles);

  hsa_status_t SPMAcquire(uint32_t preferred_node_id) const override;
  hsa_status_t SPMRelease(uint32_t preferred_node_id) const override;
  hsa_status_t SPMSetDestBuffer(uint32_t preferred_node_id, uint32_t size_bytes, uint32_t* timeout,
                                uint32_t* size_copied, void* dest_mem_addr,
                                bool* is_spm_data_loss) const override;
  hsa_status_t SetTrapHandler(uint32_t node_id, const void* base, uint64_t base_size,
                              const void* buffer_base, uint64_t buffer_base_size) const override;
  hsa_status_t GetDeviceHandle(uint32_t node_id, void** device_handle) const override;
  hsa_status_t GetClockCounters(uint32_t node_id, HsaClockCounters* clock_counter) const override;
  hsa_status_t GetTileConfig(uint32_t node_id, HsaGpuTileConfig* config) const override;
  hsa_status_t GetWallclockFrequency(uint32_t node_id, uint64_t* frequency) const override;

  hsa_status_t IsModelEnabled(bool* enable) const override;

 private:
  /// @brief Destroys @p bo_handle.
  ///
  /// This function will unmap the virtual address and close the BO, but will not return any status.
  void DestroyBOHandle(BOHandle& bo_handle);

  /// @brief Finds the BO associated with the address.
  BOHandle FindBOHandle(void* mem) const;

  /// @brief Creates a new hardware context with the given PDI BO handles.
  hsa_status_t ConfigHwCtx(const PDICache& pdi_bo_handles, HSA_QUEUEID& queue_id,
                           uint32_t num_core_tiles);

  hsa_status_t QueryDriverVersion();

  /// @brief Allocate device accesible heap space.
  ///
  /// Allocate and map a buffer object (BO) that the AIE device can access.
  hsa_status_t InitDeviceHeap();
  hsa_status_t FreeDeviceHeap();

  /// @brief Creates a command BO and returns it to @p bo_info.
  ///
  /// @param size size of memory to allocate
  /// @param bo_info allocated BO
  hsa_status_t CreateCmdBO(uint32_t size, BOHandle& bo_info);

  /// @brief Gets all BOs from a command packet payload, flushes the caches associated with them and
  /// replaces the instruction virtual address with the device address.
  ///
  /// @param count Number of entries in the command
  /// @param cmd_pkt_payload A pointer to the payload of the command
  /// @param bo_handles vector that contains all BO handles
  hsa_status_t PrepareBOs(uint32_t count, hsa_amd_aie_ert_start_kernel_data_t* cmd_pkt_payload,
                          std::vector<uint32_t>& bo_handles);

  /// @brief Executes a command and waits for its completion
  ///
  /// @param cmd_chain_bo_handle command to execute
  /// @param bo_handles handles associated with the command
  /// @param aie_queue queue to submit to
  hsa_status_t ExecCmdAndWait(const BOHandle& cmd_chain_bo_handle,
                              const std::vector<uint32_t>& bo_handles, HSA_QUEUEID queue_id);

  /// TODO: Remove this in the future and rely on the core Runtime
  /// object to track handle allocations. Using the VMEM API for mapping XDNA
  /// driver handles requires a bit more refactoring. So rely on the XDNA driver
  /// to manage some of this for now.
  std::unordered_map<uint32_t, void *> vmem_handle_mappings;
  std::map<void*, BOHandle> vmem_addr_mappings;

  /// @brief Hardware context to PDI cache mapping.
  std::unordered_map<uint32_t, PDICache> hw_ctx_pdi_cache_map;

  /// @brief Virtual address range allocated for the device heap.
  ///
  /// Allocate a large enough space so we can carve out the device heap in
  /// this range and ensure it is aligned to 64MB. Currently, npu1 supports
  /// 64MB device heap and it must be aligned to 64MB.
  BOHandle dev_heap_handle;

  /// @brief The aligned device heap.
  void *dev_heap_aligned = nullptr;

  static constexpr size_t dev_heap_size = 64 * 1024 * 1024;
  static constexpr size_t dev_heap_align = 64 * 1024 * 1024;
};

} // namespace AMD
} // namespace rocr

#endif // header guard
