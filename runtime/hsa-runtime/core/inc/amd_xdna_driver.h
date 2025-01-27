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

/// @brief: The number of arguments in the packet payload before we start passing operands
constexpr uint32_t NON_OPERAND_COUNT = 6;

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

/// @brief: Calculates the number of operands in a packet
/// given the number of arguments in the packet
/// @param: arg_count(Input), Number of arguments in the packet
/// @return: uint32_t, The number of operands in the packet
inline uint32_t GetOperandCount(uint32_t arg_count) {
  return ((arg_count - NON_OPERAND_COUNT) / 2);
}

class XdnaDriver final : public core::Driver {
public:
  XdnaDriver(std::string devnode_name);
  ~XdnaDriver() = default;

  static hsa_status_t DiscoverDriver(std::unique_ptr<core::Driver>& driver);

  /// @brief Returns the size of the dev heap in bytes.
  static uint64_t GetDevHeapByteSize();

  hsa_status_t Init() override;
  hsa_status_t ShutDown() override;
  hsa_status_t QueryKernelModeDriver(core::DriverQuery query) override;

  std::unordered_map<uint32_t, void*>& GetHandleMappings();
  std::unordered_map<void*, uint32_t>& GetAddrMappings();

  hsa_status_t Open() override;
  hsa_status_t Close() override;
  hsa_status_t GetSystemProperties(HsaSystemProperties& sys_props) const override;
  hsa_status_t GetNodeProperties(HsaNodeProperties& node_props, uint32_t node_id) const override;
  hsa_status_t GetEdgeProperties(std::vector<HsaIoLinkProperties>& io_link_props,
                                 uint32_t node_id) const override;
  hsa_status_t GetAgentProperties(core::Agent &agent) const override;
  hsa_status_t
  GetMemoryProperties(uint32_t node_id,
                      core::MemoryRegion &mem_region) const override;
  hsa_status_t AllocateMemory(const core::MemoryRegion &mem_region,
                              core::MemoryRegion::AllocateFlags alloc_flags,
                              void **mem, size_t size,
                              uint32_t node_id) override;
  hsa_status_t FreeMemory(void *mem, size_t size) override;
  hsa_status_t CreateQueue(core::Queue &queue) const override;
  hsa_status_t DestroyQueue(core::Queue &queue) const override;
  hsa_status_t ExportDMABuf(void *mem, size_t size, int *dmabuf_fd,
                            size_t *offset) override;
  hsa_status_t ImportDMABuf(int dmabuf_fd, core::Agent &agent,
                            core::ShareableHandle &handle) override;
  hsa_status_t Map(core::ShareableHandle handle, void *mem, size_t offset,
                   size_t size, hsa_access_permission_t perms) override;
  hsa_status_t Unmap(core::ShareableHandle handle, void *mem, size_t offset,
                     size_t size) override;
  hsa_status_t ReleaseShareableHandle(core::ShareableHandle &handle) override;

  // @brief Submits num_pkts packets in a command chain to the XDNA driver
  hsa_status_t SubmitCmdChain(hsa_amd_aie_ert_packet_t* first_pkt, uint32_t num_pkts,
                              uint32_t num_operands, uint32_t hw_ctx_handle);

 private:
  hsa_status_t QueryDriverVersion();
  /// @brief Allocate device accesible heap space.
  ///
  /// Allocate and map a buffer object (BO) that the AIE device can access.
  hsa_status_t InitDeviceHeap();
  hsa_status_t FreeDeviceHeap();

  /// @brief Creates a command BO and returns a pointer to the memory and
  //          the corresponding handle
  ///
  /// @param size size of memory to allocate
  /// @param handle A pointer to the BO handle
  /// @param cmd A pointer to the buffer
  hsa_status_t CreateCmd(uint32_t size, uint32_t* handle, amdxdna_cmd** cmd);

  /// @brief Adds all BOs in a command packet payload to a vector
  ///         and replaces the handles with a virtual address
  ///
  /// @param count Number of entries in the command
  /// @param bo_args A pointer to a vector that contains all bo handles
  /// @param cmd_pkt_payload A pointer to the payload of the command
  hsa_status_t RegisterCmdBOs(uint32_t count, std::vector<uint32_t>& bo_args,
                              std::vector<uint32_t>& bo_sizes, std::vector<uint64_t>& bo_addrs,
                              hsa_amd_aie_ert_start_kernel_data_t* cmd_pkt_payload,
                              const std::unordered_map<void*, uint32_t>& vmem_addr_mappings);

  /// @brief Syncs all BOs referenced in bo_args
  ///
  /// @param bo_args vector containing handles of BOs to sync
  hsa_status_t SyncBos(const std::vector<uint64_t>& bo_args, const std::vector<uint32_t>& bo_sizes);

  /// @brief Executes a command and waits for its completion
  ///
  /// @param exec_cmd Structure containing the details of the command to execute
  /// @param hw_ctx_handle the handle of the hardware context to run this
  /// command
  hsa_status_t ExecCmdAndWait(amdxdna_drm_exec_cmd* exec_cmd, uint32_t hw_ctx_handle);

  /// TODO: Remove this in the future and rely on the core Runtime
  /// object to track handle allocations. Using the VMEM API for mapping XDNA
  /// driver handles requires a bit more refactoring. So rely on the XDNA driver
  /// to manage some of this for now.
  std::unordered_map<uint32_t, void *> vmem_handle_mappings;
  std::unordered_map<void*, uint32_t> vmem_addr_mappings;

  /// @brief Virtual address range allocated for the device heap.
  ///
  /// Allocate a large enough space so we can carve out the device heap in
  /// this range and ensure it is aligned to 64MB. Currently, npu1 supports
  /// 64MB device heap and it must be aligned to 64MB.
  void *dev_heap_parent = nullptr;

  /// @brief The aligned device heap.
  void *dev_heap_aligned = nullptr;
  static constexpr size_t dev_heap_size = 64 * 1024 * 1024;
  static constexpr size_t dev_heap_align = 64 * 1024 * 1024;
};

} // namespace AMD
} // namespace rocr

#endif // header guard
