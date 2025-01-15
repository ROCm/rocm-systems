////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2022-2023, Advanced Micro Devices, Inc. All rights reserved.
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

// AMD specific HSA backend.

#ifndef HSA_RUNTIME_CORE_INC_AMD_AIE_AGENT_H_
#define HSA_RUNTIME_CORE_INC_AMD_AIE_AGENT_H_

#include "core/inc/agent.h"
#include "core/inc/runtime.h"

namespace rocr {
namespace AMD {

class AieAgent : public core::Agent {
public:
  /// @brief AIE agent constructor.
  /// @param [in] node Node id.
  AieAgent(uint32_t node);

  // @brief AIE agent destructor.
  ~AieAgent();

  hsa_status_t VisitRegion(bool include_peer,
                           hsa_status_t (*callback)(hsa_region_t region,
                                                    void *data),
                           void *data) const;
  hsa_status_t IterateRegion(hsa_status_t (*callback)(hsa_region_t region,
                                                      void *data),
                             void *data) const override;

  hsa_status_t IterateCache(hsa_status_t (*callback)(hsa_cache_t cache,
                                                     void *data),
                            void *value) const override;

  hsa_status_t IterateSupportedIsas(
                    hsa_status_t (*callback)(hsa_isa_t isa, void* data),
                                                  void* data) const override;

  hsa_status_t GetInfo(hsa_agent_info_t attribute, void *value) const override;

  hsa_status_t QueueCreate(size_t size, hsa_queue_type32_t queue_type, uint64_t flags,
                           core::HsaEventCallback event_callback, void* data,
                           uint32_t private_segment_size, uint32_t group_segment_size,
                           core::Queue** queue) override;

  // @brief Override from core::Agent.
  const std::vector<const core::Isa*>& supported_isas() const override {
    return supported_isas_;
  }

  const std::vector<const core::MemoryRegion *> &regions() const override {
    return regions_;
  }

  /// @brief Getter for the AIE system allocator.
  const std::function<void *(size_t size, size_t align,
                             core::MemoryRegion::AllocateFlags flags)> &
  system_allocator() const {
    return system_allocator_;
  }

  /// @brief Getter for the AIE system deallocator.
  const std::function<void(void*)>& system_deallocator() const { return system_deallocator_; }

  // AIE agent methods.
  /// @brief Get the number of columns on this AIE agent.
  uint32_t GetNumCols() const { return num_cols_; }
  void SetNumCols(uint32_t num_cols) { num_cols_ = num_cols; }
  /// @brief Get the number of core tile rows on this AIE agent.
  uint32_t GetNumCoreRows() const { return num_core_rows_; }
  void SetNumCoreRows(uint32_t num_core_rows) { num_core_rows_ = num_core_rows; }
  /// @brief Get the number of core tiles on this AIE agent.
  uint32_t GetNumCores() const { return num_cols_ * num_core_rows_; }

private:
  /// @brief Query the driver to get the region list owned by this agent.
  void InitRegionList();
  /// @brief Setup the memory allocators used by this agent.
  void InitAllocators();

  /// @brief Query the driver to get properties for this AIE agent.
  void GetAgentProperties();

  std::vector<const core::MemoryRegion *> regions_;
  std::function<void *(size_t size, size_t align,
                       core::MemoryRegion::AllocateFlags flags)>
      system_allocator_;


  std::function<void(void*)> system_deallocator_;

  const hsa_profile_t profile_ = HSA_PROFILE_BASE;
  const uint32_t min_aql_size_ = 0x40;
  const uint32_t max_aql_size_ = 0x40;
  const uint32_t max_queues_ = 1;

  /// @brief Number of columns in the AIE array.
  uint32_t num_cols_ = 0;
  /// @brief Number of rows of core tiles in the AIE array. Not all rows in a
  /// column are cores. Some can be memory or shim tiles.
  uint32_t num_core_rows_ = 0;
};

} // namespace AMD
} // namespace rocr

#endif // header guard
