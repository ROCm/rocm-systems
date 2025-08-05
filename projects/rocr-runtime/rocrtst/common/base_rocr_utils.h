/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
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

#ifndef ROCRTST_COMMON_BASE_ROCR_UTILS_H_
#define ROCRTST_COMMON_BASE_ROCR_UTILS_H_ 1

/// \file
/// Prototypes of utility functions that act on BaseRocR objects.

#include "common/base_rocr.h"
#include "hsa/hsa.h"

namespace rocrtst {

/// Locate kernel code object file and return path suitable for use with open().
std::string LocateKernelFile(std::string filename, hsa_agent_t agent);

/// Open binary kernel object file and set all member data related to the
/// kernel. Assumes that input test already has the kernel file name,
/// agent name and kernel function specifed
/// \param[in] test Test for which the kernel will be loaded.
/// \param[in] agent for which the kernel will be loaded .
/// \returns HSA_STATUS_SUCCESS if no errors
hsa_status_t LoadKernelFromObjFile(BaseRocR* test, hsa_agent_t* agent);

/// Do initialization tasks for HSA test program.
/// \param[in] test Test to initialize
/// \returns HSA_STATUS_SUCCESS if no errors
hsa_status_t InitAndSetupHSA(BaseRocR* test);

/// Find and set the cpu and gpu agent member variables. Also checks that
/// gpu agent meets test requirements (e.g., FULL profile vs. BASE profile).
hsa_status_t SetDefaultAgents(BaseRocR* test);

/// For the provided device agent, create an AQL queue
/// \param[in] device Device for which a queue is to be created
/// \param[out] queue Address to which created queue pointer will be written
/// \param[in] num_pkts Size of the queue to create
/// \param[in] do_profile [Optional] Specificy whether profiled queue should
///  be created
/// \returns  HSA_STATUS_SUCCESS if no errors encountered
hsa_status_t CreateQueue(hsa_agent_t device, hsa_queue_t** queue,
                         uint32_t num_pkts = 0);

/// This function sets some reasonable default values for an AQL packet.
/// Override any field as necessary after calling this function.
/// \param[in] test Test from which information to populate aql packet can
/// be drawn.
/// \param[inout] aql Caller provided pointer to aql packet that will be
/// populated
/// \returns Appropriate hsa_status_t
hsa_status_t InitializeAQLPacket(const BaseRocR* test,
                         hsa_kernel_dispatch_packet_t* aql);

/// This function writes all of the aql packet fields to the queue besides
/// "setup" and "header". This assumes all the aql fields have be set
/// appropriately.
/// \param[in] test Test containing the queue and aql packet to be written.
/// \returns Pointer to dispatch packet in queue that was written to
hsa_kernel_dispatch_packet_t* WriteAQLToQueue(BaseRocR* test, uint64_t *ind);

void WriteAQLToQueueLoc(hsa_queue_t *queue, uint64_t indx,
                                      hsa_kernel_dispatch_packet_t *aql_pkt);
/// This function writes the first 32 bits of an aql packet to the provided
/// aql packet. This function is meant to be called immediately before
/// ringing door_bell signal.
/// \param[in] header Value to be written to header field
/// \param[in] setup Value to be written to setup field
/// \param[in] queue_packet Start address of in queue memory of aql packet to
/// be written
/// \returns void
inline void AtomicSetPacketHeader(uint16_t header, uint16_t setup,
                                hsa_kernel_dispatch_packet_t* queue_packet) {
  __atomic_store_n(reinterpret_cast<uint32_t*>(queue_packet),
                                    header | (setup <<16), __ATOMIC_RELEASE);
}

/// Perform common operations to clean up after executing a test. Specifically,
/// hsa_shut_down() is called and environment variables that were changed are
/// reset to their original values.
/// \param[in] test Test for which clean up with be performed
/// \returns HSA_STATUS_SUCCESS if everything cleaned up ok, or appropriate HSA
///   error code otherwise.
hsa_status_t CommonCleanUp(BaseRocR* test);

///  Check to see if target machine has the necessary profile to run the
///  provided test.
///  \param[1] test The test that specifies the required profile.
bool CheckProfile(BaseRocR const* test);

/// Allocate memory from the kernel args pool and write the provided argument
/// data to the kernel arg memory. Assumes kern_arg memory pool has been
/// assigned. The amount of memory allocated will actually be \p arg_size
/// plus the alignment required by the kernel arguments. The argument will
/// be written with the proper alignment within the allocated buffer.
/// \p test kernarg_buffer() will point to the allocated buffer, and it should
/// be freed when the kernel is no longer being used.
/// \param test Test from which to find kern_arg pool to write arguments
/// \param args pointer to block of data containing kernel arguments to be
///  written. Arguments are assumed to be of the correct placement, length,
///  and with any padding that is expected by the OpenCL kernel
/// \param arg_size Size of the kernel arg data (including padding) to be
/// written
/// \returns HSA_STATUS_SUCCESS if no errors
hsa_status_t AllocAndSetKernArgs(BaseRocR* test, void* args,
                                 size_t arg_size);

/// Verify that the machine running the test has the required profile.
/// This function will verify that the execution machine meets any specific
/// test requirement for a profile (HSA_PROFILE_BASE or HSA_PROFILE_FULL).
/// \param[in] test Test that provides profile requirements.
/// \returns bool
///          - true Machine meets test requirements
///          - false Machine does not meet test requirements
bool CheckProfileAndInform(BaseRocR* test);

/// This function will set the cpu and gpu memory pools to the type used in
/// many applications.
/// \param[in] test Test that provides profile requirements.
/// \returns HSA_STATUS_SUCCESS if everything cleaned up ok, or appropriate HSA
///   error code otherwise.
hsa_status_t SetPoolsTypical(BaseRocR* test);

/// Work-around for hsa_amd_memory_fill, which is currently broken.
/// \param[in] ptr Pointer to start of memory location to be filled
/// \param[in] value Value to write to each byte of input buffer
/// \param[in] count Size of buffer to fill
/// \param[in] dst_ag Agent owning the buffer to be filled
/// \param[in] src_ag Agent wanting to do the fill
/// \param[in] test Test that has handles to cpu and gpu agents that can own
/// either source or destination of fill
/// \returns HSA_STATUS_OK if not errors
hsa_status_t hsa_memory_fill_workaround_gen(void* ptr, uint32_t value,
      size_t count, hsa_agent_t dst_ag, hsa_agent_t src_ag, BaseRocR* test);
}  // namespace rocrtst
#endif  // ROCRTST_COMMON_BASE_ROCR_UTILS_H_
