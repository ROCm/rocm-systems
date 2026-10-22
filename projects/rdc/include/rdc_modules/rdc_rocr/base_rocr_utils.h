/*
Copyright (c) 2021 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef RDC_MODULES_RDC_ROCR_BASE_ROCR_UTILS_H_
#define RDC_MODULES_RDC_ROCR_BASE_ROCR_UTILS_H_

/// \file
/// Prototypes of utility functions that act on RdcRocrBase objects.

#include <string>

#include "hsa/hsa.h"
#include "rdc_modules/rdc_rocr/RdcRocrBase.h"
#include "rdc_modules/rdc_rocr/RocrHandles.h"
#include "rdc_modules/rdc_rocr/common.h"

namespace amd {
namespace rdc {

/// Open binary kernel object file and set all member data related to the
/// kernel. Assumes that input test already has the kernel file name,
/// agent name and kernel function specified. The caller owns the code object
/// reader and the executable, which must live until the dispatch is done:
/// destroying the executable would make the kernel object it reports invalid.
/// \param[in] test Test for which the kernel will be loaded.
/// \param[in] agent for which the kernel will be loaded .
/// \param[out] reader Holder that takes the code object reader
/// \param[out] executable Holder that takes the executable
/// A call that fails is this GPU's failure, named with its status, and a kernel that is not
/// installed for the agent is this sub-test's skip: both leave by throwing.
void LoadKernelFromObjFile(RdcRocrBase* test, hsa_agent_t* agent, CodeObjectReaderHolder* reader,
                           ExecutableHolder* executable);

/// The agent's name, read into a buffer of the 64 bytes ROCr writes and kept to at most those
/// 64 characters: ROCr clears the buffer and copies at most 63 characters, but a name that fills
/// it leaves no NUL, which a reader of the buffer must not run past.
/// \param[in] agent Agent whose name is read
/// \returns The agent's name, at most 64 characters long
std::string read_agent_name(hsa_agent_t agent);

/// The CPU agent a check runs its host side on, the first one ROCr lists. A host that lists none
/// is that GPU's failure, "No CPU agent.".
/// \returns The CPU agent
hsa_agent_t get_cpu_agent(void);

/// The packets a check's queue holds: what the check needs, at most what the agent allows. A
/// failed query is that GPU's failure, naming the call.
/// \param[in] agent Agent the queue will be created on
/// \returns The number of packets to ask for
uint32_t get_queue_packets(hsa_agent_t agent);

/// The kernel's argument segment, as the loaded symbol reports it, checked before anything is
/// allocated for it: the loader takes an alignment of 0 as 16, an alignment that is not a power
/// of two or is above what the pool promises cannot be met, and a segment shorter than the
/// arguments the check writes would leave the kernel reading bytes nobody wrote. Either is that
/// GPU's failure, so that the check allocates nothing and dispatches nothing.
/// \param[in] pool Pool the kernel argument buffer would be allocated from
/// \param[inout] alignment The kernel's argument segment alignment, 0 being taken as 16
/// \param[in] size The kernel's argument segment size
/// \param[in] arguments The bytes of arguments the check writes into that segment
void check_kernarg_segment(hsa_amd_memory_pool_t pool, uint32_t* alignment, uint32_t size,
                           size_t arguments);

/// The address the pool handed out for a kernel argument buffer, which must meet the kernel's
/// alignment: a misaligned one is that GPU's failure, with no dispatch.
/// \param[in] address The address the pool handed out
/// \param[in] alignment The alignment the kernel asks for
void check_kernarg_address(const void* address, uint32_t alignment);

void WriteAQLToQueueLoc(hsa_queue_t* queue, uint64_t indx, hsa_kernel_dispatch_packet_t* aql_pkt);
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
  __atomic_store_n(reinterpret_cast<uint32_t*>(queue_packet), header | (setup << 16),
                   __ATOMIC_RELEASE);
}

/// Get the library directory which is loaded by current process.
/// It will search /proc/self/maps for it.
/// return empty string if fail.
std::string get_lib_dir(const char* lib_name);

/// Get the app dir by looking at link of /proc/self/exe
std::string get_app_dir();

/// The kernel file for the agent, open. Each directory the module searches is tried in order
/// and the file in it opened, so that the file the reader is made from is the file that was
/// found; a candidate that does not open, or that is not a regular file, is passed over with a
/// debug line. A file that is installed nowhere is the sub-test's skip, "No <file> is installed
/// for <agent>."; a candidate that failed to open for a reason of its own, anything but a name
/// that is not there, is that GPU's failure, naming the errno.
/// \param[in] file_name The kernel file's name
/// \param[in] agent_name The agent the kernel is for, which names a directory and the skip
/// \returns The descriptor of the file that opened, which the caller owns
FileDescriptorHolder open_kernel_file(const std::string& file_name, const std::string& agent_name);

/// The deadline of each wait on a kernel or a memory copy, in milliseconds. A test replaces it
/// through the hook of RocrTestHooks.h; the module's own build has the constant.
uint64_t wait_deadline_ms(void);

/// The deadline as a failure names it: the seconds it stands for, or milliseconds where a test
/// hook set one under a second.
/// \param[in] deadline_ms The deadline in force, in milliseconds
/// \returns The deadline as the GPU's failure names it
std::string wait_deadline_text(uint64_t deadline_ms);

/// A queue for one of the checks, with an error callback and a status block of its own. ROCr
/// reports an error of a queue, an invalid packet or scratch it cannot allocate, only through
/// that callback, and suspends the queue, so that a dispatch on a queue without one never
/// completes. The block is made just before the queue and held by the holder that is released
/// after the queue's, so that an error ROCr reports late writes into memory that was not freed.
/// The queue holds the packets the check needs, at most the agent's largest queue: a failed
/// query of that maximum throws, naming hsa_agent_get_info() and its status, rather than
/// returning it as the status below.
/// \param[in] agent Agent the queue is created on
/// \param[out] status Holder that takes the queue's status block
/// \param[out] queue Holder that takes the queue
/// \returns What hsa_queue_create answered
hsa_status_t create_check_queue(hsa_agent_t agent, QueueStatusHolder* status, QueueHolder* queue);

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCR_BASE_ROCR_UTILS_H_
