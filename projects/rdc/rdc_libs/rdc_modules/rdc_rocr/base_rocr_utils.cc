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

#include "rdc_modules/rdc_rocr/base_rocr_utils.h"

#include <assert.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <libgen.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "hsa/hsa.h"
#include "rdc_lib/RdcLogger.h"
#include "rdc_lib/rdc_common.h"
#if RDC_ROCR_TEST_HOOKS
#include "rdc_modules/rdc_rocr/RocrTestHooks.h"
#endif

namespace amd {
namespace rdc {

// The deadline of each wait on a kernel or a memory copy, which a test replaces through the hook
// at the end of this file.
static const uint64_t kWaitDeadlineMs = 10 * 1000;
static uint64_t s_wait_deadline_ms = kWaitDeadlineMs;

uint64_t wait_deadline_ms(void) { return s_wait_deadline_ms; }

// The packets a check's queue holds. One dispatch at a time needs very few, where the largest
// queue an agent allows is 0x20000 packets, 8 MiB of ring, on ROCr.
static const uint32_t kCheckQueuePackets = 128;

// The agent's name, read into a buffer of the 64 bytes ROCr writes and kept to at most those
// 64 characters: ROCr clears the buffer and copies at most 63 characters of the name
// (amd_gpu_agent.cpp), but a name that fills the buffer leaves no NUL in it.
std::string read_agent_name(hsa_agent_t agent) {
  char name[64] = {};
  const hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
  throw_if_error(err, "Fail to read the name of the GPU's agent.");
  return std::string(name, strnlen(name, sizeof(name)));
}

// The CPU agent a check runs its host side on, the first one ROCr lists.
hsa_agent_t get_cpu_agent(void) {
  std::vector<hsa_agent_t> cpus;
  const hsa_status_t err = hsa_iterate_agents(IterateCPUAgents, &cpus);
  throw_if_error(err, "Fail to iterate agents.");
  if (cpus.empty()) {
    throw_failure("No CPU agent.");
  }
  return cpus[0];
}

// The packets a check's queue holds: what the check needs, at most what the agent allows. ROCr
// reports a maximum of 0x20000 and a minimum of 64 for every GPU and refuses a size below that
// minimum, so this is 128 there and never below it.
uint32_t get_queue_packets(hsa_agent_t agent) {
  uint32_t maximum = 0;
  const hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_QUEUE_MAX_SIZE, &maximum);
  throw_if_error(err,
                 "Fail to call hsa_agent_get_info() for the size of the agent's largest queue.");
  return std::min<uint32_t>(kCheckQueuePackets, maximum);
}

// The kernel's argument segment, checked against the pool it would come from before anything is
// allocated for it, so that a kernel the check cannot serve costs no allocation and no dispatch.
void check_kernarg_segment(hsa_amd_memory_pool_t pool, uint32_t* alignment, uint32_t size,
                           size_t arguments) {
  if (*alignment == 0) {
    *alignment = 16;  // what the loader gives a kernel that reports none
  }

  size_t pool_alignment = 0;
  const hsa_status_t err = hsa_amd_memory_pool_get_info(
      pool, HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_ALIGNMENT, &pool_alignment);
  throw_if_error(err, "Fail to read the alignment of the kernel argument pool.");

  // ROCr hands out whole pages from a page-aligned block, so the pool's own alignment is all a
  // kernel can be promised, and a value that is not a power of two is not an alignment at all.
  const bool power_of_two = (*alignment & (*alignment - 1)) == 0;
  if (!power_of_two || *alignment > pool_alignment) {
    throw_failure("The kernel asks for an argument segment aligned to " +
                  std::to_string(*alignment) + " bytes, which the check cannot give it.");
  }

  if (size < arguments) {
    throw_failure("The kernel's argument segment is " + std::to_string(size) +
                  " bytes, shorter than its arguments.");
  }
}

// The address the pool handed out for a kernel argument buffer.
void check_kernarg_address(const void* address, uint32_t alignment) {
  if ((reinterpret_cast<uintptr_t>(address) & (alignment - 1)) != 0) {
    throw_failure("The kernel argument buffer is not aligned to " + std::to_string(alignment) +
                  " bytes.");
  }
}

// Load the specified kernel code from the specified file, inspect and fill
// in RdcRocrBase member variables related to the kernel and executable.
// Required Input RdcRocrBase member variables:
// - kernel_file_name()
// - kernel_name()
//
// Written RdcRocrBase member variables:
//  -kernel_object()
//  -private_segment_size()
//  -group_segment_size()
//  -kernarg_size()
//  -kernarg_align()
void LoadKernelFromObjFile(RdcRocrBase* test, hsa_agent_t* agent, CodeObjectReaderHolder* reader,
                           ExecutableHolder* executable) {
  hsa_status_t err;
  hsa_code_object_reader_t code_obj_rdr = {0};
  hsa_executable_t executable_handle = {0};

  assert(test != nullptr);
  assert(agent != nullptr);
  assert(reader != nullptr);
  assert(executable != nullptr);

  // if agent name is not set, then set the agent name
  if (!test->get_agent_name().size()) {
    test->set_agent_name(read_agent_name(*agent));
  }

  std::string kern_name = test->kernel_name();

  // The descriptor the search opened is the one the reader is made from, so that the file that is
  // loaded is the file that was found. It is held so that every path out of this function closes
  // it, and it is closed as soon as the reader has been made, whatever that call returned.
  FileDescriptorHolder kernel_file =
      open_kernel_file(test->kernel_file_name(), test->get_agent_name());

  err = hsa_code_object_reader_create_from_file(kernel_file.get(), &code_obj_rdr);
  kernel_file.reset();
  if (err == HSA_STATUS_SUCCESS) {
    reader->reset(code_obj_rdr);
  }
  throw_if_error(err, "Fail to read the kernel file.");

  err = hsa_executable_create_alt(HSA_PROFILE_FULL, HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT, NULL,
                                  &executable_handle);
  if (err == HSA_STATUS_SUCCESS) {
    executable->reset(executable_handle);
  }
  throw_if_error(err, "Fail to create the kernel's executable.");
  err = hsa_executable_load_agent_code_object(executable->get(), *agent, reader->get(), NULL, NULL);
  throw_if_error(err, "Fail to load the kernel.");
  err = hsa_executable_freeze(executable->get(), NULL);
  throw_if_error(err, "Fail to freeze the kernel's executable.");

  hsa_executable_symbol_t kern_sym;
  err = hsa_executable_get_symbol(executable->get(), NULL, (kern_name + ".kd").c_str(), *agent, 0,
                                  &kern_sym);
  throw_if_error(err, "Fail to find the kernel in its executable.");

  uint64_t codeHandle;
  err = hsa_executable_symbol_get_info(kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_OBJECT,
                                       &codeHandle);
  throw_if_error(err, "Fail to read the kernel's code handle.");
  test->set_kernel_object(codeHandle);

  uint32_t val;
  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_PRIVATE_SEGMENT_SIZE, &val);
  throw_if_error(err, "Fail to read the kernel's private segment size.");
  test->set_private_segment_size(val);

  err = hsa_executable_symbol_get_info(kern_sym,
                                       HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_GROUP_SEGMENT_SIZE, &val);
  throw_if_error(err, "Fail to read the kernel's group segment size.");
  test->set_group_segment_size(val);

  // Remaining queries only supported on code object v3.
  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_SIZE, &val);
  throw_if_error(err, "Fail to read the kernel's argument segment size.");
  test->set_kernarg_size(val);

  err = hsa_executable_symbol_get_info(
      kern_sym, HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_KERNARG_SEGMENT_ALIGNMENT, &val);
  throw_if_error(err, "Fail to read the kernel's argument segment alignment.");
  test->set_kernarg_align(val);
}

void WriteAQLToQueueLoc(hsa_queue_t* queue, uint64_t indx, hsa_kernel_dispatch_packet_t* aql_pkt) {
  assert(queue);
  assert(aql_pkt);

  void* queue_base = queue->base_address;
  const uint32_t queue_mask = queue->size - 1;
  hsa_kernel_dispatch_packet_t* queue_aql_packet;

  queue_aql_packet =
      &(reinterpret_cast<hsa_kernel_dispatch_packet_t*>(queue_base))[indx & queue_mask];

  queue_aql_packet->workgroup_size_x = aql_pkt->workgroup_size_x;
  queue_aql_packet->workgroup_size_y = aql_pkt->workgroup_size_y;
  queue_aql_packet->workgroup_size_z = aql_pkt->workgroup_size_z;
  queue_aql_packet->grid_size_x = aql_pkt->grid_size_x;
  queue_aql_packet->grid_size_y = aql_pkt->grid_size_y;
  queue_aql_packet->grid_size_z = aql_pkt->grid_size_z;
  queue_aql_packet->private_segment_size = aql_pkt->private_segment_size;
  queue_aql_packet->group_segment_size = aql_pkt->group_segment_size;
  queue_aql_packet->kernel_object = aql_pkt->kernel_object;
  queue_aql_packet->kernarg_address = aql_pkt->kernarg_address;
  queue_aql_packet->completion_signal = aql_pkt->completion_signal;

  // reserved0 and reserved2 must be 0 (hsa.h), and ROCr leaves the packet contents of a new ring
  // uninitialized, so they are written from the staging packet, which starts zeroed.
  queue_aql_packet->reserved0 = aql_pkt->reserved0;
  queue_aql_packet->reserved2 = aql_pkt->reserved2;
}

std::string get_lib_dir(const char* lib_name) {
  std::string result;
  char line[1024 * 8];

  // The map is opened with O_CLOEXEC ("e"), so that a program that execs while it is open
  // does not hand the descriptor on, and the holder closes it on every path out of this
  // function: the string work below can throw.
  const std::unique_ptr<FILE, int (*)(FILE*)> file(fopen("/proc/self/maps", "re"), &fclose);
  if (!file) return result;
  std::string lib_path = "/";
  lib_path += lib_name;
  // 7f4eacb46000 r-xp 00000 08:01 17183106 /lib/x86_64-linux-gnu/libc-2.27.so
  while (fgets(line, sizeof(line), file.get())) {
    char* end = strstr(line, lib_path.c_str());
    if (end != NULL) {
      char* start = end;
      while (start > line) {
        if (isspace(static_cast<unsigned char>(*start))) {
          start++;
          break;
        }
        start--;
      }
      result = std::string(start, end - start);
      break;
    }
  }

  return result;
}

std::string get_app_dir() {
  char buf[1024 * 8];
  int ret = readlink("/proc/self/exe", buf, 1024 * 8);
  if ((ret != -1) && ret < (1024 * 8 - 1)) {
    buf[ret] = '\0';
    return dirname(buf);
  }
  return "";
}

#if RDC_ROCR_TEST_HOOKS
// The directories a test told the module to search, which take the place of the list below. The
// module's own build does not compile the hook that fills them.
static std::vector<std::string>& test_kernel_directories(void) {
  static std::vector<std::string> directories;
  return directories;
}
#endif

// One place a kernel file may be: the directory, which the module does not always know, and the
// path of the file within it.
struct KernelCandidate {
  std::string directory;
  std::string suffix;
  const char* unknown;  // what names the directory in a log line when the module has none
};

// The places the module looks for a kernel file, in the order it tries them: the program's own
// directory, the module's directory, the two trees of kernels under it and the build tree's
// kernels directory, which is what a module run from a build tree loads from. The working
// directory is not among them: it is usually root's when the daemon runs, and a file left there
// would be loaded and run on the GPU. An entry whose directory the module does not know is
// passed over, so that nothing is looked up at the filesystem root.
static std::vector<KernelCandidate> kernel_candidates(const std::string& file_name,
                                                      const std::string& agent_name) {
  std::vector<KernelCandidate> candidates;

#if RDC_ROCR_TEST_HOOKS
  if (!test_kernel_directories().empty()) {
    for (const std::string& directory : test_kernel_directories()) {
      candidates.push_back({directory, file_name, "<a directory a test named>"});
    }
    return candidates;
  }
#endif

  const std::string app_dir = get_app_dir();
  const std::string lib_dir = get_lib_dir("librdc_rocr.so");
  const std::string of_agent = agent_name + "/" + file_name;
  const char* const module_dir = "<the module's directory>";

  candidates.push_back({app_dir, file_name, "<the program's directory>"});
  candidates.push_back({lib_dir, file_name, module_dir});
  candidates.push_back({lib_dir, "rdc/hsaco/" + of_agent, module_dir});
  candidates.push_back({lib_dir, "hsaco/" + of_agent, module_dir});
  candidates.push_back(
      {lib_dir, "../../rdc_libs/rdc_modules/kernels/hsaco/" + of_agent, module_dir});
  return candidates;
}

FileDescriptorHolder open_kernel_file(const std::string& file_name, const std::string& agent_name) {
  // The first reason a candidate gave that is not "there is no such name": a kernel file that is
  // there but cannot be read is a failure to report, where a name that is not there is not.
  int first_errno = 0;

  for (const KernelCandidate& candidate : kernel_candidates(file_name, agent_name)) {
    if (candidate.directory.empty()) {
      RDC_LOG(RDC_DEBUG, "Kernel file candidate " << candidate.unknown << "/" << candidate.suffix
                                                  << ": its directory is not known.");
      continue;
    }

    const std::string path = candidate.directory + "/" + candidate.suffix;
    // O_NONBLOCK so that a FIFO or a device node of that name cannot hold the check up, which it
    // does not change for a regular file, and O_CLOEXEC so that no process the daemon starts is
    // given the descriptor.
    FileDescriptorHolder file(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NONBLOCK));
    if (file.empty()) {
      const int reason = errno;
      if (first_errno == 0 && reason != ENOENT && reason != ENOTDIR) {
        first_errno = reason;
      }
      RDC_LOG(RDC_DEBUG, "Kernel file candidate " << path << ": errno " << reason << ".");
      continue;
    }

    struct stat about = {};
    if (fstat(file.get(), &about) != 0) {
      const int reason = errno;
      if (first_errno == 0 && reason != ENOENT && reason != ENOTDIR) {
        first_errno = reason;
      }
      RDC_LOG(RDC_DEBUG, "Kernel file candidate " << path << ": fstat errno " << reason << ".");
      continue;
    }
    if (!S_ISREG(about.st_mode)) {
      RDC_LOG(RDC_DEBUG, "Kernel file candidate " << path << ": not a regular file.");
      continue;
    }

    RDC_LOG(RDC_DEBUG, "Kernel file candidate " << path << ": opened.");
    return file;
  }

  if (first_errno != 0) {
    throw_failure("Fail to open " + file_name + " for " + agent_name + ", errno " +
                  std::to_string(first_errno) + ".");
  }

  // A kernel that is not installed for this agent is what the sub-test reports, so the log keeps
  // it at debug level: an installation without this agent's kernels is not an error of the run.
  RDC_LOG(RDC_DEBUG, "No " << file_name << " is installed for " << agent_name << ".");
  throw_if_skip("No " + file_name + " is installed for " + agent_name + ".");
  return FileDescriptorHolder();
}

std::string wait_deadline_text(uint64_t deadline_ms) {
  if (deadline_ms < 1000) {
    return std::to_string(deadline_ms) + " ms";
  }
  return std::to_string(deadline_ms / 1000) + " s";
}

namespace {

// The record of the GPUs whose work a check could not stop: one slot per GPU, each claimed once
// and never cleared, so that the checks stay disabled for the life of the process. Eight covers
// the runs that can be in flight when the first of them abandons its work.
constexpr size_t kAbandonedSlots = 8;
constexpr uint64_t kFreeSlot = UINT64_MAX;
std::atomic<uint64_t> s_abandoned_gpus[kAbandonedSlots] = {{kFreeSlot}, {kFreeSlot}, {kFreeSlot},
                                                           {kFreeSlot}, {kFreeSlot}, {kFreeSlot},
                                                           {kFreeSlot}, {kFreeSlot}};

// The queue's error, which ROCr reports from a thread of its own. The callback only stores the
// status, which each wait and the check before each doorbell read, and it never throws.
void queue_error_reported(hsa_status_t status, hsa_queue_t*, void* data) noexcept {
  if (data == nullptr) {
    return;
  }
  static_cast<QueueStatus*>(data)->store(status, std::memory_order_release);
}

// The module keeps itself mapped for the life of the process: the callback above and the record
// must stay where they are after the loader has closed the module, which it does when a program
// that embeds RDC stops it. dladdr names the file this code is in and a dlopen of that file with
// RTLD_NOLOAD takes a reference that is never closed. Where this code is part of an executable,
// which is never unmapped, that dlopen answers nothing and the line below says so.
void pin_the_module(uint32_t entity) {
  Dl_info about = {};
  if (::dladdr(reinterpret_cast<const void*>(&pin_the_module), &about) == 0 ||
      about.dli_fname == nullptr || about.dli_fname[0] == '\0') {
    RDC_LOG(RDC_ERROR, "Cannot pin the module: dladdr gave no file; the record of GPU "
                           << entity << " lasts while the module stays mapped.");
    return;
  }

  // An error another call left pending is cleared first, so that what is read below is this
  // dlopen's own, and that text is used only where there is one: a null pointer streamed into
  // std::cout would leave it bad and silence every later line of the process.
  (void)::dlerror();
  if (::dlopen(about.dli_fname, RTLD_LAZY | RTLD_NOLOAD | RTLD_NODELETE) != nullptr) {
    return;
  }
  const char* const reported = ::dlerror();
  std::string line = "Cannot pin " + std::string(about.dli_fname) +
                     " with dlopen; the record of GPU " + std::to_string(entity) +
                     " lasts while the module stays mapped.";
  if (reported != nullptr) {
    line += " ";
    line += reported;
  }
  RDC_LOG(RDC_ERROR, line);
}

}  // namespace

void abandon_gpu(uint32_t entity) {
  try {
    // A GPU that is in the record already spends no second slot, so that two runs which abandon
    // their work on the same GPU leave one entry and the texts name it once.
    const uint64_t recorded = entity;
    for (std::atomic<uint64_t>& slot : s_abandoned_gpus) {
      if (slot.load(std::memory_order_acquire) == recorded) {
        return;
      }
    }

    // The module is pinned before the claim below publishes the entity, so that nothing can read
    // the record while the code that holds it could still be unmapped; a GPU the scan above
    // found is already covered by the pin the run that recorded it took.
    pin_the_module(entity);

    for (std::atomic<uint64_t>& slot : s_abandoned_gpus) {
      uint64_t free_slot = kFreeSlot;
      if (slot.compare_exchange_strong(free_slot, recorded, std::memory_order_release,
                                       std::memory_order_acquire)) {
        return;
      }
      if (free_slot == recorded) {
        // Another run recorded this GPU while this one was looking for a slot.
        return;
      }
    }
    RDC_LOG(RDC_ERROR, "The ROCr checks could not stop their work on GPU "
                           << entity
                           << " and have no slot left to record it; the checks are disabled for "
                              "the GPUs that are recorded.");
  } catch (...) {
    // This runs on a destructor's path as well: a line that cannot be put together is dropped
    // rather than ending the process.
  }
}

std::vector<uint32_t> abandoned_gpus(void) {
  std::vector<uint32_t> entities;
  for (const std::atomic<uint64_t>& slot : s_abandoned_gpus) {
    const uint64_t recorded = slot.load(std::memory_order_acquire);
    if (recorded != kFreeSlot) {
      entities.push_back(static_cast<uint32_t>(recorded));
    }
  }
  return entities;
}

hsa_status_t create_check_queue(hsa_agent_t agent, QueueStatusHolder* status, QueueHolder* queue) {
  // The block is this queue's own and is made before it, so that the callback has somewhere to
  // write from the moment ROCr can call it.
  status->reset(new QueueStatus(HSA_STATUS_SUCCESS));

  hsa_queue_t* made = nullptr;
  const hsa_status_t err = hsa_queue_create(agent, get_queue_packets(agent), HSA_QUEUE_TYPE_MULTI,
                                            queue_error_reported, status->get(), 0, 0, &made);
  if (err == HSA_STATUS_SUCCESS) {
    queue->reset(made);
  }
  return err;
}

}  // namespace rdc
}  // namespace amd

#if RDC_ROCR_TEST_HOOKS
// The hooks of RocrTestHooks.h, which only the unit tests' build of these sources defines. A test
// calls them before a run; nothing else reads what they write.
void rdc_rocr_test_set_kernel_directories(const char* const* directories, size_t count) {
  std::vector<std::string>& named = amd::rdc::test_kernel_directories();
  named.clear();
  if (directories == nullptr) {
    return;
  }
  for (size_t i = 0; i < count; i++) {
    if (directories[i] != nullptr) {
      named.push_back(directories[i]);
    }
  }
}

uint64_t rdc_rocr_test_set_wait_deadline_ms(uint64_t deadline_ms) {
  const uint64_t replaced = amd::rdc::s_wait_deadline_ms;
  amd::rdc::s_wait_deadline_ms = deadline_ms;
  return replaced;
}
#endif
