/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef TESTS_UNIT_FAKE_HSA_H_
#define TESTS_UNIT_FAKE_HSA_H_

#include <stddef.h>
#include <stdint.h>

// The controls of the link-time HSA fake in fake_hsa.cc. They are declared
// inside an extern "C" block so that a test which loads the fake as a shared
// library, as test_rocr_pin.cc does, reaches them with dlsym.
//
// What the fake answers, unless a control below says otherwise:
// - one CPU agent (node 0, "CPU") and two GPU agents (nodes 1 and 2, "gfx1030"
//   and "gfx1100"), in that order;
// - for the CPU agent a standard pool, a kernel argument pool and a
//   coarse-grained pool, and for each GPU agent a coarse-grained pool, each of
//   64 KiB with a maximum allocation of 64 KiB, a granule and an allocation
//   alignment of 4096, and allocation allowed;
// - every agent may access every pool;
// - a queue maximum of 1024 packets, a timestamp frequency of 25 MHz, and for
//   every kernel an argument segment of 80 bytes aligned to 16, a private
//   segment of 16 bytes and a group segment of 64 bytes;
// - the entity ids a check is given map to the node ids 1, 2, 3 and so on.
//
// Every allocation sits in its own mapping, filled with 0x5A, followed by a
// 4 KiB redzone that the free checks and then a page that cannot be read, and
// stays mapped after it is freed, so that a second free is counted rather than a
// crash. Every object the fake issues (queues with their rings, signals,
// executables and code object readers) stays in a registry for the life of the
// process, which is what the destroy log, the live references and the latch
// read, and what keeps LeakSanitizer quiet about an abandoned check's objects.
// All of the state is guarded by one mutex; the gates wait on a condition
// variable under it.
//
// hsa_memory_free() does the same work as hsa_amd_memory_pool_free(), as it does
// in ROCr, so a block freed through either is counted once against that block;
// the call log and rdc_fake_hsa_call_count() record the name the caller used.
extern "C" {

// The kernel hook's behaviours, for rdc_fake_hsa_set_kernel_mode().
enum rdc_fake_kernel_mode_t {
  RDC_FAKE_KERNEL_NORMAL = 0,   // write what the kernel writes, and complete
  RDC_FAKE_KERNEL_NOTHING = 1,  // write nothing, and complete
  RDC_FAKE_KERNEL_NEVER = 2,    // write nothing, and never complete
  RDC_FAKE_KERNEL_SHIFTED = 3,  // the memory access kernel's read-back shifted by one element
  RDC_FAKE_KERNEL_OUTPUT = 4    // the search outputs of rdc_fake_hsa_set_search_output()
};

// The asynchronous copy's behaviours, for rdc_fake_hsa_set_copy_mode().
enum rdc_fake_copy_mode_t {
  RDC_FAKE_COPY_NORMAL = 0,  // copy the bytes, and complete
  RDC_FAKE_COPY_NEVER = 1,   // copy nothing, and never complete
  RDC_FAKE_COPY_FAILED = 2   // the completion signal set to -1, as ROCr does for a failed copy
};

// Puts every part of the fake back to the state described above. A case calls it
// first; the registry, the mappings and the counters go with it.
void rdc_fake_hsa_reset(void);

// --- the agents ---------------------------------------------------------------

// Adds an agent after the agents already listed and returns its index.
uint32_t rdc_fake_hsa_add_agent(int is_gpu, uint32_t node_id, const char* name);

// Drops every agent and every pool, so that a case lists its own.
void rdc_fake_hsa_clear_agents(void);

uint32_t rdc_fake_hsa_agent_count(void);

// The handle the fake issues for an agent or a pool, which a case needs to call
// an entry point itself. Any index works, so that a case can build a handle the
// fake never issued out of one past the agents or the pools it lists.
uint64_t rdc_fake_hsa_agent_handle(uint32_t agent);
uint64_t rdc_fake_hsa_pool_handle(uint32_t pool);

// Writes the agent's name as ROCr writes it, NUL-padded to 64 bytes, or, when
// terminate is 0, fills all 64 bytes with the name repeated and leaves no NUL.
void rdc_fake_hsa_set_agent_name(uint32_t agent, const char* name, int terminate);

// The HSA_AGENT_INFO_QUEUE_MAX_SIZE the agent reports.
void rdc_fake_hsa_set_agent_queue_max(uint32_t agent, uint32_t packets);

// The HSA_AGENT_INFO_PROFILE the agent reports (HSA_PROFILE_BASE or _FULL).
void rdc_fake_hsa_set_agent_profile(uint32_t agent, uint32_t profile);

// Fails one attribute of hsa_agent_get_info with the given status. An agent of
// UINT32_MAX fails that attribute for every agent.
void rdc_fake_hsa_fail_agent_info(uint32_t agent, uint32_t attribute, int32_t status);

// --- the pools ----------------------------------------------------------------

// Adds a pool to an agent, after the pools it already has, and returns the
// pool's index, which every other pool control takes.
uint32_t rdc_fake_hsa_add_pool(uint32_t agent, uint32_t segment, uint32_t global_flags,
                               uint64_t size, uint64_t max_size, int alloc_allowed,
                               uint64_t granule, uint64_t alignment, int accessible_by_all);

// Drops the agent's pools.
void rdc_fake_hsa_clear_pools(uint32_t agent);

uint32_t rdc_fake_hsa_pool_count(uint32_t agent);

// The index of the agent's nth pool, or UINT32_MAX when it has no such pool.
uint32_t rdc_fake_hsa_pool_of_agent(uint32_t agent, uint32_t n);

// HSA_AMD_MEMORY_POOL_INFO_SIZE and _ALLOC_MAX_SIZE, which are distinct: the
// size is what a check divides, the maximum what an allocation is refused above.
void rdc_fake_hsa_set_pool_size(uint32_t pool, uint64_t size);
void rdc_fake_hsa_set_pool_max(uint32_t pool, uint64_t max_size);
void rdc_fake_hsa_set_pool_flags(uint32_t pool, uint32_t global_flags);
void rdc_fake_hsa_set_pool_alloc_allowed(uint32_t pool, int allowed);
void rdc_fake_hsa_set_pool_granule(uint32_t pool, uint64_t granule, uint64_t alignment);

// The access the agent has to the pool (an hsa_amd_memory_pool_access_t).
void rdc_fake_hsa_set_pool_access(uint32_t agent, uint32_t pool, uint32_t access);

// Fails one attribute of hsa_amd_memory_pool_get_info. A pool of UINT32_MAX
// fails that attribute for every pool.
void rdc_fake_hsa_fail_pool_info(uint32_t pool, uint32_t attribute, int32_t status);

// How many access values the fake was asked to fill, and how many pool handles
// it was given that it never issued.
uint32_t rdc_fake_hsa_access_reads(void);
uint32_t rdc_fake_hsa_foreign_pools(void);

// --- the allocations ----------------------------------------------------------

uint32_t rdc_fake_hsa_alloc_count(void);
const void* rdc_fake_hsa_alloc_address(uint32_t block);
uint64_t rdc_fake_hsa_alloc_size(uint32_t block);
uint32_t rdc_fake_hsa_alloc_pool(uint32_t block);
uint32_t rdc_fake_hsa_alloc_frees(uint32_t block);

// The blocks never freed, the frees of a block already freed, the frees of a
// pointer the fake never issued, and the blocks whose redzone a free found
// written over.
uint32_t rdc_fake_hsa_live_blocks(void);
uint32_t rdc_fake_hsa_double_frees(void);
uint32_t rdc_fake_hsa_unissued_frees(void);
uint32_t rdc_fake_hsa_redzone_failures(void);

// Returns the next allocation that many bytes past the start of its page, so
// that a check which takes the address as it is dispatches an unaligned one.
void rdc_fake_hsa_set_kernarg_offset(uint32_t bytes);

// Maps the second page of each later allocation of two pages or more onto its
// first, so that a write to the second page is read back from the first.
void rdc_fake_hsa_set_alias_pages(int on);

// --- the queues ---------------------------------------------------------------

uint32_t rdc_fake_hsa_queue_count(void);
uint32_t rdc_fake_hsa_queue_agent(uint32_t queue);
uint32_t rdc_fake_hsa_queue_size(uint32_t queue);
uint32_t rdc_fake_hsa_queue_private_segment(uint32_t queue);
uint32_t rdc_fake_hsa_queue_group_segment(uint32_t queue);
int rdc_fake_hsa_queue_has_callback(uint32_t queue);
int rdc_fake_hsa_queue_destroyed(uint32_t queue);

// The data the queue was created with, which a check that keeps its error status
// in a block of its own passes, so that a case can check where that block is.
const void* rdc_fake_hsa_queue_error_data(uint32_t queue);

// Raises a queue error from a thread of the fake's: it calls the queue's error
// callback with the data the queue was created with, or, when the queue has
// none, records the error and stops the queue, as ROCr suspends it. The queue's
// hsa_queue_destroy joins that thread.
void rdc_fake_hsa_raise_queue_error(uint32_t queue, int32_t status);

// The errors raised on a queue that has no callback.
uint32_t rdc_fake_hsa_queue_errors_recorded(void);

// --- the signals and the waits --------------------------------------------------

uint32_t rdc_fake_hsa_signal_count(void);
int64_t rdc_fake_hsa_signal_value(uint32_t signal);

// The waits made on one signal, which a case that counts the waits of a dispatch
// reads with the dispatch's completion signal below.
uint32_t rdc_fake_hsa_signal_waits(uint32_t signal);

// The value the kernel hook stores in a dispatch's completion signal, 0 by
// default; -1 is a value ROCr does not give a dispatch.
void rdc_fake_hsa_set_dispatch_signal_value(int64_t value);

// Completes a dispatch, or an asynchronous copy, inside the nth wait on its
// completion signal rather than at the doorbell or at the copy itself, so that
// the waits before it return 1, as expired hints do. A count of 0 takes the
// knob off.
void rdc_fake_hsa_set_completion_wait(uint32_t nth);

// The waits made, the distinct hints they carried and each of those hints, in
// the order in which they were first used.
uint32_t rdc_fake_hsa_wait_count(void);
uint32_t rdc_fake_hsa_wait_hint_count(void);
uint64_t rdc_fake_hsa_wait_hint(uint32_t n);

// The distinct wait states the waits carried, an hsa_wait_state_t each, in the
// order in which they were first used, so that a case can say that every wait
// the module makes blocks.
uint32_t rdc_fake_hsa_wait_state_count(void);
int rdc_fake_hsa_wait_state(uint32_t n);

// --- the kernels and the dispatches ---------------------------------------------

// What the executable reports for a kernel symbol; a null or empty symbol sets
// it for every kernel the fake has no entry for.
void rdc_fake_hsa_set_kernel_info(const char* symbol, uint32_t kernarg_size, uint32_t kernarg_align,
                                  uint32_t private_segment, uint32_t group_segment);

// What the kernel hook does at the doorbell, an rdc_fake_kernel_mode_t.
void rdc_fake_hsa_set_kernel_mode(int mode);

// The outputs RDC_FAKE_KERNEL_OUTPUT writes into the search kernel's output
// buffer: its element 0 and its element 3.
void rdc_fake_hsa_set_search_output(uint32_t element0, uint32_t element3);

uint32_t rdc_fake_hsa_dispatch_count(void);
uint32_t rdc_fake_hsa_dispatch_queue(uint32_t dispatch);

// The dispatch's completion signal, or UINT32_MAX when its packet named none.
uint32_t rdc_fake_hsa_dispatch_signal(uint32_t dispatch);
uint64_t rdc_fake_hsa_dispatch_grid(uint32_t dispatch);
uint32_t rdc_fake_hsa_dispatch_workgroup(uint32_t dispatch);
uint32_t rdc_fake_hsa_dispatch_private_segment(uint32_t dispatch);
uint32_t rdc_fake_hsa_dispatch_group_segment(uint32_t dispatch);
uint64_t rdc_fake_hsa_dispatch_kernarg_address(uint32_t dispatch);
uint32_t rdc_fake_hsa_dispatch_reserved0(uint32_t dispatch);
uint64_t rdc_fake_hsa_dispatch_reserved2(uint32_t dispatch);
uint32_t rdc_fake_hsa_dispatch_header(uint32_t dispatch);
uint32_t rdc_fake_hsa_dispatch_setup(uint32_t dispatch);

// The bytes of the dispatch's argument segment the hook recorded: the kernel's
// argument segment, or what was left in the block from the argument address when
// that is less, which is a violation of its own.
uint32_t rdc_fake_hsa_dispatch_kernarg_size(uint32_t dispatch);
const unsigned char* rdc_fake_hsa_dispatch_kernarg(uint32_t dispatch);

// The dispatches whose grid did not fit the buffers its arguments point to, or
// whose argument block was shorter than the kernel's argument segment, with the
// last one's text.
uint32_t rdc_fake_hsa_grid_violations(void);
const char* rdc_fake_hsa_last_grid_violation(void);

// --- the copies, the executables and the readers ----------------------------------

// What an asynchronous copy does, an rdc_fake_copy_mode_t.
void rdc_fake_hsa_set_copy_mode(int mode);
uint32_t rdc_fake_hsa_copy_count(void);

uint32_t rdc_fake_hsa_executable_count(void);
uint32_t rdc_fake_hsa_reader_count(void);

// The identity of the file behind the descriptor the reader was made from.
uint64_t rdc_fake_hsa_reader_device(uint32_t reader);
uint64_t rdc_fake_hsa_reader_inode(uint32_t reader);

// The destroys and the frees in the order they happened, each entry a kind and
// an index, "queue 0" or "block 3" for one of them.
uint32_t rdc_fake_hsa_destroy_log_size(void);
const char* rdc_fake_hsa_destroy_log_entry(uint32_t entry);

// --- the run and its references ----------------------------------------------------

// The references hsa_init() holds, the loads (an hsa_init() that found no live
// reference and succeeded), the failed loads, the hsa_shut_down() calls made
// with no reference, and the calls refused because nothing holds a reference.
uint32_t rdc_fake_hsa_live_references(void);
uint32_t rdc_fake_hsa_load_count(void);
uint32_t rdc_fake_hsa_failed_load_count(void);
uint32_t rdc_fake_hsa_unbalanced_shutdowns(void);
uint32_t rdc_fake_hsa_refused_calls(void);

// The calls given an agent, queue, signal, executable, reader or symbol handle
// the fake never issued, which it refuses without reading the handle.
uint32_t rdc_fake_hsa_foreign_handles(void);

// Set once an hsa_shut_down() to zero left a queue alive: every later hsa_init()
// then fails with HSA_STATUS_ERROR_OUT_OF_RESOURCES, as it does on the hardware.
int rdc_fake_hsa_latched(void);

// HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, which is what a wait's hint is counted in. A
// frequency of 0 is reported as it is given, so that a case can see what the module makes of
// one; the fake's own waits take it as one tick a second.
void rdc_fake_hsa_set_timestamp_frequency(uint64_t hz);

// --- failures, gates and the call log -------------------------------------------

// Fails the nth call of that name (0: every call) with that status. The call
// does nothing else: a failed allocation hands out no block, a failed destroy
// keeps its object, and a failed hsa_shut_down() keeps its reference.
void rdc_fake_hsa_fail(const char* name, uint32_t nth, int32_t status);

// Throws an int from the nth call of that name (0: every call), a thing no ROCr
// call does, as a guard of the catch of every other exception.
void rdc_fake_hsa_throw_int(const char* name, uint32_t nth, int value);

// Drops every failure and every throw.
void rdc_fake_hsa_clear_failures(void);

// The calls of that name, and the call log, which keeps its last 10000 entries.
uint32_t rdc_fake_hsa_call_count(const char* name);
uint32_t rdc_fake_hsa_call_log_size(void);
const char* rdc_fake_hsa_call_log_entry(uint32_t entry);

// The token the calls of this thread carry, for a gate keyed to a caller. It is
// 0 unless the thread set one.
void rdc_fake_hsa_set_caller(uint64_t token);

// Holds calls of that name until the case releases them. With an nth, the one
// call of that name whose number is that nth, counted over every caller, is
// held; with 0, every call of it is. The caller token (0: any caller) narrows
// which thread a held call must be on and counts nothing of its own.
void rdc_fake_hsa_gate(const char* name, uint64_t caller, uint32_t nth);

// How many callers the gate holds now.
uint32_t rdc_fake_hsa_gate_held(const char* name);

// Waits for the gate to hold that many callers and returns 1, or returns 0 when
// the time passes.
int rdc_fake_hsa_gate_wait(const char* name, uint32_t callers, uint32_t timeout_ms);

// Lets every caller the gate holds, and every later call it would hold, go on.
void rdc_fake_hsa_gate_release(const char* name);

// --- the mapping of an entity to a KFD node ----------------------------------------

// What get_kfd_node_id() answers for an entity, and the failure it answers with
// (an amdsmi_status_t), which an entity of UINT32_MAX sets for every entity.
void rdc_fake_hsa_set_node_id(uint32_t entity, uint32_t node);
void rdc_fake_hsa_fail_node_id(uint32_t entity, int32_t status);
uint32_t rdc_fake_hsa_node_id_calls(void);
}

#ifdef RDC_FAKE_HSA_HARNESS
// The harness the two test files share. A test file defines RDC_FAKE_HSA_HARNESS
// before it includes this header; the fake itself does not, so the shared
// library the pin test dlopens needs neither GoogleTest nor these functions.

#include <errno.h>
#include <ftw.h>
#include <signal.h>
#include <string.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <functional>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#if defined(__SANITIZE_ADDRESS__)
#define RDC_FAKE_HSA_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RDC_FAKE_HSA_ASAN 1
#endif
#endif
#ifndef RDC_FAKE_HSA_ASAN
#define RDC_FAKE_HSA_ASAN 0
#endif

#if RDC_FAKE_HSA_ASAN
#include <sanitizer/lsan_interface.h>
#endif

namespace rdc_fake_hsa {

// What a case's child reports through its exit status.
enum child_status_t {
  kChildPassed = 0,
  kChildFailed = 1,  // the case's own assertions failed
  kChildSetUp = 2,   // the child could not start the case
  kChildThrew = 3,   // the case's body threw
  kChildLeaked = 4   // LeakSanitizer found a leak in an ASan build
};

// What an exit status other than the case's own failures means.
inline std::string child_status_text(int status) {
  switch (status) {
    case kChildFailed:
      return "the case failed";
    case kChildSetUp:
      return "the case could not be started";
    case kChildThrew:
      return "the case threw";
    case kChildLeaked:
      return "LeakSanitizer reported a leak";
    default:
      return "the case exited with status " + std::to_string(status);
  }
}

// A number of seconds with one decimal, so that a time limit reads as one.
inline std::string seconds_text(double seconds) {
  char text[32] = {};
  std::snprintf(text, sizeof(text), "%.1f", seconds);
  return text;
}

// The seconds a case may take: 60 s times RDC_TEST_TIME_FACTOR.
inline double time_factor() {
  const char* value = std::getenv("RDC_TEST_TIME_FACTOR");
  if (value == nullptr || *value == '\0') {
    return 1.0;
  }
  char* end = nullptr;
  const double factor = std::strtod(value, &end);
  if (end == value || !(factor > 0.0)) {
    return 1.0;
  }
  return factor;
}

// The case that timed out, if one did: every later case fails at once naming it.
inline std::string& timed_out_case() {
  static std::string name;
  return name;
}

// The name of the case that is running.
inline std::string current_case() {
  const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
  if (info == nullptr) {
    return "a case";
  }
  return std::string(info->test_suite_name()) + "." + info->name();
}

inline int remove_entry(const char* path, const struct stat*, int, struct FTW*) {
  return ::remove(path);
}

inline void remove_directory(const std::string& path) {
  ::nftw(path.c_str(), remove_entry, 16, FTW_DEPTH | FTW_PHYS);
}

// Makes the directory the case's child works in, under $TMPDIR.
inline std::string make_directory() {
  const char* tmp = std::getenv("TMPDIR");
  std::string pattern = (tmp != nullptr && *tmp != '\0') ? tmp : "/tmp";
  if (pattern.back() == '/') {
    pattern.pop_back();
  }
  pattern += "/rdc_rocr_case_XXXXXX";
  std::vector<char> buffer(pattern.begin(), pattern.end());
  buffer.push_back('\0');
  if (::mkdtemp(buffer.data()) == nullptr) {
    return std::string();
  }
  return std::string(buffer.data());
}

inline double monotonic_seconds() {
  struct timespec now = {};
  ::clock_gettime(CLOCK_MONOTONIC, &now);
  return static_cast<double>(now.tv_sec) + static_cast<double>(now.tv_nsec) / 1e9;
}

// LeakSanitizer's answer for what the case has allocated and no longer holds,
// in an ASan build, and false in every other build. run_in_child() makes this
// check at the end of the case; a case that unmaps a library (dlclose) makes
// that library's live state unreachable, which the check would report, so such
// a case calls this itself before the unmapping and the harness does not check
// again.
inline bool& leaks_checked() {
  static bool checked = false;
  return checked;
}

inline bool leaks_reported() {
  leaks_checked() = true;
#if RDC_FAKE_HSA_ASAN
  return __lsan_do_recoverable_leak_check() != 0;
#else
  return false;
#endif
}

// Runs the case's body in a child of its own, which works in its own directory,
// and gives the parent the verdict through its exit status. It is called while
// the process has one thread: a case that needs threads makes them in the child.
inline void run_in_child(const std::function<void()>& body) {
#if RDC_ROCR_TEST_HOOKS
  if (!timed_out_case().empty()) {
    FAIL() << "not run: " << timed_out_case() << " timed out";
  }
#endif
  const std::string directory = make_directory();
  ASSERT_FALSE(directory.empty()) << "mkdtemp: " << std::strerror(errno);

  const pid_t parent = ::getpid();
  std::fflush(nullptr);
  const pid_t child = ::fork();
  ASSERT_GE(child, 0) << "fork: " << std::strerror(errno);

  if (child == 0) {
    // No child outlives the binary, and none runs on without its parent.
    ::prctl(PR_SET_PDEATHSIG, SIGKILL);
    int verdict = kChildPassed;
    if (::getppid() != parent) {
      _exit(kChildSetUp);
    }
    if (::chdir(directory.c_str()) != 0) {
      std::fprintf(stderr, "the case could not enter %s: %s\n", directory.c_str(),
                   std::strerror(errno));
      verdict = kChildSetUp;
    } else {
      try {
        body();
        verdict = ::testing::Test::HasFailure() ? kChildFailed : kChildPassed;
      } catch (const std::exception& error) {
        std::fprintf(stderr, "the case threw: %s\n", error.what());
        verdict = kChildThrew;
      } catch (...) {
        std::fprintf(stderr, "the case threw something that is not a std::exception\n");
        verdict = kChildThrew;
      }
    }
#if RDC_FAKE_HSA_ASAN
    if (!leaks_checked() && __lsan_do_recoverable_leak_check() != 0) {
      std::fprintf(stderr, "LeakSanitizer reported a leak in the case\n");
      verdict = kChildLeaked;
    }
#endif
    // _exit() takes neither GoogleTest nor the sanitizers' exit-time work back
    // into this process, so what the child wrote must be flushed by hand.
    std::fflush(nullptr);
    _exit(verdict);
  }

  const double limit = 60.0 * time_factor();
  const double deadline = monotonic_seconds() + limit;
  int status = 0;
  bool reaped = false;
  while (monotonic_seconds() < deadline) {
    const pid_t done = ::waitpid(child, &status, WNOHANG);
    if (done == child) {
      reaped = true;
      break;
    }
    if (done < 0 && errno != EINTR) {
      break;
    }
    struct timespec pause = {0, 2 * 1000 * 1000};
    ::nanosleep(&pause, nullptr);
  }
  if (!reaped) {
    ::kill(child, SIGKILL);
    ::waitpid(child, &status, 0);
    timed_out_case() = current_case();
    FAIL() << "the case did not finish in " << seconds_text(limit)
           << " s; its directory is kept at " << directory;
  }
  if (WIFEXITED(status) && WEXITSTATUS(status) == kChildPassed) {
    remove_directory(directory);
    return;
  }
  if (WIFSIGNALED(status)) {
    ADD_FAILURE() << "the case died of signal " << WTERMSIG(status) << " ("
                  << ::strsignal(WTERMSIG(status)) << "); its directory is kept at " << directory;
    return;
  }
  ADD_FAILURE() << child_status_text(WEXITSTATUS(status)) << " (status " << WEXITSTATUS(status)
                << "); its directory is kept at " << directory;
}

}  // namespace rdc_fake_hsa

// A case whose body runs in a child of its own. It is variadic so that a brace
// initializer with a comma in it does not read as a second argument.
#define RDC_ROCR_CASE(...) rdc_fake_hsa::run_in_child([&]() __VA_ARGS__)

#endif  // RDC_FAKE_HSA_HARNESS

#endif  // TESTS_UNIT_FAKE_HSA_H_
