
#include "common.hpp"
#include "rocm_async.hpp"

#include <stdlib.h>
#include <assert.h>
#include <algorithm>
#include <unistd.h>
#include <cctype>
#include <sstream>

// The values are in megabytes at allocation time
const uint32_t RocmAsync::SIZE_LIST[] = { 64, 128, 256, 512 };
//const uint32_t RocmAsync::SIZE_LIST[] = { 2, 4, 8, 16, 32, 64, 128, 256, 512 };

uint32_t RocmAsync::GetIterationNum() {
  return num_iteration_ * 1.2 + 1;
}

void RocmAsync::AcquireAccess(hsa_agent_t agent, void* ptr) {
  err_ = hsa_amd_agents_allow_access(1, &agent, NULL, ptr);
  ErrorCheck(err_);
}

void RocmAsync::AllocateHostBuffers(bool bidir, uint32_t size,
                                    void*& src_fwd, void*& dst_fwd,
                                    void* buf_src_fwd, void* buf_dst_fwd,
                                    hsa_agent_t src_agent_fwd, hsa_agent_t dst_agent_fwd,
                                    void*& src_rev, void*& dst_rev,
                                    void* buf_src_rev, void* buf_dst_rev,
                                    hsa_agent_t src_agent_rev, hsa_agent_t dst_agent_rev,
                                    hsa_signal_t& signal_fwd, hsa_signal_t& signal_rev) {

  // Allocate host buffers and setup accessibility for copy operation
  err_ = hsa_amd_memory_pool_allocate(sys_pool_, size, 0, (void**)&src_fwd);
  ErrorCheck(err_);
  AcquireAccess(src_agent_fwd, src_fwd);
  AcquireAccess(cpu_agent_, buf_src_fwd);

  err_ = hsa_amd_memory_pool_allocate(sys_pool_, size, 0, (void**)&dst_fwd);
  ErrorCheck(err_);
  AcquireAccess(dst_agent_fwd, dst_fwd);
  AcquireAccess(cpu_agent_, buf_dst_fwd);

  // Initialize host buffers to a determinate value
  memset(src_fwd, 0x23, size);
  memset(dst_fwd, 0x00, size);
  
  // Create a signal to wait on copy operation
  // @TODO: replace it with a signal pool call
  err_ = hsa_signal_create(1, 0, NULL, &signal_fwd);
  ErrorCheck(err_);

  if (bidir == false) {
    return;
  }

  err_ = hsa_amd_memory_pool_allocate(sys_pool_, size, 0, (void**)&src_rev);
  ErrorCheck(err_);
  AcquireAccess(src_agent_rev, src_rev);
  AcquireAccess(cpu_agent_, buf_src_rev);

  err_ = hsa_amd_memory_pool_allocate(sys_pool_, size, 0, (void**)&dst_rev);
  ErrorCheck(err_);
  AcquireAccess(dst_agent_rev, dst_rev);
  AcquireAccess(cpu_agent_, buf_dst_rev);

  // Initialize host buffers to a determinate value
  memset(src_rev, 0x23, size);
  memset(dst_rev, 0x00, size);
  
  err_ = hsa_signal_create(1, 0, NULL, &signal_rev);
  ErrorCheck(err_);
}

void RocmAsync::AllocateCopyBuffers(bool bidir, uint32_t size,
                        void*& src_fwd, hsa_amd_memory_pool_t src_pool_fwd,
                        void*& dst_fwd, hsa_amd_memory_pool_t dst_pool_fwd,
                        hsa_agent_t src_agent_fwd, hsa_agent_t dst_agent_fwd,
                        void*& src_rev, hsa_amd_memory_pool_t src_pool_rev,
                        void*& dst_rev, hsa_amd_memory_pool_t dst_pool_rev,
                        hsa_agent_t src_agent_rev, hsa_agent_t dst_agent_rev,
                        hsa_signal_t& signal_fwd, hsa_signal_t& signal_rev) {

  // Allocate buffers in src and dst pools for forward copy
  err_ = hsa_amd_memory_pool_allocate(src_pool_fwd, size, 0, &src_fwd);
  ErrorCheck(err_);
  err_ = hsa_amd_memory_pool_allocate(dst_pool_fwd, size, 0, &dst_fwd);
  ErrorCheck(err_);

  // Allocate buffers in src and dst pools for reverse copy
  if (bidir) {
    err_ = hsa_amd_memory_pool_allocate(src_pool_rev, size, 0, &src_rev);
    ErrorCheck(err_);
    err_ = hsa_amd_memory_pool_allocate(dst_pool_rev, size, 0, &dst_rev);
    ErrorCheck(err_);
  }

  // Acquire access to src and dst buffers for forward copy
  AcquireAccess(src_agent_fwd, dst_fwd);
  AcquireAccess(dst_agent_fwd, src_fwd);

  // Acquire access to src and dst buffers for reverse copy
  if (bidir) {
    AcquireAccess(src_agent_rev, dst_rev);
    AcquireAccess(dst_agent_rev, src_rev);
  }
  
  // Create a signal to wait on copy operation
  // @TODO: replace it with a signal pool call
  err_ = hsa_signal_create(1, 0, NULL, &signal_fwd);
  ErrorCheck(err_);
  if (bidir) {
    err_ = hsa_signal_create(1, 0, NULL, &signal_rev);
    ErrorCheck(err_);
  }
}

void RocmAsync::ReleaseBuffers(bool bidir,
                               void* src_fwd, void* src_rev,
                               void* dst_fwd, void* dst_rev,
                               hsa_signal_t signal_fwd,
                               hsa_signal_t signal_rev) {

  // Free the src and dst buffers used in forward copy
  // including the signal used to wait
  err_ = hsa_amd_memory_pool_free(src_fwd);
  ErrorCheck(err_);
  err_ = hsa_amd_memory_pool_free(dst_fwd);
  ErrorCheck(err_);
  err_ = hsa_signal_destroy(signal_fwd);
  ErrorCheck(err_);

  // Free the src and dst buffers used in reverse copy
  // including the signal used to wait
  if (bidir) {
    err_ = hsa_amd_memory_pool_free(src_rev);
    ErrorCheck(err_);
    err_ = hsa_amd_memory_pool_free(dst_rev);
    ErrorCheck(err_);
    err_ = hsa_signal_destroy(signal_rev);
    ErrorCheck(err_);
  }
}

double RocmAsync::GetGpuCopyTime(bool bidir,
                                 hsa_signal_t signal_fwd,
                                 hsa_signal_t signal_rev) {

  // Obtain time taken for forward copy
  hsa_amd_profiling_async_copy_time_t async_time_fwd = {0};
  err_= hsa_amd_profiling_get_async_copy_time(signal_fwd, &async_time_fwd);
  ErrorCheck(err_);
  if (bidir == false) {
    return(async_time_fwd.end - async_time_fwd.start);
  }

  hsa_amd_profiling_async_copy_time_t async_time_rev = {0};
  err_= hsa_amd_profiling_get_async_copy_time(signal_rev, &async_time_rev);
  ErrorCheck(err_);
  double start = min(async_time_fwd.start, async_time_rev.start);
  double end = max(async_time_fwd.end, async_time_rev.end);
  return(end - start);
}

void RocmAsync::copy_buffer(void* dst, hsa_agent_t dst_agent,
                            void* src, hsa_agent_t src_agent,
                            size_t size, hsa_signal_t signal) {

  // Copy from src into dst buffer
  err_ = hsa_amd_memory_async_copy(dst, dst_agent,
                                   src, src_agent,
                                   size, 0, NULL, signal);
  ErrorCheck(err_);
  
  // Wait for the forward copy operation to complete
  while (hsa_signal_wait_acquire(signal, HSA_SIGNAL_CONDITION_LT, 1,
                                     uint64_t(-1), HSA_WAIT_STATE_ACTIVE));
}

void RocmAsync::RunCopyBenchmark(async_trans_t& trans) {

  // Bind if this transaction is bidirectional
  bool bidir = trans.copy.bidir_;

  // Initialize size of buffer to equal the largest element of allocation
  uint32_t size_len = size_list_.size();
  uint32_t max_size = size_list_.back() * 1024 * 1024;
  
  // Bind to resources such as pool and agents that are involved
  // in both forward and reverse copy operations
  void* buf_src_fwd;
  void* buf_dst_fwd;
  void* buf_src_rev;
  void* buf_dst_rev;
  void* host_src_fwd;
  void* host_dst_fwd;
  void* host_src_rev;
  void* host_dst_rev;
  hsa_signal_t signal_fwd;
  hsa_signal_t signal_rev;
  hsa_signal_t host_signal_fwd;
  hsa_signal_t host_signal_rev;
  hsa_amd_memory_pool_t src_pool_fwd = trans.copy.src_pool_;
  hsa_amd_memory_pool_t dst_pool_fwd = trans.copy.dst_pool_;
  hsa_amd_memory_pool_t src_pool_rev = dst_pool_fwd;
  hsa_amd_memory_pool_t dst_pool_rev = src_pool_fwd;
  hsa_agent_t src_agent_fwd = pool_list_[trans.copy.src_idx_].owner_agent_;
  hsa_agent_t dst_agent_fwd = pool_list_[trans.copy.dst_idx_].owner_agent_;
  hsa_agent_t src_agent_rev = dst_agent_fwd;
  hsa_agent_t dst_agent_rev = src_agent_fwd;

  // Allocate buffers and signal objects
  AllocateCopyBuffers(bidir, max_size,
                      buf_src_fwd, src_pool_fwd, 
                      buf_dst_fwd, dst_pool_fwd,
                      src_agent_fwd, dst_agent_fwd,
                      buf_src_rev, src_pool_rev, 
                      buf_dst_rev, dst_pool_rev,
                      src_agent_rev, dst_agent_rev,
                      signal_fwd, signal_rev);
  
  if (verify_) {
    AllocateHostBuffers(bidir, max_size,
                        host_src_fwd, host_dst_fwd,
                        buf_src_fwd, buf_dst_fwd,
                        src_agent_fwd, dst_agent_fwd,
                        host_src_rev, host_dst_rev,
                        buf_src_rev, buf_dst_rev,
                        src_agent_rev, dst_agent_rev,
                        host_signal_fwd, host_signal_rev);

    // Initialize source buffer with values from verification buffer
    copy_buffer(buf_src_fwd, src_agent_fwd,
                host_src_fwd, cpu_agent_,
                max_size, host_signal_fwd);
    ErrorCheck(err_);
    if (bidir) {
      copy_buffer(buf_src_rev, src_agent_rev,
                  host_src_rev, cpu_agent_,
                  max_size, host_signal_rev);
      ErrorCheck(err_);
    }
  }

  // Bind the number of iterations
  uint32_t iterations = GetIterationNum();

  // Iterate through the differnt buffer sizes to
  // compute the bandwidth as determined by copy
  for (uint32_t idx = 0; idx < size_len; idx++) {
    
    // This should not be happening
    uint32_t curr_size = size_list_[idx] * 1024 * 1024;
    if (curr_size > max_size) {
      break;
    }

    std::vector<double> cpu_time;
    std::vector<double> gpu_time;
    for (uint32_t it = 0; it < iterations; it++) {
      #if DEBUG
      printf(".");
      fflush(stdout);
      #endif

      hsa_signal_store_relaxed(signal_fwd, 1);
      if (bidir) {
        hsa_signal_store_relaxed(signal_rev, 1);
      }

      if (verify_) {
        AcquireAccess(src_agent_fwd, buf_dst_fwd);
        AcquireAccess(dst_agent_fwd, buf_src_fwd);
        if (bidir) {
          AcquireAccess(src_agent_rev, buf_dst_rev);
          AcquireAccess(dst_agent_rev, buf_src_rev);
        }
      }

      // Create a timer object and reset signals
      PerfTimer timer;
      uint32_t index = timer.CreateTimer();

      // Start the timer and launch forward copy operation
      timer.StartTimer(index);
      err_ = hsa_amd_memory_async_copy(buf_dst_fwd, dst_agent_fwd,
                                       buf_src_fwd, src_agent_fwd,
                                       curr_size, 0, NULL, signal_fwd);
      ErrorCheck(err_);

      // Launch reverse copy operation if it is bidirectional
      if (bidir) {
        err_ = hsa_amd_memory_async_copy(buf_dst_rev, dst_agent_rev,
                                         buf_src_rev, src_agent_rev,
                                         curr_size, 0, NULL, signal_rev);
        ErrorCheck(err_);
      }

      // Wait for the forward copy operation to complete
      while (hsa_signal_wait_acquire(signal_fwd, HSA_SIGNAL_CONDITION_LT, 1,
                                     uint64_t(-1), HSA_WAIT_STATE_ACTIVE));

      // Wait for the reverse copy operation to complete
      if (bidir) {
        while (hsa_signal_wait_acquire(signal_rev, HSA_SIGNAL_CONDITION_LT, 1,
                                       uint64_t(-1), HSA_WAIT_STATE_ACTIVE));
      }

      // Stop the timer object
      timer.StopTimer(index);

      // Push the time taken for copy into a vector of copy times
      cpu_time.push_back(timer.ReadTimer(index));

      // Collect time from the signal(s)
      if (trans.copy.uses_gpu_) {
        double temp = GetGpuCopyTime(bidir, signal_fwd, signal_rev);
        gpu_time.push_back(temp);
      }

      if (verify_) {

        // Re-Establish access to destination buffer and host buffer
        AcquireAccess(cpu_agent_, buf_dst_fwd);
        AcquireAccess(dst_agent_fwd, host_dst_fwd);
        
        // Init dst buffer with values from outbuffer of copy operation
        hsa_signal_store_relaxed(host_signal_fwd, 1);
        copy_buffer(host_dst_fwd, cpu_agent_,
                    buf_dst_fwd, dst_agent_fwd,
                    curr_size, host_signal_fwd);
        ErrorCheck(err_);
        
        // Compare output equals input
        err_ = (hsa_status_t)memcmp(host_src_fwd, host_dst_fwd, curr_size);
        ErrorCheck(err_);

        if (bidir) {

          // Re-Establish access to destination buffer and host buffer
          AcquireAccess(cpu_agent_, buf_dst_rev);
          AcquireAccess(dst_agent_rev, host_dst_rev);

          hsa_signal_store_relaxed(host_signal_rev, 1);
          copy_buffer(host_dst_rev, cpu_agent_,
                      buf_dst_rev, dst_agent_rev,
                      curr_size, host_signal_rev);
          ErrorCheck(err_);
        
          // Compare output equals input
          err_ = (hsa_status_t)memcmp(host_src_rev, host_dst_rev, curr_size);
          ErrorCheck(err_);
        }
      }
    }
    #if DEBUG
    std::cout << std::endl;
    #endif

    // Get Cpu min copy time
    trans.cpu_min_time_.push_back(GetMinTime(cpu_time));
    // Get Cpu mean copy time and store to the array
    trans.cpu_avg_time_.push_back(GetMeanTime(cpu_time));

    if (trans.copy.uses_gpu_) {
      // Get Gpu min copy time
      trans.gpu_min_time_.push_back(GetMinTime(gpu_time));
      // Get Gpu mean copy time and store to the array
      trans.gpu_avg_time_.push_back(GetMeanTime(gpu_time));
    }

    // Clear the stack of cpu times
    cpu_time.clear();
    gpu_time.clear();
  }
  
  // Free up buffers and signal objects used in copy operation
  ReleaseBuffers(bidir, buf_src_fwd, buf_src_rev,
                 buf_dst_fwd, buf_dst_rev, signal_fwd, signal_rev);
  
  if (verify_) {
    ReleaseBuffers(bidir, host_src_fwd, host_src_rev,
                   host_dst_fwd, host_dst_rev, host_signal_fwd, host_signal_rev);
  }
}

void RocmAsync::Run() {

  // Enable profiling of Async Copy Activity
  err_ = hsa_amd_profiling_async_copy_enable(true);
  ErrorCheck(err_);

  // Iterate through the list of transactions and execute them
  uint32_t trans_size = trans_list_.size();
  for (uint32_t idx = 0; idx < trans_size; idx++) {
    async_trans_t& trans = trans_list_[idx];
    if ((trans.req_type_ == REQ_COPY_BIDIR) ||
        (trans.req_type_ == REQ_COPY_UNIDIR) ||
        (trans.req_type_ == REQ_COPY_ALL_BIDIR) ||
        (trans.req_type_ == REQ_COPY_ALL_UNIDIR)) {
      RunCopyBenchmark(trans);
      ComputeCopyTime(trans);
    }
    if ((trans.req_type_ == REQ_READ) ||
        (trans.req_type_ == REQ_WRITE)) {
      RunIOBenchmark(trans);
    }
  }

  // Disable profiling of Async Copy Activity
  err_ = hsa_amd_profiling_async_copy_enable(false);
  ErrorCheck(err_);

}

void RocmAsync::Close() {
  hsa_status_t status = hsa_shut_down();
  ErrorCheck(status);
  return;
}

// Sets up the bandwidth test object to enable running
// the various test scenarios requested by user. The
// things this proceedure takes care of are:
//    
//    Parse user arguments
//    Discover RocR Device Topology
//    Determine validity of requested test scenarios
//    Build the list of transactions to execute
//    Miscellaneous
//
void RocmAsync::SetUp() {

  // Parse user arguments
  ParseArguments();

  // Validate input parameters
  bool status = ValidateArguments();
  if (status == false) {
    PrintHelpScreen();
    exit(1);
  }

  // Build list of transactions (copy, read, write) to execute
  status = BuildTransList();
  if (status == false) {
    PrintHelpScreen();
    exit(1);
  }
}

RocmAsync::RocmAsync(int argc, char** argv) : BaseTest() {
  usr_argc_ = argc;
  usr_argv_ = argv;
  verify_ = false;
  pool_index_ = 0;
  agent_index_ = 0;
  req_read_ = REQ_INVALID;
  req_write_ = REQ_INVALID;
  req_copy_bidir_ = REQ_INVALID;
  req_copy_unidir_ = REQ_INVALID;
  req_copy_all_bidir_ = REQ_INVALID;
  req_copy_all_unidir_ = REQ_INVALID;
}

RocmAsync::~RocmAsync() { }

