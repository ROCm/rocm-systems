
#include "common.hpp"
#include "rocm_async.hpp"

#include <assert.h>
#include <algorithm>
#include <unistd.h>
#include <cctype>
#include <sstream>

bool RocmAsync::PoolIsPresent(vector<uint32_t>& in_list) {
  
  bool is_present;
  uint32_t idx1 = 0;
  uint32_t idx2 = 0;
  uint32_t count = in_list.size();
  uint32_t pool_count = pool_list_.size();
  for (idx1 = 0; idx1 < count; idx1++) {
    is_present = false;
    for (idx2 = 0; idx2 < pool_count; idx2++) {
      if (in_list[idx1] == pool_list_[idx2].index_) {
        is_present = true;
        break;
      }
    }
    if (is_present == false) {
      return false;
    }
  }

  return true;
}

bool RocmAsync::PoolIsDuplicated(vector<uint32_t>& in_list) {
  
  uint32_t idx1 = 0;
  uint32_t idx2 = 0;
  uint32_t count = in_list.size();
  for (idx1 = 0; idx1 < count; idx1++) {
    for (idx2 = 0; idx2 < count; idx2++) {
      if ((in_list[idx1] == in_list[idx2]) && (idx1 != idx2)){
        return false;
      }
    }
  }
  return true;
}

bool RocmAsync::ValidateReadOrWriteReq(vector<uint32_t>& in_list) {

  // Determine read / write request is even
  // Request is specified as a list of memory
  // pool, agent tuples - first element identifies
  // memory pool while the second element denotes
  // an agent
  uint32_t list_size = in_list.size();
  if ((list_size % 2) != 0) {
    return false;
  }
  
  // Validate the list of pool-agent tuples
  for (uint32_t idx = 0; idx < list_size; idx+=2) {
    uint32_t pool_idx = in_list[idx];
    uint32_t exec_idx = in_list[idx + 1];
    // Determine the pool and agent exist in system
    if ((pool_idx >= pool_index_) ||
        (exec_idx >= agent_index_)) {
      return false;
    }
  }
  return true;
}

bool RocmAsync::ValidateReadReq() {
  return ValidateReadOrWriteReq(read_list_);
}

bool RocmAsync::ValidateWriteReq() {
  return ValidateReadOrWriteReq(write_list_);
}

bool RocmAsync::ValidateCopyReq(vector<uint32_t>& in_list) {
  
  // Determine pool list length is valid
  uint32_t count = in_list.size();
  uint32_t pool_count = pool_list_.size();
  if (count > pool_count) {
    return false;
  }
  
  // Determine no pool is duplicated
  bool status = PoolIsDuplicated(in_list);
  if (status == false) {
    return false;
  }
  
  // Determine every pool is present in system
  return PoolIsPresent(in_list);
}

bool RocmAsync::ValidateBidirCopyReq() {
  return ValidateCopyReq(bidir_list_);
}

bool RocmAsync::ValidateUnidirCopyReq() {
  return ((ValidateCopyReq(src_list_)) && (ValidateCopyReq(dst_list_)));
}

bool RocmAsync::ValidateArguments() {
  
  // Determine if user has requested a READ
  // operation and gave valid inputs
  bool status = false;
  if (req_read_ == REQ_READ) {
    status = ValidateReadReq();
    if (status == false) {
      return status;
    }
  }

  // Determine if user has requested a WRITE
  // operation and gave valid inputs
  status = false;
  if (req_write_ == REQ_WRITE) {
    status = ValidateWriteReq();
    if (status == false) {
      return status;
    }
  }

  // Determine if user has requested a Copy
  // operation that is bidirectional and gave
  // valid inputs. Same validation is applied
  // for all-to-all unidirectional copy operation
  status = false;
  if ((req_copy_bidir_ == REQ_COPY_BIDIR) ||
      (req_copy_all_bidir_ == REQ_COPY_ALL_BIDIR)) {
    status = ValidateBidirCopyReq();
    if (status == false) {
      return status;
    }
  }

  // Determine if user has requested a Copy
  // operation that is unidirectional and gave
  // valid inputs. Same validation is applied
  // for all-to-all bidirectional copy operation
  status = false;
  if ((req_copy_unidir_ == REQ_COPY_UNIDIR) ||
      (req_copy_all_unidir_ == REQ_COPY_ALL_UNIDIR)) {
    status = ValidateUnidirCopyReq();
    if (status == false) {
      return status;
    }
  }

  // All of the request are well formed
  return true;
}
