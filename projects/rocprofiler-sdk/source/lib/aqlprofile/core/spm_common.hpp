#pragma once

#include "aqlprofile-sdk/aql_profile_v2.h"
#include <string>
#include <stdio.h>
#include <stdexcept>
#include <memory>

inline bool operator<(const aqlprofile_handle_t& a, const aqlprofile_handle_t& b)
{
    return a.handle < b.handle;
}

#define SPM_DESC_SIZE 0x1000

// Once KFD change is merged, we should use the definition from linux/include/uapi/linux/kfd_ioctl.h
struct kfd_ioctl_spm_buffer_header {
    uint32_t version; /* 0-23: minor 24-31: major */
    uint32_t bytes_copied;
    uint32_t has_data_loss;
    uint32_t reserved[5];
};

typedef struct SpmBufferDesc_ {
  uint32_t version{1};
  uint32_t global_num_line{0};
  uint32_t se_num_line{0};
  uint32_t num_se{0};
  uint32_t num_sa{0};
  uint32_t num_xcc{0};
  size_t num_events{0};

  uint16_t* get_counter_map()
  {
    return (uint16_t*)(this+1);
  }
} SpmBufferDesc;
