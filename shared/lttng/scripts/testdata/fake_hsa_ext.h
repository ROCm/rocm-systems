/* Test fixture — HSA AMD-extensions-only header. */
#ifndef FAKE_HSA_EXT_H_
#define FAKE_HSA_EXT_H_
#include "fake_hsa_base.h"

hsa_status_t hsa_amd_signal_create(int64_t initial_value,
                                   uint32_t num_consumers,
                                   const void* consumers,
                                   uint64_t attributes,
                                   hsa_signal_t* signal);
#endif
