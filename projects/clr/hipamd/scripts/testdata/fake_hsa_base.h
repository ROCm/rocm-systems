/* Test fixture — minimal HSA-base-like header (no AMD extensions). */
#ifndef FAKE_HSA_BASE_H_
#define FAKE_HSA_BASE_H_
#include <stdint.h>
#include <stddef.h>

typedef enum { HSA_STATUS_SUCCESS = 0 } hsa_status_t;
typedef struct { uint64_t handle; } hsa_signal_t;

hsa_status_t hsa_signal_create(int64_t initial_value,
                               uint32_t num_consumers,
                               const void* consumers,
                               hsa_signal_t* signal);
#endif
