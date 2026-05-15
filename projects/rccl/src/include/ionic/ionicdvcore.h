#ifndef NCCL_IONICDV_CORE_H_
#define NCCL_IONICDV_CORE_H_

/* Basic ionic direct verbs structs.
 * Needed to dynamically load ionic direct verbs functions without
 * explicit including of ionic direct verbs header.
 */

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>
#include <unistd.h>
#include "ibvwrap.h"

enum ionicdv_reg_udma_mask {
    IONIC_UDMA_MASK_LOW    = 1,
    IONIC_UDMA_MASK_HIGH   = 2
};

#endif  // NCCL_IONICDV_CORE_H_
