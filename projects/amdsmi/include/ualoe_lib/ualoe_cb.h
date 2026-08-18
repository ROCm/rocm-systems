// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef UALOE_CB_H
#define UALOE_CB_H

#include "ualoe_lib.h"

void ualoe_cb_fini(ualoe_handle_t handle);
int ualoe_cb_init(ualoe_handle_t handle, int dev_id, ualoe_event_callback_t cb, void* user_ctx);

#endif /* UALOE_CB_H */
