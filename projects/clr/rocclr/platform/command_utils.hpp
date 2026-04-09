/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef COMMAND_UTILS_HPP_
#define COMMAND_UTILS_HPP_

// Dummy command types for Stream Wait and Write commands.
#define ROCCLR_COMMAND_STREAM_WAIT_VALUE 0x4501
#define ROCCLR_COMMAND_STREAM_WRITE_VALUE 0x4502
#define ROCCLR_COMMAND_BATCH_STREAM 0x4503
#define ROCCLR_COMMAND_BATCH_COPY_BUFFER 0x4504

// Stream Wait Value Conidtions
#define ROCCLR_STREAM_WAIT_VALUE_GTE 0x0
#define ROCCLR_STREAM_WAIT_VALUE_EQ 0x1
#define ROCCLR_STREAM_WAIT_VALUE_AND 0x2
#define ROCCLR_STREAM_WAIT_VALUE_NOR 0x3

#define ROCCLR_HIP_GL_CONTEXT_KHR 0x2100
#define ROCCLR_HIP_GLX_DISPLAY_KHR 0x2101
#define ROCCLR_HIP_WGL_HDC_KHR 0x2102

#endif  // COMMAND_UTILS_HPP_