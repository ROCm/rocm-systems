// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Interrupt makes notification writes return EINTR, leaving completed exports
// awaiting notification. Cancel requests deferred cancellation at the first
// notification write.
enum class NotificationWriteMode { Disabled, Interrupt, Cancel };
