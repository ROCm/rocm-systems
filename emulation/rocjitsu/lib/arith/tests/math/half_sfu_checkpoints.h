// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Stored FP16 special-function checkpoints and exhaustive-sweep hashes.

#pragma once

#include <stdint.h>

namespace rocjitsu::arith::test {

// RCP, RSQ, SQRT: cumulative word-wise hashes every 4096 half encodings.
// Unmodified inputs, denormals preserved, quiet NaNs, no saturation.
// Generated from rocjitsu commit ffc144c564c450b7847ae0528410dc63af9725a8.
static const uint64_t half_sfu_checkpoints[3][16] = {
    {
        0xb4867455ae932e96ULL,
        0xfcd2bc2a58071f0eULL,
        0xe84e3439731681c6ULL,
        0x409a3939eacf833eULL,
        0x3a44563b783890b6ULL,
        0x4c7b62863d8e17aeULL,
        0xaf06591469663e66ULL,
        0x061160653541c1d3ULL,
        0x707d054930b0dc20ULL,
        0xc849de8a4c141068ULL,
        0xf2289b5201929010ULL,
        0x30fdffc26eeb8298ULL,
        0xd5d61a45015dd780ULL,
        0xdb9ba5f07e7525c8ULL,
        0x8b9bd3d10fd8f7f0ULL,
        0x0667b75d288e9039ULL,
    },
    {
        0xdc4b7b26955a2920ULL,
        0x9b057350a3311084ULL,
        0xc59ccb0919eefe68ULL,
        0xae8cdbcf500f550cULL,
        0x4cc4b47b7ec247d0ULL,
        0xc8f7759956962db4ULL,
        0xd54458c98bbbb2d8ULL,
        0xcabe2b074eeccdb9ULL,
        0x06ece7e409a18fb9ULL,
        0x89405b9f0aaa4fb9ULL,
        0x888741523b4d0fb9ULL,
        0xf3df5fd096e5cfb9ULL,
        0x440c793b919a8fb9ULL,
        0xcd25d984d2a34fb9ULL,
        0xa9fa3bc643460fb9ULL,
        0x5f415b15127233b9ULL,
    },
    {
        0xc57ff883c46ba521ULL,
        0x220d48e682e802ddULL,
        0x24afaf632e1b5c29ULL,
        0x671d3bc237b9d625ULL,
        0x4477b91898f59c71ULL,
        0x362bd5b155cca5edULL,
        0xeb03d8c1b14a9df9ULL,
        0xf1c3ccdd36c6c336ULL,
        0x1f85a0d3e45e3d36ULL,
        0x16477a24c021bd36ULL,
        0xb234b7222db93d36ULL,
        0xce0fa979c97cbd36ULL,
        0xb203af7df7143d36ULL,
        0xa4d31adc52d7bd36ULL,
        0x5aa949e7406f3d36ULL,
        0x551301580720ed36ULL,
    },
};

} // namespace rocjitsu::arith::test
