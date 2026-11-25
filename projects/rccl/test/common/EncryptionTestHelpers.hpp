/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_ENCRYPTION_TEST_HELPERS_HPP
#define RCCL_ENCRYPTION_TEST_HELPERS_HPP

#include <rccl/rccl.h>

namespace RcclUnitTesting {

#if defined(NCCL_TLS_BACKEND_OPENSSL3)
constexpr bool kTlsBuild = true;
#else
constexpr bool kTlsBuild = false;
#endif

// Configures process-wide socket encryption with a pre-shared key.
inline ncclResult_t SetPsk(const char* psk) {
  ncclEncryptionConfig_t cfg = NCCL_ENCRYPTION_CONFIG_INITIALIZER;
  cfg.mode = NCCL_ENCRYPTION_MODE_PSK;
  cfg.psk = psk;
  return ncclSetEncryption(&cfg);
}

inline ncclResult_t ClearPsk() { return ncclSetEncryption(nullptr); }

} // namespace RcclUnitTesting

#endif // RCCL_ENCRYPTION_TEST_HELPERS_HPP
