/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

namespace rocr {
namespace core {
namespace Runtime {
  // Global flags stub, just for compilation not used during execution
  struct GlobalFlag {
    struct Flags {  
      bool override_cpu_affinity() const {
        return false;
      }    
    } flags_;
    const Flags& flag() const {
      return flags_;
    }
  } runtime_singleton_;
} // namespace Runtime
} // namespace core 
} // namespace rocr
