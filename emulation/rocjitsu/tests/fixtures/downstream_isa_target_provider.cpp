// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "downstream_isa_target_provider.h"

#include "downstream_isa_fixture.h"
#include "rocjitsu/isa/target_provider.h"

namespace rocjitsu::test {

std::unique_ptr<rocjitsu::Decoder> create_downstream_target_decoder() {
  return make_isa_decoder<DownstreamIsa>();
}

} // namespace rocjitsu::test
