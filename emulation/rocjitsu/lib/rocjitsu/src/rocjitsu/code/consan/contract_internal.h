// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/consan/contract.h"
#include "rocjitsu/code/consan/encoding.h"

namespace rocjitsu::consan::wire {

void encode_identity(Encoder &encoder, const ImageIdentity &identity);
bool decode_identity(Decoder &decoder, ImageIdentity &identity);
void encode_entry(Encoder &encoder, const EntryFacts &entry);
bool decode_entry(Decoder &decoder, EntryFacts &entry, uint32_t string_limit);

} // namespace rocjitsu::consan::wire
