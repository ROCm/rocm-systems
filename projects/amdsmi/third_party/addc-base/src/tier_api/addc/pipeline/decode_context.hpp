// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/decoder/dbglog/decoder.hpp"
#include "addc/decoder/wdt/decoder.hpp"
#include "addc/mca/decoder.hpp"
#include "addc/pipeline/output.hpp"

#include <cstddef>
#include <string>

namespace addc::pipeline
{

struct DecodeContext
{
    std::size_t section_index{};
    std::string timestamp;
    std::string header_severity;
    Diagnostics* diagnostics = nullptr;
    mca::McaDecodeFunction decode_mca = mca::try_decode_project_event;
    decoder::DbgLogDecodeFunction decode_dbglog =
        decoder::decode_base_dbglog;
    decoder::WdtDecodeFunction decode_wdt = decoder::decode_base_wdt;
};

} // namespace addc::pipeline
