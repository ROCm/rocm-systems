// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include "addc/cper/record.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <span>

namespace addc::cper::sections
{

/// Parses a UEFI PCI Express error section body (GUID
/// d995e954-bbc1-430f-ad91-b44dcb3c6f35) into the JSON IR that
/// addc::decoder::decode_pcie() reads. A higher tier uses it for PCIe sections
/// that arrive outside a CPER record, such as ACPI BERT entries.
[[nodiscard]] nlohmann::json parse_pcie(std::span<const uint8_t> body,
                                        const SectionDescriptor& desc);

} // namespace addc::cper::sections
