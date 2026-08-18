// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "vendor_registry.h"

#include "amd/ifoe_subsystem.h"
#include "broadcom/broadcom_subsystem.h"
#include "pensando/pensando_subsystem.h"

std::vector<std::unique_ptr<SmiNicSubsystem>> make_default_vendor_plugins() {
  std::vector<std::unique_ptr<SmiNicSubsystem>> plugins;
  plugins.push_back(std::make_unique<SmiNicSubsystemPensando>());
  plugins.push_back(std::make_unique<SmiNicSubsystemIfoe>());
  plugins.push_back(std::make_unique<SmiNicSubsystemBroadcom>());
  return plugins;
}
