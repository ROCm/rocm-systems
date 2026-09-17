// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"
#include "rocjitsu/code/patch/consan/targets/consan_target_profiles.h"

#include <iostream>
#include <string>

namespace rocjitsu::consan {
namespace {

std::string render_domain(rj_code_target_id_t target, Mode mode, CapabilityDomain domain) {
  std::string rendered;
  for (CapabilityForm form : kCapabilityForms) {
    if (capability_domain(form) != domain)
      continue;
    const CapabilityDisposition disposition = capability_disposition(target, mode, form);
    if (disposition == CapabilityDisposition::NotApplicable)
      continue;
    if (!rendered.empty())
      rendered += "<br>";
    rendered += capability_form_name(form);
    if (disposition != CapabilityDisposition::Supported) {
      rendered += " (";
      rendered += capability_disposition_name(disposition);
      rendered += ')';
    }
  }
  return rendered.empty() ? "not applicable" : rendered;
}

} // namespace

int render_manifest() {
  std::cout << "<!-- BEGIN GENERATED CONSAN CAPABILITY CONTRACT -->\n"
               "| Target | Mode | Access | Barrier | Atomic | Fence |\n"
               "| --- | --- | --- | --- | --- | --- |\n";
  for (const TargetProfile &target : kTargetProfiles) {
    for (Mode mode : kEnabledModes) {
      std::cout << "| `" << rj_code_target_name(target.target) << "` | " << mode_label(mode);
      for (CapabilityDomain domain : kCapabilityDomains)
        std::cout << " | " << render_domain(target.target, mode, domain);
      std::cout << " |\n";
    }
  }
  std::cout << "<!-- END GENERATED CONSAN CAPABILITY CONTRACT -->\n";
  return 0;
}

} // namespace rocjitsu::consan

int main() { return rocjitsu::consan::render_manifest(); }
