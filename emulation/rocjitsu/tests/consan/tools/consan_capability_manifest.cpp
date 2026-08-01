// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_capability_contract.h"

#include <iostream>
#include <string>

namespace {

using namespace rocjitsu;

std::string render_domain(rj_code_target_id_t target, ConSanCapabilityEngine engine,
                          ConSanCapabilityDomain domain) {
  std::string rendered;
  for (ConSanCapabilityForm form : kConSanCapabilityForms) {
    if (consan_capability_domain(form) != domain)
      continue;
    const ConSanCapabilityDisposition disposition =
        consan_capability_disposition(target, engine, form);
    if (disposition == ConSanCapabilityDisposition::NotApplicable)
      continue;
    if (!rendered.empty())
      rendered += "<br>";
    rendered += consan_capability_form_name(form);
    if (disposition != ConSanCapabilityDisposition::Supported) {
      rendered += " (";
      rendered += consan_capability_disposition_name(disposition);
      rendered += ')';
    }
  }
  return rendered.empty() ? "not applicable" : rendered;
}

} // namespace

int main() {
  std::cout << "<!-- BEGIN GENERATED CONSAN CAPABILITY CONTRACT -->\n"
               "| Target | Engine | Access | Barrier | Atomic | Fence |\n"
               "| --- | --- | --- | --- | --- | --- |\n";
  for (const ConSanCapabilityTarget &target : kConSanCapabilityTargets) {
    for (ConSanCapabilityEngine engine : kConSanCapabilityEngines) {
      std::cout << "| `" << rj_code_target_name(target.target) << "` | "
                << consan_capability_engine_name(engine);
      for (ConSanCapabilityDomain domain : kConSanCapabilityDomains)
        std::cout << " | " << render_domain(target.target, engine, domain);
      std::cout << " |\n";
    }
  }
  std::cout << "<!-- END GENERATED CONSAN CAPABILITY CONTRACT -->\n";
}
