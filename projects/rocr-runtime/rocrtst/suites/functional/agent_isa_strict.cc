/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Checks the ISA list every agent reports through hsa_agent_iterate_isas().
// A strict target is not a member of any generic family, so an agent whose
// first (native) ISA is a strict target must report that ISA and nothing else,
// and no name it reports may be a generic family name.
//
// On hardware whose agents have no strict target the invariant is vacuous and
// the test records the ISA list without failing.

#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "hsa/hsa.h"

namespace {

const char kStrictSuffix[] = "-strict";
const char kGenericMarker[] = "-generic";

struct AgentIsaList {
  std::string agent_name;
  std::vector<std::string> isa_names;
};

hsa_status_t CollectIsa(hsa_isa_t isa, void* data) {
  std::vector<std::string>* names =
      reinterpret_cast<std::vector<std::string>*>(data);

  uint32_t name_len = 0;
  hsa_status_t err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME_LENGTH,
                                          &name_len);
  if (err != HSA_STATUS_SUCCESS) return err;

  std::vector<char> buf(name_len + 1, '\0');
  err = hsa_isa_get_info_alt(isa, HSA_ISA_INFO_NAME, buf.data());
  if (err != HSA_STATUS_SUCCESS) return err;

  names->push_back(std::string(buf.data()));
  return HSA_STATUS_SUCCESS;
}

hsa_status_t CollectAgent(hsa_agent_t agent, void* data) {
  std::vector<AgentIsaList>* agents =
      reinterpret_cast<std::vector<AgentIsaList>*>(data);

  char name[64] = {0};
  hsa_status_t err = hsa_agent_get_info(agent, HSA_AGENT_INFO_NAME, name);
  if (err != HSA_STATUS_SUCCESS) return err;

  AgentIsaList entry;
  entry.agent_name = name;

  err = hsa_agent_iterate_isas(agent, CollectIsa, &entry.isa_names);
  if (err != HSA_STATUS_SUCCESS) return err;

  agents->push_back(entry);
  return HSA_STATUS_SUCCESS;
}

bool Contains(const std::string& haystack, const char* needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST(AgentIsaValidation, StrictTargetReportsNoGenericFamily) {
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_init());

  std::vector<AgentIsaList> agents;
  ASSERT_EQ(HSA_STATUS_SUCCESS, hsa_iterate_agents(CollectAgent, &agents));

  int strict_agents = 0;

  for (const AgentIsaList& a : agents) {
    std::cout << "Agent: " << a.agent_name << " reports "
              << a.isa_names.size() << " ISA(s)" << std::endl;
    for (const std::string& n : a.isa_names) {
      std::cout << "    " << n << std::endl;
    }

    if (a.isa_names.empty()) continue;

    if (!Contains(a.isa_names[0], kStrictSuffix)) {
      // Not a strict agent - nothing to assert for this agent.
      continue;
    }

    strict_agents++;

    EXPECT_EQ(a.isa_names.size(), 1u)
        << "Agent " << a.agent_name << " has a strict native ISA ("
        << a.isa_names[0] << ") and must report exactly one ISA.";

    for (const std::string& n : a.isa_names) {
      EXPECT_FALSE(Contains(n, kGenericMarker))
          << "Agent " << a.agent_name
          << " has a strict native ISA but reports generic ISA " << n << ".";
    }
  }

  if (strict_agents == 0) {
    std::cout << "No agent on this system reports a strict ISA - the strict "
                 "invariant was not exercised." << std::endl;
  } else {
    std::cout << "Strict agents checked: " << strict_agents << std::endl;
  }

  EXPECT_EQ(HSA_STATUS_SUCCESS, hsa_shut_down());
}
