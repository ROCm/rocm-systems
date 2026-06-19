/*
 * Test suite header for topo_expl
 */

#ifndef TOPO_EXPL_TESTS_H_
#define TOPO_EXPL_TESTS_H_

#include <vector>
#include <set>
#include <string>
#include "model_descs.h"

// Run the comprehensive test suite with command-line argument parsing
// Returns exit code (0 = all passed, 1 = some failed)
int run_test_suite_from_args(int argc, char* argv[]);

int run_test_suite(const std::set<int>& include_models,
                   const std::set<int>& include_nodes,
                   const std::set<int>& exclude_models,
                   const std::set<int>& exclude_nodes);

// Parse comma-separated list of integers
std::set<int> parse_comma_separated_ints(const std::string& str);

#endif // TOPO_EXPL_TESTS_H_
