/*
 * Test suite implementation for topo_expl
 * Tests all models with various node combinations (1, 2, 4, 8, 16)
 */

#include "topo_expl_tests.h"
#include "topo_expl_api.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <set>
#include <sstream>
#include <algorithm>

struct TestResult {
    int model_id;
    int num_nodes;
    bool success;
    std::string error_msg;
    int nranks;
    int nnodes;
};

static const int node_counts[] = {1, 2, 4, 8, 16};
static const int num_node_counts = sizeof(node_counts) / sizeof(*node_counts);

static std::vector<TestResult> test_results;

// External declaration to suppress verbose rank/host output during tests
extern thread_local int topoExplSuppressVerbose;

static bool run_single_test(int model_id, int num_nodes) {
    if (model_id >= num_models) {
        printf("ERROR: Invalid model_id %d\n", model_id);
        return false;
    }
    
    NodeModelDesc *desc = &model_descs[model_id];
    int gpusPerNode = 0;
    extractArchAndGpus(desc->description, &gpusPerNode);
    
    if (gpusPerNode == 0) {
        printf("WARN: Could not extract GPU count from model %d description\n", model_id);
        return false;
    }
    
    int expected_nranks = gpusPerNode * num_nodes;
    int expected_nnodes = num_nodes;
    
    // Suppress verbose rank/host output during test execution
    int suppress_verbose = topoExplSuppressVerbose;
    topoExplSuppressVerbose = 1;
    
    // Create topology explorer context
    TopoExplConfig config;
    config.xmlTopoFile = desc->filename;
    config.numNodes = num_nodes;
    
    TopoExplContext* context = nullptr;
    TopoExplResult result = topoExplCreate(&config, &context);
    
    TestResult test_result;
    test_result.model_id = model_id;
    test_result.num_nodes = num_nodes;
    test_result.success = (result == TOPO_EXPL_SUCCESS && context != nullptr);

    if (!test_result.success) {
        test_result.error_msg = "Context creation failed";
        test_result.nranks = 0;
        test_result.nnodes = 0;
        test_results.push_back(test_result);
        topoExplDestroy(context);
        topoExplSuppressVerbose = suppress_verbose;
        return false;
    }
    
    // Verify rank info for first and last rank
    bool rank_info_ok = true;
    for (int r : {0, expected_nranks - 1}) {
        int nodeId, cudaDev;
        uint64_t busId;
        if (topoExplGetRankInfo(context, r, &nodeId, &cudaDev, &busId) != TOPO_EXPL_SUCCESS) {
            rank_info_ok = false;
            break;
        }
    }
    
    // Try to get algorithm info and algo time for simple cases
    TopoExplAlgoInfo algoInfo, agAlgoInfo;
    TopoExplResult arAlgoResult = topoExplGetAlgoInfo(context, TOPO_FUNC_ALLREDUCE, 
                                                     16384, TOPO_DTYPE_FLOAT32, &algoInfo);
    TopoExplResult agAlgoResult = topoExplGetAlgoInfo(context, TOPO_FUNC_ALLGATHER, 
                                                     33554432, TOPO_DTYPE_FLOAT32, &agAlgoInfo);
    bool algo_info_ok = (arAlgoResult == TOPO_EXPL_SUCCESS && agAlgoResult == TOPO_EXPL_SUCCESS);

    float agTime = -1.0f;
    float arTime = -1.0f;
    TopoExplResult agTimeResult = topoExplGetAlgoTime(context, TOPO_FUNC_ALLGATHER, TOPO_ALGO_RING, TOPO_PROTO_SIMPLE, 8388608, &agTime);
    TopoExplResult arTimeResult = topoExplGetAlgoTime(context, TOPO_FUNC_ALLREDUCE, TOPO_ALGO_RING, TOPO_PROTO_LL, 262144, &arTime);
    bool algo_time_ok = (agTimeResult == TOPO_EXPL_SUCCESS && agTime >= 0.0f &&
                         arTimeResult == TOPO_EXPL_SUCCESS && arTime >= 0.0f);

    test_result.success = rank_info_ok && algo_info_ok && algo_time_ok;
    test_result.nranks = expected_nranks;
    test_result.nnodes = expected_nnodes;

    if (!test_result.success) {
        if (!rank_info_ok) {
            test_result.error_msg = "Failed to get rank info";
        } else if (!algo_info_ok) {
            test_result.error_msg = "Failed to get algorithm info";
        } else {
            test_result.error_msg = "No valid algo/proto time (tuning may be broken)";
        }
    }
    
    test_results.push_back(test_result);
    topoExplDestroy(context);
    
    topoExplSuppressVerbose = suppress_verbose;
    
    return test_result.success;
}

static void print_test_summary() {
    int total_tests = 0;
    int passed_tests = 0;
    int failed_tests = 0;
    
    printf("\n");
    printf("================================================================================\n");
    printf("TEST SUMMARY\n");
    printf("================================================================================\n");
    printf("\n");
    
    // Group by model
    for (int m = 0; m < num_models; m++) {
        std::vector<std::pair<int, TestResult>> model_results;
        
        for (const auto& result : test_results) {
            if (result.model_id == m) {
                model_results.push_back({result.num_nodes, result});
                total_tests++;
                if (result.success) {
                    passed_tests++;
                } else {
                    failed_tests++;
                }
            }
        }   
    }
    
    printf("TOTAL: %d tests | PASSED: %d | FAILED: %d\n", 
           total_tests, passed_tests, failed_tests);
    
    if (failed_tests > 0) {
        printf("\nFAILED TESTS:\n");
        printf("--------------------\n");
        for (const auto& result : test_results) {
            if (!result.success) {
                printf("Model %2d, %2d nodes : %s\n",
                       result.model_id, result.num_nodes, result.error_msg.c_str());
            }
        }
    }
    printf("\n");
}

int run_test_suite_from_args(int argc, char* argv[]) {
    // -t cannot be used with -m/--model, -n/--nodes, or -a/--arch
    if (cmdOptionExists(argv, argv + argc, {"-m", "--model", "-n", "--nodes", "-a", "--arch"})) {
        fprintf(stderr,
                "WARN: -m/--model, -n/--nodes, and -a/--arch cannot be used with -t (test mode).\n");
        return TOPO_EXPL_INVALID_ARG;
    }

    std::set<int> include_models;
    std::set<int> include_nodes;
    std::set<int> exclude_models;
    std::set<int> exclude_nodes;

    const char* val;
    if ((val = getCmdOption(argv, argv + argc, "--include-models")))
        include_models = parse_comma_separated_ints(val);
    if ((val = getCmdOption(argv, argv + argc, "--include-nodes")))
        include_nodes = parse_comma_separated_ints(val);
    if ((val = getCmdOption(argv, argv + argc, "--exclude-models")))
        exclude_models = parse_comma_separated_ints(val);
    if ((val = getCmdOption(argv, argv + argc, "--exclude-nodes")))
        exclude_nodes = parse_comma_separated_ints(val);
    
    int failed_count = run_test_suite(include_models, include_nodes, exclude_models, exclude_nodes);
    return (failed_count == 0) ? TOPO_EXPL_SUCCESS : TOPO_EXPL_ERROR;
}

std::set<int> parse_comma_separated_ints(const std::string& str) {
    std::set<int> result;
    std::stringstream ss(str);
    std::string item;
    
    while (std::getline(ss, item, ',')) {
        item.erase(0, item.find_first_not_of(" \t"));
        item.erase(item.find_last_not_of(" \t") + 1);
        
        if (!item.empty()) {
            int value = std::atoi(item.c_str());
            result.insert(value);
        }
    }
    
    return result;
}

int run_test_suite(const std::set<int>& include_models,
                   const std::set<int>& include_nodes,
                   const std::set<int>& exclude_models,
                   const std::set<int>& exclude_nodes) {
    test_results.clear();
    
    // Determine which models to test
    std::vector<int> models_to_test;
    for (int m = 0; m < num_models; m++) {
        // If include_models is specified, only test those models
        if (!include_models.empty() && include_models.find(m) == include_models.end()) {
            continue;
        }
        // Skip excluded models
        if (exclude_models.find(m) != exclude_models.end()) {
            continue;
        }
        models_to_test.push_back(m);
    }
    
    // Determine which node counts to test
    std::vector<int> nodes_to_test;
    if (!include_nodes.empty()) {
        for (int n : include_nodes) {
            if (exclude_nodes.find(n) == exclude_nodes.end()) {
                nodes_to_test.push_back(n);
            }
        }
    } else {
        // Use default node counts, excluding specified ones
        for (int i = 0; i < num_node_counts; i++) {
            if (exclude_nodes.find(node_counts[i]) == exclude_nodes.end()) {
                nodes_to_test.push_back(node_counts[i]);
            }
        }
    }
    
    printf("================================================================================\n");
    printf("Topology Explorer Test Suite\n");
    printf("================================================================================\n");
    printf("Testing %zu models with %zu node combination(s)\n", models_to_test.size(), nodes_to_test.size());
    printf("\n");
    
    printf("Number of models : %zu\n", models_to_test.size());
    printf("Number of nodes  : %zu\n", nodes_to_test.size());
    printf("Nodes to test    : ");
    for (size_t i = 0; i < nodes_to_test.size(); i++) {
        printf("%d", nodes_to_test[i]);
        if (i < nodes_to_test.size() - 1) printf(", ");
    }
    printf("\n");
    printf("Number of tests  : %zu\n\n", models_to_test.size() * nodes_to_test.size());
    
    // Run tests
    int test_num = 0;
    int total_tests = models_to_test.size() * nodes_to_test.size();
    
    for (int m : models_to_test) {
        for (int num_nodes : nodes_to_test) {
            test_num++;
            
            printf("[%3d/%3d] Testing model %2d with %2d nodes...", 
                   test_num, total_tests, m, num_nodes);
            fflush(stdout);
            
            bool success = run_single_test(m, num_nodes);
            
            if (success) {
                printf(" PASSED\n");
            } else {
                printf(" FAILED\n");
            }
        }
    }
    
    // Print summary
    print_test_summary();
    
    // Return number of failures
    int failed_count = 0;
    for (const auto& result : test_results) {
        if (!result.success) failed_count++;
    }
    
    return failed_count;
}
