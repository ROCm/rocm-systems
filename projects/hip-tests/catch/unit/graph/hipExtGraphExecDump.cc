/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
Test Case Scenarios :
Negative -
1) Pass graphExec as nullptr to hipExtGraphExecDumpCreate and verify hipErrorInvalidValue.
2) Pass dumpOut as nullptr to hipExtGraphExecDumpCreate and verify hipErrorInvalidValue.
3) Pass dump as nullptr to hipExtGraphExecDumpGet and verify hipErrorInvalidValue.
4) Pass result as nullptr to hipExtGraphExecDumpGet and verify hipErrorInvalidValue.
5) Pass out-of-range index with CodeObject query and verify hipErrorInvalidValue.
6) Pass invalid query enum to hipExtGraphExecDumpGet and verify hipErrorInvalidValue.
7) Pass dump as nullptr to hipExtGraphExecDumpDestroy and verify hipErrorInvalidValue.
Functional -
1) Create a graph with kernel nodes that have pointer args, by-value uint args,
   and a by-value uint16_t[8] array arg. Dump the graph exec and verify:
   - JSON contains correct structure (node count, kernel name, dispatch dims)
   - Pointer args are marked as "PTR"
   - By-value uint args contain correct hex values
   - By-value array arg contains correct hex bytes
   - Code object binaries are accessible in-memory with valid hash and size
2) Test with multiple kernel nodes and dependencies between them, including
   code object deduplication across nodes sharing the same binary.
3) Test that flags control what gets captured in the in-memory dump.
*/

#include <hip_test_common.hh>

#include <sstream>
#include <cstring>

// Kernel with pointer args + by-value uint32_t scalars
__global__ void kernel_scalar(float* output, const float* input, uint32_t count,
                               uint32_t multiplier) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    output[i] = input[i] * multiplier;
  }
}

// Struct to pass uint16_t[8] as a by-value argument
struct ShortArray8 {
  uint16_t data[8];
};

// Kernel with pointer arg + by-value uint16_t[8] array arg + by-value uint32_t
__global__ void kernel_array_arg(float* output, ShortArray8 coeffs, uint32_t n) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    output[i] = static_cast<float>(coeffs.data[i % 8]);
  }
}

// Helper: get the JSON string from a dump handle
static std::string getDumpJson(hipExtGraphExecDump_t dump) {
  hipExtGraphExecDumpResult_t r;
  HIP_CHECK(hipExtGraphExecDumpGet(dump, hipExtGraphExecDumpQueryJson, 0, &r));
  return std::string(r.json.ptr, r.json.len);
}

// Helper: find a JSON string or numeric value for a key (simple substring search)
static std::string findJsonValue(const std::string& json, const std::string& key) {
  std::string searchKey = "\"" + key + "\": ";
  auto pos = json.find(searchKey);
  if (pos == std::string::npos) return "";
  pos += searchKey.size();
  if (json[pos] == '"') {
    pos++;
    auto end = json.find('"', pos);
    return json.substr(pos, end - pos);
  } else {
    auto end = json.find_first_of(",}\n", pos);
    return json.substr(pos, end - pos);
  }
}

// Helper: count occurrences of a substring
static int countOccurrences(const std::string& str, const std::string& sub) {
  int count = 0;
  size_t pos = 0;
  while ((pos = str.find(sub, pos)) != std::string::npos) {
    count++;
    pos += sub.size();
  }
  return count;
}

/* Test verifies hipExtGraphExecDumpCreate/Get/Destroy negative scenarios. */
HIP_TEST_CASE(Unit_hipExtGraphExecDump_Negative) {
  hipGraph_t graph;
  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Add an empty node so the graph can be instantiated
  hipGraphNode_t emptyNode;
  HIP_CHECK(hipGraphAddEmptyNode(&emptyNode, graph, nullptr, 0));
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  hipExtGraphExecDump_t dump = nullptr;

  SECTION("hipExtGraphExecDumpCreate: graphExec nullptr") {
    HIP_CHECK_ERROR(hipExtGraphExecDumpCreate(nullptr, 0, &dump), hipErrorInvalidValue);
  }

  SECTION("hipExtGraphExecDumpCreate: dumpOut nullptr") {
    HIP_CHECK_ERROR(hipExtGraphExecDumpCreate(graphExec, 0, nullptr), hipErrorInvalidValue);
  }

  // Create a valid dump for the remaining negative tests
  HIP_CHECK(hipExtGraphExecDumpCreate(graphExec, hipExtGraphExecDumpFlagsNone, &dump));
  REQUIRE(dump != nullptr);

  SECTION("hipExtGraphExecDumpGet: dump nullptr") {
    hipExtGraphExecDumpResult_t r;
    HIP_CHECK_ERROR(hipExtGraphExecDumpGet(nullptr, hipExtGraphExecDumpQueryJson, 0, &r),
                    hipErrorInvalidValue);
  }

  SECTION("hipExtGraphExecDumpGet: result nullptr") {
    HIP_CHECK_ERROR(hipExtGraphExecDumpGet(dump, hipExtGraphExecDumpQueryJson, 0, nullptr),
                    hipErrorInvalidValue);
  }

  SECTION("hipExtGraphExecDumpGet: CodeObject out-of-range index") {
    hipExtGraphExecDumpResult_t r;
    // No code objects captured (FlagsNone), so any index is out of range
    HIP_CHECK_ERROR(hipExtGraphExecDumpGet(dump, hipExtGraphExecDumpQueryCodeObject, 0, &r),
                    hipErrorInvalidValue);
  }

  SECTION("hipExtGraphExecDumpGet: invalid query enum") {
    hipExtGraphExecDumpResult_t r;
    HIP_CHECK_ERROR(
        hipExtGraphExecDumpGet(dump, static_cast<hipExtGraphExecDumpQuery_t>(999), 0, &r),
        hipErrorInvalidValue);
  }

  SECTION("hipExtGraphExecDumpDestroy: dump nullptr") {
    HIP_CHECK_ERROR(hipExtGraphExecDumpDestroy(nullptr), hipErrorInvalidValue);
  }

  HIP_CHECK(hipExtGraphExecDumpDestroy(dump));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
}

/* Test verifies hipExtGraphExecDump captures by-value scalar args correctly. */
HIP_TEST_CASE(Unit_hipExtGraphExecDump_ScalarArgs) {
  constexpr int N = 1024;
  constexpr int blockSize = 256;
  constexpr int gridSize = (N + blockSize - 1) / blockSize;

  float *d_input, *d_output;
  HIP_CHECK(hipMalloc(&d_input, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  uint32_t count = N;        // 0x00000400
  uint32_t multiplier = 42;  // 0x0000002a
  void* args[] = {&d_output, &d_input, &count, &multiplier};

  hipKernelNodeParams kparams{};
  kparams.func = reinterpret_cast<void*>(kernel_scalar);
  kparams.gridDim = dim3(gridSize);
  kparams.blockDim = dim3(blockSize);
  kparams.kernelParams = reinterpret_cast<void**>(args);
  kparams.sharedMemBytes = 0;

  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kparams));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  hipExtGraphExecDump_t dump = nullptr;
  HIP_CHECK(hipExtGraphExecDumpCreate(graphExec, hipExtGraphExecDumpFlagsAll, &dump));
  REQUIRE(dump != nullptr);

  std::string json = getDumpJson(dump);
  REQUIRE(!json.empty());
  INFO("JSON content: " << json);

  // Verify node counts
  REQUIRE(findJsonValue(json, "node_count") == "1");
  REQUIRE(findJsonValue(json, "kernel_node_count") == "1");

  // Verify kernel name
  REQUIRE(json.find("kernel_scalar") != std::string::npos);

  // Verify dispatch dimensions
  REQUIRE(json.find("\"gridDim\": { \"x\": 4, \"y\": 1, \"z\": 1 }") != std::string::npos);
  REQUIRE(json.find("\"blockDim\": { \"x\": 256, \"y\": 1, \"z\": 1 }") != std::string::npos);

  // Verify pointer args (output, input) are marked as PTR
  REQUIRE(countOccurrences(json, "\"kind\": \"pointer\"") == 2);
  REQUIRE(countOccurrences(json, "\"value\": \"PTR\"") == 2);

  // Verify by-value args (count, multiplier) with correct hex values
  REQUIRE(countOccurrences(json, "\"kind\": \"value\"") == 2);
  REQUIRE(json.find("\"0x00000400\"") != std::string::npos);  // count = 1024
  REQUIRE(json.find("\"0x0000002a\"") != std::string::npos);  // multiplier = 42

  // Verify code objects are accessible in-memory
  hipExtGraphExecDumpResult_t r;
  HIP_CHECK(hipExtGraphExecDumpGet(dump, hipExtGraphExecDumpQueryCodeObjectCount, 0, &r));
  size_t coCount = r.count;  // save before loop — each CodeObject call overwrites the union
  REQUIRE(coCount > 0);
  for (size_t i = 0; i < coCount; i++) {
    HIP_CHECK(hipExtGraphExecDumpGet(dump, hipExtGraphExecDumpQueryCodeObject, i, &r));
    REQUIRE(r.codeObject.hash != nullptr);
    REQUIRE(r.codeObject.data != nullptr);
    REQUIRE(r.codeObject.size > 0);
    REQUIRE(strlen(r.codeObject.hash) == 16);  // 16-char hex hash
  }

  // Verify the hash is referenced in the JSON
  REQUIRE(json.find("code_object_hash") != std::string::npos);

  HIP_CHECK(hipExtGraphExecDumpDestroy(dump));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_input));
  HIP_CHECK(hipFree(d_output));
}

/* Test verifies hipExtGraphExecDump captures a by-value uint16_t[8] array arg. */
HIP_TEST_CASE(Unit_hipExtGraphExecDump_ArrayArg) {
  constexpr int N = 256;
  constexpr int blockSize = 64;
  constexpr int gridSize = (N + blockSize - 1) / blockSize;

  float* d_output;
  HIP_CHECK(hipMalloc(&d_output, N * sizeof(float)));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  ShortArray8 coeffs;
  coeffs.data[0] = 0x1111;
  coeffs.data[1] = 0x2222;
  coeffs.data[2] = 0x3333;
  coeffs.data[3] = 0x4444;
  coeffs.data[4] = 0x5555;
  coeffs.data[5] = 0x6666;
  coeffs.data[6] = 0x7777;
  coeffs.data[7] = 0x8888;
  uint32_t n = N;

  void* args[] = {&d_output, &coeffs, &n};
  hipKernelNodeParams kparams{};
  kparams.func = reinterpret_cast<void*>(kernel_array_arg);
  kparams.gridDim = dim3(gridSize);
  kparams.blockDim = dim3(blockSize);
  kparams.kernelParams = reinterpret_cast<void**>(args);
  kparams.sharedMemBytes = 0;

  hipGraphNode_t kNode;
  HIP_CHECK(hipGraphAddKernelNode(&kNode, graph, nullptr, 0, &kparams));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  hipExtGraphExecDump_t dump = nullptr;
  HIP_CHECK(hipExtGraphExecDumpCreate(graphExec, hipExtGraphExecDumpFlagsAll, &dump));
  REQUIRE(dump != nullptr);

  std::string json = getDumpJson(dump);
  REQUIRE(!json.empty());
  INFO("JSON content: " << json);

  // Verify kernel name
  REQUIRE(json.find("kernel_array_arg") != std::string::npos);

  // Verify pointer arg (d_output) is marked as PTR
  REQUIRE(countOccurrences(json, "\"kind\": \"pointer\"") == 1);

  // Verify by-value args: the struct (16 bytes) + uint32_t n
  REQUIRE(countOccurrences(json, "\"kind\": \"value\"") >= 2);

  // Verify n = 256 = 0x00000100
  REQUIRE(json.find("\"0x00000100\"") != std::string::npos);

  // Verify the array bytes are present in hex.
  // Little-endian stored, big-endian display: 0x88887777666655554444333322221111
  REQUIRE(json.find("88887777666655554444333322221111") != std::string::npos);

  HIP_CHECK(hipExtGraphExecDumpDestroy(dump));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_output));
}

/* Test verifies hipExtGraphExecDump with multiple nodes and dependencies. */
HIP_TEST_CASE(Unit_hipExtGraphExecDump_MultiNodeDeps) {
  constexpr int N = 512;
  constexpr int blockSize = 128;
  constexpr int gridSize = (N + blockSize - 1) / blockSize;

  float *d_buf1, *d_buf2;
  HIP_CHECK(hipMalloc(&d_buf1, N * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_buf2, N * sizeof(float)));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  // Node 0: kernel_scalar (no deps)
  uint32_t count = N;
  uint32_t mult1 = 7;
  void* args0[] = {&d_buf1, &d_buf2, &count, &mult1};
  hipKernelNodeParams kp0{};
  kp0.func = reinterpret_cast<void*>(kernel_scalar);
  kp0.gridDim = dim3(gridSize);
  kp0.blockDim = dim3(blockSize);
  kp0.kernelParams = reinterpret_cast<void**>(args0);

  hipGraphNode_t node0;
  HIP_CHECK(hipGraphAddKernelNode(&node0, graph, nullptr, 0, &kp0));

  // Node 1: kernel_scalar (depends on node0)
  uint32_t mult2 = 13;
  void* args1[] = {&d_buf2, &d_buf1, &count, &mult2};
  hipKernelNodeParams kp1{};
  kp1.func = reinterpret_cast<void*>(kernel_scalar);
  kp1.gridDim = dim3(gridSize);
  kp1.blockDim = dim3(blockSize);
  kp1.kernelParams = reinterpret_cast<void**>(args1);

  hipGraphNode_t node1;
  HIP_CHECK(hipGraphAddKernelNode(&node1, graph, &node0, 1, &kp1));

  // Node 2: kernel_array_arg (depends on node0, parallel with node1)
  ShortArray8 coeffs;
  for (int i = 0; i < 8; i++) coeffs.data[i] = static_cast<uint16_t>(i + 1);
  uint32_t n2 = N;
  void* args2[] = {&d_buf2, &coeffs, &n2};
  hipKernelNodeParams kp2{};
  kp2.func = reinterpret_cast<void*>(kernel_array_arg);
  kp2.gridDim = dim3(gridSize);
  kp2.blockDim = dim3(blockSize);
  kp2.kernelParams = reinterpret_cast<void**>(args2);

  hipGraphNode_t node2;
  HIP_CHECK(hipGraphAddKernelNode(&node2, graph, &node0, 1, &kp2));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  hipExtGraphExecDump_t dump = nullptr;
  HIP_CHECK(hipExtGraphExecDumpCreate(graphExec, hipExtGraphExecDumpFlagsAll, &dump));
  REQUIRE(dump != nullptr);

  std::string json = getDumpJson(dump);
  REQUIRE(!json.empty());
  INFO("JSON content: " << json);

  // Verify 3 nodes
  REQUIRE(findJsonValue(json, "node_count") == "3");
  REQUIRE(findJsonValue(json, "kernel_node_count") == "3");

  // Verify node0 has no deps; node1 and node2 both depend on node0
  REQUIRE(json.find("\"dependencies\": []") != std::string::npos);
  REQUIRE(countOccurrences(json, "\"dependencies\": [0]") == 2);

  // Verify both kernel names appear
  REQUIRE(json.find("kernel_scalar") != std::string::npos);
  REQUIRE(json.find("kernel_array_arg") != std::string::npos);

  // Verify by-value scalar args: mult1=7 (0x00000007), mult2=13 (0x0000000d)
  REQUIRE(json.find("\"0x00000007\"") != std::string::npos);
  REQUIRE(json.find("\"0x0000000d\"") != std::string::npos);

  // Verify code object deduplication: two kernels but kernel_scalar appears twice —
  // only unique binaries are stored, so expect at most 2 code objects
  hipExtGraphExecDumpResult_t r;
  HIP_CHECK(hipExtGraphExecDumpGet(dump, hipExtGraphExecDumpQueryCodeObjectCount, 0, &r));
  size_t coCount = r.count;  // save before loop — each CodeObject call overwrites the union
  REQUIRE(coCount > 0);
  REQUIRE(coCount <= 2);

  HIP_CHECK(hipExtGraphExecDumpDestroy(dump));
  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_buf1));
  HIP_CHECK(hipFree(d_buf2));
}

/* Test verifies flags control what gets captured in the in-memory dump. */
HIP_TEST_CASE(Unit_hipExtGraphExecDump_FlagsControl) {
  constexpr int N = 64;

  float* d_buf;
  HIP_CHECK(hipMalloc(&d_buf, N * sizeof(float)));

  hipGraph_t graph;
  HIP_CHECK(hipGraphCreate(&graph, 0));

  uint32_t count = N;
  uint32_t mult = 1;
  void* args[] = {&d_buf, &d_buf, &count, &mult};
  hipKernelNodeParams kp{};
  kp.func = reinterpret_cast<void*>(kernel_scalar);
  kp.gridDim = dim3(1);
  kp.blockDim = dim3(N);
  kp.kernelParams = reinterpret_cast<void**>(args);

  hipGraphNode_t node;
  HIP_CHECK(hipGraphAddKernelNode(&node, graph, nullptr, 0, &kp));

  hipGraphExec_t graphExec;
  HIP_CHECK(hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

  SECTION("No flags - only basic node info") {
    hipExtGraphExecDump_t dump = nullptr;
    HIP_CHECK(hipExtGraphExecDumpCreate(graphExec, hipExtGraphExecDumpFlagsNone, &dump));
    std::string json = getDumpJson(dump);
    REQUIRE(!json.empty());
    REQUIRE(json.find("arguments") == std::string::npos);
    REQUIRE(json.find("gridDim") == std::string::npos);
    REQUIRE(json.find("dependencies") == std::string::npos);

    // No code objects captured
    hipExtGraphExecDumpResult_t r;
    HIP_CHECK(hipExtGraphExecDumpGet(dump, hipExtGraphExecDumpQueryCodeObjectCount, 0, &r));
    REQUIRE(r.count == 0);

    HIP_CHECK(hipExtGraphExecDumpDestroy(dump));
  }

  SECTION("Dispatch flag only") {
    hipExtGraphExecDump_t dump = nullptr;
    HIP_CHECK(hipExtGraphExecDumpCreate(graphExec, hipExtGraphExecDumpFlagsDispatch, &dump));
    std::string json = getDumpJson(dump);
    REQUIRE(!json.empty());
    REQUIRE(json.find("gridDim") != std::string::npos);
    REQUIRE(json.find("arguments") == std::string::npos);
    HIP_CHECK(hipExtGraphExecDumpDestroy(dump));
  }

  SECTION("KernelArgs flag only") {
    hipExtGraphExecDump_t dump = nullptr;
    HIP_CHECK(hipExtGraphExecDumpCreate(graphExec, hipExtGraphExecDumpFlagsKernelArgs, &dump));
    std::string json = getDumpJson(dump);
    REQUIRE(!json.empty());
    REQUIRE(json.find("arguments") != std::string::npos);
    REQUIRE(json.find("gridDim") == std::string::npos);
    HIP_CHECK(hipExtGraphExecDumpDestroy(dump));
  }

  HIP_CHECK(hipGraphExecDestroy(graphExec));
  HIP_CHECK(hipGraphDestroy(graph));
  HIP_CHECK(hipFree(d_buf));
}

// ---------------------------------------------------------------------------
// Example: dumping a graph exec to files on disk.
//
// This shows the intended end-to-end usage of the API — create a dump,
// write the JSON metadata and ELF code objects to a directory, then clean up.
// Uncomment and adapt as needed.
// ---------------------------------------------------------------------------
//
// #include <filesystem>
// #include <fstream>
//
// void dumpGraphExecToDir(hipGraphExec_t exec, const std::filesystem::path& outDir) {
//   std::filesystem::create_directories(outDir);
//
//   hipExtGraphExecDump_t dump;
//   hipExtGraphExecDumpCreate(exec, hipExtGraphExecDumpFlagsAll, &dump);
//
//   hipExtGraphExecDumpResult_t r;
//
//   // Write JSON metadata
//   hipExtGraphExecDumpGet(dump, hipExtGraphExecDumpQueryJson, 0, &r);
//   std::ofstream json(outDir / "graph.json");
//   json.write(r.json.ptr, r.json.len);
//   json.close();
//
//   // Write code object ELF binaries (deduplicated by content hash)
//   hipExtGraphExecDumpGet(dump, hipExtGraphExecDumpQueryCodeObjectCount, 0, &r);
//   size_t count = r.count;
//   for (size_t i = 0; i < count; i++) {
//     hipExtGraphExecDumpGet(dump, hipExtGraphExecDumpQueryCodeObject, i, &r);
//     std::ofstream elf(outDir / (std::string(r.codeObject.hash) + ".elf"),
//                       std::ios::binary);
//     elf.write(reinterpret_cast<const char*>(r.codeObject.data), r.codeObject.size);
//   }
//
//   hipExtGraphExecDumpDestroy(dump);
//
//   // Cleanup: std::filesystem::remove_all(outDir);
// }
