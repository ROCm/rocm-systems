//===- hotswap_tool_test.cpp - Tests for gfx1250-A0 agent gating ---------===//
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for agent_is_gfx1250_a0() in hotswap_tool.cpp.
//
// The tool translation unit is #included directly so its internal
// (anonymous-namespace) helpers — agent_is_gfx1250_a0() and
// topology_node_base_dirs() — are reachable from the tests. The HSA, libdrm and
// COMGR entry points the tool calls are replaced with in-file stubs (linked in
// place of the real libraries) so the gating decision can be driven entirely
// from the test without GPU hardware:
//
//   * ISA name              <- hsa_agent_iterate_isas / hsa_isa_get_info_alt
//   * KFD node id           <- hsa_agent_get_info
//   * DRM render minor      <- real sysfs parser pointed at a temp directory
//   * chip revision         <- drmOpenRender / amdgpu_* stubs
//
//===----------------------------------------------------------------------===//

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Test-controlled fake environment. Set before each call into the tool.
// ---------------------------------------------------------------------------
namespace {
struct FakeEnv {
  std::string isa_name;          // ISA reported for the agent
  bool node_id_ok = true;        // hsa_agent_get_info success
  uint32_t node_id = 0;          // KFD topology node id
  bool drm_open_ok = true;       // drmOpenRender success
  bool device_init_ok = true;    // amdgpu_device_initialize success
  bool query_info_ok = true;     // amdgpu_query_info success
  int pci_rev = 0;               // PCI Config Space revision id read by the
                                 // tool (drm_amdgpu_info_device::pci_rev;
                                 // 0 == A0 for the gating logic)

  // Counters used to assert short-circuiting and caching behavior.
  int isa_query_calls = 0;       // get_agent_isa_name -> iterate_isas
  int agent_get_info_calls = 0;  // node-id queries
  int drm_open_calls = 0;        // query_chip_rev reached drmOpenRender
};
FakeEnv g_env;
}  // namespace

// The tool TU (brings in hsa.h / amdgpu.h / xf86drm.h and the code under test).
#include "hotswap_tool.cpp"

// ---------------------------------------------------------------------------
// Stubs replacing the real HSA / libdrm symbols referenced by the tool.
// ---------------------------------------------------------------------------
extern "C" {

hsa_status_t hsa_agent_iterate_isas(hsa_agent_t /*agent*/,
                                    hsa_status_t (*callback)(hsa_isa_t isa,
                                                             void *data),
                                    void *data) {
  ++g_env.isa_query_calls;
  hsa_isa_t isa{};
  isa.handle = 1;
  return callback(isa, data);
}

hsa_status_t hsa_isa_get_info_alt(hsa_isa_t /*isa*/, hsa_isa_info_t attribute,
                                  void *value) {
  if (attribute == HSA_ISA_INFO_NAME_LENGTH) {
    *static_cast<uint32_t *>(value) =
        static_cast<uint32_t>(g_env.isa_name.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  if (attribute == HSA_ISA_INFO_NAME) {
    std::memcpy(value, g_env.isa_name.c_str(), g_env.isa_name.size() + 1);
    return HSA_STATUS_SUCCESS;
  }
  return HSA_STATUS_ERROR;
}

hsa_status_t hsa_agent_get_info(hsa_agent_t /*agent*/,
                                hsa_agent_info_t /*attribute*/, void *value) {
  ++g_env.agent_get_info_calls;
  if (!g_env.node_id_ok) {
    return HSA_STATUS_ERROR;
  }
  *static_cast<uint32_t *>(value) = g_env.node_id;
  return HSA_STATUS_SUCCESS;
}

int drmOpenRender(int /*minor*/) {
  ++g_env.drm_open_calls;
  return g_env.drm_open_ok ? 42 : -1;
}

int drmClose(int /*fd*/) { return 0; }

int amdgpu_device_initialize(int /*fd*/, uint32_t *major, uint32_t *minor,
                             amdgpu_device_handle *handle) {
  if (major) *major = 1;
  if (minor) *minor = 0;
  if (!g_env.device_init_ok) {
    return -1;
  }
  static int dummy_device = 0;
  *handle = reinterpret_cast<amdgpu_device_handle>(&dummy_device);
  return 0;
}

int amdgpu_device_deinitialize(amdgpu_device_handle /*handle*/) { return 0; }

int amdgpu_query_info(amdgpu_device_handle /*dev*/, unsigned info_id,
                      unsigned size, void *value) {
  if (!g_env.query_info_ok) {
    return -1;
  }
  if (info_id == AMDGPU_INFO_DEV_INFO) {
    std::memset(value, 0, size);
    auto *info = static_cast<struct drm_amdgpu_info_device *>(value);
    // Production query_chip_rev() reads dev_info.pci_rev, so drive that field.
    info->pci_rev = static_cast<uint32_t>(g_env.pci_rev);
    return 0;
  }
  return -1;
}

}  // extern "C"

// COMGR rewrite is not exercised here; provide a stub so the tool TU links.
namespace rocr::hotswap {
int RetargetCodeObject(const void *elf_data, size_t elf_size,
                       const char * /*source_isa*/, const char * /*target_isa*/,
                       void **out_data, size_t *out_size) {
  if (out_data) {
    *out_data = const_cast<void *>(elf_data);
  }
  if (out_size) {
    *out_size = elf_size;
  }
  return -1;
}
}  // namespace rocr::hotswap

// ---------------------------------------------------------------------------
// Minimal test harness.
// ---------------------------------------------------------------------------
namespace {

int tests_passed = 0;
int tests_failed = 0;

void check(bool cond, const char *name) {
  if (cond) {
    ++tests_passed;
    printf("  PASS: %s\n", name);
  } else {
    ++tests_failed;
    fprintf(stderr, "  FAIL: %s\n", name);
  }
}

void reset_env() { g_env = FakeEnv{}; }

uint64_t g_next_handle = 1000;
hsa_agent_t fresh_agent() {
  hsa_agent_t a{};
  a.handle = g_next_handle++;
  return a;
}

const char *kGfx1250Isa = "amdgcn-amd-amdhsa--gfx1250";
const char *kGfx942Isa = "amdgcn-amd-amdhsa--gfx942";

// Creates a temp topology base dir containing <node_id>/properties with the
// given drm_render_minor, points the tool at it, and returns the base dir.
std::string set_topology_with_minor(uint32_t node_id, int render_minor) {
  char tmpl[] = "/tmp/hotswap_topo_XXXXXX";
  const char *base = mkdtemp(tmpl);
  if (!base) {
    return {};
  }
  const std::string node_dir = std::string(base) + "/" + std::to_string(node_id);
  mkdir(node_dir.c_str(), 0755);
  std::ofstream props(node_dir + "/properties");
  props << "cpu_cores_count 0\n"
        << "drm_render_minor " << render_minor << "\n"
        << "simd_count 4\n";
  props.close();
  const std::string base_with_slash = std::string(base) + "/";
  topology_node_base_dirs() = {base_with_slash};
  return base_with_slash;
}

// Points the tool at an empty temp dir (no node properties -> render minor
// lookup fails).
void set_empty_topology() {
  char tmpl[] = "/tmp/hotswap_topo_XXXXXX";
  const char *base = mkdtemp(tmpl);
  if (base) {
    topology_node_base_dirs() = {std::string(base) + "/"};
  } else {
    topology_node_base_dirs() = {"/nonexistent_hotswap_topo/"};
  }
}

// (7) gfx1250 silicon at pci_rev A0 -> true.
void test_Gfx1250A0IsTrue() {
  printf("TEST Gfx1250A0IsTrue...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.node_id = 3;
  g_env.pci_rev = 0;
  set_topology_with_minor(3, 128);
  check(agent_is_gfx1250_a0(fresh_agent()) == true,
        "gfx1250 + pci_rev A0 -> true");
}

// (8) Non-gfx1250 ISA -> false, and the DRM/node-id path is never touched.
void test_NonGfx1250ShortCircuits() {
  printf("TEST NonGfx1250ShortCircuits...\n");
  reset_env();
  g_env.isa_name = kGfx942Isa;
  check(agent_is_gfx1250_a0(fresh_agent()) == false, "non-gfx1250 -> false");
  check(g_env.agent_get_info_calls == 0,
        "non-gfx1250 does not query KFD node id");
  check(g_env.drm_open_calls == 0, "non-gfx1250 does not query DRM");
}

// (9) gfx1250 but a non-A0 stepping (pci_rev != 0) -> false.
void test_Gfx1250NonA0IsFalse() {
  printf("TEST Gfx1250NonA0IsFalse...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.node_id = 5;
  g_env.pci_rev = 1;  // A1
  set_topology_with_minor(5, 128);
  check(agent_is_gfx1250_a0(fresh_agent()) == false,
        "gfx1250 + pci_rev != 0 -> false");
}

// (10) Node-id query failure -> false, and DRM is not queried.
void test_NodeIdQueryFailure() {
  printf("TEST NodeIdQueryFailure...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.node_id_ok = false;
  check(agent_is_gfx1250_a0(fresh_agent()) == false,
        "node-id query failure -> false");
  check(g_env.drm_open_calls == 0,
        "no DRM query when node id is unavailable");
}

// (11) render-minor lookup failure (-1) -> false, and DRM is not queried.
void test_RenderMinorUnavailable() {
  printf("TEST RenderMinorUnavailable...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.node_id = 7;
  set_empty_topology();
  check(agent_is_gfx1250_a0(fresh_agent()) == false,
        "render minor unavailable -> false");
  check(g_env.drm_open_calls == 0,
        "no DRM query when render minor is unavailable");
}

// (12) chip-rev query failure (query_chip_rev returns -1) -> false.
void test_ChipRevQueryFailure() {
  printf("TEST ChipRevQueryFailure...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.node_id = 9;
  g_env.drm_open_ok = false;  // drmOpenRender fails -> chip_rev stays -1
  set_topology_with_minor(9, 128);
  check(agent_is_gfx1250_a0(fresh_agent()) == false,
        "query_chip_rev failure -> false");
  check(g_env.drm_open_calls == 1, "DRM open was attempted");
}

// (13) Result is cached per agent handle: a repeat call does not re-query.
void test_ResultIsCachedPerHandle() {
  printf("TEST ResultIsCachedPerHandle...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.node_id = 11;
  g_env.pci_rev = 0;
  set_topology_with_minor(11, 128);
  const hsa_agent_t agent = fresh_agent();
  const bool first = agent_is_gfx1250_a0(agent);
  const int after_first = g_env.isa_query_calls;
  const bool second = agent_is_gfx1250_a0(agent);
  check(first && second, "cached result stays true");
  check(g_env.isa_query_calls == after_first,
        "second call served from cache (no re-query)");
}

// (14) Distinct handles are evaluated independently (not conflated by cache).
void test_DistinctHandlesIndependent() {
  printf("TEST DistinctHandlesIndependent...\n");
  reset_env();
  g_env.isa_name = kGfx1250Isa;
  g_env.node_id = 13;
  g_env.pci_rev = 0;
  set_topology_with_minor(13, 128);
  const bool a1 = agent_is_gfx1250_a0(fresh_agent());

  // Different agent, different ISA: must be re-evaluated, not cached as a1.
  g_env.isa_name = kGfx942Isa;
  const bool a2 = agent_is_gfx1250_a0(fresh_agent());
  check(a1 == true, "first handle (gfx1250 A0) -> true");
  check(a2 == false, "distinct handle (gfx942) evaluated independently");
}

}  // namespace

int main() {
  test_Gfx1250A0IsTrue();
  test_NonGfx1250ShortCircuits();
  test_Gfx1250NonA0IsFalse();
  test_NodeIdQueryFailure();
  test_RenderMinorUnavailable();
  test_ChipRevQueryFailure();
  test_ResultIsCachedPerHandle();
  test_DistinctHandlesIndependent();

  printf("\n%d passed, %d failed\n", tests_passed, tests_failed);
  return tests_failed ? EXIT_FAILURE : EXIT_SUCCESS;
}