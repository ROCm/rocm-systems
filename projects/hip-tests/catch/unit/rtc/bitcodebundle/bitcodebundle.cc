#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

#include <hip_test_common.hh>
#include <hip_test_filesystem.hh>


#include <cassert>
#include <cstddef>
#include <fstream>
#include <memory>
#include <iostream>
#include <iterator>
#include <vector>

#pragma clang diagnostic ignored "-Wuninitialized"


// Load pre-generated bundled bitcode file
std::vector<char> loadBundledBitcodeFile(const std::string& filepath) {
  std::ifstream in(filepath, std::ios::binary);
  if (!in) {
    throw std::runtime_error("Failed to open bundled bitcode file: " + filepath);
  }
  
  std::vector<char> bundled_bc((std::istreambuf_iterator<char>(in)),
                               std::istreambuf_iterator<char>());
  in.close();
  
  if (bundled_bc.empty()) {
    throw std::runtime_error("Bundled bitcode file is empty: " + filepath);
  }
  
  return bundled_bc;
}


TEST_CASE("Unit_RTC_BUNDLE_FROM_FILE")  {
  // Load pre-generated offload_bundle.hipfb from the build directory
  std::string bundle_path = "offload_bundle.hipfb";
  std::vector<char> bundled_bc = loadBundledBitcodeFile(bundle_path);
  
  hiprtcLinkState linkstate;
  HIPRTC_CHECK(hiprtcLinkCreate(0, nullptr, nullptr, &linkstate));
  HIPRTC_CHECK(hiprtcLinkAddData(linkstate, HIPRTC_JIT_INPUT_LLVM_BUNDLED_BITCODE,
                                 bundled_bc.data(), bundled_bc.size(),
                                 "offload_bundle", 0, nullptr, nullptr));
  
  void* finaldata;
  size_t finalsize = 0;
  HIPRTC_CHECK(hiprtcLinkComplete(linkstate, &finaldata, &finalsize));
  
  hipModule_t module;
  hipFunction_t kernel;
  HIP_CHECK(hipModuleLoadData(&module, finaldata));
  HIP_CHECK(hipModuleGetFunction(&kernel, module, "saxpy"));

  HIPRTC_CHECK(hiprtcLinkDestroy(linkstate));
  HIP_CHECK(hipModuleUnload(module));
}