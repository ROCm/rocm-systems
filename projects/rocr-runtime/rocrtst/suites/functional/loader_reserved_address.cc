/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <fcntl.h>
#include <unistd.h>

#include <cstring>
#include <iostream>
#include <string>

#include "suites/functional/loader_reserved_address.h"
#include "common/base_rocr_utils.h"
#include "common/common.h"
#include "common/helper_funcs.h"
#include "gtest/gtest.h"
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"

static size_t AlignUp(size_t value, size_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

LoaderReservedAddressTest::LoaderReservedAddressTest(void) : TestBase() {
  set_title("Loader Reserved Address Test");
  set_description(
      "A normal device allocation is rejected. A virtual-memory mapping the "
      "agent can execute accepts the image, and its bytes match a normal load.");
  set_kernel_file_name("test_case_template_kernels.hsaco");
}

LoaderReservedAddressTest::~LoaderReservedAddressTest(void) {}

void LoaderReservedAddressTest::SetUp(void) {
  TestBase::SetUp();
  if (test_skipped_) return;

  ASSERT_SUCCESS(rocrtst::SetDefaultAgents(this));
  ASSERT_SUCCESS(rocrtst::SetPoolsTypical(this));

  uint16_t minor = 0;
  bool supported = false;
  ASSERT_SUCCESS(
      hsa_system_major_extension_supported(HSA_EXTENSION_AMD_LOADER, 1, &minor, &supported));
  ASSERT_TRUE(supported);
  ASSERT_GE(minor, 4);
  ASSERT_SUCCESS(
      hsa_system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(loader_), &loader_));
  ASSERT_NE(loader_.hsa_ven_amd_loader_code_object_reader_get_load_size, nullptr);
  ASSERT_NE(loader_.hsa_ven_amd_loader_executable_load_agent_code_object_at_address, nullptr);

  std::string path = rocrtst::LocateKernelFile(kernel_file_name(), *gpu_device1());
  ASSERT_FALSE(path.empty());
  kernel_fd_ = open(path.c_str(), O_RDONLY);
  ASSERT_NE(kernel_fd_, -1) << "Failed to open " << path;
  ASSERT_SUCCESS(hsa_code_object_reader_create_from_file(kernel_fd_, &reader_));

  page_size_ = sysconf(_SC_PAGESIZE);
}

void LoaderReservedAddressTest::Run(void) { TestBase::Run(); }

void LoaderReservedAddressTest::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void LoaderReservedAddressTest::DisplayResults(void) const { TestBase::DisplayResults(); }

void LoaderReservedAddressTest::Close(void) {
  if (reader_.handle) {
    EXPECT_SUCCESS(hsa_code_object_reader_destroy(reader_));
    reader_.handle = 0;
  }
  if (kernel_fd_ != -1) {
    close(kernel_fd_);
    kernel_fd_ = -1;
  }
  TestBase::Close();
}

void LoaderReservedAddressTest::LoadAtAddressTest(void) {
  if (::testing::Test::HasFatalFailure()) return;
  bool supp = false;
  ASSERT_SUCCESS(hsa_system_get_info(HSA_AMD_SYSTEM_INFO_VIRTUAL_MEM_API_SUPPORTED, &supp));
  if (!supp) {
    if (verbosity() > 0) {
      std::cout << "    Virtual Memory API not supported on this system - Skipping." << std::endl;
    }
    return;
  }
  hsa_agent_t gpu = *gpu_device1();

  size_t load_size = 0;
  ASSERT_SUCCESS(loader_.hsa_ven_amd_loader_code_object_reader_get_load_size(reader_, &load_size));
  ASSERT_GT(load_size, 0u);

  size_t granule = 0;
  ASSERT_SUCCESS(hsa_amd_memory_pool_get_info(
      device_pool(), HSA_AMD_MEMORY_POOL_INFO_RUNTIME_ALLOC_GRANULE, &granule));
  const size_t align = granule > page_size_ ? granule : page_size_;
  const size_t span = AlignUp(load_size, align);

  // A normal pool allocation cannot be executable.
  void* plain = nullptr;
  ASSERT_SUCCESS(hsa_amd_memory_pool_allocate(device_pool(), span, 0, &plain));
  hsa_executable_t rejected = {0};
  ASSERT_SUCCESS(hsa_executable_create_alt(profile(), HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                           nullptr, &rejected));
  EXPECT_EQ(HSA_STATUS_ERROR_INVALID_ARGUMENT,
            loader_.hsa_ven_amd_loader_executable_load_agent_code_object_at_address(
                rejected, gpu, reader_, nullptr, plain, span, nullptr));
  EXPECT_SUCCESS(hsa_executable_destroy(rejected));
  EXPECT_SUCCESS(hsa_amd_memory_pool_free(plain));

  void* address = nullptr;
  ASSERT_SUCCESS(hsa_amd_vmem_address_reserve(&address, span, 0, 0));
  hsa_amd_vmem_alloc_handle_t handle = {0};
  ASSERT_SUCCESS(hsa_amd_vmem_handle_create(device_pool(), span, MEMORY_TYPE_NONE, 0, &handle));
  ASSERT_SUCCESS(hsa_amd_vmem_map(address, span, 0, handle, 0));
  hsa_amd_memory_access_desc_t access = {
      static_cast<hsa_access_permission_t>(HSA_ACCESS_PERMISSION_RW | HSA_ACCESS_PERMISSION_EX),
      gpu};
  ASSERT_SUCCESS(hsa_amd_vmem_set_access(address, span, &access, 1));

  // Load the same executable normally and in a custom spot then compare.
  hsa_executable_t normal = {0};
  hsa_loaded_code_object_t normal_lco = {0};
  ASSERT_SUCCESS(hsa_executable_create_alt(profile(), HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                           nullptr, &normal));
  ASSERT_SUCCESS(hsa_executable_load_agent_code_object(normal, gpu, reader_, nullptr, &normal_lco));

  hsa_executable_t placed = {0};
  hsa_loaded_code_object_t placed_lco = {0};
  ASSERT_SUCCESS(hsa_executable_create_alt(profile(), HSA_DEFAULT_FLOAT_ROUNDING_MODE_DEFAULT,
                                           nullptr, &placed));
  ASSERT_SUCCESS(loader_.hsa_ven_amd_loader_executable_load_agent_code_object_at_address(
      placed, gpu, reader_, nullptr, address, span, &placed_lco));
  ASSERT_SUCCESS(hsa_executable_freeze(normal, nullptr));
  ASSERT_SUCCESS(hsa_executable_freeze(placed, nullptr));

  uint64_t normal_base = 0, placed_base = 0, image_size = 0, placed_size = 0;
  ASSERT_SUCCESS(loader_.hsa_ven_amd_loader_loaded_code_object_get_info(
      normal_lco, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &normal_base));
  ASSERT_SUCCESS(loader_.hsa_ven_amd_loader_loaded_code_object_get_info(
      normal_lco, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &image_size));
  ASSERT_SUCCESS(loader_.hsa_ven_amd_loader_loaded_code_object_get_info(
      placed_lco, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &placed_base));
  ASSERT_SUCCESS(loader_.hsa_ven_amd_loader_loaded_code_object_get_info(
      placed_lco, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &placed_size));
  ASSERT_EQ(placed_base, reinterpret_cast<uint64_t>(address));
  ASSERT_EQ(image_size, load_size);
  ASSERT_EQ(placed_size, image_size);

  const void* normal_host = nullptr;
  const void* placed_host = nullptr;
  ASSERT_SUCCESS(loader_.hsa_ven_amd_loader_query_host_address(
      reinterpret_cast<const void*>(normal_base), &normal_host));
  ASSERT_SUCCESS(loader_.hsa_ven_amd_loader_query_host_address(address, &placed_host));

  // The image has no dynamic relocations, so the two loads match byte for byte.
  EXPECT_EQ(0, std::memcmp(normal_host, placed_host, image_size));

  EXPECT_SUCCESS(hsa_executable_destroy(normal));
  EXPECT_SUCCESS(hsa_executable_destroy(placed));

  // The mapping is still owned by the application after the executable is destroyed.
  EXPECT_SUCCESS(hsa_amd_vmem_unmap(address, span));
  EXPECT_SUCCESS(hsa_amd_vmem_handle_release(handle));
  EXPECT_SUCCESS(hsa_amd_vmem_address_free(address, span));
}
