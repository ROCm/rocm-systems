/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "vulkan_test.hh"

#include <iostream>
#include <algorithm>


VkFence VulkanTest::CreateFence() {
  VkFence fence;
  VkFenceCreateInfo fence_create_info = {};
  fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
  fence_create_info.flags = 0;
  VK_CHECK_RESULT(vkCreateFence(_device, &fence_create_info, nullptr, &fence));

  _fences.push_back(fence);
  return fence;
}

VkSemaphore VulkanTest::CreateExternalSemaphore(VkSemaphoreType sem_type, uint64_t initial_value) {
  VkExportSemaphoreCreateInfoKHR export_sem_create_info = {};
  VkSemaphoreTypeCreateInfo timeline_create_info = {};

  export_sem_create_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO_KHR;
  export_sem_create_info.handleTypes = _sem_handle_type;

  if (sem_type == VK_SEMAPHORE_TYPE_TIMELINE) {
    timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timeline_create_info.initialValue = initial_value;
    export_sem_create_info.pNext = &timeline_create_info;
  } else {
    export_sem_create_info.pNext = nullptr;
  }

  VkSemaphoreCreateInfo semaphore_create_info = {};
  semaphore_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  semaphore_create_info.pNext = &export_sem_create_info;

  VkSemaphore semaphore;
  VK_CHECK_RESULT(vkCreateSemaphore(_device, &semaphore_create_info, nullptr, &semaphore));

  _semaphores.push_back(semaphore);
  return semaphore;
}

hipExternalSemaphoreHandleDesc VulkanTest::BuildSemaphoreDescriptor(VkSemaphore vk_sem,
                                                                    VkSemaphoreType sem_type) {
  hipExternalSemaphoreHandleDesc sem_handle_desc = {};
  sem_handle_desc.type = VulkanSemHandleTypeToHIPHandleType(sem_type);
#ifdef _WIN64
  sem_handle_desc.handle.win32.handle = GetSemaphoreHandle(vk_sem);
#else
  sem_handle_desc.handle.fd = GetSemaphoreHandle(vk_sem);
#endif
  sem_handle_desc.flags = 0;

  return sem_handle_desc;
}

hipExternalMemoryHandleDesc VulkanTest::BuildMemoryDescriptor(VkDeviceMemory vk_mem,
                                                              uint32_t size) {
  hipExternalMemoryHandleDesc mem_handle_desc = {};
  mem_handle_desc.type = VulkanMemHandleTypeToHIPHandleType();
#ifdef _WIN64
  mem_handle_desc.handle.win32.handle = GetMemoryHandle(vk_mem);
#else
  mem_handle_desc.handle.fd = GetMemoryHandle(vk_mem);
#endif
  mem_handle_desc.size = size;

  return mem_handle_desc;
}

void VulkanTest::CreateInstance() {
  UNSCOPED_INFO("Not all of the required instance extensions are supported");
  REQUIRE(CheckExtensionSupport(_required_instance_extensions));
  if (_enable_validation) {
    EnableValidationLayer();
  }

  VkApplicationInfo app_info = {};
  app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
  app_info.apiVersion = VK_API_VERSION_1_2;

  VkInstanceCreateInfo create_info = {};
  create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  create_info.pApplicationInfo = &app_info;
  create_info.enabledExtensionCount = static_cast<uint32_t>(_required_instance_extensions.size());
  create_info.ppEnabledExtensionNames = _required_instance_extensions.data();
  create_info.enabledLayerCount = static_cast<uint32_t>(_enabled_layers.size());
  create_info.ppEnabledLayerNames = _enabled_layers.data();

  VK_CHECK_RESULT(vkCreateInstance(&create_info, nullptr, &_instance));
}

void VulkanTest::CreateDevice() {
  UNSCOPED_INFO("Not all of the required device extensions are supported");
  REQUIRE(CheckExtensionSupport(_required_device_extensions));

  FindPhysicalDevice();

  VkDeviceQueueCreateInfo queue_create_info = {};
  queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
  queue_create_info.queueFamilyIndex = _compute_family_queue_idx = GetComputeQueueFamilyIndex();
  queue_create_info.queueCount = 1;
  float queue_priorities = 1.0;
  queue_create_info.pQueuePriorities = &queue_priorities;

  VkPhysicalDeviceVulkan12Features features = {};
  features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
  features.timelineSemaphore = true;

  VkDeviceCreateInfo device_create_info = {};
  device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
  device_create_info.enabledLayerCount = _enabled_layers.size();
  device_create_info.ppEnabledLayerNames = _enabled_layers.data();
  device_create_info.enabledExtensionCount = _required_device_extensions.size();
  device_create_info.ppEnabledExtensionNames = _required_device_extensions.data();
  device_create_info.pQueueCreateInfos = &queue_create_info;
  device_create_info.queueCreateInfoCount = 1;
  device_create_info.pNext = &features;

  VK_CHECK_RESULT(vkCreateDevice(_physical_device, &device_create_info, nullptr, &_device));
  vkGetDeviceQueue(_device, _compute_family_queue_idx, 0, &_queue);
}

void VulkanTest::CreateCommandBuffer() {
  VkCommandPoolCreateInfo command_pool_create_info = {};
  command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
  command_pool_create_info.flags = 0;
  command_pool_create_info.queueFamilyIndex = _compute_family_queue_idx;
  VK_CHECK_RESULT(vkCreateCommandPool(_device, &command_pool_create_info, nullptr, &_command_pool));

  VkCommandBufferAllocateInfo command_buffer_allocate_info = {};
  command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  command_buffer_allocate_info.commandPool = _command_pool;
  command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  command_buffer_allocate_info.commandBufferCount = 1;
  VK_CHECK_RESULT(
      vkAllocateCommandBuffers(_device, &command_buffer_allocate_info, &_command_buffer));
}

bool VulkanTest::CheckExtensionSupport(std::vector<const char*> expected_extensions) {
  uint32_t extension_count = 0;
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, nullptr);
  std::vector<VkExtensionProperties> extension_properties(extension_count);
  vkEnumerateInstanceExtensionProperties(nullptr, &extension_count, extension_properties.data());

  std::vector<const char*> supported_extensions;
  supported_extensions.reserve(extension_count);
  std::transform(extension_properties.begin(), extension_properties.end(),
                 std::back_inserter(supported_extensions),
                 [](const auto& p) { return p.extensionName; });

  auto p = [](const char* l, const char* r) { return strcmp(l, r) < 0; };
  std::sort(expected_extensions.begin(), expected_extensions.end(), p);
  std::sort(supported_extensions.begin(), supported_extensions.end(), p);

  return std::includes(supported_extensions.begin(), supported_extensions.end(),
                       expected_extensions.begin(), expected_extensions.end(),
                       [](const char* l, const char* r) { return strcmp(l, r) == 0; });
}

void VulkanTest::EnableValidationLayer() {
  uint32_t layer_count = 0;
  vkEnumerateInstanceLayerProperties(&layer_count, nullptr);
  std::vector<VkLayerProperties> layer_properties(layer_count);
  vkEnumerateInstanceLayerProperties(&layer_count, layer_properties.data());
  const bool found_val_layer =
      std::any_of(layer_properties.cbegin(), layer_properties.cend(), [](const auto& props) {
        return strcmp(props.layerName, "VK_LAYER_KHRONOS_validation") == 0;
      });


  if (found_val_layer) {
    _enabled_layers.push_back("VK_LAYER_KHRONOS_validation");
  } else {
    UNSCOPED_INFO("Validation was requested, but the validation layer could not be located");
    REQUIRE(found_val_layer);
  }
}

uint32_t VulkanTest::GetComputeQueueFamilyIndex() {
  uint32_t queue_family_count = 0u;

  vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &queue_family_count, nullptr);
  std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
  vkGetPhysicalDeviceQueueFamilyProperties(_physical_device, &queue_family_count,
                                           queue_families.data());

  const auto it =
      std::find_if(queue_families.cbegin(), queue_families.cend(), [](const auto& props) {
        return props.queueCount > 0 && (props.queueFlags & VK_QUEUE_COMPUTE_BIT);
      });
  REQUIRE(it != queue_families.cend());

  return std::distance(queue_families.cbegin(), it);
}

void VulkanTest::FindPhysicalDevice() {
  uint32_t device_count = 0;
  vkEnumeratePhysicalDevices(_instance, &device_count, nullptr);
  REQUIRE(device_count != 0u);

  std::vector<VkPhysicalDevice> physical_devices(device_count);
  vkEnumeratePhysicalDevices(_instance, &device_count, physical_devices.data());

  _physical_device = physical_devices[0];
}

uint32_t VulkanTest::FindMemoryType(uint32_t memory_type_bits, VkMemoryPropertyFlags properties) {
  VkPhysicalDeviceMemoryProperties memory_properties;
  vkGetPhysicalDeviceMemoryProperties(_physical_device, &memory_properties);
  for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
    if ((memory_type_bits & (1 << i)) &&
        ((memory_properties.memoryTypes[i].propertyFlags & properties) == properties)) {
      return i;
    }
  }
  return VK_MAX_MEMORY_TYPES;
}

hipExternalSemaphoreHandleType VulkanTest::VulkanSemHandleTypeToHIPHandleType(
    VkSemaphoreType sem_type) {
  if (sem_type == VK_SEMAPHORE_TYPE_BINARY) {
    if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
      return hipExternalSemaphoreHandleTypeOpaqueWin32;
    } else if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
      return hipExternalSemaphoreHandleTypeOpaqueWin32Kmt;
    } else if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
      return hipExternalSemaphoreHandleTypeOpaqueFd;
    }
  } else if (sem_type == VK_SEMAPHORE_TYPE_TIMELINE) {
#if HT_AMD
    throw std::invalid_argument("Timeline semaphore unsupported on AMD");
#else
    if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
      return hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    } else if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
      return hipExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    } else if (_sem_handle_type & VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT) {
      return hipExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    }
#endif
  }

  throw std::invalid_argument("Invalid vulkan semaphore handle type");
}

hipExternalMemoryHandleType VulkanTest::VulkanMemHandleTypeToHIPHandleType() {
  if (_mem_handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT) {
    return hipExternalMemoryHandleTypeOpaqueWin32;
  } else if (_mem_handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT) {
    return hipExternalMemoryHandleTypeOpaqueWin32Kmt;
  } else if (_mem_handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT) {
    return hipExternalMemoryHandleTypeOpaqueFd;
  }

  throw std::invalid_argument("Invalid vulkan memory handle type");
}

#ifdef _WIN64
HANDLE
VulkanTest::GetSemaphoreHandle(VkSemaphore semaphore) {
  HANDLE handle = 0;

  VkSemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfoKHR = {};
  semaphoreGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR;
  semaphoreGetWin32HandleInfoKHR.pNext = NULL;
  semaphoreGetWin32HandleInfoKHR.semaphore = semaphore;
  semaphoreGetWin32HandleInfoKHR.handleType = _sem_handle_type;

  PFN_vkGetSemaphoreWin32HandleKHR fpGetSemaphoreWin32HandleKHR;
  fpGetSemaphoreWin32HandleKHR = (PFN_vkGetSemaphoreWin32HandleKHR)vkGetDeviceProcAddr(
      _device, "vkGetSemaphoreWin32HandleKHR");
  if (!fpGetSemaphoreWin32HandleKHR) {
    throw std::runtime_error("Failed to retrieve vkGetSemaphoreWin32HandleKHR");
  }
  if (fpGetSemaphoreWin32HandleKHR(_device, &semaphoreGetWin32HandleInfoKHR, &handle) !=
      VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve handle for buffer!");
  }

  return handle;
}
#else
int VulkanTest::GetSemaphoreHandle(VkSemaphore semaphore) {
  int fd;

  VkSemaphoreGetFdInfoKHR semaphoreGetFdInfoKHR = {};
  semaphoreGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
  semaphoreGetFdInfoKHR.pNext = NULL;
  semaphoreGetFdInfoKHR.semaphore = semaphore;
  semaphoreGetFdInfoKHR.handleType = _sem_handle_type;

  PFN_vkGetSemaphoreFdKHR fpGetSemaphoreFdKHR;
  fpGetSemaphoreFdKHR =
      (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(_device, "vkGetSemaphoreFdKHR");
  if (!fpGetSemaphoreFdKHR) {
    throw std::runtime_error("Failed to retrieve vkGetSemaphoreFdKHR");
  }
  if (fpGetSemaphoreFdKHR(_device, &semaphoreGetFdInfoKHR, &fd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve semaphore handle");
  }

  return fd;
}
#endif

#ifdef _WIN64
HANDLE
VulkanTest::GetMemoryHandle(VkDeviceMemory memory) {
  HANDLE handle = 0;

  VkMemoryGetWin32HandleInfoKHR vkMemoryGetWin32HandleInfoKHR = {};
  vkMemoryGetWin32HandleInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR;
  vkMemoryGetWin32HandleInfoKHR.memory = memory;
  vkMemoryGetWin32HandleInfoKHR.handleType = _mem_handle_type;

  PFN_vkGetMemoryWin32HandleKHR fpGetMemoryWin32HandleKHR =
      (PFN_vkGetMemoryWin32HandleKHR)vkGetDeviceProcAddr(_device, "vkGetMemoryWin32HandleKHR");

  if (!fpGetMemoryWin32HandleKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR");
  }
  if (fpGetMemoryWin32HandleKHR(_device, &vkMemoryGetWin32HandleInfoKHR, &handle) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve memory handle");
  }

  return handle;
}
#else
int VulkanTest::GetMemoryHandle(VkDeviceMemory memory) {
  int fd;

  VkMemoryGetFdInfoKHR memoryGetFdInfoKHR = {};
  memoryGetFdInfoKHR.sType = VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR;
  memoryGetFdInfoKHR.memory = memory;
  memoryGetFdInfoKHR.handleType = _mem_handle_type;

  PFN_vkGetMemoryFdKHR fpGetMemoryFdKHR =
      (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(_device, "vkGetMemoryFdKHR");
  if (!fpGetMemoryFdKHR) {
    throw std::runtime_error("Failed to retrieve vkGetMemoryFdKHR");
  }
  if (fpGetMemoryFdKHR(_device, &memoryGetFdInfoKHR, &fd) != VK_SUCCESS) {
    throw std::runtime_error("Failed to retrieve memory handle");
  }

  return fd;
}
#endif

VkExternalSemaphoreHandleTypeFlagBits VulkanTest::GetVkSemHandlePlatformType() const {
#ifdef _WIN64
  return IsWindows8OrGreater() ? VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
                               : VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

VkExternalMemoryHandleTypeFlagBits VulkanTest::GetVkMemHandlePlatformType() const {
#ifdef _WIN64
  return IsWindows8OrGreater() ? VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                               : VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_KMT_BIT;
#else
  return VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
#endif
}

void VulkanTest::CreateBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                              VkMemoryPropertyFlags properties, VkBuffer& buffer,
                              VkDeviceMemory& buffer_memory, bool external) {
  VkBufferCreateInfo buffer_create_info = {};
  buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
  buffer_create_info.size = size;
  buffer_create_info.usage = usage;
  buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

  VkExternalMemoryBufferCreateInfo external_memory_buffer_info = {};
  if (external) {
    external_memory_buffer_info.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO;
    external_memory_buffer_info.handleTypes = _mem_handle_type;
    buffer_create_info.pNext = &external_memory_buffer_info;
  }

  VK_CHECK_RESULT(vkCreateBuffer(_device, &buffer_create_info, nullptr, &buffer));

  VkMemoryRequirements memory_requirements;
  vkGetBufferMemoryRequirements(_device, buffer, &memory_requirements);

  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize = memory_requirements.size;
  alloc_info.memoryTypeIndex = FindMemoryType(memory_requirements.memoryTypeBits, properties);

  VkExportMemoryAllocateInfoKHR vulkan_export_memory_allocate_info = {};
#ifdef _WIN64
  WindowsSecurityAttributes win_security_attributes = {};
  VkExportMemoryWin32HandleInfoKHR vulkan_export_memory_win32_handle_info = {};
#endif

  if (external) {
    vulkan_export_memory_allocate_info.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    vulkan_export_memory_allocate_info.handleTypes = _mem_handle_type;

#ifdef _WIN64
    vulkan_export_memory_win32_handle_info.sType =
        VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
    vulkan_export_memory_win32_handle_info.pNext = NULL;
    vulkan_export_memory_win32_handle_info.pAttributes = &win_security_attributes;
    vulkan_export_memory_win32_handle_info.dwAccess =
        DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
    vulkan_export_memory_win32_handle_info.name = (LPCWSTR)NULL;

    vulkan_export_memory_allocate_info.pNext =
        _mem_handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR
        ? &vulkan_export_memory_win32_handle_info
        : NULL;
#endif

    alloc_info.pNext = &vulkan_export_memory_allocate_info;
  }

  VK_CHECK_RESULT(vkAllocateMemory(_device, &alloc_info, nullptr, &buffer_memory));
  VK_CHECK_RESULT(vkBindBufferMemory(_device, buffer, buffer_memory, 0));
}

void VulkanTest::CopyBuffer(VkBuffer src_buffer, VkBuffer dst_buffer, VkDeviceSize size) {
  VkCommandBufferAllocateInfo alloc_info = {};
  alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc_info.commandPool = _command_pool;
  alloc_info.commandBufferCount = 1;

  VkCommandBuffer command_buffer;
  vkAllocateCommandBuffers(_device, &alloc_info, &command_buffer);

  VkCommandBufferBeginInfo begin_info = {};
  begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

  vkBeginCommandBuffer(command_buffer, &begin_info);

  VkBufferCopy buffer_copy = {};
  buffer_copy.size = size;
  vkCmdCopyBuffer(command_buffer, src_buffer, dst_buffer, 1, &buffer_copy);

  vkEndCommandBuffer(command_buffer);

  VkSubmitInfo submit_info = {};
  submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit_info.commandBufferCount = 1;
  submit_info.pCommandBuffers = &command_buffer;

  vkQueueSubmit(_queue, 1, &submit_info, VK_NULL_HANDLE);
  vkQueueWaitIdle(_queue);
  vkFreeCommandBuffers(_device, _command_pool, 1, &command_buffer);
}

// ── Vulkan image helpers ──────────────────────────────────────────────────────────────────────

void VulkanTest::CreateImage(uint32_t width, uint32_t height, uint32_t num_levels,
                              VkFormat format, VkImage& image, VkDeviceMemory& image_memory,
                              VkDeviceSize& image_size, bool external) {
  VkImageCreateInfo img_info = {};
  img_info.sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
  img_info.imageType     = VK_IMAGE_TYPE_2D;
  img_info.format        = format;
  img_info.extent        = {width, height, 1};
  img_info.mipLevels     = num_levels;
  img_info.arrayLayers   = 1;
  img_info.samples       = VK_SAMPLE_COUNT_1_BIT;
  img_info.tiling        = VK_IMAGE_TILING_OPTIMAL;
  img_info.usage         = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                           VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_STORAGE_BIT;
  img_info.sharingMode   = VK_SHARING_MODE_EXCLUSIVE;
  img_info.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

  VkExternalMemoryImageCreateInfo ext_img_info = {};
  if (external) {
    ext_img_info.sType       = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMAGE_CREATE_INFO;
    ext_img_info.handleTypes = _mem_handle_type;
    img_info.pNext           = &ext_img_info;
  }

  VK_CHECK_RESULT(vkCreateImage(_device, &img_info, nullptr, &image));

  VkMemoryRequirements mem_req = {};
  vkGetImageMemoryRequirements(_device, image, &mem_req);
  image_size = mem_req.size;

  VkMemoryAllocateInfo alloc_info = {};
  alloc_info.sType           = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
  alloc_info.allocationSize  = mem_req.size;
  alloc_info.memoryTypeIndex = FindMemoryType(mem_req.memoryTypeBits,
                                              VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

  VkExportMemoryAllocateInfoKHR export_info = {};
#ifdef _WIN64
  WindowsSecurityAttributes win_sec_attrs;
  VkExportMemoryWin32HandleInfoKHR win32_info = {};
  win32_info.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
  win32_info.pAttributes = &win_sec_attrs;
  win32_info.dwAccess    = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
  win32_info.name        = nullptr;
#endif

  if (external) {
    export_info.sType       = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO_KHR;
    export_info.handleTypes = _mem_handle_type;
#ifdef _WIN64
    export_info.pNext = (_mem_handle_type & VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT_KHR)
                        ? &win32_info : nullptr;
#endif
    alloc_info.pNext = &export_info;
  }

  VK_CHECK_RESULT(vkAllocateMemory(_device, &alloc_info, nullptr, &image_memory));
  VK_CHECK_RESULT(vkBindImageMemory(_device, image, image_memory, 0));
}

VkImageView VulkanTest::CreateImageView(VkImage image, VkFormat format,
                                         uint32_t base_mip_level, uint32_t level_count) {
  VkImageViewCreateInfo view_info = {};
  view_info.sType                           = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
  view_info.image                           = image;
  view_info.viewType                        = VK_IMAGE_VIEW_TYPE_2D;
  view_info.format                          = format;
  view_info.components                      = {VK_COMPONENT_SWIZZLE_R, VK_COMPONENT_SWIZZLE_G,
                                               VK_COMPONENT_SWIZZLE_B, VK_COMPONENT_SWIZZLE_A};
  view_info.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  view_info.subresourceRange.baseMipLevel   = base_mip_level;
  view_info.subresourceRange.levelCount     = level_count;
  view_info.subresourceRange.baseArrayLayer = 0;
  view_info.subresourceRange.layerCount     = 1;

  VkImageView view = VK_NULL_HANDLE;
  VK_CHECK_RESULT(vkCreateImageView(_device, &view_info, nullptr, &view));
  return view;
}

static VkCommandBuffer BeginOneTimeCmd(VkDevice device, VkCommandPool pool) {
  VkCommandBufferAllocateInfo alloc = {};
  alloc.sType              = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
  alloc.commandPool        = pool;
  alloc.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
  alloc.commandBufferCount = 1;

  VkCommandBuffer cmd = VK_NULL_HANDLE;
  vkAllocateCommandBuffers(device, &alloc, &cmd);

  VkCommandBufferBeginInfo begin = {};
  begin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
  begin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
  vkBeginCommandBuffer(cmd, &begin);
  return cmd;
}

static void EndAndSubmitCmd(VkDevice device, VkQueue queue, VkCommandPool pool,
                             VkCommandBuffer cmd) {
  vkEndCommandBuffer(cmd);

  VkSubmitInfo submit = {};
  submit.sType              = VK_STRUCTURE_TYPE_SUBMIT_INFO;
  submit.commandBufferCount = 1;
  submit.pCommandBuffers    = &cmd;

  vkQueueSubmit(queue, 1, &submit, VK_NULL_HANDLE);
  vkQueueWaitIdle(queue);
  vkFreeCommandBuffers(device, pool, 1, &cmd);
}

static void RecordImageBarrier(VkCommandBuffer cmd, VkImage image, uint32_t mip_level,
                                VkImageLayout old_layout, VkImageLayout new_layout,
                                VkAccessFlags src_access, VkAccessFlags dst_access,
                                VkPipelineStageFlags src_stage, VkPipelineStageFlags dst_stage) {
  VkImageMemoryBarrier barrier = {};
  barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout                       = old_layout;
  barrier.newLayout                       = new_layout;
  barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.image                           = image;
  barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel   = mip_level;
  barrier.subresourceRange.levelCount     = 1;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount     = 1;
  barrier.srcAccessMask                   = src_access;
  barrier.dstAccessMask                   = dst_access;
  vkCmdPipelineBarrier(cmd, src_stage, dst_stage, 0, 0, nullptr, 0, nullptr, 1, &barrier);
}

void VulkanTest::TransitionImageLayout(VkImage image, uint32_t num_levels,
                                        VkImageLayout old_layout, VkImageLayout new_layout) {
  VkCommandBuffer cmd = BeginOneTimeCmd(_device, _command_pool);

  VkImageMemoryBarrier barrier = {};
  barrier.sType                           = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER;
  barrier.oldLayout                       = old_layout;
  barrier.newLayout                       = new_layout;
  barrier.srcQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.dstQueueFamilyIndex             = VK_QUEUE_FAMILY_IGNORED;
  barrier.image                           = image;
  barrier.subresourceRange.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  barrier.subresourceRange.baseMipLevel   = 0;
  barrier.subresourceRange.levelCount     = num_levels;
  barrier.subresourceRange.baseArrayLayer = 0;
  barrier.subresourceRange.layerCount     = 1;
  barrier.srcAccessMask                   = 0;
  barrier.dstAccessMask                   = VK_ACCESS_TRANSFER_WRITE_BIT;
  vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 0, nullptr, 0, nullptr, 1, &barrier);

  EndAndSubmitCmd(_device, _queue, _command_pool, cmd);
}

void VulkanTest::CopyBufferToImage(VkBuffer src_buf, VkImage dst_image, uint32_t mip_level,
                                    uint32_t mip_width, uint32_t mip_height) {
  VkCommandBuffer cmd = BeginOneTimeCmd(_device, _command_pool);

  // UNDEFINED → TRANSFER_DST_OPTIMAL: oldLayout is UNDEFINED because we always overwrite the
  // level's content; the driver is free to discard existing data, avoiding a redundant layout
  // establishment pass before the first write.
  RecordImageBarrier(cmd, dst_image, mip_level,
                      VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                      0, VK_ACCESS_TRANSFER_WRITE_BIT,
                      VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkBufferImageCopy region = {};
  region.bufferOffset                    = 0;
  region.bufferRowLength                 = 0;
  region.bufferImageHeight               = 0;
  region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel       = mip_level;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount     = 1;
  region.imageOffset                     = {0, 0, 0};
  region.imageExtent                     = {mip_width, mip_height, 1};
  vkCmdCopyBufferToImage(cmd, src_buf, dst_image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                          1, &region);

  // TRANSFER_DST_OPTIMAL → GENERAL
  RecordImageBarrier(cmd, dst_image, mip_level,
                      VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_MEMORY_READ_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

  EndAndSubmitCmd(_device, _queue, _command_pool, cmd);
}

void VulkanTest::CopyImageToBuffer(VkImage src_image, uint32_t mip_level, uint32_t mip_width,
                                    uint32_t mip_height, VkBuffer dst_buf) {
  VkCommandBuffer cmd = BeginOneTimeCmd(_device, _command_pool);

  // GENERAL → TRANSFER_SRC_OPTIMAL
  RecordImageBarrier(cmd, src_image, mip_level,
                      VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                      VK_ACCESS_MEMORY_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT,
                      VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT);

  VkBufferImageCopy region = {};
  region.bufferOffset                    = 0;
  region.bufferRowLength                 = 0;
  region.bufferImageHeight               = 0;
  region.imageSubresource.aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT;
  region.imageSubresource.mipLevel       = mip_level;
  region.imageSubresource.baseArrayLayer = 0;
  region.imageSubresource.layerCount     = 1;
  region.imageOffset                     = {0, 0, 0};
  region.imageExtent                     = {mip_width, mip_height, 1};
  vkCmdCopyImageToBuffer(cmd, src_image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, dst_buf,
                          1, &region);

  // TRANSFER_SRC_OPTIMAL → GENERAL
  RecordImageBarrier(cmd, src_image, mip_level,
                      VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL,
                      VK_ACCESS_TRANSFER_READ_BIT, VK_ACCESS_MEMORY_WRITE_BIT,
                      VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT);

  EndAndSubmitCmd(_device, _queue, _command_pool, cmd);
}

// Sometimes in CUDA the stream is not immediately ready after a semaphore has been signaled
void PollStream(hipStream_t stream, hipError_t expected, uint32_t num_iterations) {
  hipError_t query_result;
  for (uint32_t _ = 0; _ < num_iterations; ++_) {
    if ((query_result = hipStreamQuery(stream)) != expected) {
      std::this_thread::sleep_for(std::chrono::milliseconds{5});
    } else {
      break;
    }
  }
  REQUIRE(expected == query_result);
}

hipExternalSemaphore_t ImportBinarySemaphore(VulkanTest& vkt) {
  const auto semaphore = vkt.CreateExternalSemaphore(VK_SEMAPHORE_TYPE_BINARY);
  const auto sem_handle_desc = vkt.BuildSemaphoreDescriptor(semaphore, VK_SEMAPHORE_TYPE_BINARY);
  hipExternalSemaphore_t hip_ext_semaphore;
  HIP_CHECK(hipImportExternalSemaphore(&hip_ext_semaphore, &sem_handle_desc));

  return hip_ext_semaphore;
}
