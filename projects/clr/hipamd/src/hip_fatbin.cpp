/*
Copyright (c) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "hip/hip_runtime_api.h"
#include "hip_fatbin.hpp"
#include "hip_global.hpp"
#include <unordered_map>
#include "hip_code_object.hpp"
#include "hip_platform.hpp"
#include "comgrctx.hpp"
#include "amd_hsa_elf.hpp"
namespace hip {
namespace comgr_helper {

template <typename comgr_T> class ComgrUniqueHandle {
 public:
  ComgrUniqueHandle() = default;
  // constructor which takes ownership of a correctly initialzed handle
  ComgrUniqueHandle(comgr_T& handle) : comgr_obj_(handle) { handle = {0}; };

  template <typename T = comgr_T, std::enable_if_t<std::is_same_v<T, amd_comgr_data_set_t> ||
                                                       std::is_same_v<T, amd_comgr_action_info_t>,
                                                   bool> = true>
  [[nodiscard]] amd_comgr_status_t Create() {
    if constexpr (std::is_same_v<T, amd_comgr_data_set_t>) {
      return amd::Comgr::create_data_set(&comgr_obj_);
    } else if constexpr (std::is_same_v<T, amd_comgr_action_info_t>) {
      return amd::Comgr::create_action_info(&comgr_obj_);
    }

    // Unreachable code
    return AMD_COMGR_STATUS_SUCCESS;
  }

  template <typename T = comgr_T,
            std::enable_if_t<std::is_same_v<T, amd_comgr_data_t>, bool> = true>
  [[nodiscard]] amd_comgr_status_t Create(amd_comgr_data_kind_t kind) {
    return amd::Comgr::create_data(kind, &comgr_obj_);
  }

  ~ComgrUniqueHandle() {
    if (comgr_obj_.handle != 0) {
      if constexpr (std::is_same_v<comgr_T, amd_comgr_data_set_t>) {
        amd::Comgr::destroy_data_set(comgr_obj_);
      } else if constexpr (std::is_same_v<comgr_T, amd_comgr_action_info_t>) {
        amd::Comgr::destroy_action_info(comgr_obj_);
      } else if constexpr (std::is_same_v<comgr_T, amd_comgr_data_t>) {
        amd::Comgr::release_data(comgr_obj_);
      }
    }
  }

  // Delete all copy and move operators
  ComgrUniqueHandle(ComgrUniqueHandle&) = delete;
  ComgrUniqueHandle(ComgrUniqueHandle&&) = delete;
  ComgrUniqueHandle& operator=(ComgrUniqueHandle&) = delete;
  ComgrUniqueHandle& operator=(ComgrUniqueHandle&&) = delete;

  // Method to access data
  comgr_T get() const {
    assert(comgr_obj_.handle != 0);
    return comgr_obj_;
  }

 private:
  comgr_T comgr_obj_{0};
};


typedef ComgrUniqueHandle<amd_comgr_data_set_t> ComgrDataSetUniqueHandle;
typedef ComgrUniqueHandle<amd_comgr_action_info_t> ComgrActionInfoUniqueHandle;
typedef ComgrUniqueHandle<amd_comgr_data_t> ComgrDataUniqueHandle;

}  // namespace comgr_helper

FatBinaryInfo::FatBinaryInfo(const char* fname, const void* image)
    : foffset_(0), image_(image), image_mapped_(false), uri_(std::string()) {
  if (fname != nullptr) {
    fname_ = std::string(fname);
  } else {
    fname_ = std::string();
  }

  dev_programs_.resize(g_devices.size(), nullptr);
}

FatBinaryInfo::~FatBinaryInfo() {
  // Different devices in the same model have the same binary_image_
  std::set<const void*> toDelete;
  // Release per device fat bin info.
  for (int dev_id = 0; dev_id < dev_programs_.size(); dev_id++) {
    if (dev_programs_[dev_id] != nullptr) {
      auto& binaryInfo = dev_programs_[dev_id]->binary(*g_devices[dev_id]->devices()[0]);
      if (std::get<0>(binaryInfo) && std::get<1>(binaryInfo).second == 0 &&
          std::get<0>(binaryInfo) != image_) {
        toDelete.insert(std::get<0>(binaryInfo));
      }
      dev_programs_[dev_id]->release();
      dev_programs_[dev_id] = nullptr;
    }
  }
  for (auto itemData : toDelete) {
    LogPrintfInfo("~FatBinaryInfo(%p) will delete binary_image_ %p", this, itemData);
    delete[] reinterpret_cast<const char*>(itemData);
  }
  ReleaseImageAndFile();
}

void FatBinaryInfo::ReleaseImageAndFile() {
  // Release image_ and ufd_
  if (ufd_) {
    if (image_mapped_ && !amd::Os::MemoryUnmapFile(image_, ufd_->fsize_)) {
      guarantee(false, "Cannot unmap the file");
    }

    if (!PlatformState::instance().CloseUniqueFileHandle(ufd_)) {
      guarantee(false, "Cannot close file for fdesc: %d", ufd_->fdesc_);
    }

    ufd_ = nullptr;
    image_ = nullptr;
    uri_ = std::string();
    image_mapped_ = false;
  }
}

void ListAllDeviceWithNoCOFromBundle(
    const std::unordered_map<std::string, std::pair<size_t, size_t>>& unique_isa_names) {
  LogError("Missing CO for these ISAs - ");
  for (const auto& unique_isa : unique_isa_names) {
    if (unique_isa.second.first == 0) {
      LogPrintfError("     %s", unique_isa.first.c_str());
    }
  }
}

static std::string TargetGenericMap(const std::string& input) {
  const static std::unordered_map<std::string, std::string> target_map{
      // clang-format off
      {"amdgcn-amd-amdhsa--gfx900" , "amdgcn-amd-amdhsa--gfx9-generic"   },
      {"amdgcn-amd-amdhsa--gfx902" , "amdgcn-amd-amdhsa--gfx9-generic"   },
      {"amdgcn-amd-amdhsa--gfx904" , "amdgcn-amd-amdhsa--gfx9-generic"   },
      {"amdgcn-amd-amdhsa--gfx906" , "amdgcn-amd-amdhsa--gfx9-generic"   },
      {"amdgcn-amd-amdhsa--gfx909" , "amdgcn-amd-amdhsa--gfx9-generic"   },
      {"amdgcn-amd-amdhsa--gfx90c" , "amdgcn-amd-amdhsa--gfx9-generic"   },
      {"amdgcn-amd-amdhsa--gfx942" , "amdgcn-amd-amdhsa--gfx9-4-generic" },
      {"amdgcn-amd-amdhsa--gfx950" , "amdgcn-amd-amdhsa--gfx9-4-generic" },
      {"amdgcn-amd-amdhsa--gfx1010", "amdgcn-amd-amdhsa--gfx10-1-generic"},
      {"amdgcn-amd-amdhsa--gfx1011", "amdgcn-amd-amdhsa--gfx10-1-generic"},
      {"amdgcn-amd-amdhsa--gfx1012", "amdgcn-amd-amdhsa--gfx10-1-generic"},
      {"amdgcn-amd-amdhsa--gfx1013", "amdgcn-amd-amdhsa--gfx10-1-generic"},
      {"amdgcn-amd-amdhsa--gfx1030", "amdgcn-amd-amdhsa--gfx10-3-generic"},
      {"amdgcn-amd-amdhsa--gfx1031", "amdgcn-amd-amdhsa--gfx10-3-generic"},
      {"amdgcn-amd-amdhsa--gfx1032", "amdgcn-amd-amdhsa--gfx10-3-generic"},
      {"amdgcn-amd-amdhsa--gfx1033", "amdgcn-amd-amdhsa--gfx10-3-generic"},
      {"amdgcn-amd-amdhsa--gfx1034", "amdgcn-amd-amdhsa--gfx10-3-generic"},
      {"amdgcn-amd-amdhsa--gfx1035", "amdgcn-amd-amdhsa--gfx10-3-generic"},
      {"amdgcn-amd-amdhsa--gfx1036", "amdgcn-amd-amdhsa--gfx10-3-generic"},
      {"amdgcn-amd-amdhsa--gfx1100", "amdgcn-amd-amdhsa--gfx11-generic"  },
      {"amdgcn-amd-amdhsa--gfx1101", "amdgcn-amd-amdhsa--gfx11-generic"  },
      {"amdgcn-amd-amdhsa--gfx1102", "amdgcn-amd-amdhsa--gfx11-generic"  },
      {"amdgcn-amd-amdhsa--gfx1103", "amdgcn-amd-amdhsa--gfx11-generic"  },
      {"amdgcn-amd-amdhsa--gfx1150", "amdgcn-amd-amdhsa--gfx11-generic"  },
      {"amdgcn-amd-amdhsa--gfx1151", "amdgcn-amd-amdhsa--gfx11-generic"  },
      {"amdgcn-amd-amdhsa--gfx1152", "amdgcn-amd-amdhsa--gfx11-generic"  },
      {"amdgcn-amd-amdhsa--gfx1153", "amdgcn-amd-amdhsa--gfx11-generic"  },
      {"amdgcn-amd-amdhsa--gfx1200", "amdgcn-amd-amdhsa--gfx12-generic"  },
      {"amdgcn-amd-amdhsa--gfx1201", "amdgcn-amd-amdhsa--gfx12-generic"  },
      // clang-format on
  };
  if (auto i = target_map.find(input); i != target_map.end()) {
    return i->second;
  }
  return {};
}

// For sramecc and xnack
static std::string TargetFeatureCheck(const std::string& input, std::string feature) {
  if (input.find(feature) != std::string::npos) {
    auto feature_p = feature + "+";  // feature present eg: xnack+
    auto feature_m = feature + "-";  // feature absent eg: xnack-
    if (input.find(feature_p) != std::string::npos) {
      return feature_p;
    } else if (input.find(feature_m) != std::string::npos) {
      return feature_m;
    }
  }
  return "";
}

static std::string TargetToGeneric(std::string input) {
  auto sramecc = TargetFeatureCheck(input, "sramecc");
  auto xnack = TargetFeatureCheck(input, "xnack");

  // Remove all features
  size_t index = input.find_first_of(":");
  std::string name_without_feature = input.substr(0, index);

  // Look up generic name
  auto generic_name = TargetGenericMap(name_without_feature);
  if (generic_name.empty()) {
    return generic_name;  // No generic exists
  }

  // reappend feature
  if (!sramecc.empty()) {
    generic_name += ":";
    generic_name += sramecc;
  }
  if (!xnack.empty()) {
    generic_name += ":";
    generic_name += xnack;
  }
  return generic_name;
}

static bool IsCodeObjectUncompressed(const void* image) {
  return std::memcmp(image,
                     reinterpret_cast<const void*>(symbols::kOffloadBundleUncompressedMagicStr),
                     sizeof(symbols::kOffloadBundleUncompressedMagicStr) - 1) == 0;
}

static bool IsCodeObjectCompressed(const void* image) {
  return std::memcmp(image,
                     reinterpret_cast<const void*>(symbols::kOffloadBundleCompressedMagicStr),
                     sizeof(symbols::kOffloadBundleCompressedMagicStr) - 1) == 0;
}

static bool IsCodeObjectElf(const void* image) {
  const amd::Elf64_Ehdr* ehdr = reinterpret_cast<const amd::Elf64_Ehdr*>(image);
  return ehdr->e_machine == EM_AMDGPU && ehdr->e_ident[EI_OSABI] == ELFOSABI_AMDGPU_HSA;
}

static bool UncompressAndPopulateCodeObject(
    const void* image, const std::set<std::string>& unique_isa_names,
    std::map<std::string, std::pair<const void*, size_t>>& code_obj_map) {
  auto remove_file_extension = [](const std::string& input) -> std::string {
    size_t index = input.find_last_of(".");
    std::string ret = input.substr(0, index);
    return ret;
  };

  std::vector<std::string> bundle_ids_str;
  std::set<std::string> unique_ids;

  for (const auto& isa_name : unique_isa_names) {
    bundle_ids_str.push_back(std::string(symbols::kOffloadKindHipv4_) + isa_name);
  }

  std::vector<const char*> bundle_ids;
  bundle_ids.reserve(bundle_ids_str.size());
  for (auto& bundle_id_str : bundle_ids_str) {
    bundle_ids.push_back(bundle_id_str.c_str());
  }

  const auto obheader = reinterpret_cast<const symbols::ClangOffloadBundleCompressedHeader*>(image);
  const size_t size = obheader->totalSize;

  bool passed = false;
  do {
    comgr_helper::ComgrDataSetUniqueHandle bundled_co, unbundled_co;
    comgr_helper::ComgrDataUniqueHandle input_bundle;
    if (auto comgr_status = bundled_co.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Error in creating bundled_co");
      break;
    }

    if (auto comgr_status = unbundled_co.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Error in creating unbundled_co");
      break;
    }

    if (auto comgr_status = input_bundle.Create(AMD_COMGR_DATA_KIND_OBJ_BUNDLE);
        comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Error in creating input bundle");
      break;
    }

    if (auto comgr_status =
            amd::Comgr::set_data(input_bundle.get(), size, static_cast<const char*>(image));
        comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Error in setting image data to bundle");
      break;
    }

    if (auto comgr_status = amd::Comgr::set_data_name(input_bundle.get(), symbols::kHipFatBinName);
        comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Error in setting image data to bundle");
      break;
    }

    if (auto comgr_status = amd::Comgr::data_set_add(bundled_co.get(), input_bundle.get());
        comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Error in adding data set");
      break;
    }

    comgr_helper::ComgrActionInfoUniqueHandle unbundle_action;
    if (auto comgr_status = unbundle_action.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Error in creating unbundle action");
      break;
    }

    if (auto comgr_status = amd::Comgr::action_info_set_bundle_entry_ids(
            unbundle_action.get(), bundle_ids.data(), bundle_ids.size());
        comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Error in setting bundle entry ids");
      break;
    }

    if (auto comgr_status = amd::Comgr::do_action(AMD_COMGR_ACTION_UNBUNDLE, unbundle_action.get(),
                                                  bundled_co.get(), unbundled_co.get());
        comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Failed to unbundle code object");
      break;
    }

    size_t count = 0;
    if (auto comgr_status = amd::Comgr::action_data_count(unbundled_co.get(),
                                                          AMD_COMGR_DATA_KIND_EXECUTABLE, &count);
        comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogError("Failed to get data count of unbundled code object");
      break;
    }

    for (size_t i = 0; i < count; i++) {
      amd_comgr_data_t item;
      if (auto comgr_status = amd::Comgr::action_data_get_data(
              unbundled_co.get(), AMD_COMGR_DATA_KIND_EXECUTABLE, i, &item);
          comgr_status != AMD_COMGR_STATUS_SUCCESS) {
        LogError("Failed to get data unbundled code object");
        break;
      }

      size_t item_name_size = 0;
      if (auto comgr_status = amd::Comgr::get_data_name(item, &item_name_size, nullptr);
          comgr_status != AMD_COMGR_STATUS_SUCCESS) {
        LogError("Failed to get data size");
        break;
      }

      std::string item_bundle_id(item_name_size, 0);
      if (auto comgr_status =
              amd::Comgr::get_data_name(item, &item_name_size, item_bundle_id.data());
          comgr_status != AMD_COMGR_STATUS_SUCCESS) {
        LogError("Failed to get data");
        break;
      }

      size_t item_size = 0;
      if (auto comgr_status = amd::Comgr::get_data(item, &item_size, nullptr);
          comgr_status != AMD_COMGR_STATUS_SUCCESS) {
        LogError("Failed to get data size");
        break;
      }

      if (item_size > 0) {
        char* item_data = new char[item_size];
        if (auto comgr_status = amd::Comgr::get_data(item, &item_size, item_data);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to get data");
          break;
        }

        std::string bundle_entry = remove_file_extension(
            std::string(item_bundle_id.c_str() + sizeof(symbols::kOffloadHipV4FatBinName_) - 1));
        LogPrintfInfo("Inserting bundle entry of %s : size: %d, data: %p", bundle_entry.c_str(),
                      item_size, item_data);
        code_obj_map[bundle_entry] = std::make_pair(item_data, item_size);
      }
    }
    passed = true;
  } while (0);

  return passed;
}

static bool PopulateCodeObjectMap(
    const void* image, const std::set<std::string>& unique_isa_names,
    std::map<std::string, std::pair<const void*, size_t>>& code_obj_map) {
  bool passed = false;
  do {
    comgr_helper::ComgrDataUniqueHandle data_object;
    if (auto comgr_status = data_object.Create(AMD_COMGR_DATA_KIND_FATBIN);
        comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("Creating data object failed with status %d ", comgr_status);
      break;
    }

    // There is no way to find size of offload bundle, so we pass 4096 here.
    if (auto comgr_status =
            amd::Comgr::set_data(data_object.get(), 4096, reinterpret_cast<const char*>(image));
        comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("Setting data from file slice failed with status %d ", comgr_status);
      break;
    }

    // Create a query list using COMGR info for unique ISAs.
    std::vector<amd_comgr_code_object_info_t> query_list_array;
    query_list_array.reserve(unique_isa_names.size());
    for (const auto& isa_name : unique_isa_names) {
      auto& item = query_list_array.emplace_back();
      item.isa = isa_name.c_str();
      item.size = 0;
      item.offset = 0;
    }

    // Look up the code object info passing the query list.
    if (auto comgr_status = amd::Comgr::lookup_code_object(
            data_object.get(), query_list_array.data(), unique_isa_names.size());
        comgr_status != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("Setting data from file slice failed with status %d ", comgr_status);
      break;
    }

    for (const auto& item : query_list_array) {
      if (item.size > 0) {
        char* d = new char[item.size];
        std::memcpy(reinterpret_cast<void*>(d), reinterpret_cast<const char*>(image) + item.offset,
                    item.size);
        code_obj_map[item.isa] = std::make_pair(d, item.size);
      }
    }

    passed = true;
  } while (0);
  return passed;
}

hipError_t FatBinaryInfo::ExtractFatBinaryUsingCOMGR(const std::vector<hip::Device*>& devices) {
  if (fname_.empty() && image_ == nullptr) {
    LogError("Both Filename and image cannot be null");
    return hipErrorInvalidValue;
  }

  if (image_ != nullptr) {
    if (!amd::Os::FindFileNameFromAddress(image_, &fname_, &foffset_)) {
      fname_ = std::string("");
      foffset_ = 0;
    }
  } else {
    ufd_ = PlatformState::instance().GetUniqueFileHandle(fname_.c_str());
    if (ufd_ == nullptr) {
      return hipErrorFileNotFound;
    }

    // If the file name exists but the file size is 0, the something wrong with the file or its path
    if (ufd_->fsize_ == 0) {
      return hipErrorInvalidImage;
    }

    // If image_ is nullptr, then file path is passed via hipMod* APIs, so map the file.
    if (!amd::Os::MemoryMapFileDesc(ufd_->fdesc_, ufd_->fsize_, foffset_, &image_)) {
      LogError("Cannot map the file descriptor");
      PlatformState::instance().CloseUniqueFileHandle(ufd_);
      return hipErrorInvalidValue;
    }

    image_mapped_ = true;
  }
  guarantee(image_ != nullptr, "Image cannot be nullptr, file:%s did not map for some reason",
            fname_.c_str());

  bool is_compressed = IsCodeObjectCompressed(image_),
       is_uncompressed = IsCodeObjectUncompressed(image_);

  // It better be elf if its neither compressed nor uncompressed
  if (!is_compressed && !is_uncompressed) {
    if (IsCodeObjectElf(image_)) {
      // Load the binary directly
      auto elf_size = amd::Elf::getElfSize(image_);
      for (size_t i = 0; i < devices.size(); i++) {
        if (hipSuccess != AddDevProgram(devices[i], image_, elf_size, 0))
          return hipErrorInvalidImage;
      }
      return hipSuccess;  // We are done since it was already ELF
    } else {
      LogError("The code object has invalid header: compressed, uncompressed or elf");
      return hipErrorInvalidImage;
    }
  }

  // Create a list of all targets, which the current device can run
  // For example, gfx1030 can run gfx1030, gfx10-geneeric, amdgcnspirv
  std::set<std::string> unique_isa_names;
  const std::string spirv_isa_name{"spirv64-amd-amdhsa--amdgcnspirv"};
  unique_isa_names.insert(spirv_isa_name);  // Insert SPIRV ISA name
  for (auto device : devices) {
    std::string device_name = device->devices()[0]->isa().isaName();
    unique_isa_names.insert(device_name);
    auto generic_name = TargetToGeneric(device_name);
    LogPrintfInfo("Looking up generic name of : %s - %s", device_name.c_str(),
                  generic_name.c_str());
    if (!generic_name.empty()) {
      unique_isa_names.insert(generic_name);
    }
  }

  std::map<std::string, std::pair<const void*, size_t>> code_obj_map;  //!< code object map
  if (is_compressed) {
    if (!UncompressAndPopulateCodeObject(image_, unique_isa_names, code_obj_map)) {
      return hipErrorInvalidImage;
    }
  } else {  // uncompressed code object
    if (!PopulateCodeObjectMap(image_, unique_isa_names, code_obj_map)) {
      return hipErrorInvalidImage;
    }
  }

  hipError_t hip_status = hipErrorInvalidImage;
  do {
    bool spirv_isa_found = code_obj_map.find(spirv_isa_name) != code_obj_map.end();
    for (auto device : devices) {
      std::string device_name = device->devices()[0]->isa().isaName();
      auto generic_target_name = TargetToGeneric(device_name);   // Generic Code Object
      auto native_co = code_obj_map.find(device_name);           // Native Code Object
      auto generic_co = code_obj_map.find(generic_target_name);  // generic Code Object
      LogPrintfInfo("Device name: %s Generic name: %s", device_name.c_str(),
                    generic_target_name.c_str());

      // If the size is not 0, that means we found the native isa code object
      if (native_co != code_obj_map.end() && !HIP_FORCE_SPIRV_CODEOBJECT) {
        LogPrintfInfo("Using Native code object: %s", device->devices()[0]->isa().targetId());

        // We need to do this because there is existing mechanism which deletes code object in
        // destructor. Ideally next set of refactor should sort it.
        char* co = new char[native_co->second.second];
        std::memcpy(co, reinterpret_cast<const char*>(native_co->second.first),
                    native_co->second.second);
        hip_status = AddDevProgram(device, co, native_co->second.second, 0);
        if (hip_status != hipSuccess) {
          break;
        }
      } else if (generic_co != code_obj_map.end() && !HIP_FORCE_SPIRV_CODEOBJECT) {
        LogPrintfInfo("Using Generic code object: %s : %s", device->devices()[0]->isa().targetId(),
                      generic_target_name.c_str());
        char* co = new char[generic_co->second.second];
        std::memcpy(co, reinterpret_cast<const char*>(generic_co->second.first),
                    generic_co->second.second);
        hip_status = AddDevProgram(device, co, generic_co->second.second, 0);
        if (hip_status != hipSuccess) {
          break;
        }
      } else if (spirv_isa_found) {
        std::string target_id = device->devices()[0]->isa().targetId();
        std::string isa = "amdgcn-amd-amdhsa--" + target_id;
        LogPrintfInfo("Creating ISA for: %s from spirv", target_id.c_str());

        comgr_helper::ComgrDataSetUniqueHandle spirv_data_set;
        comgr_helper::ComgrDataSetUniqueHandle reloc_data;
        comgr_helper::ComgrDataUniqueHandle spirv_data;
        comgr_helper::ComgrActionInfoUniqueHandle reloc_action;

        if (auto comgr_status = spirv_data_set.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to create SPIRV Data set");
          break;
        }

        if (auto comgr_status = spirv_data.Create(AMD_COMGR_DATA_KIND_SPIRV);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to create SPIRV Data");
          break;
        }

        auto spirv_isa_handle = code_obj_map.find(spirv_isa_name);
        if (auto comgr_status =
                amd::Comgr::set_data(spirv_data.get(), spirv_isa_handle->second.second /* size */,
                                     reinterpret_cast<const char*>(spirv_isa_handle->second.first)
                                     /* buffer */);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to assign SPIRV data");
          break;
        }

        if (auto comgr_status = amd::Comgr::set_data_name(spirv_data.get(), "hip_code_object.spv");
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to set spirv data's name");
          break;
        }

        if (auto comgr_status = amd::Comgr::data_set_add(spirv_data_set.get(), spirv_data.get());
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to add spir data to data set");
          break;
        }

        if (auto comgr_status = reloc_action.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to create reloc action");
          break;
        }

        if (auto comgr_status =
                amd::Comgr::action_info_set_isa_name(reloc_action.get(), isa.c_str());
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to set reloc action's isa name");
          break;
        }

        if (auto comgr_status = reloc_data.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to create reloc data");
          break;
        }

        if (auto comgr_status =
                amd::Comgr::action_info_set_device_lib_linking(reloc_action.get(), true);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to set device lib linking for reloc action");
          break;
        }

        if (auto comgr_status =
                amd::Comgr::do_action(AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE,
                                      reloc_action.get(), spirv_data_set.get(), reloc_data.get());
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to compile spirv to reloc");
          break;
        }

        comgr_helper::ComgrActionInfoUniqueHandle exe_action;
        comgr_helper::ComgrDataSetUniqueHandle exe_output;
        if (auto comgr_status = exe_action.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to create exe action");
          break;
        }

        if (auto comgr_status = amd::Comgr::action_info_set_isa_name(exe_action.get(), isa.c_str());
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to set exe action isa name");
          break;
        }

        if (auto comgr_status = exe_output.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to create exe output");
          break;
        }

        if (auto comgr_status =
                amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                                      exe_action.get(), reloc_data.get(), exe_output.get());
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to do action: reloc to exe");
          break;
        }

        amd_comgr_data_t exe_data_handle;
        if (auto comgr_status = amd::Comgr::action_data_get_data(
                exe_output.get(), AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &exe_data_handle);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to get exe data");
          break;
        }

        // Move ownership of exe_data_handle to exe_data
        comgr_helper::ComgrDataUniqueHandle exe_data(exe_data_handle);
        size_t co_size = 0;
        if (auto comgr_status = amd::Comgr::get_data(exe_data.get(), &co_size, NULL);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to get exe size");
          break;
        }

        char* co = new char[co_size];
        if (auto comgr_status = amd::Comgr::get_data(exe_data.get(), &co_size, co);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to get exe data");
          break;
        }

        hip_status = AddDevProgram(device, co, co_size, 0);
        if (hip_status != hipSuccess) {
          break;
        }
      } else {
        // We found neither a compatible code object nor SPIRV
        LogPrintfError(
            "No compatible code objects found for: %s, value of HIP_FORCE_SPIRV_CODEOBJECT: %d",
            device->devices()[0]->isa().targetId(), HIP_FORCE_SPIRV_CODEOBJECT);
        break;
      }
    }
  } while (0);

  // release code objects
  for (const auto& co : code_obj_map) {
    delete[] reinterpret_cast<const char*>(co.second.first);
  }

  return hip_status;
}

hipError_t FatBinaryInfo::AddDevProgram(hip::Device* device, const void* binary_image,
                                        size_t binary_size, size_t binary_offset) {
  int devID = device->deviceId();
  amd::Context* ctx = device->asContext();
  amd::Program* program = new amd::Program(*ctx);
  dev_programs_[devID] = program;
  if (program == nullptr) {
    return hipErrorOutOfMemory;
  }
  if (CL_SUCCESS !=
      program->addDeviceProgram(*ctx->devices()[0], binary_image, binary_size, false, nullptr,
                                nullptr, (ufd_ != nullptr ? ufd_->fdesc_ : amd::Os::FDescInit()),
                                binary_offset, uri_)) {
    return hipErrorInvalidKernelFile;
  }
  return hipSuccess;
}

hipError_t FatBinaryInfo::BuildProgram(const int device_id) {
  // Check for Device Id bounds and empty program to return gracefully
  DeviceIdCheck(device_id);

  if (dev_programs_[device_id] == nullptr) {
    return hipErrorInvalidKernelFile;
  }

  // If Program was already built skip this step and return success
  if (dev_programs_[device_id]->IsProgramBuilt(*g_devices[device_id]->devices()[0]) == false) {
    if (CL_SUCCESS != dev_programs_[device_id]->build(g_devices[device_id]->devices(), nullptr,
                                                      nullptr, nullptr, kOptionChangeable,
                                                      kNewDevProg)) {
      return hipErrorNoBinaryForGpu;
    }
    if (!dev_programs_[device_id]->load()) {
      return hipErrorNoBinaryForGpu;
    }
  }
  return hipSuccess;
}
}  // namespace hip
