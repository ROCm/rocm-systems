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

#include "hip_fatbin.hpp"

#include <unordered_map>
#include "hip_code_object.hpp"
#include "hip_platform.hpp"
#include "comgrctx.hpp"

namespace hip {

namespace comgr_helper {

template <typename comgr_T> class ComgrUniqueHandle {
 public:
  ComgrUniqueHandle() = default;
  // constructor which takes ownership of a correctly initialzed handle
  ComgrUniqueHandle(comgr_T& handle) : comgr_obj_(handle) { handle = {0}; };

  template <typename T = comgr_T,
            std::enable_if_t<std::is_same_v<T, amd_comgr_data_set_t> ||
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

hipError_t FatBinaryInfo::ExtractFatBinaryUsingCOMGR(const std::vector<hip::Device*>& devices,
    bool &containGenericTarget) {
  amd_comgr_data_t data_object {0};
  amd_comgr_status_t comgr_status = AMD_COMGR_STATUS_SUCCESS;
  hipError_t hip_status = hipSuccess;

  // If image was passed as a pointer to our hipMod* api, we can try to extract the file name
  // if it was mapped by the app. Otherwise use the COMGR data API.
  if (fname_.size() == 0) {
    if (image_ == nullptr) {
      LogError("Both Filename and image cannot be null");
      return hipErrorInvalidValue;
    }

    if (!amd::Os::FindFileNameFromAddress(image_, &fname_, &foffset_)) {
      fname_ = std::string("");
      foffset_ = 0;
    }
  }

  // If file name & path are available (or it is passed to you), then get the file desc to use
  // COMGR file slice APIs.
  if (image_ == nullptr && fname_.size() > 0) {
    // Get File Handle & size of the file.
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

  // At this line, image should be a valid ptr.
  guarantee(image_ != nullptr, "Image cannot be nullptr, file:%s did not map for some reason",
            fname_.c_str());

  do {
    bool isCompressed = false;
    // If the image ptr is not clang offload bundle then just directly point the image.
    if (!CodeObject::IsClangOffloadMagicBundle(image_, isCompressed)) {
      for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
        uint64_t elf_size = CodeObject::ElfSize(image_);
        if (elf_size == 0) {
          hip_status = hipErrorInvalidImage;
          break;
        }
        hip_status = AddDevProgram(devices[dev_idx], image_, elf_size, 0);
        if (hip_status != hipSuccess) {
          break;
        }
      }
      break;
    }
    if (!isCompressed) {
       if (CodeObject::containGenericTarget(image_)) {
         LogInfo("offload bundle contains generic target code object");
         containGenericTarget = true;
       }
    }
    if (isCompressed || containGenericTarget) {
      size_t major = 0, minor = 0;
      amd::Comgr::get_version(&major, &minor);
      if ((major == 2 && minor >= 8) || major > 2) {
        hip_status = ExtractFatBinaryUsingCOMGR(image_, devices);
        break;
      } else if (isCompressed) {
        LogPrintfError("comgr %zu.%zu cannot support compressed mode which requires comgr 2.8+",
                       major, minor);
        hip_status = hipErrorNotSupported;
        break;
      }
    }
    // Create a data object, if it fails return error
    if ((comgr_status = amd::Comgr::create_data(AMD_COMGR_DATA_KIND_FATBIN, &data_object)) !=
        AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("Creating data object failed with status %d ", comgr_status);
      hip_status = hipErrorInvalidValue;
      break;
    }

    if (ufd_ != nullptr && amd::Os::isValidFileDesc(ufd_->fdesc_)) {
      LogPrintfError("Have valid file!%d", ufd_->fdesc_);
    }

    if (image_ != nullptr) {
      // Using the image ptr, map the data object.
      if ((comgr_status =
               amd::Comgr::set_data(data_object, 4096, reinterpret_cast<const char*>(image_))) !=
          AMD_COMGR_STATUS_SUCCESS) {
        LogPrintfError("Setting data from file slice failed with status %d ", comgr_status);
        hip_status = hipErrorInvalidValue;
        break;
      }
    } else {
      guarantee(false, "Cannot have both fname_ and image_ as nullptr");
    }

    // Find the unique number of ISAs needed for this COMGR query.
    std::unordered_map<std::string, std::pair<size_t, size_t>> unique_isa_names;
    for (auto device : devices) {
      std::string device_name = device->devices()[0]->isa().isaName();
      unique_isa_names.insert({device_name, std::make_pair<size_t, size_t>(0, 0)});
    }

    // there are two spirv targets, spirv64-amd-amdhsa--amdgcnspirv and
    // spirv64-amd-amdhsa-unknown-amdgcnspirv.
    // eventually we will remove spirv64-amd-amdhsa--amdgcnspirv
    const std::vector<std::string> spirv_isa_names = {"spirv64-amd-amdhsa--amdgcnspirv",
                                                      "spirv64-amd-amdhsa-unknown-amdgcnspirv"};
    for (const auto& spirv_isa_name : spirv_isa_names) {
      unique_isa_names.insert({spirv_isa_name, std::make_pair<size_t, size_t>(0, 0)});
    }

    // Create a query list using COMGR info for unique ISAs.
    std::vector<amd_comgr_code_object_info_t> query_list_array;
    query_list_array.reserve(unique_isa_names.size());
    for (const auto& isa_name : unique_isa_names) {
      auto& item = query_list_array.emplace_back();
      item.isa = isa_name.first.c_str();
      item.size = 0;
      item.offset = 0;
    }

    // Look up the code object info passing the query list.
    if ((comgr_status = amd::Comgr::lookup_code_object(data_object, query_list_array.data(),
                                                       unique_isa_names.size())) !=
        AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("Setting data from file slice failed with status %d ", comgr_status);
      hip_status = hipErrorInvalidValue;
      break;
    }

    for (const auto& item : query_list_array) {
      auto unique_it = unique_isa_names.find(item.isa);
      guarantee(unique_isa_names.cend() != unique_it, "Cannot find unique isa ");
      unique_it->second = std::pair<size_t, size_t>(static_cast<size_t>(item.size),
                                                    static_cast<size_t>(item.offset));
    }

    bool spirv_isa_found = false;
    decltype(unique_isa_names.begin()) spirv_isa_handle;
    for (const auto& spirv_isa_name : spirv_isa_names) {
      auto iter = unique_isa_names.find(spirv_isa_name);
      if (iter->second.first != 0) {
        spirv_isa_found = true;
        spirv_isa_handle = iter;
      }
    }
    bool get_spirv_data_res = false;
    std::once_flag get_spirv_data_flag;
    std::unordered_map<std::string, std::pair<char*, size_t>> compiled_co;  // code object cache
    comgr_helper::ComgrDataSetUniqueHandle spirv_data_set;
    comgr_helper::ComgrDataUniqueHandle spirv_data;

    auto get_spirv_data = [&]() {
      if (comgr_status = spirv_data_set.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
        LogError("Failed to create SPIRV Data set");
        return;
      }

      if (comgr_status = spirv_data.Create(AMD_COMGR_DATA_KIND_SPIRV);
          comgr_status != AMD_COMGR_STATUS_SUCCESS) {
        LogError("Failed to create SPIRV Data");
        return;
      }

      if (comgr_status =
              amd::Comgr::set_data(spirv_data.get(), spirv_isa_handle->second.first /* size */,
                                   reinterpret_cast<char*>(const_cast<void*>(image_)) +
                                       spirv_isa_handle->second.second /* buffer */);
          comgr_status != AMD_COMGR_STATUS_SUCCESS) {
        LogError("Failed to assign data in comgr");
        return;
      }

      if (comgr_status = amd::Comgr::set_data_name(spirv_data.get(), "hip_code_object.spv");
          comgr_status != AMD_COMGR_STATUS_SUCCESS) {
        LogError("Failed to set data name");
        return;
      }

      if (comgr_status = amd::Comgr::data_set_add(spirv_data_set.get(), spirv_data.get());
          comgr_status != AMD_COMGR_STATUS_SUCCESS) {
        LogError("Failed to add spir data");
        return;
      }

      get_spirv_data_res = true;
    };

    LogPrintfInfo("Searching for code objects, HIP_FORCE_SPIRV_CODEOBJECT: %d",
                  HIP_FORCE_SPIRV_CODEOBJECT);

    for (auto device : devices) {
      std::string device_name = device->devices()[0]->isa().isaName();
      auto dev_it = unique_isa_names.find(device_name);
      // If the size is not 0, that means we found the native isa code object
      if (dev_it->second.first != 0 && !HIP_FORCE_SPIRV_CODEOBJECT) {
        LogPrintfInfo("Using Native code object: %s", device->devices()[0]->isa().targetId());
        guarantee(unique_isa_names.cend() != dev_it,
                  "Cannot find the device name in the unique device name");
        hip_status = AddDevProgram(
            device, reinterpret_cast<address>(const_cast<void*>(image_)) + dev_it->second.second,
            dev_it->second.first, dev_it->second.second);
        if (hip_status != hipSuccess) {
          break;
        }
      } else if (spirv_isa_found) {
        std::call_once(get_spirv_data_flag, get_spirv_data);

        if(!get_spirv_data_res) {
          hip_status = hipErrorInvalidValue;
          break;
        }
        std::string target_id = device->devices()[0]->isa().targetId();
        if (auto code_iter = compiled_co.find(target_id); code_iter != compiled_co.end()) {
          // We have already compiled for it, lets reuse the code object
          char* co = new char[code_iter->second.second];
          std::memcpy(co, code_iter->second.first, code_iter->second.second);
          LogPrintfInfo("reusing code object for: %s", target_id.c_str());
          hip_status = AddDevProgram(device, co, code_iter->second.second, 0);
          if (hip_status != hipSuccess) {
            break;
          }
          continue;
        }

        LogPrintfInfo("Creating ISA for: %s from spirv", target_id.c_str());
        comgr_helper::ComgrActionInfoUniqueHandle reloc_action;
        std::string isa = "amdgcn-amd-amdhsa--" + target_id;
        if (comgr_status = reloc_action.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to create action");
          break;
        }

        if (comgr_status = amd::Comgr::action_info_set_isa_name(reloc_action.get(), isa.c_str());
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to set ISA name");
          break;
        }

        if (comgr_status = amd::Comgr::action_info_set_device_lib_linking(reloc_action.get(), true);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to set device lib linking");
          break;
        }

        if (comgr_status = amd::Comgr::action_info_set_option_list(
                reloc_action.get(), nullptr /* options list */, 0 /* options size */);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to set option list");
          break;
        }

        comgr_helper::ComgrDataSetUniqueHandle reloc_data;
        if (comgr_status = reloc_data.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to create reloc data set");
          break;
        }

        if (comgr_status =
                amd::Comgr::do_action(AMD_COMGR_ACTION_COMPILE_SPIRV_TO_RELOCATABLE,
                                      reloc_action.get(), spirv_data_set.get(), reloc_data.get());
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to compile to reloc");
          LogError("Failed to do action: codegen bc ot reloc");
          break;
        }

        comgr_helper::ComgrActionInfoUniqueHandle exe_action;
        comgr_helper::ComgrDataSetUniqueHandle exe_output;

        if (comgr_status = exe_action.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to create action");
          LogError("Failed to create exe action");
          break;
        }

        if (comgr_status = amd::Comgr::action_info_set_isa_name(exe_action.get(), isa.c_str());
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to set exe action isa name");
        }

        if (comgr_status = exe_output.Create(); comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to create exe output");
          break;
        }

        if (comgr_status =
                amd::Comgr::do_action(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                                      exe_action.get(), reloc_data.get(), exe_output.get());
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to do action: reloc to exe");
          break;
        }

        amd_comgr_data_t exe_data_handle;
        if (comgr_status = amd::Comgr::action_data_get_data(
                exe_output.get(), AMD_COMGR_DATA_KIND_EXECUTABLE, 0, &exe_data_handle);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to action get exe data");
          break;
        }
        // Move ownership of exe_data_handle to exe_data
        comgr_helper::ComgrDataUniqueHandle exe_data(exe_data_handle);

        size_t co_size;
        if (comgr_status = amd::Comgr::get_data(exe_data.get(), &co_size, NULL);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to get exe size");
          break;
        }

        char* co = new char[co_size];
        if (comgr_status = amd::Comgr::get_data(exe_data.get(), &co_size, co);
            comgr_status != AMD_COMGR_STATUS_SUCCESS) {
          LogError("Failed to get exe data");
          break;
        }

        auto elf_size = CodeObject::ElfSize(co);
        hip_status = AddDevProgram(device, co, elf_size, 0);
        if (hip_status != hipSuccess) {
          break;
        }
        // Save the compiled code object
        compiled_co[target_id] = std::make_pair(co, elf_size);
      } else {
        // We found neither a compatible code object nor SPIRV
        LogPrintfError(
            "No compatible code objects found for: %s, value of HIP_FORCE_SPIRV_CODEOBJECT: %d",
            device->devices()[0]->isa().targetId(), HIP_FORCE_SPIRV_CODEOBJECT);
        hip_status = hipErrorInvalidValue;
        break;
      }
    }
  } while (0);

  if (comgr_status != AMD_COMGR_STATUS_SUCCESS) {
    LogError("comgr API call failed");
    hip_status = hipErrorInvalidValue;
  }

  // Clean up file and memory resouces if hip_status failed for some reason.
  if (hip_status != hipSuccess && hip_status != hipErrorInvalidKernelFile) {
    ReleaseImageAndFile();
  }

  if (data_object.handle) {
    if ((comgr_status = amd::Comgr::release_data(data_object)) != AMD_COMGR_STATUS_SUCCESS) {
      LogPrintfError("Releasing COMGR data failed with status %d ", comgr_status);
      return hipErrorInvalidValue;
    }
  }

  return hip_status;
}

hipError_t FatBinaryInfo::ExtractFatBinary(const std::vector<hip::Device*>& devices) {
  amd::ScopedLock lock(FatBinaryLock());

  bool containGenericTarget = false;
  return ExtractFatBinaryUsingCOMGR(devices, containGenericTarget);
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
                                nullptr, (ufd_ != nullptr ? ufd_->fdesc_ : amd::Os::FDescInit()), binary_offset, uri_)) {
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
    if (CL_SUCCESS !=
        dev_programs_[device_id]->build(g_devices[device_id]->devices(), nullptr, nullptr, nullptr,
                                        kOptionChangeable, kNewDevProg)) {
      return hipErrorNoBinaryForGpu;
    }
    if (!dev_programs_[device_id]->load()) {
      return hipErrorNoBinaryForGpu;
    }
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t FatBinaryInfo::ExtractFatBinaryUsingCOMGR(const void* data,
                                                     const std::vector<hip::Device*>& devices) {
  hipError_t hip_status = hipSuccess;
  // At this line, image should be a valid ptr.
  guarantee(data != nullptr, "Image cannot be nullptr");

  do {
    std::vector<std::pair<const void*, size_t>> code_objs;
    // Copy device names
    std::vector<std::string> device_names;
    device_names.reserve(devices.size());
    for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
      device_names.push_back(devices[dev_idx]->devices()[0]->isa().isaName());
    }

    hip_status =
        CodeObject::extractCodeObjectFromFatBinaryUsingComgr(data, 0, device_names, code_objs);
    if (hip_status == hipErrorNoBinaryForGpu || hip_status == hipSuccess) {
      for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
        if (code_objs[dev_idx].first) {
          hip_status =
              AddDevProgram(devices[dev_idx], code_objs[dev_idx].first, code_objs[dev_idx].second, 0);
          if (hip_status != hipSuccess) {
            return hip_status;
          }
        } else {
          // This is the case of hipErrorNoBinaryForGpu which will finally fail app on device
          // without code object
          LogPrintfError("Cannot find CO in the bundle %s for ISA: %s", fname_.c_str(),
                         device_names[dev_idx].c_str());
        }
      }
    } else if (hip_status == hipErrorInvalidKernelFile) {
      hip_status = hipSuccess;
      // If the image ptr is not clang offload bundle then just directly point the image.
      for (size_t dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
        hip_status = AddDevProgram(devices[dev_idx], data, CodeObject::ElfSize(data), 0);
        if (hip_status != hipSuccess) {
          return hip_status;
        }
      }
    } else {
      LogPrintfError("CodeObject::extractCodeObjectFromFatBinaryUsingComgr failed with status %d\n",
                     hip_status);
    }
  } while (0);

  return hip_status;
}
}  // namespace hip
