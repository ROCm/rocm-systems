/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#include "rocurilocator.hpp"
#include <sstream>

namespace amd::roc {
std::vector<UriLocator::UriRange>& UriLocator::table() {
  static auto* t = new std::vector<UriRange>();
  return *t;
}

std::mutex& UriLocator::tableMutex() {
  static auto* m = new std::mutex();
  return *m;
}

void UriLocator::recordCodeObjects(hsa_executable_t exec) {
  hsa_ven_amd_loader_1_03_pfn_t fn_table;
  if (Hsa::system_get_major_extension_table(HSA_EXTENSION_AMD_LOADER, 1, sizeof(fn_table),
                                            &fn_table) != HSA_STATUS_SUCCESS ||
      fn_table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects == nullptr) {
    return;
  }

  struct CbArgs {
    hsa_ven_amd_loader_1_03_pfn_t* fn;
    std::vector<UriRange> ranges;
  } args{&fn_table, {}};

  fn_table.hsa_ven_amd_loader_executable_iterate_loaded_code_objects(
      exec,
      [](hsa_executable_t, hsa_loaded_code_object_t lcobj, void* data) -> hsa_status_t {
        auto* a = static_cast<CbArgs*>(data);
        auto get = a->fn->hsa_ven_amd_loader_loaded_code_object_get_info;
        if (get == nullptr) return HSA_STATUS_ERROR;
        uint64_t base = 0, size = 0;
        int64_t delta = 0;
        uint32_t uriLen = 0;
        if (get(lcobj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_BASE, &base) !=
                HSA_STATUS_SUCCESS ||
            get(lcobj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_SIZE, &size) !=
                HSA_STATUS_SUCCESS ||
            get(lcobj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_LOAD_DELTA, &delta) !=
                HSA_STATUS_SUCCESS ||
            get(lcobj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI_LENGTH, &uriLen) !=
                HSA_STATUS_SUCCESS) {
          return HSA_STATUS_SUCCESS;
        }
        std::string uri(uriLen, '\0');
        if (get(lcobj, HSA_VEN_AMD_LOADER_LOADED_CODE_OBJECT_INFO_URI, uri.data()) !=
            HSA_STATUS_SUCCESS) {
          return HSA_STATUS_SUCCESS;
        }
        a->ranges.push_back(UriRange{base, base + size - 1, delta, std::move(uri)});
        return HSA_STATUS_SUCCESS;
      },
      &args);

  if (!args.ranges.empty()) {
    std::lock_guard<std::mutex> lock(tableMutex());
    auto& tab = table();
    tab.insert(tab.end(), std::make_move_iterator(args.ranges.begin()),
               std::make_move_iterator(args.ranges.end()));
  }
}

// Encoding of uniform-resource-identifier(URI) is detailed in
// https://llvm.org/docs/AMDGPUUsage.html#loaded-code-object-path-uniform-resource-identifier-uri
std::pair<uint64_t, uint64_t> UriLocator::decodeUriAndGetFd(UriInfo& uri,
                                                            amd::Os::FileDesc* uri_fd) {
  std::ostringstream ss;
  char cur;
  uint64_t offset = 0, size = 0;
  if (uri.uriPath.size() == 0) return {0, 0};
  auto pos = uri.uriPath.find("//");
  if (pos == std::string::npos) {
    uri.uriPath = "";
    return {0, 0};
  }
  auto rspos = uri.uriPath.find('#');
  if (rspos != std::string::npos) {
    // parse range specifier
    std::string offprefix = "offset=", sizeprefix = "size=";
    auto sbeg = uri.uriPath.find('&', rspos);
    auto offbeg = rspos + offprefix.size() + 1;
    std::string offstr = uri.uriPath.substr(offbeg, sbeg - offbeg);
    auto sizebeg = sbeg + sizeprefix.size() + 1;
    std::string sizestr = uri.uriPath.substr(sizebeg, uri.uriPath.size() - sizebeg);
    offset = std::stoull(offstr, nullptr, 0);
    size = std::stoull(sizestr, nullptr, 0);
    rspos -= 1;
  } else {
    rspos = uri.uriPath.size() - 1;
  }
  if (uri.uriPath.substr(0, pos) == "file:") {
    pos += 2;
    // decode filepath
    for (auto i = pos; i <= rspos;) {
      cur = uri.uriPath[i];
      if (isalnum(cur) || cur == '/' || cur == '-' || cur == '_' || cur == '.' || cur == '~') {
        ss << cur;
        i++;
      } else {
        // characters prefix with '%' char
        char tbits = uri.uriPath[i + 1], lbits = uri.uriPath[i + 2];
        uint8_t t = (tbits < 58) ? (tbits - 48) : ((tbits - 65) + 10);
        uint8_t l = (lbits < 58) ? (lbits - 48) : ((lbits - 65) + 10);
        ss << (char)(((0b00000000 | t) << 4) | l);
        i += 3;
      }
    }
    uri.uriPath = ss.str();
    size_t fd_size;
    (void)amd::Os::GetFileHandle(uri.uriPath.c_str(), uri_fd, &fd_size);
    // As per URI locator syntax, range_specifier is optional
    // if range_specifier is absent return total size of the file
    // and set offset to begin at 0.
    if (size == 0) size = fd_size;
  }
  return {offset, size};
}

UriLocator::UriInfo UriLocator::lookUpUri(uint64_t device_pc) {
  std::lock_guard<std::mutex> lock(tableMutex());
  // Reverse order so that if an address range was reused, the most recent load wins.
  auto& tab = table();
  for (auto it = tab.rbegin(); it != tab.rend(); ++it)
    if (it->startAddr_ <= device_pc && device_pc <= it->endAddr_)
      return UriInfo{it->Uri_.c_str(), it->elfDelta_};

  return UriInfo{"", 0};
}
}  // namespace amd::roc
#endif
#endif
