/*
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2020, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>

#include "rocm_smi/rocm_smi_utils.h"
#include "rocm_smi/rocm_smi_io_link.h"

namespace amd {
namespace smi {

static const char *kKFDNodesPathRoot = "/sys/class/kfd/kfd/topology/nodes";
static const char *kKFDLinkPath[] = {"io_links", "p2p_links"};

// IO Link Property strings
static const char *kIOLinkPropTYPEStr =  "type";
// static const char *kIOLinkPropVERSION_MAJORStr = "version_major";
// static const char *kIOLinkPropVERSION_MINORStr = "version_minor";
static const char *kIOLinkPropNODE_FROMStr = "node_from";
static const char *kIOLinkPropNODE_TOStr = "node_to";
static const char *kIOLinkPropWEIGHTStr = "weight";
// static const char *kIOLinkPropMIN_LATENCYStr = "min_latency";
// static const char *kIOLinkPropMAX_LATENCYStr = "max_latency";
static const char *kIOLinkPropMIN_BANDWIDTHStr = "min_bandwidth";
static const char *kIOLinkPropMAX_BANDWIDTHStr = "max_bandwidth";
// static const char *kIOLinkPropRECOMMENDED_TRANSFER_SIZEStr =
// "recommended_transfer_size";
// static const char *kIOLinkPropFLAGSStr = "flags";

static bool is_number(const std::string &s) {
  return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

static std::string LinkPathRoot(uint32_t node_indx,
                                LINK_DIRECTORY_TYPE directory) {
  std::string link_path_root = kKFDNodesPathRoot;
  link_path_root += '/';
  link_path_root += std::to_string(node_indx);
  link_path_root += '/';
  if (directory < sizeof(kKFDLinkPath)/sizeof(kKFDLinkPath[0])) {
    link_path_root += kKFDLinkPath[directory];
  } else {
    link_path_root = "";
  }
  return link_path_root;
}

static std::string LinkPath(uint32_t node_indx, uint32_t link_indx,
                            LINK_DIRECTORY_TYPE directory) {
  std::string link_path = LinkPathRoot(node_indx, directory);
  link_path += '/';
  link_path += std::to_string(link_indx);
  return link_path;
}

static int OpenLinkProperties(uint32_t node_indx, uint32_t link_indx,
                              std::ifstream *fs,
                              LINK_DIRECTORY_TYPE directory) {
  int ret;
  std::string f_path;
  bool reg_file;

  assert(fs != nullptr);
  if (fs == nullptr) {
    return EINVAL;
  }

  f_path = LinkPath(node_indx, link_indx, directory);
  f_path += "/";
  f_path += "properties";

  ret = isRegularFile(f_path, &reg_file);

  if (ret != 0) {
    return ret;
  }
  if (!reg_file) {
    return ENOENT;
  }

  fs->open(f_path);

  if (!fs->is_open()) {
    return errno;
  }

  return 0;
}

static int ReadLinkProperties(uint32_t node_indx, uint32_t link_indx,
                              std::vector<std::string> *retVec,
                              LINK_DIRECTORY_TYPE directory) {
  std::string line;
  int ret;
  std::ifstream fs;

  assert(retVec != nullptr);
  if (retVec == nullptr) {
    return EINVAL;
  }

  ret = OpenLinkProperties(node_indx, link_indx, &fs, directory);

  if (ret) {
    return ret;
  }

  while (std::getline(fs, line)) {
    retVec->push_back(line);
  }

  if (retVec->empty()) {
    fs.close();
    return 0;
  }

  // Remove any *trailing* empty (whitespace) lines
  while (retVec->back().find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
    retVec->pop_back();
  }

  fs.close();
  return 0;
}

static int DiscoverLinks(std::map<std::pair<uint32_t, uint32_t>,
                         std::shared_ptr<IOLink>> *links,
                         LINK_DIRECTORY_TYPE directory) {
  assert(links != nullptr);
  if (links == nullptr) {
    return EINVAL;
  }
  assert(links->empty());

  links->clear();

  auto kfd_node_dir = opendir(kKFDNodesPathRoot);

  if (kfd_node_dir == nullptr) {
    std::string err_msg = "Failed to open KFD nodes directory ";
    err_msg += kKFDNodesPathRoot;
    err_msg += ".";
    perror(err_msg.c_str());
    return 1;
  }

  auto dentry_kfd = readdir(kfd_node_dir);
  while (dentry_kfd != nullptr) {
    if (dentry_kfd->d_name[0] == '.') {
      dentry_kfd = readdir(kfd_node_dir);
      continue;
    }

    if (!is_number(dentry_kfd->d_name)) {
      dentry_kfd = readdir(kfd_node_dir);
      continue;
    }

    uint32_t node_indx = static_cast<uint32_t>(std::stoi(dentry_kfd->d_name));
    std::shared_ptr<IOLink> link;
    uint32_t link_indx;
    std::string link_path_root = LinkPathRoot(node_indx, directory);

    auto io_link_dir = opendir(link_path_root.c_str());
    assert(io_link_dir != nullptr);

    auto dentry_io_link = readdir(io_link_dir);
    while (dentry_io_link != nullptr) {
      if (dentry_io_link->d_name[0] == '.') {
        dentry_io_link = readdir(io_link_dir);
        continue;
      }

      if (!is_number(dentry_io_link->d_name)) {
        dentry_io_link = readdir(io_link_dir);
        continue;
      }

      link_indx = static_cast<uint32_t>(std::stoi(dentry_io_link->d_name));
      link = std::make_shared<IOLink>(node_indx, link_indx,
                                                directory);

      link->Initialize();

      (*links)[std::make_pair(link->node_from(), link->node_to())] = link;

      dentry_io_link = readdir(io_link_dir);
    }

    if (closedir(io_link_dir)) {
      std::string err_msg = "Failed to close KFD nodes directory ";
      err_msg += kKFDNodesPathRoot;
      err_msg += ".";
      perror(err_msg.c_str());
      return 1;
    }

    dentry_kfd = readdir(kfd_node_dir);
  }

  if (closedir(kfd_node_dir)) {
    return 1;
  }
  return 0;
}

int DiscoverIOLinks(std::map<std::pair<uint32_t, uint32_t>,
                    std::shared_ptr<IOLink>> *links) {
  return DiscoverLinks(links, IO_LINK_DIRECTORY);
}

int DiscoverP2PLinks(std::map<std::pair<uint32_t, uint32_t>,
                    std::shared_ptr<IOLink>> *links) {
  return DiscoverLinks(links, P2P_LINK_DIRECTORY);
}

static int DiscoverLinksPerNode(uint32_t node_indx, std::map<uint32_t,
                                std::shared_ptr<IOLink>> *links,
                                LINK_DIRECTORY_TYPE directory) {
  assert(links != nullptr);
  if (links == nullptr) {
    return EINVAL;
  }
  assert(links->empty());

  links->clear();

  std::shared_ptr<IOLink> link;
  uint32_t link_indx;
  std::string link_path_root = LinkPathRoot(node_indx, directory);

  auto io_link_dir = opendir(link_path_root.c_str());
  assert(io_link_dir != nullptr);

  auto dentry = readdir(io_link_dir);
  while (dentry != nullptr) {
    if (dentry->d_name[0] == '.') {
      dentry = readdir(io_link_dir);
      continue;
    }

    if (!is_number(dentry->d_name)) {
      dentry = readdir(io_link_dir);
      continue;
    }

    link_indx = static_cast<uint32_t>(std::stoi(dentry->d_name));
    link = std::make_shared<IOLink>(node_indx, link_indx,
                                              directory);

    link->Initialize();

    (*links)[link->node_to()] = link;

    dentry = readdir(io_link_dir);
  }

  if (closedir(io_link_dir)) {
    return 1;
  }
  return 0;
}

int DiscoverIOLinksPerNode(uint32_t node_indx, std::map<uint32_t,
                           std::shared_ptr<IOLink>> *links) {
  return DiscoverLinksPerNode(node_indx, links, IO_LINK_DIRECTORY);
}

int DiscoverP2PLinksPerNode(uint32_t node_indx, std::map<uint32_t,
                            std::shared_ptr<IOLink>> *links) {
  return DiscoverLinksPerNode(node_indx, links, P2P_LINK_DIRECTORY);
}

IOLink::~IOLink() = default;

int IOLink::ReadProperties(void) {
  int ret;

  std::vector<std::string> propVec;

  assert(properties_.empty());
  if (!properties_.empty()) {
    return 0;
  }

  ret = ReadLinkProperties(node_indx_, link_indx_, &propVec,
                           link_dir_type_);

  if (ret) {
    return ret;
  }

  std::string key_str;
  uint64_t val_int;  // Assume all properties are unsigned integers for now
  std::istringstream fs;

  for (const auto & i : propVec) {
    fs.str(i);
    fs >> key_str;
    fs >> val_int;

    properties_[key_str] = val_int;

    fs.str("");
    fs.clear();
  }

  return 0;
}

int
IOLink::Initialize(void) {
  int ret = 0;
  ret = ReadProperties();
  if (ret) {return ret;}

  ret = get_property_value(kIOLinkPropTYPEStr,
                           reinterpret_cast<uint64_t *>(&type_));
  if (ret) {return ret;}

  ret = get_property_value(kIOLinkPropNODE_FROMStr,
                           reinterpret_cast<uint64_t *>(&node_from_));
  if (ret) {return ret;}

  ret = get_property_value(kIOLinkPropNODE_TOStr,
                           reinterpret_cast<uint64_t *>(&node_to_));
  if (ret) {return ret;}

  ret = get_property_value(kIOLinkPropWEIGHTStr, &weight_);
  if (ret) {return ret;}

  ret = get_property_value(kIOLinkPropMIN_BANDWIDTHStr, &min_bandwidth_);
  if (ret) {return ret;}

  ret = get_property_value(kIOLinkPropMAX_BANDWIDTHStr, &max_bandwidth_);

  return ret;
}

int
IOLink::get_property_value(std::string property, uint64_t *value) {
  assert(value != nullptr);
  if (value == nullptr) {
    return EINVAL;
  }
  if (properties_.find(property) == properties_.end()) {
    return EINVAL;
  }
  *value = properties_[property];
  return 0;
}

}  // namespace smi
}  // namespace amd
