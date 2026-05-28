/*************************************************************************
 * Copyright (c) 2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <stdio.h>
#include <dirent.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <map>
#include <cassert>
#include <algorithm>
#include <iomanip>
#include <vector>
#include <limits>
#include <thread>
#include "alt_rsmi.h"
#include "debug.h"

static int ARSMI_readDeviceProperties(uint32_t node_id, std::map<std::string, uint64_t> &retVec);
static int ARSMI_readLinkProperties(uint32_t node_id, uint32_t target_id, std::map<std::string, uint64_t> &retVec);
static int read_node_properties(uint32_t node, std::string property_name, uint64_t *val,
                                std::map<std::string, uint64_t> &properties);
static int read_link_properties(uint32_t node, uint32_t target, std::string property_name, uint64_t *val,
                                std::map<std::string, uint64_t> &properties);
static int getNodeIndex(uint32_t node_id);
static int getGpuId(uint32_t node, uint64_t *gpu_id);
static int countIoLinks(uint32_t dev_id);

struct ARSMI_systemNode {
    uint32_t s_node_id = 0;
    uint64_t s_gpu_id = 0;
    uint64_t s_unique_id = 0;
    uint64_t s_location_id = 0;
    uint64_t s_bdf = 0;
    uint64_t s_domain = 0;
    uint8_t  s_bus = 0;
    uint8_t  s_device = 0;
    uint8_t  s_function = 0;
    uint8_t  s_partition_id = 0;
    std::string s_card;
};

static const char *kKFDNodesPathRoot = "/sys/class/kfd/kfd/topology/nodes";
static const uint32_t kAmdGpuId = 0x1002;

// Vector containing data about each node, ordered by bdf ID
static thread_local std::vector<ARSMI_systemNode> ARSMI_orderedNodes;

// 2-D matrix with link information between each pair of nodes.
static thread_local std::vector<std::vector<ARSMI_linkInfo>> ARSMI_orderedLinks;

// Number of devices recognized
static thread_local int ARSMI_num_devices=-1;


// Public API functions
int ARSMI_init(void)
{
    std::string err_msg;
    std::multimap<uint64_t, ARSMI_systemNode> ARSMI_allSystemNodes;

    if (ARSMI_num_devices > 0) {
        // has already been initialized
        return 0;
    }

    auto node_dir = opendir(kKFDNodesPathRoot);
    if (node_dir == nullptr) {
        WARN("Failed to open topo/nodes directory ");
        return 1;
    }
    auto dentry = readdir(node_dir);

    while (dentry != nullptr) {
        uint64_t gpu_id = 0, unique_id = 0, location_id = 0, domain = 0;
        uint64_t vendor_id = 0;
        if ((strcmp(dentry->d_name, ".") == 0) ||
            (strcmp(dentry->d_name, "..") == 0)) {
            dentry = readdir(node_dir);
            continue;
        }

        uint32_t node_id = std::stoi(dentry->d_name);

        std::map<std::string, uint64_t> properties;
        ARSMI_readDeviceProperties(node_id, properties);

        int ret_gpu_id = getGpuId(node_id, &gpu_id);

        int ret_unique_id = read_node_properties(node_id, "unique_id", &unique_id, properties);
        int ret_loc_id = read_node_properties(node_id, "location_id", &location_id, properties);
        int ret_domain = read_node_properties(node_id, "domain", &domain, properties);
        int ret_vendor = read_node_properties(node_id, "vendor_id", &vendor_id, properties);
        if (ret_gpu_id == 0 &&  !(ret_unique_id != 0 || ret_loc_id != 0 || ret_domain != 0 || ret_vendor != 0) &&
            (gpu_id != 0) && (vendor_id == kAmdGpuId)) {
            // Do not try to build a node if one of these fields
            // do not exist in KFD (0 as values okay)
            ARSMI_systemNode myNode;
            myNode.s_node_id = node_id;
            myNode.s_gpu_id = gpu_id;
            myNode.s_unique_id = unique_id;
            myNode.s_location_id = location_id;
            myNode.s_domain = domain & 0xFFFFFFFF;
            myNode.s_bdf = (myNode.s_domain << 32) | (myNode.s_location_id);
            myNode.s_location_id = myNode.s_bdf;
            myNode.s_bdf |= ((domain & 0xFFFFFFFF) << 32);
            myNode.s_location_id = myNode.s_bdf;
            myNode.s_domain = myNode.s_location_id >> 32;
            myNode.s_bus = ((myNode.s_location_id >> 8) & 0xFF);
            myNode.s_device = ((myNode.s_location_id >> 3) & 0x1F);
            myNode.s_function = myNode.s_location_id & 0x7;
            myNode.s_partition_id = ((myNode.s_location_id >> 28) & 0xF);

            ARSMI_allSystemNodes.emplace(unique_id, myNode);
        }

        dentry = readdir(node_dir);
    }

    ARSMI_num_devices = ARSMI_allSystemNodes.size();

    for (auto i : ARSMI_allSystemNodes) {
        std::ostringstream ss;
        ss << "[node_id = " << std::to_string(i.second.s_node_id)
           << "; gpu_id = " << std::to_string(i.second.s_gpu_id)
           << "; unique_id = " << std::to_string(i.second.s_unique_id)
           << "; location_id = " << std::to_string(i.second.s_location_id)
           << "; bdf = " << std::to_string(i.second.s_bdf)
           << "; domain = " << std::to_string(i.second.s_domain)
           << "; partition = " << std::to_string(i.second.s_partition_id)
           << "], ";
        std::string tempstr = ss.str();
        INFO(NCCL_INIT, "%s", tempstr.c_str());
    }

    if (closedir(node_dir)) {
        WARN("Failed to close topology/node root directory");
        return 1;
    }

    // Sort devices found. For this we need to group all devices
    // having the same unique_id, sort all devices with the same
    // unique_id by there bdf value. In addition, the groups
    // of devices with the same unique_id are sorted by
    // lowest bdf value among each others.
    std::vector<uint64_t> already_seen;
    std::vector<std::vector<ARSMI_systemNode>> sort_vecs;
    int elem=0;
    for (auto i : ARSMI_allSystemNodes) {
        auto device_uuid = i.second.s_unique_id;

        if ( std::find(already_seen.begin(), already_seen.end(), device_uuid) == already_seen.end()) {
            auto range = ARSMI_allSystemNodes.equal_range(device_uuid);

            sort_vecs.resize(sort_vecs.size()+1);
            for (auto j = range.first; j != range.second ; j++) {
                sort_vecs[elem].push_back(j->second);
            }
            already_seen.push_back(device_uuid);
            elem++;
        }
    }

    //Sort each subvector
    for (auto i = 0; i < sort_vecs.size(); i++) {
        std::sort(sort_vecs[i].begin(), sort_vecs[i].end(), []
                  (const ARSMI_systemNode &p1, const ARSMI_systemNode &p2) {
                      return p1.s_bdf < p2.s_bdf;
                  });
    }

    // Copy the first element for every uuid into the first_elem vector
    std::vector<uint64_t> first_elem;
    for (auto i=0; i < sort_vecs.size(); i++) {
        first_elem.push_back(sort_vecs[i][0].s_bdf);
    }
    std::sort (first_elem.begin(), first_elem.end(), []
               (const uint64_t &p1, const uint64_t &p2) {
                   return p1 < p2;
               });

    // Copy all elements of the sort_vecs subarrays into
    // ordered_nodes, with the sorted first_elem vector indicating
    // the order of each block.
    for (auto i=0; i < first_elem.size(); i++) {
        // Find the first_elem[i] in sort_vecs in
        for (auto j = 0; j < sort_vecs.size(); j++ ) {
            if (first_elem[i] == sort_vecs[j][0].s_bdf) {
                for (auto k=0; k<sort_vecs[j].size(); k++) {
                    ARSMI_orderedNodes.push_back(sort_vecs[j][k]);
                }
                break;
            }
        }
    }

    // Part 2: generate Link Matrix
    ARSMI_orderedLinks.resize(ARSMI_num_devices);
    for (int i=0; i<ARSMI_num_devices; i++) {
        ARSMI_orderedLinks[i].resize(ARSMI_num_devices);
        for (int j = 0; j < ARSMI_num_devices; j++) {
            ARSMI_orderedLinks[i][j].src_node = std::numeric_limits<unsigned>::max();
            ARSMI_orderedLinks[i][j].dst_node = std::numeric_limits<unsigned>::max();
        }
    }

    for (int src_idx = 0; src_idx < ARSMI_num_devices; src_idx++) {
        struct ARSMI_systemNode node = ARSMI_orderedNodes[src_idx];
        uint32_t src_id = node.s_node_id, nlinks = countIoLinks(src_id);
        for (int i = 0; i < nlinks; i++) {
            ARSMI_linkInfo info;
            std::map<std::string, uint64_t> properties;
            int ret = ARSMI_readLinkProperties(src_id, i, properties);
            if (ret != 0){
                continue;
            }

            uint64_t type;
            uint64_t weight;
            uint64_t min_bandwidth;
            uint64_t max_bandwidth;
            uint64_t dst_id;

            int ret_target = read_link_properties(src_id, i, "node_to", &dst_id, properties);
            if (ret_target != 0) {
                continue;
            }
            int dst_idx = getNodeIndex(dst_id);
            if (dst_idx == -1) {
                // Not all GPUs might be directly connected to all other GPUs.
                // Will set default values in the topo_get_link_info function.
                continue;
            }
            info.src_node = src_id;
            info.dst_node = dst_id;

            int ret_weight = read_link_properties(src_id, i, "weight", &weight, properties);
            if (ret_weight != 0) {
                WARN("Error reading link properties files");
                return 1;
            }
            info.weight = weight;

            int ret_type = read_link_properties(src_id, i, "type", &type, properties);
            if (ret_type != 0) {
                WARN("Error reading link properties files");
                return 1;
            }
            if (type == 11){
                info.type = ARSMI_IOLINK_TYPE_XGMI;
                info.hops = 1;
            }
            else if (type == 2) {
                info.type = ARSMI_IOLINK_TYPE_PCIEXPRESS;
                // hard coding for now to 2
                info.hops = 2;
            }
            else {
                info.type = ARSMI_IOLINK_TYPE_UNDEFINED;
                info.hops = 0;
            }

            int ret_min_bw = read_link_properties(src_id, i, "min_bandwidth", &min_bandwidth, properties);
            if (ret_min_bw != 0) {
                WARN("Error reading link properties files");
                return 1;
            }
            info.min_bandwidth = min_bandwidth;

            int ret_max_bw = read_link_properties(src_id, i, "max_bandwidth", &max_bandwidth, properties);
            if (ret_max_bw != 0) {
                return 1;
            }
            info.max_bandwidth = max_bandwidth;

            ARSMI_orderedLinks[src_idx][dst_idx] = info;
        }
    }

    return 0;
}


int ARSMI_get_num_devices (uint32_t *num_devices)
{
    int res = 0;

    if (ARSMI_num_devices < 0) {
        res = ARSMI_init();
    }

    *num_devices = ARSMI_num_devices;
    return res;
}

int ARSMI_dev_pci_id_get(uint32_t dv_ind, uint64_t *bdfid)
{
    if (bdfid == nullptr) {
        return EINVAL;
    }

    if (ARSMI_num_devices < 0) {
        int res = ARSMI_init();
        if (res != 0) {
            return res;
        }
    }

    *bdfid = ARSMI_orderedNodes[dv_ind].s_bdf;

    return 0;
}


int ARSMI_topo_get_link_info(uint32_t dv_ind_src, uint32_t dv_ind_dst,
                             ARSMI_linkInfo *info)
{
    if (info == nullptr) {
        return EINVAL;
    }

    if (ARSMI_num_devices < 0) {
        int res = ARSMI_init();
        if (res != 0) {
            return res;
        }
    }

    if (dv_ind_src < 0 || dv_ind_src > ARSMI_num_devices) {
        return EINVAL;
    }
    if (dv_ind_dst < 0 || dv_ind_dst > ARSMI_num_devices) {
        return EINVAL;
    }

    uint32_t src_id = ARSMI_orderedNodes[dv_ind_src].s_node_id;
    uint32_t dst_id = ARSMI_orderedNodes[dv_ind_dst].s_node_id;

    ARSMI_linkInfo tinfo = ARSMI_orderedLinks[dv_ind_src][dv_ind_dst];
    if (tinfo.src_node != src_id || tinfo.dst_node != dst_id) {
        // Setting  default values.
        tinfo.hops = 2;
        tinfo.type = ARSMI_IOLINK_TYPE_PCIEXPRESS;
        tinfo.weight = 40;
        tinfo.min_bandwidth = 0;
        tinfo.max_bandwidth = 0;
    }
    *info = tinfo;

    return 0;
}

// Internal functions
static int getNodeIndex(uint32_t node_id)
{
    int res = -1;
    assert (ARSMI_num_devices > 0);

    for (int i = 0; i < ARSMI_num_devices; i++) {
        if (ARSMI_orderedNodes[i].s_node_id == node_id) {
            res = i;
            break;
        }
    }

    return res;
}

static std::string DevicePath(uint32_t dev_id)
{
    std::string node_path = kKFDNodesPathRoot;
    node_path += '/';
    node_path += std::to_string(dev_id);

    return node_path;
}

static int isRegularFile(std::string fname, bool *is_reg)
{
    struct stat file_stat;
    int ret;

    ret = stat(fname.c_str(), &file_stat);
    if (ret) {
        return errno;
    }

    if (is_reg != nullptr) {
        *is_reg = S_ISREG(file_stat.st_mode);
    }

    return 0;
}

static bool isNumber(const std::string &s)
{
    return !s.empty() && std::all_of(s.begin(), s.end(), ::isdigit);
}

static int openNodeFile(uint32_t dev_id, std::string node_file,
                        std::ifstream *fs)
{
    std::string line;
    std::string f_path;
    bool reg_file;

    assert(fs != nullptr);

    f_path = DevicePath(dev_id);
    f_path += "/";
    f_path += node_file;

    int ret = isRegularFile(f_path, &reg_file);
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

static int countIoLinks(uint32_t dev_id)
{
    std::string f_path;
    int file_count = 0;

    f_path = DevicePath(dev_id);
    f_path += "/io_links/";

    auto node_dir = opendir(f_path.c_str());
    auto dentry = readdir(node_dir);
    while (dentry != NULL) {
        if (dentry->d_type == DT_DIR && strcmp(dentry->d_name, ".") != 0
            && strcmp(dentry->d_name, "..") != 0) {
            file_count++;
        }
        dentry = readdir(node_dir);
    }
    closedir(node_dir);
    return file_count;
}

static int openLinkFile(uint32_t dev_id, uint32_t target_id,
                        std::string node_file, std::ifstream *fs)
{
    std::string line;
    std::string f_path;
    bool reg_file;

    assert(fs != nullptr);

    f_path = DevicePath(dev_id);
    f_path += "/io_links/";
    f_path +=std::to_string(target_id);
    f_path += "/";
    f_path += node_file;

    int ret = isRegularFile(f_path, &reg_file);
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


static int readGpuId(uint32_t node_id, uint64_t *gpu_id)
{
    std::string line;
    std::ifstream fs;

    assert(gpu_id != nullptr);
    int ret = openNodeFile(node_id, "gpu_id", &fs);
    if (ret) {
        fs.close();
        return ret;
    }

    std::stringstream ss;
    ss << fs.rdbuf();
    fs.close();

    std::string gpu_id_str = ss.str();
    gpu_id_str.erase(std::remove(gpu_id_str.begin(), gpu_id_str.end(), '\n'),
                     gpu_id_str.end());
    if (!isNumber(gpu_id_str)) {
        return ENXIO;
    }

    *gpu_id = static_cast<uint64_t>(std::stoi(gpu_id_str));
    return 0;
}

static bool isNodeSupported(uint32_t node_indx)
{
    std::ifstream fs;
    bool ret = true;

    int err = openNodeFile(node_indx, "properties", &fs);
    if (err == ENOENT) {
        return false;
    }
    if (fs.peek() == std::ifstream::traits_type::eof()) {
        ret = false;
    }

    fs.close();
    return ret;
}

static int getPropertyValue(std::string property, uint64_t *value, std::map<std::string, uint64_t> &properties)
{
    if (value == nullptr) {
        return EINVAL;
    }
    if (properties.empty()) {
        return EINVAL;
    }

    if (properties.find(property) == properties.end()) {
        return EINVAL;
    }

    *value = properties[property];
    return 0;
}

static bool fileExists(char const *filename)
{
    struct stat buf;
    return (stat(filename, &buf) == 0);
}

static int ARSMI_readDeviceProperties(uint32_t node_id, std::map<std::string, uint64_t> &properties)
{
    std::string line;
    std::ifstream fs;
    std::vector<std::string> tVec;

    int ret = openNodeFile(node_id, "properties", &fs);
    if (ret) {
        return ret;
    }

    while (std::getline(fs, line)) {
        tVec.push_back(line);
    }

    if (tVec.empty()) {
        fs.close();
        return ENOENT;
    }

    // Remove any *trailing* empty (whitespace) lines
    while (tVec.back().find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
        tVec.pop_back();
    }

    fs.close();

    std::string key_str;
    std::string val_str;
    uint64_t val_int;  // Assume all properties are unsigned integers for now
    std::istringstream fs2;

    for (const auto & i : tVec) {
        fs2.str(i);
        fs2 >> key_str;
        fs2 >> val_str;

        val_int = std::stoull(val_str);
        properties[key_str] = val_int;

        fs2.str("");
        fs2.clear();
    }

    return 0;
}

static int ARSMI_readLinkProperties(uint32_t node_id, uint32_t target_node_id,
                                    std::map<std::string, uint64_t> &properties)
{
    std::string line;
    std::ifstream fs;
    std::vector<std::string> tVec;

    int ret = openLinkFile(node_id, target_node_id, "properties", &fs);
    if (ret) {
        return ret;
    }

    while (std::getline(fs, line)) {
        tVec.push_back(line);
    }

    if (tVec.empty()) {
        fs.close();
        return ENOENT;
    }

    // Remove any *trailing* empty (whitespace) lines
    while (tVec.back().find_first_not_of(" \t\n\v\f\r") == std::string::npos) {
        tVec.pop_back();
    }

    fs.close();

    std::string key_str;
    std::string val_str;
    uint64_t val_int;  // Assume all properties are unsigned integers for now
    std::istringstream fs2;

    for (const auto & i : tVec) {
        fs2.str(i);
        fs2 >> key_str;
        fs2 >> val_str;

        val_int = std::stoull(val_str);
        properties[key_str] = val_int;

        fs2.str("");
        fs2.clear();
    }

    return 0;
}

// /sys/class/kfd/kfd/topology/nodes/*/properties
static int read_node_properties(uint32_t node, std::string property_name,
                                uint64_t *val, std::map<std::string, uint64_t> &properties)
{
    int retVal = EINVAL;

    if (property_name.empty() || val == nullptr) {
        WARN("Could not read node # %u property %s", node, property_name.c_str());
        return retVal;
    }

    if (isNodeSupported(node)) {
        retVal = getPropertyValue(property_name, val, properties);
    } else {
        retVal = 1;
        WARN("Could not read node # %u",node);
    }

    return retVal;
}

// /sys/class/kfd/kfd/topology/nodes/*/io_links/*/properties
static int read_link_properties(uint32_t node, uint32_t target, std::string property_name,
                                uint64_t *val, std::map<std::string, uint64_t> &properties)
{
    int retVal = EINVAL;

    if (property_name.empty() || val == nullptr) {
        WARN("Could not read node # %u", node);
        return retVal;
    }

    if (isNodeSupported(node)) {
        retVal = getPropertyValue(property_name, val, properties);
    } else {
        retVal = 1;
        WARN("Could not read node # %u", node);
    }

    return retVal;
}

// /sys/class/kfd/kfd/topology/nodes/*/gpu_id
int getGpuId(uint32_t node, uint64_t *gpu_id)
{
    int retVal = EINVAL;

    if (gpu_id == nullptr) {
        WARN("Could not determine GPU id of node # %u", node);
        return retVal;
    }

    if (isNodeSupported(node)) {
        retVal = readGpuId(node, gpu_id);
    } else {
        retVal = 1;
        WARN("Could not read node # %u", node);
    }

  return retVal;
}
