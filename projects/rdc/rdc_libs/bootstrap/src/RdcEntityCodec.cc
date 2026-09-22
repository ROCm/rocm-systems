/*
Copyright (c) 2025 - present Advanced Micro Devices, Inc. All rights reserved.

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

#include <rdc/rdc.h>
#include <rdc_lib/RdcEntityCodec.h>

#include <algorithm>
#include <string>

rdc_entity_info_t rdc_get_info_from_entity_index(uint32_t entity_index) {
  rdc_entity_info_t info;
  info.device_type =
      (rdc_device_type_t)((entity_index >> RDC_ENTITY_TYPE_SHIFT) & RDC_ENTITY_TYPE_MASK);
  info.entity_role =
      (rdc_device_role_t)((entity_index >> RDC_ENTITY_ROLE_SHIFT) & RDC_ENTITY_ROLE_MASK);
  info.instance_index = (entity_index >> RDC_ENTITY_INSTANCE_SHIFT) & RDC_ENTITY_INSTANCE_MASK;
  info.device_index = (entity_index >> RDC_ENTITY_DEVICE_SHIFT) & RDC_ENTITY_DEVICE_MASK;
  return info;
}

uint32_t rdc_get_entity_index_from_info(rdc_entity_info_t info) {
  uint32_t entity_index = 0;
  entity_index |= ((info.device_type & RDC_ENTITY_TYPE_MASK) << RDC_ENTITY_TYPE_SHIFT);
  entity_index |= ((info.entity_role & RDC_ENTITY_ROLE_MASK) << RDC_ENTITY_ROLE_SHIFT);
  entity_index |= ((info.instance_index & RDC_ENTITY_INSTANCE_MASK) << RDC_ENTITY_INSTANCE_SHIFT);
  entity_index |= ((info.device_index & RDC_ENTITY_DEVICE_MASK) << RDC_ENTITY_DEVICE_SHIFT);
  return entity_index;
}

bool rdc_is_partition_string(const char* s) {
  if (!s || s[0] == '\0') {
    return false;
  }

  // Support both GPU ('g') and CPU ('c') partition strings
  if (s[0] != 'g' && s[0] != 'c') {
    return false;
  }

  std::string str(s);
  size_t dot_pos = str.find('.');
  if (dot_pos == std::string::npos) return false;

  if (dot_pos <= 1 || dot_pos >= str.size() - 1) return false;

  std::string socket_part = str.substr(1, dot_pos - 1);
  std::string partition_part = str.substr(dot_pos + 1);

  if (!std::all_of(socket_part.begin(), socket_part.end(), ::isdigit) ||
      !std::all_of(partition_part.begin(), partition_part.end(), ::isdigit))
    return false;

  int socket_index = std::stoi(socket_part);
  int partition_index = std::stoi(partition_part);

  if (socket_index < 0 || socket_index >= RDC_MAX_NUM_DEVICES) return false;
  if (partition_index < 0 || partition_index >= RDC_MAX_NUM_PARTITIONS) return false;

  return true;
}

bool rdc_parse_partition_string(const char* s, uint32_t* socket, uint32_t* partition) {
  if (!s) {
    return false;
  }

  if (!rdc_is_partition_string(s)) {
    return false;
  }

  std::string str(s);

  std::string rest = str.substr(1);
  size_t pos = rest.find('.');

  if (pos == std::string::npos) return false;

  std::string socket_str = rest.substr(0, pos);
  std::string partition_str = rest.substr(pos + 1);

  // Ensure both parts are a number
  if (!(!socket_str.empty() && std::all_of(socket_str.begin(), socket_str.end(), ::isdigit)) ||
      !(!partition_str.empty() &&
        std::all_of(partition_str.begin(), partition_str.end(), ::isdigit))) {
    return false;
  }

  *socket = std::stoi(socket_str);
  *partition = std::stoi(partition_str);
  return true;
}
