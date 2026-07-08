/*
 * Copyright © 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include "memory_allocation_registry.h"

namespace rocdxg {

void AllocationRegistry::Clear() {
  allocations_.clear();
}

void AllocationRegistry::Insert(const void *address,
                                const Allocation &allocation) {
  allocations_[address] = allocation;
}

bool AllocationRegistry::Erase(const void *address) {
  return allocations_.erase(address) != 0;
}

Allocation *AllocationRegistry::Find(const void *address) {
  auto it = allocations_.find(address);
  if (it == allocations_.end())
    return nullptr;
  return &it->second;
}

Allocation *AllocationRegistry::FindPreceding(const void *address) {
  auto it = allocations_.upper_bound(address);
  if (it == allocations_.begin())
    return nullptr;

  --it;
  return &it->second;
}

Allocation *AllocationRegistry::FindContaining(const void *address) {
  auto it = allocations_.upper_bound(address);
  if (it != allocations_.begin()) {
    --it;
    if (address >= it->first &&
        (address < reinterpret_cast<const uint8_t *>(it->first) +
                       it->second.size_requested))
      return &it->second;
  }

  return nullptr;
}

} // namespace rocdxg
