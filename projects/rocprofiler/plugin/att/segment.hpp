/* Copyright (c) 2023 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#pragma once
#include <string>
#include <vector>
#include <iostream>
#include <random>
#include <unordered_set>
#include <algorithm>

template<typename Type>
class ordered_vector : public std::vector<Type>
{
  using Super = std::vector<Type>;
public:
  void insert(const Type& elem)
  {
    size_t loc = lower_bound(elem.begin());
    if (this->size() && get(loc).begin() < elem.begin())
      loc ++;
    this->Super::insert(this->begin()+loc, elem);
  }
  bool remove(const Type& elem)
  {
    if (!this->size()) return false;
    size_t loc = lower_bound(elem.begin());
    if (get(loc) != elem) return false;

    this->Super::erase(this->begin()+loc);
    return true;
  }
  bool remove(uint64_t elem_begin)
  {
    if (!this->size()) return false;
    size_t loc = lower_bound(elem_begin);
    if (get(loc).begin() != elem_begin) return false;

    this->Super::erase(this->begin()+loc);
    return true;
  }
  size_t lower_bound(size_t addr) const
  {
    if (!this->size()) return 0;
    return binary_search(addr, 0, this->size()-1);
  }

  size_t binary_search(size_t addr, size_t s, size_t e) const
  {
    if (s >= e)
      return s;
    else if (s+1 == e)
      return (get(e).begin() <= addr) ? e : s;

    size_t mid = (s+e)/2;
    if (get(mid).begin() <= addr)
      return binary_search(addr, mid, e);
    else
      return binary_search(addr, s, mid);
  }
  const Type& get(size_t i) const { return this->operator[](i); }
};

struct address_range_t
{
  uint64_t vbegin;
  uint32_t size;
  uint32_t id;
  uint32_t offset;

  bool operator<(const address_range_t& other) const { return vbegin < other.vbegin; }
  bool inrange(uint64_t addr) const { return addr >= vbegin && addr < vbegin+size; };
  uint64_t begin() const { return vbegin; }
};


/**
 * @brief Finds a candidate codeobj for the given vaddr
*/
class CodeobjTableTranslator : protected ordered_vector<address_range_t>
{
  using Super = ordered_vector<address_range_t>;
public:
  CodeobjTableTranslator() { reset(); }

  const address_range_t& find_codeobj_in_range(uint64_t addr)
  {
    if (cached_segment < size() && get(cached_segment).inrange(addr))
      return get(cached_segment);

    size_t lb = lower_bound(addr);
    if (lb >= size() || !get(lb).inrange(addr))
      throw std::string("segment addr out of range");

    cached_segment = lb;
    return get(cached_segment);
  }

  uint64_t find_codeobj_addr_in_range(uint64_t addr) {
    return find_codeobj_in_range(addr).vbegin;
  }

  const address_range_t& get(size_t index) const { return data()[index]; }

  void insert(const address_range_t& elem) { this->Super::insert(elem); }
  void insert_list(std::vector<address_range_t> arange)
  {
    for (auto& elem : arange) push_back(elem);
    std::sort(
      this->begin(),
      this->end(),
      [](const address_range_t& a, const address_range_t& b) { return a < b; }
    );
  };

  void reset() { cached_segment = ~0; }
  void clear() { reset(); this->Super::clear(); }
  bool remove(uint64_t addr) { reset(); return this->Super::remove(addr); }

private:
  size_t cached_segment = ~0;
};
