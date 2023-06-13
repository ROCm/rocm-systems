/******************************************************************************
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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
*******************************************************************************/

#ifndef TEST_CTRL_TEST_KERNEL_H_
#define TEST_CTRL_TEST_KERNEL_H_

#include <string>
#include <cstdlib>
#include <cstdint>
#include <map>
#include <string>

// Class implements kernel test
class TestKernel {
 public:
  // Exported buffers IDs
  enum buf_id_t { KERNARG_EXP_ID, OUTPUT_EXP_ID, REFOUT_EXP_ID };
  // Memory descriptors IDs
  enum des_id_t { NULL_DES_ID, LOCAL_DES_ID, KERNARG_DES_ID, SYS_DES_ID, REFOUT_DES_ID };

  // Memory descriptors vector declaration
  struct mem_descr_t {
    des_id_t id;
    void* ptr;
    uint32_t size;
  };

  // Memory map declaration
  typedef std::map<uint32_t, mem_descr_t> mem_map_t;
  typedef mem_map_t::iterator mem_it_t;
  typedef mem_map_t::const_iterator mem_const_it_t;

  virtual ~TestKernel() {}

  // Initialize method
  virtual void Init() = 0;

  // Return kernel memory map
  mem_map_t& GetMemMap() { return mem_map_; }

  // Return NULL descriptor
  static mem_descr_t NullDescriptor() { return {NULL_DES_ID, NULL, 0}; }

  // Check if decripter is local
  bool IsLocal(const mem_descr_t& descr) const { return (descr.id == LOCAL_DES_ID); }

  // Methods to get the kernel attributes
  const mem_descr_t& GetKernargDescr() { return *test_map_[KERNARG_EXP_ID]; }
  const mem_descr_t& GetOutputDescr() { return *test_map_[OUTPUT_EXP_ID]; }
  void* GetKernargPtr() { return GetKernargDescr().ptr; }
  uint32_t GetKernargSize() { return GetKernargDescr().size; }
  void* GetOutputPtr() { return GetOutputDescr().ptr; }
  uint32_t GetOutputSize() { return GetOutputDescr().size; }
  bool IsOutputLocal() { return IsLocal(GetOutputDescr()); }
  virtual uint32_t GetGridSize() const = 0;

  // Return reference output
  void* GetRefOut() { return test_map_[REFOUT_EXP_ID]->ptr; }

  // Print output
  virtual void PrintOutput(const void* ptr) const = 0;

  // Return name
  virtual std::string Name() const = 0;

  // Return name
  virtual std::string SymbName() { return Name() + ".kd"; }

 protected:
  // Set buffer descriptor
  bool SetInDescr(uint32_t buf_id, des_id_t des_id, uint32_t size) {
    bool suc = SetMemDescr(buf_id, des_id, size);
    if (des_id == KERNARG_DES_ID) {
      test_map_[KERNARG_EXP_ID] = &mem_map_[buf_id];
    }
    return suc;
  }

  // Set results descriptor
  bool SetOutDescr(uint32_t buf_id, des_id_t des_id, uint32_t size) {
    bool suc = SetMemDescr(buf_id, des_id, size);
    test_map_[OUTPUT_EXP_ID] = &mem_map_[buf_id];
    return suc;
  }

  // Set host descriptor
  bool SetHostDescr(uint32_t buf_id, des_id_t des_id, uint32_t size) {
    bool suc = SetMemDescr(buf_id, des_id, size);
    if (suc) {
      mem_descr_t& descr = mem_map_[buf_id];
      descr.ptr = malloc(size);
      if (des_id == REFOUT_DES_ID) {
        test_map_[REFOUT_EXP_ID] = &descr;
      }
      if (descr.ptr == NULL) suc = false;
    }
    return suc;
  }

  // Get memory descriptor
  mem_descr_t GetDescr(uint32_t buf_id) const {
    mem_const_it_t it = mem_map_.find(buf_id);
    return (it != mem_map_.end()) ? it->second : NullDescriptor();
  }

 private:
  // Set memory descriptor
  bool SetMemDescr(uint32_t buf_id, des_id_t des_id, uint32_t size) {
    const mem_descr_t des = {des_id, NULL, size};
    auto ret = mem_map_.insert(mem_map_t::value_type(buf_id, des));
    return ret.second;
  }

  // Kernel memory map object
  mem_map_t mem_map_;
  // Test memory map object
  std::map<uint32_t, mem_descr_t*> test_map_;
};

#endif  // TEST_CTRL_TEST_KERNEL_H_
