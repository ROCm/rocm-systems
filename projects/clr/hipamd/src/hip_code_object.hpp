/*
Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc. All rights reserved.

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

#ifndef HIP_CODE_OBJECT_HPP
#define HIP_CODE_OBJECT_HPP

#include "hip_global.hpp"

#include <cstring>
#include <unordered_map>

#include "hip/hip_runtime.h"
#include "hip/hip_runtime_api.h"
#include "hip_internal.hpp"
#include "device/device.hpp"
#include "platform/program.hpp"

namespace hip {
//Forward Declaration for friend usage
class PlatformState;

//Code Object base class
class CodeObject {
 public:
  virtual ~CodeObject() {}

  // Functions to add_dev_prog and build
  static hipError_t add_program(int deviceId, hipModule_t hmod, const void* binary_ptr,
                                size_t binary_size);
  static hipError_t build_module(hipModule_t hmod, const std::vector<amd::Device*>& devices);

  // Given an file desc and file size, extracts to code object for corresponding devices,
  // return code_objs{binary_ptr, binary_size}, which could be used to determine foffset
  static hipError_t ExtractCodeObjectFromFile(amd::Os::FileDesc fdesc, size_t fsize,
                    const void ** image, const std::vector<std::string>& device_names,
                    std::vector<std::pair<const void*, size_t>>& code_objs, size_t foffset);

  // Given an ptr to memory, extracts to code object for corresponding devices,
  // returns code_objs{binary_ptr, binary_size} and uniform resource indicator
  static hipError_t ExtractCodeObjectFromMemory(const void* data,
                    const std::vector<std::string>& device_names,
                    std::vector<std::pair<const void*, size_t>>& code_objs,
                    std::string& uri);

  static uint64_t ElfSize(const void* emi);

  static bool IsClangOffloadMagicBundle(const void* data, bool& isCompressed);

  static uint32_t getGenericVersion(const void* image);

  static bool isGenericTarget(const void* image);

  static bool containGenericTarget(const void *data);

  // Return size of fat bin
  static size_t getFatbinSize(const void* data, const bool isCompressed = false);

  /**
     *  @brief Extract code object from fatbin using comgr unbundling action
     *
     *  @param[in]  data the bundle data(fatbin or loaded module data). It can be in uncompressed,
     *              compressed and even SPIR-V(to be supported later) mode.
     *  @param[in]  size the size of the bundle data
     *  @param[in]  agent_triple_target_ids isa names of concerned devices
     *  @param[out] code_objs the buffer address and size pairs of extracted code objects of
     *              concerned devices
     *  Returned error code
     *
     *  @return #hipSuccess, #hipErrorInvalidKernelFile, #hipErrorInvalidValue,
     *          #hipErrorNoBinaryForGpu
     *
     *  @see FatBinaryInfo::ExtractFatBinaryUsingCOMGR(const void* data,
     *                                             const std::vector<hip::Device*>& devices)
     */
  static hipError_t extractCodeObjectFromFatBinaryUsingComgr(
      const void* data, size_t size, const std::vector<std::string>& devices,
      std::vector<std::pair<const void*, size_t>>& code_objs);

  // Query the generic target of agent target.
  // Return true on successfull query, false on failure
   static bool QueryGenericTarget(std::string agentTarget, std::string& processor,
                char& sram_ecc, char& xnack);
 protected:
  //Given an ptr to image or file, extracts to code object
  //for corresponding devices
  static hipError_t extractCodeObjectFromFatBinary(const void*,
                    const std::vector<std::string>&,
                    std::vector<std::pair<const void*, size_t>>&);

  CodeObject() {}
private:
  friend const std::vector<hipModule_t>& modules();
};

//Dynamic Code Object
class DynCO : public CodeObject {
  // Guards Dynamic Code object
  amd::Monitor dclock_{true};

public:
  DynCO() : device_id_(ihipGetDevice()), fb_info_(nullptr), module_(nullptr) {}
  virtual ~DynCO();

  //LoadsCodeObject and its data
  hipError_t loadCodeObject(const char* fname, const void* image=nullptr);
  hipModule_t getModule() const { return module_; };

  //Gets GlobalVar/Functions from a dynamically loaded code object
  hipError_t getDynFunc(hipFunction_t* hfunc, std::string func_name);
  bool isValidDynFunc(const void* hfunc);
  hipError_t getDeviceVar(DeviceVar** dvar, std::string var_name);

  hipError_t getManagedVarPointer(std::string name, void** pointer, size_t* size_ptr) const {
    auto it = vars_.find(name);
    if (it != vars_.end() && it->second->getVarKind() == Var::DVK_Managed) {
      if (pointer != nullptr) {
        *pointer = it->second->getManagedVarPtr();
      }
      if (size_ptr != nullptr) {
        *size_ptr = it->second->getSize();
      }
    }
    return hipSuccess;
  }

private:
  int device_id_;
  FatBinaryInfo* fb_info_;
  hipModule_t module_;

  //Maps for vars/funcs, could be keyed in with std::string name
  std::unordered_map<std::string, Function*> functions_;
  std::unordered_map<std::string, Var*> vars_;

  //Populate Global Vars/Funcs from an code object(@ module_load)
  hipError_t populateDynGlobalFuncs();
  hipError_t populateDynGlobalVars();
  hipError_t initDynManagedVars(const std::string& managedVar);
};

//Static Code Object
class StatCO: public CodeObject {
  // Guards Static Code object
  amd::Monitor sclock_{true};
public:
  StatCO();
  virtual ~StatCO();

  //Add/Remove/Digest Fat Binaries passed to us from "__hipRegisterFatBinary"
  FatBinaryInfo** addFatBinary(const void* data, bool initialized, bool& success);
  hipError_t removeFatBinary(FatBinaryInfo** module);
  hipError_t digestFatBinary(const void* data, FatBinaryInfo*& programs);

  //Register vars/funcs given to use from __hipRegister[Var/Func/ManagedVar]
  hipError_t registerStatFunction(const void* hostFunction, Function* func);
  hipError_t registerStatGlobalVar(const void* hostVar, Var* var);
  hipError_t registerStatManagedVar(Var *var);

  //Retrive Vars/Funcs for a given hostSidePtr(const void*), unless stated otherwise.
  const char* getStatFuncName(const void* hostFunction);
  hipError_t getStatFunc(hipFunction_t* hfunc, const void* hostFunction, int deviceId);
  hipError_t getStatFuncAttr(hipFuncAttributes* func_attr, const void* hostFunction, int deviceId);
  hipError_t getStatGlobalVar(const void* hostVar, int deviceId, hipDeviceptr_t* dev_ptr,
                              size_t* size_ptr);

  //Managed variable is a defined symbol in code object
  //pointer to the alocated managed memory has to be copied to the address of symbol
  hipError_t initStatManagedVarDevicePtr(int deviceId);
private:
  friend class hip::PlatformState;
  //Populated during __hipRegisterFatBinary
  std::unordered_map<const void*, FatBinaryInfo*> modules_;
  //Populated during __hipRegisterFuncs
  std::unordered_map<const void*, Function*> functions_;
  //Populated during __hipRegisterVars
  std::unordered_map<const void*, Var*> vars_;
  //Populated during __hipRegisterManagedVar
  std::unordered_map<FatBinaryInfo**, std::vector<Var*> > managedVars_;
  //Reverse mapping of modules to speed up removal
  std::unordered_map<FatBinaryInfo**, const void*> module_to_hostModule_;
  std::unordered_map<FatBinaryInfo**, std::vector<const void*> > module_to_hostFunctions_;
  std::unordered_map<FatBinaryInfo**, std::vector<const void*> > module_to_hostVars_;
  std::unordered_map<int, bool> managedVarsDevicePtrInitalized_;
};

}; // namespace hip

#endif /* HIP_CODE_OBJECT_HPP */
