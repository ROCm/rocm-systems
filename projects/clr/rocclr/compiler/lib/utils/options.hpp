/* Copyright (c) 2009 - 2021 Advanced Micro Devices, Inc.

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

#ifndef _UTILS_OPTIONS_HPP_
#define _UTILS_OPTIONS_HPP_

#include <string>
#include <vector>
#include <cstdio>
#include "top.hpp"
#include "library.hpp"
#include <cassert>
#include <sstream>
#ifdef __linux__
#include <unistd.h>
#endif
#ifdef _WIN32
#include <process.h>
#include <cstdio>
#endif

namespace amd {

class Device;

namespace option {

// Option Type : Info[0:5]
enum OptionType { OT_BOOL, OT_INT32, OT_UINT32, OT_CSTRING, OT_UCHAR, OT_MASK = 0x3f };

/*
   Option's Attributes : Info[6:31]

   OxA_<sth> : Only one in each group is required.
   OA_<sth>  : Each one is independent, all OA_<sth> can be OR'ed together.
*/

// Option Value Attributes
enum OptionValue {
  OVA_OPTIONAL = 0x00,    // value is optional
  OVA_REQUIRED = 0x40,    // value is required to appear;
  OVA_DISALLOWED = 0x80,  // value may not be specified.
  OVA_MASK = 0xC0
};

// Option Form Attributes
enum OptionForm {
  OFA_NORMAL = 0x000,    // normal form, no prefix
  OFA_PREFIX_F = 0x100,  // -f<flag>, machine-independent
  OFA_PREFIX_M = 0x200,  // -m<flag>, machine-dependent
  OFA_PREFIX_W = 0x300,  // -W<flag>, warning (TODO)
  OFA_MASK = 0x300
};

// Option Group, at least one must be used.
enum OptionGroup { OA_RUNTIME = 0x400, OA_CLC = 0x800, OA_LINK_EXE = 0x1000, OA_LINK_LIB = 0x2000 };

// Option Value Separator, at least one must be used.
enum OptionValueSeparator {
  OA_SEPARATOR_NONE = 0x4000,
  OA_SEPARATOR_EQUAL = 0x8000,
  OA_SEPARATOR_SPACE = 0x10000
};

// Option visibility : at least one must be used.
enum OptionVisibility {
  OVIS_PUBLIC = 0x00000,    // ducumented
  OVIS_SUPPORT = 0x20000,   // supported, but undocumented
  OVIS_INTERNAL = 0x40000,  // internal-only, not available in product release
  OVIS_MASK = 0x60000
};

// Option other attributes : optional
enum OptionMisc {
  OA_MISC_ALIAS = 0x80000  // An alias option is one that refers to another option or options
                           // and its meaning is hard-coded in setAliasOptionVariable().
};

typedef bool OT_BOOL_t;
typedef int32_t OT_INT32_t;
typedef uint32_t OT_UINT32_t;
typedef const char* OT_CSTRING_t;
typedef unsigned char OT_UCHAR_t;

// This must be a POD
struct OptionDescriptor {
  const char* NameShort;      // short option starts with -
  const char* NameLong;       // long option starts with --
  uint32_t Info;              // defined by above enums
  uint32_t OptionOffset;      // member offset to OptionVariable
  int64_t DefaultVal;         // default value for option of non-string type
  int64_t ValMin;             // (ValMin, ValMax) is the option's Value range
  int64_t ValMax;             //     for option of non-string type
  const char* DefaultString;  // default value for string.
  const char* Description;    // Short Description for this option. -h will show this.
};

#define OPTION_sname(o) ((o)->NameShort)
#define OPTION_lname(o) ((o)->NameLong)
#define OPTION_type(o) ((o)->Info & OT_MASK)
#define OPTION_value(o) ((o)->Info & OVA_MASK)
#define OPTION_form(o) ((o)->Info & OFA_MASK)
#define OPTION_vis(o) ((o)->Info & OVIS_MASK)
#define OPTION_info(o) ((o)->Info)
#define OPTION_offset(o) ((o)->OptionOffset)
#define OPTION_default(o) ((o)->DefaultVal)
#define OPTION_min(o) ((o)->ValMin)
#define OPTION_max(o) ((o)->ValMax)
#define OPTION_defaultstr(o) ((o)->DefaultString)
#define OPTION_desc(o) ((o)->Description)

// OptionVariables must be a POD
struct OptionVariables {
#define OPTION(type, a, sn, ln, var, ideft, imin, imax, sdeft, desc) type##_t var;
#define NOPTION(type, a, sn, ln, var, ideft, imin, imax, sdeft, desc)
#define FLAG(type, a, sn, var, deft, desc) type##_t var;
#include "OPTIONS.def"
#undef OPTION
#undef NOPTION
#undef FLAG
};

#define OPTION_valueSeparator(c) (((c) == ' ') || ((c) == '='))
#define OFFSET(var) ((uint32_t)&(((OptionVariable*)0)->var))


enum OptionIdentifier {
#define OPTION(type, a, sn, ln, var, ideft, imin, imax, sdeft, desc) OID_##var,
#define NOPTION(type, a, sn, ln, var, ideft, imin, imax, sdeft, desc) OID_##var,
#define FLAG(type, a, sn, var, deft, desc) OID_##var,
#include "OPTIONS.def"
  OID_LAST
#undef OPTION
#undef NOPTION
#undef FLAG
};

// Only option that is a RUNTIME option and is not an alias option has an entry in OptionVariables
#define OPTIONHasOVariable(od) ((OPTION_info(od) & (OA_RUNTIME | OA_MISC_ALIAS)) == OA_RUNTIME)

// DumpFlags defines the values for oVariables->DumpFlags
enum DumpFlags {
  DUMP_NONE = 0x00000000,          // default
  DUMP_CL = 0x00000001,            // CL source
  DUMP_I = 0x00000002,             // pre-processed CL source
  DUMP_S = 0x00000004,             // x86 assembly text
  DUMP_O = 0x00000008,             // x86 object / GPU code object
  DUMP_DLL = 0x00000010,           // x86 DLL (.so)
  DUMP_IL = 0x00000020,            // per-kernel IL (GPU)
  DUMP_ISA = 0x00000040,           // per-kernel assembly text (GPU)
  DUMP_BIF = 0x00000080,           // binary for all kernels (GPU)
  DUMP_BC_ORIGINAL = 0x00000100,   // input bitcode (generated by clc)
  DUMP_BC_LINKED = 0x00000200,     // bitcode after linking and before optimization
  DUMP_BC_OPTIMIZED = 0x00000400,  // bitcode after optimization (llvm opt's output)
  DUMP_CGIL = 0x00000800,          // output il generated by Codegen (llvm llc's output)
  DUMP_DEBUGIL = 0x00001000,       // debug il (from MDParser, dwarf refers to)
  DUMP_SPIRV = 0x00002000,         // SPIR-V binary
  DUMP_BC_SPIRV = 0x00004000,      // bitcode translated from SPIR-V

  // For a binary is encrypted, can only dump the following
  DUMP_ENCRYPT = (DUMP_DLL | DUMP_BIF),

  // For a release product, can only dump the following
  DUMP_PRODUCT_FLAGS = (DUMP_CL | DUMP_I | DUMP_S | DUMP_O | DUMP_DLL |
                        // handling DUMP_BC_ORIGINAL would require a fair amount of work,
                        // since the output of comgr is already optimized
                        DUMP_BC_OPTIMIZED | DUMP_BC_LINKED | DUMP_IL | DUMP_CGIL | DUMP_DEBUGIL |
                        DUMP_ISA | DUMP_BIF),

  DUMP_ALL = 0x00007FFF  // Everything
};

enum OptLevelFlags : unsigned char {
  OPT_O0 = '0',  // No optimization setting.
  OPT_O1 = '1',
  OPT_O2 = '2',
  OPT_O3 = '3',
  OPT_O4 = '4',
  OPT_O5 = '5',
  OPT_OG = 'g',   // g ASCII
  OPT_OS = 's',   // s ASCII
  OPT_Error = 0,  // Invalid optimization set
  /** Canary Value that guards against enum changes
   * @warning This value cannot be changed without updating the appropriate
   * tests and should NEVER be decreased.
   */
  optLast = 117
};

class Options {
 public:
  std::string origOptionStr;

  OptionVariables* oVariables;                // pointer to a struct of all option variables
  std::string clcOptions;                     // options passed into EDG frontend (clc)
  std::vector<std::string> clangOptions;      // options passed into Clang frontend
  std::string llvmOptions;                    // options passed into backend (llvm)
  std::vector<std::string> finalizerOptions;  // options passed into finalizer

  // Given as build option
  int WorkGroupSize[3];  // -1: use default
  int NumAvailGPRs;
  unsigned kernelArgAlign;

  Options();
  ~Options();

  bool isDumpFlagSet(DumpFlags f) { return ((oVariables->DumpFlags) & f) && dumpEncrypt(f); }

  // Definition of each entry in flags
  // Note: this is 1-bit entry now, it may need to be expanded to
  //       more than 1 bit later.
  enum { FLAG_UNSEEN = 0, FLAG_SEEN = 1 };

  void setFlag(int option_ndx, uint32_t v) {
    int r = option_ndx / 32;
    int c = option_ndx % 32;
    uint32_t* p = &flags[r];
    uint32_t b = (1 << c);
    v = (v << c);
    *p = ((*p) & (~b)) | v;
  }

  uint32_t getFlag(int option_ndx) const {
    int r = option_ndx / 32;
    int c = option_ndx % 32;
    const uint32_t* p = &flags[r];
    return 1 & ((*p) >> c);
  }

  bool isOptionSeen(int option_ndx) const { return (getFlag(option_ndx) == FLAG_SEEN); }

  int getLLVMArgc() const { return llvmargc; }
  char** getLLVMArgv() { return llvmargv; }
  void setLLVMArgs(int argc, char** argv) {
    llvmargc = argc;
    llvmargv = argv;
  }

  void recordMemoryHandle(char* handle) { MemoryHandles.push_back(handle); }

  // Do post-parse processing After parsing all options,
  void postParseInit();

  void setBuildNo(unsigned int bnum) { buildNo = bnum; }
  unsigned int getBuildNo() const { return buildNo; }
  void setCurrKernelName(const char* name) { currKernelName = name; }
  const char* getCurrKernelName() const { return currKernelName; }
  std::string getDumpFileName(const std::string& ext);

  void setPerBuildInfo(const char* val, int encrypt, bool device);
  bool isCStrOptionsEqual(const char* cs1, const char* cs2) const;


  bool useDefaultWGS() const { return UseDefaultWGS; }
  void setDefaultWGS(bool V) { UseDefaultWGS = V; }

  std::string& optionsLog() { return OptionsLog; }

  // Returns whether this set of options equals to another set of options
  bool equals(const Options& other, bool ignoreClcOptions = false) const;

  // Set the option variables same as defined in "other"
  bool setOptionVariablesAs(const Options& other);

  std::string getFinalizerOptions() { return getStringFromStringVec(finalizerOptions); }

 private:
  std::string fullPath, baseName;
  long basename_max;
  std::string OptionsLog;

  // One bit for each flag.
  uint32_t flags[(OID_LAST + 31) / 32];

  int llvmargc;
  char** llvmargv;

  // buildNo is a unique number for each device build, save it here so that
  // dumping can use it in its file names.
  //
  // Note that buildNo, dumpFileRoot, and encryptCode are valid only during the lifetime
  // of a device build.
  unsigned int buildNo;
  std::string dumpFileRoot;
  const char* currKernelName;
  int encryptCode;

  std::vector<char*> MemoryHandles;

  bool UseDefaultWGS;

  bool dumpEncrypt(DumpFlags f) const {
    return ((encryptCode == 0) ||  // return true if not encrypted
            (f & DUMP_ENCRYPT));
  }

  std::string getStringFromStringVec(std::vector<std::string>& stringVec);
  void setDumpFileName(const char* val);

 public:
  LibrarySelector libraryType_;
  std::string sourceFileName_;
};

OptionDescriptor* getOptDescTable();
bool init();
bool teardown();
bool parseAllOptions(std::string& options, Options& Opts, bool linkOptsOnly, bool isLC);
inline bool parseLinkOptions(std::string& options, Options& Opts, bool isLC) {
  return parseAllOptions(options, Opts, true /*linkOptsOnly*/, isLC);
}


}  // namespace option

}  // namespace amd

#endif  // _UTILS_OPTIONS_HPP_
