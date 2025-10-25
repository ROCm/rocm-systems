/* Copyright (c) 2008 - 2021 Advanced Micro Devices, Inc.

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

#include <string>
#include <map>
#include <numeric>
#include <cstring>
#include <cstdlib>
#include <sstream>
#include <iterator>
#include <cassert>
#include <regex>
#include "options.hpp"

namespace {
using namespace amd::option;

#define SKIP_INTERNAL_OPTION 1

/*
   Macro OPTION and NOPTION are general option macros.  And FLAG is a
   special, simple macro with the form -<flag>=<val>.  The FLAG is similar
   to a flag defined in flags.hpp, so that it is easier for people who are
   familiar with flags.hpp to use FLAG.
*/
#define OPTION(type, a, sn, ln, var, ideft, imin, imax, sdeft, desc)                               \
  {sn,                                                                                             \
   ln,                                                                                             \
   (type) | (a),                                                                                   \
   (uint32_t)reinterpret_cast<size_t>(&(((OptionVariables*)0)->var)),                              \
   (int64_t)ideft,                                                                                 \
   imin,                                                                                           \
   imax,                                                                                           \
   sdeft,                                                                                          \
   desc},

#define NOPTION(type, a, sn, ln, var, ideft, imin, imax, sdeft, desc)                              \
  {sn, ln, (type) | (a), 0, (int64_t)ideft, imin, imax, sdeft, desc},

/*
   FLAG is a special option with the form
         -<flag>=<val>
*/
#define FLAG(type, a, sn, var, deft, desc)                                                         \
  {sn,                                                                                             \
   NULL,                                                                                           \
   (type) | (a) | OA_RUNTIME | OVA_REQUIRED | OA_SEPARATOR_EQUAL,                                  \
   (uint32_t)reinterpret_cast<size_t>(&(((OptionVariables*)0)->var)),                              \
   (type == OT_CSTRING) ? (int64_t)0 : (int64_t)deft,                                              \
   (type == OT_INT32) ? -0x80000000LL : 0LL,                                                       \
   (type == OT_INT32) ? 0x7FFFFFFFLL : ((type == OT_UINT32) ? 0xFFFFFFFFLL : 0LL),                 \
   (type == OT_CSTRING) ? (const char*)deft : NULL,                                                \
   desc},


OptionDescriptor OptDescTable[] = {
#include "OPTIONS.def"
    {0, 0, 0, 0, 0, 0, 0, 0, 0}  // dummy entry
};

#undef OPTION
#undef NOPTION
#undef FLAG

#define OPTION_var(ix, ovars) (reinterpret_cast<char*>(ovars) + OPTION_offset(OptDescTable + ix))

/*
   [0] : map from option's short name to OptDescTable's index
   [1] : map from option's long name to OptDescTable's index

   Any prefix option (-f/-fno, -m/-mno) has no long name, and must have
   a value separator if it requires a value.
*/
std::map<std::string, int> OptionNameMap[2] ROCCLR_INIT_PRIORITY(101);
std::map<std::string, int> NoneSeparatorOptionMap[2] ROCCLR_INIT_PRIORITY(101);
// prefix -f/-fno- options
std::map<std::string, int> FOptionMap ROCCLR_INIT_PRIORITY(101);
// prefix -m/-mno- options
std::map<std::string, int> MOptionMap ROCCLR_INIT_PRIORITY(101);

bool setOptionVariable(OptionDescriptor* oDesc, OptionVariables* oVars, int64_t IValue,
                       const char* SValue);

/*
   Note:  When adding any option that has OVA_OPTIONAL as its value attribute,
          make sure that the code for returning the second default is added
          int this function!!!

   getDefault2() : get the second default value for an option given as
                   'OptDescTableIx'.

   The second default is a default VALUE if an option is PRESENT in the option string
   but its value is NOT GIVEN (its attribute must be OVA_OPTIONAL).  This is different
   from the (first) default that is registerred in OptionDescriptor, and the first
   default gives the default value when an option is NOT PRESENT in the option string.
   For instance,

   -O[0-3] :  optimization level
     first default : 3
        This means if there is no -O[0-3] present in the option string, the default
        is 3.
     second default : 3
        This means if -O is present but no level is given, the level is 3 (second
        default).

   In this case, the first default and the second default is the same, but in general
   they may be different. Take -O in gcc, the first default is 0, the second default
   is 2.

   We don't expect many options to have attributes OVA_OPTIONAL, therefore, the second
   default is set here in code instead of registerring it in OptionDescriptor. When
   adding an option that needs the second default, make sure this function return
   the second default.
*/
void getDefault2(int OptDescTableIx, int64_t& idefault2, const char*& sdefault2) {
  switch (OptDescTableIx) {
    case OID_ShowHelp:
      sdefault2 = "public";
      break;

    case OID_SaveTemps:
    case OID_SaveTempsAll:
      sdefault2 = NULL;
      break;

    case OID_OptLevel:
      idefault2 = amd::option::OPT_O3;
      break;

    case OID_OptUseNative:
      sdefault2 = "all";
      break;

    default:
      assert(false && "The second default value is not given");
      break;
  }
}

bool setAliasOptionVariable(int OptDescTableIx, Options& Opts, int64_t IValue, const char* SValue) {
  OptionDescriptor* od;
  OptionVariables* oVars = Opts.oVariables;

  switch (OptDescTableIx) {
    case OID_OptDisable:
      //
      // -cl-opt-disable is equivalent to -O0.
      //
      Opts.setFlag(OID_OptLevel, 1);
      od = &OptDescTable[OID_OptLevel];
      assert((IValue > 0) && "Internal Error: -cl-opt-disable is present, but with wrong value");
      if (!setOptionVariable(od, oVars, (int64_t)amd::option::OPT_O0, NULL)) {
        return false;
      }
      return true;

    case OID_GPU32BitIsa:
      // -m32 == "no -m64",  ie, 64-bit code generation off.
      Opts.setFlag(OID_GPU64BitIsa, 1);
      od = &OptDescTable[OID_GPU64BitIsa];
      assert((IValue > 0) && "Internal Error: -m32 is present, but with wrong value");
      if (!setOptionVariable(od, oVars, (int64_t)0, NULL)) {
        return false;
      }
      return true;

    case OID_SaveTemps:
    case OID_SaveTempsAll:
    case OID_Output:
      uint32_t flags;
      if (OptDescTableIx == OID_SaveTemps) {
        // Dump .cl, .i(.ii), .amdil, .isa, .s, dll, calimage
        flags = DUMP_CL | DUMP_I | DUMP_S | DUMP_O | DUMP_DLL | DUMP_CGIL | DUMP_DEBUGIL | DUMP_IL |
                DUMP_ISA;
      } else if (OptDescTableIx == OID_SaveTempsAll) {
        flags = DUMP_ALL;
      } else {  // OID_Output
        flags = DUMP_BIF;
        if (SValue == NULL) {
          assert(false && "Missing value for option -o <prefix>");
          return false;
        }
      }

      // Make sure flags is or'ed with the previous ones.
      if (Opts.isOptionSeen(OID_DumpFlags)) {
        flags |= oVars->DumpFlags;
      }

      Opts.setFlag(OID_DumpFlags, 1);
      od = &OptDescTable[OID_DumpFlags];
      if (!setOptionVariable(od, oVars, (int64_t)flags, NULL)) {
        return false;
      }

      if (SValue != NULL) {
        // Reset DumpPrefix
        Opts.setFlag(OID_DumpPrefix, 1);
        od = &OptDescTable[OID_DumpPrefix];
        if (!setOptionVariable(od, oVars, (int64_t)0, SValue)) {
          return false;
        }
      }
      return true;

    default:
      break;
  }

  return true;
}

void ShowOptionsHelp(const char* helpValue, Options& Opts) {
  // -h[--help][=public|all]  : public is the second default
  bool showAll = false;
  bool showSupport = false;

  /* If SKIP_INTERNAL_OPTION is set, only use public options */
#ifndef SKIP_INTERNAL_OPTION
  if (::strcmp(helpValue, "all") == 0) {
    showAll = true;
  } else if (::strcmp(helpValue, "support") == 0) {
    showSupport = true;
  }
#endif

  Opts.optionsLog() =
      "Option Summary:\n"
      "\tUse those options in clBuildProgram() directly in the OpenCL host application or\n"
      "\tduring runtime by setting environment variables : AMD_OCL_BUILD_OPTIONS and \n"
      "\tAMD_OCL_BUILD_OPTIONS_APPEND. AMD_OCL_BUILD_OPTIONS will override the options\n"
      "\tused in the host application, whereas AMD_OCL_BUILD_OPTIONS_APPEND appends to\n"
      "\tthe options. For example, assume '-cl-opt-disable' is used in the host application,\n"
      "\tand with\n"
      "\t\tset AMD_OCL_BUILD_OPTIONS=-O\n"
      "\tit will override '-cl-opt-disable' with '-O';  with\n"
      "\t\tset AMD_OCL_BUILD_OPTIONS_APPEND=-g\n"
      "\tit will append '-g' to '-cl-opt-disable' (ie '-cl-opt-disable -g').\n\n"
      "\tThose environment variables are useful for changing build options without changing\n"
      "\tand re-building (compiling) the host application.\n\n"
      "OPTIONS:\n";
  for (int i = 0; i < OID_LAST; ++i) {
    OptionDescriptor* od = &OptDescTable[i];

    if ((OPTION_vis(od) != OVIS_PUBLIC) && !showAll &&
        !(showSupport && (OPTION_vis(od) != OVIS_INTERNAL))) {
      continue;
    }

    // Print Value -- value to be shown in the help message
    std::string pntVal;
    switch (OPTION_type(od)) {
      case OT_CSTRING:
        if ((i == OID_WFComma) || (i == OID_WBComma) || (i == OID_WHComma)) {
          pntVal = "<options>";
        } else if (i == OID_SaveTemps) {
          pntVal = "<prefix>";
        } else {
          pntVal = "<value>";
        }
        break;
      case OT_BOOL:
        pntVal = "0|1";
        break;
      case OT_INT32:
      case OT_UINT32:
        pntVal = "<number>";
        break;
      case OT_UCHAR:
        pntVal = "<0-9 | a-z>";
      default:
        break;
    }


    const char* sname = OPTION_sname(od);
    const char* lname = OPTION_lname(od);
    size_t sz = 0;
    Opts.optionsLog() += "\t";
    if (OPTION_form(od) == OFA_NORMAL) {
      for (int j = 0; j < 2; ++j) {
        if (j == 0) {
          // Short Name
          if (sname != NULL) {
            Opts.optionsLog() += "-";
            Opts.optionsLog() += sname;
          } else {
            continue;
          }
        } else {  // j == 1
          // Long name
          if (lname != NULL) {
            Opts.optionsLog() += (sname != NULL) ? "  --" : "--";
            Opts.optionsLog() += lname;
          } else {
            continue;
          }
        }
        switch (i) {
          case OID_PP_D:
            Opts.optionsLog() += " name[=<definition>]";
            break;

          case OID_PP_I:
            Opts.optionsLog() += " dir";
            break;

          default:
            if ((OPTION_value(od) == OVA_OPTIONAL) || (OPTION_value(od) == OVA_REQUIRED)) {
              if (OPTION_value(od) == OVA_OPTIONAL) {
                Opts.optionsLog() += "[";
              }
              if ((OPTION_info(od) & OA_SEPARATOR_NONE) &&
                  (OPTION_info(od) & (OA_SEPARATOR_EQUAL | OA_SEPARATOR_SPACE))) {
                if ((OPTION_info(od) & OA_SEPARATOR_EQUAL) &&
                    (OPTION_info(od) & OA_SEPARATOR_SPACE)) {
                  Opts.optionsLog() += "[ |=]";
                } else if (OPTION_info(od) & OA_SEPARATOR_EQUAL) {
                  Opts.optionsLog() += "[=]";
                } else if (OPTION_info(od) & OA_SEPARATOR_SPACE) {
                  Opts.optionsLog() += "[ ]";
                }
              } else if (OPTION_info(od) & (OA_SEPARATOR_EQUAL | OA_SEPARATOR_SPACE)) {
                if ((OPTION_info(od) & OA_SEPARATOR_EQUAL) &&
                    (OPTION_info(od) & OA_SEPARATOR_SPACE)) {
                  Opts.optionsLog() += "{ |=}";
                } else if (OPTION_info(od) & OA_SEPARATOR_EQUAL) {
                  Opts.optionsLog() += "=";
                } else {
                  Opts.optionsLog() += " ";
                }
              }

              Opts.optionsLog() += pntVal.c_str();
              if (OPTION_value(od) == OVA_OPTIONAL) {
                Opts.optionsLog() += "]";
              }
            }
            break;
        }  // switch
      }
    } else if ((OPTION_form(od) == OFA_PREFIX_F) || (OPTION_form(od) == OFA_PREFIX_M)) {
      char fORm = ((OPTION_form(od) == OFA_PREFIX_F) ? 'f' : 'm');
      if (OPTION_type(od) == OT_BOOL) {
        Opts.optionsLog() += '-';
        Opts.optionsLog() += fORm;
        Opts.optionsLog() += "[no-]";
        Opts.optionsLog() += sname;
      } else {
        Opts.optionsLog() += fORm;
        Opts.optionsLog() += sname;

        if ((OPTION_value(od) == OVA_OPTIONAL) || (OPTION_value(od) == OVA_REQUIRED)) {
          assert(!(OPTION_info(od) & OA_SEPARATOR_NONE) &&
                 (OPTION_info(od) & (OA_SEPARATOR_EQUAL | OA_SEPARATOR_SPACE)) &&
                 "prefix -f/-m need either ' ' or '=' as value separator");
          if ((OPTION_info(od) & OA_SEPARATOR_EQUAL) && (OPTION_info(od) & OA_SEPARATOR_SPACE)) {
            Opts.optionsLog() += "{ |=}";
          } else if (OPTION_info(od) & OA_SEPARATOR_EQUAL) {
            Opts.optionsLog() += "=";
          } else {
            Opts.optionsLog() += " ";
          }
        }
      }
    }
    Opts.optionsLog() += "\n\t    ";
    Opts.optionsLog() += OPTION_desc(od);
    Opts.optionsLog() += "\n\n";
  }
}

bool setOptionVariable(OptionDescriptor* oDesc, OptionVariables* oVars, int64_t IValue,
                       const char* SValue) {
  char* addr = reinterpret_cast<char*>(oVars) + oDesc->OptionOffset;
  if (OPTION_type(oDesc) == OT_BOOL) {
    OT_BOOL_t* o = reinterpret_cast<OT_BOOL_t*>(addr);
    *o = static_cast<OT_BOOL_t>(IValue != 0);
  } else if (OPTION_type(oDesc) == OT_INT32) {
    OT_INT32_t* o = reinterpret_cast<OT_INT32_t*>(addr);
    *o = static_cast<OT_INT32_t>(IValue);
  } else if (OPTION_type(oDesc) == OT_UINT32) {
    OT_UINT32_t* o = reinterpret_cast<OT_UINT32_t*>(addr);
    *o = static_cast<OT_UINT32_t>(IValue);
  } else if (OPTION_type(oDesc) == OT_CSTRING) {
    OT_CSTRING_t* o = reinterpret_cast<OT_CSTRING_t*>(addr);
    *o = static_cast<OT_CSTRING_t>(SValue);
  } else if (OPTION_type(oDesc) == OT_UCHAR) {
    OT_UCHAR_t* o = reinterpret_cast<OT_UCHAR_t*>(addr);
    *o = static_cast<OT_UCHAR_t>(IValue);
  } else {
    return false;
  }
  return true;
}

static bool setOptionVariable(OptionDescriptor* oDesc, OptionVariables* srcOVars,
                              OptionVariables* dstOVars) {
  char* srcAddr = reinterpret_cast<char*>(srcOVars) + oDesc->OptionOffset;
  char* dstAddr = reinterpret_cast<char*>(dstOVars) + oDesc->OptionOffset;
  if (OPTION_type(oDesc) == OT_BOOL) {
    OT_BOOL_t* src = reinterpret_cast<OT_BOOL_t*>(srcAddr);
    OT_BOOL_t* dst = reinterpret_cast<OT_BOOL_t*>(dstAddr);
    *dst = *src;
  } else if (OPTION_type(oDesc) == OT_INT32) {
    OT_INT32_t* src = reinterpret_cast<OT_INT32_t*>(srcAddr);
    OT_INT32_t* dst = reinterpret_cast<OT_INT32_t*>(dstAddr);
    *dst = *src;
  } else if (OPTION_type(oDesc) == OT_UINT32) {
    OT_UINT32_t* src = reinterpret_cast<OT_UINT32_t*>(srcAddr);
    OT_UINT32_t* dst = reinterpret_cast<OT_UINT32_t*>(dstAddr);
    *dst = *src;
  } else if (OPTION_type(oDesc) == OT_UCHAR) {
    OT_UCHAR_t* src = reinterpret_cast<OT_UCHAR_t*>(srcAddr);
    OT_UCHAR_t* dst = reinterpret_cast<OT_UCHAR_t*>(dstAddr);
    *dst = *src;
  } else if (OPTION_type(oDesc) == OT_CSTRING) {
    OT_CSTRING_t* src = reinterpret_cast<OT_CSTRING_t*>(srcAddr);
    OT_CSTRING_t* dst = reinterpret_cast<OT_CSTRING_t*>(dstAddr);
    *dst = *src;
  } else {
    return false;
  }
  return true;
}

int getOptionDesc(std::string& options, size_t StartPos, bool IsShortForm, OptionForm oForm,
                  size_t& EndPos, std::string& Value) {
  int option_ndx = -1;
  EndPos = std::string::npos;
  const int map_ndx = (IsShortForm ? 0 : 1);

  size_t sPos = StartPos;
  size_t ePos = options.find(' ', sPos);
  size_t pos = options.find('=', sPos);
  if (pos < ePos) {
    ePos = pos;
  }
  // The following should work even if ePos is npos
  size_t len = ePos - sPos;

  // Handle the special options-passing option : -W<l>,s0,s1,s2,s3,....sn
  // where l is a single letter. And only in this option, comma ',' is treated
  // like a separator.
  if ((oForm == OFA_NORMAL) && (options.size() > (sPos + 3))) {
    char c0 = options[sPos];
    char c1 = options[sPos + 1];
    char c2 = options[sPos + 2];
    if ((c0 == 'W') && (c2 == ',') && (c1 >= 'a') && (c1 <= 'z')) {
      ePos = sPos + 3;
      len = 3;
    }
  }
  std::string name = options.substr(sPos, len);

  std::map<std::string, int>::const_iterator I, IE;
  switch (oForm) {
    case OFA_NORMAL:
      I = OptionNameMap[map_ndx].find(name);
      IE = OptionNameMap[map_ndx].end();
      break;
    case OFA_PREFIX_F:
      I = FOptionMap.find(name);
      IE = FOptionMap.end();
      break;
    case OFA_PREFIX_M:
      I = MOptionMap.find(name);
      IE = MOptionMap.end();
      break;
    default:
      return -1;
  }

  if (I != IE) {
    // found the exact match, that's it!
    option_ndx = I->second;
    pos = ePos;
  } else if (oForm != OFA_NORMAL) {
    return -1;
  } else {
    I = NoneSeparatorOptionMap[map_ndx].begin();
    IE = NoneSeparatorOptionMap[map_ndx].end();
    size_t len1 = 0;
    for (; I != IE; ++I) {
      std::string namevalue = I->first;
      size_t n = namevalue.size();
      if (n >= len) {
        // Not substr, skip
        continue;
      }
      if (name.compare(0, n, namevalue) == 0) {
        // found the substr
        if (n > len1) {
          len1 = n;
          option_ndx = I->second;
        }
      }
    }
    if (len1 == 0) {
      // no matching
      return -1;
    }
    pos = sPos + len1;
  }

  OptionDescriptor* od = &OptDescTable[option_ndx];
  if (pos == std::string::npos) {
    if (OPTION_value(od) == OVA_REQUIRED) {
      // illegal option
      return -1;
    }
    EndPos = pos;
    return option_ndx;
  }

  char next_c = options.at(pos);
  bool optionalHasValue =
      (OPTION_value(od) == OVA_OPTIONAL) &&
      (((OPTION_info(od) & OA_SEPARATOR_EQUAL) && (next_c == '=')) ||
       ((OPTION_info(od) & OA_SEPARATOR_NONE) && !OPTION_valueSeparator(next_c)));
  bool hasValue = (OPTION_value(od) == OVA_REQUIRED) || optionalHasValue;

  if (hasValue) {
    if ((OPTION_info(od) & OA_SEPARATOR_EQUAL) && (next_c == '=')) {
      pos++;
      if (pos == options.size()) {
        return -1;
      }
    } else if ((OPTION_info(od) & OA_SEPARATOR_SPACE) && (next_c == ' ')) {
      pos = options.find_first_not_of(' ', pos);
    } else if ((OPTION_info(od) & OA_SEPARATOR_NONE) && !OPTION_valueSeparator(next_c)) {
      ;
    } else {
      // illgel option
      return -1;
    }

    if (pos == std::string::npos) {
      return -1;
    }

    if ((OPTION_type(od) == OT_CSTRING) && options.at(pos) == '"') {
      size_t sz = options.size();
      if ((pos + 1) >= sz) {
        return -1;
      }
      /* Handle quoted string value */
      ePos = options.find('"', pos + 1);
      if (ePos == std::string::npos) {
        return -1;
      }

      /* Advance ePos to the next position or npos */
      if (ePos + 1 < sz) {
        ++ePos;
        if (options.at(ePos) != ' ') {
          return -1;
        }
      } else {
        ePos = std::string::npos;
      }
    } else {
      ePos = options.find(' ', pos);
    }
    if (OPTION_info(od) & OA_RUNTIME) {
      Value = options.substr(pos, ePos - pos);
    }
    EndPos = ePos;
  } else {
    if (next_c != ' ') {
      return -1;
    }
    EndPos = pos;
  }
  return option_ndx;
}

bool processOption(int OptDescTableIx, Options& Opts, const std::string& Value, bool IsPrefixOption,
                   bool IsOffFlag, bool IsLC) {
  OptionVariables* ovars = Opts.oVariables;
  OptionDescriptor* od = &OptDescTable[OptDescTableIx];

  const char* sval = NULL;
  int64_t ival = 0;
  int otype = OPTION_type(od);

  if (Value.size() == 0) {
    if (OPTION_value(od) == OVA_OPTIONAL) {
      // Get the second default value !
      getDefault2(OptDescTableIx, ival, sval);
    } else {
      if (otype == OT_BOOL) {
        // This option may be either OVA_REQUIRED or OVA_DISALLOWED !
        ival = (IsOffFlag ? 0 : 1);
      } else {
        // This option is either OT_CSTRING or OT_{U]INT32, and requires its value!
        assert(false && "This option should have a value");
        Opts.optionsLog() = "Value is missing\n";
        return false;
      }
    }
  } else {
    // An explicit value has been provided.
    char* p;
    unsigned long uval;

    switch (otype) {
      case OT_CSTRING: {
        char* cs = new char[Value.size() + 1];
        Value.copy(cs, Value.size());
        cs[Value.size()] = 0;
        sval = cs;

        // Need to remember this and free memory later.
        Opts.recordMemoryHandle((char*)sval);
      } break;

      case OT_UCHAR: {
        ival = Value.at(0);
      } break;

      case OT_INT32:
        ival = ::strtol(Value.c_str(), &p, 0);
        if (*p != '\0') {
          // invalid value
          Opts.optionsLog() = "Value is wrong\n";
          return false;
        }

        if ((ival < OPTION_min(od)) || (ival > OPTION_max(od))) {
          std::stringstream msg;
          msg << "Value should be in [" << (int)OPTION_min(od) << ", " << (int)OPTION_max(od)
              << "]\n";
          Opts.optionsLog() = msg.str().c_str();
          return false;
        }
        break;

      case OT_UINT32:
      case OT_BOOL:
        uval = ::strtoul(Value.c_str(), &p, 0);
        if (*p != '\0') {
          // invalid value
          Opts.optionsLog() = "Value is wrong\n";
          return false;
        }

        ival = (int64_t)uval;

        if (otype == OT_BOOL) {
          if ((ival < 0) || (ival > 1)) {
            Opts.optionsLog() = "Value should be either 0 or 1\n";
            return false;
          }
        } else {
          // OT_INT32, OT_UINT32
          if ((ival < OPTION_min(od)) || (ival > OPTION_max(od))) {
            std::stringstream msg;
            msg << "Value should be in [" << (int)OPTION_min(od) << ", " << (int)OPTION_max(od)
                << "]\n";
            Opts.optionsLog() = msg.str().c_str();
            return false;
          }
        }
        break;
      default:
        break;
    }
  }

  // First, if it is an alias option, special-process it here.
  if (OPTION_info(od) & OA_MISC_ALIAS) {
    if (!setAliasOptionVariable(OptDescTableIx, Opts, ival, sval)) {
      Opts.optionsLog() = "Wrong value for the option (alias)\n";
      return false;
    }
    return true;
  }

  // Special procesing of an individual option (non-alias option)
  OptionDescriptor* tod;
  switch ((OptionIdentifier)OptDescTableIx) {
    case OID_ShowHelp:
      if ((strcmp(sval, "all") != 0) && (strcmp(sval, "support") != 0) &&
          (strcmp(sval, "public") != 0)) {
        Opts.optionsLog() = "-h/--help only supports values all|support|public\n";
        return false;
      }
      break;

    case OID_FiniteMathOnly:
      Opts.setFlag(OID_FiniteMathOnly, 1);
      tod = &OptDescTable[OID_FiniteMathOnly];
      (void)setOptionVariable(tod, ovars, (int64_t)1, NULL);
      Opts.clangOptions.push_back("-cl-finite-math-only");
      break;

    case OID_NoSignedZeros:
      Opts.setFlag(OID_NoSignedZeros, 1);
      tod = &OptDescTable[OID_NoSignedZeros];
      (void)setOptionVariable(tod, ovars, (int64_t)1, NULL);
      Opts.clangOptions.push_back("-cl-no-signed-zeros");
      break;

    case OID_FastRelaxedMath:
      // -cl-fast-relaxed-math implies:
      //    -cl-finite-math-only
      //    -cl-unsafe-math-optimizations
      Opts.setFlag(OID_FiniteMathOnly, 1);
      tod = &OptDescTable[OID_FiniteMathOnly];
      (void)setOptionVariable(tod, ovars, (int64_t)1, NULL);

      Opts.setFlag(OID_UnsafeMathOpt, 1);
      tod = &OptDescTable[OID_UnsafeMathOpt];
      (void)setOptionVariable(tod, ovars, (int64_t)1, NULL);

      Opts.clcOptions.append(" -D__FAST_RELAXED_MATH__=1");
      Opts.clangOptions.push_back("-D__FAST_RELAXED_MATH__=1");
      if (IsLC) {  // w/a for SWDEV-116690
        Opts.clangOptions.push_back("-cl-fast-relaxed-math");
      }

      // fall-through to handle UnsafeMathOpt
    case OID_UnsafeMathOpt:
      // -cl-unsafe-math-optimizations implies
      //    -cl-no-signed-zeros
      //    -cl-mad-enable.
      Opts.setFlag(OID_NoSignedZeros, 1);
      tod = &OptDescTable[OID_NoSignedZeros];
      (void)setOptionVariable(tod, ovars, (int64_t)1, NULL);

      Opts.setFlag(OID_MadEnable, 1);
      tod = &OptDescTable[OID_MadEnable];
      (void)setOptionVariable(tod, ovars, (int64_t)1, NULL);
      Opts.clangOptions.push_back("-cl-unsafe-math-optimizations");
      break;

    case OID_DenormsAreZero:
      Opts.setFlag(OID_DenormsAreZero, 1);
      tod = &OptDescTable[OID_DenormsAreZero];
      (void)setOptionVariable(tod, ovars, (int64_t)1, NULL);
      Opts.clangOptions.push_back("-cl-denorms-are-zero");
      break;

    case OID_StricAliasing:
      Opts.setFlag(OID_StricAliasing, 1);
      tod = &OptDescTable[OID_StricAliasing];
      (void)setOptionVariable(tod, ovars, (int64_t)1, NULL);
      Opts.clangOptions.push_back("-cl-strict-aliasing");
      break;

    case OID_MadEnable:
      Opts.setFlag(OID_MadEnable, 1);
      tod = &OptDescTable[OID_MadEnable];
      (void)setOptionVariable(tod, ovars, (int64_t)1, NULL);
      Opts.clangOptions.push_back("-cl-mad-enable");
      break;

    case OID_EnableDebug:
      Opts.clcOptions.append(" -g");
      Opts.clangOptions.push_back("-g");
      break;

    case OID_SinglePrecisionConstant:
      Opts.clcOptions.append(" --single_precision_constant");
      Opts.clangOptions.push_back("-cl-single-precision-constant");
      break;

    case OID_FP32RoundDivideSqrt:
      Opts.clcOptions.append(" --precise_fp32_divide_sqrt");
      Opts.clangOptions.push_back("-cl-fp32-correctly-rounded-divide-sqrt");
      Opts.setFlag(OID_FP32RoundDivideSqrt, 1);
      tod = &OptDescTable[OID_FP32RoundDivideSqrt];
      (void)setOptionVariable(tod, ovars, (int64_t)1, NULL);
      break;

    case OID_EnableC99Inline:
      if (ival != 0) {
        Opts.clangOptions.push_back("-fc99-inline");
      }
      break;

    case OID_DisableAllWarnings:
      if (ival != 0) {
        Opts.clcOptions.append(" --no_warnings");
        Opts.clangOptions.push_back("-w");
      }
      break;

    case OID_WarnToError:
      if (ival != 0) {
        Opts.clcOptions.append(" --werror");
        Opts.clangOptions.push_back("-Werror");
      }
      break;

    case OID_UniformWorkGroupSize:
      Opts.clangOptions.push_back("-cl-uniform-work-group-size");
      break;

    case OID_WorkGrpSize:
      // -wgs=x[,y[,z]]. The default for missing part is 1.
      if (sval) {
        bool isValueWrong = false;
        const char* bp = sval;
        char* ep = 0;
        int dim = 0;
        while (*bp) {
          uint32_t tval = ::strtoul(bp, &ep, 0);
          if (ep != bp) {
            if (dim > 2) {
              isValueWrong = true;
              break;
            }
            Opts.WorkGroupSize[dim] = tval;
            ++dim;
            if (*ep == ',') {
              bp = ep + 1;
            } else if (*ep == 0) {
              break;
            } else {
              isValueWrong = true;
              break;
            }
          } else {
            isValueWrong = true;
            break;
          }
        }

        if (isValueWrong) {
          Opts.optionsLog() = "Wrong option value -wgs=";
          Opts.optionsLog() += sval;
          Opts.optionsLog() += "\n";
          return false;
        }

        // If there are missing dimention, assume it is 1.
        for (int i = dim; i < 3; ++i) {
          Opts.WorkGroupSize[i] = 1;
        }
        Opts.setDefaultWGS(false);
      }
      break;

    case OID_OptUseNative:
      if (IsLC) {
        Opts.llvmOptions.append(" -mllvm -amdgpu-use-native=");
        Opts.llvmOptions.append(sval);
      }
      break;

    case OID_WFComma:
    case OID_WBComma:
    case OID_WHComma:
      if (sval != NULL) {
        // we know that sval was new'ed
        for (char* p = (char*)sval; *p; ++p) {
          if (*p == ',') {
            *p = ' ';
          }
        }

        if ((OptionIdentifier)OptDescTableIx == OID_WFComma) {
          Opts.clcOptions.append(" ");
          Opts.clcOptions.append(sval);
          if (strcmp(sval, "--force_disable_spir") && strcmp(sval, "--single_precision_constant") &&
              strcmp(sval, "--precise_fp32_divide_sqrt"))
            Opts.clangOptions.push_back(sval);
        } else if (((OptionIdentifier)OptDescTableIx) == OID_WBComma) {
          Opts.llvmOptions.append(" ");
          if (IsLC) {
            Opts.llvmOptions.append("-mllvm ");
          }
          Opts.llvmOptions.append(sval);
        } else if (((OptionIdentifier)OptDescTableIx) == OID_WHComma) {
          Opts.finalizerOptions.push_back(sval);
        }
      }
      break;

    case OID_XLang:
#ifndef SKIP_INTERNAL_OPTION
      // Don't expose "-x il | cgil" to mainline
      if ((strcmp(sval, "il") == 0) || (strcmp(sval, "cgil") == 0)) {
        break;
      }
#endif
      if ((strcmp(sval, "clc") != 0) && (strcmp(sval, "clc++") != 0) &&
          (strcmp(sval, "spir") != 0)) {
        Opts.optionsLog() = "-x only supports values clc, clc++ and spir\n";
        return false;
      }

      // ToDo: EDG is not ready to produce SPIR
      //  do not pass --spir to EDG
      if (strcmp(sval, "spir") != 0) {
        // clcOptions should be --c++, --c or --spir

        // add "--" prefix for clcOptions
        Opts.clcOptions.append(" --");

        // skip "cl" prefix in the sval for clcOptions
        Opts.clcOptions.append(sval + (strcmp(sval, "spir") != 0 ? 2 : 0));
      }

      break;

    case OID_FSanitize:
      if (sval != NULL) Opts.clangOptions.push_back("-fsanitize=" + std::string(sval));
      break;
    default:
      break;
  }

  if (!setOptionVariable(od, ovars, ival, sval)) {
    Opts.optionsLog() = "Wrong option value\n";
    return false;
  }
  return true;
}

// the option at "bpos" of the "options" string is an invalid option,
// log the error message into "log"
static void logInvalidOption(std::string& options, size_t bpos, std::string& log,
                             const std::string& msg) {
  size_t epos = options.find_first_of(' ', bpos);
  log = "Invalid option: ";
  log += options.substr(bpos, (epos == std::string::npos) ? epos : epos - bpos);
  log += msg + "\n";
}

}  // namespace

namespace amd {

namespace option {

bool parseAllOptions(std::string& options, Options& Opts, bool linkOptsOnly, bool isLC) {
  Opts.origOptionStr = options;
  OptionVariables* ovars = Opts.oVariables;
  OptionDescriptor* od = OptDescTable;

  // Initialize all options to the default
  for (int i = 0; i < OID_LAST; ++i, ++od) {
    if (!OPTIONHasOVariable(od)) {
      continue;
    }
    if (!setOptionVariable(od, ovars, OPTION_default(od), OPTION_defaultstr(od))) {
      Opts.optionsLog() = "Internal Error: option processing failed\n";
      return false;
    }
  }
  Opts.clangOptions.push_back("-cl-kernel-arg-info");

  // Parse options
  if (options.empty()) {
    Opts.postParseInit();
    return true;
  }

  bool isLibLinkOpts = false;  // is this set of options for linking library?
  bool firstOpt = true;
  for (size_t pos = options.find_first_not_of(' ', 0); pos != std::string::npos;
       pos = options.find_first_not_of(" ", pos)) {
    bool isShortName = true;  // -: short name; --: long name

    // For creating option log
    size_t bpos = pos;

    if (options.at(pos) == '-') {
      pos++;
    } else {
      // options should start with "-"
      logInvalidOption(options, bpos, Opts.optionsLog(), "  (expected - at the beginning)");
      return false;
    }

    if (options.at(pos) == '-') {
      isShortName = false;
      pos++;
    }

    if ((pos == std::string::npos) || OPTION_valueSeparator(options.at(pos))) {
      logInvalidOption(options, bpos, Opts.optionsLog(), "  (expected an option name)");
      return false;
    }

    bool isPrefix_fno = false;
    bool isPrefix_mno = false;
    bool isPrefix_option = false;

    std::string value;
    size_t sPos = pos;
    int option_ndx = getOptionDesc(options, sPos, isShortName, OFA_NORMAL, pos, value);
    if (option_ndx < 0) {
      size_t sPos1;
      pos = sPos;
      if (options.at(pos) == 'f') {
        isPrefix_option = true;
        isPrefix_fno = (options.compare(pos + 1, (size_t)3, "no-") == 0);

        sPos1 = pos + (isPrefix_fno ? 4 : 1);
        option_ndx = getOptionDesc(options, sPos1, isShortName, OFA_PREFIX_F, pos, value);
      } else if (options.at(pos) == 'm') {
        isPrefix_option = true;
        isPrefix_mno = (options.compare(pos + 1, (size_t)3, "no-") == 0);

        sPos1 = pos + (isPrefix_mno ? 4 : 1);
        option_ndx = getOptionDesc(options, sPos1, isShortName, OFA_PREFIX_M, pos, value);
      }

      if (option_ndx < 0) {
        logInvalidOption(options, bpos, Opts.optionsLog(), "");
        return false;
      }
    }

    od = &OptDescTable[option_ndx];

#ifdef SKIP_INTERNAL_OPTION
    if (OPTION_vis(od) == OVIS_INTERNAL) {
      // Internal options are not support in the product
      logInvalidOption(options, bpos, Opts.optionsLog(), "");
      return false;
    }
#endif

    if (!linkOptsOnly && (OPTION_info(od) & OA_CLC)) {
      size_t sPos1 = (isShortName ? sPos - 1 : sPos - 2);
      std::string oStr = options.substr(sPos1, pos - sPos1);
      if (OPTION_info(od) & OA_CLC) {
        Opts.clcOptions.append(" " + oStr);
        if (!oStr.compare(0, 2, "-D") || !oStr.compare(0, 2, "-I")) {
          //  strip off the leading whitespaces in macro
          //  definition and include path. Clang treats
          //  whitespaces as part of macro and include paths.
          size_t vPos1 = oStr.find_first_not_of(" ", 2);
          if (vPos1 == std::string::npos) {
            // Do not allow blank macro and include directories
            logInvalidOption(options, bpos, Opts.optionsLog(), "  (expected value)");
            return false;
          }
          std::string vStr = oStr.substr(vPos1, std::string::npos);
          if ((vStr.size() > 2) && (vStr.front() == '\"' && vStr.back() == '\"')) {
            // Unquote string
            vStr = vStr.substr(1, vStr.size() - 2);
          }
          Opts.clangOptions.push_back(oStr.substr(0, 2) + vStr);
        }
      }
    }

    if (linkOptsOnly) {
      if (firstOpt) {
        isLibLinkOpts = ((OPTION_info(od) & OA_LINK_LIB)) != 0;
        firstOpt = false;
      }

      if (!(OPTION_info(od) & OA_RUNTIME) || (!isLibLinkOpts && !(OPTION_info(od) & OA_LINK_EXE)) ||
          (isLibLinkOpts && !(OPTION_info(od) & OA_LINK_LIB))) {
        // Do not allow non-link-time options
        logInvalidOption(options, bpos, Opts.optionsLog(), "  (bad link-time option)");
        return false;
      }
    } else {
      if (!(OPTION_info(od) & OA_RUNTIME)) continue;
    }

    if (!processOption(option_ndx, Opts, value, isPrefix_option, (isPrefix_mno || isPrefix_fno),
                       isLC)) {
      // Keep the optionsLog set in processOption().
      std::string tmpStr("Invalid option: ");
      tmpStr += options.substr(bpos, (pos == std::string::npos) ? pos : pos - bpos);
      tmpStr += "\n    ";
      Opts.optionsLog().insert(0, tmpStr);
      return false;
    }

    Opts.setFlag(option_ndx, 1);
  }

  if (Opts.isOptionSeen(OID_ShowHelp)) {
    OT_CSTRING_t* p = reinterpret_cast<OT_CSTRING_t*>(OPTION_var(OID_ShowHelp, ovars));
    const char* arg = (const char*)*p;
    ShowOptionsHelp(arg, Opts);
  }

  // Set up llvmargv if llvmOptions is not empty
  if (!Opts.llvmOptions.empty()) {
    std::string* ostr = &Opts.llvmOptions;
    int n;
    size_t pos1, pos2;
    pos1 = ostr->find_first_not_of(' ');
    for (n = 0; pos1 != std::string::npos; ++n) {
      pos2 = ostr->find_first_of(' ', pos1);
      if (pos2 != std::string::npos) {
        pos2 = ostr->find_first_not_of(' ', pos2);
      }
      pos1 = pos2;
    }
    if (n > 0) {
      char* data = new char[sizeof(char*) * (n + 1) +  // for argv[0:n]
                            ostr->size() + 1];         // for all option chars
      char** t_argv = (char**)data;
      static char pseudoCmdName[] = "llvmOptCodegen";

      t_argv[0] = &pseudoCmdName[0];  // pseudo command name
      Opts.setLLVMArgs(n + 1, t_argv);
      Opts.recordMemoryHandle(data);

      // Initialize all arguments
      t_argv++;
      data = (char*)(t_argv + n);
      int i = 0;
      pos1 = ostr->find_first_not_of(' ');
      while (pos1 != std::string::npos) {
        pos2 = ostr->find_first_of(' ', pos1);
        size_t len;
        if (pos2 == std::string::npos) {
          len = ostr->size() - pos1;
        } else {
          len = pos2 - pos1;
          pos2 = ostr->find_first_not_of(' ', pos2);
        }
        ostr->copy(data, len, pos1);
        data[len] = 0;
        t_argv[i++] = data;
        data += (len + 1);
        pos1 = pos2;
      }
    }
  }

  // if the set of options is OA_LINK_LIB options, the "-create-library"
  // option should be included
  if (linkOptsOnly && isLibLinkOpts && !Opts.isOptionSeen(OID_clCreateLibrary)) {
    Opts.optionsLog() =
        "Invalid option:"
        " library link options used without -create-library";
    return false;
  }

  if (Opts.isOptionSeen(OID_UniformWorkGroupSize) && (strcmp(ovars->CLStd, "CL1.2") == 0)) {
    Opts.optionsLog() =
        "Invalid option:"
        " -cl-uniform-work-group-size is not supported in OpenCL 1.x\n";
    return false;
  }
  Opts.postParseInit();
  return true;
}

bool init() {
  // Set up Options Map
  for (int i = 0; i < OID_LAST; ++i) {
    OptionDescriptor* od = &OptDescTable[i];
    const char* sname = OPTION_sname(od);
    const char* lname = OPTION_lname(od);

    // Make sure alias option is initialized correctly.
    if (OPTION_info(od) & OA_MISC_ALIAS) {
      // alias option must be in runtime group
      OPTION_info(od) |= OA_RUNTIME;
      assert((OPTION_offset(od) == 0) &&
             "Alias option should be RUNTIME option and has offset zero");
    }

    if (OPTION_form(od) == OFA_NORMAL) {
      if (sname != NULL) {
        OptionNameMap[0][sname] = i;
      }
      if (lname != NULL) {
        OptionNameMap[1][lname] = i;
      }
      if (((OPTION_value(od) == OVA_OPTIONAL) || (OPTION_value(od) == OVA_REQUIRED)) &&
          (OPTION_info(od) & OA_SEPARATOR_NONE)) {
        if (sname != NULL) {
          NoneSeparatorOptionMap[0][sname] = i;
        }
        if (lname != NULL) {
          NoneSeparatorOptionMap[1][sname] = i;
        }
      }
    } else if (OPTION_form(od) == OFA_PREFIX_F) {
      assert(((OPTION_value(od) == OVA_DISALLOWED) || !(OPTION_info(od) & OA_SEPARATOR_NONE)) &&
             (lname == NULL) &&
             "-f/-fno- option may not have a long name, and"
             "must have a value separator if it requires a value");
      FOptionMap[sname] = i;
    } else if (OPTION_form(od) == OFA_PREFIX_M) {
      assert(((OPTION_value(od) == OVA_DISALLOWED) || !(OPTION_info(od) & OA_SEPARATOR_NONE)) &&
             (lname == NULL) &&
             "-m/-mno- option may not have a long name, and"
             "must have a value separator if it requires a value");
      MOptionMap[sname] = i;
    }
  }
#if 0
    std::map <std::string, int>::const_iterator I, IE;
    IE = OptionNameMap[0].end();
    for (I = OptionNameMap[0].begin(); I != IE; ++I) {
        printf (" %s : %d \n", I->first.c_str(), I->second);
    }
#endif
  return true;
}

bool teardown() { return true; }

OptionDescriptor* getOptDescTable() { return &OptDescTable[0]; }

Options::Options()
    : oVariables(NULL),
      clcOptions(),
      llvmOptions(),
      kernelArgAlign(0),
      basename_max(0),
      OptionsLog(),
      llvmargc(0),
      llvmargv(NULL),
      buildNo(0),
      dumpFileRoot(),
      currKernelName(NULL),
      encryptCode(0),
      MemoryHandles(),
      libraryType_(amd::LibraryUndefined) {
  oVariables = new OptionVariables();
  ::memset(flags, 0, sizeof(flags));

  // Set -1 to use default
  WorkGroupSize[0] = -1;
  WorkGroupSize[1] = -1;
  WorkGroupSize[2] = -1;
  UseDefaultWGS = true;
  NumAvailGPRs = -1;

  sourceFileName_.reserve(1024);
}

Options::~Options() {
  for (int i = 0; i < (int)MemoryHandles.size(); ++i) {
    delete[] MemoryHandles[i];
  }
  if (oVariables) {
    delete oVariables;
  }
}

void Options::postParseInit() {
  if (!oVariables->EnableDumpKernel) {
    oVariables->DumpFlags = 0;
  }

#if 0
    // set -march=gpu-64 for clc
    if (oVariables->GPU64BitIsa) {
        clcOptions.append(" --march=gpu-64");
    }
#endif

  oVariables->UseJIT = oVariables->ForceJIT || (oVariables->UseJIT && !oVariables->EnableDebug);
#ifdef SKIP_INTERNAL_OPTION
  oVariables->DumpFlags &= DUMP_PRODUCT_FLAGS;
#endif

  /*
      if (oVariables->OptEstNumGPRs == -1) {
          oVariables->OptEstNumGPRs = getDefaultGPRs();
      }
  */
}

void Options::setPerBuildInfo(const char* val, int encrypt, bool device) {
  encryptCode = encrypt;
  setDumpFileName(val);

  if (device) {
    if (useDefaultWGS()) {
      WorkGroupSize[0] = 256;
      WorkGroupSize[1] = 1;
      WorkGroupSize[2] = 1;
    }
    // Get an estimate number of available GPRs for each thread
    // Can do it more precisely. For now, it is okay doing so.
    int numWF = 4;
    int wfsize = 64;
    if (wfsize > 0) {
      numWF = WorkGroupSize[0] / wfsize;
    }
    if (numWF < 2) {
      NumAvailGPRs = 128;
    } else {
      NumAvailGPRs = 256 / numWF;
    }
  } else {
    // Unused for CPU
    NumAvailGPRs = -1;
  }
}


// HashString - Hash function for strings.
// This is the Bernstein hash function.
static inline unsigned HashString(std::string::const_iterator it, std::string::const_iterator end) {
  auto Bernstein = [](unsigned Result, unsigned char c) { return Result * 33 + c; };
  return std::accumulate(it, end, 0U, Bernstein);
}

static inline std::string to_string(unsigned number) {
  std::ostringstream ss;
  ss << number;
  return ss.str();
}

static void splitFileName(const std::string& fileName, std::string& path, std::string& baseName) {
  std::size_t found = fileName.find_last_of("/\\");
  if (found != std::string::npos) {
    path = fileName.substr(0, found + 1);
    baseName = fileName.substr(found + 1);
  } else {
    baseName = fileName;
  }
}

#ifdef _WIN32
static inline bool isFullPath(const std::string& path) {
  size_t found = path.find(":");
  return found != std::string::npos;
}

static std::string getValidDumpPath(const std::string& path) {
  if (path.empty()) {
    return path;
  }

  std::string validPath;
  unsigned long pathname_max = _MAX_PATH;
  if (path.size() > pathname_max) {
    // On Windows the max file path is 32,767 if you prefix "\\?\" with full path
    if (isFullPath(path)) {
      const unsigned new_win_path = _MAX_ENV;
      // prefix "\\?\"
      validPath = "\\\\?\\" + path;
      assert(path.size() < new_win_path);
    } else {
      assert(path.size() < pathname_max);
    }
  } else
    validPath = path;
  return validPath;
}

#else

static std::string getValidDumpPath(const std::string& path) {
  if (path.empty()) {
    return path;
  }

  const std::string curPath = path + ".";
  long pathname_max = pathconf(curPath.c_str(), _PC_PATH_MAX);
  assert(pathname_max != -1 && path.size() < static_cast<unsigned long>(pathname_max));
  return path;
}

#endif

static std::string getValidDumpBaseName(const std::string& path, const std::string& file,
                                        long basename_max, const std::string& ext) {
  if (file.size() + ext.size() < static_cast<unsigned long>(basename_max)) {
    return file;
  }

  // HashString returns unsigned, and the max lengh of unsighed is
  // 10 digits, anything over origDigits will be replaced with its hash value
  size_t origDigits = basename_max - 10 - ext.size();
  std::string truncName = file.substr(0, origDigits);
  unsigned hashVal = HashString(file.begin() + origDigits, file.end());

  return truncName + to_string(hashVal);
}

void Options::setDumpFileName(const char* val) {
  std::string dumpPrefix = oVariables->DumpPrefix;
  const size_t pidPos = dumpPrefix.find("%pid%");
  if (pidPos != std::string::npos) {
#ifdef _WIN32
    const std::int32_t pid = _getpid();
#endif
#ifdef __linux__
    const std::int32_t pid = getpid();
#endif
    dumpPrefix.replace(pidPos, 5, std::to_string(pid));
  }

  std::stringstream prefix;
  // Replace characters in the file name that are not legal with '_'.
  prefix << dumpPrefix << "_" << buildNo << "_" << std::regex_replace(val, std::regex(":"), "_");
  dumpFileRoot = prefix.str();

  // Check whether the length of path meets the system limits
  splitFileName(dumpFileRoot, fullPath, baseName);
  fullPath = getValidDumpPath(fullPath);

#ifdef _WIN32
  basename_max = FILENAME_MAX;
#endif
#ifdef __linux__
  const std::string curPath = fullPath + ".";
  basename_max = pathconf(curPath.c_str(), _PC_NAME_MAX);
  assert(basename_max != -1);
#endif

  if (!fullPath.empty()) {
    dumpFileRoot = fullPath + baseName;
  } else {
    dumpFileRoot = baseName;
  }
}

std::string Options::getDumpFileName(const std::string& ext) {
  // The length of file name meets the system limits
  if (baseName.size() + ext.size() < static_cast<unsigned long>(basename_max)) {
    return fullPath + baseName + ext;
  }

  std::size_t found = ext.find_last_of(".");
  std::string kernel_name, extension;
  if (found != std::string::npos) {
    kernel_name = ext.substr(0, found);
    extension = ext.substr(found);
  } else {
    extension = ext;
  }

  baseName = getValidDumpBaseName(fullPath, baseName + kernel_name, basename_max, extension);

  std::string fullName;
  if (!fullPath.empty()) {
    fullName = fullPath + baseName + extension;
  } else {
    fullName = baseName + extension;
  }
  return fullName;
}

bool Options::isCStrOptionsEqual(const char* cs1, const char* cs2) const {
  if ((cs1 != NULL) && (cs2 != NULL)) {
    return (strcmp(cs1, cs2) == 0);
  } else if ((cs1 == NULL || strcmp(cs1, "") == 0) && (cs2 == NULL || strcmp(cs2, "") == 0)) {
    // consider empty string and NULL ptr (no string) as equal
    return true;
  }
  return false;
}

bool Options::equals(const Options& other, bool ignoreClcOptions) const {
  OptionVariables* ovars = oVariables;
  OptionVariables* ovars2 = other.oVariables;
  OptionDescriptor* od = OptDescTable;
  for (int i = 0; i < OID_LAST; ++i, ++od) {
    if (!OPTIONHasOVariable(od)) {
      continue;
    }

    char* addr = reinterpret_cast<char*>(ovars) + od->OptionOffset;
    char* addr2 = reinterpret_cast<char*>(ovars2) + od->OptionOffset;
    if (OPTION_type(od) == OT_BOOL) {
      OT_BOOL_t* o = reinterpret_cast<OT_BOOL_t*>(addr);
      OT_BOOL_t* o2 = reinterpret_cast<OT_BOOL_t*>(addr2);
      if (*o != *o2) return false;
    } else if (OPTION_type(od) == OT_INT32) {
      OT_INT32_t* o = reinterpret_cast<OT_INT32_t*>(addr);
      OT_INT32_t* o2 = reinterpret_cast<OT_INT32_t*>(addr2);
      if (*o != *o2) return false;
    } else if (OPTION_type(od) == OT_UINT32) {
      OT_UINT32_t* o = reinterpret_cast<OT_UINT32_t*>(addr);
      OT_UINT32_t* o2 = reinterpret_cast<OT_UINT32_t*>(addr2);
      if (*o != *o2) return false;
    } else if (OPTION_type(od) == OT_CSTRING) {
      OT_CSTRING_t* o = reinterpret_cast<OT_CSTRING_t*>(addr);
      OT_CSTRING_t* o2 = reinterpret_cast<OT_CSTRING_t*>(addr2);
      if (!isCStrOptionsEqual(*o, *o2)) return false;
    } else if (OPTION_type(od) == OT_UCHAR) {
      OT_UCHAR_t* o = reinterpret_cast<OT_UCHAR_t*>(addr);
      OT_UCHAR_t* o2 = reinterpret_cast<OT_UCHAR_t*>(addr2);
      if (*o != *o2) return false;
    } else {
      return false;
    }
  }

  if (!ignoreClcOptions) {
    if (clcOptions.compare(other.clcOptions) != 0) {
      return false;
    }
  }

  if (llvmOptions.compare(other.llvmOptions) != 0) {
    return false;
  }

  for (size_t i = 0; i < 3; ++i) {
    if (WorkGroupSize[i] != other.WorkGroupSize[i]) {
      return false;
    }
  }

  if (NumAvailGPRs != other.NumAvailGPRs) {
    return false;
  }

  return true;
}

bool Options::setOptionVariablesAs(const Options& other) {
  OptionVariables* srcovars = other.oVariables;
  OptionVariables* dstovars = oVariables;
  OptionDescriptor* od = OptDescTable;
  for (int i = 0; i < OID_LAST; ++i, ++od) {
    if (!OPTIONHasOVariable(od)) {
      continue;
    }

    if (!other.isOptionSeen(i)) {
      continue;
    }

    if (!setOptionVariable(od, srcovars, dstovars)) {
      optionsLog() = "Wrong option value\n";
      return false;
    }
  }

  return true;
}

std::string Options::getStringFromStringVec(std::vector<std::string>& stringVec) {
  const char* const delim = " ";
  std::ostringstream strstr;
  std::copy(stringVec.begin(), stringVec.end(), std::ostream_iterator<std::string>(strstr, delim));
  return strstr.str();
}

}  // namespace option

}  // namespace amd
