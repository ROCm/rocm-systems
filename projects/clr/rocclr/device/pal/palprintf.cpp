/* Copyright (c) 2015 - 2021 Advanced Micro Devices, Inc.

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

#include "top.hpp"
#include "os/os.hpp"
#include "device/device.hpp"
#include "device/pal/paldefs.hpp"
#include "device/pal/palmemory.hpp"
#include "device/pal/palkernel.hpp"
#include "device/pal/palprogram.hpp"
#include "device/pal/palprintf.hpp"
#include <cstdio>
#include <algorithm>
#include <cmath>

// Functions defined in devhcprintf.cpp
namespace amd {
void handlePrintfDelayed(const uint64_t* input, uint64_t len, uint64_t control);
bool populateFormatStringHashMap(const std::vector<device::PrintfInfo>& printfInfo,
                                 std::map<uint64_t, std::string>& strMap);
}  // namespace amd

namespace amd::pal {

PrintfDbg::PrintfDbg(Device& device, FILE* file)
    : dbgBuffer_(nullptr),
      dbgFile_(file),
      gpuDevice_(device),
      wiDbgSize_(0),
      initCntValue_(device, 4) {}

PrintfDbg::~PrintfDbg() { delete dbgBuffer_; }

bool PrintfDbg::create() {
  // Create a resource for the init count value
  if (initCntValue_.create(Resource::Remote)) {
    uint32_t* value = reinterpret_cast<uint32_t*>(initCntValue_.map(nullptr));
    // The counter starts from 1
    if (nullptr != value) {
      *value = 1;
    } else {
      return false;
    }
    initCntValue_.unmap(nullptr);
    return true;
  }
  return false;
}

bool PrintfDbg::init(VirtualGPU& gpu, bool printfEnabled, const amd::NDRange& size) {
  // Set up debug output buffer (if printf active)
  if (printfEnabled) {
    if (!allocate()) {
      return false;
    }

    // Make sure that the size isn't bigger than the reported max
    if (size.product() <= dev().settings().maxWorkGroupSize_) {
      size_t wiDbgSizeTmp;

      // Calculate the debug buffer size per workitem
      wiDbgSizeTmp = std::min(dbgBuffer_->size() / size.product(), dev().xferRead().bufSize());

      // Make sure the size is DWORD aligned
      wiDbgSizeTmp = amd::alignDown(wiDbgSizeTmp, sizeof(uint32_t));

      // If the new size is different, then clear the initial values
      if (wiDbgSize_ != wiDbgSizeTmp) {
        wiDbgSize_ = wiDbgSizeTmp;
        if (!clearWorkitems(gpu, 0, size.product())) {
          wiDbgSize_ = 0;
          return false;
        }
      }
    }
  }

  return true;
}

bool PrintfDbg::output(VirtualGPU& gpu, bool printfEnabled, const amd::NDRange& size,
                       const std::vector<device::PrintfInfo>& printfInfo) {
  // Are we expected to generate debug output?
  if (printfEnabled && !printfInfo.empty()) {
    uint32_t* workitemData;
    size_t i, j, k, z;
    bool realloc = false;

    // Wait for kernel execution
    gpu.waitAllEngines();

    size_t zdim = 1;
    size_t ydim = 1;
    size_t xdim = 1;

    switch (size.dimensions()) {
      case 3:
        zdim = size[2];
      // Fall through ...
      case 2:
        ydim = size[1];
      // Fall through ...
      case 1:
        xdim = size[0];
      // Fall through ...
      default:
        break;
    }

    for (k = 0; k < zdim; ++k) {
      for (j = 0; j < ydim; ++j) {
        for (i = 0; i < xdim; ++i) {
          size_t idx = (xdim * (ydim * k + j) + i);
          workitemData = mapWorkitem(gpu, idx, &realloc);

          if (nullptr != workitemData) {
            uint32_t wp = workitemData[0];  // write pointer (i.e. first unwritten element)
            // Walk through each PrintfDbg entry
            for (z = 1; (z < (wiDbgSize() / sizeof(uint32_t))) && (z < wp);) {
              if (printfInfo.size() < workitemData[z]) {
                LogError("The format string wasn't reported");
                return false;
              }
              // Get the PrintfDbg info
              const device::PrintfInfo& info = printfInfo[workitemData[z++]];
              // There's something in this buffer
              outputDbgBuffer(info, workitemData, z);
            }
          }
          unmapWorkitem(gpu, workitemData);
        }
      }
    }

    // Reallocate debug buffer if necessary
    if (!allocate(realloc)) {
      return false;
    }
  }
  return true;
}

bool PrintfDbg::allocate(bool realloc) {
  if (nullptr == dbgBuffer_) {
    dbgBuffer_ = dev().createScratchBuffer(dev().info().printfBufferSize_);
  } else if (realloc) {
    LogWarning("Debug buffer reallocation!");
    // Double the buffer size if it's not big enough
    size_t size = dbgBuffer_->size();
    delete dbgBuffer_;
    dbgBuffer_ = dev().createScratchBuffer(size << 1);
  }

  return (nullptr != dbgBuffer_) ? true : false;
}

bool PrintfDbg::checkFloat(const std::string& fmt) const {
  switch (fmt[fmt.size() - 1]) {
    case 'e':
    case 'E':
    case 'f':
    case 'g':
    case 'G':
    case 'a':
      return true;
      break;
    default:
      break;
  }
  return false;
}

bool PrintfDbg::checkString(const std::string& fmt) const {
  if (fmt[fmt.size() - 1] == 's') return true;
  return false;
}

int PrintfDbg::checkVectorSpecifier(const std::string& fmt, size_t startPos, size_t& curPos) const {
  int vectorSize = 0;
  size_t pos = curPos;
  size_t size = curPos - startPos;

  if (size >= 3) {
    size = 0;
    // no modifiers
    if (fmt[curPos - 3] == 'v') {
      size = 2;
    }
    // the modifiers are "h" or "l"
    else if (fmt[curPos - 4] == 'v') {
      size = 3;
    }
    // the modifier is "hh"
    else if ((curPos >= 5) && (fmt[curPos - 5] == 'v')) {
      size = 4;
    }
    if (size > 0) {
      curPos = size;
      pos -= curPos;

      // Get vector size
      vectorSize = fmt[pos++] - '0';
      // PrintfDbg supports only 2, 3, 4, 8 and 16 wide vectors
      switch (vectorSize) {
        case 1:
          if ((fmt[pos++] - '0') == 6) {
            vectorSize = 16;
          } else {
            vectorSize = 0;
          }
          break;
        case 2:
        case 3:
        case 4:
        case 8:
          break;
        default:
          vectorSize = 0;
          break;
      }
    }
  }

  return vectorSize;
}

static constexpr size_t ConstStr = 0xffffffff;
static constexpr char Separator[] = ",\0";

size_t PrintfDbg::outputArgument(const std::string& fmt, bool printFloat, size_t size,
                                 const void* argument) const {
  // Serialize the output to the screen
  amd::ScopedLock k(dev().lockAsyncOps());

  size_t copiedBytes = size;
  // Print the string argument, using standard PrintfDbg()
  if (checkString(fmt.c_str())) {
    // copiedBytes should be as number of printed chars
    copiedBytes = 0;
    //(null) should be printed
    if (*(reinterpret_cast<const unsigned char*>(argument)) == 0) {
      amd::Os::printf(fmt.data(), 0);
      // copiedBytes = strlen("(null)")
      copiedBytes = 6;
    } else {
      const unsigned char* argumentStr = reinterpret_cast<const unsigned char*>(argument);
      amd::Os::printf(fmt.data(), argumentStr);
      // copiedBytes = strlen(argumentStr)
      while (argumentStr[copiedBytes++] != 0);
    }
  }

  // Print the argument(except for string ), using standard PrintfDbg()
  else {
    bool hlModifier = (strstr(fmt.c_str(), "hl") != nullptr);
    std::string hlFmt;
    if (hlModifier) {
      hlFmt = fmt;
      hlFmt.erase(hlFmt.find_first_of("hl"), 2);
    }
    switch (size) {
      case 0: {
        const char* str = reinterpret_cast<const char*>(argument);
        amd::Os::printf(fmt.data(), str);
        // Find the string length
        while (str[copiedBytes++] != 0);
      } break;
      case 1:
        amd::Os::printf(fmt.data(), *(reinterpret_cast<const unsigned char*>(argument)));
        break;
      case 2:
      case 4:
        if (printFloat) {
          const float fArg = size == 2
                                 ? amd::half2float(*(reinterpret_cast<const uint16_t*>(argument)))
                                 : *(reinterpret_cast<const float*>(argument));
          static const char* fSpecifiers = "eEfgGa";
          std::string fmtF = fmt;
          size_t posS = fmtF.find_first_of("%");
          size_t posE = fmtF.find_first_of(fSpecifiers);
          if (posS != std::string::npos && posE != std::string::npos) {
            fmtF.replace(posS + 1, posE - posS, "s");
          }
          float fSign = copysign(1.0, fArg);
          if (isinf(fArg) && !isnan(fArg)) {
            if (fSign < 0) {
              amd::Os::printf(fmtF.data(), "-infinity");
            } else {
              amd::Os::printf(fmtF.data(), "infinity");
            }
          } else if (isnan(fArg)) {
            if (fSign < 0) {
              amd::Os::printf(fmtF.data(), "-nan");
            } else {
              amd::Os::printf(fmtF.data(), "nan");
            }
          } else if (hlModifier) {
            amd::Os::printf(hlFmt.data(), fArg);
          } else {
            amd::Os::printf(fmt.data(), fArg);
          }
        } else {
          bool hhModifier = (strstr(fmt.c_str(), "hh") != nullptr);
          if (hhModifier) {
            // current implementation of printf in gcc 4.5.2 runtime libraries, doesn`t recognize
            // "hh" modifier ==>
            // argument should be explicitly converted to  unsigned char (uchar) before printing and
            // fmt should be updated not to contain "hh" modifier
            std::string hhFmt = fmt;
            hhFmt.erase(hhFmt.find_first_of("h"), 2);
            amd::Os::printf(hhFmt.data(), *(reinterpret_cast<const unsigned char*>(argument)));
          } else if (hlModifier) {
            amd::Os::printf(hlFmt.data(), size == 2
                                              ? *(reinterpret_cast<const uint16_t*>(argument))
                                              : *(reinterpret_cast<const uint32_t*>(argument)));
          } else {
            amd::Os::printf(fmt.data(), size == 2 ? *(reinterpret_cast<const uint16_t*>(argument))
                                                  : *(reinterpret_cast<const uint32_t*>(argument)));
          }
        }
        break;
      case 8:
        if (printFloat) {
          if (hlModifier) {
            amd::Os::printf(hlFmt.data(), *(reinterpret_cast<const double*>(argument)));
          } else {
            amd::Os::printf(fmt.data(), *(reinterpret_cast<const double*>(argument)));
          }
        } else {
          std::string out = fmt;
          // Use 'll' for 64 bit printf
          out.insert((out.size() - 1), 1, 'l');
          amd::Os::printf(out.data(), *(reinterpret_cast<const uint64_t*>(argument)));
        }
        break;
      case ConstStr: {
        const char* str = reinterpret_cast<const char*>(argument);
        amd::Os::printf(fmt.data(), str);
      } break;
      default:
        amd::Os::printf("Error: Unsupported data size for PrintfDbg. %d bytes",
                        static_cast<int>(size));
        return 0;
    }
  }
  fflush(stdout);
  return copiedBytes;
}

void PrintfDbg::outputDbgBuffer(const device::PrintfInfo& info, const uint32_t* workitemData,
                                size_t& i) const {
  static const char* specifiers = "cdieEfgGaosuxXp";
  static const char* modifiers = "hl";
  static const char* special = "%n";
  static const std::string sepStr = "%s";
  const uint32_t* s = workitemData;
  size_t pos = 0;

  // Find the format string
  std::string str = info.fmtString_;
  std::string fmt;
  size_t posStart, posEnd;

  // Print all arguments
  // Note: the following code walks through all arguments, provided by the kernel and
  // finds the corresponding specifier in the format string.
  // Then it splits the original string into substrings with a single specifier and
  // uses standard PrintfDbg() to print each argument
  for (uint j = 0; j < info.arguments_.size(); ++j) {
    do {
      posStart = str.find_first_of("%", pos);
      if (posStart != std::string::npos) {
        posStart++;
        // Erase all spaces after %
        while (str[posStart] == ' ') {
          str.erase(posStart, 1);
        }
        size_t tmp = str.find_first_of(special, posStart);
        size_t tmp2 = str.find_first_of(specifiers, posStart);
        // Special cases. Special symbol is located before any specifier
        if (tmp < tmp2) {
          posEnd = posStart + 1;
          fmt = str.substr(pos, posEnd - pos);
          fmt.erase(posStart - pos - 1, 1);
          pos = posStart = posEnd;
          outputArgument(sepStr, false, ConstStr, fmt.data());
          continue;
        }
        break;
      } else if (pos < str.length()) {
        outputArgument(sepStr, false, ConstStr, str.substr(pos).data());
      }
    } while (posStart != std::string::npos);

    if (posStart != std::string::npos) {
      bool printFloat = false;
      int vectorSize = 0;
      size_t idPos = 0;

      // Search for PrintfDbg specifier in the format string.
      // It will be a split point for the output
      posEnd = str.find_first_of(specifiers, posStart);
      if (posEnd == std::string::npos) {
        pos = posStart = posEnd;
        break;
      }
      posEnd++;

      size_t curPos = posEnd;
      vectorSize = checkVectorSpecifier(str, posStart, curPos);

      // Get substring from the last position to the current specifier
      fmt = str.substr(pos, posEnd - pos);

      // Readjust the string pointer if PrintfDbg outputs a vector
      if (vectorSize != 0) {
        size_t posVecSpec = fmt.length() - (curPos + 1);
        size_t posVecMod = fmt.find_first_of(modifiers, posVecSpec + 1);
        size_t posMod = str.find_first_of(modifiers, posStart);
        if (posMod < posEnd) {
          fmt = fmt.erase(posVecSpec, posVecMod - posVecSpec);
        } else {
          fmt = fmt.erase(posVecSpec, curPos);
        }
        idPos = posStart - pos - 1;
      }
      pos = posStart = posEnd;

      // Find out if the argument is a float
      printFloat = checkFloat(fmt);

      // Is it a scalar value?
      if (vectorSize == 0) {
        size_t length = outputArgument(fmt, printFloat, info.arguments_[j], &s[i]);
        if (0 == length) {
          return;
        }
        i += amd::alignUp(length, sizeof(uint32_t)) / sizeof(uint32_t);
      } else {
        // 3-component vector's size is defined as 4 * size of each scalar component
        size_t elemSize = info.arguments_[j] / (vectorSize == 3 ? 4 : vectorSize);
        size_t k = i * sizeof(uint32_t);
        std::string elementStr = fmt.substr(idPos, fmt.size());

        // Print first element with full string
        if (0 == outputArgument(fmt, printFloat, elemSize, &s[i])) {
          return;
        }

        // Print other elemnts with separator if available
        for (int e = 1; e < vectorSize; ++e) {
          const char* t = reinterpret_cast<const char*>(s);

          // Output the vector separator
          outputArgument(sepStr, false, ConstStr, Separator);

          // Output the next element
          outputArgument(elementStr, printFloat, elemSize, &t[k + e * elemSize]);
        }
        i += (amd::alignUp(info.arguments_[j], sizeof(uint32_t))) / sizeof(uint32_t);
      }
    }
  }

  if (pos != std::string::npos) {
    fmt = str.substr(pos, str.size() - pos);
    outputArgument(sepStr, false, ConstStr, fmt.data());
  }
}

bool PrintfDbg::clearWorkitems(VirtualGPU& gpu, size_t idxStart, size_t number) const {
  // Go through all locations for every thread and copy 1
  for (uint i = idxStart; i < idxStart + number; ++i) {
    amd::Coord3D dst(i * wiDbgSize(), 0, 0);
    amd::Coord3D size(sizeof(uint32_t), 0, 0);

    // Copy 1 into the corresponding location in the debug buffer
    if (!initCntValue_.partialMemCopyTo(gpu, amd::Coord3D(0, 0, 0), dst, size, *dbgBuffer_)) {
      return false;
    }
  }
  return true;
}

uint32_t* PrintfDbg::mapWorkitem(VirtualGPU& gpu, size_t idx, bool* realloc) {
  uint32_t wiSize = 0;
  amd::Coord3D src(idx * wiDbgSize(), 0, 0);
  xferBufRead_ = &(dev().xferRead().acquire());

  // Copy workitem size from the corresponding location in the debug buffer
  if (!dbgBuffer_->partialMemCopyTo(gpu, src, amd::Coord3D(0, 0, 0),
                                    amd::Coord3D(sizeof(uint32_t), 0, 0), *xferBufRead_)) {
    return nullptr;
  }

  // Get memory pointer to the satged buffer
  uint32_t* workitem = reinterpret_cast<uint32_t*>(xferBufRead_->map(&gpu));
  if (nullptr == workitem) {
    return nullptr;
  }

  // Copy size value
  wiSize = *workitem;
  xferBufRead_->unmap(&gpu);

  // Check if the cuurent workitem almost reached the size limit
  if ((wiDbgSize() - static_cast<size_t>(wiSize)) < 3) {
    *realloc = true;
  }

  // If the current workitem had any output then get the data
  if ((wiSize > 1) && (wiSize <= wiDbgSize())) {
    amd::Coord3D size(wiSize * sizeof(uint32_t), 0, 0);

    // Copy the current workitem output data to the staged buffer
    if (!dbgBuffer_->partialMemCopyTo(gpu, src, amd::Coord3D(0, 0, 0), size, *xferBufRead_) ||
        // Clear the write pointer back to index 1 for the current workitem
        !clearWorkitems(gpu, idx, 1)) {
      LogError("Reading the workitem data failed!");
      return nullptr;
    }

    // Get a pointer to the workitem data
    workitem = reinterpret_cast<uint32_t*>(xferBufRead_->map(&gpu));

    return workitem;
  }

  return nullptr;
}

void PrintfDbg::unmapWorkitem(VirtualGPU& gpu, const uint32_t* workitemData) const {
  if (nullptr != workitemData) {
    xferBufRead_->unmap(&gpu);
  }

  dev().xferRead().release(gpu, *xferBufRead_);
}

bool PrintfDbgHSA::init(VirtualGPU& gpu, bool printfEnabled) {
  // Set up debug output buffer (if printf active)
  if (printfEnabled) {
    if (!allocate()) {
      return false;
    }

    // The first two DWORDs in the printf buffer are as follows:
    // First DWORD = Offset to where next information is to
    // be written, initialized to 0
    // Second DWORD = Number of bytes available for printf data
    // = buffer size � 2*sizeof(uint32_t)
    const uint8_t initSize = 2 * sizeof(uint32_t);
    uint8_t sysMem[initSize];
    memset(sysMem, 0, initSize);
    uint32_t dbgBufferSize = dbgBuffer_->size() - initSize;
    memcpy(&sysMem[4], &dbgBufferSize, sizeof(dbgBufferSize));

    // Copy offset and number of bytes available for printf data
    // into the corresponding location in the debug buffer
    dbgBuffer_->writeRawData(gpu, 0, initSize, sysMem, true);
  }
  return true;
}

bool PrintfDbgHSA::output(VirtualGPU& gpu, bool printfEnabled,
                          const std::vector<device::PrintfInfo>& printfInfo) {
  if (printfEnabled) {
    uint32_t offsetSize = 0;
    xferBufRead_ = &(dev().xferRead().acquire());

    // Copy offset from the first DWORD in the debug buffer
    if (!dbgBuffer_->partialMemCopyTo(gpu, amd::Coord3D(0, 0, 0), amd::Coord3D(0, 0, 0),
                                      amd::Coord3D(sizeof(uint32_t), 0, 0), *xferBufRead_)) {
      return false;
    }

    // Get memory pointer to the satged buffer
    uint32_t* dbgBufferPtr = reinterpret_cast<uint32_t*>(xferBufRead_->map(&gpu));
    if (nullptr == dbgBufferPtr) {
      return false;
    }

    offsetSize = *dbgBufferPtr;
    xferBufRead_->unmap(&gpu);

    if (offsetSize == 0) {
      LogInfo("The printf buffer is empty!");
      dev().xferRead().release(gpu, *xferBufRead_);
      return true;
    }

    size_t bufSize = dev().xferRead().bufSize();
    size_t copySize = offsetSize;

    // Map between 64 bit MD5 format string hash and
    // actual format string
    std::map<uint64_t, std::string> StrMap;

    while (copySize != 0) {
      // Copy the buffer data (i.e., the printfID followed by the
      // argument data for each printf call in th kernel) to the staged buffer
      if (!dbgBuffer_->partialMemCopyTo(
              gpu, amd::Coord3D(2 * sizeof(uint32_t) + offsetSize - copySize, 0, 0),
              amd::Coord3D(0, 0, 0), std::min(copySize, bufSize), *xferBufRead_)) {
        return false;
      }

      // Get a pointer to the buffer data
      dbgBufferPtr = reinterpret_cast<uint32_t*>(xferBufRead_->map(&gpu));
      if (nullptr == dbgBufferPtr) {
        return false;
      }

      uint sb = 0;
      uint sbt = 0;

      // Handle HIP nonhostcall printf here,
      if (amd::IS_HIP) {
        auto BufferForHIP = reinterpret_cast<uint32_t*>(dbgBufferPtr);

        // Populate string map with hashes and actual
        // format strings.
        if (!amd::populateFormatStringHashMap(printfInfo, StrMap)) return false;

        while (sbt < copySize) {
          auto controlDword = *BufferForHIP++;
          uint64_t nextOffset = controlDword >> 2;

          if (sbt + nextOffset > bufSize) {
            break;  // Need new portion of data in staging buffer
          }

          auto PB = (uint64_t*)BufferForHIP;
          std::vector<uint8_t> PBuffer;
          uint64_t BufferLen = 0;
          if (controlDword & 2U) {
            // Process the contsant format string case.
            // The first value is the 64 bit format string hash
            // and remaining values are printf arguments.
            // Construct a temporary buffer with actual format
            // string followed by arguments. The format string is
            // obtained by querying StrMap populated before.
            auto ArgsLen = nextOffset - 12;
            auto Str = StrMap[*PB++];
            auto StrLenWithNull = Str.size() + 1;
            BufferLen = ArgsLen + amd::alignUp(StrLenWithNull, sizeof(uint64_t));
            PBuffer.resize(BufferLen);
            memcpy(PBuffer.data(), Str.c_str(), StrLenWithNull);
            memset(PBuffer.data() + Str.size(), 0, 8 - (StrLenWithNull % 8));
            memcpy(PBuffer.data() + amd::alignUp(StrLenWithNull, sizeof(uint64_t)), PB, ArgsLen);
          } else {
            // Process Non constant format string case.
            // Here, The buffer itself contains the actual
            // format string and hence just copy the contents
            // of format string and arguments into a temporary
            // buffer
            BufferLen = nextOffset - /*ControlDWord*/ 4;
            PBuffer.resize(BufferLen);
            memcpy(PBuffer.data(), BufferForHIP, nextOffset);
          }

          // Handle printing
          amd::handlePrintfDelayed((uint64_t*)PBuffer.data(), BufferLen / 8, controlDword);
          BufferForHIP += (nextOffset / 4) - /*ControlDWord*/ 1;
          sbt += nextOffset;
        }

        copySize -= sbt;
        xferBufRead_->unmap(&gpu);
        continue;
      }

      // parse the debug buffer
      while (sbt < copySize) {
        if (*dbgBufferPtr >= printfInfo.size()) {
          LogError("Couldn't find the reported PrintfID!");
          return false;
        }
        const device::PrintfInfo& info = printfInfo[(*dbgBufferPtr)];
        sb += sizeof(uint32_t);
        for (const auto& it : info.arguments_) {
          sb += it;
        }

        if (sbt + sb > bufSize) {
          break;  // Need new portion of data in staging buffer
        }

        size_t idx = 1;
        // There's something in the debug buffer
        outputDbgBuffer(info, dbgBufferPtr, idx);

        sbt += sb;
        dbgBufferPtr += sb / sizeof(uint32_t);
        sb = 0;
      }

      copySize -= sbt;
      xferBufRead_->unmap(&gpu);
    }

    dev().xferRead().release(gpu, *xferBufRead_);
  }

  return true;
}

}  // namespace amd::pal
