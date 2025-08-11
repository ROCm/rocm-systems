////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2022, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef _ASSEMBLE_H_
#define _ASSEMBLE_H_

#include "OSWrapper.hpp"

#define ASM_MCPU_LEN 16

/* initialize LLVM targets and assembly printers/parsers */
void Init_LLVM();
/* shutdown LLVM */
void Shutdown_LLVM();

class Assembler {
  private:
      const char* ArchName = "amdgcn";
      const char* VendorName = "amd";
      const char* OSName = "amdhsa";
      char MCPU[ASM_MCPU_LEN];

      std::string TripleName;
      std::string Error;

      char* TextData;
      size_t TextSize;

      void SetTargetAsic(const uint32_t Gfxv);

      void FlushText();
      void PrintELFHex(const std::string Data);
      int ExtractELFText(const char* RawData);

  public:
      Assembler(const uint32_t Gfxv);
      ~Assembler();

      void PrintTextHex();
      const char* GetTargetAsic();

      const char* GetInstrStream();
      const size_t GetInstrStreamSize();
      int CopyInstrStream(char* OutBuf, const size_t BufSize = PAGE_SIZE);

      int RunAssemble(const char* const AssemblySource);
      int RunAssembleBuf(const char* const AssemblySource, char* OutBuf,
                         const size_t BufSize = PAGE_SIZE);
      int RunAssembleBuf(const char* const AssemblySource, char* OutBuf,
                         const size_t BufSize, const uint32_t Gfxv);
};

#endif  // _ASSEMBLE_H_
