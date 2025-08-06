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

/**
 * Self-contained assembler that uses the LLVM MC API to assemble AMDGCN
 * instructions
 */

#include <llvm/Config/llvm-config.h>
#include <llvm/MC/MCAsmBackend.h>
#include <llvm/MC/MCAsmInfo.h>
#include <llvm/MC/MCCodeEmitter.h>
#include <llvm/MC/MCContext.h>
#include <llvm/MC/MCInstPrinter.h>
#include <llvm/MC/MCInstrInfo.h>
#include <llvm/MC/MCObjectFileInfo.h>
#include <llvm/MC/MCObjectWriter.h>
#include <llvm/MC/MCParser/AsmLexer.h>
#include <llvm/MC/MCParser/MCTargetAsmParser.h>
#include <llvm/MC/MCRegisterInfo.h>
#include <llvm/MC/MCStreamer.h>
#include <llvm/MC/MCSubtargetInfo.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#if LLVM_VERSION_MAJOR > 13
#include <llvm/MC/TargetRegistry.h>
#else
#include <llvm/Support/TargetRegistry.h>
#endif
#if LLVM_VERSION_MAJOR > 18
#include "llvm/Support/ManagedStatic.h"
#endif

#include <linux/elf.h>
#include "OSWrapper.hpp"
#include "Assemble.hpp"

using namespace llvm;

/* Assembler implementation is not multi-thread safe and is
 * asic type dependent. Instantiate it per thread/gpu use case,
 * delete each assembler after assembling
 */

void Init_LLVM() {
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
}

void Shutdown_LLVM() {
    llvm_shutdown();
}

Assembler::Assembler(const uint32_t Gfxv) {
    SetTargetAsic(Gfxv);
    TextData = nullptr;
    TextSize = 0;
}

Assembler::~Assembler() {
    FlushText();
}

const char* Assembler::GetInstrStream() {
    return TextData;
}

const size_t Assembler::GetInstrStreamSize() {
    return TextSize;
}

int Assembler::CopyInstrStream(char* OutBuf, const size_t BufSize) {
    if (TextSize > BufSize)
        return -2;

    std::copy(TextData, TextData + TextSize, OutBuf);
    return 0;
}

const char* Assembler::GetTargetAsic() {
    return MCPU;
}

/**
 * Set MCPU via GFX Version from Thunk
 * LLVM Target IDs use decimal for Maj/Min, hex for Step
 */
void Assembler::SetTargetAsic(const uint32_t Gfxv) {
    const uint8_t Major = (Gfxv >> 16) & 0xff;
    const uint8_t Minor = (Gfxv >> 8) & 0xff;
    const uint8_t Step = Gfxv & 0xff;

    snprintf(MCPU, ASM_MCPU_LEN, "gfx%d%d%x", Major, Minor, Step);
}

/**
 * Flush/reset TextData and TextSize to initial state
 */
void Assembler::FlushText() {
    if (TextData)
        delete[] TextData;
    TextData = nullptr;
    TextSize = 0;
}

/**
 * Print hex of ELF object to stdout (debug)
 */
void Assembler::PrintELFHex(const std::string Data) {
    outs() << "ASM Info: assembled ELF hex data (length " << Data.length() << "):\n";
    outs() << "0x00:\t";
    for (size_t i = 0; i < Data.length(); ++i) {
        char c = Data[i];
        outs() << format_hex(static_cast<uint8_t>(c), 4);
        if ((i+1) % 16 == 0)
            outs() << "\n" << format_hex(i+1, 4) << ":\t";
        else
            outs() << " ";
    }
    outs() << "\n";
}

/**
 * Print hex of raw instruction stream to stdout (debug)
 */
void Assembler::PrintTextHex() {
    outs() << "ASM Info: assembled .text hex data (length " << TextSize << "):\n";
    outs() << "0x00:\t";
    for (size_t i = 0; i < TextSize; i++) {
        outs() << format_hex(static_cast<uint8_t>(TextData[i]), 4);
        if ((i+1) % 16 == 0)
            outs() << "\n" << format_hex(i+1, 4) << ":\t";
        else
            outs() << " ";
    }
    outs() << "\n";
}

/**
 * Extract raw instruction stream from .text section in ELF object
 *
 * @param RawData Raw C string of ELF object
 * @return 0 on success
 */
int Assembler::ExtractELFText(const char* RawData) {
    const Elf64_Ehdr* ElfHeader;
    const Elf64_Shdr* SectHeader;
    const Elf64_Shdr* SectStrTable;
    const char* SectStrAddr;
    unsigned NumSects, SectIdx;

    if (!(ElfHeader = reinterpret_cast<const Elf64_Ehdr*>(RawData))) {
        outs() << "ASM Error: elf data is invalid or corrupted\n";
        return -1;
    }
    if (ElfHeader->e_ident[EI_CLASS] != ELFCLASS64) {
        outs() << "ASM Error: elf object must be of 64-bit type\n";
        return -1;
    }

    SectHeader = reinterpret_cast<const Elf64_Shdr*>(RawData + ElfHeader->e_shoff);
    SectStrTable = &SectHeader[ElfHeader->e_shstrndx];
    SectStrAddr = static_cast<const char*>(RawData + SectStrTable->sh_offset);

    // Loop through sections, break on .text
    NumSects = ElfHeader->e_shnum;
    for (SectIdx = 0; SectIdx < NumSects; SectIdx++) {
        std::string SectName = std::string(SectStrAddr + SectHeader[SectIdx].sh_name);
        if (SectName == std::string(".text")) {
            TextSize = SectHeader[SectIdx].sh_size;
            TextData = new char[TextSize];
            memcpy(TextData, RawData + SectHeader[SectIdx].sh_offset, TextSize);
            break;
        }
    }

    if (SectIdx >= NumSects) {
        outs() << "ASM Error: couldn't locate .text section\n";
        return -1;
    }

    return 0;
}

/**
 * Assemble shader, fill member vars, and copy to output buffer
 *
 * @param AssemblySource Shader source represented as a raw C string
 * @param OutBuf Raw instruction stream output buffer
 * @param BufSize Size of OutBuf (defaults to PAGE_SIZE)
 * @param Gfxv Optional overload to temporarily set target ASIC
 * @return Value of RunAssemble() (0 on success)
 */
int Assembler::RunAssembleBuf(const char* const AssemblySource, char* OutBuf,
                              const size_t BufSize) {
    int ret = RunAssemble(AssemblySource);
    return ret ? ret : CopyInstrStream(OutBuf, BufSize);
}
int Assembler::RunAssembleBuf(const char* const AssemblySource, char* OutBuf,
                              const size_t BufSize, const uint32_t Gfxv) {
    const char* defaultMCPU = GetTargetAsic();
    SetTargetAsic(Gfxv);
    int ret = RunAssemble(AssemblySource);
    strncpy(MCPU, defaultMCPU, ASM_MCPU_LEN);
    return ret ? ret : CopyInstrStream(OutBuf, BufSize);
}

/**
 * Assemble shader and fill member vars
 *
 * @param AssemblySource Shader source represented as a raw C string
 * @return 0 on success
 */
int Assembler::RunAssemble(const char* const AssemblySource) {
    // Ensure target ASIC has been set
    if (!*MCPU) {
        outs() << "ASM Error: target asic is uninitialized\n";
        return -1;
    }

    // Delete TextData for any previous runs
    FlushText();

#if 0
    outs() << "ASM Info: running assembly for target: " << MCPU << "\n";
    outs() << "ASM Info: source:\n";
    outs() << AssemblySource << "\n";
#endif

    // Initialize MCOptions and target triple
    const MCTargetOptions MCOptions;
    Triple TheTriple;

    const Target* TheTarget =
        TargetRegistry::lookupTarget(ArchName, TheTriple, Error);
    if (!TheTarget) {
        outs() << Error;
        return -1;
    }

    TheTriple.setArchName(ArchName);
    TheTriple.setVendorName(VendorName);
    TheTriple.setOSName(OSName);

    TripleName = TheTriple.getTriple();
    TheTriple.setTriple(Triple::normalize(TripleName));

    // Create MemoryBuffer for assembly source
    StringRef AssemblyRef(AssemblySource);
    std::unique_ptr<MemoryBuffer> BufferPtr =
        MemoryBuffer::getMemBuffer(AssemblyRef, "", false);
    if (!BufferPtr->getBufferSize()) {
        outs() << "ASM Error: assembly source is empty\n";
        return -1;
    }

    // Instantiate SrcMgr and transfer BufferPtr ownership
    SourceMgr SrcMgr;
    SrcMgr.AddNewSourceBuffer(std::move(BufferPtr), SMLoc());

    // Initialize MC interfaces and base class objects
    std::unique_ptr<const MCRegisterInfo> MRI(
            TheTarget->createMCRegInfo(TripleName));
    if (!MRI) {
        outs() << "ASM Error: no register info for target " << MCPU << "\n";
        return -1;
    }
#if LLVM_VERSION_MAJOR > 9
    std::unique_ptr<const MCAsmInfo> MAI(
            TheTarget->createMCAsmInfo(*MRI, TripleName, MCOptions));
#else
    std::unique_ptr<const MCAsmInfo> MAI(
            TheTarget->createMCAsmInfo(*MRI, TripleName));
#endif
    if (!MAI) {
        outs() << "ASM Error: no assembly info for target " << MCPU << "\n";
        return -1;
    }
    std::unique_ptr<MCInstrInfo> MCII(
            TheTarget->createMCInstrInfo());
    if (!MCII) {
        outs() << "ASM Error: no instruction info for target " << MCPU << "\n";
        return -1;
    }
    std::unique_ptr<MCSubtargetInfo> STI(
            TheTarget->createMCSubtargetInfo(TripleName, MCPU, std::string()));
    if (!STI || !STI->isCPUStringValid(MCPU)) {
        outs() << "ASM Error: no subtarget info for target " << MCPU << "\n";
        return -1;
    }

    // Set up the MCContext for creating symbols and MCExpr's
#if LLVM_VERSION_MAJOR > 12
    MCContext Ctx(TheTriple, MAI.get(), MRI.get(), STI.get(), &SrcMgr, &MCOptions);
#else
    MCObjectFileInfo MOFI;
    MCContext Ctx(MAI.get(), MRI.get(), &MOFI, &SrcMgr, &MCOptions);
    MOFI.InitMCObjectFileInfo(TheTriple, true, Ctx);
#endif

    // Finalize setup for output object code stream
    std::string Data;
    std::unique_ptr<raw_string_ostream> DataStream(std::make_unique<raw_string_ostream>(Data));
    std::unique_ptr<buffer_ostream> BOS(std::make_unique<buffer_ostream>(*DataStream));
    raw_pwrite_stream* OS = BOS.get();

#if LLVM_VERSION_MAJOR > 14
    MCCodeEmitter* CE = TheTarget->createMCCodeEmitter(*MCII, Ctx);
#else
    MCCodeEmitter* CE = TheTarget->createMCCodeEmitter(*MCII, *MRI, Ctx);
#endif
    MCAsmBackend* MAB = TheTarget->createMCAsmBackend(*STI, *MRI, MCOptions);

    if (!MAB) {
	    outs() << "ASM Error: Unable to create MCA Backend\n";
	    return -1;
    }

#if LLVM_VERSION_MAJOR > 20
    std::unique_ptr<MCStreamer> Streamer(TheTarget->createMCObjectStreamer(
        TheTriple, Ctx,
	std::unique_ptr<MCAsmBackend>(MAB), MAB->createObjectWriter(*OS),
        std::unique_ptr<MCCodeEmitter>(CE), *STI));
#else
    std::unique_ptr<MCStreamer> Streamer(TheTarget->createMCObjectStreamer(
        TheTriple, Ctx,
        std::unique_ptr<MCAsmBackend>(MAB), MAB->createObjectWriter(*OS),
        std::unique_ptr<MCCodeEmitter>(CE), *STI, MCOptions.MCRelaxAll,
        MCOptions.MCIncrementalLinkerCompatible, /*DWARFMustBeAtTheEnd*/ false));
#endif

    std::unique_ptr<MCAsmParser> Parser(
            createMCAsmParser(SrcMgr, Ctx, *Streamer, *MAI));

    // Set parser to target parser and run
    std::unique_ptr<MCTargetAsmParser> TAP(
            TheTarget->createMCAsmParser(*STI, *Parser, *MCII, MCOptions));
    if (!TAP) {
        outs() << "ASM Error: no assembly parsing support for target " << MCPU << "\n";
        return -1;
    }
    Parser->setTargetParser(*TAP);

    if (Parser->Run(true)) {
        outs() << "ASM Error: assembly parser failed\n";
        return -1;
    }

    BOS.reset();
    DataStream->flush();

    int ret = ExtractELFText(Data.data());
    if (ret < 0 || !TextData) {
        outs() << "ASM Error: .text extraction failed\n";
        return ret;
    }

#if 0
    PrintELFHex(Data);
    PrintTextHex();
#endif

    return 0;
}
