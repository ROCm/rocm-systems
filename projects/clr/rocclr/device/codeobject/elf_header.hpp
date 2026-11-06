/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

//! Code object formats: ELF code objects read in place, without an ELF parser.
namespace amd::codeobject {

//! Bound for images whose length isn't known, e.g. hipModuleLoadData's pointer
constexpr size_t kUnknownSize = static_cast<size_t>(-1);

//! Name and type of the code object v3+ MessagePack metadata note (LLVM's NT_AMDGPU_METADATA;
//! not ELFIO's NT_AMDGPU_METADATA, which is 1)
constexpr char kAmdgpuNoteName[] = "AMDGPU";
constexpr uint32_t kAmdgpuMetadataNoteType = 32;

//! @returns True if @p image starts with the ELF magic. @p image must have 4 readable bytes.
bool IsElf(const void* image);

/**
 * @returns The extent of the 64-bit ELF image at @p image (the end of its headers, sections and
 *          segments), or 0 if validation fails. With @p bound == kUnknownSize only
 *          length-independent checks apply, so a malformed image on that path is undefined
 *          behavior (see hipModuleLoadData).
 */
uint64_t ElfImageSize(const void* image, size_t bound);

/**
 * ISA name (e.g. "amdgcn-amd-amdhsa--gfx90a:sramecc+:xnack-") of a code object v3 or later from
 * its ELF header fields alone, the way ROCr's loader names it (AmdHsaCode::GetIsa). False for
 * anything else: code object v2 (its ISA is in a note), 32-bit or non-AMDGPU ELFs, an OS ABI
 * other than AMDHSA, or a processor the loader doesn't know.
 */
bool GetIsaFromHeader(uint32_t elfClass, uint16_t machine, uint32_t osAbi, uint32_t abiVersion,
                      uint32_t eFlags, std::string& isaName, unsigned* genericVersion = nullptr);

/**
 * Calls @p fn(desc, size) for every note @p name of @p type in every SHT_NOTE section of the ELF
 * image at @p image, in file order, until it returns false; desc points into the image.
 *
 * @param size  Readable bytes at @p image; the whole image is validated against it first.
 * @returns False if the image or a note section is malformed.
 */
bool ForEachNote(const void* image, size_t size, const char* name, uint32_t type,
                 const std::function<bool(const char*, uint32_t)>& fn);

/**
 * Appends the names of the function (STT_FUNC) and variable (STT_OBJECT) symbols of the first
 * SHT_DYNSYM section, in table order: the lists comgr's iterate_symbols reported. Unnamed
 * symbols are left out; an image without .dynsym has none.
 *
 * @param size  Readable bytes at @p image; the whole image is validated against it first.
 * @returns False if the image or the symbol table is malformed.
 */
bool DynamicSymbolNames(const void* image, size_t size, std::vector<std::string>& funcs,
                        std::vector<std::string>& vars);

}  // namespace amd::codeobject
