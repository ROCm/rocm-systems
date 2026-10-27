/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

//! Code object formats: clang offload bundles. No HIP or OpenCL knowledge.
namespace amd::codeobject {

//! Code object found in a bundle for one target query
struct CodeObjectRef {
  uint64_t offset = 0;  //!< Offset of the code object from the start of the bundle
  uint64_t size = 0;    //!< Size of the code object; 0 if the bundle has none for the target
};

//! @returns True if @p image starts with the uncompressed offload bundle magic.
bool IsOffloadBundle(const void* image, size_t imageBound);

/**
 * @returns The extent of the uncompressed bundle at @p bundle: the end of its header or of its
 *          last code object, whichever is further. Only the first 4 KiB of the header (and at
 *          most @p headerBound bytes) are read. 0 if the header is malformed.
 */
uint64_t OffloadBundleSize(const void* bundle, size_t headerBound);

/**
 * Finds the code object for each target in an uncompressed clang offload bundle, with the rules
 * of comgr's amd_comgr_lookup_code_object: bundle entries are walked in file order, only the hip,
 * hipv4 and hcc offload kinds are used, and each entry fills the first query that is still empty
 * and compatible with it (see IsCompatibleTarget). Only the first of concatenated bundles is read.
 * Unlike comgr, an entry holding an AMDGPU ELF is matched by the ISA its ELF header names (as
 * the loader does), not by the target spelled in its id, so LLVM's amdgpuN.M triples need no
 * parsing; entries that aren't ELF (SPIR-V) are matched by their id.
 *
 * @param bundle       Start of the bundle.
 * @param headerBound  Readable bytes for the header; reading past it fails the lookup.
 * @param imageBound   Readable bytes of the whole bundle, or kUnknownSize; bounds the ELF header
 *                     reads of the entries. With kUnknownSize (pointer loads) the entry offsets
 *                     in the header are trusted and every hip entry's ELF header is read, so a
 *                     corrupt offset in any entry can fault, not only in the selected one.
 * @param targets      Target queries, e.g. "amdgcn-amd-amdhsa--gfx1100".
 * @param found        Output, one entry per query in the same order.
 * @returns False if the header is malformed or exceeds @p headerBound. Offsets and sizes are
 *          not checked against the image; the caller does that.
 */
bool LookupCodeObjects(const void* bundle, size_t headerBound, size_t imageBound,
                       const std::vector<std::string>& targets,
                       std::vector<CodeObjectRef>& found);

/**
 * @returns True if a code object built for @p codeObjectTarget runs on @p target, with the rules
 *          of comgr's isCompatibleIsaName: identical names match; otherwise both must parse,
 *          name the same processor, and a code object that sets xnack or sramecc (+/-) needs the
 *          same setting in @p target. A code object that leaves a feature unset matches any.
 */
bool IsCompatibleTarget(const std::string& target, const std::string& codeObjectTarget);

}  // namespace amd::codeobject
