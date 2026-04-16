//===-- aegisbit/Types.h - Core AegisBit Types ------------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Core type definitions for AegisBit instrumentation framework.
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_TYPES_H
#define AEGISBIT_TYPES_H

#include "aegisbit/RegisterHelper.h"
#include "llvm/MC/MCInst.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace aegisbit {

/// Instruction category for classification
enum class InstructionCategory {
  VALU,    ///< Vector ALU operation
  SALU,    ///< Scalar ALU operation
  VMEM,    ///< Vector memory operation
  SMEM,    ///< Scalar memory operation
  LDS,     ///< Local data share operation
  BRANCH,  ///< Control flow (branch/jump)
  MFMA,    ///< Matrix fused multiply-add
  BARRIER, ///< Synchronization barrier
  OTHER    ///< Other instruction types
};

/// Decoded instruction with metadata
struct DecodedInstruction {
  llvm::MCInst Inst;           ///< LLVM MC instruction object
  uint64_t Address;            ///< PC-relative address
  uint64_t Size;               ///< Instruction size in bytes (4 or 8)
  InstructionCategory Category; ///< Instruction classification

  /// Check if this is a branch instruction
  bool isBranch() const { return Category == InstructionCategory::BRANCH; }

  /// Check if this is a memory instruction
  bool isMemory() const {
    return Category == InstructionCategory::VMEM ||
           Category == InstructionCategory::SMEM ||
           Category == InstructionCategory::LDS;
  }
};

/// Basic block in the control flow graph
struct BasicBlock {
  uint32_t ID;                              ///< Unique basic block identifier
  uint64_t StartAddress;                    ///< Starting PC address
  uint64_t EndAddress;                      ///< Ending PC address (exclusive)
  std::vector<DecodedInstruction> Instructions; ///< Instructions in this BB
  std::vector<uint32_t> Successors;         ///< Successor BB IDs
  std::vector<uint32_t> Predecessors;       ///< Predecessor BB IDs
  bool IsTerminal;                          ///< True if ends with s_endpgm
};

/// Information about a branch that couldn't be resolved
struct UnresolvedBranch {
  uint64_t InstructionAddress;  ///< Address of the branch instruction
  uint32_t BlockID;             ///< Block containing the branch
  std::string ErrorMessage;     ///< Why resolution failed
};

/// Control flow graph for a kernel
struct ControlFlowGraph {
  std::vector<BasicBlock> BasicBlocks;         ///< All basic blocks
  uint32_t EntryBlockID;                       ///< Entry basic block ID
  std::vector<UnresolvedBranch> UnresolvedBranches; ///< Branches with resolution errors

  /// Build index for O(1) block lookup. Must be called after BasicBlocks is populated.
  /// This is called automatically by CFGBuilder::build().
  void buildBlockIndex() {
    BlockIndex.clear();
    for (size_t i = 0; i < BasicBlocks.size(); ++i) {
      BlockIndex[BasicBlocks[i].ID] = static_cast<uint32_t>(i);
    }
  }

  /// Get basic block by ID - O(1) if buildBlockIndex() was called
  const BasicBlock* getBlock(uint32_t ID) const {
    auto It = BlockIndex.find(ID);
    if (It != BlockIndex.end() && It->second < BasicBlocks.size()) {
      return &BasicBlocks[It->second];
    }
    return nullptr;
  }

  /// Get mutable basic block by ID - O(1) if buildBlockIndex() was called
  BasicBlock* getBlock(uint32_t ID) {
    auto It = BlockIndex.find(ID);
    if (It != BlockIndex.end() && It->second < BasicBlocks.size()) {
      return &BasicBlocks[It->second];
    }
    return nullptr;
  }

private:
  std::unordered_map<uint32_t, uint32_t> BlockIndex; ///< ID -> vector index
};

/// AMD GPU kernel descriptor (64 bytes in .rodata section)
struct KernelDescriptor {
  uint32_t GroupSegmentFixedSize;     ///< LDS size in bytes
  uint32_t PrivateSegmentFixedSize;   ///< Scratch memory size per work-item
  uint32_t KernargSize;               ///< Kernel argument size
  uint64_t KernelCodeEntryByteOffset; ///< Offset to entry point in .text
  uint32_t ComputePgmRsrc1;           ///< VGPR/SGPR counts and other flags
  uint32_t ComputePgmRsrc2;           ///< Scratch, user SGPR counts
  uint32_t ComputePgmRsrc3;           ///< Additional resource flags

  // Extracted fields
  uint32_t VGPRCount;                 ///< Total allocated VGPRs (regular + AccVGPR)
  uint32_t SGPRCount;                 ///< Number of allocated SGPRs
  uint32_t WavefrontSize;             ///< 32 or 64
  uint32_t VGPRGranularity;           ///< VGPR allocation granularity (4 or 8)

  /// First AccVGPR index in the unified register file (gfx90a/gfx942).
  /// Regular VGPRs are v0..v(AccumOffset-1). 0 means no AccVGPRs.
  uint32_t AccumOffset = 0;

  /// Number of hardware-implicit SGPRs at the top of the allocation:
  ///   VCC(2) + FLAT_SCRATCH(2) = 4  for gfx9 (non-CDNA3+)
  ///   VCC(2) + FLAT_SCRATCH(2) + XNACK_MASK(2) = 6  for gfx940+/CDNA3+
  /// Set by DescriptorUpdater::parse() based on the target architecture.
  uint32_t ImplicitSGPRs = 4;

  // Additional descriptor fields (must be preserved through patching)
  uint16_t KernelCodeProperties;      ///< KERNEL_CODE_PROPERTY_* flags (offset 56)
  uint16_t KernargPreload;            ///< Kernarg preload length (offset 58)

  // Note: parse() and serialize() will be implemented when ELF handling is added
};

/// Code object (ELF file containing GPU kernel code)
struct CodeObject {
  std::vector<uint8_t> TextSection;    ///< .text section (machine code)
  std::vector<uint8_t> RodataSection;  ///< .rodata section (kernel descriptors)
  std::vector<uint8_t> NoteSection;    ///< .note section (metadata)
  std::string KernelName;              ///< Kernel symbol name
  KernelDescriptor Descriptor;         ///< Parsed kernel descriptor

  // Note: loadFromMemory() and serialize() will be implemented when ELF handling is added
};

/// Trace buffer arguments (appended to kernel args)
/// Must be packed to match GPU-side instrumentation layout (offsets 0, 8, 16, 24)
struct __attribute__((packed)) TraceArgs {
  uint64_t BufferPtr;      ///< Pointer to trace buffer in device memory
  uint64_t BufferSize;     ///< Size of trace buffer in bytes
  uint64_t WriteOffsetPtr; ///< Pointer to atomic write offset
  uint32_t KernelID;       ///< Unique kernel identifier for this trace
};
static_assert(sizeof(TraceArgs) == 28, "TraceArgs must be 28 bytes to match GPU-side layout");

/// Symbol table entry for ELF code objects
struct SymbolEntry {
  std::string Name;        ///< Symbol name
  uint64_t Value;          ///< Symbol value (typically offset)
  uint64_t Size;           ///< Symbol size in bytes
  uint8_t Type;            ///< Symbol type (STT_FUNC, STT_OBJECT, etc.)
  uint8_t Binding;         ///< Symbol binding (STB_LOCAL, STB_GLOBAL, etc.)
  uint16_t SectionIndex;   ///< Section index this symbol belongs to
};

/// Information about a kernel within a code object
struct KernelInfo {
  std::string Name;            ///< Kernel name (from symbol table)
  uint64_t CodeOffset;         ///< Offset of code in .text section
  uint64_t CodeSize;           ///< Size of kernel code
  uint64_t DescriptorOffset;   ///< Offset of .kd symbol in .rodata
  KernelDescriptor Descriptor; ///< Parsed kernel descriptor
};

/// Parsed AMD GPU code object (ELF)
struct ParsedCodeObject {
  uint16_t EType;                       ///< ELF type (ET_REL, ET_DYN, etc.)
  uint16_t EMachine;                    ///< ELF machine type (should be 0xE0)
  uint32_t EFlags;                      ///< ELF flags (contains EF_AMDGPU_MACH_*)
  std::string GPUArch;                  ///< GPU architecture string (e.g., "gfx942")

  std::vector<uint8_t> TextSection;     ///< .text section (machine code)
  std::vector<uint8_t> RodataSection;   ///< .rodata section (kernel descriptors)
  std::vector<uint8_t> NoteSection;     ///< .note section (AMDGPU metadata)

  std::vector<KernelInfo> Kernels;      ///< All kernels in this code object
  std::vector<SymbolEntry> Symbols;     ///< Symbol table entries

  /// Section indices for reference
  uint16_t TextSectionIndex = 0;
  uint16_t RodataSectionIndex = 0;
  uint16_t NoteSectionIndex = 0;

  /// Section virtual addresses (for computing offsets from symbol values)
  uint64_t TextSectionAddress = 0;
  uint64_t RodataSectionAddress = 0;

  /// Original program headers (to preserve for ET_DYN)
  std::vector<uint8_t> ProgramHeaders;
  uint16_t ProgramHeaderCount = 0;

  /// Complete original ELF bytes (for byte surgery approach)
  std::vector<uint8_t> OriginalBytes;

  /// Section file offsets (for byte surgery)
  uint64_t TextSectionFileOffset = 0;
  uint64_t RodataSectionFileOffset = 0;
  uint64_t NoteSectionFileOffset = 0;
};

/// Instrumentation mode
enum class InstrumentationMode {
  MEMORY_ONLY,        ///< Memory coalescing profiler
};

/// Offset mapping for branch fixup (O(1) lookup)
struct OffsetMap {
  std::unordered_map<uint64_t, uint64_t> Mappings; ///< OldOffset -> NewOffset

  /// Get new offset for original offset (O(1) average)
  uint64_t getNewOffset(uint64_t OldOffset) const {
    auto It = Mappings.find(OldOffset);
    return (It != Mappings.end()) ? It->second : OldOffset;
  }

  /// Add mapping
  void addMapping(uint64_t OldOffset, uint64_t NewOffset) {
    Mappings[OldOffset] = NewOffset;
  }

  /// Check if offset has a mapping
  bool hasMapping(uint64_t OldOffset) const {
    return Mappings.find(OldOffset) != Mappings.end();
  }

  /// Get number of mappings
  size_t size() const { return Mappings.size(); }
};

//===----------------------------------------------------------------------===//
// Trace record formats
//===----------------------------------------------------------------------===//

/// Memory trace record (per-lane addresses for each memory instruction)
struct MemoryTraceRecord {
  uint32_t InstrID;           ///< Instruction ID (unique per kernel)
  uint32_t WaveID;            ///< Wavefront ID
  uint64_t Reserved;          ///< Reserved for alignment/future use
  uint64_t Addresses[64];     ///< Per-lane addresses (64 lanes x 8 bytes)

  static constexpr size_t HeaderSize = 16;
  static constexpr size_t AddressesSize = 512;
  static constexpr size_t Size = 528;
};

//===----------------------------------------------------------------------===//
// Above-the-count register allocation (Approach 2)
//===----------------------------------------------------------------------===//

/// A memory instruction site to instrument (global_load / global_store).
/// Replaces ScavengeSite — no liveness analysis needed.
struct InstrumentationSite {
  uint64_t Address = 0;      ///< PC address in the original binary
  uint64_t Offset = 0;       ///< Byte offset from kernel start
  llvm::MCInst OrigInst;     ///< Original instruction (for displaced copy)
  uint64_t OrigInstSize = 0; ///< Size of original instruction (4 or 8)
  bool IsLoad = false;       ///< true = load, false = store
  bool IsGlobal = false;     ///< true = global (64-bit addr), false = LDS

  /// Index of the VGPR holding the low 32 bits of the memory address.
  /// For GLOBAL instructions with saddr=off, this is a VReg_64 pair:
  /// the full 64-bit address lives in v[AddrVGPRIndex : AddrVGPRIndex+1].
  /// For GLOBAL instructions with saddr, this is a single 32-bit offset VGPR.
  unsigned AddrVGPRIndex = 0;

  /// true if the address is a 64-bit VGPR pair (saddr=off), false if 32-bit.
  bool Addr64 = false;

  uint16_t DSOffset0 = 0;  ///< DS instruction offset0 (bytes), added to vaddr
  uint16_t DSOffset1 = 0;  ///< DS instruction offset1 (for DS_*2 dual-address)
  bool IsDualDS = false;    ///< true for DS_READ2/DS_WRITE2 (two offsets per instruction)

  /// Per-site minimum s_waitcnt values for the AccVGPR spill pre-drain.
  /// Computed by computePreSpillDrainValues() after site discovery.
  /// 63 / 15 = "no wait needed" (architecture max for vmcnt / lgkmcnt).
  unsigned PreSpillVmWait = 0;
  unsigned PreSpillLgkmWait = 0;
};

// ScratchRegisters has been relocated to aegisbit/ScratchRegisters.h.
// Types.h re-includes that header at the bottom so existing callers that
// only include Types.h continue to compile without edits.
#if 0 // Legacy definition retained for reference only; see ScratchRegisters.h
struct _ScratchRegisters_legacy_ {
  unsigned ReturnAddrSGPR = 0;   ///< s[N]   — return addr lo
  unsigned ReturnAddrSGPRHi = 0; ///< s[N+1] — return addr hi
  unsigned ScratchVGPR = 0;      ///< v[M]   — save RA via writelane/readlane

  // --- Additional registers for instrumented trampoline ---

  unsigned ScratchSGPR = 0;      ///< s[N+2] — general-purpose scalar scratch
  unsigned ExecSaveSGPRLo = 0;   ///< s[N+3] — save exec_lo during atomic
  unsigned ExecSaveSGPRHi = 0;   ///< s[N+4] — save exec_hi during atomic
  unsigned SAddrTempSGPR = 0;    ///< s[N+5] — SADDR temp for large-offset scratch ops
  unsigned SwapTargetSGPR = 0;   ///< s[N+6] — swappc target addr lo
  unsigned SwapTargetSGPRHi = 0; ///< s[N+7] — swappc target addr hi
  unsigned LaneVGPR = 0;         ///< v[M+1] — lane ID / per-lane write offset
  unsigned TempVGPR = 0;         ///< v[M+2] — temp for atomic value

  uint32_t ExtraVGPRs = 0;      ///< Requested extra VGPRs (descriptor delta)
  uint32_t ExtraSGPRs = 0;      ///< Requested extra SGPRs (descriptor delta)
  bool HasAccumVGPRs = false;   ///< True if kernel uses AccVGPRs (gfx90a/gfx942)
  bool NeedsAccVGPRSpill = false; ///< True if victim VGPRs must be spilled to AccVGPRs (DEPRECATED)
  bool NeedsScratchSpill = false; ///< True if victim VGPRs must be spilled to scratch memory
  bool ZeroSGPR = false;         ///< True if using VCC-temp mode (no extra SGPRs allocated)
  bool UseSwapPC = false;        ///< True if using s_swappc_b64 shared-body mode

  /// AccVGPR spill slots for saving victim VGPRs (LLVM AGPR register numbers).
  /// Used when all regular VGPRs are referenced and we must spill/restore.
  /// DEPRECATED: Use scratch memory spill instead for robustness.
  unsigned SpillAGPR0 = 0;      ///< Spill slot for ScratchVGPR
  unsigned SpillAGPR1 = 0;      ///< Spill slot for LaneVGPR
  unsigned SpillAGPR2 = 0;      ///< Spill slot for TempVGPR
  unsigned SpillAGPR3 = 0;      ///< Spill slot for AddrVGPR during counting loop

  /// Scratch memory spill configuration.
  /// Used for AccVGPR kernels where we spill victim VGPRs to scratch memory
  /// instead of AccVGPR slots. More robust than AccVGPR spill.
  uint32_t ScratchSpillOffset = 0;  ///< Byte offset within scratch for our spill area
  uint32_t ExtraScratchBytes = 0;   ///< Additional scratch bytes needed (12 = 3 VGPRs × 4 bytes)

  /// Raw SGPR/VGPR indices (not LLVM register numbers) for the first free
  /// register. Useful for encoding instructions that take raw indices.
  unsigned FirstFreeSGPRIdx = 0;
  unsigned FirstFreeVGPRIdx = 0;

  /// Minimal allocation for empty trampoline (2 SGPR + 1 VGPR).
  static ScratchRegisters fromDescriptor(const KernelDescriptor &KD) {
    ScratchRegisters SR;
    SR.FirstFreeSGPRIdx = KD.SGPRCount;
    SR.FirstFreeVGPRIdx = KD.VGPRCount;

    SR.ReturnAddrSGPR   = RegisterHelper::getSGPR(KD.SGPRCount);
    SR.ReturnAddrSGPRHi = RegisterHelper::getSGPR(KD.SGPRCount + 1);
    SR.ScratchVGPR      = RegisterHelper::getVGPR(KD.VGPRCount);
    SR.ExtraSGPRs = 2;
    SR.ExtraVGPRs = 1;
    return SR;
  }

  /// Full allocation for instrumented trampoline (5 SGPR + 3 VGPR).
  ///
  /// On GFX9/CDNA, the hardware places implicit registers at the TOP of
  /// the granulated SGPR allocation:
  ///   VCC          = s[N-2 : N-1]  (always, 2 SGPRs)
  ///   FLAT_SCRATCH = s[N-4 : N-3]  (if enabled)
  ///   XNACK_MASK   = s[N-6 : N-5]  (if enabled)
  /// where N = granulated total SGPRs.
  ///
  /// KD.SGPRCount is already granulated and includes implicit SGPRs.
  /// When we expand the allocation, implicit SGPRs move to the new top,
  /// freeing the slots they previously occupied for our scratch SGPRs.
  static ScratchRegisters fromDescriptorInstrumented(const KernelDescriptor &KD) {
    ScratchRegisters SR;
    SR.HasAccumVGPRs = (KD.AccumOffset > 0);

    static constexpr uint32_t ScratchSGPRCount = 6;
    const uint32_t ImplicitSGPRs = KD.ImplicitSGPRs;
    static constexpr uint32_t SGPRGranularity = 8;

    // KD.SGPRCount includes implicit SGPRs at the top.  When we expand
    // the allocation, those implicit regs move up, so we can reclaim
    // their old slots.  Place scratch starting at SGPRCount - ImplicitSGPRs.
    uint32_t ScratchBase = KD.SGPRCount - ImplicitSGPRs;
    SR.FirstFreeSGPRIdx = ScratchBase;

    // New total: scratch base + our 6 scratch + implicit at the new top
    uint32_t MinTotal = ScratchBase + ScratchSGPRCount + ImplicitSGPRs;
    uint32_t Granulated = ((MinTotal + SGPRGranularity - 1) / SGPRGranularity)
                          * SGPRGranularity;
    uint32_t ExtraSGPRs = Granulated - KD.SGPRCount;

    SR.ReturnAddrSGPR   = RegisterHelper::getSGPR(ScratchBase);
    SR.ReturnAddrSGPRHi = RegisterHelper::getSGPR(ScratchBase + 1);
    SR.ScratchSGPR      = RegisterHelper::getSGPR(ScratchBase + 2);
    SR.ExecSaveSGPRLo   = RegisterHelper::getSGPR(ScratchBase + 3);
    SR.ExecSaveSGPRHi   = RegisterHelper::getSGPR(ScratchBase + 4);
    SR.SAddrTempSGPR    = RegisterHelper::getSGPR(ScratchBase + 5);

    if (KD.AccumOffset > 0 && KD.AccumOffset <= 256) {
      // gfx942/gfx950 with AccVGPRs: VOP/FLAT VDST is 8 bits (v0-v255 only).
      // Scratch VGPRs will first be searched for via refineScratchVGPRs().
      // If none are free, we fall back to spilling 3 "victim" VGPRs to
      // unused AccVGPR slots at trampoline entry/exit.
      SR.FirstFreeVGPRIdx = KD.AccumOffset - 3; // placeholder
      SR.ScratchVGPR = 0; // set by refineScratchVGPRs() or setupAccVGPRSpill()
      SR.LaneVGPR    = 0;
      SR.TempVGPR    = 0;
      SR.ExtraVGPRs = 0;
    } else {
      // No AccVGPRs (or gfx90a with separate 'a' registers): place above total.
      SR.FirstFreeVGPRIdx = KD.VGPRCount;
      SR.ScratchVGPR = RegisterHelper::getVGPR(KD.VGPRCount);
      SR.LaneVGPR    = RegisterHelper::getVGPR(KD.VGPRCount + 1);
      SR.TempVGPR    = RegisterHelper::getVGPR(KD.VGPRCount + 2);
      SR.ExtraVGPRs = 3;
    }

    SR.ExtraSGPRs = ExtraSGPRs;
    return SR;
  }

  /// SwapPC allocation: 8 SGPRs + 3 VGPRs for unlimited-range shared-body mode.
  static ScratchRegisters fromDescriptorSwapPC(const KernelDescriptor &KD) {
    ScratchRegisters SR;
    SR.HasAccumVGPRs = (KD.AccumOffset > 0);
    SR.UseSwapPC = true;

    static constexpr uint32_t ScratchSGPRCount = 8;
    const uint32_t ImplicitSGPRs = KD.ImplicitSGPRs;
    static constexpr uint32_t SGPRGranularity = 8;

    uint32_t ScratchBase = KD.SGPRCount - ImplicitSGPRs;
    SR.FirstFreeSGPRIdx = ScratchBase;

    uint32_t MinTotal = ScratchBase + ScratchSGPRCount + ImplicitSGPRs;
    uint32_t Granulated =
        ((MinTotal + SGPRGranularity - 1) / SGPRGranularity) * SGPRGranularity;
    uint32_t ExtraSGPRs = Granulated - KD.SGPRCount;

    SR.ReturnAddrSGPR = RegisterHelper::getSGPR(ScratchBase);
    SR.ReturnAddrSGPRHi = RegisterHelper::getSGPR(ScratchBase + 1);
    SR.ScratchSGPR = RegisterHelper::getSGPR(ScratchBase + 2);
    SR.ExecSaveSGPRLo = RegisterHelper::getSGPR(ScratchBase + 3);
    SR.ExecSaveSGPRHi = RegisterHelper::getSGPR(ScratchBase + 4);
    SR.SAddrTempSGPR = RegisterHelper::getSGPR(ScratchBase + 5);
    SR.SwapTargetSGPR = RegisterHelper::getSGPR(ScratchBase + 6);
    SR.SwapTargetSGPRHi = RegisterHelper::getSGPR(ScratchBase + 7);

    if (KD.AccumOffset > 0 && KD.AccumOffset <= 256) {
      SR.FirstFreeVGPRIdx = KD.AccumOffset - 3;
      SR.ScratchVGPR = 0;
      SR.LaneVGPR = 0;
      SR.TempVGPR = 0;
      SR.ExtraVGPRs = 0;
    } else {
      SR.FirstFreeVGPRIdx = KD.VGPRCount;
      SR.ScratchVGPR = RegisterHelper::getVGPR(KD.VGPRCount);
      SR.LaneVGPR = RegisterHelper::getVGPR(KD.VGPRCount + 1);
      SR.TempVGPR = RegisterHelper::getVGPR(KD.VGPRCount + 2);
      SR.ExtraVGPRs = 3;
    }

    SR.ExtraSGPRs = ExtraSGPRs;
    return SR;
  }

  /// Zero-extra-SGPR allocation for kernels that max out SGPR capacity.
  ///
  /// Uses VCC_LO as the sole temp SGPR (VCC is always available as an
  /// implicit register). The trampoline uses s_branch instead of s_call_b64
  /// (no return address SGPR pair), and saves/restores EXEC and SCC
  /// directly through VCC + VGPR lanes.
  ///
  /// Requires: SupportsGPUAtomics (fire-and-forget atomics only).
  static ScratchRegisters fromDescriptorZeroSGPR(const KernelDescriptor &KD) {
    ScratchRegisters SR;
    SR.HasAccumVGPRs = (KD.AccumOffset > 0);
    SR.ZeroSGPR = true;
    SR.ExtraSGPRs = 0;

    SR.ReturnAddrSGPR = 0;
    SR.ReturnAddrSGPRHi = 0;
    SR.ScratchSGPR = 0;
    SR.ExecSaveSGPRLo = 0;
    SR.ExecSaveSGPRHi = 0;
    SR.FirstFreeSGPRIdx = 0;

    if (KD.AccumOffset > 0 && KD.AccumOffset <= 256) {
      SR.FirstFreeVGPRIdx = KD.AccumOffset - 3;
      SR.ScratchVGPR = 0;
      SR.LaneVGPR    = 0;
      SR.TempVGPR    = 0;
      SR.ExtraVGPRs = 0;
    } else {
      SR.FirstFreeVGPRIdx = KD.VGPRCount;
      SR.ScratchVGPR = RegisterHelper::getVGPR(KD.VGPRCount);
      SR.LaneVGPR    = RegisterHelper::getVGPR(KD.VGPRCount + 1);
      SR.TempVGPR    = RegisterHelper::getVGPR(KD.VGPRCount + 2);
      SR.ExtraVGPRs = 3;
    }

    return SR;
  }

  /// Refine scratch VGPRs for AccVGPR kernels by scanning the CFG for
  /// VGPRs never referenced by any instruction in v0..v(AccumOffset-1).
  /// Returns true if 3 free VGPRs were found, false if not enough free VGPRs.
  bool refineScratchVGPRs(const ControlFlowGraph &CFG,
                          const class Disassembler &Disasm,
                          uint32_t AccumOffset);

  /// Set up AccVGPR spill/restore for kernels where all regular VGPRs
  /// are in use. Picks 3 victim VGPRs (high-numbered) and 3 unused
  /// AccVGPR spill slots above the kernel's AccVGPR usage.
  /// DEPRECATED: Use setupScratchSpill() instead for robustness.
  /// \param AccumOffset  Boundary between regular and accum VGPRs
  /// \param VGPRCount    Total VGPR count (regular + accum)
  void setupAccVGPRSpill(uint32_t AccumOffset, uint32_t VGPRCount);

  /// Set up scratch memory spill/restore for AccVGPR kernels.
  /// Uses 3 high-numbered victim VGPRs and spills them to scratch memory.
  /// More robust than AccVGPR spill - doesn't interfere with MFMA state.
  /// \param AccumOffset  Boundary between regular and accum VGPRs
  /// \param CurrentScratchSize  Kernel's current PrivateSegmentFixedSize
  void setupScratchSpill(uint32_t AccumOffset, uint32_t CurrentScratchSize);

  bool isValid() const {
    if (ZeroSGPR)
      return ScratchVGPR != 0;
    return ReturnAddrSGPR != 0 && ReturnAddrSGPRHi != 0 && ScratchVGPR != 0;
  }
};
#endif // ---- END ScratchRegisters relocated body ----

/// Selects the instrumentation payload strategy for trampolines.
enum class PayloadStrategy {
  /// Compute coalescing/bank-conflict metrics on-GPU using wave intrinsics.
  /// Output: small per-site counters. No per-execution address dump.
  OnGpuReduce,

  /// Dump all 64 lane addresses per execution to a global buffer via atomic
  /// slot allocation. High fidelity but massive memory/bandwidth overhead.
  FullCapture,
};

/// Configuration for the instrumented trampoline's trace output.
/// Addresses are GPU virtual addresses set at patch time and embedded
/// as immediates in the trampoline instructions.
struct TraceConfig {
  uint64_t BufferAddr = 0;     ///< GPU VA of the trace record buffer
  uint64_t CounterAddr = 0;    ///< GPU VA of the atomic write-offset counter
  uint64_t BufferSize = 0;     ///< Total buffer size in bytes

  PayloadStrategy Strategy = PayloadStrategy::OnGpuReduce;

  /// True if the trace buffer is in fine-grained memory that supports GPU
  /// atomics.  When set, OnGpuReduce uses fire-and-forget global_atomic_add
  /// (no return) instead of load-modify-store, eliminating s_waitcnt vmcnt(0)
  /// from the payload and preserving software-pipelined VMEM traffic.
  bool SupportsGPUAtomics = false;

  /// Size of one trace record: 8-byte header + 64 × 8-byte addresses.
  /// Only used by FullCapture strategy.
  static constexpr uint32_t RecordSize = 8 + 64 * 8; // 520 bytes

  /// Per-site counter size for OnGpuReduce: {total_cache_lines, total_samples}.
  static constexpr uint32_t CounterSize = 8;
};

} // namespace aegisbit

// Re-include the relocated ScratchRegisters header so callers that only
// include Types.h continue to compile.
#include "aegisbit/ScratchRegisters.h"

#endif // AEGISBIT_TYPES_H
