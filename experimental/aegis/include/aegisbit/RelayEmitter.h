//===-- aegisbit/RelayEmitter.h - Relay Stub Generation ------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Generates relay forward/return stubs for sites beyond s_branch range,
/// manages the body island buffer, and performs long-jump fixup.
///
/// A relay stub pair consists of:
///   Forward stub: save VCC/SCC, long-jump to body island entry
///   Return stub:  restore SCC/VCC, execute displaced instruction, s_branch back
///
/// The full trampoline body lives in a distant "body island" reachable
/// only via long-jump (s_getpc_b64 + s_add + s_addc + s_setpc_b64 using VCC).
///
//===----------------------------------------------------------------------===//

#ifndef AEGISBIT_RELAY_EMITTER_H
#define AEGISBIT_RELAY_EMITTER_H

#include "aegisbit/ISAEncoder.h"
#include "aegisbit/TrampolineTypes.h"
#include "aegisbit/Types.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Error.h"
#include <cstdint>
#include <cstring>
#include <vector>

namespace aegisbit {

struct RelayFixup {
  uint64_t FwdGetPCAbs;
  uint64_t RelayReturnAbs;
  size_t BodyEntryOff;
  size_t RetGetPCBodyOff;
};

struct RelayStubs {
  std::vector<uint8_t> ForwardStub;
  std::vector<uint8_t> ReturnStub;
  size_t FwdLongJumpOffset = 0; // offset within ForwardStub of the long-jump
  bool UseSBranchBody = false;  // diagnostic: use s_branch instead of long jumps
};

class RelayEmitter {
public:
  explicit RelayEmitter(ISAEncoder &Enc);
  ~RelayEmitter();

  /// Emit the forward and return stubs for a relay site.
  llvm::Expected<RelayStubs>
  emitRelayStubs(const InstrumentationSite &Site,
                 const ScratchRegisters &Scratch,
                 llvm::ArrayRef<uint8_t> Code,
                 uint64_t ReturnTargetAbs,
                 uint64_t RetBranchPC,
                 bool ForceNoBodyJump = false);

  /// Add a trampoline body to the body island buffer.
  /// Returns the offset within the body island where this entry starts.
  size_t addBodyEntry(const std::vector<uint8_t> &BodyBytes);

  /// Append a return long-jump to the body island and record its offset.
  llvm::Expected<size_t> appendReturnLongJump();

  /// Record a fixup for later patching.
  void addFixup(RelayFixup Fix);

  /// Get the current body island cursor.
  size_t getBodyCursor() const { return BodyIslandCursor; }

  /// Get the body island bytes.
  std::vector<uint8_t> &getBodyBytes() { return BodyIslandBytes; }

  /// Check if there are any relay fixups.
  bool hasFixups() const { return !Fixups.empty(); }

  /// Perform the fixup pass: patch all long-jump offsets given the final
  /// body island absolute address.
  /// Returns the body island as a TrampolineIsland.
  llvm::Expected<TrampolineIsland>
  fixupRelays(uint64_t BodyIslandStart,
              std::vector<TrampolineIsland> &StubIslands);

  /// Get the number of relay fixups.
  size_t getFixupCount() const { return Fixups.size(); }

  static constexpr int64_t LJ_PLACEHOLDER = 0x12345678;

private:
  ISAEncoder *Enc = nullptr;
  std::vector<uint8_t> BodyIslandBytes;
  size_t BodyIslandCursor = 0;
  std::vector<RelayFixup> Fixups;
};

} // namespace aegisbit

#endif // AEGISBIT_RELAY_EMITTER_H
