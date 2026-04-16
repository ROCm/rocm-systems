#include "aegisbit/ISAEncoder.h"
#include "aegisbit/Disassembler.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/TargetSelect.h"
#include <cstdio>

using namespace aegisbit;

int main() {
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllDisassemblers();

  auto D = Disassembler::create("gfx950");
  if (!D) { printf("Failed to create disassembler\n"); return 1; }

  auto Enc = ISAEncoder::create("gfx950", **D);
  if (!Enc) { printf("Failed to create encoder\n"); return 1; }

  auto LJ = (*Enc)->encodeLongJump(20, 0x1000);
  if (!LJ) { printf("Failed to encode long jump\n"); return 1; }

  printf("Encoded long jump:\n");
  for (uint8_t b : *LJ) {
    printf("%02x ", b);
  }
  printf("\n");
  return 0;
}
