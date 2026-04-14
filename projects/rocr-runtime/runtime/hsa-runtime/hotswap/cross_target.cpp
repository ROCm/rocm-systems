////////////////////////////////////////////////////////////////////////////////
//
// Cross-Target Instruction Mapping
//
// Rewrites waveasm ops from one ISA to another by:
// 1. Remapping op names (mnemonic renames)
// 2. Translating wait counters
// 3. Updating the target attribute on waveasm.program
//
////////////////////////////////////////////////////////////////////////////////

#include "cross_target.hpp"

#include "waveasm/Dialect/WaveASMDialect.h"
#include "waveasm/Dialect/WaveASMOps.h"
#include "waveasm/Dialect/WaveASMAttrs.h"
#include "waveasm/Dialect/WaveASMTypes.h"

#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/SmallVector.h>
#include <mlir/IR/Builders.h>

using namespace mlir;

//===----------------------------------------------------------------------===//
// Mnemonic mapping tables (GFX12/1250 -> GFX9/942/950)
//===----------------------------------------------------------------------===//

struct MnemonicEntry {
  llvm::StringLiteral gfx12;
  llvm::StringLiteral gfx9;
};

static constexpr MnemonicEntry kGlobalMemMappings[] = {
    {"global_load_b32", "global_load_dword"},
    {"global_load_b64", "global_load_dwordx2"},
    {"global_load_b96", "global_load_dwordx3"},
    {"global_load_b128", "global_load_dwordx4"},
    {"global_load_u8", "global_load_ubyte"},
    {"global_load_i8", "global_load_sbyte"},
    {"global_load_u16", "global_load_ushort"},
    {"global_load_i16", "global_load_sshort"},
    {"global_store_b8", "global_store_byte"},
    {"global_store_b16", "global_store_short"},
    {"global_store_b32", "global_store_dword"},
    {"global_store_b64", "global_store_dwordx2"},
    {"global_store_b96", "global_store_dwordx3"},
    {"global_store_b128", "global_store_dwordx4"},
};

static constexpr MnemonicEntry kFlatMemMappings[] = {
    {"flat_load_b32", "flat_load_dword"},
    {"flat_load_b64", "flat_load_dwordx2"},
    {"flat_load_b96", "flat_load_dwordx3"},
    {"flat_load_b128", "flat_load_dwordx4"},
    {"flat_load_u8", "flat_load_ubyte"},
    {"flat_load_i8", "flat_load_sbyte"},
    {"flat_load_u16", "flat_load_ushort"},
    {"flat_store_b8", "flat_store_byte"},
    {"flat_store_b16", "flat_store_short"},
    {"flat_store_b32", "flat_store_dword"},
    {"flat_store_b64", "flat_store_dwordx2"},
    {"flat_store_b96", "flat_store_dwordx3"},
    {"flat_store_b128", "flat_store_dwordx4"},
};

static constexpr MnemonicEntry kDSMappings[] = {
    {"ds_load_b32", "ds_read_b32"},
    {"ds_load_b64", "ds_read_b64"},
    {"ds_load_b128", "ds_read_b128"},
    {"ds_load_u8", "ds_read_u8"},
    {"ds_load_i8", "ds_read_i8"},
    {"ds_load_u16", "ds_read_u16"},
    {"ds_store_b32", "ds_write_b32"},
    {"ds_store_b64", "ds_write_b64"},
    {"ds_store_b128", "ds_write_b128"},
    {"ds_store_b8", "ds_write_b8"},
    {"ds_store_b16", "ds_write_b16"},
};

static constexpr MnemonicEntry kSMEMMappings[] = {
    {"s_load_b32", "s_load_dword"},
    {"s_load_b64", "s_load_dwordx2"},
    {"s_load_b128", "s_load_dwordx4"},
    {"s_load_b256", "s_load_dwordx8"},
    {"s_load_b512", "s_load_dwordx16"},
    {"s_buffer_load_b32", "s_buffer_load_dword"},
    {"s_buffer_load_b64", "s_buffer_load_dwordx2"},
    {"s_buffer_load_b128", "s_buffer_load_dwordx4"},
};

static constexpr MnemonicEntry kScalarALURenames[] = {
    {"s_add_co_u32", "s_add_u32"},
    {"s_add_co_i32", "s_add_i32"},
    {"s_sub_co_u32", "s_sub_u32"},
    {"s_add_co_ci_u32", "s_addc_u32"},
    {"s_sub_co_ci_u32", "s_subb_u32"},
    {"s_and_not1_b32", "s_andn2_b32"},
    {"s_and_not1_b64", "s_andn2_b64"},
    {"s_or_not1_b32", "s_orn2_b32"},
    {"s_or_not1_b64", "s_orn2_b64"},
    {"s_ctz_i32_b32", "s_ff1_i32_b32"},
    {"s_ctz_i32_b64", "s_ff1_i32_b64"},
    {"s_get_pc_i64", "s_getpc_b64"},
    {"s_swap_pc_i64", "s_swappc_b64"},
    {"s_set_pc_i64", "s_setpc_b64"},
};

static constexpr MnemonicEntry kVALURenames[] = {
    {"v_max_num_f32", "v_max_f32"},
    {"v_min_num_f32", "v_min_f32"},
    {"v_max_num_f16", "v_max_f16"},
    {"v_min_num_f16", "v_min_f16"},
    {"v_max_num_f64", "v_max_f64"},
    {"v_min_num_f64", "v_min_f64"},
    {"v_add_nc_u32", "v_add_u32"},
    {"v_sub_nc_u32", "v_sub_u32"},
    {"v_add_nc_i32", "v_add_i32"},
    {"v_sub_nc_i32", "v_sub_i32"},
    {"v_pk_add_num_f16", "v_pk_add_f16"},
    {"v_pk_mul_num_f16", "v_pk_mul_f16"},
    {"v_pk_max_num_f16", "v_pk_max_f16"},
    {"v_pk_min_num_f16", "v_pk_min_f16"},
    {"v_pk_fma_num_f16", "v_pk_fma_f16"},
    {"v_clz_i32_u32", "v_ffbh_u32"},
    {"v_ctz_i32_b32", "v_ffbl_b32"},
    {"v_add_co_ci_u32", "v_addc_co_u32"},
    {"v_sub_co_ci_u32", "v_subb_co_u32"},
};

static constexpr MnemonicEntry kGlobalAtomicRenames[] = {
    {"global_atomic_add_u32", "global_atomic_add"},
    {"global_atomic_sub_u32", "global_atomic_sub"},
    {"global_atomic_and_b32", "global_atomic_and"},
    {"global_atomic_or_b32", "global_atomic_or"},
    {"global_atomic_xor_b32", "global_atomic_xor"},
    {"global_atomic_min_i32", "global_atomic_smin"},
    {"global_atomic_max_i32", "global_atomic_smax"},
    {"global_atomic_min_u32", "global_atomic_umin"},
    {"global_atomic_max_u32", "global_atomic_umax"},
    {"global_atomic_swap_b32", "global_atomic_swap"},
    {"global_atomic_cmpswap_b32", "global_atomic_cmpswap"},
};

static constexpr MnemonicEntry kWaitCountMappings[] = {
    {"s_wait_loadcnt", "s_waitcnt"},
    {"s_wait_storecnt", "s_waitcnt"},
    {"s_wait_kmcnt", "s_waitcnt"},
    {"s_wait_dscnt", "s_waitcnt"},
    {"s_wait_expcnt", "s_waitcnt"},
};

//===----------------------------------------------------------------------===//
// Build the mapping
//===----------------------------------------------------------------------===//

static llvm::StringMap<std::string> buildGfx12ToGfx9Map() {
  llvm::StringMap<std::string> map;

  auto add = [&](const auto &table) {
    for (const auto &entry : table)
      map[entry.gfx12] = entry.gfx9.str();
  };

  add(kGlobalMemMappings);
  add(kFlatMemMappings);
  add(kDSMappings);
  add(kSMEMMappings);
  add(kScalarALURenames);
  add(kVALURenames);
  add(kGlobalAtomicRenames);
  add(kWaitCountMappings);

  return map;
}

static const llvm::StringMap<std::string> &getGfx12ToGfx9Map() {
  static auto map = buildGfx12ToGfx9Map();
  return map;
}

//===----------------------------------------------------------------------===//
// Public API
//===----------------------------------------------------------------------===//

std::string hotswap::mapMnemonic(llvm::StringRef mnemonic,
                                 llvm::StringRef sourceISA,
                                 llvm::StringRef targetISA) {
  bool isGfx12Source = sourceISA.starts_with("gfx12") ||
                       sourceISA.starts_with("gfx1250");
  bool isGfx9Target = targetISA.starts_with("gfx9");

  if (isGfx12Source && isGfx9Target) {
    auto &map = getGfx12ToGfx9Map();
    auto it = map.find(mnemonic);
    if (it != map.end())
      return it->second;
  }

  return mnemonic.str();
}

LogicalResult hotswap::retargetModule(ModuleOp module,
                                      llvm::StringRef sourceISA,
                                      llvm::StringRef targetISA) {
  auto *ctx = module.getContext();

  bool isGfx12Source = sourceISA.starts_with("gfx12") ||
                       sourceISA.starts_with("gfx1250");
  bool isGfx9Target = targetISA.starts_with("gfx9");

  module->walk([&](waveasm::ProgramOp program) {
    Block &body = program.getBody().front();
    OpBuilder builder(ctx);

    // Update the target attribute.
    waveasm::TargetAttrInterface targetKind;
    if (targetISA == "gfx942")
      targetKind = waveasm::GFX942TargetAttr::get(ctx);
    else if (targetISA == "gfx950")
      targetKind = waveasm::GFX950TargetAttr::get(ctx);
    else if (targetISA == "gfx1250")
      targetKind = waveasm::GFX1250TargetAttr::get(ctx);
    else
      targetKind = waveasm::GFX942TargetAttr::get(ctx);

    auto newTarget = waveasm::TargetAttr::get(ctx, targetKind, 5);
    program.setTargetAttr(newTarget);

    // Walk all ops and remap mnemonics.
    llvm::SmallVector<Operation *> opsToRewrite;
    for (Operation &op : body) {
      auto name = op.getName().getStringRef();
      if (!name.starts_with("waveasm."))
        continue;

      llvm::StringRef mnemonic = name.drop_front(8); // drop "waveasm."
      std::string newMnemonic = mapMnemonic(mnemonic, sourceISA, targetISA);
      if (newMnemonic != mnemonic)
        opsToRewrite.push_back(&op);
    }

    for (Operation *op : opsToRewrite) {
      auto name = op->getName().getStringRef();
      llvm::StringRef mnemonic = name.drop_front(8);
      std::string newMnemonic = mapMnemonic(mnemonic, sourceISA, targetISA);
      std::string newOpName = ("waveasm." + newMnemonic);

      builder.setInsertionPoint(op);
      OperationState state(op->getLoc(), newOpName);
      state.addTypes(op->getResultTypes());
      state.addOperands(op->getOperands());
      for (auto attr : op->getAttrs())
        state.addAttribute(attr.getName(), attr.getValue());

      auto *newOp = builder.create(state);

      // Replace results.
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        op->getResult(i).replaceAllUsesWith(newOp->getResult(i));

      op->erase();
    }

    // ── Erase GFX12-only scheduling/hint instructions ──
    if (isGfx9Target) {
      llvm::SmallVector<Operation *> opsToErase;
      for (Operation &op : body) {
        auto name = op.getName().getStringRef();
        if (name == "waveasm.s_clause" || name == "waveasm.s_delay_alu" ||
            name == "waveasm.s_code_end")
          opsToErase.push_back(&op);
      }
      for (Operation *op : opsToErase) {
        op->dropAllUses();
        op->erase();
      }
    }

    // ── Lower scale_offset addressing on global loads/stores ──
    // GFX1250 uses: global_load_dword dst, v_index(32), s[base:base+1]
    // GFX942  needs: global_load_dword dst, v[addr_lo:addr_hi](64), off
    // We detect the pattern by operand types: first source is a single VGPR,
    // second source is an SGPR pair (64-bit base).
    if (isGfx12Source && isGfx9Target) {
      // Fixed temp register allocation: v3 = scratch, v[4:5] = address pair.
      // The gfx942 template's next_free_vgpr is 8, so these are safe.
      constexpr int64_t kScratchVgpr = 3;
      constexpr int64_t kAddrLoVgpr = 4;  // must be even-aligned
      constexpr int64_t kAddrHiVgpr = 5;

      llvm::SmallVector<Operation *> globalOps;
      for (Operation &op : body) {
        auto name = op.getName().getStringRef();
        if (!name.starts_with("waveasm.global_load_") &&
            !name.starts_with("waveasm.global_store_"))
          continue;
        globalOps.push_back(&op);
      }

      for (Operation *op : globalOps) {
        auto name = op->getName().getStringRef();
        bool isStore = name.starts_with("waveasm.global_store_");
        auto loc = op->getLoc();

        // Determine element size from the mnemonic suffix.
        llvm::StringRef mnem = name.drop_front(8);
        int shiftAmount = 2; // default: dword = 4 bytes = shift by 2
        if (mnem.contains("dwordx2") || mnem.contains("b64"))
          shiftAmount = 3;
        else if (mnem.contains("dwordx3") || mnem.contains("b96"))
          shiftAmount = -1; // 12 bytes, can't shift — need multiply
        else if (mnem.contains("dwordx4") || mnem.contains("b128"))
          shiftAmount = 4;
        else if (mnem.contains("byte") || mnem.contains("b8") ||
                 mnem.contains("u8") || mnem.contains("i8"))
          shiftAmount = 0;
        else if (mnem.contains("short") || mnem.contains("b16") ||
                 mnem.contains("u16") || mnem.contains("i16"))
          shiftAmount = 1;

        // Detect scale_offset pattern by scanning operands for a 32-bit
        // single VGPR (the index) paired with a 64-bit SGPR pair (the base).
        // The MC disassembler may order them differently for loads vs stores.
        int indexOpIdx = -1;
        int baseOpIdx = -1;
        int dataOpIdx = -1; // for stores: the value being stored

        for (unsigned i = 0; i < op->getNumOperands(); ++i) {
          Type ty = op->getOperand(i).getType();
          if (auto pv = dyn_cast<waveasm::PVRegType>(ty)) {
            if (pv.getSize() == 1 && indexOpIdx < 0)
              indexOpIdx = i;
            else if (isStore && dataOpIdx < 0)
              dataOpIdx = i;
          } else if (auto ps = dyn_cast<waveasm::PSRegType>(ty)) {
            if (ps.getSize() == 2 && ps.getIndex() < 100 && baseOpIdx < 0)
              baseOpIdx = i;
          }
        }

        if (indexOpIdx < 0 || baseOpIdx < 0)
          continue;

        Value indexVal = op->getOperand(indexOpIdx);
        Value baseVal = op->getOperand(baseOpIdx);

        auto baseSgpr = cast<waveasm::PSRegType>(baseVal.getType());
        int64_t baseIdx = baseSgpr.getIndex();

        builder.setInsertionPoint(op);

        auto createPrecolored =
            [&](llvm::StringRef regOp, int64_t idx,
                int64_t sz) -> std::pair<Value, Type> {
          Type ty;
          std::string opName;
          if (regOp == "vreg") {
            ty = waveasm::PVRegType::get(ctx, idx, sz);
            opName = "waveasm.precolored.vreg";
          } else {
            ty = waveasm::PSRegType::get(ctx, idx, sz);
            opName = "waveasm.precolored.sreg";
          }
          OperationState st(loc, opName);
          st.addTypes({ty});
          st.addAttribute(
              "index", builder.getIntegerAttr(builder.getI64Type(), idx));
          st.addAttribute(
              "size", builder.getIntegerAttr(builder.getI64Type(), sz));
          return {builder.create(st)->getResult(0), ty};
        };

        auto createImm = [&](int64_t val) -> Value {
          auto ty = waveasm::ImmType::get(ctx, val);
          OperationState st(loc, "waveasm.constant");
          st.addTypes({ty});
          st.addAttribute("value", builder.getI64IntegerAttr(val));
          return builder.create(st)->getResult(0);
        };

        auto createOp = [&](llvm::StringRef mn, TypeRange resTy,
                            ValueRange operands) -> Operation * {
          OperationState st(loc, ("waveasm." + mn).str());
          st.addTypes(resTy);
          st.addOperands(operands);
          return builder.create(st);
        };

        auto scratchTy = waveasm::PVRegType::get(ctx, kScratchVgpr, 1);
        auto addrLoTy = waveasm::PVRegType::get(ctx, kAddrLoVgpr, 1);
        auto addrHiTy = waveasm::PVRegType::get(ctx, kAddrHiVgpr, 1);
        auto addrPairTy = waveasm::PVRegType::get(ctx, kAddrLoVgpr, 2);

        // v3 = v_index << shift (= index * element_size)
        Value shiftImm = createImm(shiftAmount);
        createOp("v_lshlrev_b32_e32", {scratchTy}, {shiftImm, indexVal});

        // v5 = base_hi
        auto [baseLo, baseLoTy] = createPrecolored("sreg", baseIdx, 1);
        auto [baseHi, baseHiTy] = createPrecolored("sreg", baseIdx + 1, 1);
        createOp("v_mov_b32_e32", {addrHiTy}, {baseHi});

        // v4 = base_lo + v3 (with carry)
        auto [scratchRead, scratchReadTy] =
            createPrecolored("vreg", kScratchVgpr, 1);
        auto vccTy = waveasm::PSRegType::get(ctx, 106, 2);
        createOp("v_add_co_u32_e32", {addrLoTy, vccTy},
                 {baseLo, scratchRead});

        // v5 = 0 + v5 + carry
        auto [addrHiRead, addrHiReadTy] =
            createPrecolored("vreg", kAddrHiVgpr, 1);
        Value zero = createImm(0);
        auto [vccRead, vccReadTy] = createPrecolored("sreg", 106, 2);
        createOp("v_addc_co_u32_e32", {addrHiTy, vccTy},
                 {zero, addrHiRead, vccRead});

        // Build the new global load/store with v[4:5] as the 64-bit address.
        auto [addrPair, addrPairType] =
            createPrecolored("vreg", kAddrLoVgpr, 2);
        Value offImm = createImm(0);

        OperationState newState(loc, op->getName().getStringRef().str());
        if (isStore) {
          // global_store_dword v[4:5], v_data, off
          Value dataVal = (dataOpIdx >= 0) ? op->getOperand(dataOpIdx)
                                           : op->getOperand(1);
          newState.addOperands({addrPair, dataVal, offImm});
        } else {
          // global_load_dword dst, v[4:5], off
          newState.addOperands({addrPair, offImm});
        }
        newState.addTypes(op->getResultTypes());
        auto *newOp = builder.create(newState);

        for (unsigned i = 0; i < op->getNumResults(); ++i)
          op->getResult(i).replaceAllUsesWith(newOp->getResult(i));
        op->erase();
      }
    }
  });

  return success();
}
