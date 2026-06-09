// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/compute_unit.h"

#include "rocjitsu/vm/amdgpu/command_processor.h"

#include "rocjitsu/isa/arch/amdgpu/cdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/cdna4/isa.h"
#include "rocjitsu/isa/arch/amdgpu/gfx1250/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna1/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna2/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna3_5/isa.h"
#include "rocjitsu/isa/arch/amdgpu/rdna4/isa.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "util/except.h"
#include "util/log.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace amdgpu {
namespace {

constexpr uint64_t kDecodeProfileRawPc = UINT64_MAX;

struct DecodeProfileKey {
  uint64_t pc = 0;
  uint32_t arch = 0;
  uint32_t size = 0;
  std::array<uint32_t, 4> words{};

  bool operator==(const DecodeProfileKey &other) const {
    return pc == other.pc && arch == other.arch && size == other.size && words == other.words;
  }
};

struct DecodeProfileKeyHash {
  size_t operator()(const DecodeProfileKey &key) const {
    uint64_t h = key.pc ^ (static_cast<uint64_t>(key.arch) << 32) ^ key.size;
    for (uint32_t word : key.words) {
      h ^= static_cast<uint64_t>(word) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    }
    return static_cast<size_t>(h);
  }
};

struct DecodeProfileEntry {
  uint64_t count = 0;
  std::string mnemonic;
};

class DecodeProfile {
public:
  DecodeProfile(std::string path, uint64_t stride) : path_(std::move(path)), stride_(stride) {}

  void record(rj_code_arch_t arch, uint64_t pc, const rj_code_binary_inst_t words[4],
              std::string_view mnemonic, uint32_t inst_size) {
    uint64_t seen = total_seen_.fetch_add(1, std::memory_order_relaxed);
    if ((seen % stride_) != 0)
      return;
    sampled_.fetch_add(1, std::memory_order_relaxed);

    DecodeProfileKey key;
    key.pc = pc;
    key.arch = static_cast<uint32_t>(arch);
    key.size = inst_size;
    uint32_t used_words = std::min<uint32_t>((inst_size + 3) / 4, key.words.size());
    for (uint32_t i = 0; i < used_words; ++i)
      key.words[i] = words[i];

    record_key(pc_shards_, key, mnemonic);
    key.pc = kDecodeProfileRawPc;
    record_key(raw_shards_, key, mnemonic);
  }

  void dump() {
    std::ofstream out(path_);
    if (!out) {
      std::fprintf(stderr, "rocjitsu: could not write RJ_DECODE_PROFILE=%s\n", path_.c_str());
      return;
    }

    std::vector<Row> raw_rows;
    std::vector<Row> pc_rows;
    collect_rows(raw_shards_, raw_rows);
    collect_rows(pc_shards_, pc_rows);
    std::sort(raw_rows.begin(), raw_rows.end(), row_count_desc);
    std::sort(pc_rows.begin(), pc_rows.end(), row_count_desc);

    uint64_t total_seen = total_seen_.load(std::memory_order_relaxed);
    uint64_t sampled = sampled_.load(std::memory_order_relaxed);
    out << "metric,value\n";
    out << "total_decode_attempts," << total_seen << "\n";
    out << "sample_stride," << stride_ << "\n";
    out << "sampled_decode_attempts," << sampled << "\n";
    out << "unique_raw_instructions," << raw_rows.size() << "\n";
    out << "unique_pc_instructions," << pc_rows.size() << "\n";
    out << "sampled_per_unique_raw,"
        << (raw_rows.empty() ? 0.0 : static_cast<double>(sampled) / raw_rows.size()) << "\n";
    out << "sampled_per_unique_pc,"
        << (pc_rows.empty() ? 0.0 : static_cast<double>(sampled) / pc_rows.size()) << "\n";
    out << "\nsection,rank,count,arch,pc,size,words,mnemonic\n";
    dump_rows(out, "raw", raw_rows);
    dump_rows(out, "pc", pc_rows);
  }

private:
  struct Shard {
    std::mutex mutex;
    std::unordered_map<DecodeProfileKey, DecodeProfileEntry, DecodeProfileKeyHash> counts;
  };

  struct Row {
    DecodeProfileKey key;
    DecodeProfileEntry entry;
  };

  using Shards = std::array<Shard, 64>;

  static bool row_count_desc(const Row &a, const Row &b) { return a.entry.count > b.entry.count; }

  static void record_key(Shards &shards, const DecodeProfileKey &key, std::string_view mnemonic) {
    auto hash = DecodeProfileKeyHash{}(key);
    Shard &shard = shards[hash & (shards.size() - 1)];
    std::lock_guard lock(shard.mutex);
    auto [it, inserted] = shard.counts.try_emplace(key);
    if (inserted)
      it->second.mnemonic = std::string(mnemonic);
    ++it->second.count;
  }

  static void collect_rows(Shards &shards, std::vector<Row> &rows) {
    for (auto &shard : shards) {
      std::lock_guard lock(shard.mutex);
      rows.reserve(rows.size() + shard.counts.size());
      for (const auto &[key, entry] : shard.counts)
        rows.push_back({key, entry});
    }
  }

  static void dump_rows(std::ofstream &out, std::string_view section,
                        const std::vector<Row> &rows) {
    constexpr size_t kMaxRows = 64;
    size_t limit = std::min(kMaxRows, rows.size());
    for (size_t i = 0; i < limit; ++i) {
      const auto &row = rows[i];
      out << section << "," << (i + 1) << "," << row.entry.count << "," << row.key.arch << ",";
      if (row.key.pc == kDecodeProfileRawPc)
        out << ",";
      else
        out << "0x" << std::hex << row.key.pc << std::dec << ",";
      out << row.key.size << ",\"";
      uint32_t used_words = std::min<uint32_t>((row.key.size + 3) / 4, row.key.words.size());
      for (uint32_t w = 0; w < used_words; ++w) {
        if (w)
          out << " ";
        out << "0x" << std::hex << row.key.words[w] << std::dec;
      }
      out << "\",\"" << row.entry.mnemonic << "\"\n";
    }
  }

  std::string path_;
  uint64_t stride_ = 1;
  std::atomic<uint64_t> total_seen_{0};
  std::atomic<uint64_t> sampled_{0};
  Shards raw_shards_;
  Shards pc_shards_;
};

DecodeProfile *&decode_profile_storage() {
  static DecodeProfile *profile = nullptr;
  return profile;
}

DecodeProfile *create_decode_profile() {
  const char *env = std::getenv("RJ_DECODE_PROFILE");
  if (!env || !env[0] || std::string_view(env) == "0")
    return nullptr;

  uint64_t stride = 1;
  if (const char *stride_env = std::getenv("RJ_DECODE_PROFILE_STRIDE")) {
    char *end = nullptr;
    unsigned long long parsed = std::strtoull(stride_env, &end, 10);
    if (end != stride_env && parsed > 0)
      stride = static_cast<uint64_t>(parsed);
  }

  std::string path = std::string_view(env) == "1" ? "/tmp/rocjitsu_decode_profile.csv" : env;
  auto *profile = new DecodeProfile(std::move(path), stride);
  decode_profile_storage() = profile;
  std::atexit([] {
    if (auto *p = decode_profile_storage())
      p->dump();
  });
  return profile;
}

DecodeProfile *decode_profile() {
  static DecodeProfile *profile = create_decode_profile();
  return profile;
}

} // namespace

ComputeUnitCore::ComputeUnitCore(std::string name, const Config &config, GpuMemory *memory,
                                 L2Cache *l2, uint32_t wf_size)
    : simdojo::CompositeComponent(std::move(name)), config_(config), memory_(memory),
      wf_size_(wf_size), decoder_(Decoder::create(config.arch)), l2_(l2), l1_scalar_(l2),
      l1_vector_(l2), lds_(config.lds_size_kb), scalar_mem_pipeline_(&l1_scalar_),
      global_mem_pipeline_(&l1_vector_, l2), local_mem_pipeline_(&lds_) {
  if (!decoder_)
    throw std::runtime_error("Unsupported architecture for ComputeUnit decoder");

  // Enable pool allocation for the hot decode-execute path.
  // Instructions decoded during step() are always deleted before the CU
  // (and its decoder) are destroyed, so pool allocation is safe here.
  decoder_->enable_pool();

  wfs_.resize(config.num_wf_slots);
  sgpr_file_.init(config.num_wf_slots * config.sgprs_per_wf, config.sgprs_per_wf);
  sgpr_to_wave_.resize(config.num_wf_slots * config.sgprs_per_wf, nullptr);

  // Completer port: CP sends dispatch activation messages here.
  cpl_ = add_port(std::make_unique<simdojo::Port>("cpl", 0, this, simdojo::PortDirection::IN,
                                                  simdojo::PortProtocol::DISPATCH));
  cpl_->set_handler([this](simdojo::Tick, simdojo::Message *) { activate(); });

  // Requester port: structural connection to shared L2 cache.
  req_ = add_port(std::make_unique<simdojo::Port>("req", 1, this, simdojo::PortDirection::OUT,
                                                  simdojo::PortProtocol::MEMORY));
}

std::unique_ptr<ComputeUnitCore> ComputeUnitCore::create(std::string name, const Config &config,
                                                         GpuMemory *memory, L2Cache *l2,
                                                         simdojo::ExecMode exec_mode) {
  // Helper: instantiate the ISA-specific CU for the given execution mode.
#define ROCJITSU_CU_CASE(ARCH_ENUM, ISA_TYPE)                                                      \
  case ARCH_ENUM:                                                                                  \
    switch (exec_mode) {                                                                           \
    case simdojo::ExecMode::FUNCTIONAL:                                                            \
      return std::make_unique<IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, ISA_TYPE>>(        \
          std::move(name), config, memory, l2);                                                    \
    case simdojo::ExecMode::CLOCKED:                                                               \
      return std::make_unique<IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, ISA_TYPE>>(           \
          std::move(name), config, memory, l2);                                                    \
    }                                                                                              \
    break

  switch (config.arch) {
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA1, cdna1::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA2, cdna2::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA3, cdna3::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_CDNA4, cdna4::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA1, rdna1::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA2, rdna2::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA3, rdna3::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA3_5, rdna3_5::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_RDNA4, rdna4::Isa);
    ROCJITSU_CU_CASE(ROCJITSU_CODE_ARCH_GFX1250, gfx1250::Isa);
  default:
    break;
  }
#undef ROCJITSU_CU_CASE
  throw std::runtime_error("Unsupported architecture for ComputeUnit");
}

Wavefront *ComputeUnitCore::dispatch_wf(uint32_t wg_id, uint64_t pc, uint32_t sgprs,
                                        uint32_t vgprs) {
  assert(wfs_.size() == config_.num_wf_slots && "wavefront slots not properly initialized");
  // Free register allocations from previously halted wavefronts before claiming
  // a new slot. This is needed so SGPR/VGPR blocks can be reused. However, we
  // must NOT reset the LDS allocator here — that would zero next_lds_alloc_
  // between WF dispatches of the same WG, causing concurrent WGs to share
  // the same LDS base. The LDS reset is handled separately by the CP.
  retire_halted_wfs_no_lds_reset();
  // Find an idle slot.
  size_t slot = config_.num_wf_slots;
  for (size_t i = 0; i < wfs_.size(); ++i) {
    if (wfs_[i]->is_halted()) {
      slot = i;
      break;
    }
  }
  if (slot >= config_.num_wf_slots)
    return nullptr;

  int32_t sgpr_base = sgpr_file_.allocate(sgprs, /*clear=*/false);
  if (sgpr_base < 0)
    return nullptr;

  int32_t vgpr_base = allocate_vgprs(vgprs);
  if (vgpr_base < 0) {
    sgpr_file_.free(static_cast<uint32_t>(sgpr_base));
    return nullptr;
  }

  // Zero the allocated register blocks so reused slots don't inherit stale
  // values from previous kernel runs.
  std::fill(&sgpr_file_[sgpr_base], &sgpr_file_[sgpr_base] + config_.sgprs_per_wf, 0u);
  std::memset(vgpr_data(static_cast<uint32_t>(vgpr_base)), 0,
              vgpr_allocation_block_size() * wf_size_ * sizeof(uint32_t));

  // Invalidate the L1 scalar cache so this wavefront reads fresh kernel
  // arguments from L2/memory rather than stale lines from a prior kernel.
  // On real hardware, the driver issues s_dcache_inv at kernel launch.
  l1_scalar_.invalidate_all();
  invalidate_inst_fetch_cache();

  auto *wf = wfs_[slot].get();
  wf->wg_id_ = wg_id;
  wf->pc = pc;
  wf->sgpr_alloc_ = {static_cast<uint32_t>(sgpr_base), sgprs};
  wf->vgpr_alloc_ = {static_cast<uint32_t>(vgpr_base), vgprs};
  wf->num_sgprs_ = sgprs;
  wf->num_vgprs_ = vgprs;
  wf->exec_ = wf_size_ == 64 ? ~0ULL : (1ULL << wf_size_) - 1;
  wf->vcc_ = 0;
  wf->m0_ = 0;
  wf->set_apertures(shared_aperture_base_, shared_aperture_limit_, private_aperture_base_,
                    private_aperture_limit_);
  wf->state_ = WfState::RUNNING;
  wf->set_ready_cycle(cycle_counter_);
  wf->trace_inst_count_ = 0;

  std::fill(sgpr_to_wave_.begin() + sgpr_base, sgpr_to_wave_.begin() + sgpr_base + sgprs, wf);
  fill_vgpr_to_wave(static_cast<uint32_t>(vgpr_base), vgpr_allocation_block_size(), wf);

  util::Logger::cp("DISPATCH_WF cu=", this->full_path(), " wf=", wf->wf_id(), " slot=", slot,
                   " pc=0x", std::hex, pc, std::dec, " wg=", wg_id, " pid=", wf->process_id());
  return wf;
}

size_t ComputeUnitCore::num_wfs() const {
  size_t count = 0;
  for (const auto &w : wfs_)
    if (w->sgpr_alloc().count > 0)
      ++count;
  return count;
}

void ComputeUnitCore::reset_all_wf() {
  invalidate_inst_fetch_cache();
  for (auto &w : wfs_) {
    if (w->sgpr_alloc().count > 0) {
      sgpr_file_.free(w->sgpr_alloc().base);
      free_vgprs(w->vgpr_alloc().base);
    }
    w->reset();
  }
}

void ComputeUnitCore::retire_halted_wfs_no_lds_reset() {
  for (auto &w : wfs_) {
    if (w->is_halted() && w->sgpr_alloc().count > 0) {
      sgpr_file_.free(w->sgpr_alloc().base);
      free_vgprs(w->vgpr_alloc().base);
      w->trace_inst_count_ = 0;
      w->reset();
    }
  }
}

void ComputeUnitCore::retire_halted_wfs() {
  for (auto &w : wfs_) {
    if (w->is_halted() && w->sgpr_alloc().count > 0) {
      sgpr_file_.free(w->sgpr_alloc().base);
      free_vgprs(w->vgpr_alloc().base);
      w->trace_inst_count_ = 0;
      w->reset();
    }
  }
  if (!has_active_wfs()) {
    reset_lds_alloc();
  }
}

void ComputeUnitCore::release_wf(uint32_t dispatch_id, uint32_t wg_id) {
  auto key = wg_key(dispatch_id, wg_id);
  auto it = active_wgs_.find(key);
  if (it != active_wgs_.end() && --it->second == 0) {
    if (plugin_hooks_enabled_)
      plugin_group_->onAmdgpuWorkgroupCompleted(dispatch_id, wg_id);
    active_wgs_.erase(it);
    if (cp_)
      cp_->notify_wg_complete(dispatch_id, wg_id);
  }
}

bool ComputeUnitCore::can_accept_workgroup(uint32_t num_wfs, uint32_t lds_bytes) const {
  // Count free wavefront slots.
  uint32_t free_slots = 0;
  for (const auto &w : wfs_)
    if (w->is_halted())
      ++free_slots;
  if (free_slots < num_wfs) {
    util::Logger::vm("CU ", this->name(), " can_accept_wg: REJECT free_slots=", free_slots,
                     " < num_wfs=", num_wfs);
    return false;
  }

  // Check SGPR register blocks.
  uint32_t free_sgpr = sgpr_file_.free_block_count();
  if (free_sgpr < num_wfs) {
    util::Logger::vm("CU ", this->name(), " can_accept_wg: REJECT free_sgpr=", free_sgpr,
                     " < num_wfs=", num_wfs);
    return false;
  }

  // Check VGPR register blocks.
  uint32_t free_vgpr = free_vgpr_blocks();
  if (free_vgpr < num_wfs) {
    util::Logger::vm("CU ", this->name(), " can_accept_wg: REJECT free_vgpr=", free_vgpr,
                     " < num_wfs=", num_wfs);
    return false;
  }

  if (lds_bytes > 0) {
    uint32_t aligned = util::align_up(lds_bytes, 256u);
    uint32_t lds_capacity_bytes = config_.lds_size_kb * 1024u;
    if (next_lds_alloc_ + aligned > lds_capacity_bytes) {
      return false;
    }
  }

  return true;
}

void ComputeUnitCore::tick_pipelines() {
  scalar_mem_pipeline_.tick();
  global_mem_pipeline_.tick();
  local_mem_pipeline_.tick();
}

void ComputeUnitCore::route_memory_inst(Instruction *inst, Wavefront &wf) {
  if (plugin_hooks_enabled_)
    plugin_group_->onAmdgpuRouteMemoryInstruction(*inst, wf);

  if (inst->data()->tag() == GLOBAL_MEM && shared_aperture_base_ != 0) {
    auto &d = *inst->data_as<VectorMemState>();
    uint64_t probe = 0;
    for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
      if (d.lane_mask & (1ULL << lane)) {
        probe = d.per_lane_addr[lane];
        break;
      }
    }
    // FLAT ops targeting the shared aperture are routed to LDS (LGKMCNT,
    // not VMCNT).  Scratch-targeting FLATs stay on the global path.
    if (probe >= shared_aperture_base_ && probe <= shared_aperture_limit_) {
      for (uint32_t lane = 0; lane < d.wf_size; ++lane) {
        if (d.lane_mask & (1ULL << lane))
          d.per_lane_addr[lane] = (d.per_lane_addr[lane] - shared_aperture_base_) + wf.lds_base();
      }
      inst->data()->set_tag(LOCAL_MEM);
      d.wait_counter_type = WaitCounterType::LGKMCNT;
      local_mem_pipeline_.issue(inst, wf);
      return;
    }
  }

  switch (inst->data()->tag()) {
  case SCALAR_MEM:
    scalar_mem_pipeline_.issue(inst, wf);
    break;
  case LOCAL_MEM:
    local_mem_pipeline_.issue(inst, wf);
    break;
  case GLOBAL_MEM:
    global_mem_pipeline_.issue(inst, wf);
    break;
  default:
    break;
  }
}

void ComputeUnitCore::update_wf_states() {
  ++cycle_counter_;

  for (auto &w : wfs_) {
    if (w->state() == WfState::WAITCNT && w->wait_satisfied()) {
      w->set_state(WfState::RUNNING);
      w->set_ready_cycle(cycle_counter_);
    } else if (w->state() == WfState::ENDING && w->wait_counters().empty()) {
      w->halt();
    }
  }

  for (auto &w : wfs_) {
    if (w->state() != WfState::BARRIER)
      continue;
    uint32_t did = w->dispatch_id();
    uint32_t wg = w->wg_id();
    bool all_at_barrier = true;
    for (auto &w2 : wfs_) {
      if (w2->dispatch_id() == did && w2->wg_id() == wg && w2->state() != WfState::HALTED &&
          w2->state() != WfState::BARRIER) {
        all_at_barrier = false;
        break;
      }
    }
    if (all_at_barrier) {
      std::vector<Wavefront *> barrier_wfs;
      for (auto &w2 : wfs_)
        if (w2->dispatch_id() == did && w2->wg_id() == wg && w2->state() == WfState::BARRIER)
          barrier_wfs.push_back(w2.get());
      if (plugin_hooks_enabled_)
        plugin_group_->onAmdgpuBarrierResolved(std::span<Wavefront *>(barrier_wfs));
      for (auto *bwf : barrier_wfs) {
        bwf->set_state(WfState::RUNNING);
        bwf->set_ready_cycle(cycle_counter_);
      }
    }
  }
}

void ComputeUnitCore::issue_instruction(Wavefront *active) {
  uint32_t vmid = active->process_id();

  rj_code_binary_inst_t words[4];
  fetch_instruction_words(active->pc, vmid, words);

  active->trace_inst_count_++;

  Instruction *inst = nullptr;
  try {
    inst = decoder_->decode(words);
  } catch (const util::InvalidInst &e) {
    util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(InvalidInst) pc=0x",
                     std::hex, active->pc, " words=[0x", words[0], ",0x", words[1], ",0x", words[2],
                     ",0x", words[3], "]", std::dec, " what=", e.what());
    active->halt();
    return;
  }
  if (!inst) {
    util::Logger::vm("CU ", this->name(), ": wf", active->wf_id(), " HALT(null decode) pc=0x",
                     std::hex, active->pc, " words=[0x", words[0], ",0x", words[1], ",0x", words[2],
                     ",0x", words[3], "]", std::dec);
    active->halt();
    return;
  }

  int inst_size_signed = inst->size();
  assert(inst_size_signed > 0 && "instruction size must be positive");
  if (auto *profile = decode_profile())
    profile->record(config_.arch, active->pc, words, inst->mnemonic(),
                    static_cast<uint32_t>(inst_size_signed));
  auto inst_size = static_cast<uint64_t>(inst_size_signed);

  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_VM)) {
    if (active->num_vgprs_ > 0) {
      util::Logger::vm([&](auto &os) {
        uint32_t vb = active->vgpr_alloc().base;
        os << std::format("{} wg[{}] wf[{}] EXECUTE #{} pc={:#x} {} sz={}", this->full_path(),
                          active->wg_id(), active->wf_id(), active->trace_inst_count_, active->pc,
                          inst->mnemonic(), inst_size);
        os << " enc=";
        for (uint64_t w = 0; w < inst_size / 4; ++w)
          os << std::format("{}{:08x}", w ? "," : "", words[w]);
        os << std::format(" scc={} vcc={:x} exec={:x}", active->read_scc(), active->vcc(),
                          active->exec());
        uint32_t nvr = std::min(active->num_vgprs_, 16u);
        for (uint32_t ln = 0; ln < active->wf_size_; ++ln) {
          os << std::format("\n[rj log VM]  PRE L{}: v[0:{}]=", ln, nvr - 1);
          for (uint32_t r = 0; r < nvr; ++r)
            os << std::format("{}{:x}", r ? "," : "", read_vgpr(vb + r, ln));
        }
      });
    }
  }

  if (plugin_hooks_enabled_)
    plugin_group_->onAmdgpuBeforeExecuteInstruction(active->pc, *inst, *active);

  {
    auto mn = std::string_view(inst->mnemonic());
    if (mn.find("s_setpc") != std::string_view::npos ||
        mn.find("s_swappc") != std::string_view::npos) {
      uint32_t ssrc0_idx = words[0] & 0x7F;
      uint32_t sb = active->sgpr_alloc().base;
      uint64_t target = static_cast<uint64_t>(read_sgpr(sb + ssrc0_idx)) |
                        (static_cast<uint64_t>(read_sgpr(sb + ssrc0_idx + 1)) << 32);
      if (target == 0) {
        active->halt();
        delete inst;
        return;
      }
    }
  }

  execute_instruction(inst, *active);
  if (plugin_hooks_enabled_)
    plugin_group_->onAmdgpuAfterExecuteInstruction(active->pc, *inst, *active);

  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_VM)) {
    if (active->num_vgprs_ > 0) {
      util::Logger::vm([&](auto &os) {
        uint32_t vb = active->vgpr_alloc().base;
        os << std::format("RESULT #{} scc={} vcc={:x} exec={:x}", active->trace_inst_count_,
                          active->read_scc(), active->vcc(), active->exec());
        uint32_t nvr = std::min(active->num_vgprs_, 16u);
        for (uint32_t ln = 0; ln < active->wf_size_; ++ln) {
          os << std::format("\n[rj log VM]  POST L{}: v[0:{}]=", ln, nvr - 1);
          for (uint32_t r = 0; r < nvr; ++r)
            os << std::format("{}{:x}", r ? "," : "", read_vgpr(vb + r, ln));
        }
      });
    }
  }

  if (inst->is_memory_op()) {
    if (inst->data() && inst->data()->tag() == GLOBAL_MEM) {
      auto *d = inst->data_as<VectorMemState>();
      d->issue_pc = active->pc;
    }
    route_memory_inst(inst, *active);
  } else
    delete inst;

  active->pc += inst_size;
}

void ComputeUnitCore::fetch_instruction_words(uint64_t pc, uint32_t vmid, uint32_t words[4]) {
  static_assert((kInstFetchCacheLines & (kInstFetchCacheLines - 1)) == 0);
  const uint64_t mixed = (pc >> 2) ^ (static_cast<uint64_t>(vmid) * 0x9e3779b97f4a7c15ULL);
  auto &line = inst_fetch_cache_[mixed & (kInstFetchCacheLines - 1)];
  if (line.valid && line.pc == pc && line.vmid == vmid) {
    std::memcpy(words, line.words.data(), sizeof(line.words));
    return;
  }

  memory_->read_block(pc, reinterpret_cast<uint8_t *>(words), sizeof(line.words), vmid);
  line.pc = pc;
  line.vmid = vmid;
  std::memcpy(line.words.data(), words, sizeof(line.words));
  line.valid = true;
}

void ComputeUnitCore::invalidate_inst_fetch_cache() {
  for (auto &line : inst_fetch_cache_)
    line.valid = false;
}

bool ComputeUnitCore::step() {
  update_wf_states();

  bool issued = false;
  for (auto &wf : wfs_) {
    if (wf->state() == WfState::RUNNING) {
      issue_instruction(wf.get());
      issued = true;
    }
  }
  if (!issued)
    retire_halted_wfs();

  ++step_count_;
  if constexpr (util::Logger::group_enabled(util::Logger::GROUP_CP)) {
    if ((step_count_ & 0xFFFFF) == 0) {
      util::Logger::cp([&](auto &os) {
        os << std::format("CU[{}] steps={}M", this->full_path(), step_count_ >> 20);
        for (auto &wf : wfs_) {
          auto st = wf->state();
          if (st == WfState::RUNNING || st == WfState::WAITCNT || st == WfState::BARRIER)
            os << std::format(" wf{}:pc={:#x}:{}", wf->wf_id(), wf->pc,
                              st == WfState::RUNNING   ? "R"
                              : st == WfState::WAITCNT ? "W"
                                                       : "B");
        }
      });
    }
  }

  return has_active_wfs();
}

// Explicit template instantiations for all AMDGPU ISAs and execution modes.
#define ROCJITSU_CU_INSTANTIATE(ISA_TYPE)                                                          \
  template class IsaExecComputeUnit<simdojo::ExecMode::FUNCTIONAL, ISA_TYPE>;                      \
  template class IsaExecComputeUnit<simdojo::ExecMode::CLOCKED, ISA_TYPE>

ROCJITSU_CU_INSTANTIATE(cdna1::Isa);
ROCJITSU_CU_INSTANTIATE(cdna2::Isa);
ROCJITSU_CU_INSTANTIATE(cdna3::Isa);
ROCJITSU_CU_INSTANTIATE(cdna4::Isa);
ROCJITSU_CU_INSTANTIATE(rdna1::Isa);
ROCJITSU_CU_INSTANTIATE(rdna2::Isa);
ROCJITSU_CU_INSTANTIATE(rdna3::Isa);
ROCJITSU_CU_INSTANTIATE(rdna3_5::Isa);
ROCJITSU_CU_INSTANTIATE(rdna4::Isa);
ROCJITSU_CU_INSTANTIATE(gfx1250::Isa);

#undef ROCJITSU_CU_INSTANTIATE

} // namespace amdgpu
} // namespace rocjitsu
