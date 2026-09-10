// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pci/gpu_pci_device.h"

#include "rocjitsu/vm/amdgpu/pci/graphics_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/interrupt_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery_profile.h"
#include "rocjitsu/vm/amdgpu/pci/memory_hub_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/mes_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/mmio_registers.h"
#include "rocjitsu/vm/amdgpu/pci/sdma_block_model.h"
#include "rocjitsu/vm/amdgpu/sdma_queue_binding_factory.h"
#include "rocjitsu/vm/soc.h"
#include "simdojo/sim/simulation.h"
#include "util/log.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <format>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <utility>

namespace rocjitsu {
namespace {

bool is_supported_width(std::size_t width) {
  return width == 1 || width == 2 || width == 4 || width == 8;
}

bool is_power_of_two(uint64_t value) { return value != 0 && (value & (value - 1)) == 0; }

simdojo::Tick add_ticks_saturated(simdojo::Tick now, simdojo::Tick delay) {
  return delay > simdojo::TICK_MAX - now ? simdojo::TICK_MAX : now + delay;
}

/// @brief Address bits carried by the low index register.
///
/// @details Its top bit is the flag the driver sets to say it is addressing
/// memory rather than a register, so the address continues in the high register
/// from bit 31 rather than bit 32.
constexpr uint32_t kIndirectLowMask = 0x7fffffff;
/// @brief MM_INDEX bit selecting the memory space rather than the register space.
constexpr uint32_t kIndirectMemorySelect = 0x80000000;

/// @brief Address bit at which the high index register begins.
constexpr unsigned kIndirectHighShift = 31;

/// @brief Where an HDP flush is issued, per NBIF version.
///
/// @details The hole is not a register of any block but an address the bus
/// reserves, and which address that is depends on the bus.
/// `nbio_v7_11_set_reg_remap` points a gfx1250-class driver at this one; another
/// NBIF version has its own remap function and its own answer, so this is keyed
/// rather than applied to whatever bus the published table happens to describe.
struct HdpFlushHole {
  uint16_t major;       ///< NBIF major version.
  uint16_t minor;       ///< NBIF minor version.
  uint16_t revision;    ///< NBIF revision.
  uint64_t byte_offset; ///< Byte offset into the register BAR.
};

constexpr HdpFlushHole kHdpFlushHoles[] = {
    {.major = 7, .minor = 11, .revision = 0, .byte_offset = 0x44000},
    {.major = 7, .minor = 11, .revision = 1, .byte_offset = 0x44000},
    {.major = 7, .minor = 11, .revision = 2, .byte_offset = 0x44000},
    {.major = 7, .minor = 11, .revision = 3, .byte_offset = 0x44000},
};

/// @brief A block the device knows how to model, and what to do without it.
///
/// @details Bound by hardware id here and by version inside the factory, which
/// is how the driver arranges it: `amdgpu_discovery_set_ih_ip_blocks` is
/// reached because the table names an OSSSYS block at all, and it then picks an
/// implementation by that block's version. Keeping the two separate is what
/// lets a version table live beside the registers it describes rather than in
/// the device.
struct BlockModelBinding {
  /// @brief Builds the model for one published record, or reports nullptr.
  using Factory = std::unique_ptr<IpBlockModel> (*)(const IpBlock &, const IpDiscoverySpec &,
                                                    IpRegisterWindow);

  IpHardwareId id; ///< Which published block this models.
  Factory make;    ///< How to build it. Never null.
  bool required;   ///< Whether being unable to model it makes the device unusable.
};

/// @details The hubs are required because a device whose flush handshakes
/// cannot be answered looks correct right up until the driver waits on one, and
/// then stalls with every register it read beforehand reporting success. The
/// interrupt block is not: a device without one boots and stays quiet, which is
/// worth having and is diagnosable from the message its factory leaves.
///
/// Both hubs bind the same model because they are the same hardware at
/// different versions; `gfxhub_v12_1` and `mmhub_v4_1_0` differ in where their
/// registers sit and in nothing else this device answers.
const BlockModelBinding kBlockModels[] = {
    {IpHardwareId::Gc,
     [](const IpBlock &block, const IpDiscoverySpec &,
        IpRegisterWindow registers) -> std::unique_ptr<IpBlockModel> {
       return MemoryHubBlockModel::create(block, std::move(registers));
     },
     true},
    {IpHardwareId::Gc,
     [](const IpBlock &block, const IpDiscoverySpec &discovery,
        IpRegisterWindow registers) -> std::unique_ptr<IpBlockModel> {
       return GraphicsBlockModel::create(block, discovery.graphics, std::move(registers));
     },
     true},
    {IpHardwareId::Gc,
     [](const IpBlock &block, const IpDiscoverySpec &,
        IpRegisterWindow registers) -> std::unique_ptr<IpBlockModel> {
       return MesBlockModel::create(block, std::move(registers));
     },
     true},
    {IpHardwareId::MmHub,
     [](const IpBlock &block, const IpDiscoverySpec &,
        IpRegisterWindow registers) -> std::unique_ptr<IpBlockModel> {
       return MemoryHubBlockModel::create(block, std::move(registers));
     },
     true},
    {IpHardwareId::OssSys,
     [](const IpBlock &block, const IpDiscoverySpec &,
        IpRegisterWindow registers) -> std::unique_ptr<IpBlockModel> {
       return InterruptBlockModel::create(block, std::move(registers));
     },
     false},
    {IpHardwareId::Sdma0,
     [](const IpBlock &block, const IpDiscoverySpec &,
        IpRegisterWindow registers) -> std::unique_ptr<IpBlockModel> {
       return SdmaBlockModel::create(block, std::move(registers));
     },
     true},
};

/// @brief The instance of each block this device models.
///
/// @details One. Every block a published table names today publishes one
/// record, and the two that could name more -- graphics, and the SDMA engine
/// derived from it -- publish identical register segments per instance, which
/// no absolute-address model can tell apart. Raising this needs a profile whose
/// instances have segments of their own, and the claim map below is what would
/// refuse one that does not.
constexpr uint8_t kModelledInstance = 0;

/// @brief GFX CP end-of-pipe interrupt identifiers consumed by KFD.
constexpr uint8_t kGraphicsInterruptClient = 0x14;
constexpr uint8_t kCpEndOfPipeSource = 181;
constexpr uint8_t kSdmaInterruptClient = 0x0a;
constexpr uint8_t kSdmaTrapSource = 49;
constexpr uint8_t kFirstComputeVmid = 1;
constexpr uint8_t kFirstXccInterruptNode = 2;

} // namespace

GpuPciDevice::GpuPciDevice(std::string name, const GpuPciDeviceSpec &spec, BarAccessTrace *trace,
                           SoC *soc)
    : simdojo::PciDevice(std::move(name), spec.id), spec_(spec), trace_(trace), soc_(soc),
      blocks_(spec_.discovery), registers_(this->name(), 0) {
  doorbell_event_.set_handler(
      [this](simdojo::Tick now, simdojo::Message *) { drain_doorbell_inbox(now); });
  doorbell_retry_event_.set_handler(
      [this](simdojo::Tick now, simdojo::Message *) { drain_doorbell_retry_inbox(now); });
  if (spec_.vram_bytes == 0) {
    // A device with no memory has nothing to present; an aperture cannot stand
    // in for a capacity that was never described.
    util::Logger::warn(std::format("{}: no video memory was configured", this->name()));
    return;
  }
  // A BAR is a power-of-two window by definition; anything else is refused here
  // rather than deeper in a transport that would reject it less clearly.
  for (const auto &[what, size] : {std::pair{"video memory", spec_.vram_aperture_bytes},
                                   std::pair{"doorbell", spec_.doorbell_aperture_bytes},
                                   std::pair{"register", spec_.register_aperture_bytes}}) {
    if (!is_power_of_two(size)) {
      util::Logger::warn(std::format("{}: the {} aperture is {} bytes, which is not a power of two",
                                     this->name(), what, size));
      return;
    }
    if (size < kMinMemoryBarBytes) {
      util::Logger::warn(
          std::format("{}: the {} aperture is {} bytes, below the {}-byte minimum for a memory BAR",
                      this->name(), what, size, kMinMemoryBarBytes));
      return;
    }
  }
  if (spec_.vram_aperture_bytes > spec_.vram_bytes) {
    util::Logger::warn(std::format("{}: the video memory aperture is larger than the memory itself",
                                   this->name()));
    return;
  }
  if (spec_.register_aperture_bytes < kMinRegisterApertureBytes) {
    util::Logger::warn(std::format(
        "{}: a register aperture of {} bytes cannot reach the registers the driver reads before "
        "discovery; at least {} are needed",
        this->name(), spec_.register_aperture_bytes, kMinRegisterApertureBytes));
    return;
  }

  // Video memory is a file so the guest can map it rather than trapping to us
  // for every access, which is the only way an aperture of this size is usable.
  // It stays sparse until written.
  // The driver reads the capacity as a count of megabytes in a 32-bit register,
  // and treats zero and all-ones as "there is no usable memory here". A capacity
  // that cannot be said in those terms would produce a device that looks fine
  // and then fails discovery.
  // The driver reads the capacity as a count of megabytes, so anything finer is
  // unrepresentable and rounds down here. That is not silently lossy: this
  // rounded value is what the device reports, and everything else it derives --
  // including where it later publishes a discovery table -- is derived from the
  // same value rather than from the true byte count, so the driver never
  // computes an address the device did not use. The remainder is simply memory
  // the guest is never told about.
  if ((spec_.vram_bytes & 0xfffffULL) != 0) {
    util::Logger::warn(std::format(
        "{}: {} bytes of video memory is not a whole number of megabytes; reporting {} MiB and "
        "leaving the remaining {} bytes unreachable by the guest",
        this->name(), spec_.vram_bytes, spec_.vram_bytes >> 20, spec_.vram_bytes & 0xfffffULL));
  }
  const uint64_t megabytes = spec_.vram_bytes >> 20;
  if (megabytes == 0 || megabytes >= std::numeric_limits<uint32_t>::max()) {
    util::Logger::warn(std::format(
        "{}: {} bytes of video memory is {} megabytes, which the driver cannot read as a size",
        this->name(), spec_.vram_bytes, megabytes));
    return;
  }
  vram_megabytes_ = static_cast<uint32_t>(megabytes);

  // The whole of memory is backed, sparsely, because the driver reaches past
  // the window through the indirect registers; only the window is shared with
  // the guest.
  vram_.emplace(this->name(), spec_.vram_bytes, spec_.vram_aperture_bytes);
  if (!vram_->usable()) {
    return;
  }
  doorbells_.resize(spec_.doorbell_aperture_bytes);
  msix_table_.resize(kMsixBarBytes);

  if (soc_ != nullptr) {
    sdma_queue_binding_factory_ = std::make_shared<amdgpu::SdmaQueueBindingFactory>(
        soc_->sdma_queue_scheduler(),
        [this](const amdgpu::SdmaQueueContext &context) { return make_sdma_callbacks(context); },
        [this](const amdgpu::SdmaQueueContext &context) {
          return make_sdma_progress_observer(context);
        });
    interrupt_subscription_ =
        amdgpu::InterruptSubscription([this](uint32_t process_id, uint32_t event_id) {
          if (process_id == 0 || process_id > std::numeric_limits<uint16_t>::max()) {
            util::Logger::warn(std::format("{}: cannot deliver a CP interrupt for PASID {}",
                                           this->name(), process_id));
            return;
          }
          if (!deliver_interrupt({.client_id = kGraphicsInterruptClient,
                                  .source_id = kCpEndOfPipeSource,
                                  .vmid = kFirstComputeVmid,
                                  .pasid = static_cast<uint16_t>(process_id),
                                  .node_id = kFirstXccInterruptNode,
                                  .data = {event_id}})) {
            util::Logger::warn(std::format("{}: could not deliver CP event {} for PASID {}",
                                           this->name(), event_id, process_id));
          }
        });
  }

  // One entry per dword of the aperture.
  registers_ = RegisterAperture(this->name(), spec_.register_aperture_bytes);
  // Built once, before anything asks a block to answer: which blocks this
  // device models follows the discovery table, and that does not change while
  // the device is alive.
  build_block_models();
  reset_registers();

  // A device that answers every pre-discovery register and then has nothing at
  // the address those answers point to is worse than one that fails to
  // construct: the driver would read a zero signature and reject it, with the
  // registers all looking correct.
  if (!publish_discovery_table()) {
    return;
  }
  // A device whose flush handshakes cannot be answered looks correct right up
  // until the driver waits on one, and then stalls with every register it read
  // beforehand reporting success. Refusing here is what turns that into a
  // message at construction.
  if (!flushes_answerable_) {
    util::Logger::warn(std::format(
        "{}: this device cannot answer the VM flush handshakes the driver waits on", this->name()));
    return;
  }
  if (soc_ != nullptr && !soc_->gpu_vm().initialize_gart_address_space()) {
    util::Logger::warn(std::format(
        "{}: the device-global VMID-0 address space could not be initialized", this->name()));
    return;
  }
  usable_ = true;
}

bool GpuPciDevice::publish_discovery_table() {
  // Guarded rather than left to pwrite returning EBADF, because this is now
  // reachable from the public reset() as well as from a device whose memory
  // never came up, and because reading and writing memory guard the same way.
  if (!vram_.has_value() || !vram_->usable()) {
    util::Logger::warn(
        std::format("{}: no video memory to publish a discovery table into", name()));
    return false;
  }
  if (spec_.discovery.blocks.empty()) {
    util::Logger::warn(std::format("{}: no discovery profile for this configuration, so a guest "
                                   "driver would find nothing to attach to",
                                   name()));
    return false;
  }
  const IpDiscoveryBuild built = build_ip_discovery_table(spec_.discovery);
  if (!built.ok()) {
    util::Logger::warn(
        std::format("{}: cannot build a discovery table: {}", name(), built.problem));
    return false;
  }
  const IpDiscoveryValidation checked = validate_ip_discovery_table(built.table);
  if (!checked.valid) {
    util::Logger::warn(std::format("{}: would publish a discovery table the driver refuses: {}",
                                   name(), checked.problem));
    return false;
  }
  if (built.table.size() > kDiscoveryTableBytes) {
    util::Logger::warn(std::format("{}: the discovery table is {} bytes, more than the {} the "
                                   "driver reads",
                                   name(), built.table.size(), kDiscoveryTableBytes));
    return false;
  }
  // The driver never learns the byte count. It reads the megabyte count out of
  // RCC_CONFIG_MEMSIZE and computes the address itself, as
  // `(vram_size << 20) - DISCOVERY_TMR_OFFSET` (amdgpu_discovery.c:1996-1997).
  // Publishing at a top-of-memory derived from the unrounded byte count would
  // therefore miss by the remainder for any capacity that is not a whole number
  // of megabytes, and the driver would read zeros and reject the signature with
  // no hint that the address was the problem. Deriving the address from the
  // value the device already committed to in reset_registers() is what keeps
  // the two from drifting; configs/gfx1151.json is one of the capacities that
  // would otherwise miss, by 446464 bytes.
  const uint64_t reported_bytes = static_cast<uint64_t>(vram_megabytes_) << 20;
  if (reported_bytes < kDiscoveryOffsetFromTopOfVram) {
    util::Logger::warn(std::format("{}: {} bytes of video memory has no room for a discovery "
                                   "table {} from the top",
                                   name(), reported_bytes, kDiscoveryOffsetFromTopOfVram));
    return false;
  }

  const uint64_t at = reported_bytes - kDiscoveryOffsetFromTopOfVram;
  if (!vram_->write_all(built.table, at)) {
    util::Logger::warn(std::format("{}: cannot store the {}-byte discovery table at {:#x}: {}",
                                   name(), built.table.size(), at, std::strerror(errno)));
    return false;
  }
  return true;
}

GpuPciDevice::~GpuPciDevice() { (void)shutdown_frontend(); }

bool GpuPciDevice::shutdown_frontend() {
  const std::lock_guard reset_lock(doorbell_reset_mutex_);
  if (frontend_shutdown_)
    return frontend_shutdown_complete_;

  close_doorbell_admission_for_reset();
  const bool queues_released = teardown_frontend_queues();
  const bool gart_cleared = soc_ == nullptr || soc_->gpu_vm().clear_gart_binding();
  if (!queues_released || !gart_cleared) {
    util::Logger::warn(
        std::format("{}: PCI frontend state was not fully released during shutdown", name()));
  }
  // Keep the route live until queue teardown has drained already-admitted
  // operations, so a terminal completion is not dropped. Then drain interrupt
  // callbacks while every field they capture is still alive.
  interrupt_subscription_.reset();
  // GpuVm snapshots retain PciPhysicalMemoryAccess objects whose local-memory
  // path points back into this device. Revoke the captured transport generation
  // and drain admitted operations before any device storage is destroyed. A
  // stale snapshot then fails its session lease without dereferencing this.
  shutdown_transport();

  // The models are frontend-owned, but their routes into the core are borrowed.
  // Sever those routes while the core is known to be alive; the topology may
  // destroy that core before it later destroys this sibling PCI component.
  for (const OwnedBlockModel &owned : models_) {
    if (auto *hub = dynamic_cast<MemoryHubBlockModel *>(owned.model_.get()); hub != nullptr)
      hub->attach_soc(nullptr);
  }
  if (mes_ != nullptr) {
    if (!mes_->detach_engine())
      util::Logger::warn(std::format("{}: MES frontend callbacks could not be detached", name()));
  }
  if (sdma_ != nullptr) {
    sdma_->attach_queue_binding_factory(nullptr);
    sdma_->attach_soc(nullptr);
  }
  interrupts_ = nullptr;
  mes_ = nullptr;
  sdma_ = nullptr;
  soc_ = nullptr;
  frontend_shutdown_complete_ = queues_released && gart_cleared;
  frontend_shutdown_ = true;
  return frontend_shutdown_complete_;
}

std::vector<simdojo::BarSpec> GpuPciDevice::bars() const {
  simdojo::BarSpec vram;
  vram.index = kVramBar;
  vram.size = spec_.vram_aperture_bytes;
  vram.mem = true;
  vram.prefetch = true;
  vram.is_64bit = true;
  vram.backing_fd = vram_.has_value() ? vram_->fd() : -1;
  if (vram.backing_fd >= 0) {
    vram.mmap_areas.push_back({.offset = 0, .length = spec_.vram_aperture_bytes});
  }

  simdojo::BarSpec doorbell;
  doorbell.index = kDoorbellBar;
  doorbell.size = spec_.doorbell_aperture_bytes;
  doorbell.mem = true;
  doorbell.prefetch = true;
  doorbell.is_64bit = true;

  // Registers always trap: a doorbell can be a plain store to memory we scan,
  // but a register read has to be answered by the model.
  simdojo::BarSpec registers;
  registers.index = kRegisterBar;
  registers.size = spec_.register_aperture_bytes;
  registers.mem = true;

  // Trapped rather than mapped, so every access is seen. What this stage does
  // with what it sees is nothing beyond storing it: the table and the pending
  // bits are one undifferentiated buffer, the device is told nothing about
  // which vectors the guest has masked, and it sets no pending bit of its own.
  // That is enough for a client that emulates the table itself, which is what
  // this transport's clients do. Modelling ownership properly -- the device
  // setting pending bits for a masked vector and learning mask changes through
  // a state callback -- needs an interface this device does not have yet, and
  // trapping is the precondition for adding it without changing the bus shape.
  simdojo::BarSpec msix;
  msix.index = kMsixBar;
  msix.size = kMsixBarBytes;
  msix.mem = true;

  return {vram, doorbell, registers, msix};
}

void GpuPciDevice::reset_registers() {
  // The registers the driver reads before it can locate the IP discovery table.
  // Reporting no memory, or all-ones, makes it give up before it starts.
  registers_.define(byte_offset_of(MmioRegister::RccConfigMemsize), vram_megabytes_);
  registers_.define(byte_offset_of(MmioRegister::Mp0SmnC2pmsg33), kFirmwareInitDoneBit);
  // Zero says this is a physical function with virtualization disabled, which is
  // what lets the driver treat the device as passed through to it whole. The
  // alternative would be to model the SR-IOV mailbox a virtual function reaches
  // its host through, which this device does not have.
  registers_.define(byte_offset_of(MmioRegister::RccIovFuncIdentifier), 0);
  // Zero here tells the driver the discovery table is not published through
  // these registers, so it looks for it at the top of video memory instead.
  registers_.define(byte_offset_of(MmioRegister::DriverScratch0), 0);
  registers_.define(byte_offset_of(MmioRegister::DriverScratch1), 0);
  registers_.define(byte_offset_of(MmioRegister::DriverScratch2), 0);
  // The indirect window and its data register are how memory outside the
  // aperture is reached, so they are modelled rather than reading as absent.
  registers_.define(byte_offset_of(MmioRegister::MmIndex), 0);
  registers_.define(byte_offset_of(MmioRegister::MmIndexHi), 0);
  registers_.define(byte_offset_of(MmioRegister::MmData), 0);
  // Readable before discovery, like the registers above: it is inside the
  // pre-discovery aperture and is named, so leaving it undefined would report it
  // as a register this device does not model when in fact it has an answer.
  registers_.define(byte_offset_of(MmioRegister::IpDiscoveryVersion), kIpDiscoveryVersion);

  // Accepting the flush is the whole model; the driver orders it with a read of
  // a different register and never reads this one back. Which address the flush
  // is issued at follows the bus, so it comes from the NBIF version the table
  // declares; a bus with no known hole simply has none answered, and the driver
  // for it would be reaching somewhere this device has not modelled.
  if (const IpBlock *nbif = blocks_.find(IpHardwareId::Nbif, kModelledInstance); nbif != nullptr) {
    bool holed = false;
    for (const HdpFlushHole &hole : kHdpFlushHoles) {
      if (hole.major == nbif->major && hole.minor == nbif->minor &&
          hole.revision == nbif->revision) {
        registers_.define(hole.byte_offset, 0);
        holed = true;
        break;
      }
    }
    if (!holed) {
      util::Logger::warn(std::format(
          "{}: no HDP flush hole is known for NBIF {}.{}.{}, so a flush issued through the bus "
          "remap will not be answered",
          name(), nbif->major, nbif->minor, nbif->revision));
    }
  }

  // Every block this device models, back to its power-on values.
  //
  // A hub that cannot reach its own registers is what makes the whole device
  // unusable: a flush is issued and then waited for, and nothing else reports
  // it finishing, so an unanswered acknowledge is not a quiet gap in the
  // register model but a stall of the driver's full timeout, once per flush.
  // Whether the model exists at all was settled when the models were built.
  flushes_answerable_ = models_complete_;
  for (const OwnedBlockModel &owned : models_) {
    if (!owned.model_->reset() && owned.required_) {
      flushes_answerable_ = false;
    }
  }
}

void GpuPciDevice::build_block_models() {
  std::vector<ClaimedRegisters> claimed;
  models_complete_ = true;
  for (const BlockModelBinding &binding : kBlockModels) {
    const IpBlock *block = blocks_.find(binding.id, kModelledInstance);
    if (block == nullptr) {
      // A block the table does not name has no registers to answer, and the
      // driver will not look for them either.
      continue;
    }
    const IpRegisterWindow window(registers_, block->register_bases, name());
    std::unique_ptr<IpBlockModel> model = binding.make(*block, spec_.discovery, window);
    if (model == nullptr) {
      // The version has no known layout, and the factory has said so.
      if (binding.required) {
        models_complete_ = false;
      }
      continue;
    }

    // Resolved here rather than inside the model, which is the point of a claim:
    // a model names registers the way its own offset header names them and never
    // learns what byte that is. A claim the aperture cannot reach is left out
    // rather than refused -- the model's own reset reports that, and reports it
    // per register.
    for (const RegisterClaim &claim : model->claims()) {
      const std::optional<uint64_t> at = window.resolve(claim.segment_index, claim.first_dword);
      if (at.has_value()) {
        claimed.push_back(
            {.owner = model->what(), .first_dword = dword_index_of(*at), .count = claim.count});
      }
    }

    // The interrupt block is the one model the device itself talks to, because
    // delivering an interrupt is something the device is asked to do from
    // outside. Recognised by its type rather than by its binding, so the two
    // cannot be wired to different blocks by a later edit.
    if (auto *interrupts = dynamic_cast<InterruptBlockModel *>(model.get());
        interrupts != nullptr) {
      interrupts_ = interrupts;
    }
    if (auto *mes = dynamic_cast<MesBlockModel *>(model.get()); mes != nullptr) {
      if (soc_ != nullptr && !mes->attach_engine(soc_->mes_engine(), sdma_queue_binding_factory_,
                                                 interrupt_subscription_.sink())) {
        util::Logger::warn(std::format("{}: cannot attach the MES core engine", name()));
        models_complete_ = false;
      }
      mes_ = mes;
    }
    if (auto *hub = dynamic_cast<MemoryHubBlockModel *>(model.get()); hub != nullptr) {
      hub->attach_soc(soc_);
    }
    if (auto *sdma = dynamic_cast<SdmaBlockModel *>(model.get()); sdma != nullptr) {
      sdma->attach_soc(soc_);
      sdma->attach_queue_binding_factory(sdma_queue_binding_factory_);
      sdma_ = sdma;
    }
    models_.emplace_back(std::move(model), binding.required);
  }

  // One absolute-dword map over every claim, checked once. Blocks legitimately
  // share register segments on this family -- GC and SDMA0 publish identical
  // bases, as do the two management processors -- and stay apart only in the
  // offsets they claim within them, so an overlap is never a shared segment
  // showing through; it is two models about to answer one register, with
  // whichever defined it last winning silently.
  if (const std::string overlap = overlapping_claim(std::move(claimed)); !overlap.empty()) {
    util::Logger::warn(std::format("{}: {}", name(), overlap));
    models_complete_ = false;
  }
}

bool GpuPciDevice::teardown_frontend_queues() {
  // Destroy MES-owned queues first so GpuQueueRegistry drains their admitted
  // queue bindings and releases their GpuVm references. Independently tear
  // down register-backed SDMA queues even if MES could not release every
  // process binding; either adapter may retain callbacks into this frontend.
  const bool mes_complete = mes_ == nullptr || mes_->teardown_queues();
  if (sdma_ != nullptr)
    sdma_->teardown_queues();
  return mes_complete;
}

GpuPciDevice::SdmaPciEffectResult
GpuPciDevice::SdmaPciCallbackState::request(const std::shared_ptr<SdmaPciCallbackState> &self,
                                            const SdmaPciEffect &effect) {
  const std::lock_guard lock(mutex_);
  if (cancelled_)
    return {};
  if (pending_) {
    if (*pending_ != effect)
      return {.outcome = amdgpu::VmAccessOutcome::Malformed};
    if (!result_)
      return {.outcome = amdgpu::VmAccessOutcome::Unavailable};
    const SdmaPciEffectResult result = *result_;
    pending_.reset();
    result_.reset();
    return result;
  }
  if (!enqueue_)
    return {};
  pending_ = effect;
  if (!enqueue_(self, effect)) {
    pending_.reset();
    return {};
  }
  return {.outcome = amdgpu::VmAccessOutcome::Unavailable};
}

void GpuPciDevice::SdmaPciCallbackState::complete(const SdmaPciEffect &effect,
                                                  SdmaPciEffectResult result) {
  const std::lock_guard lock(mutex_);
  if (!cancelled_ && pending_ && *pending_ == effect && !result_)
    result_ = result;
}

void GpuPciDevice::SdmaPciCallbackState::cancel() {
  const std::lock_guard lock(mutex_);
  cancelled_ = true;
  enqueue_ = {};
  pending_.reset();
  result_.reset();
}

bool GpuPciDevice::enqueue_sdma_pci_effect(const std::shared_ptr<SdmaPciCallbackState> &state,
                                           const SdmaPciEffect &effect) {
  {
    const std::lock_guard lock(sdma_effect_mutex_);
    if (!sdma_effect_admission_open_)
      return false;
    sdma_effect_inbox_.emplace_back(state, effect, state->reset_epoch_);
  }
  if (engine() != nullptr)
    engine()->schedule_event_now(&doorbell_event_);
  return true;
}

void GpuPciDevice::drain_sdma_pci_effects() {
  std::deque<SdmaPciEffectRequest> requests;
  std::deque<SdmaQueueProgressRequest> progress_updates;
  {
    const std::lock_guard lock(sdma_effect_mutex_);
    requests.swap(sdma_effect_inbox_);
    progress_updates.swap(sdma_progress_inbox_);
  }

  for (const SdmaPciEffectRequest &request : requests) {
    const std::shared_ptr<SdmaPciCallbackState> state = request.state_.lock();
    if (state == nullptr)
      continue;
    {
      const std::lock_guard lock(sdma_effect_mutex_);
      last_sdma_effect_thread_ = std::this_thread::get_id();
      ++sdma_effect_count_;
    }
    std::shared_ptr<simdojo::PciTransportSession> session;
    uint64_t generation = 0;
    {
      const std::lock_guard state_lock(state->mutex_);
      if (state->cancelled_ || !state->pending_ || *state->pending_ != request.effect_ ||
          state->result_) {
        continue;
      }
      session = state->session_.lock();
      generation = state->generation_;
    }

    SdmaPciEffectResult result;
    simdojo::PciTransportSession::OperationLease lease =
        session != nullptr ? session->acquire() : simdojo::PciTransportSession::OperationLease{};
    FrontendOperationLease frontend = acquire_frontend_operation(request.reset_epoch_);
    if (frontend && lease && session->generation() == generation &&
        transport_session() == session) {
      std::function<void()> admitted_hook;
      {
        const std::lock_guard lock(sdma_effect_mutex_);
        admitted_hook = sdma_effect_admitted_hook_for_test_;
      }
      if (admitted_hook)
        admitted_hook();
      {
        const std::lock_guard state_lock(state->mutex_);
        if (state->cancelled_ || !state->pending_ || *state->pending_ != request.effect_ ||
            state->result_) {
          continue;
        }
      }
      switch (request.effect_.kind) {
      case SdmaPciEffectKind::ReadRegister: {
        uint32_t value = 0;
        if (!read_register(request.effect_.address, value)) {
          result.outcome = amdgpu::VmAccessOutcome::Faulted;
          break;
        }
        result.outcome = amdgpu::VmAccessOutcome::Complete;
        result.value = value;
        break;
      }
      case SdmaPciEffectKind::WriteRegister:
        result.outcome = write_register(request.effect_.address, request.effect_.value)
                             ? amdgpu::VmAccessOutcome::Complete
                             : amdgpu::VmAccessOutcome::Faulted;
        break;
      case SdmaPciEffectKind::DeliverInterrupt: {
        const uint32_t process_id = request.effect_.process_id;
        const uint32_t engine_id = request.effect_.engine_id;
        if (process_id > UINT16_MAX) {
          result.outcome = amdgpu::VmAccessOutcome::Malformed;
          break;
        }
        const bool delivered = deliver_interrupt({.client_id = kSdmaInterruptClient,
                                                  .source_id = kSdmaTrapSource,
                                                  .ring_id = static_cast<uint8_t>(engine_id << 4),
                                                  .vmid = process_id == 0 ? uint8_t{0} : uint8_t{1},
                                                  .pasid = static_cast<uint16_t>(process_id),
                                                  .node_id = kFirstXccInterruptNode,
                                                  .data = {request.effect_.value}});
        result.outcome =
            delivered ? amdgpu::VmAccessOutcome::Complete : amdgpu::VmAccessOutcome::Faulted;
        break;
      }
      }
    }
    state->complete(request.effect_, result);
  }

  for (const SdmaQueueProgressRequest &progress : progress_updates) {
    const std::shared_ptr<simdojo::PciTransportSession> session = progress.session.lock();
    if (session == nullptr || session->generation() != progress.generation)
      continue;
    simdojo::PciTransportSession::OperationLease lease = session->acquire();
    FrontendOperationLease frontend = acquire_frontend_operation(progress.reset_epoch);
    if (!lease || transport_session() != session || !frontend || sdma_ == nullptr)
      continue;
    if (progress.queue_id != (0xffff0000u | progress.engine_id))
      continue;
    sdma_->update_queue_progress(progress.engine_id, progress.consumer_cursor, progress.terminal);
  }
}

std::thread::id GpuPciDevice::last_sdma_pci_effect_thread_for_test() {
  const std::lock_guard lock(sdma_effect_mutex_);
  return last_sdma_effect_thread_;
}

uint64_t GpuPciDevice::sdma_pci_effect_count_for_test() {
  const std::lock_guard lock(sdma_effect_mutex_);
  return sdma_effect_count_;
}

void GpuPciDevice::set_sdma_pci_effect_admitted_hook_for_test(std::function<void()> hook) {
  const std::lock_guard lock(sdma_effect_mutex_);
  sdma_effect_admitted_hook_for_test_ = std::move(hook);
}

void GpuPciDevice::cancel_sdma_pci_effects() {
  std::vector<std::shared_ptr<SdmaPciCallbackState>> states;
  {
    const std::lock_guard lock(sdma_effect_mutex_);
    sdma_effect_admission_open_ = false;
    sdma_effect_inbox_.clear();
    sdma_progress_inbox_.clear();
    for (const std::weak_ptr<SdmaPciCallbackState> &weak_state : sdma_callback_states_)
      if (std::shared_ptr<SdmaPciCallbackState> state = weak_state.lock())
        states.push_back(std::move(state));
    sdma_callback_states_.clear();
  }
  for (const std::shared_ptr<SdmaPciCallbackState> &state : states)
    state->cancel();
}

amdgpu::SdmaPacketCallbacks
GpuPciDevice::make_sdma_callbacks(const amdgpu::SdmaQueueContext &context) {
  const std::shared_ptr<simdojo::PciTransportSession> session = transport_session();
  std::shared_ptr<SdmaPciCallbackState> state = std::make_shared<SdmaPciCallbackState>();
  state->session_ = session;
  state->generation_ = session != nullptr ? session->generation() : 0;
  {
    const std::lock_guard lock(doorbell_inbox_mutex_);
    state->reset_epoch_ = doorbell_reset_epoch_;
  }
  state->enqueue_ = [this](const std::shared_ptr<SdmaPciCallbackState> &callback_state,
                           const SdmaPciEffect &effect) {
    return enqueue_sdma_pci_effect(callback_state, effect);
  };
  {
    const std::lock_guard lock(sdma_effect_mutex_);
    if (!sdma_effect_admission_open_)
      state->cancelled_ = true;
    std::erase_if(sdma_callback_states_, [](const std::weak_ptr<SdmaPciCallbackState> &candidate) {
      return candidate.expired();
    });
    sdma_callback_states_.push_back(state);
  }

  amdgpu::SdmaPacketCallbacks callbacks;
  callbacks.read_register = [state](uint32_t address) {
    const SdmaPciEffectResult result =
        state->request(state, {.kind = SdmaPciEffectKind::ReadRegister, .address = address});
    return amdgpu::SdmaPacketRegisterReadResult{.outcome = result.outcome, .value = result.value};
  };
  callbacks.write_register = [state](uint32_t address, uint32_t value) {
    return state
        ->request(state,
                  {.kind = SdmaPciEffectKind::WriteRegister, .address = address, .value = value})
        .outcome;
  };
  callbacks.deliver_interrupt = [state, process_id = context.process_id,
                                 engine_id = context.engine_id](uint32_t data) {
    return state
        ->request(state, {.kind = SdmaPciEffectKind::DeliverInterrupt,
                          .value = data,
                          .process_id = process_id,
                          .engine_id = engine_id})
        .outcome;
  };
  callbacks.timestamp = [] {
    const std::chrono::steady_clock::duration now =
        std::chrono::steady_clock::now().time_since_epoch();
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(now).count());
  };
  return callbacks;
}

std::function<void(const amdgpu::SdmaQueueProgress &)>
GpuPciDevice::make_sdma_progress_observer(const amdgpu::SdmaQueueContext &context) {
  // MES-created queues share this binding factory but do not have an SDMA register
  // shadow. Their consumer pointer is already published through GPU memory.
  if (context.queue_id != (0xffff0000u | context.engine_id))
    return {};
  const std::shared_ptr<simdojo::PciTransportSession> session = transport_session();
  const uint64_t generation = session != nullptr ? session->generation() : 0;
  uint64_t reset_epoch = 0;
  {
    const std::lock_guard lock(doorbell_inbox_mutex_);
    reset_epoch = doorbell_reset_epoch_;
  }
  return [this, weak_session = std::weak_ptr<simdojo::PciTransportSession>(session), generation,
          reset_epoch, queue_id = context.queue_id,
          engine_id = context.engine_id](const amdgpu::SdmaQueueProgress &progress) {
    {
      const std::lock_guard lock(sdma_effect_mutex_);
      if (!sdma_effect_admission_open_)
        return;
      sdma_progress_inbox_.push_back({.session = weak_session,
                                      .generation = generation,
                                      .reset_epoch = reset_epoch,
                                      .queue_id = queue_id,
                                      .engine_id = engine_id,
                                      .consumer_cursor = progress.consumer_cursor,
                                      .terminal = progress.terminal});
    }
    if (engine() != nullptr)
      engine()->schedule_event_now(&doorbell_event_);
  };
}

bool GpuPciDevice::deliver_interrupt(const InterruptEntry &entry) {
  if (interrupts_ == nullptr) {
    return false;
  }
  // Taken once and used for the whole delivery. Re-reading it per use would make
  // every step its own chance to observe a detach, and a delivery interrupted
  // between the ring write and the pointer write leaves an entry the driver is
  // never told about -- worse than declining before anything was written.
  simdojo::PciTransportSession::OperationLease transport = acquire_transport();
  if (!transport || transport.dma() == nullptr || transport.irq() == nullptr) {
    return false;
  }
  const InterruptRing programmed = interrupts_->ring();
  std::optional<uint32_t> consumed;
  {
    const std::lock_guard lock(doorbell_storage_mutex_);
    consumed = interrupts_->read_pointer(doorbells_, programmed);
  }
  return interrupts_->deliver(entry, programmed, consumed, *transport.dma(), *transport.irq());
}

InterruptRing GpuPciDevice::interrupt_ring() const {
  // A published table naming no interrupt block this device knows how to model
  // has said nothing about a ring, which is what a default one reports.
  return interrupts_ == nullptr ? InterruptRing{} : interrupts_->ring();
}

uint64_t GpuPciDevice::indirect_address() const {
  const uint32_t low = registers_.value(byte_offset_of(MmioRegister::MmIndex));
  const uint32_t high = registers_.value(byte_offset_of(MmioRegister::MmIndexHi));
  return (static_cast<uint64_t>(low) & kIndirectLowMask) |
         (static_cast<uint64_t>(high) << kIndirectHighShift);
}

int64_t GpuPciDevice::access_registers(std::span<std::byte> buf, uint64_t offset, bool write) {
  // Registers are 32 bits wide and naturally aligned; anything else is a driver
  // bug or a transport bug, and answering it would hide which. A device that
  // refused its own configuration has no aperture at all, and holds() says so.
  if (buf.size() != kRegisterBytes || !registers_.holds(offset)) {
    return -1;
  }

  const bool modelled = registers_.modelled(offset);
  if (trace_ != nullptr) {
    trace_->record(kRegisterBar, offset, buf.size(), write, modelled);
  }

  // Memory outside the aperture is reached by pointing the index registers at
  // an address and then reading or writing the data register.
  if (offset == byte_offset_of(MmioRegister::MmData)) {
    // Bit 31 of MM_INDEX selects which space the window addresses. This stage
    // models the memory side only, so a request for the register side is
    // refused rather than quietly answered out of the framebuffer -- which
    // would hand the driver bytes from an unrelated address and look like
    // working hardware.
    if ((registers_.value(byte_offset_of(MmioRegister::MmIndex)) & kIndirectMemorySelect) == 0) {
      if (trace_ != nullptr) {
        trace_->record_rejected(kRegisterBar, offset, buf.size(), write);
      }
      return -1;
    }
    uint32_t value = 0;
    if (write) {
      std::memcpy(&value, buf.data(), sizeof(value));
      (void)vram_->write(indirect_address(), value);
    } else {
      (void)vram_->read(indirect_address(), value);
      std::memcpy(buf.data(), &value, sizeof(value));
    }
    return 4;
  }

  if (write) {
    // A write to something the device does not model is dropped rather than
    // remembered, so a later read still reports absent hardware instead of
    // echoing back whatever the driver put there. A read-only register keeps
    // its value for the same reason: what it reports is a property of the
    // hardware, not state the driver owns.
    if (modelled && !registers_.read_only(offset)) {
      uint32_t value = 0;
      std::memcpy(&value, buf.data(), sizeof(value));
      (void)registers_.store(offset, value);
      // The blocks learn what the driver said from the registers themselves,
      // after the write has landed. Told where rather than what, because a model
      // that was handed the value would be deciding what a field means twice --
      // once here and once when it reads the register back.
      if (interrupts_ != nullptr) {
        interrupts_->observe_write(offset);
      }
      for (const OwnedBlockModel &owned : models_) {
        owned.model_->observe_register_write(offset, *this);
      }
    }
    return 4;
  }

  const uint32_t value = modelled ? registers_.value(offset) : 0;
  std::memcpy(buf.data(), &value, sizeof(value));
  return 4;
}

int64_t GpuPciDevice::access_memory(std::span<std::byte> buf, uint64_t offset, bool write,
                                    std::span<std::byte> backing) {
  if (offset > backing.size() || buf.size() > backing.size() - offset) {
    return -1;
  }
  const auto begin = backing.begin() + static_cast<std::ptrdiff_t>(offset);
  if (write) {
    std::ranges::copy(buf, begin);
  } else {
    std::copy_n(begin, buf.size(), buf.begin());
  }
  return static_cast<int64_t>(buf.size());
}

int64_t GpuPciDevice::bar_access(int bar, std::span<std::byte> buf, uint64_t offset, bool write) {
  if (!is_supported_width(buf.size())) {
    // Deliberately not traced. The trace answers "which registers does this
    // device still not model", and an unsupported width is a malformed access
    // rather than missing hardware: recording it as unmodelled would put an
    // address into that report that may be modelled perfectly well, and send
    // whoever reads it looking for a register that is not the problem.
    return -1;
  }

  switch (bar) {
  case kRegisterBar:
    return access_registers(buf, offset, write);
  case kDoorbellBar: {
    int64_t result = 0;
    {
      const std::lock_guard lock(doorbell_storage_mutex_);
      result = access_memory(buf, offset, write, doorbells_);
    }
    if (result < 0 || !write) {
      return result;
    } else {
      uint64_t value = 0;
      std::memcpy(&value, buf.data(), buf.size());
      enqueue_doorbell_notification(offset, value, buf.size());
      return result;
    }
  }
  case kMsixBar:
    return access_memory(buf, offset, write, msix_table_);
  case kVramBar:
    // Reached only for a guest that did not map the aperture; a mapped one
    // never traps here.
    if (!vram_.has_value()) {
      return -1;
    }
    if (write) {
      return vram_->write_all(std::span<const std::byte>(buf), offset)
                 ? static_cast<int64_t>(buf.size())
                 : -1;
    }
    return vram_->read_all(buf, offset) ? static_cast<int64_t>(buf.size()) : -1;
  default:
    break;
  }

  if (trace_ != nullptr) {
    trace_->record_rejected(bar, offset, buf.size(), write);
  }
  return -1;
}

void GpuPciDevice::dma_map(const simdojo::DmaRegion & /*region*/) {}

void GpuPciDevice::dma_unmap(const simdojo::DmaRegion & /*region*/) {}

void GpuPciDevice::enqueue_doorbell_notification(uint64_t byte_offset, uint64_t value,
                                                 std::size_t width) {
  const std::shared_ptr<simdojo::PciTransportSession> session = transport_session();
  bool wake = false;
  {
    const std::lock_guard lock(doorbell_inbox_mutex_);
    if (!doorbell_admission_open_)
      return;
    doorbell_inbox_.emplace_back(session, session != nullptr ? session->generation() : 0,
                                 doorbell_reset_epoch_, byte_offset, value, width);
    if (engine() != nullptr && !doorbell_wake_pending_) {
      doorbell_wake_pending_ = true;
      wake = true;
    }
  }
  if (wake)
    engine()->schedule_event_now(&doorbell_event_);
}

GpuPciDevice::FrontendOperationLease::~FrontendOperationLease() {
  if (device_ != nullptr)
    device_->finish_frontend_operation();
}

GpuPciDevice::FrontendOperationLease::FrontendOperationLease(
    FrontendOperationLease &&other) noexcept
    : device_(std::exchange(other.device_, nullptr)) {}

GpuPciDevice::FrontendOperationLease &
GpuPciDevice::FrontendOperationLease::operator=(FrontendOperationLease &&other) noexcept {
  if (this == &other)
    return *this;
  if (device_ != nullptr)
    device_->finish_frontend_operation();
  device_ = std::exchange(other.device_, nullptr);
  return *this;
}

GpuPciDevice::FrontendOperationLease
GpuPciDevice::acquire_frontend_operation(uint64_t reset_epoch) {
  const std::lock_guard lock(doorbell_inbox_mutex_);
  if (!doorbell_admission_open_ || reset_epoch != doorbell_reset_epoch_)
    return {};
  ++active_frontend_operations_;
  return FrontendOperationLease(this);
}

void GpuPciDevice::finish_frontend_operation() {
  const std::lock_guard lock(doorbell_inbox_mutex_);
  if (active_frontend_operations_ == 0)
    return;
  --active_frontend_operations_;
  if (active_frontend_operations_ == 0)
    doorbell_idle_.notify_all();
}

bool GpuPciDevice::same_doorbell_retry_target(const DoorbellRetry &retry,
                                              const DoorbellNotification &notification,
                                              const IpBlockModel *model) {
  return retry.model_ == model && retry.notification_.generation_ == notification.generation_ &&
         retry.notification_.reset_epoch_ == notification.reset_epoch_ &&
         retry.notification_.byte_offset_ == notification.byte_offset_;
}

void GpuPciDevice::enqueue_doorbell_retry(const DoorbellNotification &notification,
                                          IpBlockModel *model, simdojo::Tick now,
                                          simdojo::Tick backoff) {
  if (model == nullptr)
    return;
  if (now == simdojo::TICK_MAX)
    return;
  const simdojo::Tick bounded_backoff = std::min(backoff, kMaximumDoorbellRetryBackoff);
  const simdojo::Tick ready_tick = add_ticks_saturated(now, bounded_backoff);
  simdojo::Tick scheduled_tick = ready_tick;
  bool wake = false;
  {
    const std::lock_guard lock(doorbell_inbox_mutex_);
    if (!doorbell_admission_open_ || notification.reset_epoch_ != doorbell_reset_epoch_)
      return;
    const std::deque<DoorbellRetry>::iterator duplicate =
        std::ranges::find_if(doorbell_retry_inbox_, [&](const DoorbellRetry &retry) {
          return same_doorbell_retry_target(retry, notification, model);
        });
    simdojo::Tick candidate_tick = ready_tick;
    if (duplicate != doorbell_retry_inbox_.end()) {
      duplicate->notification_.value_ =
          std::max(duplicate->notification_.value_, notification.value_);
      duplicate->notification_.width_ = notification.width_;
      duplicate->backoff_ = std::min(duplicate->backoff_, bounded_backoff);
      duplicate->ready_tick_ = std::min(duplicate->ready_tick_, ready_tick);
      candidate_tick = duplicate->ready_tick_;
    } else {
      doorbell_retry_inbox_.emplace_back(notification, model, bounded_backoff, ready_tick);
    }
    if (engine() != nullptr &&
        (!doorbell_retry_wake_tick_ || candidate_tick < *doorbell_retry_wake_tick_)) {
      doorbell_retry_wake_tick_ = candidate_tick;
      scheduled_tick = candidate_tick;
      wake = true;
    }
  }
  if (wake)
    schedule_event(&doorbell_retry_event_, scheduled_tick);
}

void GpuPciDevice::discard_doorbell_retry(const DoorbellNotification &notification,
                                          IpBlockModel *model) {
  const std::lock_guard lock(doorbell_inbox_mutex_);
  std::erase_if(doorbell_retry_inbox_, [&](const DoorbellRetry &retry) {
    return same_doorbell_retry_target(retry, notification, model);
  });
}

void GpuPciDevice::drain_doorbell_inbox(simdojo::Tick now) {
  drain_sdma_pci_effects();
  std::deque<DoorbellNotification> notifications;
  {
    const std::lock_guard lock(doorbell_inbox_mutex_);
    notifications.swap(doorbell_inbox_);
    doorbell_wake_pending_ = false;
  }

  for (const DoorbellNotification &notification : notifications) {
    const std::shared_ptr<simdojo::PciTransportSession> session = notification.session_.lock();
    if (session == nullptr || session->generation() != notification.generation_)
      continue;
    simdojo::PciTransportSession::OperationLease lease = session->acquire();
    FrontendOperationLease frontend = acquire_frontend_operation(notification.reset_epoch_);
    if (!lease || transport_session() != session || !frontend)
      continue;

    try {
      for (const OwnedBlockModel &owned : models_) {
        const DoorbellDisposition disposition = owned.model_->observe_doorbell_write(
            notification.byte_offset_, notification.value_, notification.width_, *this);
        if (disposition == DoorbellDisposition::Retry) {
          enqueue_doorbell_retry(notification, owned.model_.get(), now, 1);
        } else if (disposition != DoorbellDisposition::Ignored) {
          discard_doorbell_retry(notification, owned.model_.get());
        }
      }
    } catch (const std::exception &error) {
      util::Logger::warn(
          std::format("{}: deferred doorbell handling failed: {}", name(), error.what()));
    } catch (...) {
      util::Logger::warn(std::format("{}: deferred doorbell handling failed", name()));
    }
  }
}

void GpuPciDevice::drain_doorbell_retry_inbox(simdojo::Tick now, bool ignore_ready) {
  drain_sdma_pci_effects();
  std::deque<DoorbellRetry> retries;
  {
    const std::lock_guard lock(doorbell_inbox_mutex_);
    if (doorbell_retry_wake_tick_ && *doorbell_retry_wake_tick_ <= now)
      doorbell_retry_wake_tick_.reset();
    retries.swap(doorbell_retry_inbox_);
  }

  std::deque<DoorbellRetry> deferred;
  for (DoorbellRetry &retry : retries) {
    if (!ignore_ready && retry.ready_tick_ > now) {
      deferred.push_back(std::move(retry));
      continue;
    }

    const DoorbellNotification &notification = retry.notification_;
    const std::shared_ptr<simdojo::PciTransportSession> session = notification.session_.lock();
    if (session == nullptr || session->generation() != notification.generation_)
      continue;
    simdojo::PciTransportSession::OperationLease lease = session->acquire();
    FrontendOperationLease frontend = acquire_frontend_operation(notification.reset_epoch_);
    if (!lease || transport_session() != session || !frontend)
      continue;

    DoorbellDisposition disposition = DoorbellDisposition::Faulted;
    try {
      disposition = retry.model_->observe_doorbell_write(
          notification.byte_offset_, notification.value_, notification.width_, *this);
    } catch (const std::exception &error) {
      util::Logger::warn(
          std::format("{}: deferred doorbell retry failed: {}", name(), error.what()));
    } catch (...) {
      util::Logger::warn(std::format("{}: deferred doorbell retry failed", name()));
    }
    if (disposition == DoorbellDisposition::Retry) {
      if (now == simdojo::TICK_MAX)
        continue;
      retry.backoff_ = std::min(retry.backoff_ * 2, kMaximumDoorbellRetryBackoff);
      retry.ready_tick_ = add_ticks_saturated(now, retry.backoff_);
      deferred.push_back(std::move(retry));
    }
  }

  if (deferred.empty())
    return;

  simdojo::Tick next_tick = simdojo::TICK_MAX;
  bool wake = false;
  {
    const std::lock_guard lock(doorbell_inbox_mutex_);
    if (!doorbell_admission_open_)
      return;
    for (DoorbellRetry &retry : deferred) {
      if (retry.notification_.reset_epoch_ != doorbell_reset_epoch_)
        continue;
      const std::deque<DoorbellRetry>::iterator duplicate =
          std::ranges::find_if(doorbell_retry_inbox_, [&](const DoorbellRetry &candidate) {
            return same_doorbell_retry_target(candidate, retry.notification_, retry.model_);
          });
      if (duplicate == doorbell_retry_inbox_.end()) {
        next_tick = std::min(next_tick, retry.ready_tick_);
        doorbell_retry_inbox_.push_back(std::move(retry));
        continue;
      }
      duplicate->notification_.value_ =
          std::max(duplicate->notification_.value_, retry.notification_.value_);
      duplicate->notification_.width_ = retry.notification_.width_;
      duplicate->backoff_ = std::min(duplicate->backoff_, retry.backoff_);
      duplicate->ready_tick_ = std::min(duplicate->ready_tick_, retry.ready_tick_);
      next_tick = std::min(next_tick, duplicate->ready_tick_);
    }
    if (engine() != nullptr && next_tick != simdojo::TICK_MAX &&
        (!doorbell_retry_wake_tick_ || next_tick < *doorbell_retry_wake_tick_)) {
      doorbell_retry_wake_tick_ = next_tick;
      wake = true;
    }
  }
  if (wake)
    schedule_event(&doorbell_retry_event_, next_tick);
}

void GpuPciDevice::drain_doorbell_inbox_for_test() {
  if (engine() != nullptr)
    return;
  bool normal_work = false;
  {
    const std::lock_guard lock(doorbell_inbox_mutex_);
    normal_work = !doorbell_inbox_.empty();
  }
  if (normal_work)
    drain_doorbell_inbox(0);
  else
    drain_doorbell_retry_inbox(0, true);

  if (soc_ == nullptr || soc_->sdma_queue_scheduler().active_queues() == 0)
    return;

  // Let the asynchronous SDMA owner post packet side effects and final queue
  // progress, then apply both on this owner thread. The scheduler, not this
  // frontend, retries a retained packet after an unavailable side effect.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
  uint32_t idle_rounds = 0;
  bool observed_sdma_message = false;
  while (std::chrono::steady_clock::now() < deadline) {
    bool has_message = false;
    {
      const std::lock_guard lock(sdma_effect_mutex_);
      has_message = !sdma_effect_inbox_.empty() || !sdma_progress_inbox_.empty();
    }
    if (has_message) {
      observed_sdma_message = true;
      idle_rounds = 0;
      drain_sdma_pci_effects();
      continue;
    }
    if (observed_sdma_message && ++idle_rounds >= 20)
      break;
    std::this_thread::sleep_for(std::chrono::microseconds(50));
  }
}

void GpuPciDevice::close_doorbell_admission_for_reset() {
  std::unique_lock lock(doorbell_inbox_mutex_);
  doorbell_admission_open_ = false;
  ++doorbell_reset_epoch_;
  if (doorbell_reset_epoch_ == 0)
    ++doorbell_reset_epoch_;
  doorbell_inbox_.clear();
  doorbell_retry_inbox_.clear();
  doorbell_retry_wake_tick_.reset();
  doorbell_wake_pending_ = false;
  cancel_sdma_pci_effects();
  doorbell_idle_.wait(lock, [this]() { return active_frontend_operations_ == 0; });
}

void GpuPciDevice::reopen_doorbell_admission_after_reset() {
  {
    const std::lock_guard lock(doorbell_inbox_mutex_);
    doorbell_admission_open_ = true;
  }
  const std::lock_guard lock(sdma_effect_mutex_);
  sdma_effect_admission_open_ = true;
}

bool GpuPciDevice::read_vram(uint64_t offset, std::span<std::byte> bytes) {
  return vram_.has_value() && vram_->read_all(bytes, offset);
}

bool GpuPciDevice::write_vram(uint64_t offset, std::span<const std::byte> bytes) {
  return vram_.has_value() && vram_->write_all(bytes, offset);
}

amdgpu::AtomicLoadResult GpuPciDevice::atomic_load_vram(uint64_t offset, uint32_t width) {
  if (!vram_.has_value())
    return {.outcome = amdgpu::VmAccessOutcome::Unavailable};
  const std::optional<VramAtomicLoadResult> result = vram_->atomic_load(offset, width);
  if (!result.has_value())
    return {.outcome = amdgpu::VmAccessOutcome::Faulted};
  return {.outcome = amdgpu::VmAccessOutcome::Complete, .value = result->value};
}

amdgpu::VmAccessOutcome GpuPciDevice::atomic_store_vram(uint64_t offset, uint32_t width,
                                                        uint64_t value) {
  if (!vram_.has_value())
    return amdgpu::VmAccessOutcome::Unavailable;
  return vram_->atomic_store(offset, width, value) ? amdgpu::VmAccessOutcome::Complete
                                                   : amdgpu::VmAccessOutcome::Faulted;
}

amdgpu::AtomicCompareExchangeResult GpuPciDevice::compare_exchange_vram(uint64_t offset,
                                                                        uint32_t width,
                                                                        uint64_t expected,
                                                                        uint64_t desired) {
  if (!vram_.has_value())
    return {.outcome = amdgpu::VmAccessOutcome::Unavailable};
  const std::optional<VramAtomicCompareExchangeResult> result =
      vram_->compare_exchange(offset, width, expected, desired);
  if (!result.has_value())
    return {.outcome = amdgpu::VmAccessOutcome::Faulted};
  return {.outcome = amdgpu::VmAccessOutcome::Complete,
          .observed = result->observed,
          .exchanged = result->exchanged};
}

bool GpuPciDevice::read_register(uint64_t byte_offset, uint32_t &value) {
  if (!registers_.holds(byte_offset)) {
    return false;
  }
  value = registers_.value(byte_offset);
  return true;
}

bool GpuPciDevice::write_register(uint64_t byte_offset, uint32_t value) {
  if (!registers_.holds(byte_offset)) {
    return false;
  }
  if (registers_.modelled(byte_offset) && !registers_.read_only(byte_offset)) {
    if (!registers_.store(byte_offset, value))
      return false;
    for (const OwnedBlockModel &owned : models_) {
      owned.model_->observe_register_write(byte_offset, *this);
    }
  }
  return true;
}

void GpuPciDevice::reset(simdojo::ResetKind kind) {
  const std::lock_guard reset_lock(doorbell_reset_mutex_);
  close_doorbell_admission_for_reset();
  util::Logger::warn(std::format("{}: reset requested, kind {}", name(), static_cast<int>(kind)));
  try {
    const bool queues_released = teardown_frontend_queues();
    if (!queues_released)
      throw std::runtime_error("PCI queue state could not be cleared");
    if (soc_ != nullptr) {
      // Preserve the device-global VMID-0 identity, but drop the translator
      // and its retained transport backing. A later hub invalidation publishes
      // the replacement frontend's coherent GART configuration.
      if (!soc_->gpu_vm().clear_gart_binding())
        throw std::runtime_error("VMID-0 GART binding could not be cleared");
    }
    // Everything a client could have changed goes back to power-on state. Video
    // memory is not cleared here: a mapping already handed out cannot be taken
    // back, which is why a device that exports one is served to a single client.
    {
      const std::lock_guard lock(doorbell_storage_mutex_);
      std::ranges::fill(doorbells_, std::byte{0});
    }
    std::ranges::fill(msix_table_, std::byte{0});
    reset_registers();
    // The table is restored rather than assumed intact. On real hardware it lives
    // in memory the security processor reserves and the driver cannot write; here
    // it is ordinary video memory, reachable through MM_DATA, so a guest that
    // scribbles over it once would otherwise make every later bind fail.
    if (!publish_discovery_table()) {
      util::Logger::warn(std::format("{}: the discovery table could not be restored, so a driver "
                                     "binding after this reset will refuse the device",
                                     name()));
    }
  } catch (...) {
    reopen_doorbell_admission_after_reset();
    throw;
  }
  reopen_doorbell_admission_after_reset();
}

} // namespace rocjitsu
