// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file gpu_pci_device.h
/// @brief A simulated GPU as a guest driver first sees it on the bus.
///
/// @details This is the device an unmodified kernel driver attaches to. It
/// presents an identity, the apertures the driver maps, and the few registers it
/// reads before it knows anything else about the hardware.
///
/// Nothing here is specific to one ASIC. Which GPU is presented comes entirely
/// from the simulation config, the same place the rest of the machine's shape
/// comes from, so supporting a new part is a config file rather than a new class.

#pragma once

#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"
#include "simdojo/components/pci_device.h"
#include "simdojo/components/register_file.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace rocjitsu {

/// @brief The identity and bus shape a simulated GPU presents.
struct GpuPciDeviceSpec {
  simdojo::PciId id;       ///< Identity to present in configuration space.
  uint64_t vram_bytes = 0; ///< Local memory, which the driver reads back.
  /// @brief Aperture onto that memory. Must be a power of two; derive it with
  /// @ref gpu_pci_spec_from_config rather than leaving it unset.
  uint64_t vram_aperture_bytes = 0;
  uint64_t doorbell_aperture_bytes = 0; ///< Doorbell aperture size.
  uint64_t register_aperture_bytes = 0; ///< Register aperture size.
  /// @brief The blocks this device publishes to a guest driver.
  ///
  /// @details Carried in the spec rather than chosen inside the device, because
  /// the identity in @ref id and the hardware described here have to be the
  /// same GPU. A device that picked its own table could present one part's PCI
  /// IDs alongside another part's blocks, and the guest would instantiate
  /// drivers for hardware the identity says is not there. Derive it with
  /// @ref gpu_pci_spec_from_config, which reads both from one config.
  IpDiscoverySpec discovery;
};

/// @brief PCI function presenting a simulated GPU to a guest driver.
class GpuPciDevice final : public simdojo::PciDevice {
public:
  /// @brief BAR carrying the video memory aperture.
  static constexpr int kVramBar = 0;

  /// @brief BAR carrying the doorbell pages.
  static constexpr int kDoorbellBar = 2;

  /// @brief BAR carrying the register aperture.
  static constexpr int kRegisterBar = 5;

  /// @brief BAR carrying the message-signalled interrupt table.
  ///
  /// @details Its own BAR rather than a corner of the register aperture, so
  /// that a table entry can never be mistaken for a register or land on one.
  /// BARs 1 and 3 are the upper halves of the two 64-bit apertures, which
  /// leaves this as the only free index.
  static constexpr int kMsixBar = 4;

  /// @brief Size of that BAR.
  ///
  /// @details The table and the pending bits get a 4 KiB page each. Nothing
  /// requires it -- they may share a page with each other, and a client asks
  /// only that they be eight-byte aligned and not overlap -- but a page apiece
  /// means neither can ever share one with anything else, which is what the
  /// specification does care about, and it costs 8 KiB of a BAR that holds
  /// nothing else.
  static constexpr uint64_t kMsixBarBytes = 8 * 1024;

  /// @brief Where the table starts within that BAR.
  static constexpr uint64_t kMsixTableOffset = 0;

  /// @brief Where the pending bits start, one page after it.
  static constexpr uint64_t kMsixPendingOffset = 4 * 1024;

  /// @brief Message vectors advertised.
  ///
  /// @details One, because the driver asks the bus for exactly one and this
  /// device has exactly one thing to report: that its interrupt ring has
  /// something in it. Advertising more would be describing hardware that is
  /// not there.
  static constexpr uint32_t kMsixVectors = 1;

  /// @brief Smallest memory BAR the PCI specification allows.
  ///
  /// @details Checked here so a device reports its own configuration unusable
  /// rather than being rejected later while a transport is built around it.
  static constexpr uint64_t kMinMemoryBarBytes = 16;

  /// @brief Smallest register aperture leaving every pre-discovery register
  /// directly addressable.
  ///
  /// @details The furthest of them, the discovery table version, sits at byte
  /// 0x5a800, and a BAR must be a power of two, so 512 KiB is the smallest legal
  /// size that reaches it. Real hardware may expose less and serve the remainder
  /// through the indirect window, but a device that advertises an aperture too
  /// small to answer its own registers is refused rather than quietly broken.
  static constexpr uint64_t kMinRegisterApertureBytes = 512 * 1024;

  /// @brief Where the discovery table sits, measured back from the top of
  /// memory.
  ///
  /// @details The driver computes this itself, from the capacity the device
  /// reports, whenever the scratch registers do not name somewhere else. So the
  /// device does not get to choose the address: it either writes the table here
  /// or the driver reads whatever happens to be here instead.
  static constexpr uint64_t kDiscoveryOffsetFromTopOfVram = 64 * 1024;

  /// @brief Construct the function.
  /// @param[in] name Component name, used in diagnostics.
  /// @param[in] spec Identity and bus shape, from the simulation config.
  /// @param[in] trace Access diagnostics to feed, or nullptr for none. Must
  ///                  outlive this device.
  GpuPciDevice(std::string name, const GpuPciDeviceSpec &spec, BarAccessTrace *trace);
  ~GpuPciDevice() override;

  [[nodiscard]] std::vector<simdojo::BarSpec> bars() const override;

  /// @brief Advertise message-signalled interrupts.
  ///
  /// @details The driver asks the bus for one vector of any kind at all and
  /// refuses the device outright when it cannot have one, so a function with no
  /// interrupt capability whatsoever does not merely lose interrupts: its
  /// interrupt-handling block fails to initialize and the probe ends there.
  ///
  /// A legacy pin would satisfy that too, and is cheaper, but it cannot be what
  /// raises the interrupt: a client disables every mmap of every BAR while a
  /// legacy interrupt is pending and restores them only after a quiet period,
  /// and this device's largest BAR is the memory aperture the guest maps to
  /// avoid trapping. Raising pins at the rate an interrupt ring produces them
  /// would cost the guest its direct view of memory for as long as they kept
  /// arriving, and would present as a collapse of the memory path rather than
  /// as anything to do with interrupts. Messages carry no such penalty, and
  /// they are what the parts being modelled use.
  ///
  /// The cost of offering no pin at all is that a driver *forced* to legacy
  /// interrupts -- by the module parameter that turns messages off -- asks the
  /// bus for a pin, is refused, and fails the same probe this capability
  /// exists to get through. That is a debugging option rather than a
  /// configuration anyone runs, and restoring the pin would reintroduce the
  /// mapping penalty above the moment anything raised one.
  ///
  /// @returns One vector, and where its table lives.
  [[nodiscard]] simdojo::InterruptSpec interrupts() const override {
    return {.kind = simdojo::InterruptKind::MsiX,
            .vectors = kMsixVectors,
            .table_bar = kMsixBar,
            .table_offset = kMsixTableOffset,
            .pending_offset = kMsixPendingOffset};
  }

  [[nodiscard]] int64_t bar_access(int bar, std::span<std::byte> buf, uint64_t offset,
                                   bool write) override;
  void dma_map(const simdojo::DmaRegion &region) override;
  void dma_unmap(const simdojo::DmaRegion &region) override;
  void reset(simdojo::ResetKind kind) override;

  /// @brief Which address space an interrupt ring's addresses are in.
  ///
  /// @details The driver puts this in the same register as the ring's size and
  /// its enable bit, and it decides what the addresses beside it *mean*. It
  /// follows how firmware is loaded: a driver loading firmware through the
  /// security processor allocates the ring through the translation tables and
  /// programs virtual addresses, and every other way of loading gets a bus
  /// address. So identical register values denote different memory depending on
  /// a module parameter, and an address used in the wrong space does not fail --
  /// it points somewhere real and wrong.
  enum class InterruptRingSpace : uint8_t {
    /// @brief Not one of the two values the driver writes; in practice, a ring
    /// it has not programmed yet.
    Unset = 0,
    BusAddress = 2, ///< Guest physical, reachable through a shared window.
    GpuVirtual = 4  ///< Behind translation tables this device does not walk.
  };

  /// @brief The interrupt ring as the driver has programmed it so far.
  ///
  /// @details The ring is the device's side of the interrupt path: the device
  /// writes an entry into it, publishes a write pointer, and raises a message.
  /// All of that needs to know where the driver put the ring, which the driver
  /// says only by writing these registers -- so this is read back out of them
  /// rather than tracked as they are written, because the driver writes them in
  /// its own order and rewrites them on reset.
  struct InterruptRing {
    /// @brief Address of the ring in @ref space, or zero if it is not set.
    uint64_t base = 0;
    uint64_t bytes = 0;        ///< Its size, decoded from the size field.
    uint64_t wptr_address = 0; ///< Where the device publishes the write pointer.
    /// @brief What @ref base and @ref wptr_address are addresses in.
    InterruptRingSpace space = InterruptRingSpace::Unset;
    bool enabled = false;         ///< Whether the driver has switched the ring on.
    bool raises_messages = false; ///< Whether it wants an interrupt per entry.

    /// @brief Whether the driver has said anything at all about the ring.
    ///
    /// @details Every field, rather than the address and the enable bit alone:
    /// a driver that sized a ring and named a write-pointer address but had not
    /// switched it on yet has said a great deal, and reporting that as nothing
    /// programmed would hide exactly the partial state worth seeing. These read
    /// back as their defaults only while the registers are untouched.
    /// @retval false Every field still holds what a reset left.
    [[nodiscard]] bool programmed() const {
      return base != 0 || bytes != 0 || wptr_address != 0 || space != InterruptRingSpace::Unset ||
             enabled || raises_messages;
    }
  };

  /// @brief Read the interrupt ring out of the registers the driver wrote.
  ///
  /// @details Reads register state without a lock of its own, so it must be
  /// called either from the thread servicing register access or after that
  /// thread has been joined.
  ///
  /// @returns What the driver has said about the ring so far.
  [[nodiscard]] InterruptRing interrupt_ring() const;

  /// @brief One interrupt, as the ring carries it.
  ///
  /// @details The driver looks up a handler by the pair of identifiers and
  /// passes the rest to it. Everything else an entry can carry -- timestamps,
  /// process and node identifiers -- describes work this device does not run,
  /// so it is left zero rather than invented.
  struct InterruptEntry {
    uint8_t client_id = 0;             ///< Which block is reporting.
    uint8_t source_id = 0;             ///< What it is reporting.
    std::array<uint32_t, 4> data = {}; ///< Whatever that source attaches.
  };

  /// @brief Put one entry in the interrupt ring and raise the message for it.
  ///
  /// @details The whole delivery, because the parts are only meaningful
  /// together: an entry the driver never sees, a write pointer naming an entry
  /// that is not there, or a message with nothing behind it are each worse than
  /// doing nothing. The write pointer is published into guest memory as well as
  /// into its register, because the driver reads it from memory; it reads the
  /// register only when the pointer it read says an overflow happened.
  ///
  /// **Nothing tracks how much of the ring the driver has consumed.** The
  /// driver acknowledges entries by writing a doorbell, which this device
  /// records and does not read, so a ring filled faster than it is drained
  /// overwrites entries the driver has not seen, and reports no overflow. That
  /// is survivable only while deliveries are occasional; anything raising
  /// interrupts at a rate needs the read pointer followed first.
  ///
  /// Must be called from the thread servicing register access, or with that
  /// thread stopped: it reads registers and reaches guest memory through the
  /// transport, neither of which it locks.
  ///
  /// @param[in] entry What to report.
  /// @retval false Nothing was delivered, or not all of it was. A ring that is
  ///               unusable is declined before anything is written; a failure
  ///               to publish the pointer leaves an entry nobody is pointed at,
  ///               which the next delivery overwrites; a message the transport
  ///               would not take leaves the entry and the pointer in place,
  ///               for the next delivery's message to cover.
  [[nodiscard]] bool deliver_interrupt(const InterruptEntry &entry);

  /// @brief Render a ring for a diagnostic.
  ///
  /// @details A sentence fragment beginning "its interrupt ring at ...", meant
  /// to follow a subject and verb naming who did what -- "the driver enabled",
  /// "the driver left". Logged on its own it reads as though it lost a word.
  ///
  /// @param[in] ring The ring to describe.
  /// @returns Where it is, how big, and in which address space.
  [[nodiscard]] static std::string describe(const InterruptRing &ring);

  /// @brief Whether the device is usable.
  /// @retval false The configuration was rejected or its memory could not be
  ///               backed; the reason has been logged and it must not be served.
  [[nodiscard]] bool usable() const { return usable_; }

private:
  void define_register(uint64_t byte_offset, uint32_t value);
  [[nodiscard]] int64_t access_registers(std::span<std::byte> buf, uint64_t offset, bool write);
  [[nodiscard]] int64_t access_memory(std::span<std::byte> buf, uint64_t offset, bool write,
                                      std::span<std::byte> backing);

  void reset_registers();

  /// @brief One hub's invalidation engines, as three contiguous register blocks.
  ///
  /// @details A flush writes an engine's request register and polls its
  /// acknowledge register for the bit of the VMID being flushed. Nothing else
  /// reports the flush finishing, so an unanswered step is not a missing
  /// register but a stall of the driver's whole timeout, once per flush.
  ///
  /// Older flush paths bracket that pair with an acquire and release of the
  /// engine's semaphore. The one this device publishes today does not, so the
  /// semaphore block is answered for the parts that take that path rather than
  /// for this one.
  ///
  /// The three blocks are modelled together because they are laid out together
  /// -- eighteen engines of each, back to back -- and because answering only
  /// some of the handshake buys nothing: the driver waits just as long on
  /// whichever step is unanswered.
  struct InvalidationEngineBlocks {
    uint32_t semaphore = 0;   ///< Engine 0's semaphore, in dwords from the segment.
    uint32_t request = 0;     ///< Engine 0's request register, likewise.
    uint32_t acknowledge = 0; ///< Engine 0's acknowledge register, likewise.
  };

  /// @brief Answer the invalidation engines of the hub based at @p segment.
  /// @param[in] segment First register segment of the hub, from the published
  ///                    discovery table.
  /// @param[in] blocks Where each block starts, relative to that segment.
  /// @returns True when every engine of the hub was defined.
  [[nodiscard]] bool define_invalidation_engines(uint64_t segment,
                                                 const InvalidationEngineBlocks &blocks);

  /// @brief The published block with @p id, or nullptr if the table omits it.
  /// @details Returns instance 0. The version matters as much as the segment:
  /// register offsets within a block move between versions of it.
  [[nodiscard]] const IpBlock *published_block(IpHardwareId id) const;

  /// @brief Whether a published segment can be turned into a byte address.
  /// @details One place rather than each caller: every caller computes
  /// `(segment + register) * 4`, which a large enough segment wraps into a
  /// small address that would pass a bounds test and land on an unrelated
  /// register.
  [[nodiscard]] bool segment_within_aperture(uint64_t segment) const;

  /// @brief Whether every named hub's flush handshakes can be answered.
  bool flushes_answerable_ = false;

  /// @brief Define a register that answers reads and ignores writes.
  ///
  /// @details For registers whose value is a property of the hardware rather
  /// than state the driver owns. A semaphore the device always grants is the
  /// case in point: the driver releases it by writing zero, and a register that
  /// stored that write would grant the acquire once and then stall forever.
  ///
  /// @param[in] byte_offset Where the register sits in the aperture.
  /// @param[in] value What it always reads as.
  void define_read_only_register(uint64_t byte_offset, uint32_t value);

  /// @brief Address the index registers currently select.
  ///
  /// @details Recomputed from both registers on each access rather than tracked
  /// as they are written, because the driver writes them in either order and
  /// updates the high half only when it changes.
  [[nodiscard]] uint64_t indirect_address() const;
  [[nodiscard]] bool read_vram(uint64_t offset, uint32_t &value) const;
  bool write_vram(uint64_t offset, uint32_t value);

  /// @brief Write the discovery table where the driver will look for it.
  /// @returns Whether the table was built, accepted and stored.
  [[nodiscard]] bool publish_discovery_table();

  GpuPciDeviceSpec spec_;
  BarAccessTrace *trace_;

  /// @brief Register storage, one entry per dword of the aperture.
  /// @brief One dword per register in the aperture.
  /// @details Plain dense storage rather than a simdojo::RegisterFile: that
  /// type is a shader register file, whose operator[] requires the index to
  /// fall in an allocated block, and this aperture allocates none -- every
  /// index would fail that precondition and abort an assertion-enabled build
  /// during construction.
  std::vector<uint32_t> registers_;

  /// @brief Memory capacity as the driver reads it, in megabytes.
  ///
  /// @details Validated once at construction, because the register is 32 bits
  /// of megabytes and the configured capacity is 64 bits of bytes: not every
  /// capacity has a representation the driver would accept.
  uint32_t vram_megabytes_ = 0;

  /// @brief Which of those registers the device actually models.
  ///
  /// @details Kept apart from the values because zero is a legitimate register
  /// value as well as what absent hardware reads as, and telling the two apart
  /// is the whole point of the unmodelled-register report.
  std::vector<bool> modelled_;

  /// @brief Which of the modelled registers ignore writes.
  std::vector<bool> read_only_;

  /// @brief Byte offset of the interrupt ring's control register, or nothing
  /// when the published table names no block that has one. Not zero for that:
  /// byte zero is the indirect window's own index register.
  std::optional<uint64_t> ih_control_offset_;

  /// @brief Whether the ring being switched on has already been reported.
  bool announced_interrupt_ring_ = false;
  int vram_fd_ = -1;
  std::byte *vram_ = nullptr;

  std::vector<std::byte> doorbells_;

  /// @brief Backing for the message table and its pending bits.
  ///
  /// @details Plain storage. A client of this transport emulates the table's
  /// meaning itself and delivers the message, so what the device has to do is
  /// hold the bytes and not lose them.
  std::vector<std::byte> msix_table_;
  bool usable_ = false;
};

} // namespace rocjitsu
