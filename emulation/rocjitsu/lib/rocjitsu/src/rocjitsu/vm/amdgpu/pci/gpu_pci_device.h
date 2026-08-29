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

  /// @brief Advertise a single legacy interrupt pin.
  ///
  /// @details The driver asks the bus for one vector of any kind at all and
  /// refuses the device outright when it cannot have one, so a function with no
  /// interrupt capability whatsoever does not merely lose interrupts: its
  /// interrupt-handling block fails to initialize and the probe ends there. A
  /// pin is the smallest capability that satisfies that, and the only one this
  /// transport can currently advertise.
  ///
  /// Nothing built against it is wasted when this becomes MSI-X, because the
  /// driver decides how to program the interrupt ring from a module parameter
  /// rather than from what the bus actually gave it, and so programs the ring
  /// identically either way.
  ///
  /// **A pin must not be what finally raises an interrupt, though.** A client
  /// disables every mmap of every BAR while a legacy interrupt is pending, and
  /// restores them only after a quiet period; this device's largest BAR is the
  /// memory aperture the guest maps to avoid trapping. Raising pins at the rate
  /// an interrupt ring produces them would therefore cost the guest its direct
  /// view of memory for as long as they keep arriving, and would present as a
  /// collapse of the memory path rather than as anything to do with interrupts.
  ///
  /// @returns A single legacy pin. The vectors field is NOT APPLICABLE here
  ///          rather than meaningful: it counts message-signalled vectors, so
  ///          the zero reported alongside IntxPin is not a count of anything
  ///          this device declines to raise. What is advertised is a capability
  ///          the guest driver checks for before it will keep the device; this
  ///          device raises nothing through it yet.
  [[nodiscard]] simdojo::InterruptSpec interrupts() const override {
    return {.kind = simdojo::InterruptKind::IntxPin, .vectors = 0};
  }

  [[nodiscard]] int64_t bar_access(int bar, std::span<std::byte> buf, uint64_t offset,
                                   bool write) override;
  void dma_map(const simdojo::DmaRegion &region) override;
  void dma_unmap(const simdojo::DmaRegion &region) override;
  void reset(simdojo::ResetKind kind) override;

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
  int vram_fd_ = -1;
  std::byte *vram_ = nullptr;

  std::vector<std::byte> doorbells_;
  bool usable_ = false;
};

} // namespace rocjitsu
