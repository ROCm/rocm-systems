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

#include "rocjitsu/vm/amdgpu/interrupt_sink.h"
#include "rocjitsu/vm/amdgpu/pci/bar_access_trace.h"
#include "rocjitsu/vm/amdgpu/pci/interrupt_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/interrupt_ring.h"
#include "rocjitsu/vm/amdgpu/pci/ip_block_model.h"
#include "rocjitsu/vm/amdgpu/pci/ip_discovery.h"
#include "rocjitsu/vm/amdgpu/pci/register_aperture.h"
#include "rocjitsu/vm/amdgpu/pci/vram_store.h"
#include "simdojo/components/pci_device.h"
#include "simdojo/components/register_file.h"

#include <array>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace rocjitsu {

class SoC;
class MesBlockModel;
class SdmaBlockModel;

namespace amdgpu {
class SdmaPacketCallbacks;
class SdmaQueueBindingFactory;
struct SdmaQueueContext;
struct SdmaQueueProgress;
} // namespace amdgpu

/// @brief The identity and bus shape a simulated GPU presents.
class GpuPciDeviceSpec {
public:
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
class GpuPciDevice final : public simdojo::PciDevice, private PciMemoryAccess {
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
  GpuPciDevice(std::string name, const GpuPciDeviceSpec &spec, BarAccessTrace *trace,
               SoC *soc = nullptr);
  ~GpuPciDevice() override;

  /// @brief Release every core and transport resource owned by this frontend.
  ///
  /// @details Must run while the referenced SoC is still alive. The operation
  /// is idempotent and severs every borrowed core/model route before returning,
  /// so a later owner-driven destruction needs no access to the SoC.
  ///
  /// @returns Whether every frontend-owned queue and VM binding was released.
  [[nodiscard]] bool shutdown_frontend();

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

  /// @brief Advertise the PCIe AtomicOp completion widths KFD requires.
  ///
  /// @details The upstream driver enables 32- and 64-bit requests together and
  /// declines this generation when either width cannot reach the function.
  [[nodiscard]] simdojo::PcieSpec pcie() const override {
    return {.atomic_completer_32 = true, .atomic_completer_64 = true};
  }

  [[nodiscard]] int64_t bar_access(int bar, std::span<std::byte> buf, uint64_t offset,
                                   bool write) override;
  void dma_map(const simdojo::DmaRegion &region) override;
  void dma_unmap(const simdojo::DmaRegion &region) override;
  void reset(simdojo::ResetKind kind) override;

  /// @brief Drain pending semantic doorbell work without a simulation engine.
  /// @details Production callers schedule the same drain on this component's
  /// owner thread. Topology-free unit tests call this explicitly so they do not
  /// hide a synchronous transport-callback path that production must avoid.
  void drain_doorbell_inbox_for_test();
  [[nodiscard]] std::thread::id last_sdma_pci_effect_thread_for_test();
  [[nodiscard]] uint64_t sdma_pci_effect_count_for_test();
  void set_sdma_pci_effect_admitted_hook_for_test(std::function<void()> hook);

  /// @brief Read the interrupt ring out of the registers the driver wrote.
  ///
  /// @details Reads register state without a lock of its own, so it must be
  /// called either from the thread servicing register access or after that
  /// thread has been joined.
  ///
  /// @returns What the driver has said about the ring so far.
  [[nodiscard]] InterruptRing interrupt_ring() const;

  /// @brief Put one entry in the interrupt ring and raise the message for it.
  ///
  /// @details The whole delivery, because the parts are only meaningful
  /// together: an entry the driver never sees, a write pointer naming an entry
  /// that is not there, or a message with nothing behind it are each worse than
  /// doing nothing. The write pointer is published into guest memory as well as
  /// into its register, because the driver reads it from memory; it reads the
  /// register only when the pointer it read says an overflow happened.
  ///
  /// The driver's read-pointer doorbell bounds unread entries. When a delivery
  /// would catch it, the device reports overflow in both write-pointer copies
  /// and keeps that indication latched until the driver pulses the control
  /// register's overflow-clear bit, matching the handler's acknowledgement
  /// sequence.
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
  [[nodiscard]] bool deliver_interrupt(const InterruptEntry &entry) override;

  /// @brief Whether the device is usable.
  /// @retval false The configuration was rejected or its memory could not be
  ///               backed; the reason has been logged and it must not be served.
  [[nodiscard]] bool usable() const { return usable_; }

private:
  [[nodiscard]] std::shared_ptr<simdojo::PciTransportSession>
  capture_transport_session() const override {
    return transport_session();
  }
  [[nodiscard]] bool read_vram(uint64_t offset, std::span<std::byte> bytes) override;
  [[nodiscard]] bool write_vram(uint64_t offset, std::span<const std::byte> bytes) override;
  [[nodiscard]] amdgpu::AtomicLoadResult atomic_load_vram(uint64_t offset, uint32_t width) override;
  [[nodiscard]] amdgpu::VmAccessOutcome atomic_store_vram(uint64_t offset, uint32_t width,
                                                          uint64_t value) override;
  [[nodiscard]] amdgpu::AtomicCompareExchangeResult
  compare_exchange_vram(uint64_t offset, uint32_t width, uint64_t expected,
                        uint64_t desired) override;
  [[nodiscard]] bool read_register(uint64_t byte_offset, uint32_t &value) override;
  [[nodiscard]] bool write_register(uint64_t byte_offset, uint32_t value) override;

  [[nodiscard]] int64_t access_registers(std::span<std::byte> buf, uint64_t offset, bool write);
  [[nodiscard]] int64_t access_memory(std::span<std::byte> buf, uint64_t offset, bool write,
                                      std::span<std::byte> backing);

  class DoorbellNotification {
  public:
    DoorbellNotification(std::weak_ptr<simdojo::PciTransportSession> session, uint64_t generation,
                         uint64_t reset_epoch, uint64_t byte_offset, uint64_t value,
                         std::size_t width)
        : session_(std::move(session)), generation_(generation), reset_epoch_(reset_epoch),
          byte_offset_(byte_offset), value_(value), width_(width) {}

  private:
    friend class GpuPciDevice;
    std::weak_ptr<simdojo::PciTransportSession> session_;
    uint64_t generation_ = 0;
    uint64_t reset_epoch_ = 0;
    uint64_t byte_offset_ = 0;
    uint64_t value_ = 0;
    std::size_t width_ = 0;
  };

  class DoorbellRetry {
  public:
    DoorbellRetry(DoorbellNotification notification, IpBlockModel *model, simdojo::Tick backoff,
                  simdojo::Tick ready_tick)
        : notification_(std::move(notification)), model_(model), backoff_(backoff),
          ready_tick_(ready_tick) {}

  private:
    friend class GpuPciDevice;
    DoorbellNotification notification_;
    IpBlockModel *model_ = nullptr;
    simdojo::Tick backoff_ = 1;
    simdojo::Tick ready_tick_ = 0;
  };

  class FrontendOperationLease {
  public:
    FrontendOperationLease() = default;
    explicit FrontendOperationLease(GpuPciDevice *device) : device_(device) {}
    ~FrontendOperationLease();

    FrontendOperationLease(const FrontendOperationLease &) = delete;
    FrontendOperationLease &operator=(const FrontendOperationLease &) = delete;
    FrontendOperationLease(FrontendOperationLease &&other) noexcept;
    FrontendOperationLease &operator=(FrontendOperationLease &&other) noexcept;

    [[nodiscard]] explicit operator bool() const { return device_ != nullptr; }

  private:
    GpuPciDevice *device_ = nullptr;
  };

  enum class SdmaPciEffectKind : uint8_t { ReadRegister, WriteRegister, DeliverInterrupt };

  class SdmaPciEffect {
  public:
    SdmaPciEffectKind kind = SdmaPciEffectKind::ReadRegister;
    uint32_t address = 0;
    uint32_t value = 0;
    uint32_t process_id = 0;
    uint32_t engine_id = 0;

    friend bool operator==(const SdmaPciEffect &, const SdmaPciEffect &) = default;
  };

  struct SdmaPciEffectResult {
    amdgpu::VmAccessOutcome outcome = amdgpu::VmAccessOutcome::Faulted;
    uint32_t value = 0;
  };

  class SdmaPciCallbackState {
  public:
    [[nodiscard]] SdmaPciEffectResult request(const std::shared_ptr<SdmaPciCallbackState> &self,
                                              const SdmaPciEffect &effect);
    void complete(const SdmaPciEffect &effect, SdmaPciEffectResult result);
    void cancel();

  private:
    friend class GpuPciDevice;
    std::mutex mutex_;
    std::optional<SdmaPciEffect> pending_;
    std::optional<SdmaPciEffectResult> result_;
    std::function<bool(const std::shared_ptr<SdmaPciCallbackState> &, const SdmaPciEffect &)>
        enqueue_;
    std::weak_ptr<simdojo::PciTransportSession> session_;
    uint64_t generation_ = 0;
    uint64_t reset_epoch_ = 0;
    bool cancelled_ = false;
  };

  class SdmaPciEffectRequest {
  public:
    SdmaPciEffectRequest(std::weak_ptr<SdmaPciCallbackState> state, SdmaPciEffect effect,
                         uint64_t reset_epoch)
        : state_(std::move(state)), effect_(effect), reset_epoch_(reset_epoch) {}

  private:
    friend class GpuPciDevice;
    std::weak_ptr<SdmaPciCallbackState> state_;
    SdmaPciEffect effect_;
    uint64_t reset_epoch_ = 0;
  };

  class SdmaQueueProgressRequest {
  public:
    std::weak_ptr<simdojo::PciTransportSession> session;
    uint64_t generation = 0;
    uint64_t reset_epoch = 0;
    uint32_t queue_id = 0;
    uint32_t engine_id = 0;
    uint64_t consumer_cursor = 0;
    bool terminal = false;
  };

  void enqueue_doorbell_notification(uint64_t byte_offset, uint64_t value, std::size_t width);
  void drain_doorbell_inbox(simdojo::Tick now);
  void drain_doorbell_retry_inbox(simdojo::Tick now, bool ignore_ready = false);
  void enqueue_doorbell_retry(const DoorbellNotification &notification, IpBlockModel *model,
                              simdojo::Tick now, simdojo::Tick backoff);
  void discard_doorbell_retry(const DoorbellNotification &notification, IpBlockModel *model);
  [[nodiscard]] static bool same_doorbell_retry_target(const DoorbellRetry &retry,
                                                       const DoorbellNotification &notification,
                                                       const IpBlockModel *model);
  [[nodiscard]] FrontendOperationLease acquire_frontend_operation(uint64_t reset_epoch);
  void finish_frontend_operation();
  void close_doorbell_admission_for_reset();
  void reopen_doorbell_admission_after_reset();
  [[nodiscard]] bool enqueue_sdma_pci_effect(const std::shared_ptr<SdmaPciCallbackState> &state,
                                             const SdmaPciEffect &effect);
  void drain_sdma_pci_effects();
  void cancel_sdma_pci_effects();
  [[nodiscard]] bool teardown_frontend_queues();
  [[nodiscard]] amdgpu::SdmaPacketCallbacks
  make_sdma_callbacks(const amdgpu::SdmaQueueContext &context);
  [[nodiscard]] std::function<void(const amdgpu::SdmaQueueProgress &)>
  make_sdma_progress_observer(const amdgpu::SdmaQueueContext &context);

  void reset_registers();

  /// @brief Build a model for every published block whose version is known, and
  ///        check that no two of them answer the same register.
  ///
  /// @details Once, at construction, because a model's identity follows the
  /// discovery table and that does not change while the device is alive. A
  /// reset returns each model's registers to their power-on values; it does not
  /// rebuild the models, any more than a bus reset changes what a part contains.
  ///
  /// Every model's claims are resolved against the segments the table published
  /// for its own block and laid into one absolute-dword map. Two claims landing
  /// on one register is a construction-time error rather than a runtime
  /// surprise, because the loser of that collision presents as a block that
  /// stalls the driver on a register reading as somebody else's.
  void build_block_models();

  /// @brief Whether every named hub's flush handshakes can be answered.
  bool flushes_answerable_ = false;

  /// @brief Whether every block this device requires a model for got one, and
  /// no two of them claimed the same register.
  bool models_complete_ = false;

  /// @brief Address the index registers currently select.
  ///
  /// @details Recomputed from both registers on each access rather than tracked
  /// as they are written, because the driver writes them in either order and
  /// updates the high half only when it changes.
  [[nodiscard]] uint64_t indirect_address() const;

  /// @brief Write the discovery table where the driver will look for it.
  /// @returns Whether the table was built, accepted and stored.
  [[nodiscard]] bool publish_discovery_table();

  GpuPciDeviceSpec spec_;
  BarAccessTrace *trace_;
  SoC *soc_ = nullptr;

  /// One revocable route owned solely by this PCI frontend and its queues.
  amdgpu::InterruptSubscription interrupt_subscription_;

  /// @brief The blocks @ref spec_ publishes, looked up rather than scanned for.
  ///
  /// @details Points into @ref spec_, which is declared above it so it is built
  /// second and destroyed first, and which nothing mutates afterwards.
  IpBlockIndex blocks_;

  /// @brief Register storage, one entry per dword of the aperture.
  ///
  /// @details Plain dense storage rather than a simdojo::RegisterFile: that
  /// type is a shader register file, whose operator[] requires the index to
  /// fall in an allocated block, and this aperture allocates none -- every
  /// index would fail that precondition and abort an assertion-enabled build
  /// during construction.
  ///
  /// Sized once the configuration has been accepted, so a device that refuses
  /// its own spec answers no register at all rather than indexing storage it
  /// never allocated.
  RegisterAperture registers_;

  /// @brief Core SDMA queue binding factory shared by every PCI queue adapter.
  /// @details Declared before the block models so it outlives every binding factory
  /// pointer injected into them.
  std::shared_ptr<amdgpu::SdmaQueueBindingFactory> sdma_queue_binding_factory_;

  /// @brief Memory capacity as the driver reads it, in megabytes.
  ///
  /// @details Validated once at construction, because the register is 32 bits
  /// of megabytes and the configured capacity is 64 bits of bytes: not every
  /// capacity has a representation the driver would accept.
  uint32_t vram_megabytes_ = 0;

  /// @brief One modelled block, and what its absence would mean.
  class OwnedBlockModel {
  public:
    OwnedBlockModel(std::unique_ptr<IpBlockModel> model, bool required)
        : model_(std::move(model)), required_(required) {}

  private:
    friend class GpuPciDevice;
    /// @brief The model. Owned plain objects rather than components: a model
    /// answers register reads on the thread that services them, and joining the
    /// simulation graph would conscript every block into partitioning for no
    /// benefit.
    std::unique_ptr<IpBlockModel> model_;

    /// @brief Whether a device that cannot answer these registers is unusable.
    bool required_ = false;
  };

  /// @brief The blocks this device models, one per published record it can.
  std::vector<OwnedBlockModel> models_;

  /// @brief The interrupt block among them, or nullptr when the published table
  /// names none this device knows how to model. Borrowed from @ref models_.
  InterruptBlockModel *interrupts_ = nullptr;
  /// @brief Queue-owning adapters, borrowed from @ref models_.
  MesBlockModel *mes_ = nullptr;
  SdmaBlockModel *sdma_ = nullptr;

  /// @brief Video memory and the window onto it, owned as one thing.
  ///
  /// @details Optional because it is constructed only once the configuration
  /// has been accepted: a device that refuses its own spec must not go on to
  /// back gigabytes for it.
  std::optional<VramStore> vram_;

  std::vector<std::byte> doorbells_;

  /// @brief Serializes aperture copies without spanning model or transport work.
  mutable std::mutex doorbell_storage_mutex_;

  /// @brief FIFO crossing from transport callback threads to this component's
  /// simulation-owner thread.
  std::mutex doorbell_inbox_mutex_;
  std::condition_variable doorbell_idle_;
  std::deque<DoorbellNotification> doorbell_inbox_;
  std::deque<DoorbellRetry> doorbell_retry_inbox_;
  std::optional<simdojo::Tick> doorbell_retry_wake_tick_;
  bool doorbell_wake_pending_ = false;
  bool doorbell_admission_open_ = true;
  uint64_t doorbell_reset_epoch_ = 1;
  uint64_t active_frontend_operations_ = 0;
  simdojo::Event doorbell_event_{this, simdojo::EventType::TIMER_CALLBACK};
  simdojo::Event doorbell_retry_event_{this, simdojo::EventType::TIMER_CALLBACK};
  static constexpr simdojo::Tick kMaximumDoorbellRetryBackoff = 4096;
  std::mutex doorbell_reset_mutex_;

  /// @brief Cross-thread mailbox keeping SDMA worker callbacks off PCI state.
  std::mutex sdma_effect_mutex_;
  std::deque<SdmaPciEffectRequest> sdma_effect_inbox_;
  std::deque<SdmaQueueProgressRequest> sdma_progress_inbox_;
  std::vector<std::weak_ptr<SdmaPciCallbackState>> sdma_callback_states_;
  bool sdma_effect_admission_open_ = true;
  std::function<void()> sdma_effect_admitted_hook_for_test_;
  std::thread::id last_sdma_effect_thread_;
  uint64_t sdma_effect_count_ = 0;

  /// @brief Backing for the message table and its pending bits.
  ///
  /// @details Plain storage. A client of this transport emulates the table's
  /// meaning itself and delivers the message, so what the device has to do is
  /// hold the bytes and not lose them.
  std::vector<std::byte> msix_table_;
  bool frontend_shutdown_ = false;
  bool frontend_shutdown_complete_ = true;
  bool usable_ = false;
};

} // namespace rocjitsu
