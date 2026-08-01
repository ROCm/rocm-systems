enum class ConSanFaultSiteKind : uint8_t {
  Barrier,
  Atomic,
  LdsAccess,
  OrdinaryMemory,
};

enum class ConSanFaultMutationKind : uint8_t {
  DropBarrier,
  MoveBarrierPair,
  BarrierIdScope,
  BarrierParticipantCount,
  AtomicWrongAddress,
  AtomicWeakenOrder,
  AtomicWeakenScope,
  LdsWrongAddress,
  OrdinaryWrongAddress,
  OrdinaryWeakenOrder,
  OrdinaryWeakenScope,
};

enum class ConSanSyncMemoryRole : uint8_t {
  Unknown,
  None,
  Acquire,
  Release,
  AcquireRelease,
  SequentiallyConsistent,
};

enum class ConSanSemanticConfidence : uint8_t {
  Exact,
  Conservative,
  Ambiguous,
  Unsupported,
};

enum class ConSanOwnerProofKind : uint8_t {
  KernelLocal,
  DirectCall,
  RecoveredIndirectCall,
};

struct ConSanExecutionOwner {
  uint64_t descriptor_file_offset = 0;
  ConSanOwnerProofKind proof = ConSanOwnerProofKind::KernelLocal;
};

/// A currently injectable instruction with a load-order-independent identity.
struct ConSanFaultSite {
  ConSanFaultSiteKind kind = ConSanFaultSiteKind::Barrier;
  std::string identity;
  std::string code_object_fingerprint;
  std::string container_name;
  bool in_kernel = true;
  uint32_t occurrence = 0;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  uint32_t width_bits = 0;
  std::optional<uint16_t> address_vgpr;
  /// gfx1250 bank state at this instruction; the address operand uses bits 1:0.
  std::optional<uint16_t> gfx1250_vgpr_msb_mode;
  std::string mnemonic;
  std::string semantic_role;
  std::string decoded_operands;
  ConSanOrdinaryMemorySupportReason ordinary_memory_support_reason =
      ConSanOrdinaryMemorySupportReason::NotApplicable;
  std::optional<std::string> sync_event_identity;
  std::optional<std::string> sync_sequence_identity;
  ConSanSemanticConfidence sync_confidence = ConSanSemanticConfidence::Unsupported;
  ConSanSyncMemoryRole sync_memory_role = ConSanSyncMemoryRole::Unknown;
  std::vector<ConSanExecutionOwner> execution_owners;
};

struct ConSanFaultMutationPlan {
  ConSanFaultMutationKind kind = ConSanFaultMutationKind::DropBarrier;
  std::string primary_identity;
  std::optional<std::string> companion_identity;
  std::optional<std::string> logical_sequence_identity;
  std::vector<std::string> ordered_member_identities;
  std::optional<std::string> destination_identity;
  ConSanBarrierMoveDirection barrier_move_direction = ConSanBarrierMoveDirection::LegacyMarker;
  ConSanBarrierMoveCfgContract barrier_move_cfg_contract = ConSanBarrierMoveCfgContract::SameBlock;
  std::optional<uint32_t> structured_guard_block_index;
  std::optional<uint32_t> structured_destination_block_index;
  std::optional<uint32_t> structured_source_block_index;
  std::optional<uint64_t> structured_guard_offset;
  std::optional<uint64_t> structured_destination_offset;
  std::optional<uint64_t> structured_source_offset;
  std::optional<int32_t> original_barrier_id;
  std::optional<int32_t> target_barrier_id;
  ConSanBarrierSite::Scope original_barrier_scope = ConSanBarrierSite::Scope::Unknown;
  ConSanBarrierSite::Scope target_barrier_scope = ConSanBarrierSite::Scope::Unknown;
  std::optional<uint32_t> original_participant_count;
  std::optional<uint32_t> target_participant_count;
};

/// A stable real-instruction destination considered by whole-barrier movement.
struct ConSanBarrierMoveDestination {
  std::string identity;
  std::string code_object_fingerprint;
  std::string container_name;
  bool in_kernel = true;
  uint32_t basic_block_index = 0;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  std::string mnemonic;
  bool memory_operation = false;
  bool suitable = false;
  std::string rejection_reason;
  ConSanBarrierMoveCfgContract cfg_contract = ConSanBarrierMoveCfgContract::SameBlock;
  /// Present only for a narrowly proven acyclic structured CFG contract.
  std::optional<uint32_t> structured_guard_block_index;
  std::optional<uint32_t> structured_source_block_index;
  std::optional<uint64_t> structured_guard_offset;
  std::optional<uint64_t> structured_source_offset;
  std::vector<ConSanExecutionOwner> execution_owners;
};

enum class ConSanSyncEventKind : uint8_t {
  Barrier,
  Fence,
  Atomic,
  OrdinaryMemory,
};

enum class ConSanSyncOperation : uint8_t {
  Unknown,
  BarrierSignal,
  BarrierWait,
  BarrierFull,
  BarrierInit,
  BarrierJoin,
  BarrierLeave,
  BarrierWakeup,
  BarrierStateQuery,
  Fence,
  AtomicRmw,
  AtomicCompareExchange,
  OrdinaryLoad,
  OrdinaryStore,
};

enum class ConSanSyncAddressSource : uint8_t {
  NotApplicable,
  Unknown,
  LdsVector,
  FlatVector,
  GlobalScalarVector,
  BufferResource,
  ScratchVector,
};

enum class ConSanSyncRmwOutcome : uint8_t {
  NotApplicable,
  Unknown,
  NoReturn,
  ReturnsOldValue,
  CompareExchange,
};

/// Returns the weaker of two semantic claims. Association must never improve
/// the confidence of an input event.
[[nodiscard]] constexpr ConSanSemanticConfidence
combine_consan_sync_confidence(ConSanSemanticConfidence lhs, ConSanSemanticConfidence rhs) {
  return static_cast<uint8_t>(lhs) >= static_cast<uint8_t>(rhs) ? lhs : rhs;
}

/// Flavor-independent decoded synchronization semantics. Unknown information
/// stays explicit so later sequence association cannot silently invent it.
struct ConSanSyncEvent {
  ConSanSyncEventKind kind = ConSanSyncEventKind::Fence;
  ConSanSyncOperation operation = ConSanSyncOperation::Unknown;
  ConSanSyncAddressSource address_source = ConSanSyncAddressSource::NotApplicable;
  ConSanSyncMemoryRole memory_role = ConSanSyncMemoryRole::Unknown;
  ConSanSyncRmwOutcome rmw_outcome = ConSanSyncRmwOutcome::NotApplicable;
  ConSanSemanticConfidence confidence = ConSanSemanticConfidence::Unsupported;
  ConSanSemanticConfidence memory_role_confidence = ConSanSemanticConfidence::Unsupported;
  std::string confidence_reason;
  std::string identity;
  std::string code_object_fingerprint;
  std::string container_name;
  bool in_kernel = true;
  uint32_t occurrence = 0;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  uint32_t width_bits = 0;
  std::string mnemonic;
  std::optional<int64_t> static_byte_offset;
  std::optional<uint32_t> raw_scope;
  std::optional<int32_t> barrier_id;
  ConSanBarrierSite::OperandSource barrier_operand_source =
      ConSanBarrierSite::OperandSource::Unknown;
  std::optional<uint32_t> barrier_raw_operand_selector;
  std::optional<uint32_t> barrier_literal_width_bits;
  std::optional<uint64_t> barrier_literal_value;
  std::optional<uint16_t> barrier_raw_simm16;
  ConSanBarrierSite::Scope barrier_scope = ConSanBarrierSite::Scope::Unknown;
  std::optional<uint32_t> participant_count;
  std::optional<uint64_t> participant_mask;
  std::vector<ConSanExecutionOwner> execution_owners;
};

enum class ConSanSyncSequenceKind : uint8_t {
  Barrier,
  Fence,
  Atomic,
  OrdinaryMemory,
};

/// A logical synchronization operation associated from one or more decoded
/// events. Member identities stay ordered and the sequence confidence cannot
/// exceed its least-certain member.
struct ConSanSyncSequence {
  ConSanSyncSequenceKind kind = ConSanSyncSequenceKind::Fence;
  ConSanSyncOperation operation = ConSanSyncOperation::Unknown;
  ConSanSyncAddressSource address_source = ConSanSyncAddressSource::NotApplicable;
  ConSanSyncMemoryRole memory_role = ConSanSyncMemoryRole::Unknown;
  ConSanSyncRmwOutcome rmw_outcome = ConSanSyncRmwOutcome::NotApplicable;
  ConSanSemanticConfidence confidence = ConSanSemanticConfidence::Unsupported;
  ConSanSemanticConfidence memory_role_confidence = ConSanSemanticConfidence::Unsupported;
  std::string confidence_reason;
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  uint64_t begin_text_offset = 0;
  uint64_t end_text_offset = 0;
  /// Exact same-block gfx12 release-wait boundary associated with an atomic.
  /// This is not a standalone synchronization event: it is meaningful only
  /// as part of the containing atomic sequence.
  std::optional<uint64_t> release_wait_text_offset;
  std::optional<uint32_t> basic_block_index;
  bool in_cyclic_cfg_component = false;
  bool inside_scalar_clause = false;
  std::vector<std::string> member_event_identities;
  uint32_t width_bits = 0;
  std::optional<int64_t> static_byte_offset;
  std::optional<uint32_t> raw_scope;
  std::optional<int32_t> barrier_id;
  ConSanBarrierSite::OperandSource barrier_operand_source =
      ConSanBarrierSite::OperandSource::Unknown;
  std::optional<uint32_t> barrier_raw_operand_selector;
  std::optional<uint32_t> barrier_literal_width_bits;
  std::optional<uint64_t> barrier_literal_value;
  std::optional<uint16_t> barrier_raw_simm16;
  ConSanBarrierSite::Scope barrier_scope = ConSanBarrierSite::Scope::Unknown;
  std::optional<uint32_t> participant_count;
  std::optional<uint64_t> participant_mask;
  std::vector<ConSanExecutionOwner> execution_owners;
};

enum class ConSanCommunicationAddressKind : uint8_t {
  Atomic,
  Ordinary,
};

/// Typed fail-closed outcome for a synchronization sequence's normalized
/// communication address. A supported recipe proves that the complete source
/// address remains live at the first instruction after the sequence and that
/// one dead VGPR pair is available there for engine-specific materialization.
enum class ConSanCommunicationAddressSupport : uint8_t {
  Supported,
  UnsupportedSequence,
  AmbiguousCommunicationEvent,
  UnsupportedAddressSource,
  MissingAddressOperands,
  InvalidStaticOffset,
  MissingPostSequenceInstruction,
  AddressNotLiveAfterSequence,
  MissingOwners,
  InvalidOwnerDescriptor,
  NoPostSequenceScratchPair,
  ScratchOutsideOwnerAllocation,
};

struct ConSanCommunicationAddressRecipe {
  ConSanCommunicationAddressSupport support =
      ConSanCommunicationAddressSupport::UnsupportedSequence;
  ConSanCommunicationAddressKind kind = ConSanCommunicationAddressKind::Atomic;
  std::string sequence_identity;
  std::string communication_event_identity;
  ConSanSyncAddressSource address_source = ConSanSyncAddressSource::Unknown;
  uint64_t post_sequence_text_offset = 0;
  uint16_t address_vgpr = 0;
  uint16_t address_vgpr_count = 0;
  std::optional<uint16_t> address_sgpr;
  uint16_t address_sgpr_count = 0;
  int64_t static_byte_offset = 0;
  std::optional<uint16_t> scratch_vgpr;
  uint16_t scratch_vgpr_count = 0;
  std::vector<uint64_t> owner_descriptor_file_offsets;

  [[nodiscard]] bool supported() const {
    return support == ConSanCommunicationAddressSupport::Supported;
  }
};

/// A cache/fence instruction conservatively admitted for future MOI fence
/// record emission. Selection never invents a numeric communication token:
/// `communication_event_identity` names the unique address-bearing atomic
/// event whose effective address a later emitter must capture exactly.
struct ConSanMoiFenceCandidate {
  std::string identity;
  std::string sequence_identity;
  std::string fence_event_identity;
  std::string communication_event_identity;
  std::string container_name;
  bool in_kernel = true;
  uint64_t text_offset = 0;
  uint64_t file_offset = 0;
  uint32_t size = 0;
  ConSanSyncMemoryRole memory_role = ConSanSyncMemoryRole::Unknown;
  ConSanSyncAddressSource communication_address_source = ConSanSyncAddressSource::NotApplicable;
  std::optional<int64_t> communication_static_byte_offset;
  std::optional<uint32_t> raw_scope;
  bool eligible = false;
  std::string rejection_reason;
};

/// Checks the non-instruction metadata required before an ordinary load and
/// `global_inv` may form a conservative acquire sequence. Exposed for bounded
/// corruption tests; production association additionally proves the exact
/// intervening instruction run.
[[nodiscard]] bool consan_ordinary_acquire_metadata_compatible(
    const ConSanSyncEvent &load, const ConSanSyncSequence &load_sequence,
    const ConSanSyncEvent &cache, const ConSanSyncSequence &cache_sequence);

/// Checks the non-instruction metadata required before a `global_wb` and
/// ordinary store may form a conservative release sequence. Production
/// association additionally proves one exact same-block instruction run.
[[nodiscard]] bool consan_ordinary_release_metadata_compatible(
    const ConSanSyncEvent &cache, const ConSanSyncSequence &cache_sequence,
    const ConSanSyncEvent &store, const ConSanSyncSequence &store_sequence);

/// One admitted sequence boundary. The anchor remains an original sequence
/// member: release delays are placed before the first member and acquire
/// delays after the last member, never inside a synchronization sequence.
