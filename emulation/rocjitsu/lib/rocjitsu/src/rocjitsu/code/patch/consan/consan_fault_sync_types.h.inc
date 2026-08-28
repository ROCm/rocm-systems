enum class ConSanFaultSiteKind : uint8_t {
  Barrier,
  Atomic,
  LdsAccess,
  OrdinaryMemory,
};

/// Semantic corruption selected by ConSan's validation-only fault planner.
///
/// These values describe changes to the guest program, not patch encodings.
/// Architecture-specific lowering can therefore consume the same intent while
/// choosing different instruction rewrites. A new value must identify a
/// distinct behavioral fault and define its required payload in
/// `ConSanFaultMutationPlan::well_formed`.
enum class ConSanFaultMutationKind : uint8_t {
  /// Remove one barrier instruction or one explicitly selected logical group.
  DropBarrier,
  /// Move both members of a qualified full-barrier sequence as one operation.
  MoveBarrierPair,
  /// Retarget every member of one logical barrier to a different ID and scope.
  BarrierIdScope,
  /// Change the statically initialized participant count of one lifecycle.
  BarrierParticipantCount,
  /// Shift an atomic instruction's encoded address by a planned byte delta.
  AtomicWrongAddress,
  /// Remove one selected release or acquire boundary around an atomic.
  AtomicWeakenOrder,
  /// Lower an atomic instruction's synchronization scope to wave scope.
  AtomicWeakenScope,
  /// Replace the address VGPR of one LDS instruction.
  LdsWrongAddress,
  /// Shift an ordinary memory instruction's encoded address by a byte delta.
  OrdinaryWrongAddress,
  /// Remove the cache boundary from one qualified ordinary acquire sequence.
  OrdinaryWeakenOrder,
  /// Lower an ordinary memory instruction's synchronization scope to wave.
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

/// Immutable planner-to-emitter contract for one validation-only corruption.
///
/// The planner binds a semantic fault to a pristine code object, stable site
/// identities, and every value needed to perform the rewrite. The emitter may
/// validate those facts against `ProgramInventory`, but must not select a
/// different site from a caller index or reconstruct a target value from the
/// original `MutationRequest`. Generic identity fields are shared because a
/// barrier, atomic, and ordinary-memory fault all refer to the same semantic
/// inventory vocabulary; kind-specific payloads remain optional and are
/// admitted only in the combinations checked by `well_formed()`.
struct ConSanFaultMutationPlan {
  /// Content identity of the pristine image whose inventory produced this
  /// plan. Applying the plan to any other image is a stale-plan error even if
  /// a diagnostic identity string happens to collide.
  ConSanCodeObjectId source_code_object;
  /// Behavioral corruption to perform.
  ConSanFaultMutationKind kind = ConSanFaultMutationKind::DropBarrier;
  /// Stable diagnostic identity of the instruction selected as the principal
  /// mutation site.
  std::string primary_identity;
  /// Second instruction or ordering-boundary identity when the behavioral
  /// fault is defined by a pair.
  std::optional<std::string> companion_identity;
  /// Stable logical sequence or lifecycle identity whose semantics the
  /// mutation changes.
  std::optional<std::string> logical_sequence_identity;
  /// Exact inventory sequence through which the primary site was selected.
  /// This differs from `logical_sequence_identity` for a lifecycle or grouped
  /// mutation whose externally meaningful identity spans multiple sequences.
  std::optional<std::string> selection_sequence_identity;
  /// Second exact inventory sequence selected by a grouped barrier drop.
  std::optional<std::string> companion_selection_sequence_identity;
  /// Original-program member identities in semantic order. This lets
  /// validation prove that a pair or lifecycle still has the shape selected
  /// by planning without selecting it again.
  std::vector<std::string> ordered_member_identities;
  /// Exact real-instruction destination for a markerless barrier move.
  std::optional<std::string> destination_identity;
  /// Direction of a whole-barrier move. `LegacyMarker` identifies the older
  /// marker-based form, which has no real-instruction destination.
  ConSanBarrierMoveDirection barrier_move_direction = ConSanBarrierMoveDirection::LegacyMarker;
  /// Control-flow proof under which a markerless destination was admitted.
  ConSanBarrierMoveCfgContract barrier_move_cfg_contract = ConSanBarrierMoveCfgContract::SameBlock;
  /// Structured-CFG coordinates retained only for a cross-block barrier move;
  /// together they bind lowering to the exact proof established by analysis.
  std::optional<uint32_t> structured_guard_block_index;
  std::optional<uint32_t> structured_destination_block_index;
  std::optional<uint32_t> structured_source_block_index;
  std::optional<uint64_t> structured_guard_offset;
  std::optional<uint64_t> structured_destination_offset;
  std::optional<uint64_t> structured_source_offset;
  /// Barrier ID and scope before and after an ID/scope or participant-count
  /// mutation. Participant mutation retains equal values to prove that it
  /// changes cardinality only.
  std::optional<int32_t> original_barrier_id;
  std::optional<int32_t> target_barrier_id;
  ConSanBarrierSite::Scope original_barrier_scope = ConSanBarrierSite::Scope::Unknown;
  ConSanBarrierSite::Scope target_barrier_scope = ConSanBarrierSite::Scope::Unknown;
  /// Participant cardinality before and after a lifecycle-count mutation.
  std::optional<uint32_t> original_participant_count;
  std::optional<uint32_t> target_participant_count;
  /// Positive aligned byte displacement used by atomic and ordinary
  /// wrong-address mutations. No other mutation kind may carry it.
  std::optional<uint32_t> address_delta_bytes;
  /// Original and replacement encoded LDS address registers. Retaining both
  /// makes the selected corruption independently inspectable and lets lowering
  /// reject a stale decoded source operand.
  std::optional<uint16_t> original_address_vgpr;
  std::optional<uint16_t> target_address_vgpr;
  /// Exact ordering edge selected for an atomic weaken-order mutation.
  std::optional<ConSanAtomicOrderEdge> atomic_order_edge;
  /// Whether a single-instruction barrier drop was deliberately admitted even
  /// though it removes only half of a qualified logical barrier.
  bool destructive_incomplete_barrier_drop = false;

  /// Return whether all common identities and kind-specific payloads form one
  /// self-consistent semantic mutation. This is intentionally independent of
  /// any particular `ProgramInventory`; application separately proves that the
  /// source identity and referenced sites belong to its immutable inventory.
  [[nodiscard]] bool well_formed() const;

  bool operator==(const ConSanFaultMutationPlan &) const = default;
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
  /// Stable original-program identity shared by every symbol alias that
  /// decodes this same physical synchronization instruction.
  SemanticSiteId semantic_id;
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
  /// Every symbol-container name through which this physical event was
  /// decoded. Analysis canonicalizes equivalent aliases into one event while
  /// preserving this attribution for policy diagnostics and filtering.
  std::vector<std::string> source_containers;
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
  /// Exact same-block architecture-specific release-wait boundary associated
  /// with an atomic or ordinary store. This is not a standalone
  /// synchronization event: it is meaningful only as part of the containing
  /// communication sequence.
  std::optional<uint64_t> release_wait_text_offset;
  std::optional<uint32_t> basic_block_index;
  bool in_cyclic_cfg_component = false;
  bool inside_scalar_clause = false;
  /// Exact scalar-clause instruction covering the addressed communication
  /// instruction, when one exists. Engines that replace the communication
  /// instruction with a trampoline must neutralize this scheduling hint: an
  /// s_clause may not have a control-flow instruction inserted between it and
  /// the memory operations that it covers.
  std::optional<uint64_t> scalar_clause_text_offset;
  /// Stable identities of the original synchronization events in semantic
  /// order. These typed members are authoritative for joins; the adjacent
  /// string identities remain a temporary diagnostics/lowering adapter.
  std::vector<SemanticSiteId> member_semantic_ids;
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

/// Analysis outcome for associating one cache/fence instruction with an
/// address-bearing synchronization sequence.
///
/// This is an inventory fact, not an engine decision. `Qualified` means that
/// shared analysis proved one sequence, one communication event, a compatible
/// memory role, address source, and scope. Every other value preserves the
/// first failed invariant. Policy can therefore make a typed disposition
/// without interpreting human-readable diagnostics or rerunning association.
enum class ConSanFenceAssociation : uint8_t {
  /// The fence has one complete addressed communication association.
  Qualified,
  /// The fence has no unique synchronization-sequence membership.
  MissingOrAmbiguousSequence,
  /// Its sequence is not an addressed atomic or ordinary-memory operation.
  NotAddressedCommunication,
  /// Sequence or memory-role confidence is below the supported threshold.
  InsufficientConfidence,
  /// An acquire-pair prefix is intentionally represented by its tail fence.
  AcquirePairPrefixCoveredByTail,
  /// The decoded cache operation has no supported acquire/release role.
  UnsupportedCacheOperation,
  /// The fence role is not carried by the associated communication sequence.
  MemoryRoleMismatch,
  /// The sequence has no unique address-bearing communication member.
  MissingOrAmbiguousCommunicationEvent,
  /// The communication event uses an unsupported address source.
  UnsupportedAddressSource,
  /// The communication event has no supported non-wave scope.
  UnsupportedScope,
  /// One-past-the-end iteration bound; never an analysis outcome.
  Count,
};

/// Complete iterable set of fence-association outcomes.
inline constexpr std::array<ConSanFenceAssociation, 10> kConSanFenceAssociations = {
    ConSanFenceAssociation::Qualified,
    ConSanFenceAssociation::MissingOrAmbiguousSequence,
    ConSanFenceAssociation::NotAddressedCommunication,
    ConSanFenceAssociation::InsufficientConfidence,
    ConSanFenceAssociation::AcquirePairPrefixCoveredByTail,
    ConSanFenceAssociation::UnsupportedCacheOperation,
    ConSanFenceAssociation::MemoryRoleMismatch,
    ConSanFenceAssociation::MissingOrAmbiguousCommunicationEvent,
    ConSanFenceAssociation::UnsupportedAddressSource,
    ConSanFenceAssociation::UnsupportedScope,
};

/// Return the stable diagnostic spelling of a fence-association outcome.
[[nodiscard]] constexpr std::string_view
consan_fence_association_name(ConSanFenceAssociation association) {
  switch (association) {
  case ConSanFenceAssociation::Qualified:
    return "qualified";
  case ConSanFenceAssociation::MissingOrAmbiguousSequence:
    return "missing-or-ambiguous-sequence-membership";
  case ConSanFenceAssociation::NotAddressedCommunication:
    return "fence-not-associated-with-addressed-communication-sequence";
  case ConSanFenceAssociation::InsufficientConfidence:
    return "communication-sequence-confidence-insufficient";
  case ConSanFenceAssociation::AcquirePairPrefixCoveredByTail:
    return "paired-acquire-prefix-covered-by-tail";
  case ConSanFenceAssociation::UnsupportedCacheOperation:
    return "unsupported-cache-operation";
  case ConSanFenceAssociation::MemoryRoleMismatch:
    return "cache-role-not-carried-by-communication-sequence";
  case ConSanFenceAssociation::MissingOrAmbiguousCommunicationEvent:
    return "missing-or-ambiguous-communication-event";
  case ConSanFenceAssociation::UnsupportedAddressSource:
    return "communication-address-source-unsupported";
  case ConSanFenceAssociation::UnsupportedScope:
    return "communication-scope-unsupported";
  case ConSanFenceAssociation::Count:
    break;
  }
  return "invalid-fence-association";
}

static_assert(kConSanFenceAssociations.size() ==
              static_cast<size_t>(ConSanFenceAssociation::Count));

/// Immutable association between one cache/fence event and an addressed
/// synchronization sequence.
///
/// The two typed event identities refer back to the authoritative
/// `ConSanSyncEvent` inventory. This record deliberately does not copy the
/// events' container, instruction, address, or scope facts. A qualified record
/// names the unique address-bearing atomic or ordinary-memory event whose
/// effective address a later policy may request. Non-qualified records retain
/// no communication event and remain in inventory so policy can explain why
/// the fence is outside the observation contract.
struct ConSanMoiFenceCandidate {
  /// Original cache/fence event being associated.
  SemanticSiteId fence_event;
  /// Stable identity of the unique synchronization sequence, when one exists.
  std::string sequence_identity;
  /// Unique address-bearing event selected from the sequence when qualified.
  std::optional<SemanticSiteId> communication_event;
  /// Release/acquire role contributed by this fence within the sequence.
  ConSanSyncMemoryRole memory_role = ConSanSyncMemoryRole::Unknown;
  /// Typed result of shared sequence association.
  ConSanFenceAssociation association = ConSanFenceAssociation::MissingOrAmbiguousSequence;

  /// Return whether all analysis-owned association invariants were proved.
  [[nodiscard]] constexpr bool eligible() const {
    return association == ConSanFenceAssociation::Qualified;
  }
};

/// A contiguous, statically associated barrier lifecycle retained in semantic
/// inventory.
///
/// On gfx12-class targets `s_barrier_leave` has no explicit barrier operand and
/// drops the last named barrier joined by the thread. A group is therefore
/// admissible only when its ordered typed member identities and one matching
/// static join prove that association. Rejected groups remain in inventory so
/// later policy can explain the unsupported lifecycle without rediscovery.
struct ConSanBarrierLifecycleGroup {
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  std::optional<uint32_t> basic_block_index;
  uint64_t begin_text_offset = 0;
  uint64_t end_text_offset = 0;
  /// Stable original-program identities of every event in lifecycle order.
  std::vector<SemanticSiteId> member_semantic_ids;
  /// Temporary string projection retained for current fault and lowerer
  /// consumers until their typed-identity cutover.
  std::vector<std::string> member_event_identities;
  std::optional<int32_t> barrier_id;
  ConSanBarrierSite::Scope barrier_scope = ConSanBarrierSite::Scope::Unknown;
  bool admissible = false;
  ConSanSemanticConfidence confidence = ConSanSemanticConfidence::Unsupported;
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
