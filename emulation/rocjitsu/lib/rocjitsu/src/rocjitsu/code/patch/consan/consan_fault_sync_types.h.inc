struct ConSanProgramSite;

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
/// distinct behavioral fault.
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

  bool operator==(const ConSanExecutionOwner &) const = default;
};

/// Stable presentation facts for one currently injectable instruction.
///
/// Completed analysis products inherit this value and add only application-
/// private identity and operands. Runtime diagnostics copy this facet without
/// maintaining a second field-by-field schema.
struct ConSanFaultSitePresentation : ConSanDecodedSite {
  ConSanFaultSiteKind kind = ConSanFaultSiteKind::Barrier;
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  uint32_t occurrence = 0;
  uint32_t width_bits = 0;
  std::string semantic_role;
  std::string decoded_operands;
  ConSanOrdinaryMemorySupportReason ordinary_memory_support_reason =
      ConSanOrdinaryMemorySupportReason::NotApplicable;
  std::optional<std::string> sync_event_identity;
  std::optional<std::string> sync_sequence_identity;
  ConSanSemanticConfidence sync_confidence = ConSanSemanticConfidence::Unsupported;
  ConSanSyncMemoryRole sync_memory_role = ConSanSyncMemoryRole::Unknown;
  std::vector<ConSanExecutionOwner> execution_owners;

  bool operator==(const ConSanFaultSitePresentation &) const = default;
};

/// A currently injectable instruction with its application-private proof
/// identity and target operand state.
struct ConSanFaultSite : ConSanFaultSitePresentation {
  /// Exact decoded instruction from which this injectable projection was
  /// derived. Synchronization annotation follows this handle instead of
  /// reconstructing source identity from copied names and offsets.
  ConSanProgramSiteId source_site;
  ConSanProgramContainerId container_id;
  std::optional<uint16_t> address_vgpr;
  /// Target-defined selectable-bank state at this instruction.
  std::optional<uint16_t> selectable_vgpr_bank_mode;
};

/// Stable presentation facts for one selected validation-only corruption.
///
/// These fields describe the externally meaningful choice. Planning refines
/// them with source binding and exact application proof below, while runtime
/// diagnostics copy this facet without acquiring those mutation authorities.
struct ConSanFaultMutationPresentation {
  ConSanFaultMutationKind kind = ConSanFaultMutationKind::DropBarrier;
  std::string primary_identity;
  std::optional<std::string> companion_identity;
  std::optional<std::string> logical_sequence_identity;
  std::vector<std::string> ordered_member_identities;
  std::optional<std::string> destination_identity;
  ConSanBarrierMoveDirection barrier_move_direction = ConSanBarrierMoveDirection::LegacyMarker;
  ConSanBarrierMoveCfgContract barrier_move_cfg_contract = ConSanBarrierMoveCfgContract::SameBlock;
  std::optional<int32_t> original_barrier_id;
  std::optional<int32_t> target_barrier_id;
  ConSanBarrierSite::Scope original_barrier_scope = ConSanBarrierSite::Scope::Unknown;
  ConSanBarrierSite::Scope target_barrier_scope = ConSanBarrierSite::Scope::Unknown;
  std::optional<uint16_t> target_address_vgpr;

  bool operator==(const ConSanFaultMutationPresentation &) const = default;
};

/// Diagnostic description of one validation-only corruption admitted by the
/// private resolution-and-mutation transaction.
struct ConSanFaultMutationPlan : ConSanFaultMutationPresentation {
  bool operator==(const ConSanFaultMutationPlan &) const = default;
};

enum class ConSanBarrierMoveDestinationIssue : uint8_t {
  None,
  InsideScalarClause,
  BarrierSourceOrLifecycle,
  FenceOperation,
  NotMemoryOperation,
  NotRelocatable,
  Count,
};

[[nodiscard]] std::string
consan_barrier_move_destination_issue_message(ConSanBarrierMoveDestinationIssue issue,
                                              std::string_view detail = {});

/// Stable presentation facts for a real-instruction barrier-move destination.
///
/// Synchronization analysis adds the private code-object binding below;
/// runtime diagnostics copy this facet directly and cannot apply a move.
struct ConSanBarrierMoveDestinationPresentation : ConSanDecodedSite {
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  uint32_t basic_block_index = 0;
  bool memory_operation = false;
  ConSanBarrierMoveDestinationIssue issue = ConSanBarrierMoveDestinationIssue::None;
  /// Decoder/relocator detail attached only to `NotRelocatable`. The stable
  /// semantic decision remains the typed issue above.
  std::string issue_detail;
  ConSanBarrierMoveCfgContract cfg_contract = ConSanBarrierMoveCfgContract::SameBlock;
  /// Present only for a narrowly proven acyclic structured CFG contract.
  std::optional<uint32_t> structured_guard_block_index;
  std::optional<uint32_t> structured_source_block_index;
  std::optional<uint64_t> structured_guard_offset;
  std::optional<uint64_t> structured_source_offset;
  std::vector<ConSanExecutionOwner> execution_owners;

  [[nodiscard]] bool suitable() const { return issue == ConSanBarrierMoveDestinationIssue::None; }
  bool operator==(const ConSanBarrierMoveDestinationPresentation &) const = default;
};

/// Selection and diagnostics consume the same complete destination facts;
/// code-object identity is already carried by the owning program inventory.
using ConSanBarrierMoveDestination = ConSanBarrierMoveDestinationPresentation;

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

/// Semantic claim shared by one decoded synchronization event and the logical
/// sequence derived from it. Sequence association may refine this value, but
/// never maintains a second field vocabulary or projects its members one fact
/// at a time.
struct ConSanSyncSemantics {
  ConSanSyncOperation operation = ConSanSyncOperation::Unknown;
  ConSanSyncAddressSource address_source = ConSanSyncAddressSource::NotApplicable;
  ConSanSyncMemoryRole memory_role = ConSanSyncMemoryRole::Unknown;
  ConSanSyncRmwOutcome rmw_outcome = ConSanSyncRmwOutcome::NotApplicable;
  ConSanSemanticConfidence confidence = ConSanSemanticConfidence::Unsupported;
  ConSanSemanticConfidence memory_role_confidence = ConSanSemanticConfidence::Unsupported;
  std::string confidence_reason;
  std::optional<ConSanMemoryScope> scope;

  bool operator==(const ConSanSyncSemantics &) const = default;
};

/// Flavor-independent decoded synchronization semantics. Unknown information
/// stays explicit so later sequence association cannot silently invent it.
struct ConSanSyncEvent : ConSanSyncSemantics {
  /// Stable original-program identity shared by every symbol alias that
  /// decodes this same physical synchronization instruction.
  SemanticSiteId semantic_id;
  /// Exact source record from which synchronization analysis derived this
  /// event. Unlike container/name/offset joins, this remains unambiguous when
  /// several symbols alias one physical instruction.
  ConSanProgramSiteId source_site;
  ConSanSyncEventKind kind = ConSanSyncEventKind::Fence;
  std::string identity;
  /// Every symbol-container name through which this physical event was
  /// decoded. Analysis canonicalizes equivalent aliases into one event while
  /// preserving this attribution for policy diagnostics and filtering.
  std::vector<std::string> source_containers;
  /// Original text position is part of the event's canonical physical
  /// identity. Keeping a second mutable copy allowed the two identities to
  /// disagree after an event was assembled or copied.
  [[nodiscard]] uint64_t text_offset() const { return semantic_id.physical.original_text_offset; }
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
struct ConSanSyncSequence : ConSanSyncSemantics {
  ConSanSyncSequenceKind kind = ConSanSyncSequenceKind::Fence;
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
  /// order. Diagnostic and mutation-selection names are resolved from the
  /// event arena only when crossing those presentation boundaries.
  std::vector<SemanticSiteId> member_semantic_ids;
  std::optional<int32_t> barrier_id;
  ConSanBarrierSite::OperandSource barrier_operand_source =
      ConSanBarrierSite::OperandSource::Unknown;
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

enum class ConSanBarrierLifecycleIssue : uint8_t {
  None,
  InitMissingStaticIdOrScope,
  NonContiguousRun,
  MemberIdOrScopeMismatch,
  MissingJoin,
  MissingCompletingBarrier,
  MissingLeave,
  InvalidLeaveEncoding,
  Count,
};

[[nodiscard]] std::string_view
consan_barrier_lifecycle_issue_message(ConSanBarrierLifecycleIssue issue);

/// A contiguous, statically associated barrier lifecycle retained in semantic
/// inventory.
///
/// On RDNA4 `s_barrier_leave` has no explicit barrier operand and
/// drops the last named barrier joined by the thread. A group is therefore
/// admissible only when its ordered typed member identities and one matching
/// static join prove that association. Rejected groups remain in inventory so
/// later policy can explain the unsupported lifecycle without rediscovery.
struct ConSanBarrierLifecycleGroup {
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  /// Stable original-program identities of every event in lifecycle order.
  std::vector<SemanticSiteId> member_semantic_ids;
  std::optional<int32_t> barrier_id;
  ConSanBarrierSite::Scope barrier_scope = ConSanBarrierSite::Scope::Unknown;
  ConSanBarrierLifecycleIssue issue = ConSanBarrierLifecycleIssue::Count;

  [[nodiscard]] bool admissible() const { return issue == ConSanBarrierLifecycleIssue::None; }
};

/// Checks the non-instruction metadata required before an ordinary load and
/// `global_inv` may form a conservative acquire sequence. Exposed for bounded
/// corruption tests; production association additionally proves the exact
/// intervening instruction run.
[[nodiscard]] bool consan_ordinary_acquire_metadata_compatible(
    std::span<const ConSanProgramSite> program_sites, const ConSanSyncEvent &load,
    const ConSanSyncSequence &load_sequence, const ConSanSyncEvent &cache,
    const ConSanSyncSequence &cache_sequence);

/// Checks the non-instruction metadata required before a `global_wb` and
/// ordinary store may form a conservative release sequence. Production
/// association additionally proves one exact same-block instruction run.
[[nodiscard]] bool consan_ordinary_release_metadata_compatible(
    std::span<const ConSanProgramSite> program_sites, const ConSanSyncEvent &cache,
    const ConSanSyncSequence &cache_sequence, const ConSanSyncEvent &store,
    const ConSanSyncSequence &store_sequence);

/// One admitted sequence boundary. The anchor remains an original sequence
/// member: release delays are placed before the first member and acquire
/// delays after the last member, never inside a synchronization sequence.
