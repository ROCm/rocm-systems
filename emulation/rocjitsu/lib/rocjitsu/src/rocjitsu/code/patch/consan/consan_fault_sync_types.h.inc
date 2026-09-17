struct ProgramSite;

enum class FaultSiteKind : uint8_t {
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
enum class FaultMutationKind : uint8_t {
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

enum class SyncMemoryRole : uint8_t {
  Unknown,
  None,
  Acquire,
  Release,
  AcquireRelease,
  SequentiallyConsistent,
};

enum class SemanticConfidence : uint8_t {
  Exact,
  Conservative,
  Ambiguous,
  Unsupported,
};

enum class OwnerProofKind : uint8_t {
  KernelLocal,
  DirectCall,
  RecoveredIndirectCall,
};

struct ExecutionOwner {
  /// Stable kernel handle in the owning ProgramInventory revision.
  ///
  /// Descriptor offsets are physical locations and can change when a prior
  /// mutation rewrites the ELF.  Keeping the semantic owner as an arena
  /// handle prevents analysis products from treating those locations as
  /// cross-revision identities.  Lowering resolves the current descriptor
  /// only when it must read or modify descriptor bytes.
  ProgramContainerId kernel;
  OwnerProofKind proof = OwnerProofKind::KernelLocal;

  bool operator==(const ExecutionOwner &) const = default;
};

[[nodiscard]] inline bool execution_owners_equal(std::span<const ExecutionOwner> lhs,
                                                 std::span<const ExecutionOwner> rhs) {
  return std::ranges::equal(lhs, rhs);
}

[[nodiscard]] inline bool nonempty_execution_owners_equal(std::span<const ExecutionOwner> lhs,
                                                          std::span<const ExecutionOwner> rhs) {
  return !lhs.empty() && execution_owners_equal(lhs, rhs);
}

/// Owned presentation snapshot for one currently injectable instruction.
///
/// This is a diagnostic product, not the analysis representation. The live
/// fault inventory retains a source-site handle and derives these facts from
/// ProgramInventory when diagnostics are published.
struct FaultSitePresentation : DecodedSite {
  FaultSiteKind kind = FaultSiteKind::Barrier;
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  uint32_t occurrence = 0;
  uint32_t width_bits = 0;
  std::string semantic_role;
  std::string decoded_operands;
  OrdinaryMemorySupportReason ordinary_memory_support_reason =
      OrdinaryMemorySupportReason::NotApplicable;
  SemanticConfidence sync_confidence = SemanticConfidence::Unsupported;
  SyncMemoryRole sync_memory_role = SyncMemoryRole::Unknown;
  std::vector<ExecutionOwner> execution_owners;

  bool operator==(const FaultSitePresentation &) const = default;
};

/// A currently injectable facet of one authoritative program site.
///
/// Location, container, decoded operands, execution owners, and
/// synchronization semantics remain owned by ProgramInventory. Keeping only
/// selection identity and irreducible target state here prevents the fault
/// planner from becoming a second program representation.
struct FaultSite {
  FaultSiteKind kind = FaultSiteKind::Barrier;
  std::string identity;
  uint32_t occurrence = 0;
  /// Exact decoded instruction from which this injectable projection was
  /// derived.
  ProgramSiteId source_site;
  /// Target-defined selectable-bank state at this instruction.
  std::optional<uint16_t> selectable_vgpr_bank_mode;

  bool operator==(const FaultSite &) const = default;
};

/// Stable presentation facts for one selected validation-only corruption.
///
/// These fields describe the externally meaningful choice. Planning refines
/// them with source binding and exact application proof below, while runtime
/// diagnostics copy this facet without acquiring those mutation authorities.
struct FaultMutationPresentation {
  FaultMutationKind kind = FaultMutationKind::DropBarrier;
  std::string primary_identity;
  std::optional<std::string> companion_identity;
  std::optional<std::string> logical_sequence_identity;
  std::vector<std::string> ordered_member_identities;
  std::optional<std::string> destination_identity;
  BarrierMoveDirection barrier_move_direction = BarrierMoveDirection::LegacyMarker;
  BarrierMoveCfgContract barrier_move_cfg_contract = BarrierMoveCfgContract::SameBlock;
  std::optional<int32_t> original_barrier_id;
  std::optional<int32_t> target_barrier_id;
  BarrierSite::Scope original_barrier_scope = BarrierSite::Scope::Unknown;
  BarrierSite::Scope target_barrier_scope = BarrierSite::Scope::Unknown;
  std::optional<uint16_t> target_address_vgpr;

  bool operator==(const FaultMutationPresentation &) const = default;
};

/// Diagnostic description of one validation-only corruption admitted by the
/// private resolution-and-mutation transaction.
struct FaultMutationPlan : FaultMutationPresentation {
  bool operator==(const FaultMutationPlan &) const = default;
};

enum class BarrierMoveDestinationIssue : uint8_t {
  None,
  InsideScalarClause,
  BarrierSourceOrLifecycle,
  FenceOperation,
  NotMemoryOperation,
  NotRelocatable,
  Count,
};

[[nodiscard]] std::string barrier_move_destination_issue_message(BarrierMoveDestinationIssue issue,
                                                                 std::string_view detail = {});

/// Stable presentation facts for a real-instruction barrier-move destination.
///
/// Synchronization analysis adds the private code-object binding below;
/// runtime diagnostics copy this facet directly and cannot apply a move.
struct BarrierMoveDestinationPresentation : DecodedSite {
  std::string identity;
  std::string container_name;
  bool in_kernel = true;
  uint32_t basic_block_index = 0;
  bool memory_operation = false;
  BarrierMoveDestinationIssue issue = BarrierMoveDestinationIssue::None;
  /// Decoder/relocator detail attached only to `NotRelocatable`. The stable
  /// semantic decision remains the typed issue above.
  std::string issue_detail;
  BarrierMoveCfgContract cfg_contract = BarrierMoveCfgContract::SameBlock;
  /// Present only for a narrowly proven acyclic structured CFG contract.
  std::optional<uint32_t> structured_guard_block_index;
  std::optional<uint32_t> structured_source_block_index;
  std::optional<uint64_t> structured_guard_offset;
  std::optional<uint64_t> structured_source_offset;
  std::vector<ExecutionOwner> execution_owners;

  [[nodiscard]] bool suitable() const { return issue == BarrierMoveDestinationIssue::None; }
  bool operator==(const BarrierMoveDestinationPresentation &) const = default;
};

/// Selection and diagnostics consume the same complete destination facts;
/// code-object identity is already carried by the owning program inventory.
using BarrierMoveDestination = BarrierMoveDestinationPresentation;

/// Semantic category shared by a decoded synchronization event and every
/// logical sequence derived from it.
enum class SyncKind : uint8_t {
  Barrier,
  Fence,
  Atomic,
  OrdinaryMemory,
};

enum class SyncOperation : uint8_t {
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

enum class SyncAddressSource : uint8_t {
  NotApplicable,
  Unknown,
  LdsVector,
  FlatVector,
  GlobalScalarVector,
  BufferResource,
  ScratchVector,
};

enum class SyncRmwOutcome : uint8_t {
  NotApplicable,
  Unknown,
  NoReturn,
  ReturnsOldValue,
  CompareExchange,
};

/// Returns the weaker of two semantic claims. Association must never improve
/// the confidence of an input event.
[[nodiscard]] constexpr SemanticConfidence combine_sync_confidence(SemanticConfidence lhs,
                                                                   SemanticConfidence rhs) {
  return static_cast<uint8_t>(lhs) >= static_cast<uint8_t>(rhs) ? lhs : rhs;
}

/// Semantic claim shared by one decoded synchronization event and the logical
/// sequence derived from it. Sequence association may refine this value, but
/// never maintains a second field vocabulary or projects its members one fact
/// at a time.
struct SyncSemantics {
  SyncKind kind = SyncKind::Fence;
  SyncOperation operation = SyncOperation::Unknown;
  SyncAddressSource address_source = SyncAddressSource::NotApplicable;
  SyncMemoryRole memory_role = SyncMemoryRole::Unknown;
  SyncRmwOutcome rmw_outcome = SyncRmwOutcome::NotApplicable;
  SemanticConfidence confidence = SemanticConfidence::Unsupported;
  SemanticConfidence memory_role_confidence = SemanticConfidence::Unsupported;
  std::string confidence_reason;
  std::optional<MemoryScope> scope;

  bool operator==(const SyncSemantics &) const = default;
};

/// Stable handle into one immutable synchronization-event arena.
///
/// Sequence topology is inventory-local, so it uses this compact handle
/// instead of copying content-stable code-object identities onto every edge.
/// Stable semantic identities remain on events for diagnostics, mutation
/// selection, and other boundaries that outlive one inventory revision.
struct SyncEventId {
  static constexpr uint32_t kInvalidOrdinal = ~uint32_t{0};
  uint32_t ordinal = kInvalidOrdinal;

  [[nodiscard]] bool valid() const { return ordinal != kInvalidOrdinal; }

  auto operator<=>(const SyncEventId &) const = default;
};

/// Stable handle into one immutable synchronization-sequence arena.
struct SyncSequenceId {
  static constexpr uint32_t kInvalidOrdinal = ~uint32_t{0};
  uint32_t ordinal = kInvalidOrdinal;

  [[nodiscard]] bool valid() const { return ordinal != kInvalidOrdinal; }

  auto operator<=>(const SyncSequenceId &) const = default;
};

/// Published unique-sequence membership for one synchronization event.
/// Missing membership has an invalid sequence; ambiguous membership retains
/// that distinction so consumers fail closed without rescanning the graph.
struct SyncSequenceMembership {
  SyncSequenceId sequence;
  bool ambiguous = false;
};

/// Mode-independent decoded synchronization semantics. Unknown information
/// stays explicit so later sequence association cannot silently invent it.
struct SyncEvent : SyncSemantics {
  /// Stable original-program identity shared by every symbol alias that
  /// decodes this same physical synchronization instruction.
  SemanticSiteId semantic_id;
  /// Exact source record from which synchronization analysis derived this
  /// event. Unlike container/name/offset joins, this remains unambiguous when
  /// several symbols alias one physical instruction.
  ProgramSiteId source_site;
  std::string identity;
  /// Original text position is part of the event's canonical physical
  /// identity. Keeping a second mutable copy allowed the two identities to
  /// disagree after an event was assembled or copied.
  [[nodiscard]] uint64_t text_offset() const { return semantic_id.physical.original_text_offset; }
};

/// A logical synchronization operation associated from one or more decoded
/// events. Member identities stay ordered and the sequence confidence cannot
/// exceed its least-certain member.
struct SyncSequence : SyncSemantics {
  std::string identity;
  uint64_t begin_text_offset = 0;
  uint64_t end_text_offset = 0;
  /// Exact same-block architecture-specific release-wait boundary associated
  /// with an atomic or ordinary store. This is not a standalone
  /// synchronization event: it is meaningful only as part of the containing
  /// communication sequence.
  std::optional<uint64_t> release_wait_text_offset;
  /// Header of an exactly proven compiler polling loop whose successful exit
  /// reaches this acquire sequence's fence. A consumer that relocates the
  /// acquire must include this header so the loop backedge cannot re-enter a
  /// partially executed instrumentation prologue.
  std::optional<uint64_t> acquire_polling_loop_header_text_offset;
  std::optional<uint32_t> basic_block_index;
  bool in_cyclic_cfg_component = false;
  bool inside_scalar_clause = false;
  /// Exact scalar-clause instruction covering the addressed communication
  /// instruction, when one exists. Modes that replace the communication
  /// instruction with a trampoline must neutralize this scheduling hint: an
  /// s_clause may not have a control-flow instruction inserted between it and
  /// the memory operations that it covers.
  std::optional<uint64_t> scalar_clause_text_offset;
  /// Inventory-local handles of the original synchronization events in
  /// semantic order. Stable names are resolved only at presentation and
  /// mutation-selection boundaries.
  std::vector<SyncEventId> member_event_ids;
  std::optional<int32_t> barrier_id;
  BarrierSite::OperandSource barrier_operand_source = BarrierSite::OperandSource::Unknown;
  BarrierSite::Scope barrier_scope = BarrierSite::Scope::Unknown;
  std::optional<uint32_t> participant_count;
  std::optional<uint64_t> participant_mask;
};

/// Analysis outcome for associating one cache/fence instruction with an
/// address-bearing synchronization sequence.
///
/// This is an inventory fact, not an mode decision. `Qualified` means that
/// shared analysis proved one sequence, one communication event, a compatible
/// memory role, address source, and scope. Every other value preserves the
/// first failed invariant. Policy can therefore make a typed disposition
/// without interpreting human-readable diagnostics or rerunning association.
enum class FenceAssociation : uint8_t {
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
/// `SyncEvent` inventory. This record deliberately does not copy the
/// events' container, instruction, address, or scope facts. A qualified record
/// names the unique address-bearing atomic or ordinary-memory event whose
/// effective address a later policy may request. Non-qualified records retain
/// no communication event and remain in inventory so policy can explain why
/// the fence is outside the observation contract.
struct FenceCandidate {
  /// Original cache/fence event being associated.
  SyncEventId fence_event;
  /// Inventory-local handle of the unique synchronization sequence, when one
  /// exists. Stable text identity is materialized only at external plan and
  /// diagnostic boundaries.
  SyncSequenceId sequence;
  /// Unique address-bearing event selected from the sequence when qualified.
  std::optional<SyncEventId> communication_event;
  /// Release/acquire role contributed by this fence within the sequence.
  SyncMemoryRole memory_role = SyncMemoryRole::Unknown;
  /// Typed result of shared sequence association.
  FenceAssociation association = FenceAssociation::MissingOrAmbiguousSequence;

  /// Return whether all analysis-owned association invariants were proved.
  [[nodiscard]] constexpr bool eligible() const {
    return association == FenceAssociation::Qualified;
  }
};

enum class BarrierLifecycleIssue : uint8_t {
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

[[nodiscard]] std::string_view barrier_lifecycle_issue_message(BarrierLifecycleIssue issue);

/// A contiguous, statically associated barrier lifecycle retained in semantic
/// inventory.
///
/// On RDNA4 `s_barrier_leave` has no explicit barrier operand and
/// drops the last named barrier joined by the thread. A group is therefore
/// admissible only when its ordered typed member identities and one matching
/// static join prove that association. Rejected groups remain in inventory so
/// later policy can explain the unsupported lifecycle without rediscovery.
/// The initialization member remains authoritative for the lifecycle's stable
/// name, source container, barrier ID, and scope.
struct BarrierLifecycleGroup {
  /// Inventory-local handles of every event in lifecycle order.
  std::vector<SyncEventId> member_event_ids;
  BarrierLifecycleIssue issue = BarrierLifecycleIssue::Count;

  [[nodiscard]] bool admissible() const { return issue == BarrierLifecycleIssue::None; }
};

/// Checks the non-instruction metadata required before an ordinary load and
/// `global_inv` may form a conservative acquire sequence. Exposed for bounded
/// corruption tests; production association additionally proves the exact
/// intervening instruction run.
[[nodiscard]] bool ordinary_acquire_metadata_compatible(std::span<const ProgramSite> program_sites,
                                                        const SyncEvent &load,
                                                        const SyncSequence &load_sequence,
                                                        const SyncEvent &cache,
                                                        const SyncSequence &cache_sequence);

/// Checks the non-instruction metadata required before a `global_wb` and
/// ordinary store may form a conservative release sequence. Production
/// association additionally proves one exact same-block instruction run.
[[nodiscard]] bool ordinary_release_metadata_compatible(std::span<const ProgramSite> program_sites,
                                                        const SyncEvent &cache,
                                                        const SyncSequence &cache_sequence,
                                                        const SyncEvent &store,
                                                        const SyncSequence &store_sequence);

/// One admitted sequence boundary. The anchor remains an original sequence
/// member: release delays are placed before the first member and acquire
/// delays after the last member, never inside a synchronization sequence.
