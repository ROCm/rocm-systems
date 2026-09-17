// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Normalized address-space fact relevant to ConSan access policy.
///
/// The inventory distinguishes group memory, definitely non-group memory, and
/// unresolved provenance without prematurely applying an mode's strict or
/// likely admission policy.
enum class AccessAddressSpace : uint8_t {
  /// The access targets workgroup/cluster-visible LDS.
  Group,
  /// The access is known to target private, global, or another non-group
  /// address space.
  NonGroup,
  /// Analysis could not prove whether a FLAT access reaches group memory.
  Unresolved,
  /// One past the last address-space fact. This is an iteration bound and
  /// never a decoded value.
  Count,
};

/// Source of the normalized access address-space claim.
///
/// Confidence is stored separately because the same provenance mechanism can
/// yield an exact or conservative result. Consumers may select a confidence
/// threshold, but they may not rewrite this provenance to make a mode succeed.
enum class AccessProvenance : uint8_t {
  /// The instruction family itself fixes the access to LDS.
  NativeLdsOpcode,
  /// An encoded FLAT segment field fixes the address space.
  EncodedFlatSegment,
  /// Bounded pointer-flow analysis inferred the likely FLAT address space.
  PropagatedFlatPointer,
  /// Neither encoding nor pointer-flow analysis resolved the address space.
  UnresolvedFlatPointer,
  /// One past the last provenance source. This is an iteration bound and never
  /// a decoded source.
  Count,
};

/// Typed reason why part of a decoded access cannot participate in a complete
/// normalized semantic inventory.
///
/// These are analysis facts, not mode rejections. An mode-specific
/// unsupported mnemonic, sampling decision, or resource failure belongs to a
/// later policy or lowering result rather than this enum.
enum class InventoryExclusionReason : uint8_t {
  /// The decoded instruction is not an access kind modeled by ConSan.
  NonAccessInstruction,
  /// The instruction size is zero, unaligned, or otherwise unusable.
  InvalidInstructionSize,
  /// The decoded access width cannot describe a nonempty byte range.
  InvalidAccessWidth,
  /// An address-bearing form lacks its decoded address operand.
  MissingAddressOperand,
  /// A multi-range instruction's encoded offsets were outside the input image
  /// or could not be normalized.
  RangeEncodingUnavailable,
  /// One past the last reason. This is an iteration bound and never an
  /// exclusion record.
  Count,
};

/// One machine-readable limitation attached to a normalized access site.
struct InventoryExclusion {
  /// Stable typed limitation.
  InventoryExclusionReason reason = InventoryExclusionReason::Count;

  bool operator==(const InventoryExclusion &) const = default;
};

/// One normalized byte range accessed by an original instruction.
///
/// The offset is relative to the instruction's dynamic base address when the
/// encoding makes that relationship available. Its absence means “not decoded
/// into this legacy inventory,” not an offset of zero. Width is expressed in
/// bytes so subword and wide accesses share one unit.
struct AccessRange {
  /// Stable semantic identity of this exact logical range.
  SemanticSiteId id;
  /// Static byte displacement from the dynamic base address, when known.
  std::optional<int64_t> static_byte_offset;
  /// Nonzero logical width of this range in bytes.
  uint32_t byte_width = 0;

  bool operator==(const AccessRange &) const = default;
};

/// Shared decoded-program record for one container interpretation of a
/// physical instruction.
///
/// An instruction can carry an access facet and one synchronization/fault
/// facet at the same time: for example, a relaxed LDS atomic is both. The
/// tagged payload therefore does not compete with the access fields. Records
/// without an access facet retain `origin == Count`; records without a
/// synchronization/fault facet retain an access-only payload. During
/// builder-owned decode, code-object identity, logical ranges, exclusions,
/// and execution owners can remain incomplete. Publication completes those
/// facts. Aliasing symbol ranges can still produce several records with the
/// same physical identity; later policy canonicalizes them before placement.
struct ProgramSite {
  using Payload = std::variant<AccessSite, OrdinaryMemorySite, BarrierSite, FenceSite, AtomicSite>;

  /// Stable identity within one immutable published inventory revision.
  ProgramSiteId id;
  PhysicalSiteId physical_id;
  ProgramContainerId container;
  AccessOrigin origin = AccessOrigin::Count;
  LdsAccessKind kind = LdsAccessKind::Other;
  AccessAddressSpace address_space = AccessAddressSpace::Unresolved;
  AccessProvenance provenance = AccessProvenance::Count;
  SemanticConfidence confidence = SemanticConfidence::Unsupported;
  AccessLoweringClassification lowering;
  uint32_t decoded_width_bits = 0;
  FlatAddressSpaceHint flat_address_space_hint = FlatAddressSpaceHint::Unknown;
  AccessOperandFacts operands;
  /// Identical effective interval across participating lanes, proven locally
  /// for an ordinary, single-range native LDS access. Unknown is false.
  bool uniform_lds_address = false;
  /// Same-address ordinary store with every data word proven wave-uniform.
  /// Only its intra-instruction collision is benign; cross-wave checks remain.
  bool uniform_lds_store = false;
  /// CFG-derived kernels that can execute this physical instruction. This is
  /// the authoritative owner proof for access and synchronization consumers;
  /// derived events retain only `source_site` rather than copying it.
  std::vector<ExecutionOwner> execution_owners;
  std::vector<AccessRange> ranges;
  std::vector<InventoryExclusion> exclusions;
  Payload payload;

  [[nodiscard]] bool has_access() const { return origin != AccessOrigin::Count; }

  template <typename Site> [[nodiscard]] const Site *get_if() const {
    return std::get_if<Site>(&payload);
  }
  template <typename Site> [[nodiscard]] Site *get_if() { return std::get_if<Site>(&payload); }

  [[nodiscard]] const DecodedSite &decoded_site() const {
    return std::visit([](const auto &site) -> const DecodedSite & { return site; }, payload);
  }
  [[nodiscard]] DecodedSite &decoded_site() {
    return std::visit([](auto &site) -> DecodedSite & { return site; }, payload);
  }
  [[nodiscard]] uint64_t text_offset() const { return physical_id.original_text_offset; }
  [[nodiscard]] uint64_t decoded_file_offset() const { return decoded_site().file_offset; }
  [[nodiscard]] uint32_t size() const { return decoded_site().size; }
  [[nodiscard]] std::string_view mnemonic_view() const { return decoded_site().mnemonic; }

  void retain_access_only_payload() {
    const DecodedSite location = decoded_site();
    payload = AccessSite{};
    static_cast<DecodedSite &>(std::get<AccessSite>(payload)) = location;
  }

  [[nodiscard]] bool same_payload(const ProgramSite &other) const {
    return payload == other.payload;
  }

  /// Return whether construction found no inventory-level limitation. Mode
  /// policy can still reject a complete semantic access for mode-specific
  /// reasons.
  [[nodiscard]] bool complete() const { return exclusions.empty() && !ranges.empty(); }

  bool operator==(const ProgramSite &) const = default;
};

/// Build a program-site record from a synchronization/fault decode facet.
/// Access decoding may later merge its independent facet into this same
/// physical record.
template <typename Site>
[[nodiscard]] ProgramSite make_program_site(ProgramContainerId container, Site site) {
  ProgramSite result;
  result.physical_id.original_text_offset = site.text_offset;
  result.container = std::move(container);
  result.payload = std::move(site);
  return result;
}

/// One ordered owning arena for decoded program sites.
///
/// Access records occupy a contiguous prefix while synchronization/fault-only
/// records occupy the suffix. A physical instruction with both facets remains
/// one prefix record. The same arena is used for transactional decoder batches
/// and for the published inventory, so neither path can recreate parallel
/// access and semantic ownership.
class ProgramSiteArena {
public:
  [[nodiscard]] std::span<const ProgramSite> sites() const {
    materialize_semantic_sites();
    return sites_;
  }
  [[nodiscard]] std::span<ProgramSite> sites() {
    materialize_semantic_sites();
    return sites_;
  }
  [[nodiscard]] std::span<const ProgramSite> access_sites() const {
    return std::span<const ProgramSite>(sites_).first(access_site_count_);
  }
  [[nodiscard]] std::span<ProgramSite> access_sites() {
    return std::span<ProgramSite>(sites_).first(access_site_count_);
  }

  void add_access(ProgramSite site) {
    const size_t inserted_index = access_site_count_;
    if (inserted_index == sites_.size())
      sites_.push_back(std::move(site));
    else
      sites_.insert(sites_.begin() + inserted_index, std::move(site));
    ++access_site_count_;
    // Existing access records retain their ordinals. Only the inserted record
    // and the semantic suffix moved by the insertion need new handles.
    reindex_from(inserted_index);
  }

  void add_semantic(ProgramSite site) {
    // Decoders encounter access and synchronization facets in instruction
    // order. Keep the semantic suffix out of the access vector until a view
    // is requested, so every later access remains an amortized append rather
    // than shifting all previously decoded semantic records.
    pending_semantic_sites_.push_back(std::move(site));
  }

  void append(ProgramSiteArena batch) {
    materialize_semantic_sites();
    batch.materialize_semantic_sites();
    const size_t access_count = batch.access_site_count_;
    if (batch.sites_.empty())
      return;

    const size_t combined_size = sites_.size() + batch.sites_.size();
    sites_.reserve(combined_size);
    sites_.insert(sites_.begin() + access_site_count_,
                  std::make_move_iterator(batch.sites_.begin()),
                  std::make_move_iterator(batch.sites_.begin() + access_count));
    access_site_count_ += access_count;
    pending_semantic_sites_.insert(pending_semantic_sites_.end(),
                                   std::make_move_iterator(batch.sites_.begin() + access_count),
                                   std::make_move_iterator(batch.sites_.end()));
    materialize_semantic_sites();
    reindex();
  }

  template <typename Predicate> void erase_if(Predicate predicate) {
    materialize_semantic_sites();
    auto access_end = sites_.begin() + access_site_count_;
    access_end = std::remove_if(sites_.begin(), access_end, predicate);
    const size_t retained_access_count = static_cast<size_t>(access_end - sites_.begin());
    sites_.erase(access_end, sites_.begin() + access_site_count_);
    access_site_count_ = retained_access_count;
    sites_.erase(std::remove_if(sites_.begin() + access_site_count_, sites_.end(), predicate),
                 sites_.end());
    reindex();
  }

  /// Transfer semantic facets in an installed code range to their final
  /// owner without disturbing the access prefix. Symbol aliases can decode
  /// the same semantic instruction more than once: retain one facet, but
  /// retain every independently normalized access record.
  void reattribute_semantic_range(uint64_t begin, uint32_t size, const ProgramContainerId &owner) {
    materialize_semantic_sites();
    const uint64_t end = begin + size;
    std::vector<std::pair<size_t, uint64_t>> seen;
    const auto in_range = [&](const ProgramSite &site) {
      return site.text_offset() >= begin && site.text_offset() < end;
    };
    const auto first_semantic_facet = [&](const ProgramSite &site) {
      const std::pair key{site.payload.index(), site.text_offset()};
      if (std::ranges::find(seen, key) != seen.end())
        return false;
      seen.push_back(key);
      return true;
    };

    for (ProgramSite &site : access_sites()) {
      if (!in_range(site))
        continue;
      site.container = owner;
      if (!std::holds_alternative<AccessSite>(site.payload) && !first_semantic_facet(site))
        site.retain_access_only_payload();
    }

    auto semantic_begin = sites_.begin() + access_site_count_;
    sites_.erase(std::remove_if(semantic_begin, sites_.end(),
                                [&](ProgramSite &site) {
                                  if (!in_range(site))
                                    return false;
                                  site.container = owner;
                                  return !first_semantic_facet(site);
                                }),
                 sites_.end());
    reindex();
  }

private:
  struct SourceKey {
    uint32_t container = ProgramContainerId::kInvalidOrdinal;
    uint64_t text_offset = 0;
    bool operator==(const SourceKey &) const = default;
  };

  struct SourceKeyHash {
    [[nodiscard]] size_t operator()(const SourceKey &key) const {
      const size_t container = std::hash<uint32_t>{}(key.container);
      const size_t offset = std::hash<uint64_t>{}(key.text_offset);
      return container ^ (offset + 0x9e3779b9u + (container << 6u) + (container >> 2u));
    }
  };

  void materialize_semantic_sites() const {
    if (pending_semantic_sites_.empty())
      return;
    std::unordered_map<SourceKey, size_t, SourceKeyHash> access_by_source;
    access_by_source.reserve(access_site_count_);
    for (size_t index = 0; index < access_site_count_; ++index) {
      const ProgramSite &access = sites_[index];
      access_by_source.try_emplace(
          SourceKey{access.container.ordinal, access.physical_id.original_text_offset}, index);
    }
    for (ProgramSite &site : pending_semantic_sites_) {
      const auto access = access_by_source.find(
          SourceKey{site.container.ordinal, site.physical_id.original_text_offset});
      if (access != access_by_source.end() &&
          std::holds_alternative<AccessSite>(sites_[access->second].payload)) {
        sites_[access->second].payload = std::move(site.payload);
      } else {
        sites_.push_back(std::move(site));
      }
    }
    pending_semantic_sites_.clear();
    reindex_from(access_site_count_);
  }

  void reindex_from(size_t begin) const {
    for (size_t index = begin; index < sites_.size(); ++index)
      sites_[index].id = {static_cast<uint32_t>(index)};
  }

  void reindex() const { reindex_from(0); }

  mutable std::vector<ProgramSite> sites_;
  mutable std::vector<ProgramSite> pending_semantic_sites_;
  size_t access_site_count_ = 0;
};

/// Resolve one stable site handle in any immutable arena view.
[[nodiscard]] inline const ProgramSite *program_site(std::span<const ProgramSite> sites,
                                                     ProgramSiteId id) {
  return id.valid() && id.ordinal < sites.size() ? &sites[id.ordinal] : nullptr;
}

template <typename Site>
[[nodiscard]] inline const Site *program_site(std::span<const ProgramSite> sites,
                                              ProgramSiteId id) {
  const ProgramSite *site = program_site(sites, id);
  return site == nullptr ? nullptr : site->get_if<Site>();
}

[[nodiscard]] inline const ProgramSite *program_site(std::span<const ProgramSite> sites,
                                                     const SyncEvent &event) {
  return program_site(sites, event.source_site);
}

template <typename Site>
[[nodiscard]] inline const Site *program_site(std::span<const ProgramSite> sites,
                                              const SyncEvent &event) {
  return program_site<Site>(sites, event.source_site);
}

/// Derive every symbol-container alias for one physical instruction from the
/// authoritative program-site arena. Alias names are presentation data and
/// must not be copied into semantic events or policy decisions.
[[nodiscard]] inline std::vector<std::string>
source_container_names(std::span<const ProgramSite> sites,
                       std::span<const ProgramContainer> containers,
                       const PhysicalSiteId &physical_site) {
  std::vector<std::string> names;
  for (const ProgramSite &site : sites) {
    // The arena belongs to exactly one code object, so the original text
    // offset is the complete lookup key inside this inventory. This also
    // keeps builder views useful before publication binds the code-object ID.
    if (site.text_offset() != physical_site.original_text_offset || !site.container.valid() ||
        site.container.ordinal >= containers.size())
      continue;
    const std::string &name = containers[site.container.ordinal].name;
    if (std::ranges::find(names, name) == names.end())
      names.push_back(name);
  }
  std::ranges::sort(names);
  return names;
}

/// Immutable synchronization and ownership projection produced by semantic
/// analysis.
///
/// Events retain physical and typed semantic identities; sequences and
/// lifecycle groups retain compact arena-local event handles, and fence
/// candidates retain the bounded associations derived from those sequences.
/// The view owns no storage and remains valid while its `ProgramInventory`
/// owner is alive. Fault planning and mode policy may query this projection
/// but cannot modify or rebuild sanitizer semantics through it.
struct SynchronizationInventoryView {
  std::span<const ProgramContainer> program_containers;
  std::span<const ProgramSite> program_sites;
  std::span<const SyncEvent> sync_events;
  std::span<const SyncSequence> sync_sequences;
  std::span<const SyncSequenceMembership> sequence_membership_by_event;
  std::span<const BarrierLifecycleGroup> barrier_lifecycle_groups;
  std::span<const FenceCandidate> fence_candidates;

  [[nodiscard]] const ProgramContainer *container(ProgramContainerId id) const {
    return id.valid() && id.ordinal < program_containers.size() ? &program_containers[id.ordinal]
                                                                : nullptr;
  }

  [[nodiscard]] const ProgramSite *source(const SyncEvent &event) const {
    return program_site(program_sites, event);
  }

  template <typename Site> [[nodiscard]] const Site *source_as(const SyncEvent &event) const {
    const ProgramSite *decoded = source(event);
    return decoded == nullptr ? nullptr : decoded->get_if<Site>();
  }

  [[nodiscard]] std::span<const ExecutionOwner> execution_owners(const SyncEvent &event) const {
    const ProgramSite *decoded = source(event);
    return decoded == nullptr ? std::span<const ExecutionOwner>{}
                              : std::span<const ExecutionOwner>(decoded->execution_owners);
  }

  /// Derive the owners that can reach every member of a logical sequence.
  /// Member program sites are authoritative; retaining this intersection on
  /// the sequence made ownership stale whenever either arena was revised.
  [[nodiscard]] std::vector<ExecutionOwner> execution_owners(const SyncSequence &sequence) const {
    std::vector<ExecutionOwner> intersection;
    bool first = true;
    for (SyncEventId member : sequence.member_event_ids) {
      const SyncEvent *event = find_event(member);
      if (event == nullptr)
        return {};
      const std::span<const ExecutionOwner> member_owners = execution_owners(*event);
      if (first) {
        intersection.assign(member_owners.begin(), member_owners.end());
        first = false;
        continue;
      }
      std::erase_if(intersection, [&](ExecutionOwner &left) {
        const auto right = std::ranges::find(member_owners, left.kernel, &ExecutionOwner::kernel);
        if (right == member_owners.end())
          return true;
        if (static_cast<uint8_t>(right->proof) > static_cast<uint8_t>(left.proof))
          left.proof = right->proof;
        return false;
      });
    }
    return intersection;
  }

  [[nodiscard]] const ProgramContainer *container(const SyncEvent &event) const {
    const ProgramSite *decoded = source(event);
    return decoded == nullptr ? nullptr : container(decoded->container);
  }

  [[nodiscard]] std::string_view container_name(const SyncEvent &event) const {
    const ProgramContainer *record = container(event);
    return record == nullptr ? std::string_view{} : std::string_view(record->name);
  }

  [[nodiscard]] bool in_kernel(const SyncEvent &event) const {
    const ProgramContainer *record = container(event);
    return record != nullptr && record->is_kernel();
  }

  [[nodiscard]] bool same_container(const SyncEvent &lhs, const SyncEvent &rhs) const {
    const ProgramContainer *left = container(lhs);
    const ProgramContainer *right = container(rhs);
    return left != nullptr && right != nullptr && left->id == right->id;
  }

  /// Resolve the one source container shared by every sequence member.
  /// Missing members or cross-container associations fail closed.
  [[nodiscard]] const ProgramContainer *container(const SyncSequence &sequence) const {
    const ProgramContainer *result = nullptr;
    for (SyncEventId member : sequence.member_event_ids) {
      const SyncEvent *event = find_event(member);
      const ProgramContainer *current = event == nullptr ? nullptr : container(*event);
      if (current == nullptr || (result != nullptr && result->id != current->id))
        return nullptr;
      result = current;
    }
    return result;
  }

  [[nodiscard]] std::string_view container_name(const SyncSequence &sequence) const {
    const ProgramContainer *record = container(sequence);
    return record == nullptr ? std::string_view{} : std::string_view(record->name);
  }

  [[nodiscard]] bool same_container(const SyncSequence &lhs, const SyncSequence &rhs) const {
    const ProgramContainer *left = container(lhs);
    const ProgramContainer *right = container(rhs);
    return left != nullptr && right != nullptr && left->id == right->id;
  }

  [[nodiscard]] std::vector<std::string>
  source_container_names(const PhysicalSiteId &physical_site) const {
    return consan::source_container_names(program_sites, program_containers, physical_site);
  }

  [[nodiscard]] std::vector<uint64_t>
  execution_owner_descriptors(std::span<const SyncEvent *const> aliases) const {
    std::vector<uint64_t> descriptors;
    for (const SyncEvent *event : aliases) {
      for (const ExecutionOwner &owner : execution_owners(*event)) {
        const ProgramContainer *kernel = container(owner.kernel);
        if (kernel != nullptr && kernel->is_kernel())
          descriptors.push_back(kernel->descriptor_file_offset);
      }
    }
    std::ranges::sort(descriptors);
    descriptors.erase(std::ranges::unique(descriptors).begin(), descriptors.end());
    return descriptors;
  }

  /// Resolve one typed event identity to its sole authoritative record. An
  /// absent identity returns null; physical aliases have already been
  /// canonicalized before this immutable view is published.
  [[nodiscard]] const SyncEvent *find_event(const SemanticSiteId &identity) const {
    const auto event = std::ranges::find(sync_events, identity, &SyncEvent::semantic_id);
    return event == sync_events.end() ? nullptr : &*event;
  }

  /// Resolve one inventory-local synchronization-event handle in constant
  /// time. The arena is immutable for the lifetime of this view.
  [[nodiscard]] const SyncEvent *find_event(SyncEventId id) const {
    return id.valid() && id.ordinal < sync_events.size() ? &sync_events[id.ordinal] : nullptr;
  }

  /// Return the compact handle of an event owned by this arena.
  [[nodiscard]] SyncEventId event_id(const SyncEvent &event) const {
    return {static_cast<uint32_t>(&event - sync_events.data())};
  }

  /// Resolve one inventory-local synchronization-sequence handle in constant
  /// time. The arena is immutable for the lifetime of this view.
  [[nodiscard]] const SyncSequence *find_sequence(SyncSequenceId id) const {
    return id.valid() && id.ordinal < sync_sequences.size() ? &sync_sequences[id.ordinal] : nullptr;
  }

  /// Return the compact handle of a sequence owned by this arena.
  [[nodiscard]] SyncSequenceId sequence_id(const SyncSequence &sequence) const {
    return {static_cast<uint32_t>(&sequence - sync_sequences.data())};
  }

  /// Return the published membership fact when this inventory has completed
  /// graph construction. Mutable test/build views without an index fall back
  /// to the authoritative sequence scan in higher-level queries.
  [[nodiscard]] SyncSequenceMembership sequence_membership(SyncEventId event) const {
    if (!event.valid() || event.ordinal >= sync_events.size())
      return {};
    if (sequence_membership_by_event.size() == sync_events.size())
      return sequence_membership_by_event[event.ordinal];
    SyncSequenceMembership membership;
    for (const SyncSequence &sequence : sync_sequences) {
      if (std::ranges::find(sequence.member_event_ids, event) == sequence.member_event_ids.end())
        continue;
      if (membership.sequence.valid())
        return {.sequence = {}, .ambiguous = true};
      membership.sequence = sequence_id(sequence);
    }
    return membership;
  }

  /// Resolve the synchronization event derived from one exact decoded site.
  /// Unlike semantic or physical lookup, this preserves symbol-alias identity.
  [[nodiscard]] const SyncEvent *find_event(ProgramSiteId source_site) const {
    const auto event = std::ranges::find(sync_events, source_site, &SyncEvent::source_site);
    return event == sync_events.end() ? nullptr : &*event;
  }

  /// Resolve one decoded event name to its sole published record. Event names
  /// are the stable diagnostic and mutation-selection identity; target-neutral
  /// graph policy uses the typed `SemanticSiteId` overload above.
  [[nodiscard]] const SyncEvent *find_event(std::string_view identity) const {
    const auto event = std::ranges::find(sync_events, identity, &SyncEvent::identity);
    return event == sync_events.end() ? nullptr : &*event;
  }

  /// Materialize stable diagnostic names only at an external presentation or
  /// mutation-plan boundary. The synchronization graph itself stores one
  /// typed member vector.
  [[nodiscard]] std::optional<std::vector<std::string>>
  sequence_member_identities(std::span<const SyncEventId> members) const {
    std::vector<std::string> identities;
    identities.reserve(members.size());
    for (SyncEventId member : members) {
      const SyncEvent *event = find_event(member);
      if (event == nullptr)
        return std::nullopt;
      identities.push_back(event->identity);
    }
    return identities;
  }

  /// Resolve the sole initialization event that owns a barrier lifecycle's
  /// identity and static attributes. Malformed test or stale builder state
  /// fails closed instead of selecting one of several contradictory members.
  [[nodiscard]] const SyncEvent *
  barrier_lifecycle_initialization(const BarrierLifecycleGroup &group) const {
    const SyncEvent *initialization = nullptr;
    for (SyncEventId member : group.member_event_ids) {
      const SyncEvent *event = find_event(member);
      if (event == nullptr)
        return nullptr;
      if (event->operation != SyncOperation::BarrierInit)
        continue;
      if (initialization != nullptr)
        return nullptr;
      initialization = event;
    }
    return initialization;
  }

  [[nodiscard]] std::optional<std::string>
  barrier_lifecycle_identity(const BarrierLifecycleGroup &group) const {
    const SyncEvent *initialization = barrier_lifecycle_initialization(group);
    if (initialization == nullptr)
      return std::nullopt;
    return initialization->identity + "|lifecycle-group=static-run";
  }

  /// Find the unique logical sequence containing a typed event identity.
  ///
  /// Typed membership is the authoritative graph edge. Absence and ambiguous
  /// membership both return null so evidence policy cannot silently select one
  /// of several contradictory causal interpretations.
  [[nodiscard]] const SyncSequence *
  find_unique_sequence_containing(SyncEventId event_identity) const {
    const SyncSequenceMembership membership = sequence_membership(event_identity);
    return membership.ambiguous ? nullptr : find_sequence(membership.sequence);
  }

  [[nodiscard]] const SyncSequence *
  find_unique_sequence_containing(const SemanticSiteId &event_identity) const {
    const SyncEvent *event =
        find_event(event_identity.in_domain(SemanticSiteDomain::SynchronizationEvent));
    return event == nullptr ? nullptr : find_unique_sequence_containing(event_id(*event));
  }

  /// Find unique sequence membership from the exact decoded source rather
  /// than round-tripping through its presentation identity.
  [[nodiscard]] const SyncSequence *
  find_unique_sequence_containing(ProgramSiteId source_site) const {
    const SyncEvent *event = find_event(source_site);
    return event == nullptr ? nullptr : find_unique_sequence_containing(event_id(*event));
  }

  /// Find the unique logical sequence containing a decoded event name.
  /// Absence and ambiguous membership both return null so callers cannot
  /// silently choose one of several contradictory ordering interpretations.
  [[nodiscard]] const SyncSequence *
  find_unique_sequence_containing(std::string_view event_identity) const {
    const SyncEvent *event = find_event(event_identity);
    return event == nullptr ? nullptr : find_unique_sequence_containing(event_id(*event));
  }

  /// Resolve one unique synchronization sequence by its stable graph identity.
  [[nodiscard]] const SyncSequence *find_unique_sequence(std::string_view identity) const {
    const SyncSequence *match = nullptr;
    for (const SyncSequence &sequence : sync_sequences) {
      if (sequence.identity != identity)
        continue;
      if (match != nullptr)
        return nullptr;
      match = &sequence;
    }
    return match;
  }

  /// Return whether semantic analysis published no synchronization-derived
  /// records of any kind.
  [[nodiscard]] bool empty() const {
    return sync_events.empty() && sync_sequences.empty() && barrier_lifecycle_groups.empty() &&
           fence_candidates.empty();
  }
};

/// Mutable construction capability for synchronization inventory during the
/// single semantic-analysis stage.
///
/// References point into one `ProgramInventoryBuilder`. Producer passes receive
/// this value explicitly, which makes mutation visible in their signatures and
/// prevents fault planners or lowerers from acquiring write access. The build
/// view must not escape analysis; published consumers receive only
/// `SynchronizationInventoryView`.
struct SynchronizationInventoryBuildView {
  const std::vector<ProgramContainer> &program_containers;
  std::span<ProgramSite> program_sites;
  std::vector<SyncEvent> &sync_events;
  std::vector<SyncSequence> &sync_sequences;
  std::vector<SyncSequenceMembership> &sequence_membership_by_event;
  std::vector<BarrierLifecycleGroup> &barrier_lifecycle_groups;
  std::vector<FenceCandidate> &fence_candidates;

  [[nodiscard]] SynchronizationInventoryView view() const {
    return {
        .program_containers = program_containers,
        .program_sites = program_sites,
        .sync_events = sync_events,
        .sync_sequences = sync_sequences,
        .sequence_membership_by_event = sequence_membership_by_event,
        .barrier_lifecycle_groups = barrier_lifecycle_groups,
        .fence_candidates = fence_candidates,
    };
  }
};

/// Internal shared storage owned by `ProgramInventory` and populated only by
/// `ProgramInventoryBuilder` during the analysis stage.
///
/// This type is named so ownership is explicit in profiles and tests. It is
/// not a semantic API: consumers use the immutable accessors on
/// `ProgramInventory`.
struct ProgramInventoryStorage {
  CodeObjectId code_object_id;
  /// Ownership and occupied-byte facts inherited from an earlier stage of a
  /// composed transform. They are properties of this exact input image, not
  /// options that an mode may mutate while lowering it.
  PreappliedMutationLayout preapplied_mutation;
  /// True only after the pristine bytes have passed structural AMDGPU code
  /// object parsing. This is an immutable input fact, independent of whether
  /// the target has a supported semantic architecture.
  bool code_object_parsed = false;
  bool kernel_metadata_trustworthy = false;
  size_t malformed_kernel_metadata_note_count = 0;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  bool semantic_arch_required = false;
  std::vector<TextSection> text_sections;
  /// One stable arena, ordered as dispatchable kernels followed by helper
  /// functions. `kernel_count` makes both typed semantic partitions contiguous
  /// views without reviving parallel ownership.
  std::vector<ProgramContainer> containers;
  size_t kernel_count = 0;
  /// One owning arena for every decoded instruction of interest. Accesses form
  /// a contiguous prefix so the hot access-policy path keeps a zero-allocation
  /// span; synchronization and fault analysis traverse the complete arena and
  /// ignore records whose tagged payload is monostate.
  ProgramSiteArena program_sites;
  std::vector<SyncEvent> sync_events;
  std::vector<SyncSequence> sync_sequences;
  std::vector<SyncSequenceMembership> sequence_membership_by_event;
  std::vector<BarrierLifecycleGroup> barrier_lifecycle_groups;
  std::vector<FenceCandidate> fence_candidates;
};

/// Immutable owner and query interface for original code-object facts used by
/// ConSan semantic policy.
///
/// Copies share const storage, so site IDs and references remain stable while
/// results move through retries and lowering. Only `ProgramInventoryBuilder`
/// can populate the storage during a transform. The public interface exposes
/// original-program facts plus normalized access and synchronization
/// semantics; it deliberately excludes mode admission, report layout,
/// registers, patches, diagnostics, and mutable transformation state.
class ProgramInventory {
public:
  ProgramInventory() = default;

  [[nodiscard]] bool empty() const {
    return !storage_ || storage_->code_object_id.fingerprint.empty();
  }
  [[nodiscard]] const CodeObjectId &code_object_id() const {
    return storage_ ? storage_->code_object_id : empty_storage().code_object_id;
  }
  /// Return immutable geometry inherited from a prior mutation stage.
  [[nodiscard]] const PreappliedMutationLayout &preapplied_mutation() const {
    return storage_ ? storage_->preapplied_mutation : empty_storage().preapplied_mutation;
  }
  /// Return whether the input was structurally parsed as an AMDGPU code object.
  [[nodiscard]] bool code_object_parsed() const { return storage_ && storage_->code_object_parsed; }
  [[nodiscard]] bool kernel_metadata_trustworthy() const {
    return storage_ && storage_->kernel_metadata_trustworthy;
  }
  [[nodiscard]] size_t malformed_kernel_metadata_note_count() const {
    return storage_ ? storage_->malformed_kernel_metadata_note_count : 0;
  }
  [[nodiscard]] rj_code_arch_t arch() const {
    return storage_ ? storage_->arch : ROCJITSU_CODE_ARCH_INVALID;
  }
  [[nodiscard]] rj_code_target_id_t target() const {
    return storage_ ? storage_->target : ROCJITSU_CODE_TARGET_INVALID;
  }
  [[nodiscard]] bool semantic_arch_required() const {
    return storage_ && storage_->semantic_arch_required;
  }
  /// Return whether every architecture-dependent inventory stage has a
  /// concrete semantic architecture. Parse-only inventories do not require
  /// one and remain safe for an unchanged pass-through result.
  [[nodiscard]] bool has_resolved_semantic_arch() const {
    return !semantic_arch_required() || arch() != ROCJITSU_CODE_ARCH_INVALID;
  }
  [[nodiscard]] std::span<const TextSection> text_sections() const {
    return storage_ ? std::span<const TextSection>(storage_->text_sections)
                    : std::span<const TextSection>{};
  }
  [[nodiscard]] std::span<const ProgramContainer> containers() const {
    return storage_ ? std::span<const ProgramContainer>(storage_->containers)
                    : std::span<const ProgramContainer>{};
  }
  [[nodiscard]] const ProgramContainer *container(ProgramContainerId id) const {
    const std::span<const ProgramContainer> records = containers();
    return id.valid() && id.ordinal < records.size() ? &records[id.ordinal] : nullptr;
  }
  [[nodiscard]] std::span<const ProgramContainer> kernels() const {
    return containers().first(storage_ ? storage_->kernel_count : 0);
  }
  /// Resolve a kernel descriptor's immutable inventory record. Descriptor
  /// offsets are unique physical identities within one code object.
  [[nodiscard]] const ProgramContainer *find_kernel_by_descriptor(uint64_t offset) const {
    const std::span<const ProgramContainer> records = kernels();
    const auto kernel =
        std::ranges::find(records, offset, &ProgramContainer::descriptor_file_offset);
    return kernel == records.end() ? nullptr : &*kernel;
  }
  /// Resolve the first published kernel with an exact symbol name. This
  /// preserves object order for malformed inputs containing duplicate names;
  /// physical ownership should use the descriptor lookup whenever available.
  [[nodiscard]] const ProgramContainer *find_kernel_by_name(std::string_view name) const {
    const std::span<const ProgramContainer> records = kernels();
    const auto kernel = std::ranges::find(records, name, &ProgramContainer::name);
    return kernel == records.end() ? nullptr : &*kernel;
  }
  /// Resolve a semantic execution owner to the kernel record in this exact
  /// inventory revision.  A stale, invalid, or non-kernel handle fails closed.
  [[nodiscard]] const ProgramContainer *kernel(const ExecutionOwner &owner) const {
    const ProgramContainer *record = container(owner.kernel);
    return record != nullptr && record->is_kernel() ? record : nullptr;
  }
  /// Materialize kernel handles as physical descriptor locations at a
  /// byte-oriented boundary. Unlike the presentation-oriented execution-owner
  /// queries below, this fails closed if any handle is stale or is not a kernel.
  [[nodiscard]] std::optional<std::vector<uint64_t>>
  kernel_descriptors(std::span<const ProgramContainerId> kernels) const {
    std::vector<uint64_t> descriptors;
    descriptors.reserve(kernels.size());
    for (ProgramContainerId id : kernels) {
      const ProgramContainer *record = container(id);
      if (record == nullptr || !record->is_kernel())
        return std::nullopt;
      descriptors.push_back(record->descriptor_file_offset);
    }
    return descriptors;
  }
  /// Materialize physical descriptor locations at a byte-oriented boundary.
  /// Semantic analysis and policy retain kernel handles instead.
  [[nodiscard]] std::vector<uint64_t>
  execution_owner_descriptors(std::span<const ExecutionOwner> owners) const {
    std::vector<uint64_t> descriptors;
    descriptors.reserve(owners.size());
    for (const ExecutionOwner &owner : owners) {
      if (const ProgramContainer *record = kernel(owner))
        descriptors.push_back(record->descriptor_file_offset);
    }
    std::ranges::sort(descriptors);
    descriptors.erase(std::ranges::unique(descriptors).begin(), descriptors.end());
    return descriptors;
  }
  /// Return whether every published semantic owner names a kernel in this
  /// exact immutable revision.  This catches cross-revision handle leakage at
  /// the public result boundary.
  [[nodiscard]] bool execution_owners_well_formed() const {
    return std::ranges::all_of(program_sites(), [&](const ProgramSite &site) {
      return std::ranges::all_of(site.execution_owners, [&](const ExecutionOwner &owner) {
        return kernel(owner) != nullptr;
      });
    });
  }
  /// Return every kernel that can execute a site, including the containing
  /// kernel when symbol-range ownership did not already publish it.
  ///
  /// This is the semantic owner representation. Consumers should retain these
  /// handles until they reach a descriptor-byte boundary.
  [[nodiscard]] std::vector<ProgramContainerId>
  execution_owner_kernels(const ProgramSite &site) const {
    std::vector<ProgramContainerId> owners = execution_owner_kernels(site.execution_owners);
    owners.reserve(owners.size() + 1u);
    if (const ProgramContainer *record = container(site.container);
        record != nullptr && record->is_kernel())
      owners.push_back(site.container);
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());
    return owners;
  }
  [[nodiscard]] std::vector<ProgramContainerId>
  execution_owner_kernels(std::span<const ExecutionOwner> execution_owners) const {
    std::vector<ProgramContainerId> owners;
    owners.reserve(execution_owners.size());
    for (const ExecutionOwner &owner : execution_owners)
      if (kernel(owner) != nullptr)
        owners.push_back(owner.kernel);
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());
    return owners;
  }
  /// Return whether every decoded record owns its exact arena handle.
  [[nodiscard]] bool program_site_ids_well_formed() const {
    const std::span<const ProgramSite> records = program_sites();
    for (size_t index = 0; index < records.size(); ++index) {
      if (records[index].id != ProgramSiteId{static_cast<uint32_t>(index)})
        return false;
    }
    return true;
  }
  [[nodiscard]] std::span<const ProgramContainer> functions() const {
    return containers().subspan(storage_ ? storage_->kernel_count : 0);
  }
  [[nodiscard]] std::span<const ProgramSite> access_sites() const {
    return storage_ ? storage_->program_sites.access_sites() : std::span<const ProgramSite>{};
  }
  /// Return every kernel descriptor that can own execution of an access,
  /// including its containing kernel when symbol-range ownership did not
  /// already publish that descriptor. Shared helper bodies can name several
  /// execution owners; callers that mutate descriptor state must update the
  /// complete returned set.
  [[nodiscard]] std::vector<uint64_t> execution_owner_descriptors(const ProgramSite &site) const {
    return kernel_descriptors(execution_owner_kernels(site)).value_or(std::vector<uint64_t>{});
  }
  [[nodiscard]] std::vector<uint64_t> execution_owner_descriptors(ProgramSiteId site) const {
    const ProgramSite *record = program_site(site);
    return record == nullptr ? std::vector<uint64_t>{} : execution_owner_descriptors(*record);
  }
  [[nodiscard]] std::vector<uint64_t>
  execution_owner_descriptors(std::span<const ProgramSite *const> aliases) const {
    std::vector<uint64_t> owners;
    for (const ProgramSite *site : aliases) {
      for (const ExecutionOwner &owner : site->execution_owners) {
        if (const ProgramContainer *record = kernel(owner))
          owners.push_back(record->descriptor_file_offset);
      }
    }
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());
    return owners;
  }
  /// Return the authoritative decoded source sites used by synchronization
  /// and fault analysis. Payloads remain decoder facts; policy and lowering
  /// products live in later phase-owned records.
  [[nodiscard]] std::span<const ProgramSite> program_sites() const {
    return storage_ ? std::span<const ProgramSite>(storage_->program_sites.sites())
                    : std::span<const ProgramSite>{};
  }

  /// Resolve one exact source-site handle in this inventory revision.
  [[nodiscard]] const ProgramSite *program_site(ProgramSiteId id) const {
    const ProgramSite *site = consan::program_site(program_sites(), id);
    return site != nullptr && site->id == id ? site : nullptr;
  }

  /// Resolve the authoritative decoded source of a fault-planning facet.
  [[nodiscard]] const ProgramSite *program_site(const FaultSite &site) const {
    return program_site(site.source_site);
  }

  [[nodiscard]] std::vector<std::string>
  source_container_names(const PhysicalSiteId &physical_site) const {
    return consan::source_container_names(program_sites(), containers(), physical_site);
  }

  /// Resolve one exact source-site handle and require its typed payload.
  template <typename Site> [[nodiscard]] const Site *program_site(ProgramSiteId id) const {
    const ProgramSite *site = program_site(id);
    return site == nullptr ? nullptr : site->get_if<Site>();
  }
  /// Return the complete immutable synchronization projection as one value.
  [[nodiscard]] SynchronizationInventoryView sync() const {
    if (!storage_)
      return {};
    return {
        .program_containers = storage_->containers,
        .program_sites = storage_->program_sites.sites(),
        .sync_events = storage_->sync_events,
        .sync_sequences = storage_->sync_sequences,
        .sequence_membership_by_event = storage_->sequence_membership_by_event,
        .barrier_lifecycle_groups = storage_->barrier_lifecycle_groups,
        .fence_candidates = storage_->fence_candidates,
    };
  }

private:
  friend class ProgramInventoryBuilder;

  explicit ProgramInventory(std::shared_ptr<const ProgramInventoryStorage> storage)
      : storage_(std::move(storage)) {}

  [[nodiscard]] static const ProgramInventoryStorage &empty_storage() {
    static const ProgramInventoryStorage storage;
    return storage;
  }

  std::shared_ptr<const ProgramInventoryStorage> storage_;
};

/// Construction capability for one `ProgramInventory` analysis stage.
///
/// The builder is the only mutable owner during decode/provenance/ownership
/// analysis. `view()` may publish a const facade while construction continues
/// so staged analyzers can read completed containers, but the builder must not
/// escape the transform. Once it is destroyed, every surviving inventory
/// reference is immutable. Decoder output is staged only in the builder-owned
/// access collection with immediate container attribution.
/// `publish_decoded_accesses`
/// completes stable identity, ranges, and limitations in place before the
/// immutable inventory becomes a stage output.
class ProgramInventoryBuilder {
public:
  /// Start a fresh inventory for one pristine code-object image.
  explicit ProgramInventoryBuilder(std::span<const uint8_t> code_object_bytes = {},
                                   const PreappliedMutationLayout &preapplied_mutation = {})
      : storage_(std::make_shared<ProgramInventoryStorage>()) {
    storage_->code_object_id = make_code_object_id(code_object_bytes);
    storage_->preapplied_mutation = preapplied_mutation;
  }

  /// Start a distinct mutable analysis revision from an immutable inventory.
  ///
  /// Storage is deep-copied: publishing or modifying the revision cannot
  /// mutate any existing `ProgramInventory` view. This operation supports
  /// component-sized reanalysis and adversarial tests; ordinary initial
  /// analysis should use the pristine-byte constructor above.
  explicit ProgramInventoryBuilder(const ProgramInventory &source)
      : storage_(std::make_shared<ProgramInventoryStorage>(
            source.storage_ ? *source.storage_ : ProgramInventoryStorage{})) {}

  /// Publish an immutable shared view of the builder's current revision.
  [[nodiscard]] ProgramInventory view() const {
    // Freeze the arena ordering before publishing a const facade. This keeps
    // immutable queries side-effect free even when the builder is destroyed
    // immediately after returning the view.
    (void)storage_->program_sites.sites();
    return ProgramInventory(storage_);
  }

  [[nodiscard]] std::vector<TextSection> &text_sections() { return storage_->text_sections; }
  [[nodiscard]] std::span<ProgramContainer> containers() { return storage_->containers; }
  [[nodiscard]] std::span<ProgramContainer> kernels() {
    return std::span(storage_->containers).first(storage_->kernel_count);
  }
  [[nodiscard]] std::span<ProgramContainer> functions() {
    return std::span(storage_->containers).subspan(storage_->kernel_count);
  }
  ProgramContainer &add_kernel(ProgramContainer container = {}) {
    container.kind = ProgramContainerKind::Kernel;
    if (storage_->kernel_count != storage_->containers.size())
      throw std::logic_error("ConSan kernels must be staged before functions");
    container.id = {static_cast<uint32_t>(storage_->containers.size())};
    storage_->containers.push_back(std::move(container));
    ++storage_->kernel_count;
    return storage_->containers.back();
  }
  ProgramContainer &add_function(ProgramContainer container = {}) {
    container.kind = ProgramContainerKind::Function;
    container.id = {static_cast<uint32_t>(storage_->containers.size())};
    storage_->containers.push_back(std::move(container));
    return storage_->containers.back();
  }
  /// Return the normalized access prefix while the inventory is being built.
  ///
  /// Access decoding creates these records before control-flow analysis.
  /// Later analysis stages may enrich fields such as execution ownership, but
  /// must not change a site's physical identity or decoded instruction facts.
  /// This mutable capability stays on the builder so published
  /// `ProgramInventory` views remain read-only.
  [[nodiscard]] std::span<ProgramSite> access_sites() {
    return storage_->program_sites.access_sites();
  }
  [[nodiscard]] std::span<ProgramSite> program_sites() { return storage_->program_sites.sites(); }
  void add_access_site(ProgramSite site) {
    invalidate_site_derivatives();
    storage_->program_sites.add_access(std::move(site));
  }
  void add_semantic_site(ProgramSite site) {
    invalidate_site_derivatives();
    storage_->program_sites.add_semantic(std::move(site));
  }
  void add_sites(ProgramSiteArena sites) {
    invalidate_site_derivatives();
    storage_->program_sites.append(std::move(sites));
  }
  template <typename Predicate> void erase_sites_if(Predicate predicate) {
    invalidate_site_derivatives();
    storage_->program_sites.erase_if(predicate);
  }
  /// Transfer semantic facets in an installed code range to their final
  /// owner without disturbing the access prefix. Symbol aliases can decode
  /// the same semantic instruction more than once: retain one facet, but
  /// retain every independently normalized access record.
  void reattribute_semantic_range(uint64_t begin, uint32_t size, const ProgramContainer &owner) {
    invalidate_site_derivatives();
    storage_->program_sites.reattribute_semantic_range(begin, size, owner.id);
  }
  /// Return the analysis-only capability for synchronization-derived facts.
  [[nodiscard]] SynchronizationInventoryBuildView synchronization() {
    // Acquiring mutable synchronization storage invalidates the derived index.
    storage_->sequence_membership_by_event.clear();
    return {
        .program_containers = storage_->containers,
        .program_sites = storage_->program_sites.sites(),
        .sync_events = storage_->sync_events,
        .sync_sequences = storage_->sync_sequences,
        .sequence_membership_by_event = storage_->sequence_membership_by_event,
        .barrier_lifecycle_groups = storage_->barrier_lifecycle_groups,
        .fence_candidates = storage_->fence_candidates,
    };
  }

  void set_code_object_facts(bool kernel_metadata_trustworthy,
                             size_t malformed_kernel_metadata_note_count, rj_code_arch_t arch,
                             rj_code_target_id_t target) {
    storage_->code_object_parsed = true;
    storage_->kernel_metadata_trustworthy = kernel_metadata_trustworthy;
    storage_->malformed_kernel_metadata_note_count = malformed_kernel_metadata_note_count;
    storage_->arch = arch;
    storage_->target = target;
  }

  void set_semantic_arch_required(bool required) { storage_->semantic_arch_required = required; }

  /// Complete access facts staged directly in builder-owned storage.
  ///
  /// Existing completed accesses are retained so a mutable inventory revision
  /// can append a newly analyzed container without reconstructing prior facts.
  /// Repeated publication is idempotent because a completed record has a valid
  /// code-object identity and no longer qualifies as pending decode output.
  void publish_decoded_accesses(std::span<const uint8_t> code_object_bytes) {
    // Atomic/fence decoding can contribute a semantic facet whose normalized
    // offset is authoritative for completing the access facet. Merge all
    // deferred semantic records before deriving access ranges.
    (void)storage_->program_sites.sites();
    for (ProgramSite &access : access_sites()) {
      if (!access.physical_id.code_object.valid())
        access = complete_access_site(access.container, access, code_object_bytes);
    }
    for (ProgramSite &site : storage_->program_sites.sites()) {
      if (!site.physical_id.code_object.valid())
        site.physical_id.code_object = storage_->code_object_id;
    }
  }

  /// Transfer normalized accesses in a staged executable range to their
  /// dispatchable owner and merge access instructions decoded only after that
  /// range was installed.
  ///
  /// Fault composition may place a trampoline outside its source kernel's
  /// symbol range. Existing records can therefore have been decoded through
  /// an overlapping symbol, while instructions introduced by the trampoline
  /// do not occur in the pristine inventory at all. This operation resolves
  /// both cases without creating a second container-owned access collection.
  void reattribute_access_range(uint64_t begin, uint32_t size, const ProgramContainer &owner,
                                std::span<const ProgramSite> decoded_accesses,
                                std::span<const uint8_t> code_object_bytes) {
    invalidate_site_derivatives();
    const uint64_t end = begin + size;
    const ProgramContainerId reference = owner.id;
    for (ProgramSite &access : access_sites()) {
      if (access.physical_id.original_text_offset >= begin &&
          access.physical_id.original_text_offset < end)
        access.container = reference;
    }
    for (const ProgramSite &access : decoded_accesses) {
      ProgramSite completed = complete_access_site(reference, access, code_object_bytes);
      const bool duplicate = std::ranges::any_of(access_sites(), [&](const auto &existing) {
        return existing.container == reference && existing.physical_id == completed.physical_id &&
               (existing.origin == AccessOrigin::Flat) == (completed.origin == AccessOrigin::Flat);
      });
      if (!duplicate)
        add_access_site(std::move(completed));
    }
  }

private:
  /// Structural site mutation starts a new arena revision. Ordinal handles
  /// remain stable only while that revision is immutable, so no graph product
  /// derived from the preceding ordering may survive the transition.
  void invalidate_site_derivatives() {
    storage_->sync_events.clear();
    storage_->sync_sequences.clear();
    storage_->sequence_membership_by_event.clear();
    storage_->barrier_lifecycle_groups.clear();
    storage_->fence_candidates.clear();
  }

  [[nodiscard]] static std::optional<uint32_t>
  read_little_endian_u32(std::span<const uint8_t> bytes, uint64_t offset) {
    if (offset > bytes.size() || sizeof(uint32_t) > bytes.size() - offset)
      return std::nullopt;
    return static_cast<uint32_t>(bytes[offset]) | (static_cast<uint32_t>(bytes[offset + 1]) << 8u) |
           (static_cast<uint32_t>(bytes[offset + 2]) << 16u) |
           (static_cast<uint32_t>(bytes[offset + 3]) << 24u);
  }

  /// Recover the static byte displacement of a one-range native LDS access.
  /// The target-owned atomic decoder has already normalized an atomic's exact
  /// offset into this program site's semantic facet; consuming that fact here
  /// avoids a second target-dependent interpretation of the same instruction.
  /// Other admitted one-range DS/VDS forms use the low sixteen bits.
  /// Direct-to-LDS transfers address their destination through M0 and therefore
  /// contribute no encoded displacement. Returning absence preserves a missing
  /// atomic facet or unreadable pristine bytes as a typed inventory gap instead
  /// of silently treating either as offset zero.
  [[nodiscard]] std::optional<int64_t>
  single_range_native_static_byte_offset(const ProgramSite &site,
                                         std::span<const uint8_t> bytes) const {
    if (site.origin == AccessOrigin::DirectToLds)
      return 0;
    if (site.kind == LdsAccessKind::Atomic) {
      const AtomicSite *atomic = site.get_if<AtomicSite>();
      if (atomic != nullptr && atomic->raw_ioffset)
        return static_cast<int64_t>(*atomic->raw_ioffset);
      const TargetProfile *target = target_profile(storage_->arch);
      if (target == nullptr)
        return std::nullopt;
      const uint32_t mask = (1u << target->native_lds.single_range_atomic_offset_bits) - 1u;
      const std::optional<uint32_t> word0 =
          read_little_endian_u32(bytes, site.decoded_file_offset());
      return word0 ? std::optional<int64_t>{static_cast<int64_t>(*word0 & mask)} : std::nullopt;
    }
    const std::optional<uint32_t> word0 = read_little_endian_u32(bytes, site.decoded_file_offset());
    if (!word0)
      return std::nullopt;
    return static_cast<int64_t>(*word0 & 0xffffu);
  }

  [[nodiscard]] SemanticSiteId access_range_id(uint64_t text_offset, uint32_t range_ordinal) const {
    return {
        .physical = {.code_object = storage_->code_object_id, .original_text_offset = text_offset},
        .domain = SemanticSiteDomain::Access,
        .member_ordinal = 0,
        .range_ordinal = range_ordinal,
    };
  }

  [[nodiscard]] ProgramSite complete_access_site(const ProgramContainerId &container,
                                                 const ProgramSite &site,
                                                 std::span<const uint8_t> bytes) const {
    ProgramSite result = site;
    result.physical_id = {
        .code_object = storage_->code_object_id,
        .original_text_offset = site.physical_id.original_text_offset,
    };
    result.container = container;
    result.execution_owners.clear();
    result.ranges.clear();
    result.exclusions.clear();
    if (site.origin == AccessOrigin::Flat) {
      const auto [address_space, provenance, confidence] =
          flat_provenance(site.flat_address_space_hint);
      result.address_space = address_space;
      result.provenance = provenance;
      result.confidence = confidence;
      append_common_exclusions(result, false);
      if (site.decoded_width_bits != 0 && site.decoded_width_bits % 8u == 0) {
        result.ranges.push_back({.id = access_range_id(site.physical_id.original_text_offset, 0),
                                 .static_byte_offset = site.operands.raw_ioffset,
                                 .byte_width = site.decoded_width_bits / 8u});
      }
      result.lowering = classify_access_lowering(result, storage_->arch);
      return result;
    }

    result.address_space = AccessAddressSpace::Group;
    result.provenance = AccessProvenance::NativeLdsOpcode;
    result.confidence = SemanticConfidence::Exact;
    append_common_exclusions(result, site.origin == AccessOrigin::DirectToLds);
    if (const auto form = detail::decode_native_lds_two_range_shape(site.mnemonic_view())) {
      const auto word0 = read_little_endian_u32(bytes, site.decoded_file_offset());
      if (word0) {
        result.ranges.push_back({.id = access_range_id(site.physical_id.original_text_offset, 0),
                                 .static_byte_offset = (*word0 & 0xffu) * form->offset_scale_bytes,
                                 .byte_width = form->element_width_bits / 8u});
        result.ranges.push_back(
            {.id = access_range_id(site.physical_id.original_text_offset, 1),
             .static_byte_offset = ((*word0 >> 8u) & 0xffu) * form->offset_scale_bytes,
             .byte_width = form->element_width_bits / 8u});
      } else {
        result.exclusions.push_back({
            .reason = InventoryExclusionReason::RangeEncodingUnavailable,
        });
      }
    } else if (site.decoded_width_bits != 0 && site.decoded_width_bits % 8u == 0) {
      result.ranges.push_back(
          {.id = access_range_id(site.physical_id.original_text_offset, 0),
           .static_byte_offset = single_range_native_static_byte_offset(site, bytes),
           .byte_width = site.decoded_width_bits / 8u});
    }
    result.lowering = classify_access_lowering(result, storage_->arch);
    return result;
  }

  static void append_common_exclusions(ProgramSite &site, bool address_operand_not_required) {
    if (site.kind == LdsAccessKind::Other) {
      site.exclusions.push_back({
          .reason = InventoryExclusionReason::NonAccessInstruction,
      });
    }
    if (site.size() == 0 || site.size() % sizeof(uint32_t) != 0) {
      site.exclusions.push_back({
          .reason = InventoryExclusionReason::InvalidInstructionSize,
      });
    }
    if (site.decoded_width_bits == 0 || site.decoded_width_bits % 8u != 0) {
      site.exclusions.push_back({
          .reason = InventoryExclusionReason::InvalidAccessWidth,
      });
    }
    if (!address_operand_not_required && !site.operands.address_vgpr) {
      site.exclusions.push_back({
          .reason = InventoryExclusionReason::MissingAddressOperand,
      });
    }
  }

  /// Internal tuple returned while normalizing a legacy FLAT provenance hint.
  /// It keeps address-space, source, and confidence assignments adjacent so a
  /// new hint cannot update one semantic dimension while silently leaving the
  /// others at an unrelated default.
  struct FlatProvenanceFacts {
    AccessAddressSpace address_space;
    AccessProvenance provenance;
    SemanticConfidence confidence;
  };

  [[nodiscard]] static FlatProvenanceFacts flat_provenance(FlatAddressSpaceHint hint) {
    switch (hint) {
    case FlatAddressSpaceHint::Group:
      return {AccessAddressSpace::Group, AccessProvenance::EncodedFlatSegment,
              SemanticConfidence::Exact};
    case FlatAddressSpaceHint::Private:
    case FlatAddressSpaceHint::Global:
      return {AccessAddressSpace::NonGroup, AccessProvenance::EncodedFlatSegment,
              SemanticConfidence::Exact};
    case FlatAddressSpaceHint::MaybeGroup:
      return {AccessAddressSpace::Group, AccessProvenance::PropagatedFlatPointer,
              SemanticConfidence::Conservative};
    case FlatAddressSpaceHint::MaybePrivate:
      return {AccessAddressSpace::NonGroup, AccessProvenance::PropagatedFlatPointer,
              SemanticConfidence::Conservative};
    case FlatAddressSpaceHint::Unknown:
      return {AccessAddressSpace::Unresolved, AccessProvenance::UnresolvedFlatPointer,
              SemanticConfidence::Ambiguous};
    }
    return {AccessAddressSpace::Unresolved, AccessProvenance::UnresolvedFlatPointer,
            SemanticConfidence::Unsupported};
  }

  std::shared_ptr<ProgramInventoryStorage> storage_;
};
