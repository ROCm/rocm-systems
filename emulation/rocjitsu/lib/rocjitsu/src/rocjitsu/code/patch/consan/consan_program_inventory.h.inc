// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// Normalized address-space fact relevant to ConSan access policy.
///
/// The inventory distinguishes group memory, definitely non-group memory, and
/// unresolved provenance without prematurely applying an engine's strict or
/// likely admission policy.
enum class ConSanAccessAddressSpace : uint8_t {
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
enum class ConSanAccessProvenance : uint8_t {
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
/// These are analysis facts, not engine rejections. An engine-specific
/// unsupported mnemonic, sampling decision, or resource failure belongs to a
/// later policy or lowering result rather than this enum.
enum class ConSanInventoryExclusionReason : uint8_t {
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
struct ConSanInventoryExclusion {
  /// Stable typed limitation.
  ConSanInventoryExclusionReason reason = ConSanInventoryExclusionReason::Count;

  bool operator==(const ConSanInventoryExclusion &) const = default;
};

/// One normalized byte range accessed by an original instruction.
///
/// The offset is relative to the instruction's dynamic base address when the
/// encoding makes that relationship available. Its absence means “not decoded
/// into this legacy inventory,” not an offset of zero. Width is expressed in
/// bytes so subword and wide accesses share one unit.
struct ConSanAccessRange {
  /// Stable semantic identity of this exact logical range.
  SemanticSiteId id;
  /// Static byte displacement from the dynamic base address, when known.
  std::optional<int64_t> static_byte_offset;
  /// Nonzero logical width of this range in bytes.
  uint32_t byte_width = 0;

  bool operator==(const ConSanAccessRange &) const = default;
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
struct ConSanProgramSite {
  using Payload = std::variant<ConSanAccessSite, ConSanOrdinaryMemorySite, ConSanBarrierSite,
                               ConSanFenceSite, ConSanAtomicSite>;

  PhysicalSiteId physical_id;
  ConSanProgramContainerId container;
  ConSanAccessOrigin origin = ConSanAccessOrigin::Count;
  ConSanLdsAccessKind kind = ConSanLdsAccessKind::Other;
  ConSanAccessAddressSpace address_space = ConSanAccessAddressSpace::Unresolved;
  ConSanAccessProvenance provenance = ConSanAccessProvenance::Count;
  ConSanSemanticConfidence confidence = ConSanSemanticConfidence::Unsupported;
  ConSanAccessLoweringClassification lowering;
  uint32_t decoded_width_bits = 0;
  ConSanFlatAddressSpaceHint flat_address_space_hint = ConSanFlatAddressSpaceHint::Unknown;
  ConSanAccessOperandFacts operands;
  /// CFG-derived kernels that can execute this physical instruction. This is
  /// the authoritative owner proof for access and synchronization consumers;
  /// derived events retain only `source_site` rather than copying it.
  std::vector<ConSanExecutionOwner> execution_owners;
  std::vector<ConSanAccessRange> ranges;
  std::vector<ConSanInventoryExclusion> exclusions;
  Payload payload;

  [[nodiscard]] bool has_access() const { return origin != ConSanAccessOrigin::Count; }

  template <typename Site> [[nodiscard]] const Site *get_if() const {
    return std::get_if<Site>(&payload);
  }
  template <typename Site> [[nodiscard]] Site *get_if() { return std::get_if<Site>(&payload); }

  [[nodiscard]] const ConSanDecodedSite &decoded_site() const {
    return std::visit([](const auto &site) -> const ConSanDecodedSite & { return site; }, payload);
  }
  [[nodiscard]] ConSanDecodedSite &decoded_site() {
    return std::visit([](auto &site) -> ConSanDecodedSite & { return site; }, payload);
  }
  [[nodiscard]] uint64_t text_offset() const { return physical_id.original_text_offset; }
  [[nodiscard]] uint64_t decoded_file_offset() const { return decoded_site().file_offset; }
  [[nodiscard]] uint32_t size() const { return decoded_site().size; }
  [[nodiscard]] std::string_view mnemonic_view() const { return decoded_site().mnemonic; }

  void retain_access_only_payload() {
    const ConSanDecodedSite location = decoded_site();
    payload = ConSanAccessSite{};
    static_cast<ConSanDecodedSite &>(std::get<ConSanAccessSite>(payload)) = location;
  }

  [[nodiscard]] bool same_payload(const ConSanProgramSite &other) const {
    return payload == other.payload;
  }

  /// Return whether construction found no inventory-level limitation. Engine
  /// policy can still reject a complete semantic access for mode-specific
  /// reasons.
  [[nodiscard]] bool complete() const { return exclusions.empty() && !ranges.empty(); }

  bool operator==(const ConSanProgramSite &) const = default;
};

/// Build a program-site record from a synchronization/fault decode facet.
/// Access decoding may later merge its independent facet into this same
/// physical record.
template <typename Site>
[[nodiscard]] ConSanProgramSite make_consan_program_site(ConSanProgramContainerId container,
                                                         Site site) {
  ConSanProgramSite result;
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
class ConSanProgramSiteArena {
public:
  [[nodiscard]] const std::vector<ConSanProgramSite> &sites() const { return sites_; }
  [[nodiscard]] std::vector<ConSanProgramSite> &sites() { return sites_; }
  [[nodiscard]] std::span<const ConSanProgramSite> access_sites() const {
    return std::span<const ConSanProgramSite>(sites_).first(access_site_count_);
  }
  [[nodiscard]] std::span<ConSanProgramSite> access_sites() {
    return std::span<ConSanProgramSite>(sites_).first(access_site_count_);
  }

  void add_access(ConSanProgramSite site) {
    sites_.insert(sites_.begin() + access_site_count_, std::move(site));
    ++access_site_count_;
  }

  void add_semantic(ConSanProgramSite site) {
    const auto same_source = [&](const ConSanProgramSite &candidate) {
      return candidate.container == site.container &&
             candidate.physical_id.original_text_offset == site.physical_id.original_text_offset;
    };
    const auto access = std::ranges::find_if(access_sites(), same_source);
    if (access != access_sites().end() &&
        std::holds_alternative<ConSanAccessSite>(access->payload)) {
      access->payload = std::move(site.payload);
      return;
    }
    sites_.push_back(std::move(site));
  }

  void append(ConSanProgramSiteArena batch) {
    const size_t access_count = batch.access_site_count_;
    for (size_t index = 0; index < access_count; ++index)
      add_access(std::move(batch.sites_[index]));
    for (size_t index = access_count; index < batch.sites_.size(); ++index)
      add_semantic(std::move(batch.sites_[index]));
  }

  template <typename Predicate> void erase_if(Predicate predicate) {
    auto access_end = sites_.begin() + access_site_count_;
    access_end = std::remove_if(sites_.begin(), access_end, predicate);
    const size_t retained_access_count = static_cast<size_t>(access_end - sites_.begin());
    sites_.erase(access_end, sites_.begin() + access_site_count_);
    access_site_count_ = retained_access_count;
    sites_.erase(std::remove_if(sites_.begin() + access_site_count_, sites_.end(), predicate),
                 sites_.end());
  }

  /// Transfer semantic facets in an installed code range to their final
  /// owner without disturbing the access prefix. Symbol aliases can decode
  /// the same semantic instruction more than once: retain one facet, but
  /// retain every independently normalized access record.
  void reattribute_semantic_range(uint64_t begin, uint32_t size,
                                  const ConSanProgramContainerId &owner) {
    const uint64_t end = begin + size;
    std::vector<std::pair<size_t, uint64_t>> seen;
    const auto in_range = [&](const ConSanProgramSite &site) {
      return site.text_offset() >= begin && site.text_offset() < end;
    };
    const auto first_semantic_facet = [&](const ConSanProgramSite &site) {
      const std::pair key{site.payload.index(), site.text_offset()};
      if (std::ranges::find(seen, key) != seen.end())
        return false;
      seen.push_back(key);
      return true;
    };

    for (ConSanProgramSite &site : access_sites()) {
      if (!in_range(site))
        continue;
      site.container = owner;
      if (!std::holds_alternative<ConSanAccessSite>(site.payload) && !first_semantic_facet(site))
        site.retain_access_only_payload();
    }

    auto semantic_begin = sites_.begin() + access_site_count_;
    sites_.erase(std::remove_if(semantic_begin, sites_.end(),
                                [&](ConSanProgramSite &site) {
                                  if (!in_range(site))
                                    return false;
                                  site.container = owner;
                                  return !first_semantic_facet(site);
                                }),
                 sites_.end());
  }

private:
  std::vector<ConSanProgramSite> sites_;
  size_t access_site_count_ = 0;
};

/// Resolve one stable site handle in any immutable arena view.
[[nodiscard]] inline const ConSanProgramSite *
consan_program_site(std::span<const ConSanProgramSite> sites, ConSanProgramSiteId id) {
  return id.valid() && id.ordinal < sites.size() ? &sites[id.ordinal] : nullptr;
}

template <typename Site>
[[nodiscard]] inline const Site *consan_program_site(std::span<const ConSanProgramSite> sites,
                                                     ConSanProgramSiteId id) {
  const ConSanProgramSite *site = consan_program_site(sites, id);
  return site == nullptr ? nullptr : site->get_if<Site>();
}

[[nodiscard]] inline const ConSanProgramSite *
consan_program_site(std::span<const ConSanProgramSite> sites, const ConSanSyncEvent &event) {
  return consan_program_site(sites, event.source_site);
}

template <typename Site>
[[nodiscard]] inline const Site *consan_program_site(std::span<const ConSanProgramSite> sites,
                                                     const ConSanSyncEvent &event) {
  return consan_program_site<Site>(sites, event.source_site);
}

/// Derive every symbol-container alias for one physical instruction from the
/// authoritative program-site arena. Alias names are presentation data and
/// must not be copied into semantic events or policy decisions.
[[nodiscard]] inline std::vector<std::string>
consan_source_container_names(std::span<const ConSanProgramSite> sites,
                              std::span<const ConSanProgramContainer> containers,
                              const PhysicalSiteId &physical_site) {
  std::vector<std::string> names;
  for (const ConSanProgramSite &site : sites) {
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
/// owner is alive. Fault planning and engine policy may query this projection
/// but cannot modify or rebuild sanitizer semantics through it.
struct SynchronizationInventoryView {
  std::span<const ConSanProgramContainer> program_containers;
  std::span<const ConSanProgramSite> program_sites;
  std::span<const ConSanSyncEvent> sync_events;
  std::span<const ConSanSyncSequence> sync_sequences;
  std::span<const ConSanSyncSequenceMembership> sequence_membership_by_event;
  std::span<const ConSanBarrierLifecycleGroup> barrier_lifecycle_groups;
  std::span<const ConSanMoiFenceCandidate> moi_fence_candidates;

  [[nodiscard]] const ConSanProgramContainer *container(ConSanProgramContainerId id) const {
    return id.valid() && id.ordinal < program_containers.size() ? &program_containers[id.ordinal]
                                                                : nullptr;
  }

  [[nodiscard]] const ConSanProgramSite *source(const ConSanSyncEvent &event) const {
    return consan_program_site(program_sites, event);
  }

  template <typename Site> [[nodiscard]] const Site *source_as(const ConSanSyncEvent &event) const {
    const ConSanProgramSite *decoded = source(event);
    return decoded == nullptr ? nullptr : decoded->get_if<Site>();
  }

  [[nodiscard]] std::span<const ConSanExecutionOwner>
  execution_owners(const ConSanSyncEvent &event) const {
    const ConSanProgramSite *decoded = source(event);
    return decoded == nullptr ? std::span<const ConSanExecutionOwner>{}
                              : std::span<const ConSanExecutionOwner>(decoded->execution_owners);
  }

  /// Derive the owners that can reach every member of a logical sequence.
  /// Member program sites are authoritative; retaining this intersection on
  /// the sequence made ownership stale whenever either arena was revised.
  [[nodiscard]] std::vector<ConSanExecutionOwner>
  execution_owners(const ConSanSyncSequence &sequence) const {
    std::vector<ConSanExecutionOwner> intersection;
    bool first = true;
    for (ConSanSyncEventId member : sequence.member_event_ids) {
      const ConSanSyncEvent *event = find_event(member);
      if (event == nullptr)
        return {};
      const std::span<const ConSanExecutionOwner> member_owners = execution_owners(*event);
      if (first) {
        intersection.assign(member_owners.begin(), member_owners.end());
        first = false;
        continue;
      }
      std::erase_if(intersection, [&](ConSanExecutionOwner &left) {
        const auto right =
            std::ranges::find(member_owners, left.kernel, &ConSanExecutionOwner::kernel);
        if (right == member_owners.end())
          return true;
        if (static_cast<uint8_t>(right->proof) > static_cast<uint8_t>(left.proof))
          left.proof = right->proof;
        return false;
      });
    }
    return intersection;
  }

  [[nodiscard]] const ConSanProgramContainer *container(const ConSanSyncEvent &event) const {
    const ConSanProgramSite *decoded = source(event);
    return decoded == nullptr ? nullptr : container(decoded->container);
  }

  [[nodiscard]] std::string_view container_name(const ConSanSyncEvent &event) const {
    const ConSanProgramContainer *record = container(event);
    return record == nullptr ? std::string_view{} : std::string_view(record->name);
  }

  [[nodiscard]] bool in_kernel(const ConSanSyncEvent &event) const {
    const ConSanProgramContainer *record = container(event);
    return record != nullptr && record->is_kernel();
  }

  [[nodiscard]] bool same_container(const ConSanSyncEvent &lhs, const ConSanSyncEvent &rhs) const {
    const ConSanProgramContainer *left = container(lhs);
    const ConSanProgramContainer *right = container(rhs);
    return left != nullptr && right != nullptr && left->id == right->id;
  }

  /// Resolve the one source container shared by every sequence member.
  /// Missing members or cross-container associations fail closed.
  [[nodiscard]] const ConSanProgramContainer *container(const ConSanSyncSequence &sequence) const {
    const ConSanProgramContainer *result = nullptr;
    for (ConSanSyncEventId member : sequence.member_event_ids) {
      const ConSanSyncEvent *event = find_event(member);
      const ConSanProgramContainer *current = event == nullptr ? nullptr : container(*event);
      if (current == nullptr || (result != nullptr && result->id != current->id))
        return nullptr;
      result = current;
    }
    return result;
  }

  [[nodiscard]] std::string_view container_name(const ConSanSyncSequence &sequence) const {
    const ConSanProgramContainer *record = container(sequence);
    return record == nullptr ? std::string_view{} : std::string_view(record->name);
  }

  [[nodiscard]] bool same_container(const ConSanSyncSequence &lhs,
                                    const ConSanSyncSequence &rhs) const {
    const ConSanProgramContainer *left = container(lhs);
    const ConSanProgramContainer *right = container(rhs);
    return left != nullptr && right != nullptr && left->id == right->id;
  }

  [[nodiscard]] std::vector<std::string>
  source_container_names(const PhysicalSiteId &physical_site) const {
    return consan_source_container_names(program_sites, program_containers, physical_site);
  }

  [[nodiscard]] std::vector<uint64_t>
  execution_owner_descriptors(std::span<const ConSanSyncEvent *const> aliases) const {
    std::vector<uint64_t> descriptors;
    for (const ConSanSyncEvent *event : aliases) {
      for (const ConSanExecutionOwner &owner : execution_owners(*event)) {
        const ConSanProgramContainer *kernel = container(owner.kernel);
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
  [[nodiscard]] const ConSanSyncEvent *find_event(const SemanticSiteId &identity) const {
    const auto event = std::ranges::find(sync_events, identity, &ConSanSyncEvent::semantic_id);
    return event == sync_events.end() ? nullptr : &*event;
  }

  /// Resolve one inventory-local synchronization-event handle in constant
  /// time. The arena is immutable for the lifetime of this view.
  [[nodiscard]] const ConSanSyncEvent *find_event(ConSanSyncEventId id) const {
    return id.valid() && id.ordinal < sync_events.size() ? &sync_events[id.ordinal] : nullptr;
  }

  /// Return the compact handle of an event owned by this arena.
  [[nodiscard]] ConSanSyncEventId event_id(const ConSanSyncEvent &event) const {
    return {static_cast<uint32_t>(&event - sync_events.data())};
  }

  /// Resolve one inventory-local synchronization-sequence handle in constant
  /// time. The arena is immutable for the lifetime of this view.
  [[nodiscard]] const ConSanSyncSequence *find_sequence(ConSanSyncSequenceId id) const {
    return id.valid() && id.ordinal < sync_sequences.size() ? &sync_sequences[id.ordinal] : nullptr;
  }

  /// Return the compact handle of a sequence owned by this arena.
  [[nodiscard]] ConSanSyncSequenceId sequence_id(const ConSanSyncSequence &sequence) const {
    return {static_cast<uint32_t>(&sequence - sync_sequences.data())};
  }

  /// Return the published membership fact when this inventory has completed
  /// graph construction. Mutable test/build views without an index fall back
  /// to the authoritative sequence scan in higher-level queries.
  [[nodiscard]] ConSanSyncSequenceMembership sequence_membership(ConSanSyncEventId event) const {
    if (!event.valid() || event.ordinal >= sync_events.size())
      return {};
    if (sequence_membership_by_event.size() == sync_events.size())
      return sequence_membership_by_event[event.ordinal];
    ConSanSyncSequenceMembership membership;
    for (const ConSanSyncSequence &sequence : sync_sequences) {
      if (std::ranges::find(sequence.member_event_ids, event) == sequence.member_event_ids.end())
        continue;
      if (membership.sequence.valid())
        return {.sequence = {}, .ambiguous = true};
      membership.sequence = sequence_id(sequence);
    }
    return membership;
  }

  /// Resolve the synchronization event derived from one exact decoded site.
  /// Physical aliases intentionally converge on the canonical event; callers
  /// retain the source handle when symbol attribution must also be checked.
  [[nodiscard]] const ConSanSyncEvent *find_event(ConSanProgramSiteId source_site) const {
    if (!source_site.valid() || source_site.ordinal >= program_sites.size())
      return nullptr;
    return find_event(SemanticSiteId{
        .physical = program_sites[source_site.ordinal].physical_id,
        .domain = ConSanSemanticSiteDomain::SynchronizationEvent,
    });
  }

  /// Resolve one decoded event name to its sole published record. Event names
  /// are the stable diagnostic and mutation-selection identity; target-neutral
  /// graph policy uses the typed `SemanticSiteId` overload above.
  [[nodiscard]] const ConSanSyncEvent *find_event(std::string_view identity) const {
    const auto event = std::ranges::find(sync_events, identity, &ConSanSyncEvent::identity);
    return event == sync_events.end() ? nullptr : &*event;
  }

  /// Materialize stable diagnostic names only at an external presentation or
  /// mutation-plan boundary. The synchronization graph itself stores one
  /// typed member vector.
  [[nodiscard]] std::optional<std::vector<std::string>>
  sequence_member_identities(std::span<const ConSanSyncEventId> members) const {
    std::vector<std::string> identities;
    identities.reserve(members.size());
    for (ConSanSyncEventId member : members) {
      const ConSanSyncEvent *event = find_event(member);
      if (event == nullptr)
        return std::nullopt;
      identities.push_back(event->identity);
    }
    return identities;
  }

  /// Resolve the sole initialization event that owns a barrier lifecycle's
  /// identity and static attributes. Malformed test or stale builder state
  /// fails closed instead of selecting one of several contradictory members.
  [[nodiscard]] const ConSanSyncEvent *
  barrier_lifecycle_initialization(const ConSanBarrierLifecycleGroup &group) const {
    const ConSanSyncEvent *initialization = nullptr;
    for (ConSanSyncEventId member : group.member_event_ids) {
      const ConSanSyncEvent *event = find_event(member);
      if (event == nullptr)
        return nullptr;
      if (event->operation != ConSanSyncOperation::BarrierInit)
        continue;
      if (initialization != nullptr)
        return nullptr;
      initialization = event;
    }
    return initialization;
  }

  [[nodiscard]] std::optional<std::string>
  barrier_lifecycle_identity(const ConSanBarrierLifecycleGroup &group) const {
    const ConSanSyncEvent *initialization = barrier_lifecycle_initialization(group);
    if (initialization == nullptr)
      return std::nullopt;
    return initialization->identity + "|lifecycle-group=static-run";
  }

  /// Resolve one unique synchronization event by physical site and kind.
  ///
  /// This is the graph-side join used by a target-neutral probe intent, whose
  /// placement identity deliberately has no engine-specific semantic domain.
  /// Duplicate physical records fail closed even if their decoded fields are
  /// otherwise identical; published production inventories canonicalize valid
  /// aliases before policy observes them.
  [[nodiscard]] const ConSanSyncEvent *find_unique_event(ConSanSyncKind kind,
                                                         const PhysicalSiteId &physical) const {
    const ConSanSyncEvent *match = nullptr;
    for (const ConSanSyncEvent &event : sync_events) {
      if (event.kind != kind || event.semantic_id.physical != physical)
        continue;
      if (match != nullptr)
        return nullptr;
      match = &event;
    }
    return match;
  }

  /// Find the unique logical sequence containing a typed event identity.
  ///
  /// Typed membership is the authoritative graph edge. Absence and ambiguous
  /// membership both return null so evidence policy cannot silently select one
  /// of several contradictory causal interpretations.
  [[nodiscard]] const ConSanSyncSequence *
  find_unique_sequence_containing(ConSanSyncEventId event_identity) const {
    const ConSanSyncSequenceMembership membership = sequence_membership(event_identity);
    return membership.ambiguous ? nullptr : find_sequence(membership.sequence);
  }

  [[nodiscard]] const ConSanSyncSequence *
  find_unique_sequence_containing(const SemanticSiteId &event_identity) const {
    const ConSanSyncEvent *event =
        find_event(event_identity.in_domain(ConSanSemanticSiteDomain::SynchronizationEvent));
    return event == nullptr ? nullptr : find_unique_sequence_containing(event_id(*event));
  }

  /// Find unique sequence membership from the exact decoded source rather
  /// than round-tripping through its presentation identity.
  [[nodiscard]] const ConSanSyncSequence *
  find_unique_sequence_containing(ConSanProgramSiteId source_site) const {
    const ConSanSyncEvent *event = find_event(source_site);
    return event == nullptr ? nullptr : find_unique_sequence_containing(event_id(*event));
  }

  /// Find the unique logical sequence containing a decoded event name.
  /// Absence and ambiguous membership both return null so callers cannot
  /// silently choose one of several contradictory ordering interpretations.
  [[nodiscard]] const ConSanSyncSequence *
  find_unique_sequence_containing(std::string_view event_identity) const {
    const ConSanSyncEvent *event = find_event(event_identity);
    return event == nullptr ? nullptr : find_unique_sequence_containing(event_id(*event));
  }

  /// Resolve one unique synchronization sequence by its stable graph identity.
  [[nodiscard]] const ConSanSyncSequence *find_unique_sequence(std::string_view identity) const {
    const ConSanSyncSequence *match = nullptr;
    for (const ConSanSyncSequence &sequence : sync_sequences) {
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
           moi_fence_candidates.empty();
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
  const std::vector<ConSanProgramContainer> &program_containers;
  std::vector<ConSanProgramSite> &program_sites;
  std::vector<ConSanSyncEvent> &sync_events;
  std::vector<ConSanSyncSequence> &sync_sequences;
  std::vector<ConSanSyncSequenceMembership> &sequence_membership_by_event;
  std::vector<ConSanBarrierLifecycleGroup> &barrier_lifecycle_groups;
  std::vector<ConSanMoiFenceCandidate> &moi_fence_candidates;

  [[nodiscard]] SynchronizationInventoryView view() const {
    return {
        .program_containers = program_containers,
        .program_sites = program_sites,
        .sync_events = sync_events,
        .sync_sequences = sync_sequences,
        .sequence_membership_by_event = sequence_membership_by_event,
        .barrier_lifecycle_groups = barrier_lifecycle_groups,
        .moi_fence_candidates = moi_fence_candidates,
    };
  }
};

/// Internal shared storage owned by `ProgramInventory` and populated only by
/// `ProgramInventoryBuilder` during the analysis stage.
///
/// This type is named so ownership is explicit in profiles and tests. It is
/// not a semantic API: consumers use the immutable accessors on
/// `ProgramInventory`.
struct ConSanProgramInventoryStorage {
  ConSanCodeObjectId code_object_id;
  /// Ownership and occupied-byte facts inherited from an earlier stage of a
  /// composed transform. They are properties of this exact input image, not
  /// options that an engine may mutate while lowering it.
  ConSanPreappliedMutationLayout preapplied_mutation;
  /// True only after the pristine bytes have passed structural AMDGPU code
  /// object parsing. This is an immutable input fact, independent of whether
  /// the target has a supported semantic architecture.
  bool code_object_parsed = false;
  bool kernel_metadata_trustworthy = false;
  size_t malformed_kernel_metadata_note_count = 0;
  rj_code_arch_t arch = ROCJITSU_CODE_ARCH_INVALID;
  rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID;
  bool semantic_arch_required = false;
  std::vector<ConSanTextSection> text_sections;
  /// One stable arena, ordered as dispatchable kernels followed by helper
  /// functions. `kernel_count` makes both typed semantic partitions contiguous
  /// views without reviving parallel ownership.
  std::vector<ConSanProgramContainer> containers;
  size_t kernel_count = 0;
  /// One owning arena for every decoded instruction of interest. Accesses form
  /// a contiguous prefix so the hot access-policy path keeps a zero-allocation
  /// span; synchronization and fault analysis traverse the complete arena and
  /// ignore records whose tagged payload is monostate.
  ConSanProgramSiteArena program_sites;
  std::vector<ConSanSyncEvent> sync_events;
  std::vector<ConSanSyncSequence> sync_sequences;
  std::vector<ConSanSyncSequenceMembership> sequence_membership_by_event;
  std::vector<ConSanBarrierLifecycleGroup> barrier_lifecycle_groups;
  std::vector<ConSanMoiFenceCandidate> moi_fence_candidates;
};

/// Immutable owner and query interface for original code-object facts used by
/// ConSan semantic policy.
///
/// Copies share const storage, so site IDs and references remain stable while
/// results move through retries and lowering. Only `ProgramInventoryBuilder`
/// can populate the storage during a transform. The public interface exposes
/// original-program facts plus normalized access and synchronization
/// semantics; it deliberately excludes engine admission, report layout,
/// registers, patches, diagnostics, and mutable transformation state.
class ProgramInventory {
public:
  ProgramInventory() = default;

  [[nodiscard]] bool empty() const {
    return !storage_ || storage_->code_object_id.fingerprint.empty();
  }
  [[nodiscard]] const ConSanCodeObjectId &code_object_id() const {
    return storage_ ? storage_->code_object_id : empty_storage().code_object_id;
  }
  /// Return immutable geometry inherited from a prior mutation stage.
  [[nodiscard]] const ConSanPreappliedMutationLayout &preapplied_mutation() const {
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
  [[nodiscard]] std::span<const ConSanTextSection> text_sections() const {
    return storage_ ? std::span<const ConSanTextSection>(storage_->text_sections)
                    : std::span<const ConSanTextSection>{};
  }
  [[nodiscard]] std::span<const ConSanProgramContainer> containers() const {
    return storage_ ? std::span<const ConSanProgramContainer>(storage_->containers)
                    : std::span<const ConSanProgramContainer>{};
  }
  [[nodiscard]] const ConSanProgramContainer *container(ConSanProgramContainerId id) const {
    const std::span<const ConSanProgramContainer> records = containers();
    return id.valid() && id.ordinal < records.size() ? &records[id.ordinal] : nullptr;
  }
  [[nodiscard]] std::span<const ConSanProgramContainer> kernels() const {
    return containers().first(storage_ ? storage_->kernel_count : 0);
  }
  /// Resolve a kernel descriptor's immutable inventory record. Descriptor
  /// offsets are unique physical identities within one code object.
  [[nodiscard]] const ConSanProgramContainer *find_kernel_by_descriptor(uint64_t offset) const {
    const std::span<const ConSanProgramContainer> records = kernels();
    const auto kernel =
        std::ranges::find(records, offset, &ConSanProgramContainer::descriptor_file_offset);
    return kernel == records.end() ? nullptr : &*kernel;
  }
  /// Resolve the first published kernel with an exact symbol name. This
  /// preserves object order for malformed inputs containing duplicate names;
  /// physical ownership should use the descriptor lookup whenever available.
  [[nodiscard]] const ConSanProgramContainer *find_kernel_by_name(std::string_view name) const {
    const std::span<const ConSanProgramContainer> records = kernels();
    const auto kernel = std::ranges::find(records, name, &ConSanProgramContainer::name);
    return kernel == records.end() ? nullptr : &*kernel;
  }
  /// Resolve a semantic execution owner to the kernel record in this exact
  /// inventory revision.  A stale, invalid, or non-kernel handle fails closed.
  [[nodiscard]] const ConSanProgramContainer *kernel(const ConSanExecutionOwner &owner) const {
    const ConSanProgramContainer *record = container(owner.kernel);
    return record != nullptr && record->is_kernel() ? record : nullptr;
  }
  /// Materialize kernel handles as physical descriptor locations at a
  /// byte-oriented boundary. Unlike the presentation-oriented execution-owner
  /// queries below, this fails closed if any handle is stale or is not a kernel.
  [[nodiscard]] std::optional<std::vector<uint64_t>>
  kernel_descriptors(std::span<const ConSanProgramContainerId> kernels) const {
    std::vector<uint64_t> descriptors;
    descriptors.reserve(kernels.size());
    for (ConSanProgramContainerId id : kernels) {
      const ConSanProgramContainer *record = container(id);
      if (record == nullptr || !record->is_kernel())
        return std::nullopt;
      descriptors.push_back(record->descriptor_file_offset);
    }
    return descriptors;
  }
  /// Materialize physical descriptor locations at a byte-oriented boundary.
  /// Semantic analysis and policy retain kernel handles instead.
  [[nodiscard]] std::vector<uint64_t>
  execution_owner_descriptors(std::span<const ConSanExecutionOwner> owners) const {
    std::vector<uint64_t> descriptors;
    descriptors.reserve(owners.size());
    for (const ConSanExecutionOwner &owner : owners) {
      if (const ConSanProgramContainer *record = kernel(owner))
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
    return std::ranges::all_of(program_sites(), [&](const ConSanProgramSite &site) {
      return std::ranges::all_of(site.execution_owners, [&](const ConSanExecutionOwner &owner) {
        return kernel(owner) != nullptr;
      });
    });
  }
  [[nodiscard]] std::span<const ConSanProgramContainer> functions() const {
    return containers().subspan(storage_ ? storage_->kernel_count : 0);
  }
  /// Resolve the first published non-dispatchable function with an exact
  /// symbol name, preserving code-object order for malformed duplicate names.
  [[nodiscard]] const ConSanProgramContainer *find_function_by_name(std::string_view name) const {
    const std::span<const ConSanProgramContainer> records = functions();
    const auto function = std::ranges::find(records, name, &ConSanProgramContainer::name);
    return function == records.end() ? nullptr : &*function;
  }
  [[nodiscard]] std::span<const ConSanProgramSite> access_sites() const {
    return storage_ ? storage_->program_sites.access_sites() : std::span<const ConSanProgramSite>{};
  }
  /// Return every kernel descriptor that can own execution of an access,
  /// including its containing kernel when symbol-range ownership did not
  /// already publish that descriptor. Shared helper bodies can name several
  /// execution owners; callers that mutate descriptor state must update the
  /// complete returned set.
  [[nodiscard]] std::vector<uint64_t>
  execution_owner_descriptors(const ConSanProgramSite &site) const {
    std::vector<uint64_t> owners;
    owners.reserve(site.execution_owners.size() + 1u);
    for (const ConSanExecutionOwner &owner : site.execution_owners) {
      if (const ConSanProgramContainer *record = kernel(owner))
        owners.push_back(record->descriptor_file_offset);
    }
    if (const ConSanProgramContainer *record = container(site.container);
        record != nullptr && record->is_kernel())
      owners.push_back(record->descriptor_file_offset);
    std::ranges::sort(owners);
    owners.erase(std::ranges::unique(owners).begin(), owners.end());
    return owners;
  }
  [[nodiscard]] std::vector<uint64_t> execution_owner_descriptors(ConSanProgramSiteId site) const {
    const ConSanProgramSite *record = program_site(site);
    return record == nullptr ? std::vector<uint64_t>{} : execution_owner_descriptors(*record);
  }
  [[nodiscard]] std::vector<uint64_t>
  execution_owner_descriptors(std::span<const ConSanProgramSite *const> aliases) const {
    std::vector<uint64_t> owners;
    for (const ConSanProgramSite *site : aliases) {
      for (const ConSanExecutionOwner &owner : site->execution_owners) {
        if (const ConSanProgramContainer *record = kernel(owner))
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
  [[nodiscard]] std::span<const ConSanProgramSite> program_sites() const {
    return storage_ ? std::span<const ConSanProgramSite>(storage_->program_sites.sites())
                    : std::span<const ConSanProgramSite>{};
  }

  /// Resolve one exact source-site handle in this inventory revision.
  [[nodiscard]] const ConSanProgramSite *program_site(ConSanProgramSiteId id) const {
    return consan_program_site(program_sites(), id);
  }

  /// Resolve the authoritative decoded source of a fault-planning facet.
  [[nodiscard]] const ConSanProgramSite *program_site(const ConSanFaultSite &site) const {
    return program_site(site.source_site);
  }

  [[nodiscard]] std::vector<std::string>
  source_container_names(const PhysicalSiteId &physical_site) const {
    return consan_source_container_names(program_sites(), containers(), physical_site);
  }

  /// Resolve one exact source-site handle and require its typed payload.
  template <typename Site> [[nodiscard]] const Site *program_site(ConSanProgramSiteId id) const {
    const ConSanProgramSite *site = program_site(id);
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
        .moi_fence_candidates = storage_->moi_fence_candidates,
    };
  }

private:
  friend class ProgramInventoryBuilder;

  explicit ProgramInventory(std::shared_ptr<const ConSanProgramInventoryStorage> storage)
      : storage_(std::move(storage)) {}

  [[nodiscard]] static const ConSanProgramInventoryStorage &empty_storage() {
    static const ConSanProgramInventoryStorage storage;
    return storage;
  }

  std::shared_ptr<const ConSanProgramInventoryStorage> storage_;
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
                                   const ConSanPreappliedMutationLayout &preapplied_mutation = {})
      : storage_(std::make_shared<ConSanProgramInventoryStorage>()) {
    storage_->code_object_id = make_consan_code_object_id(code_object_bytes);
    storage_->preapplied_mutation = preapplied_mutation;
  }

  /// Start a distinct mutable analysis revision from an immutable inventory.
  ///
  /// Storage is deep-copied: publishing or modifying the revision cannot
  /// mutate any existing `ProgramInventory` view. This operation supports
  /// component-sized reanalysis and adversarial tests; ordinary initial
  /// analysis should use the pristine-byte constructor above.
  explicit ProgramInventoryBuilder(const ProgramInventory &source)
      : storage_(std::make_shared<ConSanProgramInventoryStorage>(
            source.storage_ ? *source.storage_ : ConSanProgramInventoryStorage{})) {}

  /// Publish an immutable shared view of the builder's current revision.
  [[nodiscard]] ProgramInventory view() const { return ProgramInventory(storage_); }

  [[nodiscard]] std::vector<ConSanTextSection> &text_sections() { return storage_->text_sections; }
  [[nodiscard]] std::span<ConSanProgramContainer> containers() { return storage_->containers; }
  [[nodiscard]] std::span<ConSanProgramContainer> kernels() {
    return std::span(storage_->containers).first(storage_->kernel_count);
  }
  [[nodiscard]] std::span<ConSanProgramContainer> functions() {
    return std::span(storage_->containers).subspan(storage_->kernel_count);
  }
  ConSanProgramContainer &add_kernel(ConSanProgramContainer container = {}) {
    container.kind = ConSanProgramContainerKind::Kernel;
    if (storage_->kernel_count != storage_->containers.size())
      throw std::logic_error("ConSan kernels must be staged before functions");
    container.id = {static_cast<uint32_t>(storage_->containers.size())};
    storage_->containers.push_back(std::move(container));
    ++storage_->kernel_count;
    return storage_->containers.back();
  }
  ConSanProgramContainer &add_function(ConSanProgramContainer container = {}) {
    container.kind = ConSanProgramContainerKind::Function;
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
  [[nodiscard]] std::span<ConSanProgramSite> access_sites() {
    return storage_->program_sites.access_sites();
  }
  [[nodiscard]] std::vector<ConSanProgramSite> &program_sites() {
    return storage_->program_sites.sites();
  }
  void add_access_site(ConSanProgramSite site) {
    storage_->program_sites.add_access(std::move(site));
  }
  void add_semantic_site(ConSanProgramSite site) {
    storage_->program_sites.add_semantic(std::move(site));
  }
  void add_sites(ConSanProgramSiteArena sites) { storage_->program_sites.append(std::move(sites)); }
  template <typename Predicate> void erase_sites_if(Predicate predicate) {
    storage_->program_sites.erase_if(predicate);
  }
  /// Transfer semantic facets in an installed code range to their final
  /// owner without disturbing the access prefix. Symbol aliases can decode
  /// the same semantic instruction more than once: retain one facet, but
  /// retain every independently normalized access record.
  void reattribute_semantic_range(uint64_t begin, uint32_t size,
                                  const ConSanProgramContainer &owner) {
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
        .moi_fence_candidates = storage_->moi_fence_candidates,
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
    for (ConSanProgramSite &access : access_sites()) {
      if (!access.physical_id.code_object.valid())
        access = complete_access_site(access.container, access, code_object_bytes);
    }
    for (ConSanProgramSite &site : storage_->program_sites.sites()) {
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
  void reattribute_access_range(uint64_t begin, uint32_t size, const ConSanProgramContainer &owner,
                                std::span<const ConSanProgramSite> decoded_accesses,
                                std::span<const uint8_t> code_object_bytes) {
    const uint64_t end = begin + size;
    const ConSanProgramContainerId reference = owner.id;
    for (ConSanProgramSite &access : access_sites()) {
      if (access.physical_id.original_text_offset >= begin &&
          access.physical_id.original_text_offset < end)
        access.container = reference;
    }
    for (const ConSanProgramSite &access : decoded_accesses) {
      ConSanProgramSite completed = complete_access_site(reference, access, code_object_bytes);
      const bool duplicate = std::ranges::any_of(access_sites(), [&](const auto &existing) {
        return existing.container == reference && existing.physical_id == completed.physical_id &&
               (existing.origin == ConSanAccessOrigin::Flat) ==
                   (completed.origin == ConSanAccessOrigin::Flat);
      });
      if (!duplicate)
        add_access_site(std::move(completed));
    }
  }

private:
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
  single_range_native_static_byte_offset(const ConSanProgramSite &site,
                                         std::span<const uint8_t> bytes) const {
    if (site.origin == ConSanAccessOrigin::DirectToLds)
      return 0;
    if (site.kind == ConSanLdsAccessKind::Atomic) {
      const ConSanAtomicSite *atomic = site.get_if<ConSanAtomicSite>();
      if (atomic != nullptr && atomic->raw_ioffset)
        return static_cast<int64_t>(*atomic->raw_ioffset);
      const ConSanTargetProfile *target = consan_target_profile(storage_->arch);
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
        .domain = ConSanSemanticSiteDomain::Access,
        .member_ordinal = 0,
        .range_ordinal = range_ordinal,
    };
  }

  [[nodiscard]] ConSanProgramSite complete_access_site(const ConSanProgramContainerId &container,
                                                       const ConSanProgramSite &site,
                                                       std::span<const uint8_t> bytes) const {
    ConSanProgramSite result = site;
    result.physical_id = {
        .code_object = storage_->code_object_id,
        .original_text_offset = site.physical_id.original_text_offset,
    };
    result.container = container;
    result.execution_owners.clear();
    result.ranges.clear();
    result.exclusions.clear();
    if (site.origin == ConSanAccessOrigin::Flat) {
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
      result.lowering = classify_consan_access_lowering(result, storage_->arch);
      return result;
    }

    result.address_space = ConSanAccessAddressSpace::Group;
    result.provenance = ConSanAccessProvenance::NativeLdsOpcode;
    result.confidence = ConSanSemanticConfidence::Exact;
    append_common_exclusions(result, site.origin == ConSanAccessOrigin::DirectToLds);
    if (const auto form = consan_detail::decode_native_lds_two_range_shape(site.mnemonic_view())) {
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
            .reason = ConSanInventoryExclusionReason::RangeEncodingUnavailable,
        });
      }
    } else if (site.decoded_width_bits != 0 && site.decoded_width_bits % 8u == 0) {
      result.ranges.push_back(
          {.id = access_range_id(site.physical_id.original_text_offset, 0),
           .static_byte_offset = single_range_native_static_byte_offset(site, bytes),
           .byte_width = site.decoded_width_bits / 8u});
    }
    result.lowering = classify_consan_access_lowering(result, storage_->arch);
    return result;
  }

  static void append_common_exclusions(ConSanProgramSite &site, bool address_operand_not_required) {
    if (site.kind == ConSanLdsAccessKind::Other) {
      site.exclusions.push_back({
          .reason = ConSanInventoryExclusionReason::NonAccessInstruction,
      });
    }
    if (site.size() == 0 || site.size() % sizeof(uint32_t) != 0) {
      site.exclusions.push_back({
          .reason = ConSanInventoryExclusionReason::InvalidInstructionSize,
      });
    }
    if (site.decoded_width_bits == 0 || site.decoded_width_bits % 8u != 0) {
      site.exclusions.push_back({
          .reason = ConSanInventoryExclusionReason::InvalidAccessWidth,
      });
    }
    if (!address_operand_not_required && !site.operands.address_vgpr) {
      site.exclusions.push_back({
          .reason = ConSanInventoryExclusionReason::MissingAddressOperand,
      });
    }
  }

  /// Internal tuple returned while normalizing a legacy FLAT provenance hint.
  /// It keeps address-space, source, and confidence assignments adjacent so a
  /// new hint cannot update one semantic dimension while silently leaving the
  /// others at an unrelated default.
  struct FlatProvenanceFacts {
    ConSanAccessAddressSpace address_space;
    ConSanAccessProvenance provenance;
    ConSanSemanticConfidence confidence;
  };

  [[nodiscard]] static FlatProvenanceFacts flat_provenance(ConSanFlatAddressSpaceHint hint) {
    switch (hint) {
    case ConSanFlatAddressSpaceHint::Group:
      return {ConSanAccessAddressSpace::Group, ConSanAccessProvenance::EncodedFlatSegment,
              ConSanSemanticConfidence::Exact};
    case ConSanFlatAddressSpaceHint::Private:
    case ConSanFlatAddressSpaceHint::Global:
      return {ConSanAccessAddressSpace::NonGroup, ConSanAccessProvenance::EncodedFlatSegment,
              ConSanSemanticConfidence::Exact};
    case ConSanFlatAddressSpaceHint::MaybeGroup:
      return {ConSanAccessAddressSpace::Group, ConSanAccessProvenance::PropagatedFlatPointer,
              ConSanSemanticConfidence::Conservative};
    case ConSanFlatAddressSpaceHint::MaybePrivate:
      return {ConSanAccessAddressSpace::NonGroup, ConSanAccessProvenance::PropagatedFlatPointer,
              ConSanSemanticConfidence::Conservative};
    case ConSanFlatAddressSpaceHint::Unknown:
      return {ConSanAccessAddressSpace::Unresolved, ConSanAccessProvenance::UnresolvedFlatPointer,
              ConSanSemanticConfidence::Ambiguous};
    }
    return {ConSanAccessAddressSpace::Unresolved, ConSanAccessProvenance::UnresolvedFlatPointer,
            ConSanSemanticConfidence::Unsupported};
  }

  std::shared_ptr<ConSanProgramInventoryStorage> storage_;
};
