/// Return the physical offsets whose ordinary-access bodies must be placed
/// after ordinary candidates so a later ordering pass can still wrap their
/// relocated guest instruction within SOPP reach. ConSan
/// use the same qualified synchronization inventory; disabled atomic tracking
/// deliberately produces no placement constraint.
[[nodiscard]] std::unordered_set<uint64_t>
collect_ordering_atomic_offsets(const ObservationPlan &observation_plan, bool track_atomics) {
  std::unordered_set<uint64_t> offsets;
  if (!track_atomics)
    return offsets;
  for (const AtomicSiteDecision &decision : observation_plan.atomic_site_decisions) {
    if (decision.kind == SiteDecisionKind::Admitted)
      offsets.insert(decision.semantic_site.physical.original_text_offset);
  }
  return offsets;
}

/// Return the shared placement order for replayable access bodies. Physical
/// order is deterministic, while an access that is also an admitted atomic
/// communication event moves to the tail so the later synchronization pass
/// can still wrap its relocated guest within direct-branch reach. This is a
/// placement constraint shared by ConSan, not evidence
/// selection; the input already contains only policy-admitted candidates.
[[nodiscard]] std::vector<const Candidate *>
order_replay_access_candidates(std::span<const Candidate> admitted,
                               const ObservationPlan &observation_plan, bool track_atomics) {
  std::vector<const Candidate *> candidates = detail::make_candidate_pointer_view(admitted);
  std::ranges::sort(candidates, [](const Candidate *lhs, const Candidate *rhs) {
    return lhs->anchor() < rhs->anchor();
  });
  const std::unordered_set<uint64_t> ordering_atomic_offsets =
      collect_ordering_atomic_offsets(observation_plan, track_atomics);
  std::stable_partition(candidates.begin(), candidates.end(), [&](const auto *candidate) {
    return !ordering_atomic_offsets.contains(candidate->anchor());
  });
  return candidates;
}

[[nodiscard]] ScalarPlacementModel scalar_placement_model(rj_code_arch_t arch) {
  const TargetProfile *target = target_profile(arch);
  return target ? target->scalar_placement_model : ScalarPlacementModel::Unsupported;
}

/// Return the first aligned register range admitted by one placement policy.
/// Candidate order is part of the generated-code contract: callers provide
/// their original search floor and alignment while this shared traversal owns
/// the common first-fit mechanics.
template <typename Admitted>
[[nodiscard]] std::optional<uint16_t>
first_admitted_register_range(uint32_t begin, uint16_t width, uint16_t limit, uint16_t alignment,
                              Admitted &&admitted) {
  for (uint32_t base = util::align_up(begin, static_cast<uint32_t>(alignment));
       base + width <= limit; base += alignment) {
    const uint16_t candidate = static_cast<uint16_t>(base);
    if (admitted(candidate))
      return candidate;
  }
  return std::nullopt;
}

[[nodiscard]] uint32_t descriptor_user_sgpr_count(const KD &desc) {
  return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
}

[[nodiscard]] uint16_t descriptor_system_sgpr_count(const KD &desc) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  return static_cast<uint16_t>(
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) != 0 ? 1u : 0u));
}

[[nodiscard]] uint16_t descriptor_dispatch_id_prefix_sgpr_count(const KD &desc) {
  const uint16_t properties = desc.kernel_code_properties;
  return amdhsa_dispatch_id_prefix_sgpr_count(
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) != 0);
}

[[nodiscard]] std::optional<uint16_t> descriptor_kernarg_segment_ptr_sgpr(const KD &desc) {
  const uint16_t properties = desc.kernel_code_properties;
  if (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) == 0)
    return std::nullopt;
  return amdhsa_dispatch_id_prefix_sgpr_count(
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0, false);
}

[[nodiscard]] DispatchIdPreloadPlan descriptor_dispatch_id_preload_plan(const KD &desc,
                                                                        rj_code_arch_t arch) {
  const uint16_t properties = desc.kernel_code_properties;
  const uint16_t original_user_count = static_cast<uint16_t>(descriptor_user_sgpr_count(desc));
  const uint16_t system_count = descriptor_system_sgpr_count(desc);
  const bool queue_enabled =
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0;
  const bool dispatch_enabled =
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) != 0;
  const uint16_t private_and_dispatch_ptr = static_cast<uint16_t>(
      (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) != 0
           ? 4u
           : 0u) +
      (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) != 0 ? 2u
                                                                                           : 0u));
  const bool kernarg_enabled =
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) != 0;
  const uint16_t prefix =
      static_cast<uint16_t>(private_and_dispatch_ptr + 2u + (kernarg_enabled ? 2u : 0u));
  const uint16_t named_suffix = static_cast<uint16_t>(
      (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT) != 0
           ? 2u
           : 0u) +
      (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE) != 0
           ? 1u
           : 0u) +
      AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH));
  const uint16_t original_named_count = static_cast<uint16_t>(
      private_and_dispatch_ptr + (queue_enabled ? 2u : 0u) + (kernarg_enabled ? 2u : 0u) +
      (dispatch_enabled ? 2u : 0u) + named_suffix);
  const uint32_t minimum_declared_user_count = original_named_count;
  if (minimum_declared_user_count > original_user_count) {
    DispatchIdPreloadPlan invalid;
    invalid.support = DispatchIdPreloadSupport::InvalidDispatchPosition;
    invalid.dispatch_id_sgpr = prefix;
    invalid.original_user_sgpr_count = original_user_count;
    invalid.expanded_user_sgpr_count = original_user_count;
    invalid.system_sgpr_count = system_count;
    return invalid;
  }
  const TargetProfile *profile = target_profile(arch);
  const uint16_t user_sgpr_initialization_limit =
      profile ? profile->user_sgpr_initialization_limit : 16u;
  DispatchIdPreloadPlan plan;
  plan.queue_ptr_was_enabled = queue_enabled;
  plan.dispatch_id_was_enabled = dispatch_enabled;
  plan.dispatch_id_sgpr = prefix;
  plan.identity_salt_sgpr = private_and_dispatch_ptr;
  plan.original_user_sgpr_count = original_user_count;
  plan.system_sgpr_count = system_count;
  const uint16_t inserted =
      static_cast<uint16_t>((queue_enabled ? 0u : 2u) + (dispatch_enabled ? 0u : 2u));
  const uint16_t desired_named_count = static_cast<uint16_t>(original_named_count + inserted);
  uint16_t expanded_user_count = std::max(original_user_count, desired_named_count);
  const uint16_t preload_length =
      AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH);
  plan.original_kernarg_preload_length = preload_length;
  plan.replacement_kernarg_preload_length = preload_length;
  const auto kernarg_ptr = descriptor_kernarg_segment_ptr_sgpr(desc);
  // RDNA3 and CDNA3/4 initialize at most 16 user SGPRs. When dispatch ID would
  // exceed that window by one or two dwords, shorten the hardware kernarg
  // preload by exactly that overflow and recover those guest values from the
  // still-live kernarg pointer in the entry prologue. System SGPRs move only
  // by the remaining user-window growth, so record that smaller repair
  // separately.
  const uint16_t overflow =
      expanded_user_count > user_sgpr_initialization_limit
          ? static_cast<uint16_t>(expanded_user_count - user_sgpr_initialization_limit)
          : 0u;
  if (profile && profile->supports_kernarg_preload_overflow_recovery && inserted != 0u &&
      overflow >= 1u && overflow <= 4u && kernarg_ptr && preload_length > overflow &&
      original_named_count == original_user_count) {
    expanded_user_count = user_sgpr_initialization_limit;
    plan.kernarg_reload_sgpr = static_cast<uint16_t>(original_user_count - overflow);
    // Reload executes after the explicit guest-SGPR repair.  At that point
    // the kernarg pointer is back in its original ABI position; its shifted
    // replacement position may already have been overwritten by another
    // restored guest preload.
    plan.kernarg_reload_base_sgpr = *kernarg_ptr;
    plan.kernarg_reload_offset_dwords = static_cast<uint16_t>(
        AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET) + preload_length -
        overflow);
    plan.kernarg_reload_count = overflow;
    plan.replacement_kernarg_preload_length = static_cast<uint16_t>(preload_length - overflow);
  }
  if (expanded_user_count > user_sgpr_initialization_limit) {
    plan.support = DispatchIdPreloadSupport::UserSgprInitializationLimit;
    plan.expanded_user_sgpr_count = original_user_count;
    return plan;
  }
  if (static_cast<uint32_t>(expanded_user_count) + system_count > kMaxSgprs) {
    plan.support = DispatchIdPreloadSupport::SgprAllocationLimit;
    plan.expanded_user_sgpr_count = original_user_count;
    return plan;
  }
  plan.support = inserted == 0u ? DispatchIdPreloadSupport::SupportedAlreadyEnabled
                                : DispatchIdPreloadSupport::SupportedInsert;
  plan.expanded_user_sgpr_count = expanded_user_count;
  plan.system_sgpr_shift = static_cast<uint16_t>(expanded_user_count - original_user_count);
  plan.shifted_system_sgpr_count = plan.system_sgpr_shift != 0u ? system_count : 0u;
  plan.required_sgpr_count = static_cast<uint16_t>(expanded_user_count + system_count);

  uint16_t original_pos = 0u;
  uint16_t replacement_pos = 0u;
  const auto map_segment = [&](uint16_t count) {
    for (uint16_t i = 0u; i < count; ++i) {
      if (original_pos != replacement_pos &&
          original_pos < original_user_count - plan.kernarg_reload_count &&
          plan.guest_restore_count < plan.guest_restore_sources.size()) {
        plan.guest_restore_destinations[plan.guest_restore_count] = original_pos;
        plan.guest_restore_sources[plan.guest_restore_count] = replacement_pos;
        ++plan.guest_restore_count;
      }
      ++original_pos;
      ++replacement_pos;
    }
  };
  map_segment(private_and_dispatch_ptr);
  if (queue_enabled)
    map_segment(2u);
  else
    replacement_pos += 2u;
  if (kernarg_enabled)
    map_segment(2u);
  if (dispatch_enabled)
    map_segment(2u);
  else
    replacement_pos += 2u;
  map_segment(static_cast<uint16_t>(named_suffix - plan.kernarg_reload_count));
  if (plan.guest_restore_count != 0u) {
    plan.first_shifted_guest_sgpr = plan.guest_restore_destinations[0];
    plan.shifted_guest_sgpr_count = plan.guest_restore_count;
  }
  return plan;
}

[[nodiscard]] std::optional<uint16_t> descriptor_workgroup_id_sgpr(const KD &desc,
                                                                   uint32_t dimension) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  const bool enabled[3] = {
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0,
  };

  uint32_t sgpr = descriptor_user_sgpr_count(desc);
  for (uint32_t i = 0; i < 3; ++i) {
    if (!enabled[i])
      continue;
    if (i == dimension)
      return static_cast<uint16_t>(sgpr);
    ++sgpr;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<WorkgroupSources>
descriptor_workgroup_sources(std::span<const uint8_t> image, uint64_t descriptor_file_offset,
                             rj_code_arch_t arch, std::vector<std::string> &errors,
                             bool uses_cluster_workgroup_id,
                             std::optional<uint16_t> cdna_full_payload_user_sgpr_count) {
  const auto descriptor = read_kernel_descriptor(image, descriptor_file_offset);
  if (!descriptor) {
    errors.emplace_back("ConSan workgroup-id descriptor exceeds ELF bytes");
    return std::nullopt;
  }

  const TargetProfile *profile = target_profile(arch);
  if (profile && profile->command_processor_workgroup_identity) {
    const auto &identity = *profile->command_processor_workgroup_identity;
    // These are translated command-processor ABI TTMPs, not the
    // descriptor-selected entry SGPRs copied from them by the kernel prologue.
    // ConSan finalization enables the complete command-processor grid payload on
    // every patched owner descriptor. Preserve that full identity: an x-only
    // guest kernel can still be dispatched over a multidimensional grid, and
    // each workgroup owns a distinct LDS instance.
    return WorkgroupSources{
        .x = WorkgroupSource::scalar(ttmp_scalar_operand(identity.grid_x_ttmp)),
        .y = WorkgroupSource::scalar(ttmp_scalar_operand(identity.grid_yz_ttmp),
                                     /*right_shift=*/0u,
                                     /*low_bit_count=*/16u),
        .z = WorkgroupSource::scalar(ttmp_scalar_operand(identity.grid_yz_ttmp),
                                     /*right_shift=*/16u),
        .cluster_workgroup_id =
            uses_cluster_workgroup_id && identity.cluster_workgroup_id_ttmp
                ? WorkgroupSource::scalar(ttmp_scalar_operand(*identity.cluster_workgroup_id_ttmp),
                                          /*right_shift=*/0u,
                                          identity.cluster_workgroup_id_low_bit_count)
                : WorkgroupSource{},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  if (cdna_full_payload_user_sgpr_count && profile) {
    const uint16_t full_base = *cdna_full_payload_user_sgpr_count;
    if (static_cast<uint32_t>(full_base) + 3u > kMaxSgprs) {
      errors.emplace_back("ConSan full CDNA workgroup-ID payload exceeds scalar registers");
      return std::nullopt;
    }
    const uint32_t rsrc2 = descriptor->compute_pgm_rsrc2;
    uint8_t guest_mask = 0;
    guest_mask |= AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0
                      ? 1u << 0u
                      : 0u;
    guest_mask |= AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0
                      ? 1u << 1u
                      : 0u;
    guest_mask |= AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0
                      ? 1u << 2u
                      : 0u;
    guest_mask |= AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) != 0
                      ? 1u << 3u
                      : 0u;
    return WorkgroupSources{
        .x = WorkgroupSource::scalar(full_base),
        .y = WorkgroupSource::scalar(static_cast<uint16_t>(full_base + 1u)),
        .z = WorkgroupSource::scalar(static_cast<uint16_t>(full_base + 2u)),
        .cluster_workgroup_id = {},
        .cdna_full_payload_base = full_base,
        .cdna_guest_payload_base = static_cast<uint16_t>(descriptor_user_sgpr_count(*descriptor)),
        .cdna_guest_payload_mask = guest_mask,
    };
  }
  const auto sgpr_source = [](std::optional<uint16_t> sgpr) {
    return sgpr ? WorkgroupSource::scalar(*sgpr) : WorkgroupSource{};
  };
  return WorkgroupSources{
      .x = sgpr_source(descriptor_workgroup_id_sgpr(*descriptor, 0)),
      .y = sgpr_source(descriptor_workgroup_id_sgpr(*descriptor, 1)),
      .z = sgpr_source(descriptor_workgroup_id_sgpr(*descriptor, 2)),
      .cluster_workgroup_id = {},
      .cdna_full_payload_base = std::nullopt,
      .cdna_guest_payload_base = std::nullopt,
      .cdna_guest_payload_mask = 0u,
  };
}

/// Exact-coordinate probes execute after arbitrary guest code. Capture the
/// complete tuple at entry instead of depending on descriptor SGPRs or RDNA
/// launch TTMPs retaining their entry values.
[[nodiscard]] std::optional<WorkgroupSources>
exact_entry_workgroup_sources(const OperatingPoint &point,
                              const PersistentWorkgroupPrivateOffsets *private_offsets) {
  if (!has_exact_entry_workgroup_capture(point, private_offsets) ||
      !exact_entry_workgroup_capture_is_unambiguous(point, private_offsets)) {
    return std::nullopt;
  }
  const auto scalar_source = [](uint16_t sgpr) { return WorkgroupSource::scalar(sgpr); };
  const auto vector_source = [](uint16_t vgpr) { return WorkgroupSource::vector(vgpr); };
  const auto private_source = [](uint32_t offset) {
    return WorkgroupSource::private_state(offset);
  };
  if (point.persistent_sgprs.exact_workgroup.complete()) {
    return WorkgroupSources{
        .x = scalar_source(*point.persistent_sgprs.exact_workgroup.x()),
        .y = scalar_source(*point.persistent_sgprs.exact_workgroup.y()),
        .z = scalar_source(*point.persistent_sgprs.exact_workgroup.z()),
        .cluster_workgroup_id =
            point.persistent_sgprs.exact_workgroup.cluster_workgroup_id()
                ? scalar_source(*point.persistent_sgprs.exact_workgroup.cluster_workgroup_id())
                : WorkgroupSource{},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  if (point.exact_workgroup_vgprs.complete()) {
    return WorkgroupSources{
        .x = vector_source(*point.exact_workgroup_vgprs.x()),
        .y = vector_source(*point.exact_workgroup_vgprs.y()),
        .z = vector_source(*point.exact_workgroup_vgprs.z()),
        .cluster_workgroup_id =
            point.exact_workgroup_vgprs.cluster_workgroup_id()
                ? vector_source(*point.exact_workgroup_vgprs.cluster_workgroup_id())
                : WorkgroupSource{},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  if (private_offsets && private_offsets->complete()) {
    return WorkgroupSources{
        .x = private_source(*private_offsets->x()),
        .y = private_source(*private_offsets->y()),
        .z = private_source(*private_offsets->z()),
        .cluster_workgroup_id = private_offsets->cluster_workgroup_id()
                                    ? private_source(*private_offsets->cluster_workgroup_id())
                                    : WorkgroupSource{},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  return std::nullopt;
}

struct KernelResourceContext {
  const ProgramContainer *kernel = nullptr;
  KernelCfgScope scope;
  std::unordered_set<BasicBlock *> block_set;
  std::unique_ptr<LivenessAnalysis> liveness;
  uint16_t current_vgpr_count = 0;
  uint16_t max_referenced_vgpr_count = 0;
  uint16_t ordinary_vgpr_limit = kMaxVgprs;
  bool accumulator_boundary_is_allocation_tail = false;
  bool can_move_empty_accumulator_boundary = false;
  uint16_t current_sgpr_count = 0;
  uint16_t max_referenced_sgpr_count = 0;
  uint16_t initialized_sgpr_count = 0;
  uint32_t original_private_segment_size = 0;
  std::optional<bool> uses_dynamic_stack;
  bool has_indirect_sgpr_access = false;
  bool sgpr_reference_coverage_complete = false;
  bool has_indirect_vgpr_access = false;
  bool descriptor_valid = false;

  [[nodiscard]] detail::ScalarOwnerContextSummary scalar_summary() const {
    return {
        .descriptor_file_offset =
            kernel ? std::optional{kernel->descriptor_file_offset} : std::nullopt,
        .current_sgpr_count = current_sgpr_count,
        .max_referenced_sgpr_count = max_referenced_sgpr_count,
        .has_indirect_sgpr_access = has_indirect_sgpr_access,
        .sgpr_reference_coverage_complete = sgpr_reference_coverage_complete,
        .descriptor_valid = descriptor_valid,
    };
  }
};

[[nodiscard]] const PersistentVgprAssignment *
persistent_vgpr_assignment(OwnerAssignments assignments, uint64_t descriptor_offset) {
  const auto assignment = std::ranges::find(assignments.persistent_vgprs, descriptor_offset,
                                            &PersistentVgprAssignment::descriptor_file_offset);
  return assignment == assignments.persistent_vgprs.end() ? nullptr : &*assignment;
}

[[nodiscard]] const TransientSgprAssignment *transient_sgpr_assignment(OwnerAssignments assignments,
                                                                       uint64_t descriptor_offset) {
  const auto assignment = std::ranges::find(assignments.transient_sgprs, descriptor_offset,
                                            &TransientSgprAssignment::descriptor_file_offset);
  return assignment == assignments.transient_sgprs.end() ? nullptr : &*assignment;
}

/// Returns true when automatic scalar spilling selected after the initial
/// resource pass must share a site-local frame with a dynamic-stack owner.
/// The first pass is still useful: it avoids reserving the dynamic frame ABI
/// for ordinary owners that can keep their transient state in registers.
[[nodiscard]] bool automatic_scalar_spill_needs_dynamic_stack_planning(
    const OperatingPoint &point, const ProgramInventory &inventory,
    std::span<const CandidateResourcePlan> site_plans) {
  const auto descriptor_owner_uses_dynamic_stack = [&](uint64_t descriptor_offset) {
    const ProgramContainer *kernel = inventory.find_kernel_by_descriptor(descriptor_offset);
    return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
  };
  if (std::ranges::any_of(
          point.owner_transient_sgprs, [&](const TransientSgprAssignment &assignment) {
            return assignment.spill_backed &&
                   descriptor_owner_uses_dynamic_stack(assignment.descriptor_file_offset);
          })) {
    return true;
  }
  if (!point.has_scalar_spill()) {
    return false;
  }
  return std::ranges::any_of(site_plans, [&](const CandidateResourcePlan &plan) {
    return plan.source != RegisterAllocationSource::Unsupported &&
           std::ranges::any_of(plan.owner_kernel_ids, [&](ProgramContainerId owner) {
             const ProgramContainer *kernel = inventory.container(owner);
             return kernel != nullptr && kernel->is_kernel() &&
                    kernel->uses_dynamic_stack.value_or(false);
           });
  });
}

/// Applies the one transient scalar assignment shared by every owner of a
/// site. Owners connected through a shared helper must either all carry the
/// same assignment or keep the code-object-wide fallback.
[[nodiscard]] bool
apply_transient_sgpr_assignment(const Request &, OperatingPoint &point,
                                OwnerAssignments assignments,
                                std::span<const uint64_t> owner_descriptor_offsets) {
  const TransientSgprAssignment *selected = nullptr;
  bool saw_unassigned_owner = false;
  for (uint64_t descriptor_offset : owner_descriptor_offsets) {
    const auto *assignment = transient_sgpr_assignment(assignments, descriptor_offset);
    if (!assignment) {
      saw_unassigned_owner = true;
      continue;
    }
    if (selected) {
      TransientSgprAssignment comparable = *assignment;
      comparable.descriptor_file_offset = selected->descriptor_file_offset;
      if (comparable != *selected)
        return false;
    }
    selected = assignment;
  }
  if (!selected) {
    return point.exec_save_sgpr.has_value();
  }
  if (saw_unassigned_owner)
    return false;
  point.exec_save_sgpr = selected->exec_save_sgpr;
  point.owner_sgpr.set(selected->owner_sgpr, point.owner_sgpr.automatic());
  // A component assignment replaces the object-wide scalar ABI. Its exact
  // persistent outputs are represented by the fields copied below, so the
  // broad object-wide persistence marker must not reject a spill-backed
  // window whose incoming contents are deliberately backed up at entry.
  if (selected->spill_backed)
    point.exec_save_sgprs_persistent = false;
  point.dispatch_sgpr.set(selected->dispatch_id_sgpr, point.dispatch_sgpr.automatic());
  point.scalar_spill_setup = selected->scalar_spill_setup;

  point.automatic_scalar_spill_layout =
      selected->spill_backed ? ScalarSpillLayout::Compact : ScalarSpillLayout::None;
  point.branch_only_spill.reset();

  return true;
}

[[nodiscard]] bool
apply_persistent_vgpr_assignment(OperatingPoint &point, OwnerAssignments assignments,
                                 std::span<const uint64_t> owner_descriptor_offsets) {
  if (assignments.persistent_vgprs.empty()) {
    return point.owner_epoch_vgprs.owner().has_value() || point.persistent_sgprs.complete() ||
           point.automatic_private_epoch;
  }
  const PersistentVgprAssignment *selected = nullptr;
  size_t assigned_owner_count = 0u;
  for (uint64_t descriptor_offset : owner_descriptor_offsets) {
    const auto *assignment = persistent_vgpr_assignment(assignments, descriptor_offset);
    if (!assignment)
      continue;
    ++assigned_owner_count;
    if (selected) {
      PersistentVgprAssignment comparable = *assignment;
      comparable.descriptor_file_offset = selected->descriptor_file_offset;
      if (comparable != *selected)
        return false;
    }
    selected = assignment;
  }
  if (!selected) {
    if (!point.automatic_private_epoch)
      return false;
    point.reset_owner_epoch_vgprs();
    point.exact_workgroup_vgprs = {};
    return true;
  }
  // One physical site cannot consume two different persistent
  // representations. Components are constructed from shared ownership, so a
  // partial match indicates stale or incomplete placement metadata.
  if (assigned_owner_count != owner_descriptor_offsets.size())
    return false;
  point.automatic_private_epoch = false;
  point.set_owner_epoch_vgprs(selected->owner_epoch_vgprs.owner, selected->owner_epoch_vgprs.epoch);
  point.exact_workgroup_vgprs = selected->exact_workgroup_vgprs;
  return true;
}

[[nodiscard]] bool is_indirect_sgpr_access(const Instruction &instruction) {
  return instruction.mnemonic().starts_with("s_movrel");
}

[[nodiscard]] bool is_indirect_vgpr_access(const Instruction &instruction) {
  const std::string_view mnemonic = instruction.mnemonic();
  return mnemonic.starts_with("v_movrel") || mnemonic.starts_with("v_swaprel") ||
         mnemonic.starts_with("s_set_gpr_idx") || mnemonic == "s_setreg_b32" ||
         mnemonic == "s_setreg_imm32_b32";
}

[[nodiscard]] uint16_t max_referenced_vgpr_count(const KernelCfgScope &scope) {
  uint32_t max_count = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      const InstDefUse du(inst);
      const RegisterSet referenced = du.defs | du.uses;
      referenced.for_each([&](RegisterRef ref) {
        if (ref.cls == RegClass::VGPR)
          max_count = std::max<uint32_t>(max_count, static_cast<uint32_t>(ref.index) + 1u);
      });
    }
  }
  return static_cast<uint16_t>(std::min<uint32_t>(max_count, kMaxVgprs));
}

[[nodiscard]] uint16_t max_referenced_sgpr_count(const KernelCfgScope &scope) {
  uint32_t max_count = 0;
  for (BasicBlock *block : scope.blocks) {
    if (block == nullptr)
      continue;
    for (const Instruction &inst : block->instructions()) {
      const InstDefUse du(inst);
      const RegisterSet referenced = du.defs | du.uses;
      referenced.for_each([&](RegisterRef ref) {
        if (ref.cls == RegClass::SGPR)
          max_count = std::max<uint32_t>(max_count, static_cast<uint32_t>(ref.index) + 1u);
      });
    }
  }
  return static_cast<uint16_t>(std::min<uint32_t>(max_count, kMaxSgprs));
}

enum class UnclosedControlKind {
  Branch,
  Call,
  Fallthrough,
};

struct OwnerRegisterReferenceSummary {
  RegisterSet references;
  bool uses_indirect_sgprs = false;
  bool uses_indirect_vgprs = false;
  bool control_flow_closed = true;
  std::optional<uint64_t> first_unclosed_control_offset;
  std::string first_unclosed_control_mnemonic;
  UnclosedControlKind first_unclosed_control_kind = UnclosedControlKind::Branch;
};

[[nodiscard]] OwnerRegisterReferenceSummary
summarize_owner_register_references(std::span<const KernelResourceContext *const> owner_contexts) {
  OwnerRegisterReferenceSummary result;
  std::unordered_set<BasicBlock *> visited;
  const auto classify_unclosed = [](const Instruction &instruction) {
    if ((instruction.flags() & INDIRECT_CALL) != 0)
      return UnclosedControlKind::Call;
    if ((instruction.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH)) != 0)
      return UnclosedControlKind::Branch;
    return UnclosedControlKind::Fallthrough;
  };
  const auto mark_unclosed = [&](const Instruction &instruction,
                                 std::optional<UnclosedControlKind> kind = std::nullopt) {
    result.control_flow_closed = false;
    if (!result.first_unclosed_control_offset) {
      result.first_unclosed_control_offset = instruction.src_loc();
      result.first_unclosed_control_mnemonic = instruction.mnemonic();
      result.first_unclosed_control_kind = kind.value_or(classify_unclosed(instruction));
    }
  };
  const auto classify_static_issue = [](BasicBlock::StaticSuccessorIssue issue) {
    switch (issue) {
    case BasicBlock::StaticSuccessorIssue::MissingCallTarget:
    case BasicBlock::StaticSuccessorIssue::MissingCallContinuation:
      return UnclosedControlKind::Call;
    case BasicBlock::StaticSuccessorIssue::MissingBranchTarget:
      return UnclosedControlKind::Branch;
    case BasicBlock::StaticSuccessorIssue::MissingFallthrough:
    case BasicBlock::StaticSuccessorIssue::None:
      return UnclosedControlKind::Fallthrough;
    }
    return UnclosedControlKind::Fallthrough;
  };
  for (const KernelResourceContext *context : owner_contexts) {
    if (context == nullptr)
      continue;
    // Reuse liveness's architecture-aware completeness proof. In particular,
    // S_SET_GPR_IDX_ON makes later ordinary VOP operands indirect even though
    // no V_MOVREL mnemonic appears in the kernel.
    result.uses_indirect_vgprs |= !context->liveness->global_vgpr_usage_is_complete();

    // Closure is context-specific: a setpc can be a validated return for one
    // owner while remaining an unproven branch for another. Require every CFG
    // edge and every exhaustive recovered target to remain inside this owner's
    // scope before using owner-local references as a whole-program proof.
    std::unordered_set<const BasicBlock *> scope_blocks;
    std::unordered_set<uint64_t> scope_block_offsets;
    for (BasicBlock *block : context->scope.blocks) {
      if (block == nullptr)
        continue;
      scope_blocks.insert(block);
      scope_block_offsets.insert(block->start_offset());
    }

    for (BasicBlock *block : context->scope.blocks) {
      if (block == nullptr)
        continue;

      const Instruction *terminator = block->terminator();
      const auto mark_unclosed_terminator = [&]() {
        if (terminator != nullptr)
          mark_unclosed(*terminator);
      };
      if (!block->static_successors_complete() && terminator != nullptr)
        mark_unclosed(*terminator, classify_static_issue(block->static_successor_issue()));
      if (std::ranges::any_of(block->successors(), [&](const BasicBlock *successor) {
            return successor == nullptr || !scope_blocks.contains(successor);
          })) {
        mark_unclosed_terminator();
      }
      if (std::ranges::any_of(block->call_edges(), [&](const BasicBlock::CallEdge &call) {
            return call.callee == nullptr || call.continuation == nullptr ||
                   !scope_blocks.contains(call.callee) || !scope_blocks.contains(call.continuation);
          })) {
        mark_unclosed_terminator();
      }

      const bool scan_references = visited.insert(block).second;
      for (const Instruction &instruction : block->instructions()) {
        const uint64_t offset = instruction.src_loc();
        const bool indirect_call =
            (instruction.flags() & INDIRECT_CALL) != 0 && !instruction.branch_offset_bytes();
        const bool indirect_branch = (instruction.flags() & INDIRECT_BRANCH) != 0;
        const bool validated_return = context->scope.call_return_offsets.contains(offset);
        if (indirect_call || (indirect_branch && !validated_return)) {
          bool found_fixup = false;
          bool targets_exhaustive = true;
          for (const IndirectCallFixup &fixup : block->static_indirect_call_fixups()) {
            if (fixup.source_call_offset != offset)
              continue;
            found_fixup = true;
            targets_exhaustive = targets_exhaustive && fixup.source_targets_exhaustive &&
                                 scope_block_offsets.contains(fixup.source_target_offset);
          }
          if (!found_fixup || !targets_exhaustive)
            mark_unclosed(instruction);
        }

        if (!scan_references)
          continue;
        if (is_indirect_sgpr_access(instruction))
          result.uses_indirect_sgprs = true;
        if (is_indirect_vgpr_access(instruction))
          result.uses_indirect_vgprs = true;
        const InstDefUse def_use(instruction);
        result.references |= def_use.defs | def_use.uses;
      }
    }
  }
  return result;
}

[[nodiscard]] const Instruction *instruction_at(BasicBlock *block, uint64_t text_offset) {
  if (block == nullptr)
    return nullptr;
  const auto it =
      std::ranges::find_if(block->instructions(), [text_offset](const Instruction &inst) {
        return inst.src_loc() == text_offset;
      });
  return it == block->instructions().end() ? nullptr : &*it;
}

struct ResourcePlanningState {
  std::span<const uint8_t> bytes;
  rj_code_arch_t arch;
  ProgramInventory inventory;
  /// Register allocation selected by this lowering attempt. The referenced
  /// operating point is stable for the lifetime of the planning context even
  /// while placement appends owner-local assignments to its vectors.
  const OperatingPoint &register_allocation;
  /// Bytes unavailable to this lowering attempt. The initial entries come
  /// from the immutable staged-image inventory; ConSan may append reservations
  /// for deferred prologue anchors without mutating that inventory.
  std::vector<PreappliedReservedRange> reserved_ranges;
  AmdGpuCodeObject code_object;
  std::unique_ptr<Decoder> decoder;
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  BlockOffsetIndex block_index;
  std::vector<KernelResourceContext> contexts;
  std::unordered_map<uint32_t, KernelResourceContext *> context_by_kernel_id;
  std::unordered_map<uint64_t, KernelResourceContext *> context_by_descriptor;
  mutable std::unordered_map<uint64_t, RegisterSet> dynamic_frame_saves_by_site;
  bool valid = false;

  ResourcePlanningState(const ResourceProblem &problem, const OperatingPoint &allocation,
                        std::span<const CandidateResourcePlan> prior_site_plans = {})
      : bytes(problem.image()), arch(problem.arch()), inventory(problem.inventory()),
        register_allocation(allocation),
        reserved_ranges(problem.inventory().preapplied_mutation().reserved_ranges),
        code_object(bytes.data(), bytes.size()), decoder(Decoder::create(arch)) {
    if (!code_object.is_valid() || !decoder)
      return;

    const detail::CfgBuildInputs cfg = detail::build_cfg_inputs_for_selection(
        code_object, problem.inventory().containers(),
        problem.inventory().preapplied_mutation().code_ranges, problem.request(), DebugOverrides{});
    blocks = BasicBlock::build(code_object, *decoder, arch, cfg.leaders, cfg.code_ranges);
    block_index = build_block_offset_index(blocks);
    CodeObjectPatcher patcher(code_object);
    const std::span<const uint8_t> text = patcher.text_bytes();

    // Resource planning only asks for liveness at instrumentation anchors and
    // kernel entries. Keeping a snapshot for every decoded instruction makes
    // large generated libraries scale with their complete instruction count,
    // even though most instructions are never queried. Per-owner transient
    // assignments are the one exception: deriving a component-wide safe
    // window may inspect any relocatable instruction in an owning block.
    const bool restrict_liveness_queries = register_allocation.owner_transient_sgprs.empty();
    const SynchronizationInventoryView sync = problem.inventory().sync();
    std::unordered_map<BasicBlock *, std::vector<const Instruction *>> liveness_queries_by_block;
    if (restrict_liveness_queries) {
      std::unordered_set<uint64_t> query_offsets;
      query_offsets.reserve(sync.sync_events.size() + prior_site_plans.size() +
                            problem.observation_plan().probe_intents.size() +
                            problem.inventory().kernels().size());
      for (const SyncEvent &event : sync.sync_events)
        query_offsets.insert(event.text_offset());
      for (const CandidateResourcePlan &plan : prior_site_plans)
        query_offsets.insert(plan.text_offset);
      for (const ProbeIntent &intent : problem.observation_plan().probe_intents)
        query_offsets.insert(intent.physical_site.original_text_offset);
      for (const ProgramContainer &kernel : problem.inventory().kernels()) {
        if (kernel.has_text_range)
          query_offsets.insert(kernel.entry_text_offset);
      }
      for (uint64_t offset : query_offsets) {
        BasicBlock *block = block_for_offset(block_index, offset);
        if (const Instruction *instruction = instruction_at(block, offset))
          liveness_queries_by_block[block].push_back(instruction);
      }
    }

    contexts.reserve(problem.inventory().kernels().size());
    for (const ProgramContainer &kernel : problem.inventory().kernels()) {
      if (!kernel.has_text_range)
        continue;
      std::vector<uint64_t> additional_entry_offsets;
      for (const PreappliedCodeRange &range :
           problem.inventory().preapplied_mutation().code_ranges) {
        if (range.kernel_name != kernel.name)
          continue;
        if (block_for_offset(block_index, range.text_offset) != nullptr)
          additional_entry_offsets.push_back(range.text_offset);
        if (range.continuation_text_offset &&
            block_for_offset(block_index, *range.continuation_text_offset) != nullptr)
          additional_entry_offsets.push_back(*range.continuation_text_offset);
      }
      auto scope = build_kernel_cfg_scope(
          blocks, block_index,
          KernelScopeRequest{.entry_offset = kernel.entry_text_offset,
                             .additional_entry_offsets = additional_entry_offsets},
          cfg.kernel_entries, text);
      if (!scope)
        continue;

      KernelResourceContext context;
      context.kernel = &kernel;
      context.scope = std::move(*scope);
      contexts.push_back(std::move(context));
      KernelResourceContext &stored = contexts.back();
      stored.block_set.insert(stored.scope.blocks.begin(), stored.scope.blocks.end());
      std::vector<const Instruction *> context_liveness_queries;
      LivenessAnalysisOptions liveness_options;
      liveness_options.arch = arch;
      liveness_options.entry_block = stored.scope.entry;
      liveness_options.text = text;
      std::vector<BasicBlock *> additional_entry_blocks;
      for (uint64_t offset : additional_entry_offsets) {
        if (BasicBlock *block = block_for_offset(block_index, offset))
          additional_entry_blocks.push_back(block);
      }
      liveness_options.additional_entry_blocks = additional_entry_blocks;
      if (restrict_liveness_queries) {
        for (BasicBlock *block : stored.scope.blocks) {
          const auto queries = liveness_queries_by_block.find(block);
          if (queries == liveness_queries_by_block.end())
            continue;
          context_liveness_queries.insert(context_liveness_queries.end(), queries->second.begin(),
                                          queries->second.end());
        }
        liveness_options.restrict_live_before_to_instructions = true;
        liveness_options.live_before_instructions = context_liveness_queries;
      }
      stored.liveness = std::make_unique<LivenessAnalysis>(
          KernelBlockScope(stored.scope.blocks), liveness_options, stored.scope.liveness_edges);
      stored.max_referenced_vgpr_count = max_referenced_vgpr_count(stored.scope);
      stored.max_referenced_sgpr_count = std::max<uint16_t>(max_referenced_sgpr_count(stored.scope),
                                                            kernel.sgpr_count.value_or(0u));
      const std::array<const KernelResourceContext *, 1> owner = {&stored};
      const OwnerRegisterReferenceSummary scalar_references =
          summarize_owner_register_references(owner);
      stored.has_indirect_sgpr_access = scalar_references.uses_indirect_sgprs;
      stored.sgpr_reference_coverage_complete = scalar_references.control_flow_closed;
      // Static live-before sets cannot name registers reached through relative
      // or runtime-indexed VGPR operands. Use the liveness analysis's shared
      // completeness gate instead of maintaining a second, narrower opcode
      // inventory here.
      stored.has_indirect_vgpr_access = !stored.liveness->global_vgpr_usage_is_complete();

      const auto descriptor = read_kernel_descriptor(bytes, kernel.descriptor_file_offset);
      if (!descriptor) {
        continue;
      }
      const DescriptorVgprAllocation vgprs = descriptor_vgpr_allocation(*descriptor, arch);
      stored.current_vgpr_count = vgprs.unified_count;
      if (vgprs.accumulator_base) {
        stored.ordinary_vgpr_limit = *vgprs.accumulator_base;
        stored.accumulator_boundary_is_allocation_tail =
            vgprs.unified_count <= *vgprs.accumulator_base;
        stored.can_move_empty_accumulator_boundary =
            stored.accumulator_boundary_is_allocation_tail || kernel.agpr_count == 0u;
      }
      stored.current_sgpr_count =
          static_cast<uint16_t>(descriptor_sgpr_allocation_count(*descriptor, arch));
      stored.initialized_sgpr_count = static_cast<uint16_t>(
          descriptor_user_sgpr_count(*descriptor) + descriptor_system_sgpr_count(*descriptor));
      stored.original_private_segment_size = descriptor->private_segment_fixed_size;
      stored.uses_dynamic_stack = kernel.uses_dynamic_stack;
      if (!stored.uses_dynamic_stack &&
          AMDHSA_BITS_GET(descriptor->compute_pgm_rsrc2,
                          kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT) == 0u) {
        stored.uses_dynamic_stack = false;
      }
      stored.descriptor_valid = true;
    }
    context_by_kernel_id.reserve(contexts.size());
    context_by_descriptor.reserve(contexts.size());
    for (KernelResourceContext &context : contexts) {
      context_by_kernel_id.try_emplace(context.kernel->id.ordinal, &context);
      context_by_descriptor.try_emplace(context.kernel->descriptor_file_offset, &context);
    }
    valid = true;
  }

  ResourcePlanningState(std::span<const uint8_t> input, rj_code_arch_t arch,
                        const ProgramInventory &program_inventory,
                        const ObservationPlan &observation_plan,
                        std::span<const CandidateResourcePlan> site_plans, const Request &request,
                        const BoundRuntimeResources &resources, const OperatingPoint &point)
      : ResourcePlanningState(ResourceProblem(input, arch, request, resources, program_inventory,
                                              observation_plan, {}),
                              point, site_plans) {}
};

struct CfgForwardDistanceIndex {
  const ResourcePlanningState *state = nullptr;
  BasicBlock *target = nullptr;
  uint64_t target_text_offset = 0;
  std::unordered_map<BasicBlock *, uint32_t> block_edges;
};

bool resource_reserve_range_if_uncovered(ResourcePlanningState &state,
                                         PreappliedReservedRange range) {
  if (range.size == 0u)
    return false;
  const bool covered =
      std::ranges::any_of(state.reserved_ranges, [&](const PreappliedReservedRange &reserved) {
        return reserved.size != 0u && range.text_offset >= reserved.text_offset &&
               range.text_offset - reserved.text_offset < reserved.size;
      });
  if (covered)
    return false;
  state.reserved_ranges.push_back(range);
  return true;
}

bool resource_is_valid(const ResourcePlanningState &state) { return state.valid; }

rj_code_arch_t resource_arch(const ResourcePlanningState &state) { return state.arch; }

bool resource_has_owner_context(const ResourcePlanningState &state, uint64_t descriptor_offset) {
  const auto found = state.context_by_descriptor.find(descriptor_offset);
  return found != state.context_by_descriptor.end() && found->second != nullptr;
}

void CfgForwardDistanceIndexDeleter::operator()(CfgForwardDistanceIndex *index) const {
  delete index;
}

CfgForwardDistanceIndexPtr make_cfg_forward_distance_index(const ResourcePlanningState &state,
                                                           uint64_t descriptor_offset,
                                                           uint64_t target_text_offset) {
  const auto context_it = state.context_by_descriptor.find(descriptor_offset);
  BasicBlock *target = block_for_offset(state.block_index, target_text_offset);
  if (context_it == state.context_by_descriptor.end() || context_it->second == nullptr ||
      target == nullptr) {
    return {};
  }
  const KernelResourceContext &context = *context_it->second;
  const std::unordered_set<BasicBlock *> scope(context.scope.blocks.begin(),
                                               context.scope.blocks.end());
  if (!scope.contains(target))
    return {};

  std::unordered_map<BasicBlock *, std::vector<BasicBlock *>> predecessors;
  predecessors.reserve(scope.size());
  for (BasicBlock *block : context.scope.blocks) {
    for (BasicBlock *successor : block->successors()) {
      if (scope.contains(successor))
        predecessors[successor].push_back(block);
    }
  }
  for (const ScopedCfgEdge &edge : context.scope.liveness_edges) {
    if (edge.from != nullptr && edge.to != nullptr && scope.contains(edge.from) &&
        scope.contains(edge.to)) {
      predecessors[edge.to].push_back(edge.from);
    }
  }

  CfgForwardDistanceIndexPtr index(new CfgForwardDistanceIndex);
  index->state = &state;
  index->target = target;
  index->target_text_offset = target_text_offset;
  index->block_edges.reserve(scope.size());
  std::vector<BasicBlock *> pending{target};
  index->block_edges.emplace(target, 0u);
  for (size_t cursor = 0; cursor < pending.size(); ++cursor) {
    BasicBlock *block = pending[cursor];
    const uint32_t next_distance = index->block_edges.at(block) + 1u;
    auto found = predecessors.find(block);
    if (found == predecessors.end())
      continue;
    std::ranges::sort(found->second);
    found->second.erase(std::ranges::unique(found->second).begin(), found->second.end());
    for (BasicBlock *predecessor : found->second) {
      if (index->block_edges.emplace(predecessor, next_distance).second)
        pending.push_back(predecessor);
    }
  }
  return index;
}

std::optional<CfgDistance> cfg_forward_distance(const CfgForwardDistanceIndex &index,
                                                uint64_t from_text_offset) {
  if (index.state == nullptr)
    return std::nullopt;
  BasicBlock *from = block_for_offset(index.state->block_index, from_text_offset);
  if (from == nullptr || (from == index.target && from_text_offset > index.target_text_offset))
    return std::nullopt;
  const auto found = index.block_edges.find(from);
  if (found == index.block_edges.end())
    return std::nullopt;
  const uint64_t text_distance = from_text_offset < index.target_text_offset
                                     ? index.target_text_offset - from_text_offset
                                     : from_text_offset - index.target_text_offset;
  return CfgDistance{found->second, text_distance};
}

void ResourcePlanningStateDeleter::operator()(ResourcePlanningState *state) const { delete state; }

ResourcePlanningStatePtr
make_resource_planning_state(const ResourceProblem &problem, const OperatingPoint &allocation,
                             std::span<const CandidateResourcePlan> prior_site_plans) {
  return ResourcePlanningStatePtr(new ResourcePlanningState(problem, allocation, prior_site_plans));
}

ResourcePlanningStatePtr make_resource_planning_state(
    std::span<const uint8_t> input, rj_code_arch_t arch, const ProgramInventory &program_inventory,
    const ObservationPlan &observation_plan, std::span<const CandidateResourcePlan> site_plans,
    const Request &request, const BoundRuntimeResources &resources, const OperatingPoint &point) {
  return ResourcePlanningStatePtr(new ResourcePlanningState(
      input, arch, program_inventory, observation_plan, site_plans, request, resources, point));
}

/// Conservatively summarize every instruction in a code object's declared
/// function ranges when those ranges cover the complete text section except
/// for verified compiler/linker padding.
///
/// This is the fail-safe fallback for generated kernels whose local-label
/// calls cannot be reconstructed by CFG analysis. Scanning unrelated kernels
/// can only make a hole unavailable. An unresolved target is treated as
/// represented by the summary only when every byte outside a declared
/// function is an aligned zero or s_nop word, so an uncovered instruction that
/// could clobber the selected state still fails closed.
[[nodiscard]] std::optional<OwnerRegisterReferenceSummary>
summarize_complete_text_register_references(const ResourcePlanningState &state) {
  // Function offsets are currently normalized against the code object's sole
  // executable text span. Multi-text objects need section-aware function
  // provenance before their ranges can be tiled independently, so retain the
  // conservative fail-closed boundary here.
  if (!state.valid || state.code_object.text_sections().size() != 1u)
    return std::nullopt;

  const Section &text_section = *state.code_object.text_sections().front();
  const uint64_t text_size = text_section.size();
  const std::span<const uint8_t> text(reinterpret_cast<const uint8_t *>(text_section.data()),
                                      text_section.size());
  const uint32_t padding_nop = build_s_nop(0, state.arch);
  const auto is_padding = [&](uint64_t begin, uint64_t end) {
    if (begin > end || end > text.size() || begin % sizeof(uint32_t) != 0u ||
        end % sizeof(uint32_t) != 0u) {
      return false;
    }
    for (uint64_t offset = begin; offset < end; offset += sizeof(uint32_t)) {
      uint32_t word = 0u;
      std::memcpy(&word, text.data() + offset, sizeof(word));
      if (word != 0u && word != padding_nop)
        return false;
    }
    return true;
  };
  std::vector<std::pair<uint64_t, uint64_t>> ranges;
  ranges.reserve(state.code_object.functions().size());
  for (const AmdGpuFunctionInfo &function : state.code_object.functions()) {
    if (function.code_size == 0u || function.entry_text_offset > text_size ||
        function.code_size > text_size - function.entry_text_offset) {
      return std::nullopt;
    }
    ranges.emplace_back(function.entry_text_offset,
                        function.entry_text_offset + function.code_size);
  }
  std::ranges::sort(ranges);
  uint64_t covered_end = 0u;
  for (const auto &[begin, end] : ranges) {
    if (begin > covered_end && !is_padding(covered_end, begin))
      return std::nullopt;
    covered_end = std::max(covered_end, end);
  }
  if (covered_end > text_size || !is_padding(covered_end, text_size))
    return std::nullopt;

  OwnerRegisterReferenceSummary result;
  for (const std::unique_ptr<BasicBlock> &block : state.blocks) {
    if (!block)
      continue;
    for (const Instruction &instruction : block->instructions()) {
      if (is_indirect_sgpr_access(instruction))
        result.uses_indirect_sgprs = true;
      if (is_indirect_vgpr_access(instruction))
        result.uses_indirect_vgprs = true;
      const InstDefUse def_use(instruction);
      result.references |= def_use.defs | def_use.uses;
    }
  }
  return result;
}

[[nodiscard]] bool resource_plan_can_be_recovered_by_persistent_placement(
    const CandidateResourcePlan &plan, const DebugOverrides &debug, rj_code_arch_t arch) {
  if (plan.source != RegisterAllocationSource::Unsupported)
    return false;
  if (scalar_placement_model(arch) != ScalarPlacementModel::DescriptorPartitioned) {
    return false;
  }
  // Persistent placement and resource planning form a fixed point for the
  // CDNA3/4 stateful ABIs. An owner-local persistent assignment or a
  // component-local transient assignment can recover an automatic window that
  // was rejected against provisional code-object-wide state. Conflicts against
  // persistent and transient state become owner-local. Explicit scratch conflicts and
  // structural failures cannot be repaired here and must not make an unrelated
  // owner appear admitted.
  return !plan.owner_kernel_ids.empty() &&
         (plan.reason == RegisterPlanReason::NoLegalWindow ||
          (plan.reason == RegisterPlanReason::ForbiddenOverlap && !debug.scratch_vgpr));
}

[[nodiscard]] std::vector<ProgramContainerId>
scalar_owner_kernel_ids(std::span<const CandidateResourcePlan> site_plans) {
  std::vector<ProgramContainerId> owners;
  for (const CandidateResourcePlan &plan : site_plans) {
    // Scalar persistent state is code-object-wide and must be proven against
    // every semantically admitted owner, including a site whose provisional
    // resource plan failed. Persistent placement can change that site's ABI
    // and recover it on the subsequent resource-plan rebuild; omitting it here
    // would instead select SGPRs that clobber the very owner being recovered.
    for (ProgramContainerId owner : plan.owner_kernel_ids) {
      if (std::ranges::find(owners, owner) == owners.end())
        owners.push_back(owner);
    }
  }
  return owners;
}

/// Partition resource-plan owners into the connected components that must
/// share one scalar ABI. A shared helper joins all kernels that can reach it;
/// unrelated kernels remain independently placeable.
[[nodiscard]] std::vector<std::vector<ProgramContainerId>>
scalar_owner_components(std::span<const CandidateResourcePlan> plans) {
  std::vector<std::vector<ProgramContainerId>> components;
  for (const CandidateResourcePlan &plan : plans) {
    if (plan.owner_kernel_ids.empty())
      continue;
    std::optional<size_t> destination;
    for (size_t index = 0; index < components.size();) {
      const bool intersects =
          std::ranges::any_of(plan.owner_kernel_ids, [&](ProgramContainerId owner) {
            return std::ranges::find(components[index], owner) != components[index].end();
          });
      if (!intersects) {
        ++index;
        continue;
      }
      if (!destination) {
        destination = index++;
        continue;
      }
      for (ProgramContainerId owner : components[index]) {
        if (std::ranges::find(components[*destination], owner) == components[*destination].end())
          components[*destination].push_back(owner);
      }
      components.erase(components.begin() + static_cast<ptrdiff_t>(index));
    }
    if (!destination) {
      destination = components.size();
      components.emplace_back();
    }
    for (ProgramContainerId owner : plan.owner_kernel_ids) {
      if (std::ranges::find(components[*destination], owner) == components[*destination].end())
        components[*destination].push_back(owner);
    }
  }
  return components;
}

struct ScalarOwnerContexts {
  std::vector<const KernelResourceContext *> contexts;
  std::vector<detail::ScalarOwnerContextSummary> summaries;
  uint32_t tail_floor = 0;
  uint32_t initialized_sgpr_count = 0;
  bool all_use_dynamic_stack = true;
  bool all_use_fixed_stack = true;
};

[[nodiscard]] std::optional<ScalarOwnerContexts>
resolve_scalar_owner_contexts(const ResourcePlanningState &state,
                              std::span<const uint64_t> owners) {
  std::vector<detail::ScalarOwnerContextSummary> summaries;
  summaries.reserve(state.contexts.size());
  for (const KernelResourceContext &context : state.contexts)
    summaries.push_back(context.scalar_summary());
  const auto resolved = detail::resolve_scalar_owner_contexts(state.valid, summaries, owners);
  if (!resolved)
    return std::nullopt;

  ScalarOwnerContexts result;
  result.contexts.reserve(resolved->context_indices.size());
  result.summaries.reserve(resolved->context_indices.size());
  for (size_t context_index : resolved->context_indices) {
    result.contexts.push_back(&state.contexts[context_index]);
    result.summaries.push_back(summaries[context_index]);
    const KernelResourceContext &context = state.contexts[context_index];
    result.initialized_sgpr_count =
        std::max<uint32_t>(result.initialized_sgpr_count, context.initialized_sgpr_count);
    result.all_use_dynamic_stack &= context.uses_dynamic_stack.value_or(false);
    result.all_use_fixed_stack &=
        context.uses_dynamic_stack.has_value() && !*context.uses_dynamic_stack;
  }
  result.tail_floor = resolved->tail_floor;
  return result;
}

[[nodiscard]] std::optional<ScalarOwnerContexts>
resolve_scalar_owner_contexts(const ResourcePlanningState &state,
                              std::span<const ProgramContainerId> owners) {
  if (!state.valid)
    return std::nullopt;
  ScalarOwnerContexts result;
  result.contexts.reserve(owners.size());
  result.summaries.reserve(owners.size());
  for (ProgramContainerId owner : owners) {
    const auto found = state.context_by_kernel_id.find(owner.ordinal);
    if (!owner.valid() || found == state.context_by_kernel_id.end() || found->second == nullptr ||
        !found->second->descriptor_valid) {
      return std::nullopt;
    }
    const KernelResourceContext &context = *found->second;
    result.contexts.push_back(&context);
    result.summaries.push_back(context.scalar_summary());
    result.tail_floor = std::max<uint32_t>(
        result.tail_floor, detail::scalar_owner_tail_floor(result.summaries.back()));
    result.initialized_sgpr_count =
        std::max<uint32_t>(result.initialized_sgpr_count, context.initialized_sgpr_count);
    result.all_use_dynamic_stack &= context.uses_dynamic_stack.value_or(false);
    result.all_use_fixed_stack &=
        context.uses_dynamic_stack.has_value() && !*context.uses_dynamic_stack;
  }
  return result;
}

bool resource_owner_sgpr_window_admitted(const ResourcePlanningState &state,
                                         std::span<const ProgramContainerId> owners, uint16_t base,
                                         uint16_t count, uint32_t required_sgpr_count_floor) {
  const auto contexts = resolve_scalar_owner_contexts(state, owners);
  if (!contexts || !std::ranges::all_of(contexts->summaries, [&](const auto &context) {
        return detail::scalar_owner_tail_floor(context) <= base;
      })) {
    return false;
  }
  if (scalar_placement_model(state.arch) != ScalarPlacementModel::DescriptorPartitioned) {
    return true;
  }
  const std::array ranges{detail::ScalarOwnerSgprRange{base, count}};
  return !detail::scalar_owner_contexts_conflict_with_physical_vcc(contexts->summaries, ranges,
                                                                   required_sgpr_count_floor);
}

struct ScalarOwnerContextGroup {
  bool supported = false;
  ScalarOwnerContexts owners;
};

struct ScalarOwnerContextGroupsResolution {
  bool planning_state_valid = false;
  std::vector<ScalarOwnerContextGroup> groups;
  std::optional<ProgramContainerId> unresolved_owner;
};

[[nodiscard]] ScalarOwnerContextGroupsResolution
resolve_resource_plan_scalar_owner_contexts(const ResourcePlanningState &state,
                                            std::span<const CandidateResourcePlan> plans) {
  ScalarOwnerContextGroupsResolution resolved;
  resolved.planning_state_valid = state.valid;
  if (!state.valid)
    return resolved;
  resolved.groups.reserve(plans.size());
  for (const CandidateResourcePlan &plan : plans) {
    if (plan.owner_kernel_ids.empty())
      continue;
    auto owners = resolve_scalar_owner_contexts(state, plan.owner_kernel_ids);
    if (!owners) {
      for (ProgramContainerId owner : plan.owner_kernel_ids) {
        const auto found = state.context_by_kernel_id.find(owner.ordinal);
        if (!owner.valid() || found == state.context_by_kernel_id.end() ||
            found->second == nullptr || !found->second->descriptor_valid) {
          resolved.unresolved_owner = owner;
          break;
        }
      }
      return resolved;
    }
    resolved.groups.push_back({
        .supported = plan.source != RegisterAllocationSource::Unsupported,
        .owners = std::move(*owners),
    });
  }
  return resolved;
}

[[nodiscard]] ScalarOwnerContexts
collect_scalar_owner_contexts(std::span<const ScalarOwnerContextGroup> groups,
                              bool include_unsupported) {
  ScalarOwnerContexts result;
  for (const ScalarOwnerContextGroup &group : groups) {
    if (!include_unsupported && !group.supported)
      continue;
    for (size_t index = 0; index < group.owners.contexts.size(); ++index) {
      const KernelResourceContext *context = group.owners.contexts[index];
      if (std::ranges::find(result.contexts, context) != result.contexts.end())
        continue;
      result.contexts.push_back(context);
      result.summaries.push_back(group.owners.summaries[index]);
      result.tail_floor = std::max<uint32_t>(result.tail_floor, group.owners.tail_floor);
      result.initialized_sgpr_count =
          std::max(result.initialized_sgpr_count, group.owners.initialized_sgpr_count);
      result.all_use_dynamic_stack &= group.owners.all_use_dynamic_stack;
      result.all_use_fixed_stack &= group.owners.all_use_fixed_stack;
    }
  }
  return result;
}

[[nodiscard]] bool scalar_contexts_conflict_with_physical_vcc(
    std::span<const detail::ScalarOwnerContextSummary> contexts, std::span<const SgprRange> ranges,
    rj_code_arch_t arch) {
  return scalar_placement_model(arch) == ScalarPlacementModel::DescriptorPartitioned &&
         detail::scalar_owner_contexts_conflict_with_physical_vcc(contexts, ranges);
}

[[nodiscard]] bool
scalar_contexts_admit_reserved_window(std::span<const detail::ScalarOwnerContextSummary> contexts,
                                      uint16_t base, uint16_t width, rj_code_arch_t arch) {
  return detail::scalar_owner_contexts_admit_reserved_window(
      contexts, base, width,
      scalar_placement_model(arch) == ScalarPlacementModel::DescriptorPartitioned);
}

/// One register-file view for an ordered scalar-placement attempt.
///
/// Fixed occupants exclude overlap but do not participate in the CDNA
/// physical-VCC proof: they were qualified by their own lifetime. Selected
/// ranges form one simultaneously live tuple and are checked together. The
/// optional unavailable set is the site-local liveness/def-use union; callers
/// explicitly choose whether a role must be dead or is preserved by spill.
class ScalarPlacementDomain {
public:
  ScalarPlacementDomain(rj_code_arch_t arch, uint16_t limit,
                        std::span<const detail::ScalarOwnerContextSummary> owner_contexts,
                        const RegisterSet *unavailable = nullptr)
      : arch_(arch), limit_(limit), owner_contexts_(owner_contexts), unavailable_(unavailable) {}

  void occupy(uint16_t base, uint16_t width) {
    if (width != 0u) {
      occupied_.push_back({base, width});
      required_sgpr_count_floor_ =
          std::max<uint32_t>(required_sgpr_count_floor_, static_cast<uint32_t>(base) + width);
    }
  }

  void occupy(std::optional<uint16_t> base, uint16_t width) {
    if (base)
      occupy(*base, width);
  }

  void occupy_dynamic_stack_state() {
    occupy(kDynamicStackTopSgpr, 1u);
    occupy(kDynamicFrameBaseSgpr, 1u);
  }

  [[nodiscard]] bool legal_range(uint16_t base, uint16_t width) const {
    return static_cast<uint32_t>(base) + width <= limit_ &&
           !persistent_sgpr_range_overlaps_reserved_ordinary_range(base, width, arch_);
  }

  [[nodiscard]] bool can_select(uint16_t base, uint16_t width, bool require_dead = true) const {
    if (width == 0u || !legal_range(base, width)) {
      return false;
    }
    if (std::ranges::any_of(occupied_, [&](const SgprRange &range) {
          return range_overlaps(base, width, range.base, range.width);
        })) {
      return false;
    }
    if (require_dead && unavailable_ != nullptr) {
      for (uint16_t index = 0u; index < width; ++index) {
        if (unavailable_->contains({RegClass::SGPR, static_cast<uint16_t>(base + index), 1u})) {
          return false;
        }
      }
    }
    std::vector<detail::ScalarOwnerSgprRange> combined = selected_;
    combined.push_back({base, width});
    return owner_contexts_.empty() || !detail::scalar_owner_contexts_conflict_with_physical_vcc(
                                          owner_contexts_, combined, required_sgpr_count_floor_);
  }

  [[nodiscard]] std::optional<uint16_t>
  select_first(uint32_t begin, uint16_t width, uint16_t alignment = 1u, bool require_dead = true) {
    const auto selected =
        first_admitted_register_range(begin, width, limit_, alignment, [&](uint16_t base) {
          return can_select(base, width, require_dead);
        });
    if (selected)
      (void)select(*selected, width, require_dead);
    return selected;
  }

  [[nodiscard]] std::optional<uint16_t>
  select_last(uint32_t begin, uint16_t width, uint16_t alignment = 1u, bool require_dead = true) {
    if (width > limit_)
      return std::nullopt;
    const int32_t first = static_cast<int32_t>(util::align_up(begin, uint32_t{alignment}));
    for (int32_t base = static_cast<int32_t>((limit_ - width) / alignment * alignment);
         base >= first; base -= alignment) {
      if (select(static_cast<uint16_t>(base), width, require_dead))
        return static_cast<uint16_t>(base);
    }
    return std::nullopt;
  }

  [[nodiscard]] bool select(uint16_t base, uint16_t width, bool require_dead = true) {
    if (!can_select(base, width, require_dead))
      return false;
    occupied_.push_back({base, width});
    selected_.push_back({base, width});
    return true;
  }

private:
  rj_code_arch_t arch_;
  uint16_t limit_ = 0u;
  std::span<const detail::ScalarOwnerContextSummary> owner_contexts_;
  const RegisterSet *unavailable_ = nullptr;
  std::vector<SgprRange> occupied_;
  std::vector<detail::ScalarOwnerSgprRange> selected_;
  // Fixed instrumentation state has already been qualified independently,
  // but it still contributes to descriptor growth and therefore to the
  // relocated CDNA physical-VCC position used to qualify later selections.
  uint32_t required_sgpr_count_floor_ = 0u;
};

void occupy_transient_scalar_state(ScalarPlacementDomain &domain, const Request &request,
                                   const BoundRuntimeResources &resources,
                                   const OperatingPoint &point, const OperatingPoint &allocation,
                                   rj_code_arch_t arch) {
  const uint16_t exec_width =
      exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, point), arch);
  domain.occupy(point.exec_save_sgpr, exec_width);
  domain.occupy(point.owner_sgpr.base(), 1u);
  domain.occupy(point.dispatch_sgpr.base(), kDispatchStateSgprCount);
  point.for_each_auxiliary_sgpr_range(
      [&](std::optional<uint16_t> base, uint16_t width) { domain.occupy(base, width); });
  for (const TransientSgprAssignment &assignment : allocation.owner_transient_sgprs) {
    domain.occupy(assignment.exec_save_sgpr, exec_width);
    domain.occupy(assignment.owner_sgpr, 1u);
    domain.occupy(assignment.dispatch_id_sgpr, kDispatchStateSgprCount);
    if (assignment.scalar_spill_setup) {
      assignment.scalar_spill_setup->for_each_sgpr_range(
          [&](uint16_t base, uint16_t width) { domain.occupy(base, width); });
    }
    if (assignment.branch_only_spill)
      domain.occupy(assignment.branch_only_spill->dynamic_stack_borrowed_sgpr, 2u);
  }
}

[[nodiscard]] uint32_t required_sgpr_count_floor(const Request &request,
                                                 const BoundRuntimeResources &resources,
                                                 const OperatingPoint &point, rj_code_arch_t arch) {
  uint32_t floor = 0u;
  const auto include = [&](std::optional<uint16_t> base, uint16_t width) {
    if (base)
      floor = std::max<uint32_t>(floor, static_cast<uint32_t>(*base) + width);
  };
  const uint16_t exec_width =
      exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, point), arch);
  include(point.exec_save_sgpr, exec_width);
  include(point.owner_sgpr.base(), 1u);
  include(point.dispatch_sgpr.base(), kDispatchStateSgprCount);
  point.for_each_auxiliary_sgpr_range(include);
  for (const TransientSgprAssignment &assignment : point.owner_transient_sgprs) {
    include(assignment.exec_save_sgpr, exec_width);
    include(assignment.owner_sgpr, 1u);
    include(assignment.dispatch_id_sgpr, kDispatchStateSgprCount);
    if (assignment.scalar_spill_setup) {
      assignment.scalar_spill_setup->for_each_sgpr_range(
          [&](uint16_t base, uint16_t width) { include(base, width); });
    }
    if (assignment.branch_only_spill)
      include(assignment.branch_only_spill->dynamic_stack_borrowed_sgpr, 2u);
  }
  return floor;
}

struct ScalarSiteAvailability {
  std::unordered_map<uint32_t, RegisterSet> unavailable_by_kernel_id;
  RegisterSet unavailable;
  bool saw_site = false;
};

[[nodiscard]] RegisterSet dynamic_frame_save_registers_for_site(const ResourcePlanningState &state,
                                                                uint64_t text_offset);

[[nodiscard]] ScalarSiteAvailability
collect_scalar_site_availability(const ResourcePlanningState &state,
                                 std::span<const CandidateResourcePlan> site_plans,
                                 const ObservationPlan &observation_plan,
                                 const ProgramInventory &inventory, bool include_kernel_entries) {
  ScalarSiteAvailability result;
  std::unordered_map<BasicBlock *, std::vector<const KernelResourceContext *>> owners_by_block;
  for (const KernelResourceContext &context : state.contexts) {
    for (BasicBlock *block : context.scope.blocks) {
      if (block != nullptr)
        owners_by_block[block].push_back(&context);
    }
  }

  std::unordered_set<uint64_t> included_offsets;
  included_offsets.reserve(site_plans.size() + observation_plan.probe_intents.size() +
                           inventory.kernels().size());
  const auto include = [&](uint64_t text_offset) {
    if (!included_offsets.insert(text_offset).second)
      return;
    BasicBlock *block = block_for_offset(state.block_index, text_offset);
    const Instruction *instruction = instruction_at(block, text_offset);
    if (block == nullptr || instruction == nullptr)
      return;
    const auto owners = owners_by_block.find(block);
    if (owners == owners_by_block.end())
      return;
    result.saw_site = true;
    for (const KernelResourceContext *context : owners->second) {
      RegisterSet site_unavailable = context->liveness->live_before(*instruction);
      const InstDefUse def_use(*instruction);
      site_unavailable |= def_use.defs | def_use.uses;
      if (context->kernel != nullptr && context->kernel->uses_dynamic_stack.value_or(false))
        site_unavailable |= dynamic_frame_save_registers_for_site(state, text_offset);
      result.unavailable |= site_unavailable;
      if (context->kernel != nullptr)
        result.unavailable_by_kernel_id[context->kernel->id.ordinal] |= site_unavailable;
    }
  };
  for (const CandidateResourcePlan &plan : site_plans)
    include(plan.text_offset);
  for (const ProbeIntent &intent : observation_plan.probe_intents)
    include(intent.physical_site.original_text_offset);
  if (include_kernel_entries) {
    for (const ProgramContainer &kernel : inventory.kernels())
      include(kernel.entry_text_offset);
  }
  return result;
}

[[nodiscard]] std::optional<std::vector<PrologueScratchVgprAssignment>>
plan_scalar_prologue_scratch(const ResourcePlanningState &state,
                             const ScalarOwnerContexts &resolution, bool needs_entry_vgpr_pair) {
  std::vector<PrologueScratchVgprAssignment> assignments;
  assignments.reserve(resolution.contexts.size());
  for (const KernelResourceContext *context : resolution.contexts) {
    if (!needs_entry_vgpr_pair) {
      assignments.push_back({context->kernel->descriptor_file_offset, 0u});
      continue;
    }
    BasicBlock *entry_block =
        block_for_offset(state.block_index, context->kernel->entry_text_offset);
    const Instruction *entry = instruction_at(entry_block, context->kernel->entry_text_offset);
    if (entry_block == nullptr || entry == nullptr)
      return std::nullopt;
    RegisterSet unavailable_vgprs = context->liveness->live_before(*entry);
    const InstDefUse entry_def_use(*entry);
    unavailable_vgprs |= entry_def_use.defs | entry_def_use.uses;
    // The same trusted zero-AGPR metadata that permits site-local descriptor
    // growth also permits an entry-only scalar-transfer pair above the
    // compiler's rounded ordinary boundary. Descriptor mutation repeats that
    // proof before moving ACCUM_OFFSET. Without it, remain below the original
    // boundary even though the pair is dead before the first guest instruction.
    const uint16_t scratch_limit =
        context->can_move_empty_accumulator_boundary ? kMaxVgprs : context->ordinary_vgpr_limit;
    std::optional<uint16_t> scratch;
    for (uint16_t candidate = 0; candidate + 2u <= scratch_limit; ++candidate) {
      if (!unavailable_vgprs.contains({RegClass::VGPR, candidate, 1u}) &&
          !unavailable_vgprs.contains(
              {RegClass::VGPR, static_cast<uint16_t>(candidate + 1u), 1u})) {
        scratch = candidate;
        break;
      }
    }
    if (!scratch)
      return std::nullopt;
    assignments.push_back({context->kernel->descriptor_file_offset, *scratch});
  }
  return assignments;
}

[[nodiscard]] const CandidateResourcePlan *
resource_plan_for_candidate(std::span<const CandidateResourcePlan> plans,
                            const Candidate &candidate);

/// Returns the merged original-text ranges occupied by synchronization
/// instructions that access-mode relocation must preserve.
///
/// Barrier, atomic, and fence inventories are deliberately projected through
/// one query so ConSan probes cannot silently
/// disagree about which synchronization instructions are unavailable as
/// relocation donors. Empty sites are ignored and adjacent ranges are merged;
/// consumers may therefore also use the result for logarithmic overlap tests.

[[nodiscard]] RegisterSet dynamic_frame_save_registers_for_site(const ResourcePlanningState &state,
                                                                uint64_t text_offset) {
  if (const auto cached = state.dynamic_frame_saves_by_site.find(text_offset);
      cached != state.dynamic_frame_saves_by_site.end()) {
    return cached->second;
  }
  RegisterSet saved;
  constexpr uint64_t kMaxDynamicFramePrologueDistance = 16u * 1024u;
  CodeObjectPatcher patcher(state.code_object);
  const std::span<const uint8_t> text = patcher.text_bytes();
  if (text_offset >= sizeof(uint32_t) && text_offset <= text.size()) {
    const uint64_t begin = text_offset > kMaxDynamicFramePrologueDistance
                               ? text_offset - kMaxDynamicFramePrologueDistance
                               : 0u;
    for (uint64_t offset = text_offset - sizeof(uint32_t);; offset -= sizeof(uint32_t)) {
      if (offset + 2u * sizeof(uint32_t) <= text.size()) {
        uint32_t save_word = 0;
        uint32_t establish_word = 0;
        std::memcpy(&save_word, text.data() + offset, sizeof(save_word));
        std::memcpy(&establish_word, text.data() + offset + sizeof(uint32_t),
                    sizeof(establish_word));
        if ((save_word & 0xff800000u) == 0xbe800000u && (save_word & 0xffu) == 33u &&
            establish_word == build_s_mov_b32(/*dst=*/33u, /*src=*/32u, state.arch)) {
          saved.expand({RegClass::SGPR, static_cast<uint16_t>((save_word >> 16u) & 0x7fu), 1u});
          break;
        }
      }
      if (offset < begin + sizeof(uint32_t))
        break;
    }
  }
  return state.dynamic_frame_saves_by_site.emplace(text_offset, std::move(saved)).first->second;
}

void forbid_persistent_vgprs(RegisterSet &forbidden, const OperatingPoint &point,
                             const OperatingPoint &allocation,
                             std::span<const uint64_t> owner_descriptors) {
  const auto forbid_vgpr = [&](std::optional<uint16_t> reg, uint16_t count) {
    if (reg)
      forbidden.expand({RegClass::VGPR, *reg, static_cast<uint8_t>(count)});
  };
  const auto forbid_state = [&](const PersistentVgprStateView &state) {
    state.for_each_range(forbid_vgpr);
  };
  forbid_state(persistent_vgpr_state_view(point));
  for (uint64_t descriptor_offset : owner_descriptors) {
    const auto *assignment = persistent_vgpr_assignment(allocation, descriptor_offset);
    if (assignment)
      forbid_state(persistent_vgpr_state_view(*assignment));
  }
}

[[nodiscard]] CandidateResourcePlan plan_resource_site(
    ResourcePlanningState &state, const Request &semantic_request, const DebugOverrides &debug,
    const OperatingPoint &point, ResourceSiteKind site_kind, size_t candidate_index,
    uint64_t text_offset, std::optional<ProgramSiteId> owner_site, uint16_t scratch_count,
    const Candidate *access_candidate, const AtomicLoweringForm *atomic_form, bool require_spill) {
  CandidateResourcePlan plan;
  plan.site_kind = site_kind;
  plan.candidate_index = candidate_index;
  plan.text_offset = text_offset;
  if (access_candidate != nullptr)
    plan.intent_ids = access_candidate->intent_ids;
  plan.scratch_vgpr_count = scratch_count;
  if (!state.valid) {
    plan.reason = RegisterPlanReason::MissingInstruction;
    return plan;
  }

  BasicBlock *anchor_block = block_for_offset(state.block_index, text_offset);
  const Instruction *anchor = instruction_at(anchor_block, text_offset);
  if (anchor == nullptr) {
    plan.reason = RegisterPlanReason::MissingInstruction;
    return plan;
  }

  std::vector<KernelResourceContext *> owners;
  const auto add_owner = [&](KernelResourceContext &context) {
    if (!context.block_set.contains(anchor_block))
      return;
    owners.push_back(&context);
    plan.owner_kernel_ids.push_back(context.kernel->id);
  };
  const auto add_execution_owners = [&](std::span<const ExecutionOwner> execution_owners) {
    for (const ExecutionOwner &execution_owner : execution_owners) {
      const auto owner = state.context_by_kernel_id.find(execution_owner.kernel.ordinal);
      if (owner != state.context_by_kernel_id.end())
        add_owner(*owner->second);
    }
  };
  if (access_candidate != nullptr) {
    add_execution_owners(access_candidate->site().execution_owners);
  } else if (owner_site) {
    const ProgramSite *site = state.inventory.program_site(*owner_site);
    if (site != nullptr)
      add_execution_owners(site->execution_owners);
  } else {
    for (KernelResourceContext &context : state.contexts)
      add_owner(context);
  }
  if (owners.empty()) {
    plan.reason = RegisterPlanReason::MissingOwner;
    return plan;
  }
  std::vector<uint64_t> owner_descriptor_offsets;
  owner_descriptor_offsets.reserve(owners.size());
  for (const KernelResourceContext *owner : owners)
    owner_descriptor_offsets.push_back(owner->kernel->descriptor_file_offset);
  const bool owners_have_spill_backed_scalar_state =
      !owner_descriptor_offsets.empty() &&
      std::ranges::all_of(owner_descriptor_offsets, [&](uint64_t descriptor_offset) {
        const TransientSgprAssignment *assignment =
            transient_sgpr_assignment(point, descriptor_offset);
        return assignment != nullptr && assignment->spill_backed;
      });
  const bool fixed_lane_scalar_reservoir =
      ((point.exec_save_sgpr && point.has_scalar_spill()) ||
       owners_have_spill_backed_scalar_state) &&
      std::ranges::none_of(owners, [](const KernelResourceContext *owner) {
        return owner->kernel->uses_dynamic_stack.value_or(false);
      });
  if (fixed_lane_scalar_reservoir && scratch_count == REGISTER_SET_MAX_VGPRS) {
    plan.reason = RegisterPlanReason::NoLegalWindow;
    return plan;
  }
  // Fixed-stack spill-backed scalar state lives in one VGPR that the mode
  // never uses as scratch. Include it in allocation itself so liveness,
  // persistent-state exclusions, guest operands, and descriptor growth all
  // cover the real clobber range rather than discovering an adjacent register
  // after planning.
  const uint16_t allocation_count =
      static_cast<uint16_t>(scratch_count + static_cast<uint16_t>(fixed_lane_scalar_reservoir));
  plan.current_vgpr_count = kMaxVgprs;
  plan.ordinary_vgpr_limit = kMaxVgprs;
  plan.current_sgpr_count = kMaxSgprs;
  const TargetProfile *target = target_profile(state.arch);
  bool every_owner_can_move_empty_accumulator_boundary =
      target && target->accumulator_model == AccumulatorModel::DescriptorPartitioned;
  bool every_owner_accumulator_boundary_is_allocation_tail =
      every_owner_can_move_empty_accumulator_boundary;
  RegisterSet live_before_all_owners;
  bool descriptors_valid = true;
  for (KernelResourceContext *owner : owners) {
    plan.current_vgpr_count = std::min(plan.current_vgpr_count, owner->current_vgpr_count);
    plan.ordinary_vgpr_limit = std::min(plan.ordinary_vgpr_limit, owner->ordinary_vgpr_limit);
    every_owner_accumulator_boundary_is_allocation_tail &=
        owner->accumulator_boundary_is_allocation_tail;
    every_owner_can_move_empty_accumulator_boundary &= owner->can_move_empty_accumulator_boundary;
    plan.max_referenced_vgpr_count =
        std::max(plan.max_referenced_vgpr_count, owner->max_referenced_vgpr_count);
    plan.current_sgpr_count = std::min(plan.current_sgpr_count, owner->current_sgpr_count);
    plan.max_referenced_sgpr_count =
        std::max(plan.max_referenced_sgpr_count, owner->max_referenced_sgpr_count);
    plan.original_private_segment_size =
        std::max(plan.original_private_segment_size, owner->original_private_segment_size);
    live_before_all_owners |= owner->liveness->live_before(*anchor);
    // Relative VGPR instructions address a register array through M0. Their
    // dynamically selected elements are intentionally absent from ordinary
    // explicit-operand def/use sets, so a normal liveness proof can otherwise
    // misclassify a live array element as instrumentation scratch. Conservatively
    // reserve the whole guest allocation; the planner can grow the descriptor or
    // use the existing spill path instead of clobbering indirect guest state.
    if (owner->has_indirect_vgpr_access) {
      plan.has_indirect_vgpr_access = true;
      for (uint16_t index = 0; index < owner->current_vgpr_count; ++index)
        live_before_all_owners.expand({RegClass::VGPR, index, 1});
    }
    descriptors_valid &= owner->descriptor_valid;
  }
  plan.required_vgpr_count = plan.current_vgpr_count;
  if (!descriptors_valid) {
    plan.reason = RegisterPlanReason::InvalidDescriptor;
    return plan;
  }

  const InstDefUse anchor_def_use(*anchor);
  RegisterRequest request;
  request.reg_class = RegClass::VGPR;
  request.count = allocation_count;
  // ConSan probes use the first two scratch registers as a FLAT
  // address pair. The target placement contract owns its tuple alignment.
  request.alignment = target ? target->placement.scratch_vgpr_alignment : 1u;
  request.current_allocation_count = std::min(plan.current_vgpr_count, plan.ordinary_vgpr_limit);
  request.max_referenced_count = plan.max_referenced_vgpr_count;
  request.architecture_limit =
      every_owner_accumulator_boundary_is_allocation_tail ? kMaxVgprs : plan.ordinary_vgpr_limit;
  request.explicit_base = debug.scratch_vgpr;
  request.force_spill = debug.test_force_vgpr_spill || require_spill;
  request.forbidden = anchor_def_use.defs | anchor_def_use.uses;
  // Candidate decoding is the authoritative operand-width source for native
  // two-address LDS operations.  Some generic instruction def/use models only
  // expose the first destination register, so supplement them before choosing
  // a liveness-dead window rather than rejecting the chosen window later.
  if (access_candidate != nullptr) {
    const AccessLoweringForm *form =
        access_candidate->site().lowering.form ? &*access_candidate->site().lowering.form : nullptr;
    if (form != nullptr) {
      if (form->address_vgpr && form->address_vgpr_count != 0u) {
        request.forbidden.expand(
            {RegClass::VGPR, *form->address_vgpr, static_cast<uint8_t>(form->address_vgpr_count)});
      }
      if (form->destination_vgpr && form->destination_register_count != 0u) {
        request.forbidden.expand({RegClass::VGPR, *form->destination_vgpr,
                                  static_cast<uint8_t>(form->destination_register_count)});
      }
      const uint8_t data_count = static_cast<uint8_t>(
          form->second_data_vgpr ? form->element_register_count : form->data_register_count);
      if (form->data_vgpr)
        request.forbidden.expand({RegClass::VGPR, *form->data_vgpr, data_count});
      if (form->second_data_vgpr) {
        request.forbidden.expand({RegClass::VGPR, *form->second_data_vgpr, data_count});
      }
    }
  }
  // Atomic address/payload widths come from the classifier-owned lowering form.
  // Reserve them explicitly: generic def/use can understate native two-address
  // operations, and rejecting an overlap only during emission invalidates the
  // whole code-object transform instead of selecting another scratch window.
  if (atomic_form != nullptr) {
    request.forbidden.expand({RegClass::VGPR, atomic_form->address_vgpr,
                              static_cast<uint8_t>(atomic_form->address_vgpr_count)});
    request.forbidden.expand({RegClass::VGPR, atomic_form->data_vgpr,
                              static_cast<uint8_t>(atomic_form->data_register_count)});
    if (atomic_form->destination_vgpr && atomic_form->destination_register_count != 0u) {
      request.forbidden.expand({RegClass::VGPR, *atomic_form->destination_vgpr,
                                static_cast<uint8_t>(atomic_form->destination_register_count)});
    }
  }
  if (!point.owner_epoch_vgprs.owner() &&
      state.register_allocation.owner_persistent_vgprs.empty() &&
      semantic_request.owner_source == OwnerSource::WorkitemId)
    request.forbidden.expand({RegClass::VGPR, kAmdGpuWorkitemIdX, 1});
  forbid_persistent_vgprs(request.forbidden, point, state.register_allocation,
                          owner_descriptor_offsets);

  RegisterPlan register_plan = plan_registers(request, live_before_all_owners);
  if (register_plan.source == RegisterAllocationSource::Unsupported &&
      register_plan.reason == RegisterPlanReason::NoLegalWindow &&
      every_owner_can_move_empty_accumulator_boundary && !request.explicit_base &&
      !request.force_spill) {
    // Trusted metadata can prove that a rounded unified-VGPR tail contains no
    // accumulator state even when ACCUM_OFFSET conservatively places the
    // ordinary-register boundary below it. Retry descriptor growth only after
    // ordinary placement fails. This keeps already-supported kernels on their
    // original allocation while avoiding hot-path private-memory spills for
    // the otherwise unplaceable sites.
    RegisterRequest growth_request = request;
    growth_request.architecture_limit = kMaxVgprs;
    growth_request.allow_spill = false;
    register_plan = plan_registers(growth_request, live_before_all_owners);
    plan.alternatives.push_back({
        .kind = ResourcePlanAlternativeKind::EmptyAccumulatorDescriptorGrowth,
        .source = register_plan.source,
        .reason = register_plan.reason,
        .scratch_vgpr_count = scratch_count,
        .outcome = register_plan.source == RegisterAllocationSource::DescriptorGrowth
                       ? ResourcePlanAlternativeOutcome::Selected
                       : ResourcePlanAlternativeOutcome::Rejected,
    });
  }
  const AccessResourceFacts access_resource_facts =
      access_candidate != nullptr
          ? resolve_access_resource_facts(point, *access_candidate, state.arch)
          : AccessResourceFacts{.target_available = is_capability_arch(state.arch)};
  const OperandOverlapSpillPolicy operand_overlap_spill = detail::operand_overlap_spill({
      semantic_request,
      access_resource_facts,
      access_candidate,
      site_kind,
  });
  if (register_plan.source == RegisterAllocationSource::Unsupported &&
      register_plan.reason == RegisterPlanReason::NoLegalWindow &&
      operand_overlap_spill.supported && !request.explicit_base) {
    // These appended-cave bodies preserve guest operands even when no
    // disjoint ordinary-VGPR window exists. ConSan atomics and the admitted
    // ConSan stores execute the guest before clobbering an overlapping saved
    // window; ConSan access bodies restore the window before the
    // displaced guest. Persistent state remains forbidden because it must
    // survive the complete kernel lifetime. Shared physical code can have
    // owners with different descriptor allocations, so this capability-gated
    // fallback may also grow the smaller descriptors to hold the spill window.
    RegisterRequest spill_request = request;
    spill_request.force_spill = true;
    spill_request.allow_spill_descriptor_growth = true;
    if (every_owner_can_move_empty_accumulator_boundary)
      spill_request.architecture_limit = kMaxVgprs;
    spill_request.forbidden = {};
    forbid_persistent_vgprs(spill_request.forbidden, point, state.register_allocation,
                            owner_descriptor_offsets);
    if (operand_overlap_spill.protected_vgpr) {
      spill_request.forbidden.expand({RegClass::VGPR, *operand_overlap_spill.protected_vgpr,
                                      operand_overlap_spill.protected_vgpr_count});
    }
    register_plan = plan_registers(spill_request, live_before_all_owners);
    plan.alternatives.push_back({
        .kind = ResourcePlanAlternativeKind::GuestOperandOverlapSpill,
        .source = register_plan.source,
        .reason = register_plan.reason,
        .scratch_vgpr_count = scratch_count,
        .outcome = register_plan.source == RegisterAllocationSource::Unsupported
                       ? ResourcePlanAlternativeOutcome::Rejected
                       : ResourcePlanAlternativeOutcome::Selected,
    });
  }
  plan.source = register_plan.source;
  plan.reason = register_plan.reason;
  plan.scratch_vgpr = register_plan.base;
  plan.required_vgpr_count = register_plan.required_descriptor_count;
  if (plan.source == RegisterAllocationSource::SpillRequired) {
    const bool has_dynamic_owner =
        std::ranges::any_of(owners, [](const KernelResourceContext *owner) {
          return owner->kernel->uses_dynamic_stack.value_or(false);
        });
    const bool every_owner_dynamic =
        std::ranges::all_of(owners, [](const KernelResourceContext *owner) {
          return owner->kernel->uses_dynamic_stack.value_or(false);
        });
    // A shared private frame recipe requires every owner to use a dynamic stack.
    const bool supported_dynamic_spill = is_capability_arch(state.arch) && every_owner_dynamic;
    if (has_dynamic_owner && !supported_dynamic_spill) {
      plan.source = RegisterAllocationSource::Unsupported;
      plan.reason = RegisterPlanReason::DynamicStack;
      plan.scratch_vgpr.reset();
      plan.required_vgpr_count = plan.current_vgpr_count;
    }
  }
  return plan;
}

/// Preserve the complete feasibility-search trace when an outer fallback
/// retries resource planning with a different scratch shape. A successful
/// retry supersedes the first selected path and incorporates its nested
/// choices; a failed retry leaves the original plan selected and records its
/// nested selections as vetoed.
void adopt_resource_plan_fallback(CandidateResourcePlan &plan, CandidateResourcePlan fallback,
                                  ResourcePlanAlternativeKind kind, uint16_t scratch_count) {
  const bool selected = fallback.source == RegisterAllocationSource::SpillRequired;
  ResourcePlanAlternative outer{
      .kind = kind,
      .source = fallback.source,
      .reason = fallback.reason,
      .scratch_vgpr_count = scratch_count,
      .outcome = selected ? ResourcePlanAlternativeOutcome::Selected
                          : ResourcePlanAlternativeOutcome::Rejected,
  };
  std::vector<ResourcePlanAlternative> initial = std::move(plan.alternatives);
  std::vector<ResourcePlanAlternative> nested = std::move(fallback.alternatives);
  if (selected) {
    for (ResourcePlanAlternative &alternative : initial) {
      if (alternative.outcome == ResourcePlanAlternativeOutcome::Selected)
        alternative.outcome = ResourcePlanAlternativeOutcome::Superseded;
    }
  }
  for (ResourcePlanAlternative &alternative : nested) {
    if (alternative.outcome == ResourcePlanAlternativeOutcome::Selected) {
      alternative.outcome = selected ? ResourcePlanAlternativeOutcome::Contributed
                                     : ResourcePlanAlternativeOutcome::Vetoed;
    }
  }
  std::vector<ResourcePlanAlternative> trace;
  trace.reserve(initial.size() + 1u + nested.size());
  trace.insert(trace.end(), std::make_move_iterator(initial.begin()),
               std::make_move_iterator(initial.end()));
  trace.push_back(outer);
  trace.insert(trace.end(), std::make_move_iterator(nested.begin()),
               std::make_move_iterator(nested.end()));
  if (selected)
    plan = std::move(fallback);
  plan.alternatives = std::move(trace);
}

void append_resource_plans(ResourcePlanningState &state, const Request &request,
                           const DebugOverrides &debug, const OperatingPoint &point,
                           std::span<const Candidate> candidates,
                           std::vector<CandidateResourcePlan> &plans) {
  plans.reserve(plans.size() + candidates.size());
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const Candidate &candidate = candidates[candidate_index];
    OperatingPoint site_point = point;
    AccessResourceFacts site_resource_facts =
        resolve_access_resource_facts(site_point, candidate, state.arch);
    uint16_t scratch_count = direct_scratch_count(request, site_resource_facts);
    CandidateResourcePlan plan = plan_resource_site(
        state, request, debug, site_point, ResourceSiteKind::Access, candidate_index,
        candidate.anchor(), std::nullopt, scratch_count, &candidate);
    // Persistent placement can deliberately mix private state with
    // component-local VGPR tuples. The first plan discovers the physical
    // site's complete owner set; size and place its final scratch window for
    // that owner's actual ABI, rather than the code-object-wide fallback.
    OperatingPoint owner_point = site_point;
    const auto owner_descriptors = state.inventory.kernel_descriptors(plan.owner_kernel_ids);
    if (owner_descriptors && apply_persistent_vgpr_assignment(
                                 owner_point, state.register_allocation, *owner_descriptors)) {
      const AccessResourceFacts owner_resource_facts =
          resolve_access_resource_facts(owner_point, candidate, state.arch);
      const uint16_t owner_scratch_count = direct_scratch_count(request, owner_resource_facts);
      if (owner_scratch_count != scratch_count) {
        scratch_count = owner_scratch_count;
        plan = plan_resource_site(state, request, debug, owner_point, ResourceSiteKind::Access,
                                  candidate_index, candidate.anchor(), std::nullopt, scratch_count,
                                  &candidate);
      }
      site_resource_facts = owner_resource_facts;
      site_point = std::move(owner_point);
    }
    bool initial_spill_overlaps_guest = false;
    if (plan.source == RegisterAllocationSource::SpillRequired && plan.scratch_vgpr) {
      std::vector<std::string> ignored_overlap_errors;
      initial_spill_overlaps_guest = reject_candidate_scratch_range_overlap(
          candidate, *plan.scratch_vgpr, plan.scratch_vgpr_count, ignored_overlap_errors);
    }
    const bool no_ordinary_window = plan.source == RegisterAllocationSource::Unsupported &&
                                    plan.reason == RegisterPlanReason::NoLegalWindow;
    const std::optional<uint16_t> spill_scratch_count = access_spill_fallback(
        {request, site_resource_facts, no_ordinary_window, initial_spill_overlaps_guest});
    if (spill_scratch_count) {
      CandidateResourcePlan spill_plan = plan_resource_site(
          state, request, debug, site_point, ResourceSiteKind::Access, candidate_index,
          candidate.anchor(), std::nullopt, *spill_scratch_count, &candidate,
          /*atomic_form=*/nullptr, /*require_spill=*/true);
      adopt_resource_plan_fallback(plan, std::move(spill_plan),
                                   ResourcePlanAlternativeKind::SpillBackedOperandRecovery,
                                   *spill_scratch_count);
    }
    plans.push_back(std::move(plan));
  }
}

bool resource_offsets_share_block(const ResourcePlanningState &state, uint64_t first_offset,
                                  uint64_t second_offset) {
  BasicBlock *first = block_for_offset(state.block_index, first_offset);
  return first != nullptr && block_for_offset(state.block_index, second_offset) == first;
}

/// Find the access-site plan whose scratch tuple was checked against every owning kernel.
[[nodiscard]] const CandidateResourcePlan *
resource_plan_for_candidate(std::span<const CandidateResourcePlan> plans,
                            const Candidate &candidate) {
  const auto plan = std::ranges::find_if(plans, [&](const auto &candidate_plan) {
    return candidate_plan.site_kind == ResourceSiteKind::Access &&
           candidate_plan.text_offset == candidate.anchor();
  });
  return plan == plans.end() ? nullptr : &*plan;
}

[[nodiscard]] const CandidateResourcePlan *
resource_plan_for_site(std::span<const CandidateResourcePlan> plans, ResourceSiteKind site_kind,
                       uint64_t text_offset) {
  const auto it = std::ranges::find_if(plans, [&](const auto &plan) {
    return plan.site_kind == site_kind &&
           plan.semantic_text_offset.value_or(plan.text_offset) == text_offset;
  });
  return it == plans.end() ? nullptr : &*it;
}

[[nodiscard]] std::optional<ResolvedScratchPlan>
resolve_scratch_plan(const CandidateResourcePlan &plan, const ProgramInventory &inventory,
                     const OperatingPoint &site_point, OwnerAssignments assignments,
                     uint16_t expected_count) {
  if (plan.scratch_vgpr && plan.scratch_vgpr_count == expected_count &&
      (plan.source == RegisterAllocationSource::Explicit ||
       plan.source == RegisterAllocationSource::LivenessDead ||
       plan.source == RegisterAllocationSource::DescriptorGrowth ||
       plan.source == RegisterAllocationSource::SpillRequired)) {
    ResolvedScratchPlan resolved;
    resolved.base = *plan.scratch_vgpr;
    resolved.count = expected_count;
    resolved.required_vgpr_count = plan.required_vgpr_count;
    resolved.original_private_segment_size = plan.original_private_segment_size;
    auto owner_descriptors = inventory.kernel_descriptors(plan.owner_kernel_ids);
    if (!owner_descriptors)
      return std::nullopt;
    resolved.owner_descriptor_file_offsets = std::move(*owner_descriptors);
    resolved.source = plan.source;
    const auto include_in_required_count = [&](std::optional<uint16_t> reg, uint16_t width) {
      if (reg) {
        resolved.required_vgpr_count =
            std::max<uint16_t>(resolved.required_vgpr_count, static_cast<uint16_t>(*reg + width));
      }
    };
    persistent_vgpr_state_view(site_point).for_each_range(include_in_required_count);
    for (uint64_t descriptor_offset : resolved.owner_descriptor_file_offsets) {
      const auto *assignment = persistent_vgpr_assignment(assignments, descriptor_offset);
      if (!assignment)
        continue;
      persistent_vgpr_state_view(*assignment).for_each_range(include_in_required_count);
    }
    return resolved;
  }

  return std::nullopt;
}

[[nodiscard]] std::optional<ResolvedScratchPlan>
resolve_scratch(std::span<const CandidateResourcePlan> plans, const ProgramInventory &inventory,
                const Candidate &candidate, const OperatingPoint &operating_point,
                uint16_t expected_count) {
  const CandidateResourcePlan *plan = resource_plan_for_candidate(plans, candidate);
  return plan == nullptr ? std::nullopt
                         : resolve_scratch_plan(*plan, inventory, operating_point, operating_point,
                                                expected_count);
}

// GCC 13 loses std::optional engagement facts across the component search
// loops. Every selected/spill payload below is guarded before use; suppress
// this optimizer false positive only for this function and compiler version.
#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 13
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
[[nodiscard]] bool configure_automatic_transient_sgpr_assignments(
    OperatingPoint &options, const ResourceProblem &problem,
    std::span<const CandidateResourcePlan> site_plans, std::vector<std::string> &warnings,
    const ResourcePlanningState &planning_state, bool dispatch_only = false) {
  std::vector<TransientSgprAssignment> &owner_transient_sgprs = options.owner_transient_sgprs;
  const rj_code_arch_t arch = planning_state.arch;
  const Request &request = problem.request();
  const BoundRuntimeResources &resources = problem.resources();
  const bool capability_target = is_capability_arch(arch);
  const bool placement_enabled = capability_target;
  if (!placement_enabled || (!options.automatic_partial_exec_save_sgprs && !dispatch_only) ||
      !options.exec_save_sgpr)
    return false;

  const ResourcePlanningState &state = planning_state;
  if (!state.valid)
    return false;
  const std::vector<std::vector<ProgramContainerId>> owner_components =
      scalar_owner_components(site_plans);
  const ScalarSiteAvailability availability =
      collect_scalar_site_availability(state, site_plans, problem.observation_plan(),
                                       problem.inventory(), options.initialize_owner_epoch);

  const uint16_t count =
      exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, options), arch);
  const uint16_t limit = ordinary_sgpr_limit(arch);

  bool assigned_any = false;
  bool missed_required_assignment = false;
  for (const std::vector<ProgramContainerId> &owners : owner_components) {
    const auto component_contexts = resolve_scalar_owner_contexts(state, owners);
    if (!component_contexts) {
      missed_required_assignment = true;
      continue;
    }
    const uint32_t liveness_search_floor =
        std::ranges::any_of(component_contexts->summaries,
                            [](const auto &context) {
                              return detail::scalar_owner_tail_floor(context) >
                                     context.max_referenced_sgpr_count;
                            })
            ? component_contexts->tail_floor
            : 0u;
    const bool needs_exec_override = !scalar_contexts_admit_reserved_window(
        component_contexts->summaries, *options.exec_save_sgpr, count, arch);
    const bool needs_dispatch_override =
        options.dispatch_sgpr.base() &&
        (!scalar_contexts_admit_reserved_window(component_contexts->summaries,
                                                *options.dispatch_sgpr.base(),
                                                kDispatchStateSgprCount, arch) ||
         range_overlaps(*options.exec_save_sgpr, count, *options.dispatch_sgpr.base(),
                        kDispatchStateSgprCount));
    const bool needs_override = dispatch_only ? needs_dispatch_override : needs_exec_override;
    if (!needs_override)
      continue;
    RegisterSet unavailable;
    for (ProgramContainerId owner : owners) {
      if (const auto it = availability.unavailable_by_kernel_id.find(owner.ordinal);
          it != availability.unavailable_by_kernel_id.end())
        unavailable |= it->second;
    }
    // A code-object-wide EXEC window may already have a complete liveness
    // proof while only the persistent dispatch pair excludes this component.
    // Retain that transient window and override just the dispatch
    // representation rather than inventing a second scalar ABI.
    std::optional<uint16_t> selected;
    if ((dispatch_only || (!needs_exec_override && needs_dispatch_override)) &&
        options.exec_save_sgpr) {
      selected = options.exec_save_sgpr;
    }
    std::optional<uint16_t> spill_frame_base;
    std::optional<uint16_t> spill_scc;
    bool branch_only_scalar_spill = false;
    std::optional<uint16_t> dynamic_stack_borrowed_sgpr;
    const bool all_owners_use_fixed_stack =
        !owners.empty() && component_contexts->all_use_fixed_stack;
    if (!selected) {
      ScalarPlacementDomain domain(arch, limit, component_contexts->summaries, &unavailable);
      domain.occupy(options.owner_sgpr.base(), 1u);
      domain.occupy(options.dispatch_sgpr.base(), kDispatchStateSgprCount);
      if (options.dynamic_stack_spill)
        domain.occupy_dynamic_stack_state();
      const auto candidate = domain.select_first(liveness_search_floor, count, 2u);
      if (candidate)
        selected = candidate;
    }

    const bool supports_compact_scalar_spill = capability_target;
    if (!selected && supports_compact_scalar_spill && address_free_private_limit(arch)) {
      ScalarPlacementDomain domain(arch, limit, component_contexts->summaries, &unavailable);
      domain.occupy(options.owner_sgpr.base(), 1u);
      domain.occupy(options.dispatch_sgpr.base(), kDispatchStateSgprCount);
      if (options.dynamic_stack_spill)
        domain.occupy_dynamic_stack_state();
      // This path preserves every borrowed guest SGPR in address-free private
      // storage before using it. Unlike a merely liveness-dead window, it is
      // safe below a scalar-relative owner's allocation tail.
      spill_frame_base = domain.select_first(0u, 2u, 2u);
      if (spill_frame_base)
        spill_scc = domain.select_first(0u, 1u);
      const uint16_t spilled_scalar_count = 8u;
      if (spill_scc) {
        const auto spill_window =
            domain.select_first(0u, spilled_scalar_count, 2u, /*require_dead=*/false);
        if (spill_window)
          selected = spill_window;
      }
    }
    // Fixed-stack compact owners can preserve the complete borrowed
    // window in private state. RDNA dynamic-stack owners additionally reserve
    // one bootstrap pair.
    const TargetProfile *target = target_profile(arch);
    const bool setup_state_branch_only = supports_compact_scalar_spill && target &&
                                         target->placement.branch_only_spill_embeds_setup_state &&
                                         all_owners_use_fixed_stack;
    if (!selected && address_free_private_limit(arch) && setup_state_branch_only) {
      uint16_t allocated_limit = limit;
      for (const auto &summary : component_contexts->summaries)
        allocated_limit = std::min(allocated_limit, summary.current_sgpr_count);
      ScalarPlacementDomain branch_domain(arch, allocated_limit, component_contexts->summaries);

      branch_domain.occupy_dynamic_stack_state();
      branch_domain.occupy(options.dispatch_sgpr.base(), kDispatchStateSgprCount);
      options.persistent_sgprs.for_each_range(
          [&](std::optional<uint16_t> reg, uint16_t width) { branch_domain.occupy(reg, width); });
      for (uint32_t base = 0u; !selected && base + count <= allocated_limit; base += 2u) {
        ScalarPlacementDomain attempt = branch_domain;
        if (!attempt.select(static_cast<uint16_t>(base), count, /*require_dead=*/false))
          continue;

        selected = static_cast<uint16_t>(base);
      }
      branch_only_scalar_spill = selected.has_value();
      spill_frame_base.reset();
      spill_scc.reset();
    }
    if (!selected) {
      missed_required_assignment = true;
      continue;
    }
    const uint16_t selected_sgpr = *selected;
    const bool spill_backed = (spill_frame_base && spill_scc) || branch_only_scalar_spill ||
                              dynamic_stack_borrowed_sgpr.has_value();
    const std::optional<uint16_t> dispatch_id_sgpr =
        options.dispatch_sgpr.base() &&
                scalar_contexts_admit_reserved_window(component_contexts->summaries,
                                                      *options.dispatch_sgpr.base(),
                                                      kDispatchStateSgprCount, arch) &&
                !range_overlaps(selected_sgpr, count, *options.dispatch_sgpr.base(),
                                kDispatchStateSgprCount)
            ? options.dispatch_sgpr.base()
            : std::nullopt;
    const auto owner_descriptors = state.inventory.kernel_descriptors(owners);
    if (!owner_descriptors) {
      missed_required_assignment = true;
      continue;
    }
    for (uint64_t owner : *owner_descriptors) {
      owner_transient_sgprs.push_back(
          {.descriptor_file_offset = owner,
           .exec_save_sgpr = selected_sgpr,
           .owner_sgpr = options.owner_sgpr.base(),
           .dispatch_id_sgpr = dispatch_id_sgpr,
           .spill_backed = spill_backed,
           .scalar_spill_setup =
               spill_frame_base && spill_scc
                   ? std::optional{ScalarSpillSetup{
                         .temporaries = ScalarSpillTemporaries{*spill_frame_base, *spill_scc}}}
                   : std::nullopt,
           .branch_only_spill = branch_only_scalar_spill ? std::optional{BranchOnlyScalarSpill{
                                                               dynamic_stack_borrowed_sgpr}}
                                                         : std::nullopt});
    }
    assigned_any = true;
  }
  const bool can_exclude_incompatible_component =
      scalar_placement_model(arch) == ScalarPlacementModel::DescriptorPartitioned;
  if (missed_required_assignment && !can_exclude_incompatible_component) {
    owner_transient_sgprs.clear();
    return false;
  }
  if (!assigned_any)
    return false;
  warnings.emplace_back(
      "ConSan assigned owner-local transient SGPR windows to independent owner components");
  return true;
}

#if defined(__GNUC__) && !defined(__clang__) && __GNUC__ == 13
#pragma GCC diagnostic pop
#endif

[[nodiscard]] bool
configure_automatic_exec_save_sgprs(OperatingPoint &options, const ResourceProblem &problem,
                                    std::span<const CandidateResourcePlan> site_plans,
                                    std::vector<std::string> &warnings,
                                    const ResourcePlanningState &planning_state) {
  const rj_code_arch_t arch = problem.arch();
  const Request &request = problem.request();
  const BoundRuntimeResources &resources = problem.resources();
  if (options.exec_save_sgpr)
    return false;
  const uint16_t count =
      exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, options), arch);
  if (count == 0)
    return false;

  const auto plan_contexts =
      resolve_resource_plan_scalar_owner_contexts(planning_state, site_plans);
  if (!plan_contexts.planning_state_valid || plan_contexts.unresolved_owner) {
    std::string warning = "ConSan could not resolve an owner context for automatic EXEC-save "
                          "SGPR placement";
    if (plan_contexts.unresolved_owner)
      warning +=
          " at kernel arena index " + std::to_string(plan_contexts.unresolved_owner->ordinal);
    warnings.push_back(std::move(warning));
    return false;
  }
  // A provisional CDNA resource failure may disappear after owner-local
  // persistent placement shrinks the probe ABI. Prove a global transient
  // window against those resolved owners too; otherwise a window selected
  // from the successful subset is later (correctly) rejected for the newly
  // recovered sites. The partial-owner path below remains the fallback when
  // no one code-object-wide window exists.
  const ScalarOwnerContexts scalar_contexts =
      collect_scalar_owner_contexts(plan_contexts.groups, true);
  if (scalar_contexts.contexts.empty()) {
    warnings.emplace_back(
        "ConSan could not derive automatic EXEC-save SGPRs without an owned candidate "
        "scope");
    return false;
  }

  const uint32_t first = util::align_up(scalar_contexts.tail_floor, 2u);
  const uint16_t ordinary_sgpr_limit = detail::ordinary_sgpr_limit(arch);
  const bool scalar_tail_covers_implicit_references =
      std::ranges::any_of(scalar_contexts.summaries, [](const auto &context) {
        return detail::scalar_owner_tail_floor(context) > context.max_referenced_sgpr_count;
      });
  // Scalar-relative and open-CFG owners may still need a smaller persistent
  // tuple above the same tail. Prefer the high end for the wide transient
  // window so lower legal holes remain available to that later placement.
  ScalarPlacementDomain fresh_domain(arch, ordinary_sgpr_limit, scalar_contexts.summaries);
  fresh_domain.occupy(options.owner_sgpr.base(), 1u);
  fresh_domain.occupy(options.dispatch_sgpr.base(), kDispatchStateSgprCount);
  if (options.dynamic_stack_spill)
    fresh_domain.occupy_dynamic_stack_state();
  const auto fresh_window = scalar_tail_covers_implicit_references
                                ? fresh_domain.select_last(first, count, 2u)
                                : fresh_domain.select_first(first, count, 2u);
  if (fresh_window) {
    options.exec_save_sgpr = *fresh_window;
    options.exec_save_sgprs_persistent = false;
    warnings.emplace_back("ConSan automatically assigned EXEC-save SGPRs s" +
                          std::to_string(*fresh_window) + ":s" +
                          std::to_string(*fresh_window + count - 1u));
    return true;
  }

  // Preserve one owner-wide ABI for the components that admit it, then let
  // the component solver override only the incompatible remainder. This is
  // stronger than an object-wide site-liveness proof because the retained
  // window is safe across every guest instruction in each admitted owner.
  if (is_capability_arch(arch)) {
    const std::span<const detail::ScalarOwnerContextSummary> no_contexts;
    ScalarPlacementDomain base_domain(arch, ordinary_sgpr_limit, no_contexts);
    base_domain.occupy(options.owner_sgpr.base(), 1u);
    if (options.dynamic_stack_spill)
      base_domain.occupy_dynamic_stack_state();
    for (int32_t base = static_cast<int32_t>((ordinary_sgpr_limit - count) & ~1u); base >= 0;
         base -= 2) {
      const uint16_t candidate = static_cast<uint16_t>(base);
      ScalarPlacementDomain layout = base_domain;
      if (!layout.select(candidate, count, /*require_dead=*/false))
        continue;

      const std::optional<uint16_t> prior_dispatch = options.dispatch_sgpr.base();
      const bool prior_dispatch_automatic = options.dispatch_sgpr.automatic();
      std::optional<uint16_t> dispatch = prior_dispatch;
      const uint16_t dispatch_count = kDispatchStateSgprCount;
      const auto group_admits = [&](const ScalarOwnerContextGroup &group,
                                    std::optional<uint16_t> dispatch_base) {
        if (!group.supported ||
            !scalar_contexts_admit_reserved_window(group.owners.summaries, candidate, count, arch))
          return false;
        if (!dispatch_base)
          return true;
        const std::array ranges{SgprRange{candidate, count},
                                SgprRange{*dispatch_base, dispatch_count}};
        return layout.can_select(*dispatch_base, dispatch_count, /*require_dead=*/false) &&
               scalar_contexts_admit_reserved_window(group.owners.summaries, *dispatch_base,
                                                     dispatch_count, arch) &&
               !scalar_contexts_conflict_with_physical_vcc(group.owners.summaries, ranges, arch);
      };

      if (dispatch && range_overlaps(candidate, count, *dispatch, dispatch_count))
        continue;

      const auto admits = [&](const ScalarOwnerContextGroup &group) {
        return group_admits(group, dispatch);
      };
      if (!std::ranges::any_of(plan_contexts.groups, admits) ||
          std::ranges::all_of(plan_contexts.groups, admits))
        continue;
      options.exec_save_sgpr = candidate;
      options.automatic_partial_exec_save_sgprs = true;
      options.exec_save_sgprs_persistent = true;
      options.dispatch_sgpr.set(dispatch, prior_dispatch_automatic);
      const bool assigned_component_local = configure_automatic_transient_sgpr_assignments(
          options, problem, site_plans, warnings, planning_state);
      const bool relocation_recovers_component =
          dispatch == prior_dispatch || assigned_component_local;
      if (relocation_recovers_component &&
          (assigned_component_local ||
           scalar_placement_model(arch) == ScalarPlacementModel::DescriptorPartitioned)) {
        warnings.emplace_back("ConSan reserved an owner-wide EXEC-save window and assigned "
                              "component-local scalar state where needed");
        if (dispatch != prior_dispatch)
          warnings.emplace_back("ConSan relocated automatic dispatch-ID state beside it");
        return true;
      }
      options.owner_transient_sgprs.clear();
      options.exec_save_sgpr.reset();
      options.automatic_partial_exec_save_sgprs = false;
      options.exec_save_sgprs_persistent = false;
      options.dispatch_sgpr.set(prior_dispatch, prior_dispatch_automatic);
    }
  }

  // A high-numbered scalar operand anywhere in a kernel does not imply that
  // every lower SGPR is live at every instrumentation site. Large generated
  // kernels commonly use s104 for a short-lived address calculation while a
  // five-register window is dead throughout all LDS/synchronization sites.
  // Prove one common window with the existing CFG liveness before falling
  // back to an unsupported resource result.
  const ResourcePlanningState &state = planning_state;
  const ScalarSiteAvailability availability =
      collect_scalar_site_availability(state, site_plans, problem.observation_plan(),
                                       problem.inventory(), options.initialize_owner_epoch);
  const RegisterSet &unavailable = availability.unavailable;
  if (state.valid && availability.saw_site) {
    const uint32_t liveness_search_floor = scalar_tail_covers_implicit_references ? first : 0u;
    ScalarPlacementDomain liveness_domain(arch, ordinary_sgpr_limit, scalar_contexts.summaries,
                                          &unavailable);
    liveness_domain.occupy(options.owner_sgpr.base(), 1u);
    liveness_domain.occupy(options.dispatch_sgpr.base(), kDispatchStateSgprCount);
    if (options.dynamic_stack_spill)
      liveness_domain.occupy_dynamic_stack_state();
    const auto liveness_window = liveness_domain.select_first(liveness_search_floor, count, 2u);
    if (liveness_window) {
      options.exec_save_sgpr = *liveness_window;
      warnings.emplace_back("ConSan automatically assigned liveness-dead EXEC-save SGPRs s" +
                            std::to_string(*liveness_window) + ":s" +
                            std::to_string(*liveness_window + count - 1u));
      return true;
    }
  }

  // A monolithic generated library can contain a few full-pressure kernels
  // without making every independent owner unusable.  Keep one global window
  // (and therefore the existing probe ABI), but choose the highest legal
  // window that at least one owner can preserve.  Resource-plan rebuilding
  // rejects every owner group whose compiler allocation reaches the window.
  // This is deliberately more conservative than site-local liveness: the
  // retained window is safe across entry setup and every guest instruction
  // in each admitted kernel.
  if (scalar_placement_model(arch) == ScalarPlacementModel::SpillBacked) {
    const std::span<const detail::ScalarOwnerContextSummary> no_contexts;
    ScalarPlacementDomain high_domain(arch, ordinary_sgpr_limit, no_contexts);
    high_domain.occupy(options.owner_sgpr.base(), 1u);
    high_domain.occupy(options.dispatch_sgpr.base(), kDispatchStateSgprCount);
    if (options.dynamic_stack_spill)
      high_domain.occupy_dynamic_stack_state();
    const int32_t last_base = static_cast<int32_t>((ordinary_sgpr_limit - count) & ~1u);
    for (int32_t base = last_base; base >= 0; base -= 2) {
      const uint16_t candidate = static_cast<uint16_t>(base);
      if (!high_domain.can_select(candidate, count, /*require_dead=*/false))
        continue;
      const bool admits_owner =
          std::ranges::any_of(plan_contexts.groups, [&](const ScalarOwnerContextGroup &group) {
            return group.supported && detail::scalar_owner_contexts_admit_reserved_window(
                                          group.owners.summaries, candidate, count,
                                          /*protect_physical_vcc=*/false);
          });
      if (!admits_owner)
        continue;
      options.exec_save_sgpr = candidate;
      options.automatic_partial_exec_save_sgprs = true;
      options.exec_save_sgprs_persistent = true;
      warnings.emplace_back("ConSan reserved a high EXEC-save SGPR window and excluded only "
                            "incompatible full-pressure owner groups");
      (void)configure_automatic_transient_sgpr_assignments(options, problem, site_plans, warnings,
                                                           state);
      return true;
    }
  }

  // A code object made entirely of full-pressure owners has no owner from
  // which to seed the ordinary high-window fallback. Seed component planning
  // with a placeholder; every admitted owner must replace it with either a
  // liveness-dead window or a spill-backed access-probe layout.
  if (is_capability_arch(arch)) {
    options.exec_save_sgpr = 0u;
    options.automatic_partial_exec_save_sgprs = true;
    if (configure_automatic_transient_sgpr_assignments(options, problem, site_plans, warnings,
                                                       state)) {
      warnings.emplace_back(
          "ConSan admitted full-pressure owners with component-local scalar state");
      return true;
    }
    options.exec_save_sgpr.reset();
    options.automatic_partial_exec_save_sgprs = false;
  }

  warnings.emplace_back("ConSan could not place a fresh automatic EXEC-save SGPR window");
  return false;
}

[[nodiscard]] bool configure_automatic_owner_sgpr(OperatingPoint &point,
                                                  const ResourceProblem &problem,
                                                  std::span<const CandidateResourcePlan> site_plans,
                                                  std::vector<std::string> &diagnostics,
                                                  const ResourcePlanningState &state) {
  const rj_code_arch_t arch = state.arch;
  if (problem.request().owner_source != OwnerSource::HwId || point.owner_sgpr.base())
    return false;

  const auto plan_contexts = resolve_resource_plan_scalar_owner_contexts(state, site_plans);
  if (!plan_contexts.planning_state_valid || plan_contexts.unresolved_owner) {
    std::string warning =
        "ConSan could not resolve an owner context for automatic hw_id owner SGPR placement";
    if (plan_contexts.unresolved_owner)
      warning +=
          " at kernel arena index " + std::to_string(plan_contexts.unresolved_owner->ordinal);
    diagnostics.push_back(std::move(warning));
    return false;
  }
  const ScalarOwnerContexts scalar_contexts =
      collect_scalar_owner_contexts(plan_contexts.groups, true);
  if (scalar_contexts.contexts.empty()) {
    diagnostics.emplace_back(
        "ConSan could not derive an automatic hw_id owner SGPR without an owned candidate "
        "scope");
    return false;
  }

  ScalarPlacementDomain domain(arch, ordinary_sgpr_limit(arch), scalar_contexts.summaries);
  domain.occupy(
      point.exec_save_sgpr,
      exec_save_sgpr_count(
          resolve_exec_save_requirement(problem.request(), problem.resources(), point), arch));
  if (const auto sgpr = domain.select_first(scalar_contexts.tail_floor, 1u)) {
    point.owner_sgpr.set(*sgpr, true);
    diagnostics.emplace_back("ConSan automatically assigned hw_id owner temporary s" +
                             std::to_string(*sgpr));
    return true;
  }

  diagnostics.emplace_back("ConSan could not place a fresh automatic hw_id owner SGPR");
  return false;
}

[[nodiscard]] OperatingPointUpdate
configure_automatic_dispatch_id_sgprs(const OperatingPoint &base, const ResourceProblem &problem,
                                      std::span<const CandidateResourcePlan> site_plans,
                                      const ResourcePlanningState &state) {
  OperatingPointUpdate update{
      .attempted_operating_point = base,
      .rejection = std::nullopt,
      .diagnostics = {},
  };
  OperatingPoint &options = update.attempted_operating_point;
  const Request &request = problem.request();
  const BoundRuntimeResources &resources = problem.resources();
  const rj_code_arch_t arch = state.arch;
  const TargetProfile *target = target_profile(arch);
  const bool has_access_or_atomic_consumer =
      std::ranges::any_of(site_plans, [](const CandidateResourcePlan &plan) {
        return !plan.owner_kernel_ids.empty() && (plan.site_kind == ResourceSiteKind::Access ||
                                                  plan.site_kind == ResourceSiteKind::Atomic);
      });
  const bool needs_dispatch_id = requires_dispatch_identity(
      request, {.access_reports_need_explicit_identity =
                    !target || target->access_reports_need_explicit_dispatch_identity,
                .has_access_or_atomic_consumer = has_access_or_atomic_consumer});
  if (!needs_dispatch_id || (target && target->placement.prefer_literal_dispatch_identity) ||
      !resources.report_buffer_address || options.dispatch_sgpr.base()) {
    return update;
  }
  const auto plan_contexts = resolve_resource_plan_scalar_owner_contexts(state, site_plans);
  if (!plan_contexts.planning_state_valid || plan_contexts.unresolved_owner) {
    std::string warning =
        "ConSan could not resolve an owner context for automatic dispatch-ID SGPR placement";
    if (plan_contexts.unresolved_owner)
      warning +=
          " at kernel arena index " + std::to_string(plan_contexts.unresolved_owner->ordinal);
    update.diagnostics.push_back(std::move(warning));
    return update;
  }
  const ScalarOwnerContexts scalar_contexts =
      collect_scalar_owner_contexts(plan_contexts.groups, true);
  if (scalar_contexts.contexts.empty()) {
    update.diagnostics.emplace_back(
        "ConSan could not derive automatic dispatch-ID SGPRs without an owned candidate "
        "scope");
    return update;
  }

  // Sixteen initialized user SGPRs plus X/Y/Z and WORKGROUP_INFO is the
  // largest legal entry-input window after insertion. Keeping the persistent
  // pair above that window makes it safe for every per-kernel preload map.
  uint32_t first = std::max<uint32_t>(20u, scalar_contexts.tail_floor);

  const auto qualify_owner_local_assignments = [&](uint16_t dispatch_base) {
    for (TransientSgprAssignment &assignment : options.owner_transient_sgprs) {
      const std::array owner = {assignment.descriptor_file_offset};
      const auto contexts = resolve_scalar_owner_contexts(state, owner);
      assignment.dispatch_id_sgpr = contexts && scalar_contexts_admit_reserved_window(
                                                    contexts->summaries, dispatch_base, 2u, arch)
                                        ? std::optional(dispatch_base)
                                        : std::nullopt;
    }
  };

  first = util::align_up(first, 2u);
  const uint16_t dispatch_state_count = kDispatchStateSgprCount;
  const uint16_t ordinary_sgpr_limit = detail::ordinary_sgpr_limit(arch);
  ScalarPlacementDomain dispatch_domain(arch, ordinary_sgpr_limit, scalar_contexts.summaries);
  occupy_transient_scalar_state(dispatch_domain, request, resources, options, options, arch);
  const auto dispatch_base = dispatch_domain.select_first(first, dispatch_state_count, 2u);
  if (dispatch_base) {
    options.dispatch_sgpr.set(*dispatch_base, true);
    qualify_owner_local_assignments(*dispatch_base);
    update.diagnostics.emplace_back("ConSan automatically assigned persistent dispatch-ID "
                                    "state SGPRs s" +
                                    std::to_string(*dispatch_base) + ":s" +
                                    std::to_string(*dispatch_base + dispatch_state_count - 1u));
    return update;
  }

  // A large generated code object can contain one full-SGPR kernel alongside
  // many independent lower-pressure kernels.  Requiring one pair above the
  // maximum reference in the entire object would reject every site because
  // of that one owner. Reserve the highest pair that is legal for at least one
  // owner and let resource-plan rebuilding reject only incompatible groups.
  // The pair remains code-object-wide, so this does not change the report ABI
  // or any emitted probe sequence.
  if (target && target->dispatch_identity == DispatchIdentitySource::PreloadedSgprPair) {
    const std::span<const detail::ScalarOwnerContextSummary> no_contexts;
    ScalarPlacementDomain high_domain(arch, ordinary_sgpr_limit, no_contexts);
    high_domain.occupy(options.owner_sgpr.base(), 1u);
    high_domain.occupy(
        options.exec_save_sgpr,
        exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, options), arch));
    const int32_t last_base =
        static_cast<int32_t>((ordinary_sgpr_limit - dispatch_state_count) & ~1u);
    for (int32_t base = last_base; base >= 20; base -= 2) {
      const uint16_t candidate = static_cast<uint16_t>(base);
      if (!high_domain.can_select(candidate, dispatch_state_count, /*require_dead=*/false))
        continue;
      const bool admits_owner =
          std::ranges::any_of(plan_contexts.groups, [&](const ScalarOwnerContextGroup &group) {
            return group.supported &&
                   scalar_contexts_admit_reserved_window(group.owners.summaries, candidate,
                                                         dispatch_state_count, arch);
          });
      if (!admits_owner)
        continue;
      options.dispatch_sgpr.set(candidate, true);
      qualify_owner_local_assignments(candidate);
      update.diagnostics.emplace_back(
          "ConSan reserved a high dispatch-ID SGPR pair and excluded only "
          "incompatible full-pressure owner groups");
      return update;
    }
  }

  if (target && target->dispatch_identity == DispatchIdentitySource::CodeObjectLiteral) {
    // A dynamic-stack kernel can allocate the full ordinary scalar file while
    // leaving a lower pair completely untouched. Dispatch identity crosses
    // arbitrary guest code, so site-local liveness is not sufficient: require
    // a closed owner CFG, or conservatively scan every declared function when
    // those ranges cover the complete text section. The pair also remains
    // above the largest possible entry-input window so dispatch-ID insertion
    // and guest-input restoration cannot overwrite it.
    OwnerRegisterReferenceSummary references =
        summarize_owner_register_references(scalar_contexts.contexts);
    bool used_complete_text_summary = false;
    if (!references.control_flow_closed) {
      if (auto complete = summarize_complete_text_register_references(state)) {
        references = std::move(*complete);
        used_complete_text_summary = true;
      }
    }
    const bool has_dynamic_stack_owner =
        std::ranges::any_of(problem.inventory().kernels(), [](const ProgramContainer &kernel) {
          return kernel.uses_dynamic_stack.value_or(false);
        });
    if (references.control_flow_closed && !references.uses_indirect_sgprs) {
      ScalarPlacementDomain unused_domain(arch, ordinary_sgpr_limit, scalar_contexts.summaries,
                                          &references.references);
      unused_domain.occupy(options.owner_sgpr.base(), 1u);
      unused_domain.occupy(
          options.exec_save_sgpr,
          exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, options), arch));
      if (has_dynamic_stack_owner)
        unused_domain.occupy_dynamic_stack_state();
      const auto unused_dispatch_base = unused_domain.select_first(20u, dispatch_state_count, 2u);
      if (unused_dispatch_base) {
        options.dispatch_sgpr.set(*unused_dispatch_base, true);
        qualify_owner_local_assignments(*unused_dispatch_base);
        update.diagnostics.emplace_back(
            "ConSan assigned persistent dispatch-ID state to a proven-unused " +
            std::string(rj_code_arch_name(arch)) + " SGPR pair" +
            std::string(used_complete_text_summary ? " using complete-text coverage" : ""));
        return update;
      }
    }
  }

  // Record the stable reader identity literal when no persistent hardware
  // pair survives every owner.
  update.diagnostics.emplace_back(
      "ConSan is using a literal report dispatch ID because no persistent hardware "
      "dispatch-ID SGPR pair fits");
  return update;
}

[[nodiscard]] OperatingPointAttempt
plan_dispatch_id_fallback(const Request &request, const BoundRuntimeResources &resources,
                          const OperatingPoint &base, const ResourceProblem &problem,
                          std::span<const CandidateResourcePlan> site_plans) {
  OperatingPoint point = base;
  OperatingPointAttempt attempt{
      .attempted_operating_point = base,
      .accepted_fallback = std::nullopt,
      .diagnostics = {},
  };
  const rj_code_arch_t arch = problem.arch();
  if (!is_capability_arch(arch) || !point.dispatch_sgpr.automatic() ||
      !point.dispatch_sgpr.base() || !point.exec_save_sgpr) {
    return attempt;
  }

  // Dispatch placement precedes transient EXEC-save placement. Retain the
  // independently proven EXEC window and replace only a conflicting dispatch
  // pair with its literal identity representation.
  std::vector<TransientSgprAssignment> &owner_transient_sgprs = point.owner_transient_sgprs;
  owner_transient_sgprs.clear();
  point.automatic_partial_exec_save_sgprs = true;
  ResourcePlanningState planning_state(problem, point, site_plans);
  std::vector<std::string> diagnostics;
  if (!configure_automatic_transient_sgpr_assignments(point, problem, site_plans, diagnostics,
                                                      planning_state, /*dispatch_only=*/true)) {
    attempt.attempted_operating_point = point;
    return attempt;
  }

  // EXEC and dispatch placement are independent proofs. Preserve every
  // previously selected owner-local EXEC ABI and overlay only the dispatch
  // representation derived above. Replacing the whole assignment here can
  // resurrect a code-object-wide EXEC window that the owner had already
  // rejected, silently dropping all sites in that component.
  const auto dispatch_assignments = std::move(owner_transient_sgprs);
  owner_transient_sgprs = base.owner_transient_sgprs;
  for (const TransientSgprAssignment &dispatch_assignment : dispatch_assignments) {
    auto prior =
        std::ranges::find(owner_transient_sgprs, dispatch_assignment.descriptor_file_offset,
                          &TransientSgprAssignment::descriptor_file_offset);
    if (prior == owner_transient_sgprs.end()) {
      // A partial EXEC proof deliberately leaves an incompatible owner
      // without an assignment. Dispatch-only planning starts from the
      // global EXEC base, so admitting that newly synthesized assignment
      // would resurrect the excluded owner with an unsafe transient
      // window. Owners that safely use the global EXEC ABI remain eligible.
      if (base.automatic_partial_exec_save_sgprs) {
        const std::array owners{dispatch_assignment.descriptor_file_offset};
        const auto owner_contexts = resolve_scalar_owner_contexts(planning_state, owners);
        if (!owner_contexts ||
            !scalar_contexts_admit_reserved_window(
                owner_contexts->summaries, *point.exec_save_sgpr,
                exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, point),
                                     arch),
                arch)) {
          continue;
        }
      }
      owner_transient_sgprs.push_back(dispatch_assignment);
    } else {
      prior->dispatch_id_sgpr = dispatch_assignment.dispatch_id_sgpr;
    }
  }
  // The dispatch-only proof must not reclassify a globally liveness-dead
  // EXEC window as an owner-wide reserved window. The latter is deliberately
  // stricter and would reject guest uses outside instrumentation sites.
  point.automatic_partial_exec_save_sgprs = base.automatic_partial_exec_save_sgprs;

  // The component planner connects all callers of shared text. It retains the
  // hardware pair per owner exactly when that owner's scalar pressure permits
  // it; otherwise the assignment's established contract is to record zero
  // generation.
  const bool selected_zero_generation = std::ranges::any_of(
      owner_transient_sgprs, [](const auto &assignment) { return !assignment.dispatch_id_sgpr; });
  if (!selected_zero_generation) {
    attempt.attempted_operating_point = point;
    return attempt;
  }

  diagnostics.emplace_back(
      "ConSan selected owner-local literal dispatch IDs where the hardware pair "
      "overlaps guest scalar state");
  attempt.attempted_operating_point = point;
  attempt.accepted_fallback = FallbackKind::LiteralDispatchId;
  attempt.diagnostics = std::move(diagnostics);
  return attempt;
}

[[nodiscard]] std::optional<std::string>
validate_scalar_state(const Request &request, const BoundRuntimeResources &resources,
                      const OperatingPoint &point, rj_code_arch_t arch) {
  if (point.dispatch_sgpr.base()) {
    const uint16_t base = *point.dispatch_sgpr.base();
    const uint16_t width = kDispatchStateSgprCount;
    const bool overlaps_owner =
        point.owner_sgpr.base() && range_overlaps(base, width, *point.owner_sgpr.base(), 1u);
    const bool overlaps_exec =
        point.exec_save_sgpr &&
        range_overlaps(
            base, width, *point.exec_save_sgpr,
            exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, point), arch));
    const std::array violations{
        std::pair{base < 20u, std::string_view{"entry window"}},
        std::pair{base % 2u != 0u, std::string_view{"alignment"}},
        std::pair{static_cast<uint32_t>(base) + width > ordinary_sgpr_limit(arch),
                  std::string_view{"ordinary-file bound"}},
        std::pair{persistent_sgpr_range_overlaps_reserved_ordinary_range(base, width, arch),
                  std::string_view{"architectural reservation"}},
        std::pair{overlaps_owner, std::string_view{"owner temporary"}},
        std::pair{overlaps_exec, std::string_view{"EXEC-save window"}},
    };
    std::string reason;
    for (const auto &[violated, item] : violations) {
      if (!violated)
        continue;
      if (!reason.empty())
        reason += ", ";
      reason += item;
    }
    if (!reason.empty()) {
      return "ConSan persistent dispatch-ID SGPR pair s" + std::to_string(base) + ":s" +
             std::to_string(base + width - 1u) + " violates " + reason;
    }
  }
  const std::span<const detail::ScalarOwnerContextSummary> no_contexts;
  const ScalarPlacementDomain domain(arch, ordinary_sgpr_limit(arch), no_contexts);
  const auto invalid = [&](std::optional<uint16_t> base, uint16_t width) {
    return base && !domain.legal_range(*base, width);
  };
  const uint16_t exec_count =
      exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, point), arch);
  const bool exec_invalid = invalid(point.exec_save_sgpr, exec_count);
  const bool owner_invalid = invalid(point.owner_sgpr.base(), 1u);
  bool persistent_invalid = false;
  point.persistent_sgprs.for_each_range([&](std::optional<uint16_t> base, uint16_t width) {
    persistent_invalid |= invalid(base, width);
  });
  bool auxiliary_invalid = false;
  point.for_each_auxiliary_sgpr_range([&](std::optional<uint16_t> base, uint16_t width) {
    auxiliary_invalid |= invalid(base, width);
  });
  const bool branch_only_scalar_invalid = point.branch_only_spill.has_value();
  if (!exec_invalid && !owner_invalid && !persistent_invalid && !auxiliary_invalid &&
      !branch_only_scalar_invalid)
    return std::nullopt;
  return "ConSan scalar instrumentation state aliases an architectural special SGPR";
}

[[nodiscard]] PersistentPlacementUpdate
configure_automatic_persistent_vgprs(const OperatingPoint &base, const ResourceProblem &problem,
                                     const DebugOverrides &debug,
                                     std::span<const CandidateResourcePlan> resource_plans,
                                     const ResourcePlanningState &planning_state) {
  PersistentPlacementUpdate update{
      .attempted_operating_point = base,
      .rejection = std::nullopt,
      .diagnostics = {},
      .prologue_scratch_assignments = {},
  };
  OperatingPoint &options = update.attempted_operating_point;
  const Request &request = problem.request();
  const BoundRuntimeResources &resources = problem.resources();
  const ProgramInventory &program_inventory = problem.inventory();
  const std::span<const uint8_t> bytes = planning_state.bytes;
  const rj_code_arch_t arch = planning_state.arch;
  std::vector<std::string> &diagnostics = update.diagnostics;
  std::vector<PrologueScratchVgprAssignment> &prologue_scratch_assignments =
      update.prologue_scratch_assignments;
  PersistentStateFacts persistent_facts;
  bool has_selectable_bank_transition_owner = false;
  for (const CandidateResourcePlan &plan : resource_plans) {
    if (plan.source == RegisterAllocationSource::Unsupported || plan.owner_kernel_ids.empty()) {
      continue;
    }
    persistent_facts.access_count += plan.site_kind == ResourceSiteKind::Access;
    persistent_facts.atomic_count += plan.site_kind == ResourceSiteKind::Atomic;
    persistent_facts.barrier_count += plan.site_kind == ResourceSiteKind::Barrier;
    for (ProgramContainerId owner : plan.owner_kernel_ids) {
      const ProgramContainer *kernel = program_inventory.container(owner);
      persistent_facts.has_operational_dynamic_stack_owner |=
          kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
      if (kernel != nullptr && kernel->has_text_range && kernel->text_file_offset <= bytes.size() &&
          kernel->entry_text_offset <= bytes.size() - kernel->text_file_offset) {
        const uint64_t begin = kernel->text_file_offset + kernel->entry_text_offset;
        if (kernel->code_size <= bytes.size() - begin) {
          has_selectable_bank_transition_owner |= selectable_vgpr_bank_transition_in_range(
              arch, bytes, begin, begin + kernel->code_size);
        }
      }
    }
  }
  const PersistentStateDemand persistent_demand =
      plan_persistent_state_demand(request, options, persistent_facts);
  const bool needs_exact_workgroup_tuple = persistent_demand.needs_entry_workgroup_tuple;
  const bool needs_cluster_workgroup_id =
      needs_exact_workgroup_tuple && arch_has_cluster_facilities(arch) &&
      std::ranges::any_of(program_inventory.kernels(), [](const ProgramContainer &kernel) {
        return kernel.uses_cluster_workgroup_id;
      });
  const TargetProfile *target = target_profile(arch);
  // Persistent VGPR values must survive arbitrary guest bank-selection state.
  // Once a CDNA5 owner changes that state, the eight-bit register numbers in
  // its instructions no longer provide a complete physical-register bound.
  // Unlike a site-local scratch window, persistent values cannot be selected
  // transiently around every guest use. Treat the descriptor's complete
  // selectable-bank allocation as occupied and route an overflowing tuple
  // through scalar or private state. Owners which never change bank retain
  // the ordinary precise explicit-operand placement path.
  constexpr uint32_t kDirectlyEncodedVgprCount = 256u;
  const uint32_t persistent_vgpr_limit =
      target && target->has_selectable_vgpr_bank ? kDirectlyEncodedVgprCount : kMaxVgprs;
  const bool needs_scratch_floor =
      target && target->accumulator_model != AccumulatorModel::DescriptorPartitioned;
  const bool has_explicit_owner_epoch = options.owner_epoch_vgprs.complete();
  if (!exact_entry_workgroup_capture_is_unambiguous(options)) {
    update.rejection = PlacementRejection::AmbiguousWorkgroupCapture;
    diagnostics.emplace_back("ConSan has multiple persistent workgroup-tuple representations");
    return update;
  }
  if (debug.requested_owner_vgpr.has_value() != debug.requested_epoch_vgpr.has_value()) {
    update.rejection = PlacementRejection::PartialOwnerEpochVgprOverride;
    diagnostics.emplace_back(
        "ConSan persistent VGPR overrides require both owner and epoch registers");
    return update;
  }
  if (has_explicit_owner_epoch && !needs_exact_workgroup_tuple)
    return update;
  if (!persistent_demand.needs_persistent_state)
    return update;
  if (!resources.report_buffer_address)
    return update;

  uint32_t persistent_base = 0;
  bool found_owned_scope = false;
  for (const CandidateResourcePlan &plan : resource_plans) {
    if (plan.owner_kernel_ids.empty())
      continue;
    found_owned_scope = true;
    // Persistent state crosses arbitrary guest code, so it must sit above
    // every statically referenced ordinary VGPR. Indirect v_movrel access has
    // no complete static bound and conservatively reserves the ordinary guest
    // allocation instead. On CDNA3/4, ACCUM_OFFSET bounds this search below the
    // physical AccVGPR bank even when the descriptor's total VGPR allocation
    // extends through that bank.
    const uint32_t safe_base = plan.has_indirect_vgpr_access || has_selectable_bank_transition_owner
                                   ? plan.current_vgpr_count
                                   : plan.max_referenced_vgpr_count;
    persistent_base = std::max(persistent_base, safe_base);
  }
  if (has_explicit_owner_epoch) {
    persistent_base = std::max<uint32_t>(
        persistent_base,
        std::max(options.owner_epoch_vgprs->owner, options.owner_epoch_vgprs->epoch) + 1u);
  }

  if (!found_owned_scope) {
    diagnostics.emplace_back(
        "ConSan could not derive automatic persistent VGPRs without an owned candidate "
        "scope");
    return update;
  }
  if (needs_scratch_floor) {
    // Persistent owner/epoch and exact-workgroup state shares the ordinary
    // VGPR file with site-local ConSan probes. Keep it above
    // the largest operational scratch window. Guest operands may overlap a
    // spill-backed window because emission recovers them from private state;
    // persistent values cannot overlap it because they must survive for the
    // complete dispatch.
    uint32_t scratch_floor = 0;
    for (const CandidateResourcePlan &plan : resource_plans) {
      if (plan.source != RegisterAllocationSource::Unsupported && !plan.owner_kernel_ids.empty()) {
        const uint32_t scratch_end =
            plan.scratch_vgpr ? static_cast<uint32_t>(*plan.scratch_vgpr) + plan.scratch_vgpr_count
                              : plan.scratch_vgpr_count;
        scratch_floor = std::max(scratch_floor, scratch_end);
      }
    }
    persistent_base = std::max(persistent_base, scratch_floor);
  }
  const uint32_t automatic_owner_epoch_count = has_explicit_owner_epoch ? 0u : 2u;
  const uint32_t exact_workgroup_count =
      needs_exact_workgroup_tuple ? (needs_cluster_workgroup_id ? 4u : 3u) : 0u;
  const uint32_t persistent_count = automatic_owner_epoch_count + exact_workgroup_count;
  const uint32_t persistent_end = persistent_base + persistent_count;
  const auto has_dynamic_stack_resource_owner = [&] {
    return std::ranges::any_of(program_inventory.kernels(), [&](const ProgramContainer &kernel) {
      return kernel.uses_dynamic_stack.value_or(false) &&
             std::ranges::any_of(resource_plans, [&](const CandidateResourcePlan &plan) {
               // Resource planning precedes persistent-state placement.
               // A full-pressure site can therefore be provisionally
               // unsupported only because the code-object-wide fallback
               // ABI is larger than its eventual owner-local ABI. Keep
               // every semantically admitted, resolved owner in this
               // query so placement can recover that site on the rebuild.
               return (plan.source != RegisterAllocationSource::Unsupported ||
                       resource_plan_can_be_recovered_by_persistent_placement(plan, debug, arch)) &&
                      std::ranges::find(plan.owner_kernel_ids, kernel.id) !=
                          plan.owner_kernel_ids.end();
             });
    });
  };
  const auto assign_scalar_persistent_state = [&](uint16_t base) {
    uint16_t next = base;
    if (!has_explicit_owner_epoch) {
      options.persistent_sgprs.set_owner_epoch(next, static_cast<uint16_t>(next + 1u));
      next = static_cast<uint16_t>(next + 2u);
    }
    if (needs_exact_workgroup_tuple) {
      options.persistent_sgprs.exact_workgroup = PersistentWorkgroupRegisters{
          next, static_cast<uint16_t>(next + 1u), static_cast<uint16_t>(next + 2u),
          needs_cluster_workgroup_id ? std::optional<uint16_t>(next + 3u) : std::nullopt};
    }
  };
  const bool crosses_cdna_accvgpr_boundary =
      scalar_placement_model(program_inventory.arch()) ==
          ScalarPlacementModel::DescriptorPartitioned &&
      std::ranges::any_of(resource_plans, [&](const CandidateResourcePlan &plan) {
        return plan.source != RegisterAllocationSource::Unsupported &&
               !plan.owner_kernel_ids.empty() &&
               ((plan.ordinary_vgpr_limit != kMaxVgprs &&
                 persistent_end > plan.ordinary_vgpr_limit) ||
                persistent_end > kMaxVgprs);
      });
  if (crosses_cdna_accvgpr_boundary) {
    // ACCUM_OFFSET defines the compiler's ordinary/accumulator register split.
    // Moving that split after code generation changes which physical registers
    // existing MFMA and accumulator operands address and has corrupted native
    // gfx950 workloads. Until persistent tuples are selected per owning kernel,
    // reject a code object whose one global tuple would cross any such boundary.
    const bool has_dynamic_stack_owner = has_dynamic_stack_resource_owner();

    if (request.owner_source == OwnerSource::WorkitemId && !has_dynamic_stack_owner) {
      options.initialize_owner_epoch = true;
      options.automatic_private_epoch = true;
      diagnostics.emplace_back(
          "ConSan selected entry-snapshotted private owner/epoch state below a CDNA ordinary "
          "VGPR boundary for ConSan instrumentation");
      return update;
    }

    // Persistent placement and resource planning form a fixed point. Failed
    // plans still contribute ownership so a smaller component-local ABI can
    // recover them on the subsequent resource-plan rebuild.
    const std::vector<std::vector<ProgramContainerId>> components =
        scalar_owner_components(resource_plans);

    std::string scalar_persistence_failure;
    const auto try_assign_scalar_persistent_state = [&] {
      // Owner, epoch, and the optional workgroup key are uniform within a
      // wave. A fresh SGPR tail preserves them across arbitrary guest code
      // without moving ACCUM_OFFSET or adding a private-memory transaction at
      // every barrier. Prove the choice across the complete owner union: this
      // is a code-object-wide representation, not a component-local shortcut.
      const uint16_t scalar_count = static_cast<uint16_t>(persistent_count);
      const std::vector<ProgramContainerId> scalar_owners = scalar_owner_kernel_ids(resource_plans);
      const ResourcePlanningState &state = planning_state;
      const auto resolved_contexts = resolve_scalar_owner_contexts(state, scalar_owners);
      if (!resolved_contexts) {
        scalar_persistence_failure = "could not resolve every scalar-state owner descriptor";
        return false;
      }
      auto prologue_scratch = plan_scalar_prologue_scratch(state, *resolved_contexts,
                                                           /*needs_entry_vgpr_pair=*/true);
      if (!prologue_scratch) {
        scalar_persistence_failure =
            "could not place entry-local VGPR scratch for scalar persistent state";
        return false;
      }
      // This state crosses arbitrary guest code. Closed direct-only owners may
      // reuse descriptor padding above their highest named operand. Owners
      // with scalar-relative access or open reference coverage start beyond
      // their original allocation. The candidate filter separately preserves
      // physical VCC before and after descriptor growth.
      const uint32_t first = resolved_contexts->tail_floor;
      const uint16_t scalar_limit = ordinary_sgpr_limit(arch);
      ScalarPlacementDomain scalar_domain(arch, scalar_limit, resolved_contexts->summaries);
      occupy_transient_scalar_state(scalar_domain, request, resources, options, options, arch);
      scalar_domain.occupy_dynamic_stack_state();
      const auto scalar_tail = scalar_domain.select_first(first, scalar_count);
      if (scalar_tail) {
        // A prior disconnected component may already have accumulated a VGPR
        // tuple. Commit the scalar transition only after entry scratch has
        // been proven for the complete union.
        options.owner_persistent_vgprs.clear();
        assign_scalar_persistent_state(*scalar_tail);
        options.initialize_owner_epoch = true;
        prologue_scratch_assignments = std::move(*prologue_scratch);
        diagnostics.emplace_back("ConSan assigned persistent scalar owner/epoch"
                                 " state s" +
                                 std::to_string(*scalar_tail) + ":s" +
                                 std::to_string(*scalar_tail + scalar_count - 1u) +
                                 " below a full CDNA ordinary VGPR bank for " + "default");
        return true;
      }
      scalar_persistence_failure =
          "found no persistent scalar tail outside guest and instrumentation state";
      return false;
    };

    bool assigned_private_component = false;
    for (const std::vector<ProgramContainerId> &component : components) {
      uint32_t component_base = 0u;
      uint32_t component_limit = kMaxVgprs;
      uint32_t component_encoded_limit = kMaxVgprs;
      bool resolved_every_component_context = true;
      for (ProgramContainerId owner : component) {
        const auto found = planning_state.context_by_kernel_id.find(owner.ordinal);
        if (!owner.valid() || found == planning_state.context_by_kernel_id.end() ||
            found->second == nullptr) {
          resolved_every_component_context = false;
          continue;
        }
        const KernelResourceContext &context = *found->second;
        component_encoded_limit =
            std::min<uint32_t>(component_encoded_limit, context.ordinary_vgpr_limit);
        // A component-local tuple may extend beyond the encoded ordinary
        // boundary only when metadata proves that owner has no allocated
        // accumulator bank. Resource replanning then grows the unified VGPR
        // allocation and moves the empty boundary together. A real AccVGPR
        // bank remains a hard limit for every owner that shares this tuple.
        if (!context.can_move_empty_accumulator_boundary)
          component_limit = std::min<uint32_t>(component_limit, context.ordinary_vgpr_limit);
      }
      for (const CandidateResourcePlan &plan : resource_plans) {
        if (std::ranges::none_of(plan.owner_kernel_ids, [&](ProgramContainerId owner) {
              return std::ranges::find(component, owner) != component.end();
            })) {
          continue;
        }
        const uint32_t safe_base =
            plan.has_indirect_vgpr_access
                ? std::min<uint32_t>(plan.current_vgpr_count, plan.ordinary_vgpr_limit)
                : plan.max_referenced_vgpr_count;
        component_base = std::max(component_base, safe_base);
        component_encoded_limit =
            std::min<uint32_t>(component_encoded_limit, plan.ordinary_vgpr_limit);
        if (!resolved_every_component_context)
          component_limit = std::min<uint32_t>(component_limit, plan.ordinary_vgpr_limit);
      }
      const bool crosses_encoded_component_boundary =
          component_base + persistent_count > component_encoded_limit;
      // Scalar state avoids descriptor growth even when trusted metadata says
      // an empty accumulator boundary can move. Prefer that smaller, more
      // conservative representation when it is legal across the complete
      // owner union; component-local VGPR growth remains the fallback when a
      // different owner makes the global scalar tail unavailable.
      if (crosses_encoded_component_boundary && !debug.test_force_private_epoch &&
          try_assign_scalar_persistent_state()) {
        return update;
      }
      if (component_base + persistent_count > component_limit) {
        const bool has_dynamic_stack_owner =
            std::ranges::any_of(component, [&](ProgramContainerId owner) {
              const ProgramContainer *kernel = program_inventory.container(owner);
              return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
            });
        if (!crosses_encoded_component_boundary && !debug.test_force_private_epoch &&
            try_assign_scalar_persistent_state()) {
          return update;
        }
        if (has_dynamic_stack_owner) {
          update.rejection = PlacementRejection::PersistentStateUnavailable;
          diagnostics.emplace_back(
              "ConSan cannot place persistent scalar state (" + scalar_persistence_failure +
              ") for a dynamic-stack CDNA owner with a full ordinary VGPR bank: component " +
              std::to_string(component_base) + ":" +
              std::to_string(component_base + persistent_count) + " exceeds limit " +
              std::to_string(component_limit));
          return update;
        }
        if (!has_dynamic_stack_owner && persistent_demand.private_state_supported) {
          options.initialize_owner_epoch = true;
          options.automatic_private_epoch = true;
          assigned_private_component = true;
          diagnostics.emplace_back(
              "ConSan selected component-local entry-snapshotted private persistent state "
              "below a full CDNA ordinary VGPR bank after scalar placement " +
              (scalar_persistence_failure.empty() ? std::string("was not requested")
                                                  : scalar_persistence_failure));
          continue;
        }
        update.rejection = PlacementRejection::PersistentStateUnavailable;
        if (!scalar_persistence_failure.empty()) {
          diagnostics.emplace_back(
              "ConSan cannot place persistent scalar state (" + scalar_persistence_failure +
              ") or persistent VGPR state below a connected CDNA AccVGPR boundary");
        } else {
          diagnostics.emplace_back(
              "ConSan cannot place persistent VGPR state below a connected CDNA AccVGPR "
              "boundary");
        }
        return update;
      }
      for (ProgramContainerId owner : component) {
        const ProgramContainer *kernel = program_inventory.container(owner);
        if (kernel == nullptr || !kernel->is_kernel()) {
          update.rejection = PlacementRejection::PersistentStateUnavailable;
          diagnostics.emplace_back("ConSan found a stale component owner during placement");
          return update;
        }
        PersistentVgprAssignment assignment;
        assignment.descriptor_file_offset = kernel->descriptor_file_offset;
        uint16_t next = static_cast<uint16_t>(component_base);
        if (has_explicit_owner_epoch) {
          assignment.owner_epoch_vgprs = OwnerEpochRegisters{options.owner_epoch_vgprs->owner,
                                                             options.owner_epoch_vgprs->epoch};
        } else {
          assignment.owner_epoch_vgprs =
              OwnerEpochRegisters{next, static_cast<uint16_t>(next + 1u)};
          next = static_cast<uint16_t>(next + 2u);
        }
        if (needs_exact_workgroup_tuple) {
          const uint16_t x = next++;
          const uint16_t y = next++;
          const uint16_t z = next++;
          assignment.exact_workgroup_vgprs = PersistentWorkgroupRegisters{
              x, y, z, needs_cluster_workgroup_id ? std::optional<uint16_t>(next++) : std::nullopt};
        }
        options.owner_persistent_vgprs.push_back(assignment);
      }
    }
    options.initialize_owner_epoch = true;
    options.automatic_persistent_vgprs = !options.owner_persistent_vgprs.empty();
    diagnostics.emplace_back(
        assigned_private_component && options.automatic_persistent_vgprs
            ? "ConSan assigned component-local private and persistent VGPR state below CDNA "
              "AccVGPR boundaries"
        : assigned_private_component
            ? "ConSan assigned component-local private state below CDNA AccVGPR "
              "boundaries"
            : "ConSan assigned per-owner persistent VGPR tuples below CDNA AccVGPR "
              "boundaries");
    return update;
  }
  const bool would_grow_descriptor =
      std::ranges::any_of(resource_plans, [&](const CandidateResourcePlan &plan) {
        return !plan.owner_kernel_ids.empty() &&
               plan.source != RegisterAllocationSource::Unsupported &&
               persistent_end > plan.current_vgpr_count;
      });
  const bool has_dynamic_stack_owner = has_dynamic_stack_resource_owner();
  // Private epoch storage is address-free scratch and therefore cannot be
  // composed with a guest dynamic stack. If persistent state fits in the
  // architectural VGPR file, grow those owner descriptors instead. The
  // rebuilt resource plans reserve the selected tail VGPRs before allocating
  // per-site scratch, so this growth cannot overlap instrumentation scratch.
  const bool prefer_private_epoch =
      persistent_demand.private_state_supported && !has_dynamic_stack_owner &&
      !has_explicit_owner_epoch &&
      (!needs_exact_workgroup_tuple || persistent_demand.private_workgroup_tuple_supported) &&
      !persistent_demand.synchronization_requires_persistent_owner && would_grow_descriptor;
  const ScalarPlacementModel scalar_model = scalar_placement_model(arch);
  const bool rdna_needs_scalar_epoch =
      scalar_model != ScalarPlacementModel::Unsupported &&
      scalar_model != ScalarPlacementModel::DescriptorPartitioned &&
      !debug.test_force_private_epoch &&
      persistent_demand.scalar_state_required_for_private_or_overflow &&
      (persistent_end > persistent_vgpr_limit || prefer_private_epoch);
  if (rdna_needs_scalar_epoch) {
    const std::string rdna_arch_name = rj_code_arch_name(arch);

    const uint16_t scalar_count = static_cast<uint16_t>(persistent_count);
    const ResourcePlanningState &scalar_state = planning_state;
    const std::vector<ProgramContainerId> scalar_owners = scalar_owner_kernel_ids(resource_plans);
    const auto resolved_contexts = resolve_scalar_owner_contexts(scalar_state, scalar_owners);
    if (!resolved_contexts) {
      if (has_dynamic_stack_owner) {
        update.rejection = PlacementRejection::PersistentStateUnavailable;
        diagnostics.emplace_back("ConSan could not resolve every " + rdna_arch_name +
                                 " dynamic-stack scalar-state owner");
        return update;
      }
      diagnostics.emplace_back("ConSan could not resolve every " + rdna_arch_name +
                               " scalar-state owner");
      return update;
    }
    auto prologue_scratch = plan_scalar_prologue_scratch(scalar_state, *resolved_contexts, true);
    if (!prologue_scratch) {
      if (has_dynamic_stack_owner)
        update.rejection = PlacementRejection::PersistentStateUnavailable;
      diagnostics.emplace_back("ConSan could not place entry-local VGPR scratch for " +
                               rdna_arch_name + " scalar persistent state");
      return update;
    }
    OwnerRegisterReferenceSummary scalar_references =
        summarize_owner_register_references(resolved_contexts->contexts);
    if (!scalar_references.control_flow_closed) {
      if (auto complete = summarize_complete_text_register_references(scalar_state))
        scalar_references = std::move(*complete);
    }
    const uint16_t scalar_limit = ordinary_sgpr_limit(arch);
    ScalarPlacementDomain tail_domain(arch, scalar_limit, resolved_contexts->summaries);
    occupy_transient_scalar_state(tail_domain, request, resources, options, options, arch);
    if (has_dynamic_stack_owner)
      tail_domain.occupy_dynamic_stack_state();
    std::optional<uint16_t> scalar_base =
        tail_domain.select_first(resolved_contexts->tail_floor, scalar_count);
    const uint32_t hole_limit = std::min<uint32_t>(resolved_contexts->tail_floor, scalar_limit);
    if (has_dynamic_stack_owner && !scalar_base && scalar_references.control_flow_closed &&
        !scalar_references.uses_indirect_sgprs) {
      ScalarPlacementDomain hole_domain(arch, static_cast<uint16_t>(hole_limit),
                                        resolved_contexts->summaries,
                                        &scalar_references.references);
      occupy_transient_scalar_state(hole_domain, request, resources, options, options, arch);
      hole_domain.occupy_dynamic_stack_state();
      scalar_base = hole_domain.select_first(0u, scalar_count);
    }
    if (scalar_base) {
      const uint16_t base = *scalar_base;
      assign_scalar_persistent_state(base);
      options.initialize_owner_epoch = true;
      prologue_scratch_assignments = std::move(*prologue_scratch);
      diagnostics.emplace_back(
          "ConSan assigned persistent scalar " +
          std::string(has_explicit_owner_epoch ? "" : "owner/epoch") +
          std::string(needs_exact_workgroup_tuple ? "/exact-workgroup-tuple" : "") +
          " state for an " + rdna_arch_name + " " + "default" +
          " owner without growing the VGPR allocation");
      return update;
    }
    if (has_dynamic_stack_owner) {
      update.rejection = PlacementRejection::PersistentStateUnavailable;
      diagnostics.emplace_back(
          std::string("ConSan could not place persistent scalar owner/epoch state for an ") +
          rdna_arch_name + " dynamic-stack " + "default" + " owner");
      return update;
    }
    if (needs_exact_workgroup_tuple) {
      diagnostics.emplace_back("ConSan could not place the exact workgroup tuple in persistent " +
                               rdna_arch_name +
                               " scalar state; retaining it in entry-captured private state");
    }
    diagnostics.emplace_back(
        std::string("ConSan could not place persistent scalar owner/epoch state for an ") +
        rdna_arch_name + " " + "default" + " owner; retaining the existing private-state fallback");
  }
  const bool needs_private_epoch = options.automatic_private_epoch ||
                                   debug.test_force_private_epoch ||
                                   persistent_end > persistent_vgpr_limit || prefer_private_epoch;
  if (needs_private_epoch && !has_dynamic_stack_owner &&
      persistent_demand.private_state_supported) {
    options.initialize_owner_epoch = true;
    options.automatic_private_epoch = true;
    diagnostics.emplace_back(
        std::string("ConSan automatically selected derived-owner/private-"
                    "epoch persistent state") +
        (would_grow_descriptor ? " to preserve the guest VGPR allocation" : ""));
    return update;
  }
  if (needs_private_epoch) {
    update.rejection = PlacementRejection::PersistentStateUnavailable;
    diagnostics.emplace_back(
        "ConSan cannot retain exact entry state for a dynamic-stack owner without a "
        "persistent register window");
    return update;
  }

  uint16_t next_persistent_vgpr = static_cast<uint16_t>(persistent_base);
  if (!has_explicit_owner_epoch) {
    const uint16_t owner_vgpr = next_persistent_vgpr++;
    const uint16_t epoch_vgpr = next_persistent_vgpr++;
    options.set_owner_epoch_vgprs(owner_vgpr, epoch_vgpr);
  }
  if (needs_exact_workgroup_tuple) {
    const uint16_t x = next_persistent_vgpr++;
    const uint16_t y = next_persistent_vgpr++;
    const uint16_t z = next_persistent_vgpr++;
    options.exact_workgroup_vgprs = PersistentWorkgroupRegisters{
        x, y, z,
        needs_cluster_workgroup_id ? std::optional<uint16_t>(next_persistent_vgpr++)
                                   : std::nullopt};
  }

  options.initialize_owner_epoch = true;
  options.automatic_persistent_vgprs = true;
  diagnostics.emplace_back(
      "ConSan automatically assigned persistent " +
      std::string(has_explicit_owner_epoch ? "" : "owner/epoch") +
      std::string(needs_exact_workgroup_tuple ? "/exact-workgroup-tuple" : "") + " VGPRs v" +
      std::to_string(persistent_base) + ":v" + std::to_string(next_persistent_vgpr - 1u) +
      (would_grow_descriptor && has_dynamic_stack_owner
           ? " with descriptor growth because private epoch storage is incompatible with a "
             "dynamic-stack owner"
           : ""));
  return update;
}

void note_descriptor_requirements(DescriptorVgprRequirements &requirements,
                                  const ResolvedScratchPlan &plan) {
  note_maximum_descriptor_extents(requirements, plan.owner_descriptor_file_offsets,
                                  plan.required_vgpr_count);
}

void note_sgpr_requirements(DescriptorSgprRequirements &requirements,
                            const ResolvedScratchPlan &plan, const Request &request,
                            const BoundRuntimeResources &resources, const OperatingPoint &point,
                            rj_code_arch_t arch) {
  uint16_t required_count = 0;
  const auto note_range = [&](std::optional<uint16_t> base, uint16_t width) {
    if (base)
      required_count = std::max<uint16_t>(required_count, static_cast<uint16_t>(*base + width));
  };
  note_range(point.exec_save_sgpr,
             exec_save_sgpr_count(resolve_exec_save_requirement(request, resources, point), arch));
  if (request.owner_source == OwnerSource::HwId)
    note_range(point.owner_sgpr.base(), 1u);
  note_range(point.dispatch_sgpr.base(), kDispatchStateSgprCount);
  point.persistent_sgprs.for_each_range(note_range);
  point.for_each_auxiliary_sgpr_range(note_range);
  if (required_count == 0)
    return;
  note_maximum_descriptor_extents(requirements, plan.owner_descriptor_file_offsets, required_count);
}

void note_spill_descriptor_requirements(DescriptorPrivateRequirements &requirements,
                                        const ResolvedScratchPlan &plan,
                                        const VgprSpillSequence &spill) {
  note_maximum_descriptor_extents(requirements, plan.owner_descriptor_file_offsets,
                                  spill.total_private_bytes);
}

void note_dynamic_stack_private_requirement(PatchAbiEffects &effects,
                                            const VgprSpillSequence *spill) {
  if (spill == nullptr || !spill->uses_dynamic_stack_frame)
    return;
  effects.dynamic_private_segment_addend =
      std::max(effects.dynamic_private_segment_addend, spill->dynamic_frame_bytes);
}
