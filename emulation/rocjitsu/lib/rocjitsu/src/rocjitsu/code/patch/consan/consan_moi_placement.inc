/// Return the physical offsets whose ordinary-access bodies must be placed
/// after ordinary candidates so a later ordering pass can still wrap their
/// relocated guest instruction within SOPP reach. Record/Replay and Sampled
/// use the same qualified synchronization inventory; disabled atomic tracking
/// deliberately produces no placement constraint.
[[nodiscard]] std::unordered_set<uint64_t>
collect_moi_ordering_atomic_offsets(const ConSanObservationPlan &observation_plan,
                                    bool track_atomics) {
  std::unordered_set<uint64_t> offsets;
  if (!track_atomics)
    return offsets;
  for (const ConSanAtomicSiteDecision &decision : observation_plan.atomic_site_decisions) {
    if (decision.kind == ConSanSiteDecisionKind::Admitted)
      offsets.insert(decision.semantic_site.physical.original_text_offset);
  }
  return offsets;
}

/// Return the shared placement order for replayable access bodies. Physical
/// order is deterministic, while an access that is also an admitted atomic
/// communication event moves to the tail so the later synchronization pass
/// can still wrap its relocated guest within direct-branch reach. This is a
/// placement constraint shared by Record/Replay and Sampled, not evidence
/// selection; the input already contains only policy-admitted candidates.
[[nodiscard]] std::vector<const ConSanMoiCandidate *>
order_moi_replay_access_candidates(std::span<const ConSanMoiCandidate> admitted,
                                   const ConSanObservationPlan &observation_plan,
                                   bool track_atomics) {
  std::vector<const ConSanMoiCandidate *> candidates =
      consan_detail::make_moi_candidate_pointer_view(admitted);
  std::ranges::sort(candidates, [](const ConSanMoiCandidate *lhs, const ConSanMoiCandidate *rhs) {
    return lhs->anchor() < rhs->anchor();
  });
  const std::unordered_set<uint64_t> ordering_atomic_offsets =
      collect_moi_ordering_atomic_offsets(observation_plan, track_atomics);
  std::stable_partition(candidates.begin(), candidates.end(), [&](const auto *candidate) {
    return !ordering_atomic_offsets.contains(candidate->anchor());
  });
  return candidates;
}

/// Return the first aligned register range admitted by one placement policy.
/// Candidate order is part of the generated-code contract: callers provide
/// their original search floor and alignment while this shared traversal owns
/// the common first-fit mechanics.
template <typename Admitted>
[[nodiscard]] std::optional<uint16_t>
first_admitted_moi_register_range(uint32_t begin, uint16_t width, uint16_t limit,
                                  uint16_t alignment, Admitted &&admitted) {
  for (uint32_t base = util::align_up(begin, static_cast<uint32_t>(alignment));
       base + width <= limit; base += alignment) {
    const uint16_t candidate = static_cast<uint16_t>(base);
    if (admitted(candidate))
      return candidate;
  }
  return std::nullopt;
}

void append_nop_padding_to_alignment(std::vector<uint8_t> &bytes, uint64_t alignment,
                                     rj_code_arch_t arch) {
  if (alignment == 0)
    return;
  while (bytes.size() % alignment != 0)
    append_word_bytes(bytes, build_s_nop(0, arch));
}

[[nodiscard]] bool write_word_bytes(std::vector<uint8_t> &bytes, uint64_t offset, uint32_t word) {
  if (offset > bytes.size() || sizeof(word) > bytes.size() - offset)
    return false;
  std::memcpy(bytes.data() + offset, &word, sizeof(word));
  return true;
}

[[nodiscard]] uint32_t moi_descriptor_user_sgpr_count(const KD &desc) {
  return AMDHSA_BITS_GET(desc.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_USER_SGPR_COUNT);
}

[[nodiscard]] uint16_t moi_descriptor_system_sgpr_count(const KD &desc) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  return static_cast<uint16_t>(
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0 ? 1u : 0u) +
      (AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_INFO) != 0 ? 1u : 0u));
}

[[nodiscard]] uint16_t moi_descriptor_dispatch_id_prefix_sgpr_count(const KD &desc) {
  const uint16_t properties = desc.kernel_code_properties;
  return consan_moi_amdhsa_dispatch_id_prefix_sgpr_count(
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) != 0);
}

[[nodiscard]] std::optional<uint16_t> moi_descriptor_kernarg_segment_ptr_sgpr(const KD &desc) {
  const uint16_t properties = desc.kernel_code_properties;
  if (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_KERNARG_SEGMENT_PTR) == 0)
    return std::nullopt;
  return consan_moi_amdhsa_dispatch_id_prefix_sgpr_count(
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_BUFFER) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_PTR) != 0,
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_QUEUE_PTR) != 0, false);
}

[[nodiscard]] bool moi_descriptor_has_kernarg_preload(const KD &desc) {
  return AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH) != 0;
}

[[nodiscard]] ConSanMoiDispatchIdPreloadPlan
moi_descriptor_dispatch_id_preload_plan(const KD &desc, rj_code_arch_t arch) {
  const uint16_t properties = desc.kernel_code_properties;
  const uint16_t original_user_count = static_cast<uint16_t>(moi_descriptor_user_sgpr_count(desc));
  const uint16_t system_count = moi_descriptor_system_sgpr_count(desc);
  const bool dispatch_enabled =
      AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_DISPATCH_ID) != 0;
  const uint16_t prefix = moi_descriptor_dispatch_id_prefix_sgpr_count(desc);
  const uint16_t named_suffix = static_cast<uint16_t>(
      (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_FLAT_SCRATCH_INIT) != 0
           ? 2u
           : 0u) +
      (AMDHSA_BITS_GET(properties, kd::KERNEL_CODE_PROPERTY_ENABLE_SGPR_PRIVATE_SEGMENT_SIZE) != 0
           ? 1u
           : 0u) +
      AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH));
  const uint32_t minimum_declared_user_count =
      static_cast<uint32_t>(prefix) + (dispatch_enabled ? 2u : 0u) + named_suffix;
  if (minimum_declared_user_count > original_user_count) {
    ConSanMoiDispatchIdPreloadPlan invalid;
    invalid.support = ConSanMoiDispatchIdPreloadSupport::InvalidDispatchPosition;
    invalid.dispatch_id_sgpr = prefix;
    invalid.original_user_sgpr_count = original_user_count;
    invalid.expanded_user_sgpr_count = original_user_count;
    invalid.system_sgpr_count = system_count;
    return invalid;
  }
  const ConSanTargetProfile *profile = consan_target_profile(arch);
  const uint16_t user_sgpr_initialization_limit =
      profile ? profile->user_sgpr_initialization_limit : 16u;
  ConSanMoiDispatchIdPreloadPlan plan;
  if (!dispatch_enabled && original_user_count <= user_sgpr_initialization_limit &&
      minimum_declared_user_count + 2u <= original_user_count &&
      static_cast<uint32_t>(original_user_count) + system_count <= kMaxSgprs) {
    // Some producers reserve a larger user-SGPR window than their named
    // preloads occupy. Insert dispatch ID into that padding and shift only the
    // named suffix; neither the declared window nor the following system
    // SGPRs need to move.
    plan.support = ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
    plan.dispatch_id_sgpr = prefix;
    plan.original_user_sgpr_count = original_user_count;
    plan.expanded_user_sgpr_count = original_user_count;
    plan.system_sgpr_count = system_count;
    plan.first_shifted_guest_sgpr = prefix;
    plan.shifted_guest_sgpr_count = named_suffix;
    plan.required_sgpr_count = static_cast<uint16_t>(original_user_count + system_count);
  } else {
    plan = consan_moi_plan_dispatch_id_preload(original_user_count, system_count, prefix,
                                               dispatch_enabled, kMaxSgprs,
                                               user_sgpr_initialization_limit);
  }
  const uint16_t preload_length =
      AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_LENGTH);
  plan.original_kernarg_preload_length = preload_length;
  plan.replacement_kernarg_preload_length = preload_length;
  const auto kernarg_ptr = moi_descriptor_kernarg_segment_ptr_sgpr(desc);
  // RDNA3 and CDNA3/4 initialize at most 16 user SGPRs. When dispatch ID would
  // exceed that window by one or two dwords, shorten the hardware kernarg
  // preload by exactly that overflow and recover those guest values from the
  // still-live kernarg pointer in the entry prologue. System SGPRs move only
  // by the remaining user-window growth, so record that smaller repair
  // separately.
  const uint16_t overflow =
      original_user_count + 2u > user_sgpr_initialization_limit
          ? static_cast<uint16_t>(original_user_count + 2u - user_sgpr_initialization_limit)
          : 0u;
  if (profile && profile->supports_kernarg_preload_overflow_recovery &&
      plan.support == ConSanMoiDispatchIdPreloadSupport::UserSgprInitializationLimit &&
      !dispatch_enabled && overflow >= 1u && overflow <= 2u && kernarg_ptr &&
      preload_length > overflow && prefix + named_suffix == original_user_count) {
    plan.support = ConSanMoiDispatchIdPreloadSupport::SupportedInsert;
    plan.expanded_user_sgpr_count = user_sgpr_initialization_limit;
    plan.first_shifted_guest_sgpr = prefix;
    plan.shifted_guest_sgpr_count = static_cast<uint16_t>(original_user_count - prefix - overflow);
    plan.kernarg_reload_sgpr = static_cast<uint16_t>(original_user_count - overflow);
    plan.kernarg_reload_base_sgpr = *kernarg_ptr;
    plan.kernarg_reload_offset_dwords = static_cast<uint16_t>(
        AMDHSA_BITS_GET(desc.kernarg_preload, kd::KERNARG_PRELOAD_SPEC_OFFSET) + preload_length -
        overflow);
    plan.kernarg_reload_count = overflow;
    plan.system_sgpr_shift = static_cast<uint16_t>(2u - overflow);
    plan.shifted_system_sgpr_count = plan.system_sgpr_shift != 0u ? system_count : 0u;
    plan.original_kernarg_preload_length = preload_length;
    plan.replacement_kernarg_preload_length = static_cast<uint16_t>(preload_length - overflow);
    plan.required_sgpr_count = static_cast<uint16_t>(plan.expanded_user_sgpr_count + system_count);
  }
  return plan;
}

[[nodiscard]] std::optional<uint16_t> moi_descriptor_workgroup_id_sgpr(const KD &desc,
                                                                       uint32_t dimension) {
  const uint32_t rsrc2 = desc.compute_pgm_rsrc2;
  const bool enabled[3] = {
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Y) != 0,
      AMDHSA_BITS_GET(rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_Z) != 0,
  };

  uint32_t sgpr = moi_descriptor_user_sgpr_count(desc);
  for (uint32_t i = 0; i < 3; ++i) {
    if (!enabled[i])
      continue;
    if (i == dimension)
      return static_cast<uint16_t>(sgpr);
    ++sgpr;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<ConSanMoiWorkgroupSources>
moi_descriptor_workgroup_sources(std::span<const uint8_t> image, uint64_t descriptor_file_offset,
                                 rj_code_arch_t arch, std::vector<std::string> &errors,
                                 bool uses_cluster_workgroup_id,
                                 std::optional<uint16_t> cdna_full_payload_user_sgpr_count) {
  const auto descriptor = read_kernel_descriptor(image, descriptor_file_offset);
  if (!descriptor) {
    errors.emplace_back("ConSan MOI workgroup-id descriptor exceeds ELF bytes");
    return std::nullopt;
  }

  const ConSanTargetProfile *profile = consan_target_profile(arch);
  if (profile && profile->command_processor_workgroup_identity) {
    const auto &identity = *profile->command_processor_workgroup_identity;
    // These are translated command-processor ABI TTMPs, not the
    // descriptor-selected entry SGPRs copied from them by the kernel prologue.
    // MOI finalization enables the complete command-processor grid payload on
    // every patched owner descriptor. Preserve that full identity: an x-only
    // guest kernel can still be dispatched over a multidimensional grid, and
    // each workgroup owns a distinct LDS instance.
    return ConSanMoiWorkgroupSources{
        .x = ConSanMoiWorkgroupSource::scalar(ttmp_scalar_operand(identity.grid_x_ttmp)),
        .y = ConSanMoiWorkgroupSource::scalar(ttmp_scalar_operand(identity.grid_yz_ttmp),
                                              /*shift_right_16=*/false,
                                              /*mask_low_16=*/true),
        .z = ConSanMoiWorkgroupSource::scalar(ttmp_scalar_operand(identity.grid_yz_ttmp),
                                              /*shift_right_16=*/true),
        .cluster_workgroup_id = uses_cluster_workgroup_id && identity.cluster_workgroup_id_ttmp
                                    ? ConSanMoiWorkgroupSource::scalar(
                                          ttmp_scalar_operand(*identity.cluster_workgroup_id_ttmp))
                                    : ConSanMoiWorkgroupSource{},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  if (cdna_full_payload_user_sgpr_count && profile) {
    const uint16_t full_base = *cdna_full_payload_user_sgpr_count;
    if (static_cast<uint32_t>(full_base) + 3u > kMaxSgprs) {
      errors.emplace_back("ConSan MOI full CDNA workgroup-ID payload exceeds scalar registers");
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
    return ConSanMoiWorkgroupSources{
        .x = ConSanMoiWorkgroupSource::scalar(full_base),
        .y = ConSanMoiWorkgroupSource::scalar(static_cast<uint16_t>(full_base + 1u)),
        .z = ConSanMoiWorkgroupSource::scalar(static_cast<uint16_t>(full_base + 2u)),
        .cluster_workgroup_id = {},
        .cdna_full_payload_base = full_base,
        .cdna_guest_payload_base =
            static_cast<uint16_t>(moi_descriptor_user_sgpr_count(*descriptor)),
        .cdna_guest_payload_mask = guest_mask,
    };
  }
  const auto sgpr_source = [](std::optional<uint16_t> sgpr) {
    return sgpr ? ConSanMoiWorkgroupSource::scalar(*sgpr) : ConSanMoiWorkgroupSource{};
  };
  return ConSanMoiWorkgroupSources{
      .x = sgpr_source(moi_descriptor_workgroup_id_sgpr(*descriptor, 0)),
      .y = sgpr_source(moi_descriptor_workgroup_id_sgpr(*descriptor, 1)),
      .z = sgpr_source(moi_descriptor_workgroup_id_sgpr(*descriptor, 2)),
      .cluster_workgroup_id = {},
      .cdna_full_payload_base = std::nullopt,
      .cdna_guest_payload_base = std::nullopt,
      .cdna_guest_payload_mask = 0u,
  };
}

/// Exact-coordinate probes execute after arbitrary guest code. Capture the
/// complete tuple at entry instead of depending on descriptor SGPRs or RDNA
/// launch TTMPs retaining their entry values.
[[nodiscard]] std::optional<ConSanMoiWorkgroupSources> moi_exact_entry_workgroup_sources(
    const ConSanMoiOperatingPoint &point,
    const ConSanMoiPersistentWorkgroupPrivateOffsets *private_offsets) {
  if (!moi_has_exact_entry_workgroup_capture(point, private_offsets) ||
      !moi_exact_entry_workgroup_capture_is_unambiguous(point, private_offsets)) {
    return std::nullopt;
  }
  const auto scalar_source = [](uint16_t sgpr) { return ConSanMoiWorkgroupSource::scalar(sgpr); };
  const auto vector_source = [](uint16_t vgpr) { return ConSanMoiWorkgroupSource::vector(vgpr); };
  const auto private_source = [](uint32_t offset) {
    return ConSanMoiWorkgroupSource::private_state(offset);
  };
  if (point.moi_persistent_sgprs.exact_workgroup.complete()) {
    return ConSanMoiWorkgroupSources{
        .x = scalar_source(*point.moi_persistent_sgprs.exact_workgroup.x()),
        .y = scalar_source(*point.moi_persistent_sgprs.exact_workgroup.y()),
        .z = scalar_source(*point.moi_persistent_sgprs.exact_workgroup.z()),
        .cluster_workgroup_id =
            point.moi_persistent_sgprs.exact_workgroup.cluster_workgroup_id()
                ? scalar_source(*point.moi_persistent_sgprs.exact_workgroup.cluster_workgroup_id())
                : ConSanMoiWorkgroupSource{},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  if (point.moi_exact_workgroup_vgprs.complete()) {
    return ConSanMoiWorkgroupSources{
        .x = vector_source(*point.moi_exact_workgroup_vgprs.x()),
        .y = vector_source(*point.moi_exact_workgroup_vgprs.y()),
        .z = vector_source(*point.moi_exact_workgroup_vgprs.z()),
        .cluster_workgroup_id =
            point.moi_exact_workgroup_vgprs.cluster_workgroup_id()
                ? vector_source(*point.moi_exact_workgroup_vgprs.cluster_workgroup_id())
                : ConSanMoiWorkgroupSource{},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  if (private_offsets && private_offsets->complete()) {
    return ConSanMoiWorkgroupSources{
        .x = private_source(*private_offsets->x()),
        .y = private_source(*private_offsets->y()),
        .z = private_source(*private_offsets->z()),
        .cluster_workgroup_id = private_offsets->cluster_workgroup_id()
                                    ? private_source(*private_offsets->cluster_workgroup_id())
                                    : ConSanMoiWorkgroupSource{},
        .cdna_full_payload_base = std::nullopt,
        .cdna_guest_payload_base = std::nullopt,
        .cdna_guest_payload_mask = 0u,
    };
  }
  return std::nullopt;
}

struct MoiKernelResourceContext {
  const ConSanProgramContainer *kernel = nullptr;
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
  bool liveness_is_restricted = false;

  [[nodiscard]] consan_detail::ScalarOwnerContextSummary scalar_summary() const {
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

[[nodiscard]] const ConSanMoiPersistentVgprAssignment *
moi_persistent_vgpr_assignment(MoiOwnerAssignments assignments, uint64_t descriptor_offset) {
  const auto assignment =
      std::ranges::find(assignments.persistent_vgprs, descriptor_offset,
                        &ConSanMoiPersistentVgprAssignment::descriptor_file_offset);
  return assignment == assignments.persistent_vgprs.end() ? nullptr : &*assignment;
}

[[nodiscard]] const ConSanMoiTransientSgprAssignment *
moi_transient_sgpr_assignment(MoiOwnerAssignments assignments, uint64_t descriptor_offset) {
  const auto assignment =
      std::ranges::find(assignments.transient_sgprs, descriptor_offset,
                        &ConSanMoiTransientSgprAssignment::descriptor_file_offset);
  return assignment == assignments.transient_sgprs.end() ? nullptr : &*assignment;
}

/// Returns true when automatic scalar spilling selected after the initial
/// resource pass must share a site-local frame with a dynamic-stack owner.
/// The first pass is still useful: it avoids reserving the dynamic frame ABI
/// for ordinary owners that can keep their transient state in registers.
[[nodiscard]] bool automatic_moi_scalar_spill_needs_dynamic_stack_planning(
    const ConSanMoiOperatingPoint &point, const ProgramInventory &inventory,
    std::span<const ConSanCandidateResourcePlan> site_plans) {
  const auto owner_uses_dynamic_stack = [&](uint64_t descriptor_offset) {
    const ConSanProgramContainer *kernel = inventory.find_kernel_by_descriptor(descriptor_offset);
    return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
  };
  if (std::ranges::any_of(point.owner_transient_sgprs,
                          [&](const ConSanMoiTransientSgprAssignment &assignment) {
                            return assignment.spill_backed &&
                                   owner_uses_dynamic_stack(assignment.descriptor_file_offset);
                          })) {
    return true;
  }
  if (!point.has_moi_scalar_spill()) {
    return false;
  }
  return std::ranges::any_of(site_plans, [&](const ConSanCandidateResourcePlan &plan) {
    return plan.source != ConSanRegisterAllocationSource::Unsupported &&
           std::ranges::any_of(plan.owner_descriptor_file_offsets, owner_uses_dynamic_stack);
  });
}

/// Applies the one transient scalar assignment shared by every owner of a
/// site. Owners connected through a shared helper must either all carry the
/// same assignment or keep the code-object-wide fallback.
[[nodiscard]] bool
apply_moi_transient_sgpr_assignment(const ConSanRequest &request, ConSanMoiOperatingPoint &point,
                                    MoiOwnerAssignments assignments,
                                    std::span<const uint64_t> owner_descriptor_offsets) {
  const ConSanMoiScalarSpillLayout spill_layout =
      moi_mode_operations(request.moi_engine).transient_scalar_placement.spill_layout;
  const ConSanMoiTransientSgprAssignment *selected = nullptr;
  bool saw_unassigned_owner = false;
  for (uint64_t descriptor_offset : owner_descriptor_offsets) {
    const auto *assignment = moi_transient_sgpr_assignment(assignments, descriptor_offset);
    if (!assignment) {
      saw_unassigned_owner = true;
      continue;
    }
    if (selected) {
      ConSanMoiTransientSgprAssignment comparable = *assignment;
      comparable.descriptor_file_offset = selected->descriptor_file_offset;
      if (comparable != *selected)
        return false;
    }
    selected = assignment;
  }
  if (!selected) {
    return point.moi_exec_save_sgpr.has_value() ||
           (point.has_inline_moi_scalar_spill() &&
            (point.moi_scalar_spill_setup || point.moi_branch_only_spill));
  }
  if (saw_unassigned_owner)
    return false;
  point.moi_exec_save_sgpr = selected->exec_save_sgpr;
  point.moi_owner_sgpr.set(selected->owner_sgpr, point.moi_owner_sgpr.automatic());
  // A component assignment replaces the object-wide scalar ABI. Its exact
  // persistent outputs are represented by the fields copied below, so the
  // broad object-wide persistence marker must not reject a spill-backed
  // window whose incoming contents are deliberately backed up at entry.
  if (selected->spill_backed)
    point.moi_exec_save_sgprs_persistent = false;
  point.moi_dispatch_identity.set_sgpr(selected->dispatch_id_sgpr,
                                       point.moi_dispatch_identity.automatic_sgpr());
  point.moi_scalar_spill_setup = selected->scalar_spill_setup;
  if (spill_layout == ConSanMoiScalarSpillLayout::Inline) {
    point.automatic_moi_scalar_spill_layout = selected->spill_backed
                                                  ? ConSanMoiScalarSpillLayout::Inline
                                                  : ConSanMoiScalarSpillLayout::None;
    point.moi_branch_only_spill = selected->branch_only_spill;
  } else {
    point.automatic_moi_scalar_spill_layout = selected->spill_backed
                                                  ? ConSanMoiScalarSpillLayout::Compact
                                                  : ConSanMoiScalarSpillLayout::None;
    point.moi_branch_only_spill.reset();
  }
  return true;
}

[[nodiscard]] bool
moi_transient_sgpr_assignment_is_branch_only(MoiOwnerAssignments assignments,
                                             std::span<const uint64_t> owner_descriptor_offsets) {
  return !owner_descriptor_offsets.empty() &&
         std::ranges::all_of(owner_descriptor_offsets, [&](uint64_t descriptor_offset) {
           const ConSanMoiTransientSgprAssignment *assignment =
               moi_transient_sgpr_assignment(assignments, descriptor_offset);
           return assignment != nullptr && assignment->branch_only_spill;
         });
}

[[nodiscard]] bool
apply_moi_persistent_vgpr_assignment(ConSanMoiOperatingPoint &point,
                                     MoiOwnerAssignments assignments,
                                     std::span<const uint64_t> owner_descriptor_offsets) {
  if (assignments.persistent_vgprs.empty()) {
    return point.moi_owner_epoch_vgprs.owner().has_value() ||
           point.moi_persistent_sgprs.complete() || point.automatic_moi_private_epoch;
  }
  const ConSanMoiPersistentVgprAssignment *selected = nullptr;
  size_t assigned_owner_count = 0u;
  for (uint64_t descriptor_offset : owner_descriptor_offsets) {
    const auto *assignment = moi_persistent_vgpr_assignment(assignments, descriptor_offset);
    if (!assignment)
      continue;
    ++assigned_owner_count;
    if (selected) {
      ConSanMoiPersistentVgprAssignment comparable = *assignment;
      comparable.descriptor_file_offset = selected->descriptor_file_offset;
      if (comparable != *selected)
        return false;
    }
    selected = assignment;
  }
  if (!selected) {
    if (!point.automatic_moi_private_epoch)
      return false;
    point.reset_moi_owner_epoch_vgprs();
    point.moi_workgroup_key_vgpr.reset();
    point.moi_exact_workgroup_vgprs = {};
    point.moi_dispatch_identity.reset_vgpr();
    return true;
  }
  // One physical site cannot consume two different persistent
  // representations. Components are constructed from shared ownership, so a
  // partial match indicates stale or incomplete placement metadata.
  if (assigned_owner_count != owner_descriptor_offsets.size())
    return false;
  point.automatic_moi_private_epoch = false;
  // Owner/epoch/workgroup state and dispatch identity are independently
  // placed. A full-pressure CDNA owner can admit the former in persistent
  // VGPRs while still needing the hardware dispatch preload captured in
  // private entry state. Only a selected dispatch VGPR pair supersedes that
  // private representation.
  if (selected->dispatch_id_vgpr)
    point.moi_dispatch_identity.set_private_fallback(false);
  point.set_moi_owner_epoch_vgprs(selected->owner_epoch_vgprs.owner,
                                  selected->owner_epoch_vgprs.epoch);
  point.moi_workgroup_key_vgpr = selected->workgroup_key_vgpr;
  point.moi_exact_workgroup_vgprs = selected->exact_workgroup_vgprs;
  point.moi_dispatch_identity.set_vgpr(selected->dispatch_id_vgpr);
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

enum class MoiUnclosedControlKind {
  Branch,
  Call,
  Fallthrough,
};

struct MoiOwnerRegisterReferenceSummary {
  RegisterSet references;
  bool uses_indirect_sgprs = false;
  bool uses_indirect_vgprs = false;
  bool control_flow_closed = true;
  std::optional<uint64_t> first_unclosed_control_offset;
  std::string first_unclosed_control_mnemonic;
  MoiUnclosedControlKind first_unclosed_control_kind = MoiUnclosedControlKind::Branch;
};

[[nodiscard]] MoiOwnerRegisterReferenceSummary summarize_moi_owner_register_references(
    std::span<const MoiKernelResourceContext *const> owner_contexts) {
  MoiOwnerRegisterReferenceSummary result;
  std::unordered_set<BasicBlock *> visited;
  const auto classify_unclosed = [](const Instruction &instruction) {
    if ((instruction.flags() & INDIRECT_CALL) != 0)
      return MoiUnclosedControlKind::Call;
    if ((instruction.flags() & (BRANCH | COND_BRANCH | INDIRECT_BRANCH)) != 0)
      return MoiUnclosedControlKind::Branch;
    return MoiUnclosedControlKind::Fallthrough;
  };
  const auto mark_unclosed = [&](const Instruction &instruction,
                                 std::optional<MoiUnclosedControlKind> kind = std::nullopt) {
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
      return MoiUnclosedControlKind::Call;
    case BasicBlock::StaticSuccessorIssue::MissingBranchTarget:
      return MoiUnclosedControlKind::Branch;
    case BasicBlock::StaticSuccessorIssue::MissingFallthrough:
    case BasicBlock::StaticSuccessorIssue::None:
      return MoiUnclosedControlKind::Fallthrough;
    }
    return MoiUnclosedControlKind::Fallthrough;
  };
  for (const MoiKernelResourceContext *context : owner_contexts) {
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

struct MoiResourcePlanningState {
  std::span<const uint8_t> bytes;
  rj_code_arch_t arch;
  ProgramInventory inventory;
  /// Register allocation selected by this lowering attempt. The referenced
  /// operating point is stable for the lifetime of the planning context even
  /// while placement appends owner-local assignments to its vectors.
  const ConSanMoiOperatingPoint &register_allocation;
  /// Bytes unavailable to this lowering attempt. The initial entries come
  /// from the immutable staged-image inventory; MOI may append reservations
  /// for deferred prologue anchors without mutating that inventory.
  std::vector<ConSanPreappliedReservedRange> reserved_ranges;
  AmdGpuCodeObject code_object;
  std::unique_ptr<Decoder> decoder;
  std::vector<std::unique_ptr<BasicBlock>> blocks;
  BlockOffsetIndex block_index;
  std::vector<MoiKernelResourceContext> contexts;
  std::unordered_map<uint32_t, MoiKernelResourceContext *> context_by_kernel_id;
  std::unordered_map<uint64_t, MoiKernelResourceContext *> context_by_descriptor;
  mutable std::unordered_map<uint64_t, RegisterSet> dynamic_frame_saves_by_site;
  bool valid = false;

  MoiResourcePlanningState(const MoiResourceProblem &problem,
                           const ConSanMoiOperatingPoint &allocation,
                           std::span<const ConSanCandidateResourcePlan> prior_site_plans = {})
      : bytes(problem.image()), arch(problem.arch()), inventory(problem.inventory()),
        register_allocation(allocation),
        reserved_ranges(problem.inventory().preapplied_mutation().reserved_ranges),
        code_object(bytes.data(), bytes.size()), decoder(Decoder::create(arch)) {
    if (!code_object.is_valid() || !decoder)
      return;

    const consan_detail::ConSanCfgBuildInputs cfg = consan_detail::build_consan_cfg_inputs(
        code_object, problem.inventory().containers(),
        problem.inventory().preapplied_mutation().code_ranges);
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
      for (const ConSanSyncEvent &event : sync.sync_events)
        query_offsets.insert(event.text_offset());
      for (const ConSanCandidateResourcePlan &plan : prior_site_plans)
        query_offsets.insert(plan.text_offset);
      for (const ConSanProbeIntent &intent : problem.observation_plan().probe_intents)
        query_offsets.insert(intent.physical_site.original_text_offset);
      for (const ConSanProgramContainer &kernel : problem.inventory().kernels()) {
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
    for (const ConSanProgramContainer &kernel : problem.inventory().kernels()) {
      if (!kernel.has_text_range)
        continue;
      std::vector<uint64_t> additional_entry_offsets;
      for (const ConSanPreappliedCodeRange &range :
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

      MoiKernelResourceContext context;
      context.kernel = &kernel;
      context.scope = std::move(*scope);
      contexts.push_back(std::move(context));
      MoiKernelResourceContext &stored = contexts.back();
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
      stored.liveness_is_restricted = restrict_liveness_queries;
      stored.max_referenced_vgpr_count = max_referenced_vgpr_count(stored.scope);
      stored.max_referenced_sgpr_count = std::max<uint16_t>(max_referenced_sgpr_count(stored.scope),
                                                            kernel.sgpr_count.value_or(0u));
      const std::array<const MoiKernelResourceContext *, 1> owner = {&stored};
      const MoiOwnerRegisterReferenceSummary scalar_references =
          summarize_moi_owner_register_references(owner);
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
      const ConSanDescriptorVgprAllocation vgprs = descriptor_vgpr_allocation(*descriptor, arch);
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
      stored.initialized_sgpr_count =
          static_cast<uint16_t>(moi_descriptor_user_sgpr_count(*descriptor) +
                                moi_descriptor_system_sgpr_count(*descriptor));
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
    for (MoiKernelResourceContext &context : contexts) {
      context_by_kernel_id.try_emplace(context.kernel->id.ordinal, &context);
      context_by_descriptor.try_emplace(context.kernel->descriptor_file_offset, &context);
    }
    valid = true;
  }

  MoiResourcePlanningState(std::span<const uint8_t> input, rj_code_arch_t arch,
                           const ProgramInventory &program_inventory,
                           const ConSanObservationPlan &observation_plan,
                           std::span<const ConSanCandidateResourcePlan> site_plans,
                           const ConSanRequest &request, const BoundRuntimeResources &resources,
                           const ConSanMoiOperatingPoint &point)
      : MoiResourcePlanningState(MoiResourceProblem(input, arch, request, resources,
                                                    program_inventory, observation_plan, {}, {}),
                                 point, site_plans) {}
};

struct MoiCfgForwardDistanceIndex {
  const MoiResourcePlanningState *state = nullptr;
  BasicBlock *target = nullptr;
  uint64_t target_text_offset = 0;
  std::unordered_map<BasicBlock *, uint32_t> block_edges;
};

std::span<const ConSanPreappliedReservedRange>
moi_resource_reserved_ranges(const MoiResourcePlanningState &state) {
  return state.reserved_ranges;
}

bool moi_resource_reserve_range_if_uncovered(MoiResourcePlanningState &state,
                                             ConSanPreappliedReservedRange range) {
  if (range.size == 0u)
    return false;
  const bool covered = std::ranges::any_of(
      state.reserved_ranges, [&](const ConSanPreappliedReservedRange &reserved) {
        return reserved.size != 0u && range.text_offset >= reserved.text_offset &&
               range.text_offset - reserved.text_offset < reserved.size;
      });
  if (covered)
    return false;
  state.reserved_ranges.push_back(range);
  return true;
}

Decoder *moi_resource_decoder(MoiResourcePlanningState &state) { return state.decoder.get(); }

bool moi_resource_is_valid(const MoiResourcePlanningState &state) { return state.valid; }

rj_code_arch_t moi_resource_arch(const MoiResourcePlanningState &state) { return state.arch; }

bool moi_resource_has_owner_context(const MoiResourcePlanningState &state,
                                    uint64_t descriptor_offset) {
  const auto found = state.context_by_descriptor.find(descriptor_offset);
  return found != state.context_by_descriptor.end() && found->second != nullptr;
}

void MoiCfgForwardDistanceIndexDeleter::operator()(MoiCfgForwardDistanceIndex *index) const {
  delete index;
}

MoiCfgForwardDistanceIndexPtr
make_moi_cfg_forward_distance_index(const MoiResourcePlanningState &state,
                                    uint64_t descriptor_offset, uint64_t target_text_offset) {
  const auto context_it = state.context_by_descriptor.find(descriptor_offset);
  BasicBlock *target = block_for_offset(state.block_index, target_text_offset);
  if (context_it == state.context_by_descriptor.end() || context_it->second == nullptr ||
      target == nullptr) {
    return {};
  }
  const MoiKernelResourceContext &context = *context_it->second;
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

  MoiCfgForwardDistanceIndexPtr index(new MoiCfgForwardDistanceIndex);
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

std::optional<MoiCfgDistance> moi_cfg_forward_distance(const MoiCfgForwardDistanceIndex &index,
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
  return MoiCfgDistance{found->second, text_distance};
}

void MoiResourcePlanningStateDeleter::operator()(MoiResourcePlanningState *state) const {
  delete state;
}

MoiResourcePlanningStatePtr
make_moi_resource_planning_state(const MoiResourceProblem &problem,
                                 const ConSanMoiOperatingPoint &allocation,
                                 std::span<const ConSanCandidateResourcePlan> prior_site_plans) {
  return MoiResourcePlanningStatePtr(
      new MoiResourcePlanningState(problem, allocation, prior_site_plans));
}

MoiResourcePlanningStatePtr make_moi_resource_planning_state(
    std::span<const uint8_t> input, rj_code_arch_t arch, const ProgramInventory &program_inventory,
    const ConSanObservationPlan &observation_plan,
    std::span<const ConSanCandidateResourcePlan> site_plans, const ConSanRequest &request,
    const BoundRuntimeResources &resources, const ConSanMoiOperatingPoint &point) {
  return MoiResourcePlanningStatePtr(new MoiResourcePlanningState(
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
[[nodiscard]] std::optional<MoiOwnerRegisterReferenceSummary>
summarize_moi_complete_text_register_references(const MoiResourcePlanningState &state) {
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

  MoiOwnerRegisterReferenceSummary result;
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

[[nodiscard]] bool
moi_resource_plan_can_be_recovered_by_persistent_placement(const ConSanCandidateResourcePlan &plan,
                                                           const ConSanDebugOverrides &debug,
                                                           rj_code_arch_t arch) {
  if (plan.source != ConSanRegisterAllocationSource::Unsupported)
    return false;
  if (!consan_arch_is_cdna3_or_cdna4(arch)) {
    return false;
  }
  // Persistent placement and resource planning form a fixed point for the
  // CDNA3/4 stateful ABIs. An owner-local persistent assignment or a
  // component-local transient assignment can recover an automatic window that
  // was rejected against provisional code-object-wide state. Record/Replay's
  // first-light scratch ABI remains unchanged by persistent placement; the
  // fixed point is useful because the conflicts against persistent and
  // transient state become owner-local. Explicit scratch conflicts and
  // structural failures cannot be repaired here and must not make an unrelated
  // owner appear admitted.
  return !plan.owner_descriptor_file_offsets.empty() &&
         (plan.reason == ConSanRegisterPlanReason::NoLegalWindow ||
          (plan.reason == ConSanRegisterPlanReason::ForbiddenOverlap && !debug.scratch_vgpr));
}

[[nodiscard]] std::vector<uint64_t>
moi_scalar_owner_descriptors(std::span<const ConSanCandidateResourcePlan> site_plans) {
  std::vector<uint64_t> owners;
  for (const ConSanCandidateResourcePlan &plan : site_plans) {
    // Scalar persistent state is code-object-wide and must be proven against
    // every semantically admitted owner, including a site whose provisional
    // resource plan failed. Persistent placement can change that site's ABI
    // and recover it on the subsequent resource-plan rebuild; omitting it here
    // would instead select SGPRs that clobber the very owner being recovered.
    for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
      if (std::ranges::find(owners, descriptor_offset) == owners.end())
        owners.push_back(descriptor_offset);
    }
  }
  return owners;
}

/// Partition resource-plan owners into the connected components that must
/// share one scalar ABI. A shared helper joins all kernels that can reach it;
/// unrelated kernels remain independently placeable.
[[nodiscard]] std::vector<std::vector<uint64_t>>
moi_scalar_owner_components(std::span<const ConSanCandidateResourcePlan> plans) {
  std::vector<std::vector<uint64_t>> components;
  for (const ConSanCandidateResourcePlan &plan : plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    std::optional<size_t> destination;
    for (size_t index = 0; index < components.size();) {
      const bool intersects =
          std::ranges::any_of(plan.owner_descriptor_file_offsets, [&](uint64_t owner) {
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
      for (uint64_t owner : components[index]) {
        if (std::ranges::find(components[*destination], owner) == components[*destination].end())
          components[*destination].push_back(owner);
      }
      components.erase(components.begin() + static_cast<ptrdiff_t>(index));
    }
    if (!destination) {
      destination = components.size();
      components.emplace_back();
    }
    for (uint64_t owner : plan.owner_descriptor_file_offsets) {
      if (std::ranges::find(components[*destination], owner) == components[*destination].end())
        components[*destination].push_back(owner);
    }
  }
  return components;
}

struct MoiScalarOwnerContextResolution {
  std::vector<const MoiKernelResourceContext *> contexts;
  std::vector<consan_detail::ScalarOwnerContextSummary> summaries;
  uint32_t tail_floor = 0;
  uint32_t initialized_sgpr_count = 0;
  bool all_use_dynamic_stack = true;
  bool all_use_fixed_stack = true;
};

[[nodiscard]] std::optional<MoiScalarOwnerContextResolution>
resolve_moi_scalar_owner_contexts(const MoiResourcePlanningState &state,
                                  std::span<const uint64_t> owners) {
  std::vector<consan_detail::ScalarOwnerContextSummary> summaries;
  summaries.reserve(state.contexts.size());
  for (const MoiKernelResourceContext &context : state.contexts)
    summaries.push_back(context.scalar_summary());
  const auto resolved =
      consan_detail::resolve_scalar_owner_contexts(state.valid, summaries, owners);
  if (!resolved)
    return std::nullopt;

  MoiScalarOwnerContextResolution result;
  result.contexts.reserve(resolved->context_indices.size());
  result.summaries.reserve(resolved->context_indices.size());
  for (size_t context_index : resolved->context_indices) {
    result.contexts.push_back(&state.contexts[context_index]);
    result.summaries.push_back(summaries[context_index]);
    const MoiKernelResourceContext &context = state.contexts[context_index];
    result.initialized_sgpr_count =
        std::max<uint32_t>(result.initialized_sgpr_count, context.initialized_sgpr_count);
    result.all_use_dynamic_stack &= context.uses_dynamic_stack.value_or(false);
    result.all_use_fixed_stack &=
        context.uses_dynamic_stack.has_value() && !*context.uses_dynamic_stack;
  }
  result.tail_floor = resolved->tail_floor;
  return result;
}

bool moi_resource_owner_sgpr_window_admitted(const MoiResourcePlanningState &state,
                                             std::span<const uint64_t> owners, uint16_t base,
                                             uint16_t count) {
  const auto contexts = resolve_moi_scalar_owner_contexts(state, owners);
  return contexts &&
         consan_detail::scalar_owner_contexts_admit_reserved_window(
             contexts->summaries, base, count, consan_arch_is_cdna3_or_cdna4(state.arch));
}

struct MoiScalarOwnerContextGroup {
  bool supported = false;
  MoiScalarOwnerContextResolution owners;
};

struct MoiScalarOwnerContextGroupsResolution {
  bool planning_state_valid = false;
  std::vector<MoiScalarOwnerContextGroup> groups;
  std::optional<uint64_t> unresolved_owner;
};

[[nodiscard]] MoiScalarOwnerContextGroupsResolution resolve_moi_resource_plan_scalar_owner_contexts(
    const MoiResourcePlanningState &state, std::span<const ConSanCandidateResourcePlan> plans) {
  MoiScalarOwnerContextGroupsResolution resolved;
  resolved.planning_state_valid = state.valid;
  if (!state.valid)
    return resolved;
  resolved.groups.reserve(plans.size());
  for (const ConSanCandidateResourcePlan &plan : plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    auto owners = resolve_moi_scalar_owner_contexts(state, plan.owner_descriptor_file_offsets);
    if (!owners) {
      resolved.unresolved_owner = plan.owner_descriptor_file_offsets.front();
      for (uint64_t owner : plan.owner_descriptor_file_offsets) {
        const auto found = state.context_by_descriptor.find(owner);
        if (found == state.context_by_descriptor.end() || found->second == nullptr ||
            !found->second->descriptor_valid) {
          resolved.unresolved_owner = owner;
          break;
        }
      }
      return resolved;
    }
    resolved.groups.push_back({
        .supported = plan.source != ConSanRegisterAllocationSource::Unsupported,
        .owners = std::move(*owners),
    });
  }
  return resolved;
}

[[nodiscard]] MoiScalarOwnerContextResolution
collect_moi_scalar_owner_contexts(std::span<const MoiScalarOwnerContextGroup> groups,
                                  bool include_unsupported) {
  MoiScalarOwnerContextResolution result;
  for (const MoiScalarOwnerContextGroup &group : groups) {
    if (!include_unsupported && !group.supported)
      continue;
    for (size_t index = 0; index < group.owners.contexts.size(); ++index) {
      const MoiKernelResourceContext *context = group.owners.contexts[index];
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

[[nodiscard]] bool moi_scalar_contexts_conflict_with_physical_vcc(
    std::span<const consan_detail::ScalarOwnerContextSummary> contexts,
    std::span<const MoiSgprRange> ranges, rj_code_arch_t arch) {
  return consan_arch_is_cdna3_or_cdna4(arch) &&
         consan_detail::scalar_owner_contexts_conflict_with_physical_vcc(contexts, ranges);
}

[[nodiscard]] bool moi_scalar_contexts_admit_reserved_window(
    std::span<const consan_detail::ScalarOwnerContextSummary> contexts, uint16_t base,
    uint16_t width, rj_code_arch_t arch) {
  return consan_detail::scalar_owner_contexts_admit_reserved_window(
      contexts, base, width, consan_arch_is_cdna3_or_cdna4(arch));
}

/// One register-file view for an ordered scalar-placement attempt.
///
/// Fixed occupants exclude overlap but do not participate in the CDNA
/// physical-VCC proof: they were qualified by their own lifetime. Selected
/// ranges form one simultaneously live tuple and are checked together. The
/// optional unavailable set is the site-local liveness/def-use union; callers
/// explicitly choose whether a role must be dead or is preserved by spill.
class MoiScalarPlacementDomain {
public:
  MoiScalarPlacementDomain(rj_code_arch_t arch, uint16_t limit,
                           std::span<const consan_detail::ScalarOwnerContextSummary> owner_contexts,
                           const RegisterSet *unavailable = nullptr)
      : arch_(arch), limit_(limit), owner_contexts_(owner_contexts), unavailable_(unavailable) {}

  void occupy(uint16_t base, uint16_t width) {
    if (width != 0u)
      occupied_.push_back({base, width});
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
    if (std::ranges::any_of(occupied_, [&](const MoiSgprRange &range) {
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
    std::vector<consan_detail::ScalarOwnerSgprRange> combined = selected_;
    combined.push_back({base, width});
    return owner_contexts_.empty() ||
           !consan_detail::scalar_owner_contexts_conflict_with_physical_vcc(owner_contexts_,
                                                                            combined);
  }

  [[nodiscard]] std::optional<uint16_t>
  select_first(uint32_t begin, uint16_t width, uint16_t alignment = 1u, bool require_dead = true) {
    const auto selected =
        first_admitted_moi_register_range(begin, width, limit_, alignment, [&](uint16_t base) {
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
  std::span<const consan_detail::ScalarOwnerContextSummary> owner_contexts_;
  const RegisterSet *unavailable_ = nullptr;
  std::vector<MoiSgprRange> occupied_;
  std::vector<consan_detail::ScalarOwnerSgprRange> selected_;
};

void occupy_moi_transient_scalar_state(MoiScalarPlacementDomain &domain,
                                       const ConSanRequest &request,
                                       const BoundRuntimeResources &resources,
                                       const ConSanMoiOperatingPoint &point,
                                       const ConSanMoiOperatingPoint &allocation,
                                       const MoiObjectModeSemantics &mode_semantics,
                                       rj_code_arch_t arch) {
  const uint16_t exec_width = moi_exec_save_sgpr_count(
      resolve_moi_exec_save_requirement(request, resources, point, mode_semantics), arch);
  domain.occupy(point.moi_exec_save_sgpr, exec_width);
  domain.occupy(point.moi_owner_sgpr.base(), 1u);
  domain.occupy(point.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
  point.for_each_auxiliary_sgpr_range(
      [&](std::optional<uint16_t> base, uint16_t width) { domain.occupy(base, width); });
  for (const ConSanMoiTransientSgprAssignment &assignment : allocation.owner_transient_sgprs) {
    domain.occupy(assignment.exec_save_sgpr, exec_width);
    domain.occupy(assignment.owner_sgpr, 1u);
    domain.occupy(assignment.dispatch_id_sgpr, kMoiDispatchStateSgprCount);
    if (assignment.scalar_spill_setup) {
      assignment.scalar_spill_setup->for_each_sgpr_range(
          true, [&](uint16_t base, uint16_t width) { domain.occupy(base, width); });
    }
    if (assignment.branch_only_spill)
      domain.occupy(assignment.branch_only_spill->dynamic_stack_borrowed_sgpr, 2u);
  }
}

struct MoiScalarSiteAvailability {
  std::unordered_map<uint64_t, RegisterSet> unavailable_by_owner;
  RegisterSet unavailable;
  bool saw_site = false;
};

[[nodiscard]] RegisterSet
dynamic_frame_save_registers_for_site(const MoiResourcePlanningState &state, uint64_t text_offset);

[[nodiscard]] MoiScalarSiteAvailability collect_moi_scalar_site_availability(
    const MoiResourcePlanningState &state, std::span<const ConSanCandidateResourcePlan> site_plans,
    const ConSanObservationPlan &observation_plan, const ProgramInventory &inventory,
    bool include_kernel_entries) {
  MoiScalarSiteAvailability result;
  std::unordered_map<BasicBlock *, std::vector<const MoiKernelResourceContext *>> owners_by_block;
  for (const MoiKernelResourceContext &context : state.contexts) {
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
    for (const MoiKernelResourceContext *context : owners->second) {
      RegisterSet site_unavailable = context->liveness->live_before(*instruction);
      const InstDefUse def_use(*instruction);
      site_unavailable |= def_use.defs | def_use.uses;
      if (context->kernel != nullptr && context->kernel->uses_dynamic_stack.value_or(false))
        site_unavailable |= dynamic_frame_save_registers_for_site(state, text_offset);
      result.unavailable |= site_unavailable;
      if (context->kernel != nullptr)
        result.unavailable_by_owner[context->kernel->descriptor_file_offset] |= site_unavailable;
    }
  };
  for (const ConSanCandidateResourcePlan &plan : site_plans)
    include(plan.text_offset);
  for (const ConSanProbeIntent &intent : observation_plan.probe_intents)
    include(intent.physical_site.original_text_offset);
  if (include_kernel_entries) {
    for (const ConSanProgramContainer &kernel : inventory.kernels())
      include(kernel.entry_text_offset);
  }
  return result;
}

[[nodiscard]] std::optional<std::vector<ConSanMoiPrologueScratchVgprAssignment>>
plan_moi_scalar_prologue_scratch(const MoiResourcePlanningState &state,
                                 const MoiScalarOwnerContextResolution &resolution,
                                 bool needs_entry_vgpr_pair) {
  std::vector<ConSanMoiPrologueScratchVgprAssignment> assignments;
  assignments.reserve(resolution.contexts.size());
  for (const MoiKernelResourceContext *context : resolution.contexts) {
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

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_candidate(std::span<const ConSanCandidateResourcePlan> plans,
                            const ConSanMoiCandidate &candidate);

/// Returns the merged original-text ranges occupied by synchronization
/// instructions that access-engine relocation must preserve.
///
/// Barrier, atomic, and fence inventories are deliberately projected through
/// one query so Record/Replay, Sampled, and Inline Shadow cannot silently
/// disagree about which synchronization instructions are unavailable as
/// relocation donors. Empty sites are ignored and adjacent ranges are merged;
/// consumers may therefore also use the result for logarithmic overlap tests.

[[nodiscard]] RegisterSet
dynamic_frame_save_registers_for_site(const MoiResourcePlanningState &state, uint64_t text_offset) {
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

void forbid_moi_persistent_vgprs(RegisterSet &forbidden, const ConSanMoiOperatingPoint &point,
                                 const ConSanMoiOperatingPoint &allocation,
                                 std::span<const uint64_t> owner_descriptors) {
  const auto forbid_vgpr = [&](std::optional<uint16_t> reg, uint16_t count) {
    if (reg)
      forbidden.expand({RegClass::VGPR, *reg, static_cast<uint8_t>(count)});
  };
  const auto forbid_state = [&](const MoiPersistentVgprStateView &state) {
    state.for_each_range(forbid_vgpr);
  };
  forbid_state(moi_persistent_vgpr_state_view(point));
  for (uint64_t descriptor_offset : owner_descriptors) {
    const auto *assignment = moi_persistent_vgpr_assignment(allocation, descriptor_offset);
    if (assignment)
      forbid_state(moi_persistent_vgpr_state_view(*assignment));
  }
}

[[nodiscard]] ConSanCandidateResourcePlan
plan_moi_resource_site(MoiResourcePlanningState &state, const ConSanRequest &semantic_request,
                       const ConSanDebugOverrides &debug, const ConSanMoiOperatingPoint &point,
                       ConSanResourceSiteKind site_kind, size_t candidate_index,
                       uint64_t text_offset, std::optional<ConSanProgramSiteId> owner_site,
                       uint16_t scratch_count, const ConSanMoiCandidate *access_candidate,
                       const ConSanAtomicLoweringForm *atomic_form, bool require_spill) {
  ConSanCandidateResourcePlan plan;
  plan.site_kind = site_kind;
  plan.candidate_index = candidate_index;
  plan.text_offset = text_offset;
  if (access_candidate != nullptr)
    plan.intent_ids = access_candidate->intent_ids;
  plan.scratch_vgpr_count = scratch_count;
  if (!state.valid) {
    plan.reason = ConSanRegisterPlanReason::MissingInstruction;
    return plan;
  }

  BasicBlock *anchor_block = block_for_offset(state.block_index, text_offset);
  const Instruction *anchor = instruction_at(anchor_block, text_offset);
  if (anchor == nullptr) {
    plan.reason = ConSanRegisterPlanReason::MissingInstruction;
    return plan;
  }

  std::vector<MoiKernelResourceContext *> owners;
  const auto add_owner = [&](MoiKernelResourceContext &context) {
    if (!context.block_set.contains(anchor_block))
      return;
    owners.push_back(&context);
    plan.owner_descriptor_file_offsets.push_back(context.kernel->descriptor_file_offset);
  };
  const auto add_execution_owners = [&](std::span<const ConSanExecutionOwner> execution_owners) {
    for (const ConSanExecutionOwner &execution_owner : execution_owners) {
      const auto owner = state.context_by_kernel_id.find(execution_owner.kernel.ordinal);
      if (owner != state.context_by_kernel_id.end())
        add_owner(*owner->second);
    }
  };
  if (access_candidate != nullptr) {
    add_execution_owners(access_candidate->execution_owners);
  } else if (owner_site) {
    const ConSanProgramSite *site = state.inventory.program_site(*owner_site);
    if (site != nullptr)
      add_execution_owners(site->execution_owners);
  } else {
    for (MoiKernelResourceContext &context : state.contexts)
      add_owner(context);
  }
  if (owners.empty()) {
    plan.reason = ConSanRegisterPlanReason::MissingOwner;
    return plan;
  }
  const bool owners_have_spill_backed_scalar_state =
      !plan.owner_descriptor_file_offsets.empty() &&
      std::ranges::all_of(plan.owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
        const ConSanMoiTransientSgprAssignment *assignment =
            moi_transient_sgpr_assignment(point, descriptor_offset);
        return assignment != nullptr && assignment->spill_backed;
      });
  const bool fixed_lane_scalar_reservoir =
      ((point.moi_exec_save_sgpr && point.has_moi_scalar_spill()) ||
       owners_have_spill_backed_scalar_state) &&
      std::ranges::none_of(owners, [](const MoiKernelResourceContext *owner) {
        return owner->kernel->uses_dynamic_stack.value_or(false);
      });
  if (fixed_lane_scalar_reservoir && scratch_count == REGISTER_SET_MAX_VGPRS) {
    plan.reason = ConSanRegisterPlanReason::NoLegalWindow;
    return plan;
  }
  // Fixed-stack spill-backed scalar state lives in one VGPR that the engine
  // never uses as scratch. Include it in allocation itself so liveness,
  // persistent-state exclusions, guest operands, and descriptor growth all
  // cover the real clobber range rather than discovering an adjacent register
  // after planning.
  const uint16_t allocation_count =
      static_cast<uint16_t>(scratch_count + static_cast<uint16_t>(fixed_lane_scalar_reservoir));
  plan.current_vgpr_count = kMaxVgprs;
  plan.ordinary_vgpr_limit = kMaxVgprs;
  plan.current_sgpr_count = kMaxSgprs;
  bool every_owner_can_move_empty_accumulator_boundary = consan_arch_is_cdna3_or_cdna4(state.arch);
  bool every_owner_accumulator_boundary_is_allocation_tail =
      every_owner_can_move_empty_accumulator_boundary;
  RegisterSet live_before_all_owners;
  bool descriptors_valid = true;
  for (MoiKernelResourceContext *owner : owners) {
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
    plan.reason = ConSanRegisterPlanReason::InvalidDescriptor;
    return plan;
  }

  const InstDefUse anchor_def_use(*anchor);
  ConSanRegisterRequest request;
  request.reg_class = RegClass::VGPR;
  request.count = allocation_count;
  // Every ConSan engine uses the first two scratch registers as a FLAT
  // address pair. CDNA3/4 require that pair to begin at an even VGPR.
  request.alignment = consan_arch_is_cdna3_or_cdna4(state.arch) ? 2u : 1u;
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
    const ConSanAccessLoweringForm *form =
        access_candidate->lowering.form ? &*access_candidate->lowering.form : nullptr;
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
  if (!point.moi_owner_epoch_vgprs.owner() &&
      state.register_allocation.owner_persistent_vgprs.empty() &&
      semantic_request.moi_owner_source == ConSanMoiOwnerSource::WorkitemId)
    request.forbidden.expand({RegClass::VGPR, kAmdGpuWorkitemIdX, 1});
  forbid_moi_persistent_vgprs(request.forbidden, point, state.register_allocation,
                              plan.owner_descriptor_file_offsets);

  ConSanRegisterPlan register_plan = plan_consan_registers(request, live_before_all_owners);
  if (register_plan.source == ConSanRegisterAllocationSource::Unsupported &&
      register_plan.reason == ConSanRegisterPlanReason::NoLegalWindow &&
      every_owner_can_move_empty_accumulator_boundary && !request.explicit_base &&
      !request.force_spill) {
    // Trusted metadata can prove that a rounded unified-VGPR tail contains no
    // accumulator state even when ACCUM_OFFSET conservatively places the
    // ordinary-register boundary below it. Retry descriptor growth only after
    // ordinary placement fails. This keeps already-supported kernels on their
    // original allocation while avoiding hot-path private-memory spills for
    // the otherwise unplaceable sites.
    ConSanRegisterRequest growth_request = request;
    growth_request.architecture_limit = kMaxVgprs;
    growth_request.allow_spill = false;
    register_plan = plan_consan_registers(growth_request, live_before_all_owners);
    plan.alternatives.push_back({
        .kind = ConSanResourcePlanAlternativeKind::EmptyAccumulatorDescriptorGrowth,
        .source = register_plan.source,
        .reason = register_plan.reason,
        .scratch_vgpr_count = scratch_count,
        .outcome = register_plan.source == ConSanRegisterAllocationSource::DescriptorGrowth
                       ? ConSanResourcePlanAlternativeOutcome::Selected
                       : ConSanResourcePlanAlternativeOutcome::Rejected,
    });
  }
  const MoiAccessResourceFacts access_resource_facts =
      access_candidate != nullptr
          ? resolve_moi_access_resource_facts(point, *access_candidate, state.arch)
          : MoiAccessResourceFacts{.target_available = consan_is_capability_arch(state.arch)};
  const MoiOperandOverlapSpillPolicy operand_overlap_spill = plan_moi_operand_overlap_spill({
      semantic_request,
      access_resource_facts,
      access_candidate,
      site_kind,
  });
  if (register_plan.source == ConSanRegisterAllocationSource::Unsupported &&
      register_plan.reason == ConSanRegisterPlanReason::NoLegalWindow &&
      operand_overlap_spill.supported && !request.explicit_base) {
    // These appended-cave bodies preserve guest operands even when no
    // disjoint ordinary-VGPR window exists. Sampled atomics and the admitted
    // Sampled stores execute the guest before clobbering an overlapping saved
    // window; Record/Replay access bodies restore the window before the
    // displaced guest. Persistent state remains forbidden because it must
    // survive the complete kernel lifetime. Shared physical code can have
    // owners with different descriptor allocations, so this capability-gated
    // fallback may also grow the smaller descriptors to hold the spill window.
    ConSanRegisterRequest spill_request = request;
    spill_request.force_spill = true;
    spill_request.allow_spill_descriptor_growth = true;
    if (every_owner_can_move_empty_accumulator_boundary)
      spill_request.architecture_limit = kMaxVgprs;
    spill_request.forbidden = {};
    forbid_moi_persistent_vgprs(spill_request.forbidden, point, state.register_allocation,
                                plan.owner_descriptor_file_offsets);
    if (operand_overlap_spill.protected_vgpr) {
      spill_request.forbidden.expand({RegClass::VGPR, *operand_overlap_spill.protected_vgpr,
                                      operand_overlap_spill.protected_vgpr_count});
    }
    register_plan = plan_consan_registers(spill_request, live_before_all_owners);
    plan.alternatives.push_back({
        .kind = ConSanResourcePlanAlternativeKind::GuestOperandOverlapSpill,
        .source = register_plan.source,
        .reason = register_plan.reason,
        .scratch_vgpr_count = scratch_count,
        .outcome = register_plan.source == ConSanRegisterAllocationSource::Unsupported
                       ? ConSanResourcePlanAlternativeOutcome::Rejected
                       : ConSanResourcePlanAlternativeOutcome::Selected,
    });
  }
  plan.source = register_plan.source;
  plan.reason = register_plan.reason;
  plan.scratch_vgpr = register_plan.base;
  plan.required_vgpr_count = register_plan.required_descriptor_count;
  if (plan.source == ConSanRegisterAllocationSource::SpillRequired) {
    const bool has_dynamic_owner =
        std::ranges::any_of(owners, [](const MoiKernelResourceContext *owner) {
          return owner->kernel->uses_dynamic_stack.value_or(false);
        });
    const bool every_owner_dynamic =
        std::ranges::all_of(owners, [](const MoiKernelResourceContext *owner) {
          return owner->kernel->uses_dynamic_stack.value_or(false);
        });
    const MoiDynamicStackSpillPolicy dynamic_stack_spill =
        plan_moi_dynamic_stack_spill(semantic_request.moi_engine, state.arch);
    const bool supported_dynamic_spill =
        dynamic_stack_spill.backend_supported &&
        (!dynamic_stack_spill.requires_every_owner_dynamic || every_owner_dynamic);
    if (has_dynamic_owner && !supported_dynamic_spill) {
      plan.source = ConSanRegisterAllocationSource::Unsupported;
      plan.reason = ConSanRegisterPlanReason::DynamicStack;
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
void adopt_moi_resource_plan_fallback(ConSanCandidateResourcePlan &plan,
                                      ConSanCandidateResourcePlan fallback,
                                      ConSanResourcePlanAlternativeKind kind,
                                      uint16_t scratch_count) {
  const bool selected = fallback.source == ConSanRegisterAllocationSource::SpillRequired;
  ConSanResourcePlanAlternative outer{
      .kind = kind,
      .source = fallback.source,
      .reason = fallback.reason,
      .scratch_vgpr_count = scratch_count,
      .outcome = selected ? ConSanResourcePlanAlternativeOutcome::Selected
                          : ConSanResourcePlanAlternativeOutcome::Rejected,
  };
  std::vector<ConSanResourcePlanAlternative> initial = std::move(plan.alternatives);
  std::vector<ConSanResourcePlanAlternative> nested = std::move(fallback.alternatives);
  if (selected) {
    for (ConSanResourcePlanAlternative &alternative : initial) {
      if (alternative.outcome == ConSanResourcePlanAlternativeOutcome::Selected)
        alternative.outcome = ConSanResourcePlanAlternativeOutcome::Superseded;
    }
  }
  for (ConSanResourcePlanAlternative &alternative : nested) {
    if (alternative.outcome == ConSanResourcePlanAlternativeOutcome::Selected) {
      alternative.outcome = selected ? ConSanResourcePlanAlternativeOutcome::Contributed
                                     : ConSanResourcePlanAlternativeOutcome::Vetoed;
    }
  }
  std::vector<ConSanResourcePlanAlternative> trace;
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

void append_moi_resource_plans(MoiResourcePlanningState &state, const ConSanRequest &request,
                               const BoundRuntimeResources &resources,
                               const ConSanDebugOverrides &debug,
                               const ConSanMoiOperatingPoint &point,
                               std::span<const ConSanMoiCandidate> candidates,
                               std::vector<ConSanCandidateResourcePlan> &plans) {
  plans.reserve(plans.size() + candidates.size());
  for (size_t candidate_index = 0; candidate_index < candidates.size(); ++candidate_index) {
    const ConSanMoiCandidate &candidate = candidates[candidate_index];
    ConSanMoiOperatingPoint site_point = point;
    MoiAccessResourceFacts site_resource_facts =
        resolve_moi_access_resource_facts(site_point, candidate, state.arch);
    uint16_t scratch_count =
        moi_mode_operations(request.moi_engine)
            .access_scratch_vgpr_count(request, resources, site_resource_facts, candidate);
    ConSanCandidateResourcePlan plan = plan_moi_resource_site(
        state, request, debug, site_point, ConSanResourceSiteKind::Access, candidate_index,
        candidate.anchor(), std::nullopt, scratch_count, &candidate);
    // Persistent placement can deliberately mix private state with
    // component-local VGPR tuples. The first plan discovers the physical
    // site's complete owner set; size and place its final scratch window for
    // that owner's actual ABI, rather than the code-object-wide fallback.
    ConSanMoiOperatingPoint owner_point = site_point;
    if (apply_moi_persistent_vgpr_assignment(owner_point, state.register_allocation,
                                             plan.owner_descriptor_file_offsets)) {
      const MoiAccessResourceFacts owner_resource_facts =
          resolve_moi_access_resource_facts(owner_point, candidate, state.arch);
      const uint16_t owner_scratch_count =
          moi_mode_operations(request.moi_engine)
              .access_scratch_vgpr_count(request, resources, owner_resource_facts, candidate);
      if (owner_scratch_count != scratch_count) {
        scratch_count = owner_scratch_count;
        plan = plan_moi_resource_site(state, request, debug, owner_point,
                                      ConSanResourceSiteKind::Access, candidate_index,
                                      candidate.anchor(), std::nullopt, scratch_count, &candidate);
      }
      site_resource_facts = owner_resource_facts;
      site_point = std::move(owner_point);
    }
    bool initial_spill_overlaps_guest = false;
    if (plan.source == ConSanRegisterAllocationSource::SpillRequired && plan.scratch_vgpr) {
      std::vector<std::string> ignored_overlap_errors;
      initial_spill_overlaps_guest = reject_candidate_scratch_range_overlap(
          candidate, *plan.scratch_vgpr, plan.scratch_vgpr_count, ignored_overlap_errors);
    }
    const bool no_ordinary_window = plan.source == ConSanRegisterAllocationSource::Unsupported &&
                                    plan.reason == ConSanRegisterPlanReason::NoLegalWindow;
    const auto fallback_planner = moi_mode_operations(request.moi_engine).access_spill_fallback;
    const std::optional<uint16_t> spill_scratch_count =
        fallback_planner
            ? fallback_planner({request, site_resource_facts, candidate, no_ordinary_window,
                                plan.source == ConSanRegisterAllocationSource::SpillRequired,
                                initial_spill_overlaps_guest})
            : std::nullopt;
    if (spill_scratch_count) {
      ConSanCandidateResourcePlan spill_plan = plan_moi_resource_site(
          state, request, debug, site_point, ConSanResourceSiteKind::Access, candidate_index,
          candidate.anchor(), std::nullopt, *spill_scratch_count, &candidate,
          /*atomic_form=*/nullptr, /*require_spill=*/true);
      adopt_moi_resource_plan_fallback(
          plan, std::move(spill_plan),
          ConSanResourcePlanAlternativeKind::SpillBackedOperandRecovery, *spill_scratch_count);
    }
    plans.push_back(std::move(plan));
  }
}

bool moi_resource_offsets_share_block(const MoiResourcePlanningState &state, uint64_t first_offset,
                                      uint64_t second_offset) {
  BasicBlock *first = block_for_offset(state.block_index, first_offset);
  return first != nullptr && block_for_offset(state.block_index, second_offset) == first;
}

/// contiguous scratch tuple and is proven against every owning kernel scope.
[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_candidate(std::span<const ConSanCandidateResourcePlan> plans,
                            const ConSanMoiCandidate &candidate) {
  const auto plan = std::ranges::find_if(plans, [&](const auto &candidate_plan) {
    return candidate_plan.site_kind == ConSanResourceSiteKind::Access &&
           candidate_plan.text_offset == candidate.anchor();
  });
  return plan == plans.end() ? nullptr : &*plan;
}

[[nodiscard]] const ConSanCandidateResourcePlan *
resource_plan_for_site(std::span<const ConSanCandidateResourcePlan> plans,
                       ConSanResourceSiteKind site_kind, uint64_t text_offset) {
  const auto it = std::ranges::find_if(plans, [&](const auto &plan) {
    return plan.site_kind == site_kind &&
           plan.semantic_text_offset.value_or(plan.text_offset) == text_offset;
  });
  return it == plans.end() ? nullptr : &*it;
}

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch_plan(const ConSanCandidateResourcePlan &plan,
                         const ConSanMoiOperatingPoint &site_point, MoiOwnerAssignments assignments,
                         uint16_t expected_count) {
  if (plan.scratch_vgpr && plan.scratch_vgpr_count == expected_count &&
      (plan.source == ConSanRegisterAllocationSource::Explicit ||
       plan.source == ConSanRegisterAllocationSource::LivenessDead ||
       plan.source == ConSanRegisterAllocationSource::DescriptorGrowth ||
       plan.source == ConSanRegisterAllocationSource::SpillRequired)) {
    ResolvedMoiScratchPlan resolved;
    resolved.base = *plan.scratch_vgpr;
    resolved.count = expected_count;
    resolved.required_vgpr_count = plan.required_vgpr_count;
    resolved.original_private_segment_size = plan.original_private_segment_size;
    resolved.owner_descriptor_file_offsets = plan.owner_descriptor_file_offsets;
    resolved.source = plan.source;
    const auto include_in_required_count = [&](std::optional<uint16_t> reg, uint16_t width) {
      if (reg) {
        resolved.required_vgpr_count =
            std::max<uint16_t>(resolved.required_vgpr_count, static_cast<uint16_t>(*reg + width));
      }
    };
    moi_persistent_vgpr_state_view(site_point).for_each_range(include_in_required_count);
    for (uint64_t descriptor_offset : plan.owner_descriptor_file_offsets) {
      const auto *assignment = moi_persistent_vgpr_assignment(assignments, descriptor_offset);
      if (!assignment)
        continue;
      moi_persistent_vgpr_state_view(*assignment).for_each_range(include_in_required_count);
    }
    return resolved;
  }

  return std::nullopt;
}

[[nodiscard]] std::optional<ResolvedMoiScratchPlan>
resolve_moi_scratch(std::span<const ConSanCandidateResourcePlan> plans,
                    const ConSanMoiCandidate &candidate,
                    const ConSanMoiOperatingPoint &operating_point, uint16_t expected_count) {
  const ConSanCandidateResourcePlan *plan = resource_plan_for_candidate(plans, candidate);
  return plan == nullptr
             ? std::nullopt
             : resolve_moi_scratch_plan(*plan, operating_point, operating_point, expected_count);
}

[[nodiscard]] bool configure_automatic_moi_transient_sgpr_assignments(
    ConSanMoiOperatingPoint &options, const MoiResourceProblem &problem,
    std::span<const ConSanCandidateResourcePlan> site_plans, std::vector<std::string> &warnings,
    const MoiResourcePlanningState &planning_state, bool dispatch_only = false) {
  std::vector<ConSanMoiTransientSgprAssignment> &owner_transient_sgprs =
      options.owner_transient_sgprs;
  const rj_code_arch_t arch = planning_state.arch;
  const ConSanRequest &request = problem.request();
  const BoundRuntimeResources &resources = problem.resources();
  const MoiTransientScalarPlacementTraits &placement =
      moi_mode_operations(request.moi_engine).transient_scalar_placement;
  const bool capability_target = consan_is_capability_arch(arch);
  const bool placement_enabled = !placement.requires_capability_target || capability_target;
  const bool inline_layout = placement.spill_layout == ConSanMoiScalarSpillLayout::Inline;
  const bool compact_layout = placement.spill_layout == ConSanMoiScalarSpillLayout::Compact;
  if (!placement_enabled || (!options.automatic_moi_partial_exec_save_sgprs && !dispatch_only) ||
      !options.moi_exec_save_sgpr)
    return false;

  const MoiResourcePlanningState &state = planning_state;
  if (!state.valid)
    return false;
  const std::vector<std::vector<uint64_t>> owner_components =
      moi_scalar_owner_components(site_plans);
  // Inline's component-local transient window is established only inside an
  // access or synchronization body. Its frame setup uses separate scalar
  // temporaries, while persistent owner/epoch values live in VGPRs. Do not
  // turn entry ABI inputs into false live-through constraints for that
  // site-local window.
  const MoiScalarSiteAvailability availability = collect_moi_scalar_site_availability(
      state, site_plans, problem.observation_plan(), problem.inventory(),
      options.moi_initialize_owner_epoch && !inline_layout);

  const uint16_t count = moi_exec_save_sgpr_count(
      resolve_moi_exec_save_requirement(request, resources, options, problem.mode_semantics()),
      arch);
  const uint16_t limit = moi_ordinary_sgpr_limit(arch);

  bool assigned_any = false;
  bool missed_required_assignment = false;
  for (const std::vector<uint64_t> &owners : owner_components) {
    const auto component_contexts = resolve_moi_scalar_owner_contexts(state, owners);
    if (!component_contexts) {
      missed_required_assignment = true;
      continue;
    }
    const uint32_t liveness_search_floor =
        std::ranges::any_of(component_contexts->summaries,
                            [](const auto &context) {
                              return consan_detail::scalar_owner_tail_floor(context) >
                                     context.max_referenced_sgpr_count;
                            })
            ? component_contexts->tail_floor
            : 0u;
    const bool needs_exec_override = !moi_scalar_contexts_admit_reserved_window(
        component_contexts->summaries, *options.moi_exec_save_sgpr, count, arch);
    const bool needs_dispatch_override =
        options.moi_dispatch_identity.sgpr() &&
        (!moi_scalar_contexts_admit_reserved_window(component_contexts->summaries,
                                                    *options.moi_dispatch_identity.sgpr(),
                                                    kMoiDispatchStateSgprCount, arch) ||
         range_overlaps(*options.moi_exec_save_sgpr, count, *options.moi_dispatch_identity.sgpr(),
                        kMoiDispatchStateSgprCount));
    const bool needs_override =
        dispatch_only ? needs_dispatch_override
                      : needs_exec_override || (inline_layout && needs_dispatch_override);
    if (!needs_override)
      continue;
    RegisterSet unavailable;
    for (uint64_t owner : owners) {
      if (const auto it = availability.unavailable_by_owner.find(owner);
          it != availability.unavailable_by_owner.end())
        unavailable |= it->second;
    }
    // A code-object-wide EXEC window may already have a complete liveness
    // proof while only the persistent dispatch pair excludes this component.
    // Retain that transient window and override just the dispatch
    // representation rather than inventing a second scalar ABI.
    std::optional<uint16_t> selected;
    if ((dispatch_only || (!needs_exec_override && needs_dispatch_override)) &&
        options.moi_exec_save_sgpr) {
      selected = options.moi_exec_save_sgpr;
    }
    std::optional<uint16_t> spill_frame_base;
    std::optional<uint16_t> spill_scc;
    std::optional<uint16_t> spill_visible;
    std::optional<uint16_t> inline_dead_fallback;
    bool branch_only_scalar_spill = false;
    std::optional<uint16_t> dynamic_stack_borrowed_sgpr;
    const bool all_owners_use_dynamic_stack =
        !owners.empty() && component_contexts->all_use_dynamic_stack;
    const bool all_owners_use_fixed_stack =
        !owners.empty() && component_contexts->all_use_fixed_stack;
    const bool rdna_branch_only = inline_layout && consan_arch_is_rdna4_or_cdna5(arch) &&
                                  (all_owners_use_dynamic_stack || all_owners_use_fixed_stack);
    if (!selected && compact_layout) {
      MoiScalarPlacementDomain domain(arch, limit, component_contexts->summaries, &unavailable);
      domain.occupy(options.moi_owner_sgpr.base(), 1u);
      domain.occupy(options.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
      if (options.moi_dynamic_stack_spill)
        domain.occupy_dynamic_stack_state();
      const auto candidate = domain.select_first(liveness_search_floor, count, 2u);
      if (candidate)
        selected = candidate;
    }
    if (!selected && inline_layout && capability_target &&
        consan_address_free_private_limit(arch)) {
      const uint32_t initialized_end = component_contexts->initialized_sgpr_count;
      MoiScalarPlacementDomain component_domain(arch, limit, component_contexts->summaries,
                                                &unavailable);
      component_domain.occupy(options.moi_owner_sgpr.base(), 1u);
      component_domain.occupy(options.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
      if (options.moi_dynamic_stack_spill)
        component_domain.occupy_dynamic_stack_state();
      if (consan_arch_is_cdna3_or_cdna4(arch))
        component_domain.occupy(0u, static_cast<uint16_t>(initialized_end));
      const uint32_t fresh_begin = consan_arch_is_cdna3_or_cdna4(arch)
                                       ? std::max(initialized_end, component_contexts->tail_floor)
                                       : component_contexts->tail_floor;
      // The Inline scalar window is saved after the VGPR frame is established
      // and restored before that frame is released. RDNA4 may therefore
      // borrow initialized guest SGPRs; only the frame-bootstrap temporaries
      // must remain liveness-dead. CDNA retains its stricter preload boundary.
      const uint32_t scalar_spill_begin = consan_arch_is_rdna4_or_cdna5(arch)
                                              ? 0u
                                              : std::max(initialized_end, liveness_search_floor);
      for (uint32_t base = util::align_up(scalar_spill_begin, 2u);
           !selected && base + count <= limit; base += 2u) {
        const uint16_t spill_base = static_cast<uint16_t>(base);
        MoiScalarPlacementDomain attempt = component_domain;
        if (!attempt.select(spill_base, count, /*require_dead=*/false))
          continue;
        spill_visible = attempt.select_first(fresh_begin, 1u);
        spill_frame_base = attempt.select_first(liveness_search_floor, 2u, 2u);
        if (spill_frame_base)
          spill_scc = attempt.select_first(liveness_search_floor, 1u);
        if (spill_scc)
          selected = spill_base;
        else
          spill_visible = spill_frame_base = std::nullopt;
      }
      // A full-pressure owner may have no scalar register that remains dead
      // from an access anchor through the complete probe. Preserve the
      // complete Inline window at entry instead. Prefer this path to a
      // site-local dead window because the latter cannot retain the
      // entry-to-probe visible-evidence latch and therefore cannot preserve
      // local LDS exact-shadow eligibility.
      // Dynamic-stack owners additionally need one ordinary bootstrap pair
      // while establishing their private frame. Every choice stays inside
      // every owner's existing allocation and is derived from
      // descriptor/runtime ABI state rather than a fixed range.
      // Fixed-stack RDNA4-family owners prefer entry-time scalar backup even
      // when probe-local temporaries happen to fit: an empty EXEC cannot use
      // per-lane private storage. The prologue backup remains valid for both
      // empty and partially active waves.
      if (consan_arch_is_rdna4_or_cdna5(arch) && all_owners_use_fixed_stack) {
        selected.reset();
        spill_frame_base.reset();
        spill_scc.reset();
        spill_visible.reset();
      }
      // If no complete temporary set exists, an owner-local window that is
      // dead at every instrumentation site remains useful for external-shadow
      // probes.  It cannot carry the entry-to-probe visible-evidence latch;
      // access selection therefore keeps such owners off the local LDS mirror.
      if (!selected) {
        const auto candidate = component_domain.select_first(
            std::max(initialized_end, liveness_search_floor), count, 2u);
        if (rdna_branch_only)
          inline_dead_fallback = candidate;
        else if (candidate)
          selected = candidate;
      }
    }
    const bool supports_compact_scalar_spill = compact_layout && capability_target;
    if (!selected && supports_compact_scalar_spill && consan_address_free_private_limit(arch)) {
      MoiScalarPlacementDomain domain(arch, limit, component_contexts->summaries, &unavailable);
      domain.occupy(options.moi_owner_sgpr.base(), 1u);
      domain.occupy(options.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
      if (options.moi_dynamic_stack_spill)
        domain.occupy_dynamic_stack_state();
      // This path preserves every borrowed guest SGPR in address-free private
      // storage before using it. Unlike a merely liveness-dead window, it is
      // safe below a scalar-relative owner's allocation tail.
      spill_frame_base = domain.select_first(0u, 2u, 2u);
      if (spill_frame_base)
        spill_scc = domain.select_first(0u, 1u);
      const uint16_t spilled_scalar_count =
          placement.compact_spill_scalar_count ? placement.compact_spill_scalar_count : count;
      if (spill_scc) {
        const auto spill_window =
            domain.select_first(0u, spilled_scalar_count, 2u, /*require_dead=*/false);
        if (spill_window)
          selected = spill_window;
      }
    }
    // Fixed-stack Inline/compact owners can preserve the complete borrowed
    // window in private state. RDNA dynamic-stack owners additionally reserve
    // one bootstrap pair.
    const bool cdna_branch_only =
        supports_compact_scalar_spill && consan_arch_is_cdna(arch) && all_owners_use_fixed_stack;
    if (!selected && consan_address_free_private_limit(arch) &&
        (rdna_branch_only || cdna_branch_only)) {
      uint16_t allocated_limit = limit;
      for (const auto &summary : component_contexts->summaries)
        allocated_limit = std::min(allocated_limit, summary.current_sgpr_count);
      MoiScalarPlacementDomain branch_domain(arch, allocated_limit, component_contexts->summaries);
      if (rdna_branch_only) {
        branch_domain.occupy(options.moi_owner_sgpr.base(), 1u);
        branch_domain.occupy(options.moi_scalar_spill_setup
                                 ? options.moi_scalar_spill_setup->visible_evidence_sgpr
                                 : std::nullopt,
                             1u);
      }
      branch_domain.occupy_dynamic_stack_state();
      branch_domain.occupy(options.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
      options.moi_persistent_sgprs.for_each_range(
          [&](std::optional<uint16_t> reg, uint16_t width) { branch_domain.occupy(reg, width); });
      for (uint32_t base = 0u; !selected && base + count <= allocated_limit; base += 2u) {
        MoiScalarPlacementDomain attempt = branch_domain;
        if (!attempt.select(static_cast<uint16_t>(base), count, /*require_dead=*/false))
          continue;
        if (rdna_branch_only && all_owners_use_dynamic_stack) {
          dynamic_stack_borrowed_sgpr = attempt.select_first(0u, 2u, 2u, /*require_dead=*/false);
          if (!dynamic_stack_borrowed_sgpr)
            continue;
        }
        selected = static_cast<uint16_t>(base);
      }
      branch_only_scalar_spill = selected.has_value();
      if (cdna_branch_only && selected && placement.branch_only_spill_preserves_setup_state &&
          count >= 4u) {
        spill_frame_base = static_cast<uint16_t>(*selected + 2u);
        spill_scc = selected;
      } else {
        spill_frame_base.reset();
        spill_scc.reset();
      }
      spill_visible.reset();
    }
    if (!selected)
      selected = inline_dead_fallback;
    if (!selected) {
      missed_required_assignment = true;
      continue;
    }
    const uint16_t selected_sgpr = *selected;
    const bool spill_backed = (spill_frame_base && spill_scc) || branch_only_scalar_spill ||
                              dynamic_stack_borrowed_sgpr.has_value();
    const std::optional<uint16_t> dispatch_id_sgpr =
        options.moi_dispatch_identity.sgpr() &&
                moi_scalar_contexts_admit_reserved_window(component_contexts->summaries,
                                                          *options.moi_dispatch_identity.sgpr(),
                                                          kMoiDispatchStateSgprCount, arch) &&
                !range_overlaps(selected_sgpr, count, *options.moi_dispatch_identity.sgpr(),
                                kMoiDispatchStateSgprCount)
            ? options.moi_dispatch_identity.sgpr()
            : std::nullopt;
    for (uint64_t owner : owners) {
      owner_transient_sgprs.push_back(
          {.descriptor_file_offset = owner,
           .exec_save_sgpr = selected_sgpr,
           .owner_sgpr =
               inline_layout && request.moi_owner_source == ConSanMoiOwnerSource::HwId &&
                       (!options.moi_owner_sgpr.base() || options.moi_owner_sgpr.automatic())
                   ? std::optional<uint16_t>(selected_sgpr)
                   : options.moi_owner_sgpr.base(),
           .dispatch_id_sgpr = dispatch_id_sgpr,
           .spill_backed = spill_backed,
           .scalar_spill_setup =
               spill_frame_base && spill_scc
                   ? std::optional{ConSanMoiScalarSpillSetup{
                         .temporaries =
                             ConSanMoiScalarSpillTemporaries{*spill_frame_base, *spill_scc},
                         .visible_evidence_sgpr = spill_visible}}
                   : std::nullopt,
           .branch_only_spill =
               branch_only_scalar_spill
                   ? std::optional{ConSanMoiBranchOnlyScalarSpill{dynamic_stack_borrowed_sgpr}}
                   : std::nullopt});
    }
    assigned_any = true;
  }
  const bool can_exclude_incompatible_component = consan_arch_is_cdna3_or_cdna4(arch);
  if (missed_required_assignment && !can_exclude_incompatible_component) {
    owner_transient_sgprs.clear();
    return false;
  }
  if (!assigned_any)
    return false;
  // A code-object-wide dispatch pair may be legal for one disconnected owner
  // and overlap another owner's guest scalar state. Preserve that partial
  // result: fixed-stack owners without the pair can capture it in their
  // private entry state, while dynamic-stack owners receive an owner-local
  // persistent VGPR pair during the later persistent-state placement pass.
  // The marker is intentionally set only after component assignment has
  // proved which owners cannot use the global pair.
  if (inline_layout && options.moi_dispatch_identity.sgpr() &&
      std::ranges::any_of(owner_transient_sgprs,
                          [](const ConSanMoiTransientSgprAssignment &assignment) {
                            return !assignment.dispatch_id_sgpr;
                          })) {
    options.moi_dispatch_identity.set_private_fallback(true);
    warnings.emplace_back(
        "ConSan MOI deferred incompatible owner-local dispatch identity to private or "
        "persistent VGPR state");
  }
  warnings.emplace_back(
      "ConSan MOI assigned owner-local transient SGPR windows to independent owner components");
  return true;
}

[[nodiscard]] bool configure_automatic_moi_exec_save_sgprs(
    ConSanMoiOperatingPoint &options, const MoiResourceProblem &problem,
    std::span<const ConSanCandidateResourcePlan> site_plans, std::vector<std::string> &warnings,
    const MoiResourcePlanningState &planning_state) {
  const rj_code_arch_t arch = problem.arch();
  const ConSanRequest &request = problem.request();
  const BoundRuntimeResources &resources = problem.resources();
  const MoiTransientScalarPlacementTraits &placement =
      moi_mode_operations(request.moi_engine).transient_scalar_placement;
  const bool inline_layout = placement.spill_layout == ConSanMoiScalarSpillLayout::Inline;
  if (options.moi_exec_save_sgpr)
    return false;
  const uint16_t count = moi_exec_save_sgpr_count(
      resolve_moi_exec_save_requirement(request, resources, options, problem.mode_semantics()),
      arch);
  if (count == 0)
    return false;

  const auto plan_contexts =
      resolve_moi_resource_plan_scalar_owner_contexts(planning_state, site_plans);
  if (!plan_contexts.planning_state_valid || plan_contexts.unresolved_owner) {
    std::string warning = "ConSan MOI could not resolve an owner context for automatic EXEC-save "
                          "SGPR placement";
    if (plan_contexts.unresolved_owner)
      warning += " at descriptor file offset " + std::to_string(*plan_contexts.unresolved_owner);
    warnings.push_back(std::move(warning));
    return false;
  }
  // A provisional CDNA resource failure may disappear after owner-local
  // persistent placement shrinks the probe ABI. Prove a global transient
  // window against those resolved owners too; otherwise a window selected
  // from the successful subset is later (correctly) rejected for the newly
  // recovered sites. The partial-owner path below remains the fallback when
  // no one code-object-wide window exists.
  const MoiScalarOwnerContextResolution scalar_contexts =
      collect_moi_scalar_owner_contexts(plan_contexts.groups, true);
  if (scalar_contexts.contexts.empty()) {
    warnings.emplace_back(
        "ConSan MOI could not derive automatic EXEC-save SGPRs without an owned candidate "
        "scope");
    return false;
  }

  const uint32_t first = util::align_up(scalar_contexts.tail_floor, 2u);
  const uint16_t ordinary_sgpr_limit = moi_ordinary_sgpr_limit(arch);
  const bool scalar_tail_covers_implicit_references =
      std::ranges::any_of(scalar_contexts.summaries, [](const auto &context) {
        return consan_detail::scalar_owner_tail_floor(context) > context.max_referenced_sgpr_count;
      });
  // Scalar-relative and open-CFG owners may still need a smaller persistent
  // tuple above the same tail. Prefer the high end for the wide transient
  // window so lower legal holes remain available to that later placement.
  MoiScalarPlacementDomain fresh_domain(arch, ordinary_sgpr_limit, scalar_contexts.summaries);
  fresh_domain.occupy(options.moi_owner_sgpr.base(), 1u);
  fresh_domain.occupy(options.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
  if (options.moi_dynamic_stack_spill)
    fresh_domain.occupy_dynamic_stack_state();
  const auto fresh_window = scalar_tail_covers_implicit_references
                                ? fresh_domain.select_last(first, count, 2u)
                                : fresh_domain.select_first(first, count, 2u);
  if (fresh_window) {
    options.moi_exec_save_sgpr = *fresh_window;
    options.moi_exec_save_sgprs_persistent =
        inline_layout && problem.mode_semantics().inline_access_present;
    warnings.emplace_back("ConSan MOI automatically assigned EXEC-save SGPRs s" +
                          std::to_string(*fresh_window) + ":s" +
                          std::to_string(*fresh_window + count - 1u));
    return true;
  }

  // Preserve one owner-wide ABI for the components that admit it, then let
  // the component solver override only the incompatible remainder. This is
  // stronger than an object-wide site-liveness proof because the retained
  // window is safe across every guest instruction in each admitted owner.
  if (consan_is_capability_arch(arch)) {
    const std::span<const consan_detail::ScalarOwnerContextSummary> no_contexts;
    MoiScalarPlacementDomain base_domain(arch, ordinary_sgpr_limit, no_contexts);
    base_domain.occupy(options.moi_owner_sgpr.base(), 1u);
    if (options.moi_dynamic_stack_spill)
      base_domain.occupy_dynamic_stack_state();
    for (int32_t base = static_cast<int32_t>((ordinary_sgpr_limit - count) & ~1u); base >= 0;
         base -= 2) {
      const uint16_t candidate = static_cast<uint16_t>(base);
      MoiScalarPlacementDomain layout = base_domain;
      if (!layout.select(candidate, count, /*require_dead=*/false))
        continue;

      const std::optional<uint16_t> prior_dispatch = options.moi_dispatch_identity.sgpr();
      const bool prior_dispatch_automatic = options.moi_dispatch_identity.automatic_sgpr();
      std::optional<uint16_t> dispatch = prior_dispatch;
      const uint16_t dispatch_count = kMoiDispatchStateSgprCount;
      const auto group_admits = [&](const MoiScalarOwnerContextGroup &group,
                                    std::optional<uint16_t> dispatch_base) {
        if (!group.supported || !moi_scalar_contexts_admit_reserved_window(group.owners.summaries,
                                                                           candidate, count, arch))
          return false;
        if (!dispatch_base)
          return true;
        const std::array ranges{MoiSgprRange{candidate, count},
                                MoiSgprRange{*dispatch_base, dispatch_count}};
        return layout.can_select(*dispatch_base, dispatch_count, /*require_dead=*/false) &&
               moi_scalar_contexts_admit_reserved_window(group.owners.summaries, *dispatch_base,
                                                         dispatch_count, arch) &&
               !moi_scalar_contexts_conflict_with_physical_vcc(group.owners.summaries, ranges,
                                                               arch);
      };

      if (dispatch && range_overlaps(candidate, count, *dispatch, dispatch_count)) {
        if (!inline_layout || !prior_dispatch_automatic)
          continue;
        dispatch.reset();
        for (int32_t proposed = static_cast<int32_t>((ordinary_sgpr_limit - dispatch_count) & ~1u);
             proposed >= 20; proposed -= 2) {
          const uint16_t dispatch_candidate = static_cast<uint16_t>(proposed);
          if (std::ranges::any_of(plan_contexts.groups, [&](const auto &group) {
                return group_admits(group, dispatch_candidate);
              })) {
            dispatch = dispatch_candidate;
            break;
          }
        }
        if (!dispatch)
          continue;
      }

      const auto admits = [&](const MoiScalarOwnerContextGroup &group) {
        return group_admits(group, dispatch);
      };
      if (!std::ranges::any_of(plan_contexts.groups, admits) ||
          std::ranges::all_of(plan_contexts.groups, admits))
        continue;
      options.moi_exec_save_sgpr = candidate;
      options.automatic_moi_partial_exec_save_sgprs = true;
      options.moi_exec_save_sgprs_persistent = true;
      options.moi_dispatch_identity.set_sgpr(dispatch, prior_dispatch_automatic);
      const bool assigned_component_local = configure_automatic_moi_transient_sgpr_assignments(
          options, problem, site_plans, warnings, planning_state);
      const bool relocation_recovers_component =
          dispatch == prior_dispatch || assigned_component_local;
      if (relocation_recovers_component &&
          (assigned_component_local || consan_arch_is_cdna3_or_cdna4(arch))) {
        warnings.emplace_back("ConSan MOI reserved an owner-wide EXEC-save window and assigned "
                              "component-local scalar state where needed");
        if (dispatch != prior_dispatch)
          warnings.emplace_back("ConSan MOI relocated automatic dispatch-ID state beside it");
        return true;
      }
      options.owner_transient_sgprs.clear();
      options.moi_exec_save_sgpr.reset();
      options.automatic_moi_partial_exec_save_sgprs = false;
      options.moi_exec_save_sgprs_persistent = false;
      options.moi_dispatch_identity.set_sgpr(prior_dispatch, prior_dispatch_automatic);
    }
  }

  // A high-numbered scalar operand anywhere in a kernel does not imply that
  // every lower SGPR is live at every instrumentation site. Large generated
  // kernels commonly use s104 for a short-lived address calculation while a
  // five-register window is dead throughout all LDS/synchronization sites.
  // Prove one common window with the existing CFG liveness before falling
  // back to an unsupported resource result.
  const MoiResourcePlanningState &state = planning_state;
  const MoiScalarSiteAvailability availability =
      collect_moi_scalar_site_availability(state, site_plans, problem.observation_plan(),
                                           problem.inventory(), options.moi_initialize_owner_epoch);
  const RegisterSet &unavailable = availability.unavailable;
  if (state.valid && availability.saw_site) {
    const uint32_t liveness_search_floor = scalar_tail_covers_implicit_references ? first : 0u;
    MoiScalarPlacementDomain liveness_domain(arch, ordinary_sgpr_limit, scalar_contexts.summaries,
                                             &unavailable);
    liveness_domain.occupy(options.moi_owner_sgpr.base(), 1u);
    liveness_domain.occupy(options.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
    if (options.moi_dynamic_stack_spill)
      liveness_domain.occupy_dynamic_stack_state();
    const auto liveness_window = liveness_domain.select_first(liveness_search_floor, count, 2u);
    if (liveness_window) {
      options.moi_exec_save_sgpr = *liveness_window;
      warnings.emplace_back("ConSan MOI automatically assigned liveness-dead EXEC-save SGPRs s" +
                            std::to_string(*liveness_window) + ":s" +
                            std::to_string(*liveness_window + count - 1u));
      return true;
    }
  }

  // RDNA4 Inline probes may preserve their wide transient window in private
  // memory, but frame setup still needs one object-wide dead scalar tuple.
  // Establish both parts as one transaction so later component overrides do
  // not accidentally overlap the persistent setup state.
  if (consan_arch_is_rdna4_or_cdna5(arch) && inline_layout &&
      scalar_contexts.all_use_dynamic_stack && problem.mode_semantics().inline_access_present &&
      state.valid && availability.saw_site) {
    MoiScalarPlacementDomain domain(arch, ordinary_sgpr_limit, scalar_contexts.summaries,
                                    &unavailable);
    domain.occupy(options.moi_owner_sgpr.base(), 1u);
    domain.occupy(options.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
    if (options.moi_dynamic_stack_spill)
      domain.occupy_dynamic_stack_state();
    const auto spill = domain.select_first(scalar_contexts.initialized_sgpr_count, count, 2u,
                                           /*require_dead=*/false);
    const auto visible = spill ? domain.select_first(first, 1u) : std::nullopt;
    const auto frame_base = spill ? domain.select_first(0u, 2u, 2u) : std::nullopt;
    const auto scc = spill ? domain.select_first(0u, 1u) : std::nullopt;
    if (spill && frame_base && scc) {
      options.moi_exec_save_sgpr = spill;
      options.automatic_moi_scalar_spill_layout = ConSanMoiScalarSpillLayout::Inline;
      options.moi_scalar_spill_setup = ConSanMoiScalarSpillSetup{
          .temporaries = ConSanMoiScalarSpillTemporaries{*frame_base, *scc},
          .visible_evidence_sgpr = visible};
      warnings.emplace_back("ConSan MOI automatically assigned spill-backed Inline SGPRs");
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
  if (consan_arch_is_rdna4_or_cdna5(arch)) {
    const std::span<const consan_detail::ScalarOwnerContextSummary> no_contexts;
    MoiScalarPlacementDomain high_domain(arch, ordinary_sgpr_limit, no_contexts);
    high_domain.occupy(options.moi_owner_sgpr.base(), 1u);
    high_domain.occupy(options.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
    if (options.moi_dynamic_stack_spill)
      high_domain.occupy_dynamic_stack_state();
    const int32_t last_base = static_cast<int32_t>((ordinary_sgpr_limit - count) & ~1u);
    for (int32_t base = last_base; base >= 0; base -= 2) {
      const uint16_t candidate = static_cast<uint16_t>(base);
      if (!high_domain.can_select(candidate, count, /*require_dead=*/false))
        continue;
      const bool admits_owner =
          std::ranges::any_of(plan_contexts.groups, [&](const MoiScalarOwnerContextGroup &group) {
            return group.supported && consan_detail::scalar_owner_contexts_admit_reserved_window(
                                          group.owners.summaries, candidate, count,
                                          /*protect_physical_vcc=*/false);
          });
      if (!admits_owner)
        continue;
      options.moi_exec_save_sgpr = candidate;
      options.automatic_moi_partial_exec_save_sgprs = true;
      options.moi_exec_save_sgprs_persistent = true;
      warnings.emplace_back("ConSan MOI reserved a high EXEC-save SGPR window and excluded only "
                            "incompatible full-pressure owner groups");
      (void)configure_automatic_moi_transient_sgpr_assignments(options, problem, site_plans,
                                                               warnings, state);
      return true;
    }
  }

  // A code object made entirely of full-pressure owners has no owner from
  // which to seed the ordinary high-window fallback. Seed component planning
  // with a placeholder; every admitted owner must replace it with either a
  // liveness-dead window or a spill-backed access-probe layout.
  if (consan_is_capability_arch(arch)) {
    options.moi_exec_save_sgpr = 0u;
    options.automatic_moi_partial_exec_save_sgprs = true;
    if (configure_automatic_moi_transient_sgpr_assignments(options, problem, site_plans, warnings,
                                                           state)) {
      warnings.emplace_back(
          "ConSan MOI admitted full-pressure owners with component-local scalar state");
      return true;
    }
    options.moi_exec_save_sgpr.reset();
    options.automatic_moi_partial_exec_save_sgprs = false;
  }

  warnings.emplace_back("ConSan MOI could not place a fresh automatic EXEC-save SGPR window");
  return false;
}

[[nodiscard]] bool configure_automatic_moi_owner_sgpr(
    ConSanMoiOperatingPoint &point, const MoiResourceProblem &problem,
    std::span<const ConSanCandidateResourcePlan> site_plans, std::vector<std::string> &diagnostics,
    const MoiResourcePlanningState &state) {
  const rj_code_arch_t arch = state.arch;
  if (problem.request().moi_owner_source != ConSanMoiOwnerSource::HwId ||
      point.moi_owner_sgpr.base())
    return false;
  if (moi_mode_operations(problem.request().moi_engine).transient_scalar_placement.spill_layout ==
      ConSanMoiScalarSpillLayout::Inline)
    return false;

  const auto plan_contexts = resolve_moi_resource_plan_scalar_owner_contexts(state, site_plans);
  if (!plan_contexts.planning_state_valid || plan_contexts.unresolved_owner) {
    std::string warning =
        "ConSan MOI could not resolve an owner context for automatic hw_id owner SGPR placement";
    if (plan_contexts.unresolved_owner)
      warning += " at descriptor file offset " + std::to_string(*plan_contexts.unresolved_owner);
    diagnostics.push_back(std::move(warning));
    return false;
  }
  const MoiScalarOwnerContextResolution scalar_contexts =
      collect_moi_scalar_owner_contexts(plan_contexts.groups, true);
  if (scalar_contexts.contexts.empty()) {
    diagnostics.emplace_back(
        "ConSan MOI could not derive an automatic hw_id owner SGPR without an owned candidate "
        "scope");
    return false;
  }

  MoiScalarPlacementDomain domain(arch, moi_ordinary_sgpr_limit(arch), scalar_contexts.summaries);
  domain.occupy(point.moi_exec_save_sgpr,
                moi_exec_save_sgpr_count(
                    resolve_moi_exec_save_requirement(problem.request(), problem.resources(), point,
                                                      problem.mode_semantics()),
                    arch));
  if (const auto sgpr = domain.select_first(scalar_contexts.tail_floor, 1u)) {
    point.moi_owner_sgpr.set(*sgpr, true);
    diagnostics.emplace_back("ConSan MOI automatically assigned hw_id owner temporary s" +
                             std::to_string(*sgpr));
    return true;
  }

  diagnostics.emplace_back("ConSan MOI could not place a fresh automatic hw_id owner SGPR");
  return false;
}

[[nodiscard]] bool configure_inline_moi_owner_sgpr(const ConSanRequest &request,
                                                   ConSanMoiOperatingPoint &point,
                                                   std::vector<std::string> &diagnostics) {
  const bool uses_planned_owner =
      moi_mode_operations(request.moi_engine).transient_scalar_placement.spill_layout ==
          ConSanMoiScalarSpillLayout::Inline &&
      request.moi_owner_source == ConSanMoiOwnerSource::HwId &&
      (!point.moi_owner_sgpr.base() || point.moi_owner_sgpr.automatic());
  if (!uses_planned_owner || point.moi_owner_sgpr.automatic())
    return false;
  if (!point.moi_exec_save_sgpr) {
    diagnostics.emplace_back(
        "ConSan MOI could not place its Inline resident-wave owner in a transient scalar window");
    return false;
  }
  point.moi_owner_sgpr.set(*point.moi_exec_save_sgpr, true);
  diagnostics.emplace_back(
      "ConSan MOI assigned its Inline resident-wave owner inside the saved scalar window");
  return true;
}

[[nodiscard]] ConSanMoiOperatingPointUpdate
configure_automatic_moi_dispatch_id_sgprs(const ConSanMoiOperatingPoint &base,
                                          const MoiResourceProblem &problem,
                                          std::span<const ConSanCandidateResourcePlan> site_plans,
                                          const MoiResourcePlanningState &state) {
  ConSanMoiOperatingPointUpdate update{
      .attempted_operating_point = base,
      .rejection = std::nullopt,
      .diagnostics = {},
  };
  ConSanMoiOperatingPoint &options = update.attempted_operating_point;
  const ConSanRequest &request = problem.request();
  const BoundRuntimeResources &resources = problem.resources();
  const rj_code_arch_t arch = state.arch;
  const ConSanTargetProfile *target = consan_target_profile(arch);
  const ConSanMoiDispatchIdentityPlacement target_placement =
      target ? target->moi_dispatch_identity_placement
             : ConSanMoiDispatchIdentityPlacement::Unsupported;
  const bool has_access_or_atomic_consumer =
      std::ranges::any_of(site_plans, [](const ConSanCandidateResourcePlan &plan) {
        return !plan.owner_descriptor_file_offsets.empty() &&
               (plan.site_kind == ConSanResourceSiteKind::Access ||
                plan.site_kind == ConSanResourceSiteKind::Atomic);
      });
  const MoiDispatchIdentityPlan dispatch_identity = plan_moi_dispatch_identity(
      request, {.access_reports_need_explicit_identity =
                    !target || target->moi_access_reports_need_explicit_dispatch_identity,
                .has_access_or_atomic_consumer = has_access_or_atomic_consumer});
  if (!dispatch_identity.needs_dispatch_id || !resources.moi_report_buffer_address ||
      options.moi_dispatch_identity.sgpr() || options.moi_dispatch_identity.vgpr()) {
    return update;
  }
  const auto plan_contexts = resolve_moi_resource_plan_scalar_owner_contexts(state, site_plans);
  if (!plan_contexts.planning_state_valid || plan_contexts.unresolved_owner) {
    std::string warning =
        "ConSan MOI could not resolve an owner context for automatic dispatch-ID SGPR placement";
    if (plan_contexts.unresolved_owner)
      warning += " at descriptor file offset " + std::to_string(*plan_contexts.unresolved_owner);
    update.diagnostics.push_back(std::move(warning));
    return update;
  }
  const MoiScalarOwnerContextResolution scalar_contexts =
      collect_moi_scalar_owner_contexts(plan_contexts.groups, true);
  if (scalar_contexts.contexts.empty()) {
    update.diagnostics.emplace_back(
        "ConSan MOI could not derive automatic dispatch-ID SGPRs without an owned candidate "
        "scope");
    return update;
  }

  // A target may prefer descriptor-grown persistent VGPRs to a long-lived
  // reservation in the guest scalar file. A kernel that references the
  // complete VGPR file has no such pair; in that capacity case, continue into
  // the target's other qualified scalar searches.
  if (target_placement == ConSanMoiDispatchIdentityPlacement::PersistentVectorPreferred) {
    const bool lacks_persistent_dispatch_vgpr_pair =
        std::ranges::any_of(scalar_contexts.contexts, [](const MoiKernelResourceContext *context) {
          const uint32_t safe_base = context->has_indirect_vgpr_access
                                         ? context->current_vgpr_count
                                         : context->max_referenced_vgpr_count;
          return safe_base + 2u > kMaxVgprs;
        });
    if (!lacks_persistent_dispatch_vgpr_pair)
      return update;
  }

  // Sixteen initialized user SGPRs plus X/Y/Z and WORKGROUP_INFO is the
  // largest legal entry-input window after insertion. Keeping the persistent
  // pair above that window makes it safe for every per-kernel preload map.
  uint32_t first = std::max<uint32_t>(20u, scalar_contexts.tail_floor);

  const auto qualify_owner_local_assignments = [&](uint16_t dispatch_base) {
    for (ConSanMoiTransientSgprAssignment &assignment : options.owner_transient_sgprs) {
      const std::array owner = {assignment.descriptor_file_offset};
      const auto contexts = resolve_moi_scalar_owner_contexts(state, owner);
      assignment.dispatch_id_sgpr = contexts && moi_scalar_contexts_admit_reserved_window(
                                                    contexts->summaries, dispatch_base, 2u, arch)
                                        ? std::optional(dispatch_base)
                                        : std::nullopt;
    }
  };

  first = util::align_up(first, 2u);
  const uint16_t dispatch_state_count = kMoiDispatchStateSgprCount;
  const uint16_t ordinary_sgpr_limit = moi_ordinary_sgpr_limit(arch);
  MoiScalarPlacementDomain dispatch_domain(arch, ordinary_sgpr_limit, scalar_contexts.summaries);
  dispatch_domain.occupy(options.moi_owner_sgpr.base(), 1u);
  dispatch_domain.occupy(
      options.moi_exec_save_sgpr,
      moi_exec_save_sgpr_count(
          resolve_moi_exec_save_requirement(request, resources, options, problem.mode_semantics()),
          arch));
  const auto dispatch_base = dispatch_domain.select_first(first, dispatch_state_count, 2u);
  if (dispatch_base) {
    options.moi_dispatch_identity.set_sgpr(*dispatch_base, true);
    qualify_owner_local_assignments(*dispatch_base);
    update.diagnostics.emplace_back("ConSan MOI automatically assigned persistent dispatch-ID "
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
  if (target_placement == ConSanMoiDispatchIdentityPlacement::PreloadedScalar ||
      target_placement == ConSanMoiDispatchIdentityPlacement::ScalarThenPersistentVector) {
    const std::span<const consan_detail::ScalarOwnerContextSummary> no_contexts;
    MoiScalarPlacementDomain high_domain(arch, ordinary_sgpr_limit, no_contexts);
    high_domain.occupy(options.moi_owner_sgpr.base(), 1u);
    high_domain.occupy(
        options.moi_exec_save_sgpr,
        moi_exec_save_sgpr_count(resolve_moi_exec_save_requirement(request, resources, options,
                                                                   problem.mode_semantics()),
                                 arch));
    const int32_t last_base =
        static_cast<int32_t>((ordinary_sgpr_limit - dispatch_state_count) & ~1u);
    for (int32_t base = last_base; base >= 20; base -= 2) {
      const uint16_t candidate = static_cast<uint16_t>(base);
      if (!high_domain.can_select(candidate, dispatch_state_count, /*require_dead=*/false))
        continue;
      const bool admits_owner =
          std::ranges::any_of(plan_contexts.groups, [&](const MoiScalarOwnerContextGroup &group) {
            return group.supported &&
                   moi_scalar_contexts_admit_reserved_window(group.owners.summaries, candidate,
                                                             dispatch_state_count, arch);
          });
      if (target_placement == ConSanMoiDispatchIdentityPlacement::PreloadedScalar && !admits_owner)
        continue;
      options.moi_dispatch_identity.set_sgpr(candidate, true);
      qualify_owner_local_assignments(candidate);
      update.diagnostics.emplace_back(
          "ConSan MOI reserved a high dispatch-ID SGPR pair and excluded only "
          "incompatible full-pressure owner groups");
      return update;
    }
  }

  if (target_placement == ConSanMoiDispatchIdentityPlacement::PersistentVectorPreferred ||
      target_placement == ConSanMoiDispatchIdentityPlacement::ScalarThenLiteral) {
    // A dynamic-stack kernel can allocate the full ordinary scalar file while
    // leaving a lower pair completely untouched. Dispatch identity crosses
    // arbitrary guest code, so site-local liveness is not sufficient: require
    // a closed owner CFG, or conservatively scan every declared function when
    // those ranges cover the complete text section. The pair also remains
    // above the largest possible entry-input window so dispatch-ID insertion
    // and guest-input restoration cannot overwrite it.
    MoiOwnerRegisterReferenceSummary references =
        summarize_moi_owner_register_references(scalar_contexts.contexts);
    bool used_complete_text_summary = false;
    if (!references.control_flow_closed) {
      if (auto complete = summarize_moi_complete_text_register_references(state)) {
        references = std::move(*complete);
        used_complete_text_summary = true;
      }
    }
    const bool has_dynamic_stack_owner = std::ranges::any_of(
        problem.inventory().kernels(), [](const ConSanProgramContainer &kernel) {
          return kernel.uses_dynamic_stack.value_or(false);
        });
    if (references.control_flow_closed && !references.uses_indirect_sgprs) {
      MoiScalarPlacementDomain unused_domain(arch, ordinary_sgpr_limit, scalar_contexts.summaries,
                                             &references.references);
      unused_domain.occupy(options.moi_owner_sgpr.base(), 1u);
      unused_domain.occupy(
          options.moi_exec_save_sgpr,
          moi_exec_save_sgpr_count(resolve_moi_exec_save_requirement(request, resources, options,
                                                                     problem.mode_semantics()),
                                   arch));
      if (has_dynamic_stack_owner)
        unused_domain.occupy_dynamic_stack_state();
      const auto unused_dispatch_base = unused_domain.select_first(20u, dispatch_state_count, 2u);
      if (unused_dispatch_base) {
        options.moi_dispatch_identity.set_sgpr(*unused_dispatch_base, true);
        qualify_owner_local_assignments(*unused_dispatch_base);
        update.diagnostics.emplace_back(
            "ConSan MOI assigned persistent dispatch-ID state to a proven-unused " +
            std::string(rj_code_arch_name(arch)) + " SGPR pair" +
            std::string(used_complete_text_summary ? " using complete-text coverage" : ""));
        return update;
      }
    }
  }

  if (target_placement == ConSanMoiDispatchIdentityPlacement::ScalarThenLiteral ||
      dispatch_identity.fallback_kind == ConSanMoiFallbackKind::SampledLiteralDispatchId) {
    // Automatic banked Record/Replay can capture the preload into persistent
    // descriptor-grown VGPRs during the later owner-state placement pass.
    // Sampled instead records the stable reader identity literal when no
    // persistent hardware pair survives every owner.
    if (problem.mode_semantics().automatic_banked_record_capture)
      return update;
    update.diagnostics.emplace_back(
        "ConSan MOI is using a literal report dispatch ID because no persistent hardware "
        "dispatch-ID SGPR pair fits");
    return update;
  }
  if (target_placement == ConSanMoiDispatchIdentityPlacement::PreloadedScalar &&
      dispatch_identity.permits_private_entry_capture &&
      std::ranges::any_of(site_plans, [](const ConSanCandidateResourcePlan &plan) {
        return plan.source != ConSanRegisterAllocationSource::Unsupported &&
               plan.site_kind == ConSanResourceSiteKind::Access;
      })) {
    const bool has_dynamic_stack_access_owner = std::ranges::any_of(
        problem.inventory().kernels(), [&](const ConSanProgramContainer &kernel) {
          return kernel.uses_dynamic_stack.value_or(false) &&
                 std::ranges::any_of(site_plans, [&](const ConSanCandidateResourcePlan &plan) {
                   return plan.source != ConSanRegisterAllocationSource::Unsupported &&
                          plan.site_kind == ConSanResourceSiteKind::Access &&
                          std::ranges::find(plan.owner_descriptor_file_offsets,
                                            kernel.descriptor_file_offset) !=
                              plan.owner_descriptor_file_offsets.end();
                 });
        });
    // A non-dynamic owner can capture the hardware dispatch preload directly
    // in the same address-free private entry state as owner/epoch. A dynamic
    // stack cannot compose with that representation; leave the dispatch marker
    // set so persistent-state placement includes an entry-captured VGPR pair.
    options.moi_dispatch_identity.set_private_fallback(true);
    if (!has_dynamic_stack_access_owner) {
      options.moi_initialize_owner_epoch = true;
      options.automatic_moi_private_epoch = true;
      update.diagnostics.emplace_back(
          "ConSan MOI selected owner-private hardware dispatch-ID state because no persistent "
          "SGPR pair fits");
    } else {
      update.diagnostics.emplace_back(
          "ConSan MOI deferred hardware dispatch-ID state to persistent VGPR placement because "
          "owner-private state is incompatible with a dynamic stack");
    }
    return update;
  }
  update.rejection = ConSanMoiPlacementRejection::DispatchIdSgprUnavailable;
  update.diagnostics.emplace_back(
      "ConSan MOI could not place a persistent hardware dispatch-ID SGPR pair");
  return update;
}

[[nodiscard]] ConSanMoiOperatingPointAttempt
plan_moi_dispatch_id_fallback(const ConSanOptions &input, const ConSanMoiOperatingPoint &base,
                              const MoiResourceProblem &problem,
                              std::span<const ConSanCandidateResourcePlan> site_plans) {
  ConSanMoiOperatingPoint point = base;
  ConSanMoiOperatingPointAttempt attempt{
      .attempted_operating_point = base,
      .accepted_fallback = std::nullopt,
      .diagnostics = {},
  };
  const rj_code_arch_t arch = problem.arch();
  const MoiDispatchIdentityPlan dispatch_identity =
      plan_moi_dispatch_identity(problem.request(), {});
  if (!consan_is_capability_arch(arch) || !dispatch_identity.fallback_kind ||
      !point.moi_dispatch_identity.automatic_sgpr() || !point.moi_dispatch_identity.sgpr() ||
      !point.moi_exec_save_sgpr) {
    return attempt;
  }

  // Dispatch placement runs before transient EXEC-save placement. Record/Replay
  // needs a complete owner-local scalar ABI for every newly included component.
  // Sampled instead retains the independently proven EXEC window and overrides
  // only a conflicting dispatch pair with its literal identity representation.
  std::vector<ConSanMoiTransientSgprAssignment> &owner_transient_sgprs =
      point.owner_transient_sgprs;
  owner_transient_sgprs.clear();
  point.automatic_moi_partial_exec_save_sgprs = true;
  const bool dispatch_only = dispatch_identity.fallback_replans_dispatch_only;
  MoiResourcePlanningState planning_state(problem, point, site_plans);
  std::vector<std::string> diagnostics;
  if (!configure_automatic_moi_transient_sgpr_assignments(point, problem, site_plans, diagnostics,
                                                          planning_state, dispatch_only)) {
    attempt.attempted_operating_point = point;
    return attempt;
  }

  if (dispatch_only) {
    // EXEC and dispatch placement are independent proofs. Preserve every
    // previously selected owner-local EXEC ABI and overlay only the dispatch
    // representation derived above. Replacing the whole assignment here can
    // resurrect a code-object-wide EXEC window that the owner had already
    // rejected, silently dropping all sites in that component.
    const auto dispatch_assignments = std::move(owner_transient_sgprs);
    owner_transient_sgprs = base.owner_transient_sgprs;
    for (const ConSanMoiTransientSgprAssignment &dispatch_assignment : dispatch_assignments) {
      auto prior =
          std::ranges::find(owner_transient_sgprs, dispatch_assignment.descriptor_file_offset,
                            &ConSanMoiTransientSgprAssignment::descriptor_file_offset);
      if (prior == owner_transient_sgprs.end()) {
        // A partial EXEC proof deliberately leaves an incompatible owner
        // without an assignment. Dispatch-only planning starts from the
        // global EXEC base, so admitting that newly synthesized assignment
        // would resurrect the excluded owner with an unsafe transient
        // window. Owners that safely use the global EXEC ABI remain eligible.
        if (base.automatic_moi_partial_exec_save_sgprs) {
          const std::array owners{dispatch_assignment.descriptor_file_offset};
          const auto owner_contexts = resolve_moi_scalar_owner_contexts(planning_state, owners);
          if (!owner_contexts ||
              !moi_scalar_contexts_admit_reserved_window(
                  owner_contexts->summaries, *point.moi_exec_save_sgpr,
                  moi_exec_save_sgpr_count(resolve_moi_exec_save_requirement(
                                               input, input, point, problem.mode_semantics()),
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
    point.automatic_moi_partial_exec_save_sgprs = base.automatic_moi_partial_exec_save_sgprs;
  }

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

  diagnostics.emplace_back(dispatch_identity.fallback_diagnostic);
  attempt.attempted_operating_point = point;
  attempt.accepted_fallback = dispatch_identity.fallback_kind;
  attempt.diagnostics = std::move(diagnostics);
  return attempt;
}

[[nodiscard]] std::optional<std::string>
validate_moi_scalar_state(const ConSanRequest &request, const BoundRuntimeResources &resources,
                          const ConSanMoiOperatingPoint &point,
                          const MoiObjectModeSemantics &mode_semantics, rj_code_arch_t arch) {
  if (point.moi_dispatch_identity.sgpr()) {
    const uint16_t base = *point.moi_dispatch_identity.sgpr();
    const uint16_t width = kMoiDispatchStateSgprCount;
    const bool overlaps_owner = point.moi_owner_sgpr.base() &&
                                range_overlaps(base, width, *point.moi_owner_sgpr.base(), 1u);
    const bool owner_components_replace_global_exec =
        point.automatic_moi_partial_exec_save_sgprs &&
        point.moi_dispatch_identity.private_fallback() && !point.owner_transient_sgprs.empty();
    // A partial operating point keeps the object-wide pair as the entry
    // preload destination, but every affected owner replaces the overlapping
    // probe-local EXEC window and captures dispatch identity privately. Those
    // values are not live concurrently. The component planner established
    // complete coverage before setting both markers above.
    const bool overlaps_exec =
        !owner_components_replace_global_exec && point.moi_exec_save_sgpr &&
        range_overlaps(base, width, *point.moi_exec_save_sgpr,
                       moi_exec_save_sgpr_count(resolve_moi_exec_save_requirement(
                                                    request, resources, point, mode_semantics),
                                                arch));
    const std::array violations{
        std::pair{base < 20u, std::string_view{"entry window"}},
        std::pair{base % 2u != 0u, std::string_view{"alignment"}},
        std::pair{static_cast<uint32_t>(base) + width > moi_ordinary_sgpr_limit(arch),
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
      return "ConSan MOI persistent dispatch-ID SGPR pair s" + std::to_string(base) + ":s" +
             std::to_string(base + width - 1u) + " violates " + reason;
    }
  }
  const bool inline_layout =
      moi_mode_operations(request.moi_engine).transient_scalar_placement.spill_layout ==
      ConSanMoiScalarSpillLayout::Inline;
  const std::span<const consan_detail::ScalarOwnerContextSummary> no_contexts;
  const MoiScalarPlacementDomain domain(arch, moi_ordinary_sgpr_limit(arch), no_contexts);
  const auto invalid = [&](std::optional<uint16_t> base, uint16_t width) {
    return base && !domain.legal_range(*base, width);
  };
  const uint16_t exec_count = moi_exec_save_sgpr_count(
      resolve_moi_exec_save_requirement(request, resources, point, mode_semantics), arch);
  const bool exec_invalid = invalid(point.moi_exec_save_sgpr, exec_count);
  const bool owner_invalid = invalid(point.moi_owner_sgpr.base(), 1u);
  bool persistent_invalid = false;
  point.moi_persistent_sgprs.for_each_range([&](std::optional<uint16_t> base, uint16_t width) {
    persistent_invalid |= invalid(base, width);
  });
  bool auxiliary_invalid = false;
  point.for_each_auxiliary_sgpr_range([&](std::optional<uint16_t> base, uint16_t width) {
    auxiliary_invalid |= invalid(base, width);
  });
  const bool branch_only_scalar_invalid =
      point.moi_branch_only_spill && (!inline_layout || !point.has_inline_moi_scalar_spill() ||
                                      !consan_arch_is_rdna4_or_cdna5(arch) ||
                                      !point.moi_exec_save_sgpr || point.moi_scalar_spill_setup);
  const bool dynamic_bootstrap_invalid = [&] {
    if (!point.moi_branch_only_spill || !point.moi_branch_only_spill->dynamic_stack_borrowed_sgpr)
      return false;
    const uint16_t base = *point.moi_branch_only_spill->dynamic_stack_borrowed_sgpr;
    MoiScalarPlacementDomain bootstrap_domain = domain;
    bootstrap_domain.occupy_dynamic_stack_state();
    bootstrap_domain.occupy(point.moi_exec_save_sgpr, exec_count);
    point.moi_persistent_sgprs.for_each_range(
        [&](std::optional<uint16_t> reg, uint16_t width) { bootstrap_domain.occupy(reg, width); });
    return !inline_layout || !point.has_inline_moi_scalar_spill() ||
           !point.moi_dynamic_stack_spill || !consan_arch_is_rdna4_or_cdna5(arch) ||
           base % 2u != 0u || point.moi_scalar_spill_setup ||
           !bootstrap_domain.can_select(base, 2u, /*require_dead=*/false);
  }();
  if (!exec_invalid && !owner_invalid && !persistent_invalid && !auxiliary_invalid &&
      !branch_only_scalar_invalid && !dynamic_bootstrap_invalid)
    return std::nullopt;
  return "ConSan MOI scalar instrumentation state aliases an architectural special SGPR";
}

[[nodiscard]] std::optional<std::string>
validate_moi_dispatch_id_vgprs(const ConSanMoiOperatingPoint &point) {
  if (!point.moi_dispatch_identity.vgpr())
    return std::nullopt;

  const uint16_t base = *point.moi_dispatch_identity.vgpr();
  bool overlaps_persistent_state = false;
  moi_persistent_vgpr_state_view(point).for_each_range(
      [&](std::optional<uint16_t> reg, uint16_t width) {
        if (reg)
          overlaps_persistent_state |= range_overlaps(base, 2u, *reg, width);
      },
      false);
  if (static_cast<uint32_t>(base) + 2u > kMaxVgprs || overlaps_persistent_state) {
    return "ConSan MOI persistent dispatch-ID VGPR pair overlaps persistent state or exceeds "
           "the architectural VGPR file";
  }
  return std::nullopt;
}

[[nodiscard]] ConSanMoiPersistentPlacementUpdate configure_automatic_moi_persistent_vgprs(
    const ConSanMoiOperatingPoint &base, const MoiResourceProblem &problem,
    const ConSanDebugOverrides &debug, std::span<const ConSanCandidateResourcePlan> resource_plans,
    const MoiResourcePlanningState &planning_state) {
  ConSanMoiPersistentPlacementUpdate update{
      .attempted_operating_point = base,
      .rejection = std::nullopt,
      .diagnostics = {},
      .prologue_scratch_assignments = {},
  };
  ConSanMoiOperatingPoint &options = update.attempted_operating_point;
  const ConSanRequest &request = problem.request();
  const BoundRuntimeResources &resources = problem.resources();
  const ProgramInventory &program_inventory = problem.inventory();
  std::vector<std::string> &diagnostics = update.diagnostics;
  std::vector<ConSanMoiPrologueScratchVgprAssignment> &prologue_scratch_assignments =
      update.prologue_scratch_assignments;
  const std::span<const uint8_t> bytes = planning_state.bytes;
  const rj_code_arch_t arch = planning_state.arch;
  MoiPersistentStateFacts persistent_facts;
  for (const ConSanCandidateResourcePlan &plan : resource_plans) {
    if (plan.source == ConSanRegisterAllocationSource::Unsupported ||
        plan.owner_descriptor_file_offsets.empty()) {
      continue;
    }
    persistent_facts.access_count += plan.site_kind == ConSanResourceSiteKind::Access;
    persistent_facts.atomic_count += plan.site_kind == ConSanResourceSiteKind::Atomic;
    persistent_facts.barrier_count += plan.site_kind == ConSanResourceSiteKind::Barrier;
    persistent_facts.fence_count += plan.site_kind == ConSanResourceSiteKind::Fence;
    for (uint64_t owner : plan.owner_descriptor_file_offsets) {
      const ConSanProgramContainer *kernel = program_inventory.find_kernel_by_descriptor(owner);
      persistent_facts.has_operational_dynamic_stack_owner |=
          kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
      persistent_facts.has_operational_dynamic_lds_owner |=
          kernel != nullptr && kernel->has_dynamic_lds;
    }
  }
  const MoiPersistentStateDemand persistent_demand =
      plan_moi_persistent_state_demand(request, resources, options, persistent_facts);
  const MoiCdnaPersistentOverflowStrategy cdna_overflow_strategy =
      persistent_demand.cdna_overflow_strategy;
  const bool uses_resident_wave_private_state =
      cdna_overflow_strategy == MoiCdnaPersistentOverflowStrategy::ResidentWavePrivateState;
  const bool needs_exact_workgroup_tuple = persistent_demand.needs_entry_workgroup_tuple;
  const bool needs_cluster_workgroup_id =
      needs_exact_workgroup_tuple && consan_arch_has_cluster_facilities(arch) &&
      std::ranges::any_of(program_inventory.kernels(), [](const ConSanProgramContainer &kernel) {
        return kernel.uses_cluster_workgroup_id;
      });
  const ConSanTargetProfile *target = consan_target_profile(arch);
  const bool target_uses_persistent_dispatch_vgprs =
      target && target->moi_dispatch_identity_placement !=
                    ConSanMoiDispatchIdentityPlacement::PreloadedScalar;
  const bool needs_persistent_dispatch_vgprs =
      (target_uses_persistent_dispatch_vgprs &&
       persistent_demand.needs_persistent_dispatch_capture) ||
      persistent_demand.private_dispatch_incompatible_with_dynamic_stack;
  const bool needs_workgroup_key = persistent_demand.needs_workgroup_key;
  const bool has_explicit_owner_epoch = options.moi_owner_epoch_vgprs.complete();
  if (!moi_exact_entry_workgroup_capture_is_unambiguous(options)) {
    update.rejection = ConSanMoiPlacementRejection::AmbiguousWorkgroupCapture;
    diagnostics.emplace_back("ConSan MOI has multiple persistent workgroup-tuple representations");
    return update;
  }
  if (debug.requested_moi_owner_vgpr.has_value() != debug.requested_moi_epoch_vgpr.has_value()) {
    update.rejection = ConSanMoiPlacementRejection::PartialOwnerEpochVgprOverride;
    diagnostics.emplace_back(
        "ConSan MOI persistent VGPR overrides require both owner and epoch registers");
    return update;
  }
  if ((options.moi_workgroup_key_vgpr && !needs_exact_workgroup_tuple) ||
      (has_explicit_owner_epoch && !needs_exact_workgroup_tuple))
    return update;
  if (!persistent_demand.needs_persistent_state)
    return update;
  if (!resources.moi_report_buffer_address)
    return update;

  uint32_t persistent_base = 0;
  bool found_owned_scope = false;
  for (const ConSanCandidateResourcePlan &plan : resource_plans) {
    if (plan.owner_descriptor_file_offsets.empty())
      continue;
    found_owned_scope = true;
    // Persistent state crosses arbitrary guest code, so it must sit above
    // every statically referenced ordinary VGPR. Indirect v_movrel access has
    // no complete static bound and conservatively reserves the ordinary guest
    // allocation instead. On CDNA3/4, ACCUM_OFFSET bounds this search below the
    // physical AccVGPR bank even when the descriptor's total VGPR allocation
    // extends through that bank.
    const uint32_t safe_base =
        plan.has_indirect_vgpr_access
            ? std::min<uint32_t>(plan.current_vgpr_count, plan.ordinary_vgpr_limit)
            : plan.max_referenced_vgpr_count;
    persistent_base = std::max(persistent_base, safe_base);
  }
  if (has_explicit_owner_epoch) {
    persistent_base = std::max<uint32_t>(
        persistent_base,
        std::max(options.moi_owner_epoch_vgprs->owner, options.moi_owner_epoch_vgprs->epoch) + 1u);
  }

  if (!found_owned_scope) {
    diagnostics.emplace_back(
        "ConSan MOI could not derive automatic persistent VGPRs without an owned candidate "
        "scope");
    return update;
  }
  if (target_uses_persistent_dispatch_vgprs && !uses_resident_wave_private_state) {
    // Persistent owner/epoch and exact-workgroup state shares the ordinary
    // VGPR file with site-local Record/Replay and Sampled probes. Keep it above
    // the largest operational scratch window. Guest operands may overlap a
    // spill-backed window because emission recovers them from private state;
    // persistent values cannot overlap it because they must survive for the
    // complete dispatch.
    uint32_t scratch_floor = 0;
    for (const ConSanCandidateResourcePlan &plan : resource_plans) {
      if (plan.source != ConSanRegisterAllocationSource::Unsupported &&
          !plan.owner_descriptor_file_offsets.empty()) {
        const uint32_t scratch_end =
            plan.scratch_vgpr ? static_cast<uint32_t>(*plan.scratch_vgpr) + plan.scratch_vgpr_count
                              : plan.scratch_vgpr_count;
        scratch_floor = std::max(scratch_floor, scratch_end);
      }
    }
    persistent_base = std::max(persistent_base, scratch_floor);
  }
  // A target with a packed-128 workgroup-shadow clear uses three additional
  // tail VGPRs while clearing. They must lie above every guest reference even
  // though owner/epoch remain the only persistent values.
  const uint32_t automatic_owner_epoch_count = has_explicit_owner_epoch ? 0u : 2u;
  const uint32_t exact_workgroup_count =
      needs_exact_workgroup_tuple ? (needs_cluster_workgroup_id ? 4u : 3u) : 0u;
  const uint32_t dispatch_id_count = needs_persistent_dispatch_vgprs ? 2u : 0u;
  const uint32_t named_persistent_count = automatic_owner_epoch_count +
                                          static_cast<uint32_t>(needs_workgroup_key) +
                                          exact_workgroup_count + dispatch_id_count;
  const uint32_t persistent_count = uses_resident_wave_private_state && consan_arch_is_cdna5(arch)
                                        ? std::max<uint32_t>(named_persistent_count, 5u)
                                        : named_persistent_count;
  const uint32_t persistent_end = persistent_base + persistent_count;
  const auto has_dynamic_stack_resource_owner = [&] {
    return std::ranges::any_of(
        program_inventory.kernels(), [&](const ConSanProgramContainer &kernel) {
          return kernel.uses_dynamic_stack.value_or(false) &&
                 std::ranges::any_of(resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
                   // Resource planning precedes persistent-state placement.
                   // A full-pressure site can therefore be provisionally
                   // unsupported only because the code-object-wide fallback
                   // ABI is larger than its eventual owner-local ABI. Keep
                   // every semantically admitted, resolved owner in this
                   // query so placement can recover that site on the rebuild.
                   return (plan.source != ConSanRegisterAllocationSource::Unsupported ||
                           moi_resource_plan_can_be_recovered_by_persistent_placement(plan, debug,
                                                                                      arch)) &&
                          std::ranges::find(plan.owner_descriptor_file_offsets,
                                            kernel.descriptor_file_offset) !=
                              plan.owner_descriptor_file_offsets.end();
                 });
        });
  };
  const auto assign_scalar_persistent_state = [&](uint16_t base) {
    uint16_t next = base;
    if (!has_explicit_owner_epoch) {
      options.moi_persistent_sgprs.set_owner_epoch(next, static_cast<uint16_t>(next + 1u));
      next = static_cast<uint16_t>(next + 2u);
    }
    if (needs_workgroup_key)
      options.moi_persistent_sgprs.workgroup_key = next++;
    if (needs_exact_workgroup_tuple) {
      options.moi_persistent_sgprs.exact_workgroup = ConSanMoiPersistentWorkgroupRegisters{
          next, static_cast<uint16_t>(next + 1u), static_cast<uint16_t>(next + 2u),
          needs_cluster_workgroup_id ? std::optional<uint16_t>(next + 3u) : std::nullopt};
    }
  };
  const bool crosses_cdna_accvgpr_boundary =
      (consan_arch_is_cdna3_or_cdna4(program_inventory.arch())) &&
      std::ranges::any_of(resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        return plan.source != ConSanRegisterAllocationSource::Unsupported &&
               !plan.owner_descriptor_file_offsets.empty() &&
               ((plan.ordinary_vgpr_limit != kMaxVgprs &&
                 persistent_end > plan.ordinary_vgpr_limit) ||
                persistent_end > kMaxVgprs);
      });
  // A partial CDNA scalar ABI also needs per-owner persistent placement when
  // one component cannot keep the object-wide dispatch pair.  That component
  // must receive an entry-captured VGPR pair even if owner/epoch/workgroup-key
  // state by itself fits below every AccVGPR boundary.  The other components
  // retain the smaller scalar-dispatch representation.
  const bool needs_cdna_component_dispatch_placement =
      (consan_arch_is_cdna3_or_cdna4(program_inventory.arch())) &&
      uses_resident_wave_private_state && options.moi_dispatch_identity.private_fallback() &&
      !needs_persistent_dispatch_vgprs;
  if (crosses_cdna_accvgpr_boundary || needs_cdna_component_dispatch_placement) {
    // ACCUM_OFFSET defines the compiler's ordinary/accumulator register split.
    // Moving that split after code generation changes which physical registers
    // existing MFMA and accumulator operands address and has corrupted native
    // gfx950 workloads. Until persistent tuples are selected per owning kernel,
    // reject a code object whose one global tuple would cross any such boundary.
    const bool has_dynamic_stack_owner = has_dynamic_stack_resource_owner();
    if (uses_resident_wave_private_state && options.automatic_moi_private_epoch &&
        !has_dynamic_stack_owner) {
      diagnostics.emplace_back(
          "ConSan MOI retained owner/epoch and dispatch identity in entry-captured private state "
          "below a CDNA AccVGPR boundary");
      return update;
    }
    if (cdna_overflow_strategy == MoiCdnaPersistentOverflowStrategy::ExactWorkgroupState &&
        debug.test_force_private_epoch && !has_dynamic_stack_owner) {
      options.moi_initialize_owner_epoch = true;
      options.automatic_moi_private_epoch = true;
      diagnostics.emplace_back(
          "ConSan MOI selected entry-captured private Record/Replay state below a CDNA "
          "ordinary-VGPR boundary");
      return update;
    }
    if (cdna_overflow_strategy == MoiCdnaPersistentOverflowStrategy::OwnerSnapshot &&
        request.moi_owner_source == ConSanMoiOwnerSource::WorkitemId && !has_dynamic_stack_owner) {
      options.moi_initialize_owner_epoch = true;
      options.automatic_moi_private_epoch = true;
      diagnostics.emplace_back(
          "ConSan MOI selected entry-snapshotted private owner/epoch state below a CDNA ordinary "
          "VGPR boundary for Sampled instrumentation");
      return update;
    }
    if (cdna_overflow_strategy == MoiCdnaPersistentOverflowStrategy::ExactWorkgroupState &&
        !debug.test_force_private_epoch) {
      const uint16_t scalar_count = static_cast<uint16_t>(named_persistent_count);
      const MoiResourcePlanningState &scalar_state = planning_state;
      const std::vector<uint64_t> scalar_owners = moi_scalar_owner_descriptors(resource_plans);
      const auto resolved_contexts = resolve_moi_scalar_owner_contexts(scalar_state, scalar_owners);
      const std::vector<const MoiKernelResourceContext *> scalar_contexts =
          resolved_contexts ? resolved_contexts->contexts
                            : std::vector<const MoiKernelResourceContext *>{};
      // Per-context allocation floors and VCC checks are intentionally
      // stricter than the former plan-min scan. Resolution failure leaves the
      // candidate range empty so placement fails closed.
      const uint32_t first =
          resolved_contexts ? resolved_contexts->tail_floor : moi_ordinary_sgpr_limit(arch);
      const std::span<const consan_detail::ScalarOwnerContextSummary> scalar_summaries =
          resolved_contexts ? std::span<const consan_detail::ScalarOwnerContextSummary>(
                                  resolved_contexts->summaries)
                            : std::span<const consan_detail::ScalarOwnerContextSummary>{};
      MoiScalarPlacementDomain scalar_domain(arch, moi_ordinary_sgpr_limit(arch), scalar_summaries);
      scalar_domain.occupy(
          options.moi_exec_save_sgpr,
          moi_exec_save_sgpr_count(resolve_moi_exec_save_requirement(request, resources, options,
                                                                     problem.mode_semantics()),
                                   arch));
      scalar_domain.occupy(options.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
      scalar_domain.occupy(options.moi_owner_sgpr.base(), 1u);
      const auto scalar_tail = scalar_domain.select_first(first, scalar_count);
      const std::string scalar_state_description =
          "ConSan MOI assigned persistent scalar " +
          std::string(has_explicit_owner_epoch ? "" : "owner/epoch") +
          std::string(needs_workgroup_key ? "/workgroup-key" : "") +
          std::string(needs_exact_workgroup_tuple ? "/exact-workgroup-tuple" : "");
      const auto commit_exact_workgroup_scalar_state = [&](uint16_t base,
                                                           std::string diagnostic_suffix) {
        std::vector<ConSanMoiPrologueScratchVgprAssignment> prologue_scratch;
        // Record/Replay initializes both persistent values directly in SGPRs;
        // its shared prologue API does not use the placeholder VGPR pair.
        // Resource plans are rebuilt after persistent-state selection. Keep a
        // placeholder for every possible kernel owner so a site admitted by
        // that rebuild can still receive its scalar prologue. Prologue
        // lowering itself filters this list to kernels with emitted patches.
        for (const ConSanProgramContainer &kernel : program_inventory.kernels()) {
          if (kernel.has_text_range)
            prologue_scratch.push_back({kernel.descriptor_file_offset, 0u});
        }
        assign_scalar_persistent_state(base);
        options.moi_initialize_owner_epoch = true;
        prologue_scratch_assignments = std::move(prologue_scratch);
        diagnostics.push_back(scalar_state_description + std::move(diagnostic_suffix));
      };
      if (scalar_tail) {
        commit_exact_workgroup_scalar_state(*scalar_tail, " state for CDNA exact-coordinate mode");
        return update;
      }

      // A linked kernel can reference a high SGPR transiently while leaving a
      // lower window completely untouched throughout every owning kernel and
      // shared helper. Persistent state may use such a hole only after proving
      // that the pair is absent from the complete owner scopes; site-local
      // liveness is insufficient because the values cross arbitrary guest
      // code between the entry prologue and every probe.
      ConSanMoiOperatingPoint scalar_options = options;
      const bool scalar_state_valid =
          resolved_contexts.has_value() &&
          apply_moi_transient_sgpr_assignment(request, scalar_options, options, scalar_owners);
      bool every_owner_resolved = scalar_state_valid;
      MoiOwnerRegisterReferenceSummary scalar_references =
          summarize_moi_owner_register_references(scalar_contexts);
      bool used_complete_text_summary = false;
      if (resolved_contexts && !scalar_references.control_flow_closed) {
        // Some generated kernels use indirect local-label control flow whose
        // recovered target set is absent or explicitly non-exhaustive. When
        // declared function ranges cover the whole text section, scanning all
        // of them is a conservative substitute for a closed owner CFG:
        // unrelated code may reject a hole, but omitted destinations cannot
        // make an occupied register look free.
        if (auto complete = summarize_moi_complete_text_register_references(scalar_state)) {
          scalar_references = std::move(*complete);
          used_complete_text_summary = true;
        }
      }
      for (const MoiKernelResourceContext *context : scalar_contexts) {
        const uint64_t descriptor_offset = context->kernel->descriptor_file_offset;
        if (context->kernel->uses_dynamic_stack.value_or(false)) {
          scalar_references.references.expand({RegClass::SGPR, kDynamicStackTopSgpr, 1u});
          scalar_references.references.expand({RegClass::SGPR, kDynamicFrameBaseSgpr, 1u});
        }
        const auto descriptor = read_kernel_descriptor(bytes, descriptor_offset);
        if (!descriptor) {
          every_owner_resolved = false;
          break;
        }
        const uint32_t initialized_count = moi_descriptor_user_sgpr_count(*descriptor) +
                                           moi_descriptor_system_sgpr_count(*descriptor);
        for (uint32_t sgpr = 0u;
             sgpr < std::min<uint32_t>(initialized_count, moi_ordinary_sgpr_limit(arch)); ++sgpr) {
          scalar_references.references.expand({RegClass::SGPR, static_cast<uint16_t>(sgpr), 1u});
        }
      }
      std::optional<uint16_t> persistent_hole;
      // A closed KernelCfgScope includes every direct edge and every member of
      // an explicitly exhaustive recovered target set. Unrelated linked
      // kernels must not veto a safe owner-local hole, while an open control
      // transfer or dynamic scalar index still makes the proof fail closed.
      if (every_owner_resolved && scalar_references.control_flow_closed &&
          !scalar_references.uses_indirect_sgprs) {
        uint16_t hole_limit =
            static_cast<uint16_t>(std::min<uint32_t>(first, moi_ordinary_sgpr_limit(arch)));
        for (const MoiKernelResourceContext *context : scalar_contexts)
          hole_limit = std::min(hole_limit, context->current_sgpr_count);
        MoiScalarPlacementDomain hole_domain(arch, hole_limit, resolved_contexts->summaries,
                                             &scalar_references.references);
        occupy_moi_transient_scalar_state(hole_domain, request, resources, scalar_options, options,
                                          problem.mode_semantics(), arch);
        persistent_hole = hole_domain.select_first(0u, scalar_count);
      }
      if (persistent_hole) {
        commit_exact_workgroup_scalar_state(
            *persistent_hole,
            " state to a proven-unused CDNA SGPR window below the maximum guest reference" +
                std::string(used_complete_text_summary ? " using complete-text coverage" : ""));
        return update;
      }
      // Some full-pressure CDNA kernels leave independently proven holes in
      // both register files without leaving five consecutive registers in
      // either one. Owner and epoch are vector-uniform values, so retaining
      // them in an unused ordinary VGPR pair while retaining the exact
      // workgroup tuple in an unused SGPR triple is semantically identical to
      // either homogeneous representation. Require whole-owner reference and
      // indirect-index closure for both files; this is not a calling-
      // convention or preserved-register assumption.
      if (!has_explicit_owner_epoch && needs_exact_workgroup_tuple && every_owner_resolved &&
          scalar_references.control_flow_closed && !scalar_references.uses_indirect_sgprs &&
          !scalar_references.uses_indirect_vgprs) {
        constexpr uint16_t kWorkgroupScalarCount = 3u;
        constexpr uint16_t kOwnerEpochVectorCount = 2u;
        MoiScalarPlacementDomain workgroup_domain(arch, moi_ordinary_sgpr_limit(arch),
                                                  resolved_contexts->summaries,
                                                  &scalar_references.references);
        occupy_moi_transient_scalar_state(workgroup_domain, request, resources, scalar_options,
                                          options, problem.mode_semantics(), arch);
        const auto workgroup_scalar_base = workgroup_domain.select_first(0u, kWorkgroupScalarCount);

        uint16_t common_ordinary_vgpr_limit = kMaxVgprs;
        for (const MoiKernelResourceContext *context : scalar_contexts)
          common_ordinary_vgpr_limit =
              std::min(common_ordinary_vgpr_limit, context->ordinary_vgpr_limit);
        // Automatic site-local scratch plans are provisional here and are
        // rebuilt after persistent placement, at which point the allocator
        // excludes this selected pair. A caller-pinned scratch base cannot
        // move during that rebuild, so exclude its widest supported window
        // now.
        const auto vector_pair_unavailable = [&](uint16_t base) {
          for (uint16_t index = 0u; index < kOwnerEpochVectorCount; ++index) {
            if (scalar_references.references.contains(
                    {RegClass::VGPR, static_cast<uint16_t>(base + index), 1u})) {
              return true;
            }
          }
          if (!debug.scratch_vgpr)
            return false;
          const uint16_t fixed_scratch_count =
              std::accumulate(resource_plans.begin(), resource_plans.end(), uint16_t{0},
                              [](uint16_t count, const auto &plan) {
                                return plan.source == ConSanRegisterAllocationSource::Unsupported
                                           ? count
                                           : std::max(count, plan.scratch_vgpr_count);
                              });
          return range_overlaps(base, kOwnerEpochVectorCount, *debug.scratch_vgpr,
                                fixed_scratch_count);
        };
        const auto owner_epoch_vector_base = first_admitted_moi_register_range(
            0u, kOwnerEpochVectorCount, common_ordinary_vgpr_limit, 1u,
            [&](uint16_t candidate) { return !vector_pair_unavailable(candidate); });

        if (workgroup_scalar_base && owner_epoch_vector_base) {
          const uint16_t scalar_base = *workgroup_scalar_base;
          const uint16_t vector_base = *owner_epoch_vector_base;
          options.set_moi_owner_epoch_vgprs(vector_base, static_cast<uint16_t>(vector_base + 1u));
          options.moi_persistent_sgprs.exact_workgroup = ConSanMoiPersistentWorkgroupRegisters{
              scalar_base, static_cast<uint16_t>(scalar_base + 1u),
              static_cast<uint16_t>(scalar_base + 2u)};
          options.moi_initialize_owner_epoch = true;
          options.automatic_moi_persistent_vgprs = true;
          prologue_scratch_assignments.clear();
          diagnostics.emplace_back(
              "ConSan MOI assigned hybrid persistent vector owner/epoch and scalar "
              "exact-workgroup-tuple state to proven-unused CDNA register holes" +
              std::string(used_complete_text_summary ? " using complete-text coverage" : ""));
          return update;
        }
      }
      if (!scalar_state_valid) {
        diagnostics.emplace_back(
            "ConSan MOI could not build owner scopes for a proven-unused persistent SGPR window");
      } else if (!every_owner_resolved) {
        diagnostics.emplace_back(
            "ConSan MOI could not resolve every owner descriptor for a proven-unused persistent "
            "SGPR window");
      } else if (!scalar_references.control_flow_closed) {
        std::string control_description;
        switch (scalar_references.first_unclosed_control_kind) {
        case MoiUnclosedControlKind::Call:
          control_description = "call ";
          break;
        case MoiUnclosedControlKind::Branch:
          control_description = "branch ";
          break;
        case MoiUnclosedControlKind::Fallthrough:
          control_description = "fallthrough after ";
          break;
        }
        diagnostics.emplace_back(
            "ConSan MOI cannot prove a persistent SGPR hole across unresolved guest " +
            control_description + scalar_references.first_unclosed_control_mnemonic +
            " at text offset " +
            std::to_string(scalar_references.first_unclosed_control_offset.value_or(0u)));
      } else if (scalar_references.uses_indirect_sgprs) {
        diagnostics.emplace_back(
            "ConSan MOI cannot prove a persistent SGPR hole across scalar-relative guest access");
      } else {
        diagnostics.emplace_back(
            "ConSan MOI found no persistent SGPR window unused by every admitted owner");
      }
      if (!has_dynamic_stack_owner) {
        options.moi_initialize_owner_epoch = true;
        options.automatic_moi_private_epoch = true;
        diagnostics.emplace_back("ConSan MOI retained exact entry state in private storage below a "
                                 "CDNA AccVGPR boundary");
        return update;
      }
      // A dynamic stack makes fixed private offsets non-persistent, but it
      // does not make the encoded ordinary-VGPR boundary immutable. Continue
      // into the component planner: trusted metadata can prove an empty
      // accumulator bank and move that boundary for an owner-local VGPR
      // tuple. A component with real accumulators still fails closed against
      // its hard limit below.
      diagnostics.emplace_back(
          "ConSan MOI deferred dynamic-stack exact entry state below a CDNA AccVGPR "
          "boundary to owner-local persistent VGPR placement");
    }

    // Persistent placement and resource planning form a fixed point. Failed
    // plans still contribute ownership so a smaller component-local ABI can
    // recover them on the subsequent resource-plan rebuild.
    const std::vector<std::vector<uint64_t>> components =
        moi_scalar_owner_components(resource_plans);

    std::string scalar_persistence_failure;
    const auto try_assign_scalar_persistent_state = [&] {
      // Owner, epoch, and the optional workgroup key are uniform within a
      // wave. A fresh SGPR tail preserves them across arbitrary guest code
      // without moving ACCUM_OFFSET or adding a private-memory transaction at
      // every barrier. Prove the choice across the complete owner union: this
      // is a code-object-wide representation, not a component-local shortcut.
      const uint16_t scalar_count = static_cast<uint16_t>(named_persistent_count);
      const std::vector<uint64_t> scalar_owners = moi_scalar_owner_descriptors(resource_plans);
      const MoiResourcePlanningState &state = planning_state;
      const auto resolved_contexts = resolve_moi_scalar_owner_contexts(state, scalar_owners);
      if (!resolved_contexts) {
        scalar_persistence_failure = "could not resolve every scalar-state owner descriptor";
        return false;
      }
      auto prologue_scratch = plan_moi_scalar_prologue_scratch(state, *resolved_contexts,
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
      const uint16_t scalar_limit = moi_ordinary_sgpr_limit(arch);
      MoiScalarPlacementDomain scalar_domain(arch, scalar_limit, resolved_contexts->summaries);
      occupy_moi_transient_scalar_state(scalar_domain, request, resources, options, options,
                                        problem.mode_semantics(), arch);
      scalar_domain.occupy_dynamic_stack_state();
      const auto scalar_tail = scalar_domain.select_first(first, scalar_count);
      if (scalar_tail) {
        // A prior disconnected component may already have accumulated a VGPR
        // tuple. Commit the scalar transition only after entry scratch has
        // been proven for the complete union.
        options.owner_persistent_vgprs.clear();
        assign_scalar_persistent_state(*scalar_tail);
        options.moi_initialize_owner_epoch = true;
        prologue_scratch_assignments = std::move(*prologue_scratch);
        diagnostics.emplace_back("ConSan MOI assigned persistent scalar owner/epoch" +
                                 std::string(needs_workgroup_key ? "/workgroup-key" : "") +
                                 " state s" + std::to_string(*scalar_tail) + ":s" +
                                 std::to_string(*scalar_tail + scalar_count - 1u) +
                                 " below a full CDNA ordinary VGPR bank for " +
                                 consan_moi_engine_name(request.moi_engine));
        return true;
      }
      scalar_persistence_failure =
          "found no persistent scalar tail outside guest and instrumentation state";
      return false;
    };

    bool assigned_private_component = false;
    for (const std::vector<uint64_t> &component : components) {
      uint32_t component_base = 0u;
      uint32_t component_limit = kMaxVgprs;
      uint32_t component_encoded_limit = kMaxVgprs;
      bool resolved_every_component_context = true;
      ConSanMoiOperatingPoint component_options = options;
      const bool component_transient_state_resolved =
          options.owner_transient_sgprs.empty() ||
          apply_moi_transient_sgpr_assignment(request, component_options, options, component);
      const bool component_needs_persistent_dispatch_vgprs =
          (needs_persistent_dispatch_vgprs || options.moi_dispatch_identity.private_fallback()) &&
          component_transient_state_resolved &&
          !moi_has_runtime_hardware_dispatch_id(component_options);
      const uint32_t component_persistent_count =
          persistent_count +
          (component_needs_persistent_dispatch_vgprs && !needs_persistent_dispatch_vgprs ? 2u : 0u);
      for (uint64_t descriptor_offset : component) {
        const auto found = planning_state.context_by_descriptor.find(descriptor_offset);
        if (found == planning_state.context_by_descriptor.end() || found->second == nullptr) {
          resolved_every_component_context = false;
          continue;
        }
        const MoiKernelResourceContext &context = *found->second;
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
      for (const ConSanCandidateResourcePlan &plan : resource_plans) {
        if (std::ranges::none_of(plan.owner_descriptor_file_offsets, [&](uint64_t offset) {
              return std::ranges::find(component, offset) != component.end();
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
          component_base + component_persistent_count > component_encoded_limit;
      // Scalar state avoids descriptor growth even when trusted metadata says
      // an empty accumulator boundary can move. Prefer that smaller, more
      // conservative representation when it is legal across the complete
      // owner union; component-local VGPR growth remains the fallback when a
      // different owner makes the global scalar tail unavailable.
      if (crosses_encoded_component_boundary && persistent_demand.scalar_state_supported &&
          !debug.test_force_private_epoch && !component_needs_persistent_dispatch_vgprs &&
          try_assign_scalar_persistent_state()) {
        return update;
      }
      if (component_base + component_persistent_count > component_limit) {
        const bool has_dynamic_stack_owner =
            std::ranges::any_of(component, [&](uint64_t descriptor_offset) {
              const ConSanProgramContainer *kernel =
                  program_inventory.find_kernel_by_descriptor(descriptor_offset);
              return kernel != nullptr && kernel->uses_dynamic_stack.value_or(false);
            });
        if (!crosses_encoded_component_boundary && persistent_demand.scalar_state_supported &&
            !debug.test_force_private_epoch && !component_needs_persistent_dispatch_vgprs &&
            try_assign_scalar_persistent_state()) {
          return update;
        }
        if (persistent_demand.scalar_state_supported && has_dynamic_stack_owner) {
          update.rejection = ConSanMoiPlacementRejection::PersistentStateUnavailable;
          diagnostics.emplace_back(
              "ConSan MOI cannot place persistent scalar state (" + scalar_persistence_failure +
              ") for a dynamic-stack CDNA owner with a full ordinary VGPR bank: component " +
              std::to_string(component_base) + ":" +
              std::to_string(component_base + component_persistent_count) + " exceeds limit " +
              std::to_string(component_limit));
          return update;
        }
        if (!has_dynamic_stack_owner && persistent_demand.private_state_supported) {
          options.moi_initialize_owner_epoch = true;
          options.automatic_moi_private_epoch = true;
          assigned_private_component = true;
          diagnostics.emplace_back(
              "ConSan MOI selected component-local entry-snapshotted private persistent state "
              "below a full CDNA ordinary VGPR bank after scalar placement " +
              (scalar_persistence_failure.empty() ? std::string("was not requested")
                                                  : scalar_persistence_failure));
          continue;
        }
        update.rejection = ConSanMoiPlacementRejection::PersistentStateUnavailable;
        if (persistent_demand.scalar_state_supported &&
            persistent_demand.prefer_private_epoch_for_descriptor_growth &&
            !scalar_persistence_failure.empty()) {
          diagnostics.emplace_back(
              "ConSan MOI cannot place persistent scalar state (" + scalar_persistence_failure +
              ") or persistent VGPR state below a connected CDNA AccVGPR boundary");
        } else {
          diagnostics.emplace_back(
              "ConSan MOI cannot place persistent VGPR state below a connected CDNA AccVGPR "
              "boundary");
        }
        return update;
      }
      for (uint64_t descriptor_offset : component) {
        ConSanMoiPersistentVgprAssignment assignment;
        assignment.descriptor_file_offset = descriptor_offset;
        uint16_t next = static_cast<uint16_t>(component_base);
        if (has_explicit_owner_epoch) {
          assignment.owner_epoch_vgprs = ConSanMoiOwnerEpochRegisters{
              options.moi_owner_epoch_vgprs->owner, options.moi_owner_epoch_vgprs->epoch};
        } else {
          assignment.owner_epoch_vgprs =
              ConSanMoiOwnerEpochRegisters{next, static_cast<uint16_t>(next + 1u)};
          next = static_cast<uint16_t>(next + 2u);
        }
        if (needs_workgroup_key)
          assignment.workgroup_key_vgpr = next++;
        if (needs_exact_workgroup_tuple) {
          const uint16_t x = next++;
          const uint16_t y = next++;
          const uint16_t z = next++;
          assignment.exact_workgroup_vgprs = ConSanMoiPersistentWorkgroupRegisters{
              x, y, z, needs_cluster_workgroup_id ? std::optional<uint16_t>(next++) : std::nullopt};
        }
        if (component_needs_persistent_dispatch_vgprs) {
          assignment.dispatch_id_vgpr = next;
          next = static_cast<uint16_t>(next + 2u);
        }
        options.owner_persistent_vgprs.push_back(assignment);
      }
    }
    options.moi_initialize_owner_epoch = true;
    options.automatic_moi_persistent_vgprs = !options.owner_persistent_vgprs.empty();
    const bool assigned_persistent_dispatch_vgprs = std::ranges::any_of(
        options.owner_persistent_vgprs, [](const ConSanMoiPersistentVgprAssignment &assignment) {
          return assignment.dispatch_id_vgpr.has_value();
        });
    if (assigned_persistent_dispatch_vgprs) {
      if (!assigned_private_component)
        options.moi_dispatch_identity.set_private_fallback(false);
    }
    diagnostics.emplace_back(
        assigned_private_component && options.automatic_moi_persistent_vgprs
            ? "ConSan MOI assigned component-local private and persistent VGPR state below CDNA "
              "AccVGPR boundaries"
        : assigned_private_component
            ? "ConSan MOI assigned component-local private state below CDNA AccVGPR "
              "boundaries"
            : "ConSan MOI assigned per-owner persistent VGPR tuples below CDNA AccVGPR "
              "boundaries");
    return update;
  }
  const bool would_grow_descriptor =
      std::ranges::any_of(resource_plans, [&](const ConSanCandidateResourcePlan &plan) {
        return !plan.owner_descriptor_file_offsets.empty() &&
               plan.source != ConSanRegisterAllocationSource::Unsupported &&
               persistent_end > plan.current_vgpr_count;
      });
  const bool has_dynamic_stack_owner = has_dynamic_stack_resource_owner();
  // Private epoch storage is address-free scratch and therefore cannot be
  // composed with a guest dynamic stack. If persistent state fits in the
  // architectural VGPR file, grow those owner descriptors instead. The
  // rebuilt resource plans reserve the selected tail VGPRs before allocating
  // per-site scratch, so this growth cannot overlap instrumentation scratch.
  // A bounded Record/Replay barrier inventory is another exception: without
  // persistent epoch state it requires one dynamic record per participating
  // workitem and barrier execution. That can turn a small static inventory
  // into millions of records. Prefer bounded descriptor growth so those
  // barriers use the compact inline-epoch path instead.
  // InlineShadow reads owner state on every instrumented LDS access. A
  // scratch-backed owner made a complete multi-kernel CLIP invocation spend
  // more than a minute repeatedly loading that state, whereas two persistent
  // tail VGPRs complete the same clean row in about 1.5 seconds. Prefer the
  // descriptor-backed representation whenever it fits; private state remains
  // the capacity fallback for near-full VGPR files. Dynamic-LDS kernels are an
  // exception: their external exact-shadow path exercises arbitrary guest
  // VGPR-bank transitions, so entry-snapshotted private state avoids treating
  // a remapped tail VGPR as a different owner while retaining the same exact
  // shadow semantics.
  const bool prefer_private_epoch =
      !has_dynamic_stack_owner && !persistent_demand.prefer_compact_barriers &&
      !needs_workgroup_key && !has_explicit_owner_epoch &&
      (!needs_exact_workgroup_tuple || persistent_demand.private_workgroup_tuple_supported) &&
      !persistent_demand.synchronization_requires_persistent_owner &&
      (persistent_demand.prefer_private_epoch_for_dynamic_lds ||
       (would_grow_descriptor && persistent_demand.prefer_private_epoch_for_descriptor_growth));
  const bool rdna_needs_scalar_epoch =
      consan_arch_is_rdna3_rdna4_or_cdna5(arch) && !debug.test_force_private_epoch &&
      persistent_demand.scalar_state_required_for_private_or_overflow &&
      (persistent_end > kMaxVgprs || prefer_private_epoch);
  if (rdna_needs_scalar_epoch) {
    const std::string rdna_arch_name = rj_code_arch_name(arch);
    if (needs_persistent_dispatch_vgprs) {
      update.rejection = ConSanMoiPlacementRejection::PersistentStateUnavailable;
      diagnostics.emplace_back(
          "ConSan MOI cannot retain automatic Record/Replay dispatch identity without a "
          "persistent VGPR pair");
      return update;
    }
    const uint16_t scalar_count = static_cast<uint16_t>(named_persistent_count);
    const MoiResourcePlanningState &scalar_state = planning_state;
    const std::vector<uint64_t> scalar_owners = moi_scalar_owner_descriptors(resource_plans);
    const auto resolved_contexts = resolve_moi_scalar_owner_contexts(scalar_state, scalar_owners);
    if (!resolved_contexts) {
      if (has_dynamic_stack_owner) {
        update.rejection = ConSanMoiPlacementRejection::PersistentStateUnavailable;
        diagnostics.emplace_back("ConSan MOI could not resolve every " + rdna_arch_name +
                                 " dynamic-stack scalar-state owner");
        return update;
      }
      diagnostics.emplace_back("ConSan MOI could not resolve every " + rdna_arch_name +
                               " scalar-state owner");
      return update;
    }
    auto prologue_scratch = plan_moi_scalar_prologue_scratch(
        scalar_state, *resolved_contexts, persistent_demand.scalar_state_supported);
    if (!prologue_scratch) {
      if (has_dynamic_stack_owner)
        update.rejection = ConSanMoiPlacementRejection::PersistentStateUnavailable;
      diagnostics.emplace_back("ConSan MOI could not place entry-local VGPR scratch for " +
                               rdna_arch_name + " scalar persistent state");
      return update;
    }
    MoiOwnerRegisterReferenceSummary scalar_references =
        summarize_moi_owner_register_references(resolved_contexts->contexts);
    if (!scalar_references.control_flow_closed) {
      if (auto complete = summarize_moi_complete_text_register_references(scalar_state))
        scalar_references = std::move(*complete);
    }
    const uint16_t scalar_limit = moi_ordinary_sgpr_limit(arch);
    MoiScalarPlacementDomain tail_domain(arch, scalar_limit, resolved_contexts->summaries);
    occupy_moi_transient_scalar_state(tail_domain, request, resources, options, options,
                                      problem.mode_semantics(), arch);
    if (has_dynamic_stack_owner)
      tail_domain.occupy_dynamic_stack_state();
    std::optional<uint16_t> scalar_base =
        tail_domain.select_first(resolved_contexts->tail_floor, scalar_count);
    const uint32_t hole_limit = std::min<uint32_t>(resolved_contexts->tail_floor, scalar_limit);
    if (has_dynamic_stack_owner && !scalar_base && scalar_references.control_flow_closed &&
        !scalar_references.uses_indirect_sgprs) {
      MoiScalarPlacementDomain hole_domain(arch, static_cast<uint16_t>(hole_limit),
                                           resolved_contexts->summaries,
                                           &scalar_references.references);
      occupy_moi_transient_scalar_state(hole_domain, request, resources, options, options,
                                        problem.mode_semantics(), arch);
      hole_domain.occupy_dynamic_stack_state();
      scalar_base = hole_domain.select_first(0u, scalar_count);
    }
    if (scalar_base) {
      const uint16_t base = *scalar_base;
      assign_scalar_persistent_state(base);
      options.moi_initialize_owner_epoch = true;
      prologue_scratch_assignments = std::move(*prologue_scratch);
      diagnostics.emplace_back(
          "ConSan MOI assigned persistent scalar " +
          std::string(has_explicit_owner_epoch ? "" : "owner/epoch") +
          std::string(needs_workgroup_key ? "/workgroup-key" : "") +
          std::string(needs_exact_workgroup_tuple ? "/exact-workgroup-tuple" : "") +
          " state for an " + rdna_arch_name + " " + consan_moi_engine_name(request.moi_engine) +
          " owner without growing the VGPR allocation");
      return update;
    }
    if (has_dynamic_stack_owner) {
      update.rejection = ConSanMoiPlacementRejection::PersistentStateUnavailable;
      diagnostics.emplace_back(
          std::string("ConSan MOI could not place persistent scalar owner/epoch state for an ") +
          rdna_arch_name + " dynamic-stack " + consan_moi_engine_name(request.moi_engine) +
          " owner");
      return update;
    }
    if (needs_exact_workgroup_tuple) {
      diagnostics.emplace_back(
          "ConSan MOI could not place the exact workgroup tuple in persistent " + rdna_arch_name +
          " scalar state; retaining it in entry-captured private state");
    }
    diagnostics.emplace_back(
        std::string("ConSan MOI could not place persistent scalar owner/epoch state for an ") +
        rdna_arch_name + " " + consan_moi_engine_name(request.moi_engine) +
        " owner; retaining the existing private-state fallback");
  }
  const bool needs_private_epoch = options.automatic_moi_private_epoch ||
                                   debug.test_force_private_epoch || persistent_end > kMaxVgprs ||
                                   prefer_private_epoch;
  if (needs_private_epoch && !has_dynamic_stack_owner) {
    options.moi_initialize_owner_epoch = true;
    options.automatic_moi_private_epoch = true;
    diagnostics.emplace_back(
        std::string("ConSan MOI automatically selected derived-owner/private-"
                    "epoch persistent state") +
        (would_grow_descriptor ? " to preserve the guest VGPR allocation" : ""));
    return update;
  }
  if (needs_private_epoch) {
    update.rejection = ConSanMoiPlacementRejection::PersistentStateUnavailable;
    diagnostics.emplace_back(
        "ConSan MOI cannot retain exact entry state for a dynamic-stack owner without a "
        "persistent register window");
    return update;
  }

  uint16_t next_persistent_vgpr = static_cast<uint16_t>(persistent_base);
  if (!has_explicit_owner_epoch) {
    const uint16_t owner_vgpr = next_persistent_vgpr++;
    const uint16_t epoch_vgpr = next_persistent_vgpr++;
    options.set_moi_owner_epoch_vgprs(owner_vgpr, epoch_vgpr);
  }
  if (needs_workgroup_key)
    options.moi_workgroup_key_vgpr = next_persistent_vgpr++;
  if (needs_exact_workgroup_tuple) {
    const uint16_t x = next_persistent_vgpr++;
    const uint16_t y = next_persistent_vgpr++;
    const uint16_t z = next_persistent_vgpr++;
    options.moi_exact_workgroup_vgprs = ConSanMoiPersistentWorkgroupRegisters{
        x, y, z,
        needs_cluster_workgroup_id ? std::optional<uint16_t>(next_persistent_vgpr++)
                                   : std::nullopt};
  }
  if (needs_persistent_dispatch_vgprs) {
    options.moi_dispatch_identity.set_vgpr(next_persistent_vgpr);
    next_persistent_vgpr = static_cast<uint16_t>(next_persistent_vgpr + 2u);
    options.moi_dispatch_identity.set_private_fallback(false);
  }
  options.moi_initialize_owner_epoch = true;
  options.automatic_moi_persistent_vgprs = true;
  diagnostics.emplace_back(
      "ConSan MOI automatically assigned persistent " +
      std::string(has_explicit_owner_epoch ? "" : "owner/epoch") +
      std::string(needs_workgroup_key ? "/workgroup-key" : "") +
      std::string(needs_exact_workgroup_tuple ? "/exact-workgroup-tuple" : "") +
      std::string(needs_persistent_dispatch_vgprs ? "/dispatch-ID" : "") + " VGPRs v" +
      std::to_string(persistent_base) + ":v" + std::to_string(next_persistent_vgpr - 1u) +
      (would_grow_descriptor && has_dynamic_stack_owner
           ? " with descriptor growth because private epoch storage is incompatible with a "
             "dynamic-stack owner"
           : ""));
  return update;
}

void note_descriptor_requirements(MoiDescriptorVgprRequirements &requirements,
                                  const ResolvedMoiScratchPlan &plan) {
  note_maximum_descriptor_extents(requirements, plan.owner_descriptor_file_offsets,
                                  plan.required_vgpr_count);
}

void note_moi_sgpr_requirements(MoiDescriptorSgprRequirements &requirements,
                                const ResolvedMoiScratchPlan &plan, const ConSanRequest &request,
                                const BoundRuntimeResources &resources,
                                const ConSanMoiOperatingPoint &point,
                                const MoiObjectModeSemantics &mode_semantics, rj_code_arch_t arch) {
  uint16_t required_count = 0;
  const auto note_range = [&](std::optional<uint16_t> base, uint16_t width) {
    if (base)
      required_count = std::max<uint16_t>(required_count, static_cast<uint16_t>(*base + width));
  };
  note_range(
      point.moi_exec_save_sgpr,
      moi_exec_save_sgpr_count(
          resolve_moi_exec_save_requirement(request, resources, point, mode_semantics), arch));
  if (request.moi_owner_source == ConSanMoiOwnerSource::HwId)
    note_range(point.moi_owner_sgpr.base(), 1u);
  note_range(point.moi_dispatch_identity.sgpr(), kMoiDispatchStateSgprCount);
  point.moi_persistent_sgprs.for_each_range(note_range);
  point.for_each_auxiliary_sgpr_range(note_range);
  if (required_count == 0)
    return;
  note_maximum_descriptor_extents(requirements, plan.owner_descriptor_file_offsets, required_count);
}

void note_spill_descriptor_requirements(MoiDescriptorPrivateRequirements &requirements,
                                        const ResolvedMoiScratchPlan &plan,
                                        const VgprSpillSequence &spill) {
  note_maximum_descriptor_extents(requirements, plan.owner_descriptor_file_offsets,
                                  spill.total_private_bytes);
}

void note_dynamic_stack_private_requirement(ConSanPatchAbiEffects &effects,
                                            const VgprSpillSequence *spill) {
  if (spill == nullptr || !spill->uses_dynamic_stack_frame)
    return;
  effects.dynamic_private_segment_addend =
      std::max(effects.dynamic_private_segment_addend, spill->dynamic_frame_bytes);
}

/// Accumulate the fixed private-memory requirement owned by one planned MOI
/// access body.
///
/// Access engines can attach different evidence to a body, but descriptor
/// growth depends only on the common plan's kernel owners and maximum private
/// extent. Keeping that rule here prevents admission loops from evolving
/// engine-specific interpretations of the same allocation.
void note_moi_lds_requirements(MoiDescriptorLdsRequirements &requirements,
                               const ResolvedMoiScratchPlan &plan,
                               const ConSanMoiWorkgroupShadowLayout &layout) {
  note_maximum_descriptor_extents(requirements, plan.owner_descriptor_file_offsets,
                                  layout.required_group_segment_size);
}
