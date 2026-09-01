[[nodiscard]] std::optional<uint16_t>
operational_atomic_scratch_count(ConSanProbeIntentKind evidence,
                                 const ConSanAtomicLoweringForm &form, rj_code_arch_t arch) {
  switch (evidence) {
  case ConSanProbeIntentKind::AtomicRecord:
    return atomic_record_scratch_count(form);
  case ConSanProbeIntentKind::SampledAtomicOrdering:
    return sampled_atomic_scratch_count();
  case ConSanProbeIntentKind::ExactAtomicOrdering:
    return inline_shadow_atomic_scratch_count(arch);
  default:
    return std::nullopt;
  }
}

void append_moi_operational_resource_plans(MoiResourcePlanningState &state,
                                           const ConSanDebugOverrides &debug,
                                           const ConSanMoiOperatingPoint &point,
                                           const MoiResourceProblem &problem,
                                           std::vector<std::string> &errors,
                                           std::vector<ConSanCandidateResourcePlan> &plans) {
  const ConSanRequest &request = problem.request();
  const BoundRuntimeResources &bound_resources = problem.resources();
  const MoiObjectModeSemantics &mode_semantics = problem.mode_semantics();
  const MoiOperationalEvidenceKinds &evidence =
      moi_mode_operations(request.moi_engine).operational_evidence;
  if (request.moi_track_barriers) {
    std::vector<MoiBarrierEvidenceSitePlan> candidates = build_moi_barrier_evidence_site_plans(
        problem.inventory(), problem.observation_plan(), evidence.barrier, errors);
    if (!errors.empty())
      return;
    for (size_t index = 0; index < candidates.size(); ++index) {
      const MoiBarrierEvidenceSitePlan &candidate = candidates[index];
      const auto scratch_count_for = [&](const ConSanMoiOperatingPoint &site_point) {
        return operational_barrier_scratch_count(evidence.barrier, bound_resources, site_point,
                                                 mode_semantics);
      };
      ConSanMoiOperatingPoint site_point = point;
      const std::optional<uint16_t> scratch_count = scratch_count_for(site_point);
      if (!scratch_count) {
        errors.emplace_back("ConSan MOI mode selected an invalid barrier evidence operation");
        return;
      }
      ConSanCandidateResourcePlan plan = plan_moi_resource_site(
          state, request, debug, site_point, ConSanResourceSiteKind::Barrier, index,
          candidate.site.text_offset, candidate.kernel_descriptor_file_offset, *scratch_count);
      ConSanMoiOperatingPoint owner_point = site_point;
      if (apply_moi_persistent_vgpr_assignment(owner_point, point,
                                               plan.owner_descriptor_file_offsets)) {
        const std::optional<uint16_t> owner_scratch_count = scratch_count_for(owner_point);
        if (!owner_scratch_count) {
          errors.emplace_back("ConSan MOI mode selected an invalid barrier evidence operation");
          return;
        }
        if (*owner_scratch_count != *scratch_count) {
          plan = plan_moi_resource_site(
              state, request, debug, owner_point, ConSanResourceSiteKind::Barrier, index,
              candidate.site.text_offset, candidate.kernel_descriptor_file_offset,
              *owner_scratch_count);
        }
      }
      plan.intent_ids = {candidate.evidence_intent};
      plans.push_back(std::move(plan));
    }
  }
  if (request.moi_track_atomics) {
    std::vector<MoiAtomicEvidenceSitePlan> candidates = build_moi_atomic_evidence_site_plans(
        problem.inventory(), problem.observation_plan(), evidence.atomic, errors);
    if (!errors.empty())
      return;
    for (size_t index = 0; index < candidates.size(); ++index) {
      const MoiAtomicEvidenceSitePlan &candidate = candidates[index];
      const std::optional<uint16_t> scratch_count = operational_atomic_scratch_count(
          evidence.atomic, candidate.lowering_form, moi_resource_arch(state));
      if (!scratch_count) {
        errors.emplace_back("ConSan MOI mode selected an invalid atomic evidence operation");
        return;
      }
      ConSanCandidateResourcePlan plan = plan_moi_resource_site(
          state, request, debug, point, ConSanResourceSiteKind::Atomic, index,
          candidate.site.text_offset, candidate.kernel_descriptor_file_offset, *scratch_count,
          nullptr, &candidate.lowering_form);
      plan.intent_ids = {candidate.address_capture_intent, candidate.evidence_intent};
      plans.push_back(std::move(plan));
    }
  }
  if (request.moi_track_atomics && evidence.fence) {
    std::vector<MoiFenceEvidenceSitePlan> candidates = build_moi_fence_evidence_site_plans(
        problem.inventory(), problem.observation_plan(), errors);
    if (!errors.empty())
      return;
    for (size_t index = 0; index < candidates.size(); ++index) {
      const MoiFenceEvidenceSitePlan &candidate = candidates[index];
      ConSanCandidateResourcePlan plan = plan_moi_resource_site(
          state, request, debug, point, ConSanResourceSiteKind::Fence, index,
          candidate.patch_text_offset, candidate.kernel_descriptor_file_offset,
          fence_record_scratch_count(candidate.communication_lowering_form), nullptr,
          &candidate.communication_lowering_form);
      if (candidate.patch_text_offset != candidate.semantic_site.physical.original_text_offset)
        plan.semantic_text_offset = candidate.semantic_site.physical.original_text_offset;
      plan.intent_ids = {candidate.address_capture_intent, candidate.evidence_intent};
      plans.push_back(std::move(plan));
    }
  }
}

/// Close every MOI intent that did not participate in a committed byte
/// mutation. Unsupported resource plans reject the exact intent IDs they
/// governed; all other still-pending intents receive an explicit placement
/// rejection. This terminal path never searches patches, anchors, or graph
/// geometry to rediscover what lowering attempted.
void publish_pending_moi_lowering_rejections(
    ConSanTransformArtifacts &result,
    std::optional<ConSanRegisterPlanReason> whole_transform_resource_failure) {
  for (const ConSanCandidateResourcePlan &plan : result.resource_plans) {
    if (plan.source != ConSanRegisterAllocationSource::Unsupported || plan.intent_ids.empty())
      continue;
    std::vector<ConSanProbeIntentId> pending;
    for (ConSanProbeIntentId id : plan.intent_ids) {
      const ConSanIntentCoverageEntry *entry = result.coverage_ledger.intent_entry(id);
      if (entry != nullptr && entry->lowering == ConSanLoweringOutcomeKind::Pending &&
          std::ranges::find(pending, id) == pending.end()) {
        pending.push_back(id);
      }
    }
    if (pending.empty())
      continue;
    if (!result.coverage_ledger.publish_lowering_rejection(
            pending, ConSanLoweringOutcomeKind::ResourceRejected,
            "MOI resource plan rejected intents: " +
                std::string(consan_register_plan_reason_name(plan.reason)),
            plan.reason)) {
      result.errors.emplace_back("ConSan MOI could not publish a resource rejection");
      return;
    }
  }

  for (const ConSanProbeIntent &intent : result.observation_plan().probe_intents) {
    const ConSanIntentCoverageEntry *entry = result.coverage_ledger.intent_entry(intent.id);
    if (entry == nullptr || entry->lowering != ConSanLoweringOutcomeKind::Pending)
      continue;
    const std::array<ConSanProbeIntentId, 1> ids = {intent.id};
    const ConSanLoweringOutcomeKind outcome = whole_transform_resource_failure
                                                  ? ConSanLoweringOutcomeKind::ResourceRejected
                                                  : ConSanLoweringOutcomeKind::PlacementRejected;
    const std::string_view detail = whole_transform_resource_failure
                                        ? "MOI whole-transform resource validation rejected the "
                                          "admitted intent"
                                        : "MOI lowerer selected no placement for the admitted "
                                          "intent";
    if (!result.coverage_ledger.publish_lowering_rejection(ids, outcome, std::string(detail),
                                                           whole_transform_resource_failure)) {
      result.errors.emplace_back("ConSan MOI could not publish a placement rejection");
      return;
    }
  }
}

/// Derive one complete resource-plan set for the current MOI operating-point
/// attempt. The returned vector is private until every access and operational
/// site has been planned and refined, so a retry cannot expose a half-cleared
/// or half-rebuilt plan set through the coordinator transaction.
[[nodiscard]] ConSanMoiResourcePlanningResult
plan_moi_resources(MoiResourcePlanningState &state, const ConSanDebugOverrides &debug,
                   const ConSanMoiOperatingPoint &point, const MoiResourceProblem &problem) {
  ConSanMoiResourcePlanningResult planning;
  const ConSanMoiOperatingPoint &attempted_point = point;
  const rj_code_arch_t arch = moi_resource_arch(state);
  const ConSanRequest &request = problem.request();
  const BoundRuntimeResources &bound_resources = problem.resources();
  planning.attempted_operating_point = attempted_point;
  append_moi_resource_plans(state, request, bound_resources, debug, point, problem.candidates(),
                            planning.plans);
  append_moi_operational_resource_plans(state, debug, point, problem, planning.errors,
                                        planning.plans);
  if (consan_arch_is_cdna(arch) && point.moi_dispatch_identity.automatic_sgpr() &&
      point.moi_dispatch_identity.sgpr()) {
    for (ConSanCandidateResourcePlan &plan : planning.plans) {
      ConSanMoiOperatingPoint owner_point = point;
      if (!attempted_point.owner_transient_sgprs.empty() &&
          !apply_moi_transient_sgpr_assignment(request, owner_point, attempted_point,
                                               plan.owner_descriptor_file_offsets)) {
        if (consan_arch_uses_per_kernel_owner_translation(arch))
          continue;
        owner_point.moi_dispatch_identity.reset_sgpr();
      }
      if (plan.owner_descriptor_file_offsets.empty())
        continue;
      // A component-local transient assignment may deliberately omit an
      // overlapping persistent dispatch pair. Its established fallback ABI
      // records generation zero; rejecting the resource plan here would make
      // that safe fallback specific to direct code transport even though the
      // report contract is target-neutral.
      if (!owner_point.moi_dispatch_identity.sgpr() &&
          std::ranges::all_of(plan.owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
            const ConSanMoiTransientSgprAssignment *assignment =
                moi_transient_sgpr_assignment(attempted_point, descriptor_offset);
            return assignment != nullptr && !assignment->dispatch_id_sgpr;
          })) {
        continue;
      }
      const bool missing_dispatch = !owner_point.moi_dispatch_identity.sgpr();
      const uint16_t dispatch_base = owner_point.moi_dispatch_identity.sgpr().value_or(0u);
      const bool admitted = !missing_dispatch && moi_resource_owner_sgpr_window_admitted(
                                                     state, plan.owner_descriptor_file_offsets,
                                                     dispatch_base, kMoiDispatchStateSgprCount);
      if (!admitted) {
        plan.source = ConSanRegisterAllocationSource::Unsupported;
        plan.reason = ConSanRegisterPlanReason::ForbiddenOverlap;
        plan.scratch_vgpr.reset();
        plan.required_vgpr_count = plan.current_vgpr_count;
      }
    }
  }
  if (consan_is_capability_arch(arch) && point.automatic_moi_partial_exec_save_sgprs &&
      point.moi_exec_save_sgpr) {
    const uint16_t exec_base = *point.moi_exec_save_sgpr;
    const uint16_t exec_count =
        moi_exec_save_sgpr_count(resolve_moi_exec_save_requirement(request, bound_resources, point,
                                                                   problem.mode_semantics()),
                                 arch);
    for (ConSanCandidateResourcePlan &plan : planning.plans) {
      const bool has_owner_override =
          std::ranges::any_of(plan.owner_descriptor_file_offsets, [&](uint64_t descriptor_offset) {
            return moi_transient_sgpr_assignment(attempted_point, descriptor_offset) != nullptr;
          });
      if (has_owner_override) {
        ConSanMoiOperatingPoint owner_point = point;
        if (apply_moi_transient_sgpr_assignment(request, owner_point, attempted_point,
                                                plan.owner_descriptor_file_offsets))
          continue;
      }
      if (!plan.owner_descriptor_file_offsets.empty() &&
          !moi_resource_owner_sgpr_window_admitted(state, plan.owner_descriptor_file_offsets,
                                                   exec_base, exec_count)) {
        plan.source = ConSanRegisterAllocationSource::Unsupported;
        plan.reason = ConSanRegisterPlanReason::ForbiddenOverlap;
        plan.scratch_vgpr.reset();
        plan.required_vgpr_count = plan.current_vgpr_count;
      }
    }
  }
  return planning;
}

/// Solve automatic transient-scalar placement without publishing intermediate
/// points, plans, or diagnostics. The ordinary attempt and its dynamic-stack
/// retry each own a complete point/plan pair; only the selected attempt is
/// returned to orchestration.
[[nodiscard]] ConSanMoiResourcePlanningResult solve_automatic_moi_exec_save_resources(
    MoiResourcePlanningState &state, const ConSanDebugOverrides &debug,
    const ConSanMoiOperatingPoint &base, const MoiResourceProblem &problem) {
  struct ExecAttempt {
    ConSanMoiResourcePlanningResult planning;
    bool needs_dynamic_stack_retry = false;
  };
  const auto attempt = [&](ConSanMoiOperatingPoint point,
                           std::optional<ConSanMoiFallbackKind> fallback) {
    ExecAttempt exec;
    exec.planning = plan_moi_resources(state, debug, point, problem);
    if (!exec.planning.success())
      return exec;

    std::vector<std::string> diagnostics;
    const bool changed = configure_automatic_moi_exec_save_sgprs(
        point, problem, exec.planning.plans, diagnostics, state);
    exec.needs_dynamic_stack_retry =
        !point.moi_dynamic_stack_spill && automatic_moi_scalar_spill_needs_dynamic_stack_planning(
                                              point, problem.inventory(), exec.planning.plans);
    if (changed)
      exec.planning = plan_moi_resources(state, debug, point, problem);
    exec.planning.attempted_operating_point = point;
    exec.planning.selected_fallback = fallback;
    exec.planning.diagnostics = std::move(diagnostics);
    return exec;
  };

  ExecAttempt selected = attempt(base, std::nullopt);
  if (!selected.planning.success() || !selected.needs_dynamic_stack_retry)
    return std::move(selected.planning);

  ConSanMoiOperatingPoint dynamic_stack_point = base;
  dynamic_stack_point.moi_dynamic_stack_spill = true;
  return std::move(
      attempt(std::move(dynamic_stack_point), ConSanMoiFallbackKind::DynamicStackScalarSpill)
          .planning);
}
