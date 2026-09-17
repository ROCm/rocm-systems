void append_operational_resource_plans(ResourcePlanningState &state, const DebugOverrides &debug,
                                       const OperatingPoint &point, const ResourceProblem &problem,
                                       std::vector<std::string> &errors,
                                       std::vector<CandidateResourcePlan> &plans) {
  const Request &request = problem.request();
  const ModeProbeVocabulary &evidence = *mode_probe_vocabulary(problem.observation_plan().mode);
  if (request.track_barriers) {
    std::vector<BarrierEvidenceSitePlan> candidates = build_barrier_evidence_site_plans(
        problem.inventory(), problem.observation_plan(), evidence.barrier, errors);
    if (!errors.empty())
      return;
    for (size_t index = 0; index < candidates.size(); ++index) {
      const BarrierEvidenceSitePlan &candidate = candidates[index];
      const ProgramSite *source = problem.inventory().program_site(candidate.source_site);
      if (source == nullptr) {
        errors.emplace_back("ConSan barrier resource planning lost its decoded source");
        return;
      }
      const uint64_t text_offset = source->text_offset();
      const auto scratch_count_for = [&](const OperatingPoint &site_point) {
        return barrier_scratch_vgpr_count(project_barrier_scratch_facts(site_point));
      };
      OperatingPoint site_point = point;
      const uint16_t scratch_count = scratch_count_for(site_point);
      CandidateResourcePlan plan =
          plan_resource_site(state, request, debug, site_point, ResourceSiteKind::Barrier, index,
                             text_offset, candidate.source_site, scratch_count);
      OperatingPoint owner_point = site_point;
      const auto owner_descriptors = problem.inventory().kernel_descriptors(plan.owner_kernel_ids);
      if (owner_descriptors &&
          apply_persistent_vgpr_assignment(owner_point, point, *owner_descriptors)) {
        const uint16_t owner_scratch_count = scratch_count_for(owner_point);
        if (owner_scratch_count != scratch_count) {
          plan = plan_resource_site(state, request, debug, owner_point, ResourceSiteKind::Barrier,
                                    index, text_offset, candidate.source_site, owner_scratch_count);
        }
      }
      plan.intent_ids = {candidate.evidence_intent};
      plans.push_back(std::move(plan));
    }
  }
  if (request.track_atomics) {
    std::vector<AtomicEvidenceSitePlan> candidates = build_atomic_evidence_site_plans(
        problem.inventory(), problem.observation_plan(), evidence.atomic, errors);
    if (!errors.empty())
      return;
    for (size_t index = 0; index < candidates.size(); ++index) {
      const AtomicEvidenceSitePlan &candidate = candidates[index];
      const std::optional<AtomicEvidenceSourceView> source =
          resolve_atomic_evidence_source(problem.inventory(), candidate);
      if (!source) {
        errors.emplace_back("ConSan atomic resource planning lost its source site");
        return;
      }
      const bool relocates_atomic_polling_loop = source->relocates_polling_loop();
      const uint64_t patch_text_offset =
          relocates_atomic_polling_loop ? source->patch_text_offset() : source->site.text_offset;
      const uint16_t scratch_count = atomic_scratch_count();
      CandidateResourcePlan plan = plan_resource_site(
          state, request, debug, point, ResourceSiteKind::Atomic, index, patch_text_offset,
          candidate.source_site, scratch_count, nullptr, &candidate.lowering_form,
          /*require_spill=*/relocates_atomic_polling_loop);
      if (patch_text_offset != source->site.text_offset)
        plan.semantic_text_offset = source->site.text_offset;
      plan.intent_ids = {candidate.address_capture_intent, candidate.evidence_intent};
      plans.push_back(std::move(plan));
    }
  }
}

/// Derive one complete resource-plan set for the current ConSan operating-point
/// attempt. The returned vector is private until every access and operational
/// site has been planned and refined, so a retry cannot expose a half-cleared
/// or half-rebuilt plan set through the coordinator transaction.
[[nodiscard]] ResourcePlanningResult plan_resources(ResourcePlanningState &state,
                                                    const DebugOverrides &debug,
                                                    const OperatingPoint &point,
                                                    const ResourceProblem &problem) {
  const BoundRuntimeResources &bound_resources = problem.resources();
  ResourcePlanningResult planning;
  const OperatingPoint &attempted_point = point;
  const rj_code_arch_t arch = resource_arch(state);
  const Request &request = problem.request();
  const uint32_t required_sgpr_count_floor =
      detail::required_sgpr_count_floor(request, bound_resources, point, arch);
  planning.attempted_operating_point = attempted_point;
  append_resource_plans(state, request, debug, point, problem.candidates(), planning.plans);
  append_operational_resource_plans(state, debug, point, problem, planning.errors, planning.plans);
  const TargetProfile *target = target_profile(arch);
  if (target && target->placement.automatic_dispatch_sgpr_requires_owner_admission &&
      point.dispatch_sgpr.automatic() && point.dispatch_sgpr.base()) {
    for (CandidateResourcePlan &plan : planning.plans) {
      const auto owner_descriptors = problem.inventory().kernel_descriptors(plan.owner_kernel_ids);
      if (!owner_descriptors) {
        plan.source = RegisterAllocationSource::Unsupported;
        plan.reason = RegisterPlanReason::MissingOwner;
        plan.scratch_vgpr.reset();
        continue;
      }
      OperatingPoint owner_point = point;
      if (!attempted_point.owner_transient_sgprs.empty() &&
          !apply_transient_sgpr_assignment(request, owner_point, attempted_point,
                                           *owner_descriptors)) {
        if (target->code_transport == CodeTransportModel::PerKernelOwnerTranslation)
          continue;
        owner_point.dispatch_sgpr.reset();
      }
      if (plan.owner_kernel_ids.empty())
        continue;
      // A component-local transient assignment may deliberately omit an
      // overlapping persistent dispatch pair. Its established fallback ABI
      // records generation zero; rejecting the resource plan here would make
      // that safe fallback specific to direct code transport even though the
      // report contract is target-neutral.
      if (!owner_point.dispatch_sgpr.base() &&
          std::ranges::all_of(*owner_descriptors, [&](uint64_t descriptor_offset) {
            const TransientSgprAssignment *assignment =
                transient_sgpr_assignment(attempted_point, descriptor_offset);
            return assignment != nullptr && !assignment->dispatch_id_sgpr;
          })) {
        continue;
      }
      const bool missing_dispatch = !owner_point.dispatch_sgpr.base();
      const uint16_t dispatch_base = owner_point.dispatch_sgpr.base().value_or(0u);
      const bool admitted =
          !missing_dispatch &&
          resource_owner_sgpr_window_admitted(state, plan.owner_kernel_ids, dispatch_base,
                                              kDispatchStateSgprCount, required_sgpr_count_floor);
      if (!admitted) {
        plan.source = RegisterAllocationSource::Unsupported;
        plan.reason = RegisterPlanReason::ForbiddenOverlap;
        plan.scratch_vgpr.reset();
        plan.required_vgpr_count = plan.current_vgpr_count;
      }
    }
  }
  if (is_capability_arch(arch) && point.automatic_partial_exec_save_sgprs && point.exec_save_sgpr) {
    const uint16_t exec_base = *point.exec_save_sgpr;
    const uint16_t exec_count =
        exec_save_sgpr_count(resolve_exec_save_requirement(request, bound_resources, point), arch);
    for (CandidateResourcePlan &plan : planning.plans) {
      const auto owner_descriptors = problem.inventory().kernel_descriptors(plan.owner_kernel_ids);
      if (!owner_descriptors) {
        plan.source = RegisterAllocationSource::Unsupported;
        plan.reason = RegisterPlanReason::MissingOwner;
        plan.scratch_vgpr.reset();
        continue;
      }
      const bool has_owner_override =
          std::ranges::any_of(*owner_descriptors, [&](uint64_t descriptor_offset) {
            return transient_sgpr_assignment(attempted_point, descriptor_offset) != nullptr;
          });
      if (has_owner_override) {
        OperatingPoint owner_point = point;
        if (apply_transient_sgpr_assignment(request, owner_point, attempted_point,
                                            *owner_descriptors))
          continue;
      }
      if (!plan.owner_kernel_ids.empty() &&
          !resource_owner_sgpr_window_admitted(state, plan.owner_kernel_ids, exec_base, exec_count,
                                               required_sgpr_count_floor)) {
        plan.source = RegisterAllocationSource::Unsupported;
        plan.reason = RegisterPlanReason::ForbiddenOverlap;
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
[[nodiscard]] ResourcePlanningResult
solve_automatic_exec_save_resources(ResourcePlanningState &state, const DebugOverrides &debug,
                                    const OperatingPoint &base, const ResourceProblem &problem) {
  struct ExecAttempt {
    ResourcePlanningResult planning;
    bool needs_dynamic_stack_retry = false;
  };
  const auto attempt = [&](OperatingPoint point, std::optional<FallbackKind> fallback) {
    ExecAttempt exec;
    exec.planning = plan_resources(state, debug, point, problem);
    if (!exec.planning.success())
      return exec;

    std::vector<std::string> diagnostics;
    const bool changed = configure_automatic_exec_save_sgprs(point, problem, exec.planning.plans,
                                                             diagnostics, state);
    exec.needs_dynamic_stack_retry =
        !point.dynamic_stack_spill && automatic_scalar_spill_needs_dynamic_stack_planning(
                                          point, problem.inventory(), exec.planning.plans);
    if (changed)
      exec.planning = plan_resources(state, debug, point, problem);
    exec.planning.attempted_operating_point = point;
    exec.planning.selected_fallback = fallback;
    exec.planning.diagnostics = std::move(diagnostics);
    return exec;
  };

  ExecAttempt selected = attempt(base, std::nullopt);
  if (!selected.planning.success() || !selected.needs_dynamic_stack_retry)
    return std::move(selected.planning);

  OperatingPoint dynamic_stack_point = base;
  dynamic_stack_point.dynamic_stack_spill = true;
  return std::move(
      attempt(std::move(dynamic_stack_point), FallbackKind::DynamicStackScalarSpill).planning);
}
