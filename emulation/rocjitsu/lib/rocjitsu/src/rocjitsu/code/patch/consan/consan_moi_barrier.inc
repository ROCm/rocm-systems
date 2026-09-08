/// Project admitted barrier evidence directly into the common MOI lowering
/// contract shared by Record/Replay, Sampled, and InlineShadow.
///
/// Policy may map several barrier events to one completing intent. This join
/// therefore starts from the authoritative intent inventory and resolves
/// exactly one graph event and decoded instruction per insertion site.
/// Redundant barriers, runtime kernels, container filters, and engine-specific
/// signal/wait rules have already been decided by policy and are never
/// rediscovered here.
[[nodiscard]] std::vector<MoiBarrierEvidenceSitePlan> build_moi_barrier_evidence_site_plans(
    const ProgramInventory &inventory, const ConSanObservationPlan &observation,
    ConSanProbeIntentKind evidence_kind, std::vector<std::string> &errors) {
  const SynchronizationInventoryView graph = inventory.sync();
  std::vector<MoiBarrierEvidenceSitePlan> plans;
  plans.reserve(observation.probe_intents.size());
  for (const ConSanProbeIntent &intent : observation.probe_intents) {
    if (intent.kind != evidence_kind)
      continue;
    const ConSanSyncEvent *event = graph.find_event(intent.source_site);
    const ConSanSyncSequence *sequence =
        event == nullptr ? nullptr : graph.find_unique_sequence_containing(event->semantic_id);
    if (event == nullptr || sequence == nullptr || intent.position != ConSanProbePosition::After) {
      errors.emplace_back("ConSan MOI admitted barrier evidence lost its synchronization graph");
      return {};
    }
    bool coverage_is_exact = !intent.covered_semantic_sites.empty();
    for (const SemanticSiteId &covered : intent.covered_semantic_sites) {
      const ConSanSyncEvent *member = graph.find_event(covered);
      const ConSanSyncEventId member_id =
          member == nullptr ? ConSanSyncEventId{} : graph.event_id(*member);
      coverage_is_exact &= member != nullptr && member->kind == ConSanSyncKind::Barrier &&
                           std::ranges::find(sequence->member_event_ids, member_id) !=
                               sequence->member_event_ids.end();
    }
    if (!coverage_is_exact) {
      errors.emplace_back(
          "ConSan MOI admitted barrier evidence has inconsistent sequence coverage");
      return {};
    }

    MoiBarrierEvidenceSitePlan plan;
    plan.event = graph.event_id(*event);
    plan.sequence = graph.sequence_id(*sequence);
    plan.source_site = event->source_site;
    plan.evidence_intent = intent.id;
    const ConSanProgramContainer *container =
        resolve_moi_evidence_container(inventory, event->source_site);
    const ConSanBarrierSite *site = inventory.program_site<ConSanBarrierSite>(event->source_site);
    if (container == nullptr || site == nullptr) {
      errors.emplace_back("ConSan MOI admitted barrier evidence lost its decoded lowering site");
      return {};
    }
    if (!plan.is_well_formed() || site->text_offset != event->text_offset()) {
      errors.emplace_back("ConSan MOI admitted barrier evidence lost its decoded lowering site");
      return {};
    }
    plans.push_back(std::move(plan));
  }

  std::ranges::sort(plans, {}, [&](const MoiBarrierEvidenceSitePlan &plan) {
    return graph.find_event(plan.event)->text_offset();
  });
  return plans;
}
