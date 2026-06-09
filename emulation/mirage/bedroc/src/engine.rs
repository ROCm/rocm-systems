//! The engine: lowers a request + environment + catalogue into a solver
//! problem and lifts the solver's answer back into a [`Proof`].

use std::collections::BTreeMap;

use mirage_solver::{Cache, Goal, PlanError, State, Step, plan, plan_with_cache};
use thiserror::Error;

use crate::environment::Environment;
use crate::manifest::ToolCatalog;
use crate::proof::{GoalOutcome, Proof, ProofStep, UnavailableTool};
use crate::request::BedrocRequest;

/// Metadata about an expanded step, retained so solver results can be
/// rendered back into a [`Proof`].
#[derive(Debug, Clone)]
pub(crate) struct StepMeta {
    pub(crate) manifest_id: String,
    pub(crate) name: String,
}

/// The lowered planning problem for a request: the propositional inputs
/// the solver consumes, plus the bookkeeping needed to explain results.
#[derive(Debug, Clone)]
pub struct Lowering {
    /// Initial known facts (source + environment).
    pub initial: State,
    /// Expanded, environment-filtered steps.
    pub steps: Vec<Step>,
    /// The goal (every requested property across every target).
    pub goal: Goal,
    /// Per-step metadata, keyed by step id.
    pub(crate) meta: BTreeMap<String, StepMeta>,
    /// Ids of tools available in this environment.
    pub available_tools: Vec<String>,
    /// Tools excluded by the environment, with reasons.
    pub unavailable_tools: Vec<UnavailableTool>,
}

/// Drives the bedroc planning pipeline for a fixed catalogue and
/// environment.
#[derive(Debug, Clone)]
pub struct Engine {
    catalog: ToolCatalog,
    environment: Environment,
}

/// Errors from the engine that are not ordinary "unsupported goal"
/// outcomes (those are reported in the [`Proof`] instead).
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EngineError {
    /// The planning search was exhausted — only possible with an
    /// extraordinarily large/dense catalogue.
    #[error("planning search exhausted after expanding {0} states")]
    SearchExhausted(usize),
}

impl Engine {
    /// Construct an engine over `catalog`, gated by `environment`.
    pub fn new(catalog: ToolCatalog, environment: Environment) -> Self {
        Self {
            catalog,
            environment,
        }
    }

    /// The catalogue this engine plans over.
    pub fn catalog(&self) -> &ToolCatalog {
        &self.catalog
    }

    /// The environment this engine is gated by.
    pub fn environment(&self) -> &Environment {
        &self.environment
    }

    /// Why a manifest cannot run in the current environment, or `None`
    /// if it can.
    fn unavailable_reason(&self, m: &crate::manifest::ToolManifest) -> Option<String> {
        let missing: Vec<&String> = m
            .needs_tools
            .iter()
            .filter(|t| !self.environment.has_tool(t))
            .collect();
        if !missing.is_empty() {
            let names: Vec<String> = missing.iter().map(|s| s.to_string()).collect();
            return Some(format!("requires tool(s) not installed: {}", names.join(", ")));
        }
        if m.needs_gpu && !self.environment.has_gpu() {
            return Some("requires a physical GPU but none is present".to_string());
        }
        None
    }

    /// Lower a request into the solver problem.
    pub fn lower(&self, request: &BedrocRequest) -> Lowering {
        let mut steps = Vec::new();
        let mut meta = BTreeMap::new();
        let mut available_tools = Vec::new();
        let mut unavailable_tools = Vec::new();

        for m in self.catalog.iter() {
            if let Some(reason) = self.unavailable_reason(m) {
                unavailable_tools.push(UnavailableTool {
                    id: m.id.clone(),
                    reason,
                });
                continue;
            }
            available_tools.push(m.id.clone());

            if m.per_target {
                for t in &request.targets {
                    let sub = |v: &[String]| -> Vec<String> {
                        v.iter()
                            .map(|s| crate::manifest::ToolManifest::substitute(s, &t.gfx))
                            .collect()
                    };
                    let id = format!("{}:{}", m.id, t.gfx);
                    let step = Step {
                        id: id.clone(),
                        requires: sub(&m.requires),
                        produces: sub(&m.produces),
                        cost: m.cost,
                        cacheable: m.cacheable,
                    };
                    meta.insert(
                        id,
                        StepMeta {
                            manifest_id: m.id.clone(),
                            name: m.name.clone(),
                        },
                    );
                    steps.push(step);
                }
            } else {
                let step = Step {
                    id: m.id.clone(),
                    requires: m.requires.clone(),
                    produces: m.produces.clone(),
                    cost: m.cost,
                    cacheable: m.cacheable,
                };
                meta.insert(
                    m.id.clone(),
                    StepMeta {
                        manifest_id: m.id.clone(),
                        name: m.name.clone(),
                    },
                );
                steps.push(step);
            }
        }

        let mut initial_facts = request.source_kind.source_facts(&request.targets);
        initial_facts.extend(self.environment.facts());
        let initial = State::from_facts(initial_facts);
        let goal = Goal::new(request.goal_facts());

        Lowering {
            initial,
            steps,
            goal,
            meta,
            available_tools,
            unavailable_tools,
        }
    }

    /// Plan a request, producing a full [`Proof`]. Unsupported goals are
    /// reported in the proof, not as errors.
    pub fn prove(&self, request: &BedrocRequest) -> Result<Proof, EngineError> {
        self.prove_inner(request, None)
    }

    /// Like [`Engine::prove`], but discounts steps whose results are in
    /// `cache`, steering toward reuse.
    pub fn prove_with_cache(
        &self,
        request: &BedrocRequest,
        cache: &Cache,
    ) -> Result<Proof, EngineError> {
        self.prove_inner(request, Some(cache))
    }

    fn prove_inner(
        &self,
        request: &BedrocRequest,
        cache: Option<&Cache>,
    ) -> Result<Proof, EngineError> {
        let (lowering, solver_plan, missing) = self.solve(request, cache)?;

        let plan_steps: Vec<ProofStep> = solver_plan
            .steps
            .iter()
            .map(|ps| {
                let meta = lowering.meta.get(&ps.step.id);
                ProofStep {
                    tool_id: ps.step.id.clone(),
                    manifest_id: meta.map(|m| m.manifest_id.clone()).unwrap_or_default(),
                    name: meta.map(|m| m.name.clone()).unwrap_or_default(),
                    produces: ps.step.produces.clone(),
                    cached: ps.cached,
                    cost: ps.effective_cost,
                }
            })
            .collect();

        let goals = self.goal_outcomes(request, &missing);

        Ok(Proof {
            request: request.clone(),
            targets: request.targets.clone(),
            plan: plan_steps,
            total_cost: solver_plan.total_cost,
            goals,
            available_tools: lowering.available_tools,
            unavailable_tools: lowering.unavailable_tools,
        })
    }

    /// Lower the request and find the minimum-cost plan over the
    /// *satisfiable subset* of goal facts, returning the lowering, the
    /// solver plan, and the goal facts that remain unreachable.
    pub(crate) fn solve(
        &self,
        request: &BedrocRequest,
        cache: Option<&Cache>,
    ) -> Result<(Lowering, mirage_solver::Plan, Vec<String>), EngineError> {
        let lowering = self.lower(request);
        let goal_facts = request.goal_facts();

        let run_plan = |g: &Goal| match cache {
            Some(c) => plan_with_cache(&lowering.initial, &lowering.steps, g, c),
            None => plan(&lowering.initial, &lowering.steps, g),
        };

        let (solver_plan, missing) = match run_plan(&lowering.goal) {
            Ok(p) => (p, Vec::new()),
            Err(PlanError::Unreachable { missing }) => {
                let reachable: Vec<String> = goal_facts
                    .iter()
                    .filter(|f| !missing.contains(f))
                    .cloned()
                    .collect();
                let plan = if reachable.is_empty() {
                    mirage_solver::Plan {
                        steps: Vec::new(),
                        total_cost: 0,
                    }
                } else {
                    run_plan(&Goal::new(reachable)).map_err(map_search_exhausted)?
                };
                (plan, missing)
            }
            Err(e) => return Err(map_search_exhausted(e)),
        };

        Ok((lowering, solver_plan, missing))
    }

    /// Build per (goal, target) outcomes, attaching a reason to each
    /// unproven property.
    fn goal_outcomes(&self, request: &BedrocRequest, missing: &[String]) -> Vec<GoalOutcome> {
        let mut outcomes = Vec::new();
        for g in &request.goals {
            for t in &request.targets {
                let fact = g.fact_for(&t.gfx);
                let proven = !missing.contains(&fact);
                let reason = if proven {
                    None
                } else {
                    Some(self.explain_unsupported(&fact))
                };
                outcomes.push(GoalOutcome {
                    goal: *g,
                    target: t.gfx.clone(),
                    fact,
                    proven,
                    reason,
                });
            }
        }
        outcomes
    }

    /// Explain why a goal fact could not be established: which tool would
    /// produce it and what is blocking it (a missing install, a broken
    /// prerequisite chain, or simply no such tool).
    fn explain_unsupported(&self, fact: &str) -> String {
        let (_, target) = fact.split_once(':').unwrap_or((fact, ""));
        // Manifests that could produce this fact (for this target).
        let producers: Vec<&crate::manifest::ToolManifest> = self
            .catalog
            .iter()
            .filter(|m| {
                m.produces.iter().any(|p| {
                    let expanded = if m.per_target {
                        crate::manifest::ToolManifest::substitute(p, target)
                    } else {
                        p.clone()
                    };
                    expanded == fact
                })
            })
            .collect();

        if producers.is_empty() {
            return format!("no tool in the catalogue produces `{fact}`");
        }

        let available: Vec<&&crate::manifest::ToolManifest> = producers
            .iter()
            .filter(|m| self.unavailable_reason(m).is_none())
            .collect();

        if available.is_empty() {
            // Every producing tool is gated out by the environment.
            let mut needs: std::collections::BTreeSet<String> = Default::default();
            for m in &producers {
                needs.extend(
                    m.needs_tools
                        .iter()
                        .filter(|t| !self.environment.has_tool(t))
                        .cloned(),
                );
                if m.needs_gpu && !self.environment.has_gpu() {
                    needs.insert("<physical GPU>".to_string());
                }
            }
            let ids: Vec<&str> = producers.iter().map(|m| m.id.as_str()).collect();
            return format!(
                "`{fact}` is produced by {} but requires {} in this environment",
                ids.join(", "),
                needs.into_iter().collect::<Vec<_>>().join(", ")
            );
        }

        // A producing tool is available, but a prerequisite of its input
        // chain cannot be established here.
        let ids: Vec<&str> = available.iter().map(|m| m.id.as_str()).collect();
        format!(
            "`{fact}` is produced by {} but its prerequisites cannot be established in this environment",
            ids.join(", ")
        )
    }
}

fn map_search_exhausted(e: PlanError) -> EngineError {
    match e {
        PlanError::SearchExhausted(n) => EngineError::SearchExhausted(n),
        // Unreachable is handled by the caller before reaching here.
        PlanError::Unreachable { .. } => EngineError::SearchExhausted(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::request::{GoalKind, SourceKind};

    fn full_env() -> Environment {
        Environment::empty().with_tool("rocjitsu")
    }

    fn req(goals: Vec<GoalKind>) -> BedrocRequest {
        BedrocRequest::build("k.hip", SourceKind::Hip, ["mi350"], goals)
    }

    #[test]
    fn proves_hazards_and_output_with_full_env() {
        let engine = Engine::new(ToolCatalog::builtin(), full_env());
        let proof = engine
            .prove(&req(vec![GoalKind::NoDataHazards, GoalKind::CorrectOutput]))
            .unwrap();
        assert!(proof.fully_proven(), "{}", proof.summary());
        // Compile is shared by both goals.
        let compiles = proof
            .plan
            .iter()
            .filter(|s| s.manifest_id == "compile-hip")
            .count();
        assert_eq!(compiles, 1);
    }

    #[test]
    fn races_unsupported_without_rocjitsu() {
        let engine = Engine::new(ToolCatalog::builtin(), Environment::empty());
        let proof = engine.prove(&req(vec![GoalKind::NoDataRaces])).unwrap();
        assert!(!proof.fully_proven());
        let outcome = &proof.goals[0];
        assert!(!outcome.proven);
        let reason = outcome.reason.as_ref().unwrap();
        assert!(reason.contains("rocjitsu"), "reason was: {reason}");
    }

    #[test]
    fn hazards_provable_without_any_emulator() {
        // The static wait check needs no installed tools or GPU.
        let engine = Engine::new(ToolCatalog::builtin(), Environment::empty());
        let proof = engine.prove(&req(vec![GoalKind::NoDataHazards])).unwrap();
        assert!(proof.fully_proven(), "{}", proof.summary());
    }

    #[test]
    fn unknown_goal_predicate_reports_no_tool() {
        // Build a request whose only goal has no producing tool by using
        // an empty catalogue.
        let engine = Engine::new(ToolCatalog::new(), full_env());
        let proof = engine.prove(&req(vec![GoalKind::CorrectOutput])).unwrap();
        assert!(!proof.fully_proven());
        assert!(
            proof.goals[0]
                .reason
                .as_ref()
                .unwrap()
                .contains("no tool")
        );
    }
}
