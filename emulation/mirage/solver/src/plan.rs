//! The planner: minimum-cost sequencing of steps to satisfy a goal.
//!
//! Because step effects are additive (a step only ever *adds* facts),
//! the reachable states form a monotone lattice and the optimal plan is
//! a shortest path through it. [`plan`] runs a Dijkstra search over
//! reachable [`State`]s, where each edge applies one [`Step`] and is
//! weighted by that step's (possibly cache-discounted) cost. The first
//! goal-satisfying state popped from the frontier is therefore reached
//! by a globally minimum-cost sequence.
//!
//! [`enumerate_plans`] explores *alternative* goal-satisfying step sets,
//! which higher layers use for differential fuzzing: several distinct
//! tool sequences that establish the same goal should agree on the
//! facts they prove.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BinaryHeap};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::cache::{Cache, CacheKey};
use crate::state::{Fact, Goal, State};
use crate::step::Step;

/// Hard cap on the number of states the Dijkstra search will expand,
/// guarding against pathological catalogues. Real tool catalogues are
/// tiny (tens of steps) so this is never hit in practice; exceeding it
/// surfaces as [`PlanError::SearchExhausted`].
const MAX_EXPANDED_STATES: usize = 200_000;

/// One scheduled step within a [`Plan`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlanStep {
    /// The step to run.
    pub step: Step,
    /// Whether this step's result was found in the cache (and so its
    /// cost was discounted and execution may be skipped).
    pub cached: bool,
    /// The cost actually attributed to this step in the plan (the
    /// nominal [`Step::cost`], or a discount when cached).
    pub effective_cost: u64,
}

/// An ordered, minimum-cost sequence of steps that satisfies a goal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Plan {
    /// Steps in the order they must run.
    pub steps: Vec<PlanStep>,
    /// Sum of the effective costs of every step.
    pub total_cost: u64,
}

impl Plan {
    /// The ids of the steps in this plan, in order.
    pub fn step_ids(&self) -> Vec<String> {
        self.steps.iter().map(|s| s.step.id.clone()).collect()
    }

    /// The final state produced by running every step from `initial`.
    pub fn final_state(&self, initial: &State) -> State {
        let mut s = initial.clone();
        for ps in &self.steps {
            for f in &ps.step.produces {
                s.insert(f.clone());
            }
        }
        s
    }
}

/// Why a plan could not be produced.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum PlanError {
    /// One or more goal facts can never be produced from the initial
    /// state with the available steps (no tool establishes them).
    #[error("goal is unreachable; no available step can produce: {missing:?}")]
    Unreachable {
        /// Goal facts that no reachable step can establish.
        missing: Vec<Fact>,
    },
    /// The search hit its expansion cap before finding the goal. This
    /// indicates a catalogue far larger/denser than expected.
    #[error("search exhausted after expanding {0} states")]
    SearchExhausted(usize),
}

/// Effective cost the planner attributes to running `step` in `state`,
/// consulting `cache` if present. Returns the cost and whether it was a
/// cache hit.
fn effective_cost(step: &Step, state: &State, cache: Option<&Cache>) -> (u64, bool) {
    if step.cacheable
        && let Some(cache) = cache
    {
        let inputs: Vec<Fact> = step
            .requires
            .iter()
            .filter(|f| state.has(f))
            .cloned()
            .collect();
        let key = CacheKey::for_step(step, &inputs);
        if cache.contains(&key) {
            // Reuse is effectively free aside from a lookup; model it
            // as zero so the planner prefers reusing prior work.
            return (0, true);
        }
    }
    (step.cost, false)
}

/// The maximal state reachable from `initial` by applying every
/// applicable step to a fixpoint, ignoring cost. Used to decide
/// reachability cheaply before the weighted search.
fn reachable_closure(initial: &State, steps: &[Step]) -> State {
    let mut state = initial.clone();
    loop {
        let mut grew = false;
        for step in steps {
            if step.applicable_in(&state) && step.advances(&state) {
                for f in &step.produces {
                    if state.insert(f.clone()) {
                        grew = true;
                    }
                }
            }
        }
        if !grew {
            return state;
        }
    }
}

/// Find a minimum-cost plan from `initial` that satisfies `goal`.
pub fn plan(initial: &State, steps: &[Step], goal: &Goal) -> Result<Plan, PlanError> {
    plan_inner(initial, steps, goal, None)
}

/// Like [`plan`], but discounts steps whose results are already in
/// `cache`, steering the planner toward reusing prior work.
pub fn plan_with_cache(
    initial: &State,
    steps: &[Step],
    goal: &Goal,
    cache: &Cache,
) -> Result<Plan, PlanError> {
    plan_inner(initial, steps, goal, Some(cache))
}

fn plan_inner(
    initial: &State,
    steps: &[Step],
    goal: &Goal,
    cache: Option<&Cache>,
) -> Result<Plan, PlanError> {
    if goal.is_satisfied_by(initial) {
        return Ok(Plan {
            steps: Vec::new(),
            total_cost: 0,
        });
    }

    // Cheap reachability gate: if the fixpoint closure can't satisfy the
    // goal, no weighted path can either. This also yields a precise
    // "missing facts" diagnostic.
    let closure = reachable_closure(initial, steps);
    if !goal.is_satisfied_by(&closure) {
        return Err(PlanError::Unreachable {
            missing: goal.missing_in(&closure),
        });
    }

    // Dijkstra over reachable fact-sets.
    let mut dist: BTreeMap<State, u64> = BTreeMap::new();
    let mut came_from: BTreeMap<State, (State, usize, bool, u64)> = BTreeMap::new();
    let mut heap: BinaryHeap<Reverse<(u64, State)>> = BinaryHeap::new();

    dist.insert(initial.clone(), 0);
    heap.push(Reverse((0, initial.clone())));

    let mut expanded = 0usize;
    while let Some(Reverse((cost, state))) = heap.pop() {
        // Stale heap entry (a cheaper path to this state was found).
        if dist.get(&state).is_some_and(|&best| cost > best) {
            continue;
        }
        if goal.is_satisfied_by(&state) {
            return Ok(reconstruct(initial, &state, steps, &came_from));
        }
        expanded += 1;
        if expanded > MAX_EXPANDED_STATES {
            return Err(PlanError::SearchExhausted(expanded));
        }
        for (idx, step) in steps.iter().enumerate() {
            if !step.applicable_in(&state) || !step.advances(&state) {
                continue;
            }
            let (ecost, cached) = effective_cost(step, &state, cache);
            let next = state.with_all(step.produces.iter());
            let ncost = cost + ecost;
            if dist.get(&next).is_none_or(|&best| ncost < best) {
                dist.insert(next.clone(), ncost);
                came_from.insert(next.clone(), (state.clone(), idx, cached, ecost));
                heap.push(Reverse((ncost, next)));
            }
        }
    }

    // The closure said the goal was reachable, so the search must find
    // it unless the expansion cap intervened; treat falling through as
    // exhaustion.
    Err(PlanError::SearchExhausted(expanded))
}

/// Walk the `came_from` chain back to `initial` to materialize the plan.
fn reconstruct(
    initial: &State,
    goal_state: &State,
    steps: &[Step],
    came_from: &BTreeMap<State, (State, usize, bool, u64)>,
) -> Plan {
    let mut rev: Vec<PlanStep> = Vec::new();
    let mut cur = goal_state.clone();
    while cur != *initial {
        let (prev, idx, cached, ecost) = came_from
            .get(&cur)
            .expect("every non-initial reached state has a predecessor");
        rev.push(PlanStep {
            step: steps[*idx].clone(),
            cached: *cached,
            effective_cost: *ecost,
        });
        cur = prev.clone();
    }
    rev.reverse();
    let total_cost = rev.iter().map(|s| s.effective_cost).sum();
    Plan {
        steps: rev,
        total_cost,
    }
}

/// Enumerate up to `max_plans` distinct goal-satisfying plans from
/// `initial`, deduplicated by the *set* of steps they run.
///
/// Where [`plan`] returns the single cheapest sequence, this explores
/// genuinely different routes to the same goal. Higher layers use it for
/// differential fuzzing: every returned plan establishes the goal, so
/// the facts they prove must agree even though the tool sequences
/// differ. Plans are returned cheapest-first.
///
/// The search is bounded by [`MAX_EXPANDED_STATES`] explored nodes so it
/// always terminates on dense catalogues.
pub fn enumerate_plans(
    initial: &State,
    steps: &[Step],
    goal: &Goal,
    max_plans: usize,
) -> Vec<Plan> {
    let mut found: Vec<Plan> = Vec::new();
    let mut seen_sets: std::collections::BTreeSet<Vec<usize>> = std::collections::BTreeSet::new();
    let mut explored = 0usize;

    // DFS over ordered step choices. `used` is the ordered list of step
    // indices chosen so far; a step index is never reused (re-applying
    // adds nothing). Distinct plans are deduplicated by their sorted
    // index set so independent reorderings collapse to one route.
    fn dfs(
        state: &State,
        steps: &[Step],
        goal: &Goal,
        used: &mut Vec<usize>,
        found: &mut Vec<Plan>,
        seen_sets: &mut std::collections::BTreeSet<Vec<usize>>,
        explored: &mut usize,
        max_plans: usize,
    ) {
        if found.len() >= max_plans || *explored > MAX_EXPANDED_STATES {
            return;
        }
        *explored += 1;
        if goal.is_satisfied_by(state) {
            let mut key = used.clone();
            key.sort_unstable();
            if seen_sets.insert(key) {
                let plan_steps: Vec<PlanStep> = used
                    .iter()
                    .map(|&i| PlanStep {
                        step: steps[i].clone(),
                        cached: false,
                        effective_cost: steps[i].cost,
                    })
                    .collect();
                let total_cost = plan_steps.iter().map(|s| s.effective_cost).sum();
                found.push(Plan {
                    steps: plan_steps,
                    total_cost,
                });
            }
            return;
        }
        for (idx, step) in steps.iter().enumerate() {
            if used.contains(&idx) || !step.applicable_in(state) || !step.advances(state) {
                continue;
            }
            let next = state.with_all(step.produces.iter());
            used.push(idx);
            dfs(
                &next, steps, goal, used, found, seen_sets, explored, max_plans,
            );
            used.pop();
            if found.len() >= max_plans {
                return;
            }
        }
    }

    let mut used = Vec::new();
    dfs(
        initial,
        steps,
        goal,
        &mut used,
        &mut found,
        &mut seen_sets,
        &mut explored,
        max_plans,
    );
    found.sort_by_key(|p| p.total_cost);
    found
}

#[cfg(test)]
mod tests {
    use super::*;

    fn catalogue() -> Vec<Step> {
        vec![
            Step::new("compile", ["source"], ["compiled"], 100),
            Step::new("waitcheck", ["compiled"], ["no_hazards"], 10),
            Step::new("emulate", ["compiled"], ["correct_output"], 500),
        ]
    }

    #[test]
    fn trivial_goal_already_satisfied() {
        let init = State::from_facts(["x"]);
        let p = plan(&init, &catalogue(), &Goal::new(["x"])).unwrap();
        assert!(p.steps.is_empty());
        assert_eq!(p.total_cost, 0);
    }

    #[test]
    fn finds_minimal_chain() {
        let init = State::from_facts(["source"]);
        let p = plan(&init, &catalogue(), &Goal::new(["no_hazards"])).unwrap();
        assert_eq!(p.step_ids(), vec!["compile", "waitcheck"]);
        assert_eq!(p.total_cost, 110);
    }

    #[test]
    fn shares_prefix_across_goals() {
        let init = State::from_facts(["source"]);
        let goal = Goal::new(["no_hazards", "correct_output"]);
        let p = plan(&init, &catalogue(), &goal).unwrap();
        // compile is paid once even though two goals depend on it.
        assert_eq!(p.total_cost, 100 + 10 + 500);
        assert_eq!(p.step_ids()[0], "compile");
    }

    #[test]
    fn reports_unreachable() {
        let init = State::from_facts(["source"]);
        let err = plan(&init, &catalogue(), &Goal::new(["teleported"])).unwrap_err();
        match err {
            PlanError::Unreachable { missing } => assert_eq!(missing, vec!["teleported"]),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn cache_discounts_reused_work() {
        use crate::cache::{CacheKey, CachedResult};
        let init = State::from_facts(["source"]);
        let steps = vec![Step::new("compile", ["source"], ["compiled"], 100).cacheable(true)];
        let goal = Goal::new(["compiled"]);
        // Without cache, the compile costs its full price.
        assert_eq!(plan(&init, &steps, &goal).unwrap().total_cost, 100);
        // With a cache hit on the compile step, the plan is free.
        let mut cache = Cache::new();
        let key = CacheKey::for_step(&steps[0], &["source".into()]);
        cache.put(
            key,
            CachedResult {
                produced: vec!["compiled".into()],
                fingerprint: "x".into(),
            },
        );
        let p = plan_with_cache(&init, &steps, &goal, &cache).unwrap();
        assert_eq!(p.total_cost, 0);
        assert!(p.steps[0].cached);
    }

    #[test]
    fn enumerates_distinct_routes() {
        // Two independent tools both establish `correct_output`: a fast
        // emulator and a slower reference run. Both are valid routes.
        let init = State::from_facts(["source"]);
        let steps = vec![
            Step::new("compile", ["source"], ["compiled"], 100),
            Step::new("emulate", ["compiled"], ["correct_output"], 500),
            Step::new("reference", ["compiled"], ["correct_output"], 900),
        ];
        let goal = Goal::new(["correct_output"]);
        let plans = enumerate_plans(&init, &steps, &goal, 8);
        assert_eq!(plans.len(), 2);
        // Cheapest route first.
        assert!(plans[0].step_ids().contains(&"emulate".to_string()));
        assert!(plans[1].step_ids().contains(&"reference".to_string()));
        // Every enumerated route actually establishes the goal.
        for p in &plans {
            assert!(goal.is_satisfied_by(&p.final_state(&init)));
        }
    }
}
