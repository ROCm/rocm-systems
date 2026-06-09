//! Differential fuzzing of the tool catalogue.
//!
//! Every tool declares the transformation it performs (its preconditions
//! and effects). That makes the catalogue *self-checkable*: for a given
//! property there are often several independent tool sequences that
//! should all establish it (e.g. `correct_output` via the fast emulator
//! or via the slow reference oracle). [`differential`] enumerates those
//! distinct routes and confirms each one, when executed, actually
//! establishes the property the planner promised. A route that the
//! planner accepts but that fails to establish the goal on execution is
//! a catalogue or engine bug.
//!
//! [`fuzz_catalog`] drives this across a deterministically-generated
//! batch of requests so it can run as a CLI/CI check.

use mirage_solver::{Goal, State, enumerate_plans};
use serde::{Deserialize, Serialize};

use crate::engine::Engine;
use crate::request::{BedrocRequest, GoalKind, SourceKind};

/// One enumerated route to a goal and whether executing it establishes
/// that goal.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteOutcome {
    /// The tool ids run, in order.
    pub tool_ids: Vec<String>,
    /// The route's total cost.
    pub total_cost: u64,
    /// Whether replaying the route's steps establishes the goal fact.
    pub establishes_goal: bool,
}

/// All distinct routes to one goal fact, and whether they agree.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RouteReport {
    /// The goal fact under test, e.g. `correct_output:gfx950`.
    pub goal_fact: String,
    /// The distinct routes found.
    pub routes: Vec<RouteOutcome>,
    /// `true` when every route establishes the goal (the invariant the
    /// planner promises).
    pub consistent: bool,
}

impl RouteReport {
    /// The number of genuinely distinct routes to this goal.
    pub fn route_count(&self) -> usize {
        self.routes.len()
    }
}

/// Enumerate and check the distinct routes to every goal fact in
/// `request`, finding up to `max_routes_per_goal` routes each.
pub fn differential(
    engine: &Engine,
    request: &BedrocRequest,
    max_routes_per_goal: usize,
) -> Vec<RouteReport> {
    let lowering = engine.lower(request);
    let mut reports = Vec::new();

    for fact in request.goal_facts() {
        let goal = Goal::new([fact.clone()]);
        let plans = enumerate_plans(
            &lowering.initial,
            &lowering.steps,
            &goal,
            max_routes_per_goal,
        );
        let mut routes = Vec::new();
        for plan in &plans {
            // Replay the route from the initial state and confirm it
            // establishes the goal fact — independently of the planner's
            // own bookkeeping.
            let mut state = lowering.initial.clone();
            for ps in &plan.steps {
                for f in &ps.step.produces {
                    state.insert(f.clone());
                }
            }
            routes.push(RouteOutcome {
                tool_ids: plan.step_ids(),
                total_cost: plan.total_cost,
                establishes_goal: state.has(&fact),
            });
        }
        let consistent = routes.iter().all(|r| r.establishes_goal);
        reports.push(RouteReport {
            goal_fact: fact,
            routes,
            consistent,
        });
    }

    reports
}

/// Aggregate result of fuzzing the catalogue over many requests.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FuzzSummary {
    /// Number of requests generated and checked.
    pub requests: usize,
    /// Number of distinct (goal, target) facts examined.
    pub goals_checked: usize,
    /// Total number of distinct routes discovered across all goals.
    pub total_routes: usize,
    /// Goals for which two or more genuinely different routes exist
    /// (the interesting, cross-validating cases).
    pub multi_route_goals: usize,
    /// Human-readable descriptions of any consistency violations found.
    /// An empty list means every route agreed with the planner.
    pub inconsistencies: Vec<String>,
}

impl FuzzSummary {
    /// Whether the fuzz run found no inconsistencies.
    pub fn passed(&self) -> bool {
        self.inconsistencies.is_empty()
    }
}

/// Fuzz the engine's catalogue by generating `iterations` deterministic
/// requests (varying source kind, targets, and goals from `seed`) and
/// running [`differential`] on each. The result aggregates how many
/// cross-validating multi-route goals were exercised and lists any
/// route that diverged from the planner's promise.
pub fn fuzz_catalog(engine: &Engine, iterations: usize, seed: u64) -> FuzzSummary {
    const TARGETS: &[&str] = &["mi300x", "mi350", "mi450", "gfx942", "gfx950"];
    const GOALS: &[GoalKind] = &[
        GoalKind::NoDataHazards,
        GoalKind::CorrectOutput,
        GoalKind::NoDataRaces,
        GoalKind::FpCorrect,
    ];
    const SOURCES: &[SourceKind] = &[SourceKind::Hip, SourceKind::Asm, SourceKind::CodeObject];

    let mut rng = Lcg::new(seed);
    let mut summary = FuzzSummary::default();

    for _ in 0..iterations {
        // Pick a source kind.
        let source_kind = SOURCES[rng.below(SOURCES.len() as u64) as usize];
        // Pick 1..=2 targets.
        let n_targets = 1 + rng.below(2) as usize;
        let targets: Vec<&str> = (0..n_targets)
            .map(|_| TARGETS[rng.below(TARGETS.len() as u64) as usize])
            .collect();
        // Pick 1..=3 goals.
        let n_goals = 1 + rng.below(3) as usize;
        let goals: Vec<GoalKind> = (0..n_goals)
            .map(|_| GOALS[rng.below(GOALS.len() as u64) as usize])
            .collect();

        let request = BedrocRequest::build("fuzz.kernel", source_kind, targets, goals);
        let reports = differential(engine, &request, 8);

        summary.requests += 1;
        for r in reports {
            summary.goals_checked += 1;
            summary.total_routes += r.route_count();
            if r.route_count() >= 2 {
                summary.multi_route_goals += 1;
            }
            if !r.consistent {
                summary.inconsistencies.push(format!(
                    "goal `{}` has a route that does not establish it",
                    r.goal_fact
                ));
            }
        }
    }

    summary
}

/// A tiny deterministic linear congruential generator. Used so fuzzing
/// is reproducible from a seed without pulling in an RNG dependency.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        // Avoid a zero state, which would stick.
        Self {
            state: seed ^ 0x9e37_79b9_7f4a_7c15,
        }
    }

    fn next_u64(&mut self) -> u64 {
        // Numerical Recipes LCG constants.
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// A value in `[0, bound)` (bound must be > 0).
    fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}

/// Replay a list of step ids over a state — exposed for tests that want
/// to drive routes directly.
#[doc(hidden)]
pub fn replay(initial: &State, steps: &[mirage_solver::Step]) -> State {
    let mut s = initial.clone();
    for step in steps {
        for f in &step.produces {
            s.insert(f.clone());
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Environment;
    use crate::manifest::ToolCatalog;

    fn engine() -> Engine {
        Engine::new(
            ToolCatalog::builtin(),
            Environment::empty().with_tool("rocjitsu"),
        )
    }

    #[test]
    fn correct_output_has_two_routes_that_agree() {
        let engine = engine();
        let request =
            BedrocRequest::build("k.hip", SourceKind::Hip, ["mi350"], [GoalKind::CorrectOutput]);
        let reports = differential(&engine, &request, 8);
        assert_eq!(reports.len(), 1);
        let r = &reports[0];
        // Both the emulator and the reference oracle establish it.
        assert!(r.route_count() >= 2, "routes: {:?}", r.routes);
        assert!(r.consistent);
        for route in &r.routes {
            assert!(route.establishes_goal);
        }
    }

    #[test]
    fn fuzzing_finds_multi_route_goals_and_stays_consistent() {
        let engine = engine();
        let summary = fuzz_catalog(&engine, 50, 0xdead_beef);
        assert!(summary.passed(), "inconsistencies: {:?}", summary.inconsistencies);
        assert!(summary.requests == 50);
        assert!(summary.total_routes > 0);
        assert!(summary.multi_route_goals > 0);
    }
}
