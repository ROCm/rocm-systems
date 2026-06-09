//! Executing a plan, with caching of idempotent steps.
//!
//! The [`Executor`] runs the plan the [`Engine`] produced, step by step.
//! For each step it derives a content-addressed [`mirage_solver::CacheKey`]
//! from the step and its inputs (including a fingerprint of the kernel
//! source), so an idempotent step whose result is already cached is
//! **reused** rather than re-run. The default execution is *simulated*:
//! running a step computes a deterministic artifact fingerprint rather
//! than shelling out, which keeps the engine fast and hermetic. A
//! deployment can layer real command execution on top using each
//! manifest's optional `command`.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use mirage_solver::{Cache, CacheKey, CachedResult, State};
use serde::{Deserialize, Serialize};

use crate::engine::{Engine, EngineError};
use crate::request::BedrocRequest;

/// The outcome of executing a single step.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutedStep {
    /// The step (tool) id.
    pub tool_id: String,
    /// Human-readable tool name.
    pub name: String,
    /// Facts the step established.
    pub produces: Vec<String>,
    /// Whether the result was reused from cache (`true`) or freshly
    /// computed (`false`).
    pub cached: bool,
    /// The artifact fingerprint (stable across identical inputs).
    pub fingerprint: String,
}

/// The result of executing a whole plan.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutionReport {
    /// Steps in execution order.
    pub steps: Vec<ExecutedStep>,
    /// How many steps were reused from cache.
    pub cache_hits: usize,
    /// How many steps were freshly executed.
    pub executed: usize,
    /// Goal facts established by the end of execution.
    pub established: Vec<String>,
    /// Goal facts that could not be established in this environment.
    pub unestablished: Vec<String>,
}

/// Runs plans produced by an [`Engine`], memoizing idempotent steps in a
/// [`Cache`].
#[derive(Debug)]
pub struct Executor<'a> {
    engine: &'a Engine,
}

impl<'a> Executor<'a> {
    /// Build an executor over `engine`.
    pub fn new(engine: &'a Engine) -> Self {
        Self { engine }
    }

    /// Execute the plan for `request`, consulting and updating `cache`.
    ///
    /// `source_fingerprint` is mixed into every cache key so a changed
    /// kernel invalidates prior results. Use [`fingerprint_source`] to
    /// derive it from a source path.
    pub fn run(
        &self,
        request: &BedrocRequest,
        cache: &mut Cache,
        source_fingerprint: &str,
    ) -> Result<ExecutionReport, EngineError> {
        let (lowering, plan, missing) = self.engine.solve(request, Some(cache))?;

        let mut state = lowering.initial.clone();
        let mut steps = Vec::new();
        let mut cache_hits = 0;
        let mut executed = 0;

        for ps in &plan.steps {
            let step = &ps.step;
            // Inputs that determine this step's result: its satisfied
            // preconditions plus the source fingerprint.
            let mut inputs: Vec<String> = step
                .requires
                .iter()
                .filter(|f| state.has(f))
                .cloned()
                .collect();
            inputs.push(format!("__source__:{source_fingerprint}"));
            let key = CacheKey::for_step(step, &inputs);

            let (cached, fingerprint) = if step.cacheable {
                if let Some(hit) = cache.get(&key) {
                    cache_hits += 1;
                    (true, hit.fingerprint.clone())
                } else {
                    let fp = simulate(step, &inputs);
                    cache.put(
                        key,
                        CachedResult {
                            produced: step.produces.clone(),
                            fingerprint: fp.clone(),
                        },
                    );
                    executed += 1;
                    (false, fp)
                }
            } else {
                executed += 1;
                (false, simulate(step, &inputs))
            };

            let meta = lowering.meta.get(&step.id);
            steps.push(ExecutedStep {
                tool_id: step.id.clone(),
                name: meta.map(|m| m.name.clone()).unwrap_or_default(),
                produces: step.produces.clone(),
                cached,
                fingerprint,
            });

            apply(&mut state, step);
        }

        let goal_facts = request.goal_facts();
        let established: Vec<String> = goal_facts
            .iter()
            .filter(|f| state.has(f))
            .cloned()
            .collect();

        Ok(ExecutionReport {
            steps,
            cache_hits,
            executed,
            established,
            unestablished: missing,
        })
    }
}

/// Apply a step's effects to a state.
fn apply(state: &mut State, step: &mirage_solver::Step) {
    for f in &step.produces {
        state.insert(f.clone());
    }
}

/// Deterministically "execute" a step by hashing its identity and
/// inputs into an artifact fingerprint. This stands in for real tool
/// execution while remaining stable and reproducible.
fn simulate(step: &mirage_solver::Step, inputs: &[String]) -> String {
    let mut h = DefaultHasher::new();
    step.id.hash(&mut h);
    for i in inputs {
        i.hash(&mut h);
    }
    format!("{:016x}", h.finish())
}

/// Derive a stable fingerprint for a kernel source path. The file's
/// bytes are hashed when readable; otherwise the path string is used so
/// callers always get a usable value.
pub fn fingerprint_source(path: &str) -> String {
    let mut h = DefaultHasher::new();
    match std::fs::read(path) {
        Ok(bytes) => bytes.hash(&mut h),
        Err(_) => path.hash(&mut h),
    }
    format!("{:016x}", h.finish())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Environment;
    use crate::manifest::ToolCatalog;
    use crate::request::{GoalKind, SourceKind};

    fn engine() -> Engine {
        Engine::new(
            ToolCatalog::builtin(),
            Environment::empty().with_tool("rocjitsu"),
        )
    }

    fn request() -> BedrocRequest {
        BedrocRequest::build(
            "k.hip",
            SourceKind::Hip,
            ["mi350"],
            [GoalKind::NoDataHazards, GoalKind::CorrectOutput],
        )
    }

    #[test]
    fn executes_then_reuses_from_cache() {
        let engine = engine();
        let exec = Executor::new(&engine);
        let mut cache = Cache::new();

        let first = exec.run(&request(), &mut cache, "fp1").unwrap();
        assert!(first.cache_hits == 0);
        assert!(first.executed > 0);
        assert!(first.established.contains(&"no_hazards:gfx950".to_string()));
        assert!(first.established.contains(&"correct_output:gfx950".to_string()));

        // Second run with the same source: every cacheable step is a hit.
        let second = exec.run(&request(), &mut cache, "fp1").unwrap();
        assert!(second.cache_hits > 0);
        assert_eq!(second.executed, 0);
    }

    #[test]
    fn changed_source_invalidates_cache() {
        let engine = engine();
        let exec = Executor::new(&engine);
        let mut cache = Cache::new();
        exec.run(&request(), &mut cache, "fp1").unwrap();
        let changed = exec.run(&request(), &mut cache, "fp2").unwrap();
        // New fingerprint ⇒ nothing is reused.
        assert_eq!(changed.cache_hits, 0);
    }
}
