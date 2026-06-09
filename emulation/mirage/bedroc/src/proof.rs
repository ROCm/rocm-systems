//! The proof: the engine's explanation of what was planned, what it
//! establishes, and what it cannot.

use serde::{Deserialize, Serialize};

use crate::request::{BedrocRequest, GoalKind, ResolvedTarget};

/// One step in the planned proof.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProofStep {
    /// The tool's manifest id (with target suffix for per-target tools).
    pub tool_id: String,
    /// The originating manifest id (without target suffix).
    pub manifest_id: String,
    /// Human-readable tool name.
    pub name: String,
    /// Facts this step establishes.
    pub produces: Vec<String>,
    /// Whether this step's result was already cached (and so would be
    /// reused rather than re-run).
    pub cached: bool,
    /// Cost attributed to this step in the plan.
    pub cost: u64,
}

/// Whether a requested (goal, target) property is established by the
/// plan, and if not, why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GoalOutcome {
    /// The property requested.
    pub goal: GoalKind,
    /// The target architecture.
    pub target: String,
    /// The underlying goal fact, e.g. `no_hazards:gfx950`.
    pub fact: String,
    /// Whether the plan establishes the property.
    pub proven: bool,
    /// When not proven, a precise reason (missing tool, no tool, …).
    pub reason: Option<String>,
}

/// A tool excluded from planning because the environment cannot run it,
/// with the reason — surfaced so users know what to install.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UnavailableTool {
    /// Manifest id.
    pub id: String,
    /// Why it is unavailable in this environment.
    pub reason: String,
}

/// The full result of planning a request: the ordered plan, which goals
/// it establishes, and the environment-driven gaps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    /// The normalized request (targets resolved to gfx).
    pub request: BedrocRequest,
    /// Resolved targets (friendly → gfx mapping), for display.
    pub targets: Vec<ResolvedTarget>,
    /// The ordered plan of tool invocations.
    pub plan: Vec<ProofStep>,
    /// Total cost of the plan.
    pub total_cost: u64,
    /// Per (goal, target) outcomes.
    pub goals: Vec<GoalOutcome>,
    /// Ids of tools available in this environment.
    pub available_tools: Vec<String>,
    /// Tools excluded due to environment constraints, with reasons.
    pub unavailable_tools: Vec<UnavailableTool>,
}

impl Proof {
    /// Whether every requested property is established.
    pub fn fully_proven(&self) -> bool {
        self.goals.iter().all(|g| g.proven)
    }

    /// The requested properties that are not established.
    pub fn unproven(&self) -> impl Iterator<Item = &GoalOutcome> {
        self.goals.iter().filter(|g| !g.proven)
    }

    /// Render a concise human-readable summary.
    pub fn summary(&self) -> String {
        let proven = self.goals.iter().filter(|g| g.proven).count();
        let total = self.goals.len();
        let mut out = format!(
            "plan: {} step(s), cost {}; goals proven {}/{}\n",
            self.plan.len(),
            self.total_cost,
            proven,
            total
        );
        for step in &self.plan {
            out.push_str(&format!(
                "  {} {} -> {}\n",
                if step.cached { "[cached]" } else { "[run]   " },
                step.tool_id,
                step.produces.join(", ")
            ));
        }
        for g in &self.goals {
            if g.proven {
                out.push_str(&format!("  PROVEN     {} ({})\n", g.goal.label(), g.target));
            } else {
                out.push_str(&format!(
                    "  UNSUPPORTED {} ({}): {}\n",
                    g.goal.label(),
                    g.target,
                    g.reason.as_deref().unwrap_or("no path")
                ));
            }
        }
        out
    }
}
