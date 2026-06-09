import { useEffect, useState } from "react";
import * as api from "../api/client";
import type {
  BedrocExecutionReport,
  BedrocFuzzSummary,
  BedrocProof,
  BedrocTool,
} from "../api/types";
import { useToast } from "../components/ui/Toast";

/// The correctness properties a user can ask bedroc to prove. The values
/// match `mirage_bedroc::GoalKind`'s snake_case serialization / aliases.
const GOAL_OPTIONS: { value: string; label: string }[] = [
  { value: "no_data_hazards", label: "No data hazards" },
  { value: "correct_output", label: "Correct output" },
  { value: "no_data_races", label: "No data races" },
  { value: "fp_correct", label: "Correct floating-point" },
];

const SOURCE_KINDS = ["hip", "asm", "codeobject"];

export function BedrocPage() {
  const toast = useToast();
  const [tools, setTools] = useState<BedrocTool[]>([]);
  const [source, setSource] = useState("kernel.hip");
  const [sourceKind, setSourceKind] = useState("hip");
  const [targetsText, setTargetsText] = useState("mi350, mi450");
  const [goals, setGoals] = useState<string[]>([
    "no_data_hazards",
    "correct_output",
  ]);
  const [proof, setProof] = useState<BedrocProof | null>(null);
  const [execution, setExecution] = useState<BedrocExecutionReport | null>(null);
  const [fuzz, setFuzz] = useState<BedrocFuzzSummary | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .listBedrocTools()
      .then(setTools)
      .catch((e) => setError(String(e)));
  }, []);

  function parseTargets(): string[] {
    return targetsText
      .split(/[,\s]+/)
      .map((t) => t.trim())
      .filter(Boolean);
  }

  function toggleGoal(value: string) {
    setGoals((prev) =>
      prev.includes(value)
        ? prev.filter((g) => g !== value)
        : [...prev, value],
    );
  }

  function body() {
    return {
      source,
      source_kind: sourceKind,
      targets: parseTargets(),
      goals,
    };
  }

  async function onPlan() {
    setBusy(true);
    setError("");
    setExecution(null);
    try {
      const p = await api.bedrocPlan(body());
      setProof(p);
      toast.success(
        p.plan.length
          ? `Planned ${p.plan.length} step(s)`
          : "Nothing to plan",
      );
    } catch (e) {
      setError(String(e));
      toast.error(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onRun() {
    setBusy(true);
    setError("");
    try {
      const r = await api.bedrocRun(body());
      setProof(r.proof);
      setExecution(r.execution);
      toast.success(
        `Executed ${r.execution.executed}, reused ${r.execution.cache_hits}`,
      );
    } catch (e) {
      setError(String(e));
      toast.error(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function onFuzz() {
    setBusy(true);
    setError("");
    try {
      const s = await api.bedrocFuzz(200, 0);
      setFuzz(s);
      if (s.inconsistencies.length === 0) {
        toast.success(
          `Fuzz OK: ${s.multi_route_goals} multi-route goals checked`,
        );
      } else {
        toast.error(`Fuzz found ${s.inconsistencies.length} inconsistencies`);
      }
    } catch (e) {
      setError(String(e));
      toast.error(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="page page-wide" data-testid="bedroc-page">
      <div className="page-hero">
        <div>
          <h2>Bedroc</h2>
          <p className="page-subtitle">
            Define what you want proven about a kernel; bedroc combines it
            with this machine's constraints to produce an optimal plan and
            proof.
          </p>
        </div>
      </div>

      {error && (
        <div className="error" role="alert">
          {error}
        </div>
      )}

      <div className="card">
        <h3>Request</h3>
        <div className="form-grid">
          <label>
            Source
            <input
              value={source}
              onChange={(e) => setSource(e.target.value)}
              data-testid="bedroc-source"
            />
          </label>
          <label>
            Source kind
            <select
              value={sourceKind}
              onChange={(e) => setSourceKind(e.target.value)}
              data-testid="bedroc-source-kind"
            >
              {SOURCE_KINDS.map((k) => (
                <option key={k} value={k}>
                  {k}
                </option>
              ))}
            </select>
          </label>
          <label>
            Targets (comma/space separated)
            <input
              value={targetsText}
              onChange={(e) => setTargetsText(e.target.value)}
              data-testid="bedroc-targets"
              placeholder="mi350, mi450, gfx942"
            />
          </label>
        </div>
        <fieldset className="goal-set">
          <legend>Prove</legend>
          {GOAL_OPTIONS.map((g) => (
            <label key={g.value} className="goal-option">
              <input
                type="checkbox"
                checked={goals.includes(g.value)}
                onChange={() => toggleGoal(g.value)}
                data-testid={`bedroc-goal-${g.value}`}
              />
              {g.label}
            </label>
          ))}
        </fieldset>
        <div className="button-row">
          <button onClick={onPlan} disabled={busy} data-testid="bedroc-plan">
            Plan
          </button>
          <button onClick={onRun} disabled={busy} data-testid="bedroc-run">
            Run (with cache)
          </button>
          <button onClick={onFuzz} disabled={busy} data-testid="bedroc-fuzz">
            Fuzz catalogue
          </button>
        </div>
      </div>

      {proof && (
        <div className="card" data-testid="bedroc-proof">
          <h3>
            Proof — {proof.plan.length} step(s), cost {proof.total_cost}
          </h3>
          <table className="table">
            <thead>
              <tr>
                <th>Property</th>
                <th>Target</th>
                <th>Status</th>
                <th>Detail</th>
              </tr>
            </thead>
            <tbody>
              {proof.goals.map((g) => (
                <tr key={g.fact}>
                  <td>{g.goal}</td>
                  <td>{g.target}</td>
                  <td>
                    <span
                      className={g.proven ? "pill pill-ok" : "pill pill-warn"}
                      data-testid={`bedroc-outcome-${g.fact}`}
                    >
                      {g.proven ? "PROVEN" : "UNSUPPORTED"}
                    </span>
                  </td>
                  <td className="muted">{g.reason ?? ""}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h4>Plan</h4>
          <ol className="plan-list">
            {proof.plan.map((s, i) => (
              <li key={`${s.tool_id}-${i}`}>
                <code>{s.tool_id}</code>{" "}
                <span className="muted">→ {s.produces.join(", ")}</span>{" "}
                {s.cached && <span className="pill pill-ok">cached</span>}
                <span className="muted"> (cost {s.cost})</span>
              </li>
            ))}
          </ol>

          {execution && (
            <p className="muted" data-testid="bedroc-exec-summary">
              Executed {execution.executed} step(s), reused{" "}
              {execution.cache_hits} from cache.
            </p>
          )}

          {proof.unavailable_tools.length > 0 && (
            <>
              <h4>Unavailable tools here</h4>
              <ul className="muted">
                {proof.unavailable_tools.map((t) => (
                  <li key={t.id}>
                    <code>{t.id}</code> — {t.reason}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}

      {fuzz && (
        <div className="card" data-testid="bedroc-fuzz-result">
          <h3>Differential fuzz</h3>
          <p>
            {fuzz.requests} requests · {fuzz.goals_checked} goals ·{" "}
            {fuzz.total_routes} routes · {fuzz.multi_route_goals} multi-route
            goals
          </p>
          {fuzz.inconsistencies.length === 0 ? (
            <p className="pill pill-ok">All routes agree with the planner</p>
          ) : (
            <ul className="error">
              {fuzz.inconsistencies.map((i, idx) => (
                <li key={idx}>{i}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="card">
        <h3>Tool catalogue</h3>
        <table className="table" data-testid="bedroc-tools">
          <thead>
            <tr>
              <th>Id</th>
              <th>Category</th>
              <th>Produces</th>
              <th>Available</th>
            </tr>
          </thead>
          <tbody>
            {tools.map((t) => (
              <tr key={t.id}>
                <td>
                  <code>{t.id}</code>
                </td>
                <td>{t.category}</td>
                <td className="muted">{t.produces.join(", ")}</td>
                <td>
                  <span className={t.available ? "pill pill-ok" : "pill pill-warn"}>
                    {t.available ? "yes" : "no"}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
