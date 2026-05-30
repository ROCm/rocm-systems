import { useEffect, useState, useCallback } from "react";
import {
  listProfiles,
  listSimulators,
  createProfile,
  deleteProfile,
} from "../api/client";
import type { ProfileDef, SimulatorSummary } from "../api/types";

export function ProfileListPage() {
  const [profiles, setProfiles] = useState<ProfileDef[]>([]);
  const [simulators, setSimulators] = useState<SimulatorSummary[]>([]);
  const [error, setError] = useState("");
  const [showForm, setShowForm] = useState(false);
  const [selectedSim, setSelectedSim] = useState("");

  const refresh = useCallback(() => {
    listProfiles().then(setProfiles).catch((e) => setError(String(e)));
  }, []);

  useEffect(refresh, [refresh]);

  useEffect(() => {
    listSimulators().then(setSimulators).catch(() => {});
  }, []);

  const activeSim = simulators.find((s) => s.name === selectedSim);

  const handleCreate = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const fd = new FormData(e.currentTarget);
    const profile: ProfileDef = {
      name: fd.get("name") as string,
      simulator: fd.get("simulator") as string,
      gpu: fd.get("gpu") as string,
      mode: (fd.get("mode") as ProfileDef["mode"]) || "Functional",
      num_gpus: Number(fd.get("num_gpus")) || 1,
      num_nodes: Number(fd.get("num_nodes")) || 1,
    };
    const res = await createProfile(profile);
    if (!res.ok) {
      setError(res.error);
    } else {
      setShowForm(false);
      refresh();
    }
  };

  const handleDelete = async (name: string) => {
    const res = await deleteProfile(name);
    if (!res.ok) {
      setError(res.error);
    } else {
      refresh();
    }
  };

  if (error) return <div className="error">{error}</div>;

  return (
    <div className="page">
      <div className="page-header">
        <h2>Profiles</h2>
        <button className="btn-primary" onClick={() => setShowForm(!showForm)}>
          {showForm ? "Cancel" : "+ New Profile"}
        </button>
      </div>

      {showForm && (
        <form className="create-form labeled-form" onSubmit={handleCreate}>
          <div className="form-field">
            <label htmlFor="pf-name">Profile Name</label>
            <input id="pf-name" name="name" required />
          </div>

          <div className="form-field">
            <label htmlFor="pf-simulator">Simulator</label>
            <select
              id="pf-simulator"
              name="simulator"
              required
              value={selectedSim}
              onChange={(e) => setSelectedSim(e.target.value)}
            >
              <option value="" disabled>
                Select simulator
              </option>
              {simulators.map((s) => (
                <option key={s.name} value={s.name}>
                  {s.name} v{s.version}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label htmlFor="pf-gpu">GPU</label>
            <select id="pf-gpu" name="gpu" required disabled={!activeSim}>
              <option value="" disabled>
                {activeSim ? "Select GPU" : "Pick a simulator first"}
              </option>
              {activeSim?.supported_gpus.map((g) => (
                <option key={g.name} value={g.name}>
                  {g.name} ({g.arch})
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label htmlFor="pf-mode">Mode</label>
            <select id="pf-mode" name="mode" required disabled={!activeSim}>
              <option value="" disabled>
                {activeSim ? "Select mode" : "Pick a simulator first"}
              </option>
              {activeSim?.supported_modes.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </div>

          <div className="form-field">
            <label htmlFor="pf-gpus">GPU Count</label>
            <input
              id="pf-gpus"
              name="num_gpus"
              type="number"
              min="1"
              defaultValue="1"
            />
          </div>

          <div className="form-field">
            <label htmlFor="pf-nodes">Node Count</label>
            <input
              id="pf-nodes"
              name="num_nodes"
              type="number"
              min="1"
              defaultValue="1"
            />
          </div>

          <div className="form-field form-actions">
            <button type="submit" className="btn-primary">
              Create
            </button>
          </div>
        </form>
      )}

      {profiles.length === 0 ? (
        <p className="empty">No profiles.</p>
      ) : (
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Simulator</th>
              <th>GPU</th>
              <th>Mode</th>
              <th>GPUs</th>
              <th>Nodes</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {profiles.map((p) => (
              <tr key={p.name}>
                <td><strong>{p.name}</strong></td>
                <td>{p.simulator}</td>
                <td><code>{p.gpu}</code></td>
                <td>{p.mode}</td>
                <td>{p.num_gpus}</td>
                <td>{p.num_nodes}</td>
                <td>
                  <button
                    className="btn-danger-sm"
                    onClick={() => handleDelete(p.name)}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
