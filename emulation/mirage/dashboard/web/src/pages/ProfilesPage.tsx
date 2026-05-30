import { useEffect, useState, type FormEvent } from "react";
import * as api from "../api/client";

const DEFAULT_NEW_PROFILE = {
  name: "",
  emulator: "noop",
  nodes: 1,
  gpus_per_node: 1,
  description: "",
};

export function ProfilesPage() {
  const [names, setNames] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [form, setForm] = useState(DEFAULT_NEW_PROFILE);

  async function refresh() {
    try {
      setNames(await api.listProfiles());
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function onCreate(e: FormEvent) {
    e.preventDefault();
    setError("");
    try {
      await api.putProfile({
        name: form.name,
        description: form.description || undefined,
        emulator: {
          emulator: form.emulator,
          plugins: {},
          nodes: form.nodes,
          gpus_per_node: form.gpus_per_node,
          exec_mode: "Functional",
          options: {},
          topology: { root: { name: "", type: "" } },
        },
      });
      setForm(DEFAULT_NEW_PROFILE);
      await refresh();
    } catch (err) {
      setError(String(err));
    }
  }

  async function onDelete(name: string) {
    if (!confirm(`Delete profile ${name}?`)) return;
    try {
      await api.deleteProfile(name);
      await refresh();
    } catch (e) {
      setError(String(e));
    }
  }

  return (
    <div className="page">
      <h2>Profiles</h2>
      {error && <div className="error" role="alert">{error}</div>}

      <h3>Existing</h3>
      {names.length === 0 ? (
        <p>(no profiles)</p>
      ) : (
        <table className="data-table" data-testid="profiles-table">
          <thead>
            <tr>
              <th>Name</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {names.map((n) => (
              <tr key={n} data-testid={`profile-row-${n}`}>
                <td>
                  <code>{n}</code>
                </td>
                <td>
                  <button
                    type="button"
                    onClick={() => onDelete(n)}
                    data-testid={`delete-profile-${n}`}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <h3>Create profile</h3>
      <form onSubmit={onCreate} className="form" data-testid="create-profile">
        <label>
          Name
          <input
            required
            value={form.name}
            onChange={(e) => setForm({ ...form, name: e.target.value })}
            data-testid="new-profile-name"
          />
        </label>
        <label>
          Emulator
          <input
            value={form.emulator}
            onChange={(e) => setForm({ ...form, emulator: e.target.value })}
            data-testid="new-profile-emulator"
          />
        </label>
        <label>
          Nodes
          <input
            type="number"
            min={1}
            value={form.nodes}
            onChange={(e) => setForm({ ...form, nodes: +e.target.value })}
          />
        </label>
        <label>
          GPUs / node
          <input
            type="number"
            min={1}
            value={form.gpus_per_node}
            onChange={(e) =>
              setForm({ ...form, gpus_per_node: +e.target.value })
            }
          />
        </label>
        <label>
          Description
          <input
            value={form.description}
            onChange={(e) => setForm({ ...form, description: e.target.value })}
          />
        </label>
        <button type="submit" data-testid="submit-profile">
          Create
        </button>
      </form>
    </div>
  );
}
