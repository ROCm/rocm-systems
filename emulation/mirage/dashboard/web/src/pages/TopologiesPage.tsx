import { useEffect, useState, type FormEvent } from "react";
import * as api from "../api/client";
import type { TopologyDef } from "../api/types";
import { Dropdown } from "../components/ui/Dropdown";
import { Modal } from "../components/ui/Modal";
import { useToast } from "../components/ui/Toast";

const DEFAULT_FORM = {
  name: "",
  agent: "MI350X",
  racks: 1,
  nodes_per_rack: 1,
  gpus_per_node: 1,
};

export function TopologiesPage() {
  const toast = useToast();
  const [names, setNames] = useState<string[]>([]);
  const [agents, setAgents] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<TopologyDef | null>(null);
  const [createOpen, setCreateOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [form, setForm] = useState(DEFAULT_FORM);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      const [tops, ags] = await Promise.all([
        api.listTopologies(),
        api.listAgents().catch(() => [] as string[]),
      ]);
      setNames(tops);
      setAgents(ags);
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    api
      .getTopology(selected)
      .then(setDetail)
      .catch((e) => setError(String(e)));
  }, [selected]);

  async function onCreate(e: FormEvent) {
    e.preventDefault();
    const name = form.name.trim();
    if (!name) {
      setError("Name is required");
      return;
    }
    setBusy(true);
    try {
      await api.putTopology(name, {
        racks: form.racks,
        nodes_per_rack: form.nodes_per_rack,
        gpus_per_node: form.gpus_per_node,
        agent: form.agent,
      });
      toast.success(`Topology "${name}" saved`);
      setForm(DEFAULT_FORM);
      setCreateOpen(false);
      await refresh();
      setSelected(name);
    } catch (err) {
      const msg = String(err);
      setError(msg);
      toast.error(msg);
    } finally {
      setBusy(false);
    }
  }

  async function onConfirmDelete() {
    if (!confirmDelete) return;
    const target = confirmDelete;
    try {
      await api.deleteTopology(target);
      toast.success(`Topology "${target}" deleted`);
      setConfirmDelete(null);
      if (selected === target) setSelected(null);
      await refresh();
    } catch (e) {
      const msg = String(e);
      setError(msg);
      toast.error(msg);
    }
  }

  return (
    <div className="page page-wide">
      <div className="page-hero">
        <div>
          <h2>Topologies</h2>
          <p className="page-subtitle">
            Reusable rack / node / GPU layouts referenced by profiles.
          </p>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={() => setCreateOpen(true)}
          data-testid="open-topology-create"
        >
          + New topology
        </button>
      </div>
      {error && <div className="error" role="alert">{error}</div>}

      {names.length === 0 ? (
        <div className="empty-state" data-testid="topologies-empty">
          <h3>No topologies yet</h3>
          <p>Create one and reference it from a profile.</p>
        </div>
      ) : (
        <table className="data-table" data-testid="topologies-table">
          <thead>
            <tr>
              <th>Name</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {names.map((n) => (
              <tr key={n} data-testid={`topology-row-${n}`}>
                <td>
                  <button
                    type="button"
                    className="link-button"
                    onClick={() => setSelected(n)}
                    data-testid={`topology-show-${n}`}
                  >
                    <code>{n}</code>
                  </button>
                </td>
                <td className="row-actions">
                  <button
                    type="button"
                    className="btn-danger-sm"
                    onClick={() => setConfirmDelete(n)}
                    data-testid={`delete-topology-${n}`}
                  >
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {selected && detail && (
        <pre className="json-detail" data-testid={`topology-detail-${selected}`}>
          {JSON.stringify(detail, null, 2)}
        </pre>
      )}

      <Modal
        open={createOpen}
        onClose={() => setCreateOpen(false)}
        title="New topology"
        testId="topology-create-modal"
        footer={
          <>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setCreateOpen(false)}
            >
              Cancel
            </button>
            <button
              type="submit"
              form="topology-create-form"
              className="btn-primary"
              disabled={busy}
              data-testid="topology-create-submit"
            >
              {busy ? "Saving…" : "Save"}
            </button>
          </>
        }
      >
        <form
          id="topology-create-form"
          className="form-grid"
          onSubmit={onCreate}
        >
          <label className="form-field span-2">
            <span>Name</span>
            <input
              required
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              data-testid="topology-create-name"
              placeholder="e.g. MI350X-2x4"
            />
          </label>
          <label className="form-field">
            <span>Racks</span>
            <input
              type="number"
              min={1}
              value={form.racks}
              data-testid="topology-create-racks"
              onChange={(e) =>
                setForm({ ...form, racks: Math.max(1, +e.target.value || 1) })
              }
            />
          </label>
          <label className="form-field">
            <span>Nodes per rack</span>
            <input
              type="number"
              min={1}
              value={form.nodes_per_rack}
              data-testid="topology-create-nodes"
              onChange={(e) =>
                setForm({
                  ...form,
                  nodes_per_rack: Math.max(1, +e.target.value || 1),
                })
              }
            />
          </label>
          <label className="form-field">
            <span>GPUs per node</span>
            <input
              type="number"
              min={1}
              value={form.gpus_per_node}
              data-testid="topology-create-gpus"
              onChange={(e) =>
                setForm({
                  ...form,
                  gpus_per_node: Math.max(1, +e.target.value || 1),
                })
              }
            />
          </label>
          <div className="form-field">
            <span>Agent</span>
            <Dropdown
              testId="topology-create-agent"
              ariaLabel="Agent"
              value={form.agent}
              onChange={(v) => setForm({ ...form, agent: v })}
              options={
                agents.length
                  ? agents.map((a) => ({ value: a, label: a }))
                  : [{ value: form.agent, label: form.agent }]
              }
            />
          </div>
        </form>
      </Modal>

      <Modal
        open={confirmDelete !== null}
        onClose={() => setConfirmDelete(null)}
        title={`Delete "${confirmDelete}"?`}
        testId="confirm-delete-topology"
        footer={
          <>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setConfirmDelete(null)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn-danger"
              onClick={onConfirmDelete}
              data-testid="confirm-delete-topology-confirm"
            >
              Delete topology
            </button>
          </>
        }
      >
        <p>
          This removes the topology definition from disk. Profiles that
          reference it by name will fail to load until you recreate it.
        </p>
      </Modal>
    </div>
  );
}
