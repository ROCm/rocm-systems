import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import * as api from "../api/client";
import type { AgentDef } from "../api/types";
import { Modal } from "../components/ui/Modal";
import { useToast } from "../components/ui/Toast";

export function AgentsPage() {
  const navigate = useNavigate();
  const toast = useToast();
  const [names, setNames] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<AgentDef | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [error, setError] = useState("");
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [importName, setImportName] = useState("");
  const [importOpen, setImportOpen] = useState(false);
  const [importText, setImportText] = useState("");
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      setNames(await api.listAgents());
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
      .getAgent(selected)
      .then(setDetail)
      .catch((e) => setError(String(e)));
  }, [selected]);

  async function onImport() {
    const name = importName.trim();
    if (!name) {
      setError("Name is required");
      return;
    }
    setBusy(true);
    try {
      const parsed = JSON.parse(importText) as AgentDef;
      await api.putAgent(name, parsed);
      toast.success(`Agent "${name}" imported`);
      setImportOpen(false);
      setImportName("");
      setImportText("");
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
      await api.deleteAgent(target);
      toast.success(`Agent "${target}" deleted`);
      setConfirmDelete(null);
      if (selected === target) setSelected(null);
      await refresh();
    } catch (e) {
      const msg = String(e);
      setError(msg);
      toast.error(msg);
    }
  }

  async function onFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    setImportText(await file.text());
    setImportName(file.name.replace(/\.json$/i, ""));
    setImportOpen(true);
  }

  return (
    <div className="page page-wide">
      <div className="page-hero">
        <div>
          <h2>Agents</h2>
          <p className="page-subtitle">
            Hardware GPU definitions (VM + per-agent topology) used by
            topologies.
          </p>
        </div>
        <div className="row-actions">
          <input
            ref={fileInputRef}
            type="file"
            accept=".json,application/json"
            onChange={onFileChange}
            style={{ display: "none" }}
            data-testid="agent-import-file"
          />
          <button
            type="button"
            className="btn-secondary"
            onClick={() => fileInputRef.current?.click()}
            data-testid="open-agent-import"
          >
            Import JSON…
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={() => navigate("/agents/new")}
            data-testid="open-agent-editor-new"
          >
            + New agent
          </button>
        </div>
      </div>
      {error && <div className="error" role="alert">{error}</div>}

      {names.length === 0 ? (
        <div className="empty-state" data-testid="agents-empty">
          <h3>No agents yet</h3>
          <p>Import an agent definition to get started.</p>
        </div>
      ) : (
        <table className="data-table" data-testid="agents-table">
          <thead>
            <tr>
              <th>Name</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {names.map((n) => (
              <tr key={n} data-testid={`agent-row-${n}`}>
                <td>
                  <button
                    type="button"
                    className="link-button"
                    onClick={() => setSelected(n)}
                    data-testid={`agent-show-${n}`}
                  >
                    <code>{n}</code>
                  </button>
                </td>
                <td className="row-actions">
                  <Link
                    to={`/agents/edit/${encodeURIComponent(n)}`}
                    className="btn-secondary-sm"
                    data-testid={`edit-agent-${n}`}
                  >
                    Edit
                  </Link>
                  <button
                    type="button"
                    className="btn-danger-sm"
                    onClick={() => setConfirmDelete(n)}
                    data-testid={`delete-agent-${n}`}
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
        <pre className="json-detail" data-testid={`agent-detail-${selected}`}>
          {JSON.stringify(detail, null, 2)}
        </pre>
      )}

      <Modal
        open={importOpen}
        onClose={() => setImportOpen(false)}
        title="Import agent"
        testId="agent-import-modal"
        footer={
          <>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setImportOpen(false)}
            >
              Cancel
            </button>
            <button
              type="button"
              className="btn-primary"
              disabled={busy}
              onClick={onImport}
              data-testid="agent-import-submit"
            >
              {busy ? "Importing…" : "Import"}
            </button>
          </>
        }
      >
        <label className="form-field">
          <span>Name</span>
          <input
            value={importName}
            onChange={(e) => setImportName(e.target.value)}
            data-testid="agent-import-name"
          />
        </label>
        <label className="form-field">
          <span>JSON</span>
          <textarea
            rows={12}
            value={importText}
            onChange={(e) => setImportText(e.target.value)}
            data-testid="agent-import-json"
          />
        </label>
      </Modal>

      <Modal
        open={confirmDelete !== null}
        onClose={() => setConfirmDelete(null)}
        title={`Delete "${confirmDelete}"?`}
        testId="confirm-delete-agent"
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
              data-testid="confirm-delete-agent-confirm"
            >
              Delete agent
            </button>
          </>
        }
      >
        <p>
          This removes the agent definition from disk. Topologies that
          reference it by name will fail to load until you recreate it.
        </p>
      </Modal>
    </div>
  );
}
