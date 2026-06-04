import { useEffect, useState, type FormEvent } from "react";
import * as api from "../api/client";
import type { EmulatorEntry } from "../api/types";
import { Dropdown, MultiSelect } from "../components/ui/Dropdown";
import { Modal } from "../components/ui/Modal";
import { Pill } from "../components/ui/Status";
import { useToast } from "../components/ui/Toast";

const DEFAULT_NEW_PROFILE = {
  name: "",
  emulator: "noop",
  topology_mode: "new" as "new" | "existing",
  topology_pick: "",
  topology_save_as: "",
  num_nodes: 1,
  gpus_per_node: 1,
  agent: "MI350X",
  exec_mode: "Functional" as "Functional" | "Clocked",
  plugins: [] as string[],
  description: "",
  containerize: false,
  image: "",
  provider: "" as "" | "podman" | "docker",
  mounts: [] as { host: string; container: string; read_only: boolean }[],
};

export function ProfilesPage() {
  const toast = useToast();
  const [names, setNames] = useState<string[]>([]);
  const [emulators, setEmulators] = useState<EmulatorEntry[]>([]);
  const [topologies, setTopologies] = useState<string[]>([]);
  const [agents, setAgents] = useState<string[]>([]);
  const [error, setError] = useState("");
  const [wizardOpen, setWizardOpen] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);
  const [form, setForm] = useState(DEFAULT_NEW_PROFILE);
  const [busy, setBusy] = useState(false);

  async function refresh() {
    try {
      const [n, em, tops, ags] = await Promise.all([
        api.listProfiles(),
        api.listEmulators(),
        api.listTopologies().catch(() => [] as string[]),
        api.listAgents().catch(() => [] as string[]),
      ]);
      setNames(n);
      setEmulators(em);
      setTopologies(tops);
      setAgents(ags);
      if (em.length && !em.some((e) => e.name === form.emulator)) {
        const def = em.find((e) => e.is_default) ?? em[0];
        setForm((f) => ({ ...f, emulator: def.name }));
      }
      if (tops.length) {
        setForm((f) =>
          f.topology_pick ? f : { ...f, topology_pick: tops[0] },
        );
      }
    } catch (e) {
      setError(String(e));
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const selectedEmu = emulators.find((e) => e.name === form.emulator);

  async function onCreate(e: FormEvent) {
    e.preventDefault();
    setError("");
    const name = form.name.trim().toLowerCase();
    if (!name) {
      setError("Name is required");
      return;
    }
    setBusy(true);
    try {
      const pluginsRecord: Record<string, Record<string, unknown>> = {};
      for (const p of form.plugins) pluginsRecord[p] = {};

      let topology: import("../api/types").EmulatorDef["topology"];
      if (form.topology_mode === "existing") {
        if (!form.topology_pick) {
          setError("Pick a topology");
          setBusy(false);
          return;
        }
        topology = form.topology_pick;
      } else {
        const inline = {
          num_nodes: form.num_nodes,
          gpus_per_node: form.gpus_per_node,
          agent: form.agent,
        };
        const saveAs = form.topology_save_as.trim();
        if (saveAs) {
          await api.putTopology(saveAs, inline);
          topology = saveAs;
        } else {
          topology = inline;
        }
      }

      let containerize: import("../api/types").ContainerizedDef | undefined;
      if (form.containerize) {
        const image = form.image.trim();
        if (!image) {
          setError("Image is required when containerization is enabled");
          setBusy(false);
          return;
        }
        const mounts = form.mounts
          .filter((m) => m.host.trim())
          .map((m) => ({
            host_path: m.host.trim(),
            container_path: (m.container.trim() || m.host.trim()),
            read_only: m.read_only,
          }));
        containerize = {
          image,
          ...(form.provider ? { provider: form.provider } : {}),
          ...(mounts.length ? { mounts } : {}),
        };
      }

      await api.putProfile({
        name,
        description: form.description || undefined,
        emulator: {
          emulator: form.emulator,
          plugins: pluginsRecord,
          exec_mode: form.exec_mode,
          options: {},
          topology,
        },
        ...(containerize ? { containerize } : {}),
      });
      toast.success(`Profile "${name}" created`);
      setForm(DEFAULT_NEW_PROFILE);
      setWizardOpen(false);
      await refresh();
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
      await api.deleteProfile(target);
      toast.success(`Profile "${target}" deleted`);
      setConfirmDelete(null);
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
          <h2>Profiles</h2>
          <p className="page-subtitle">
            Reusable definitions of an emulator topology + plugin set.
          </p>
        </div>
        <button
          type="button"
          className="btn-primary"
          onClick={() => setWizardOpen(true)}
          data-testid="open-profile-wizard"
        >
          + New profile
        </button>
      </div>
      {error && <div className="error" role="alert">{error}</div>}

      {names.length === 0 ? (
        <div className="empty-state" data-testid="profiles-empty">
          <h3>No profiles yet</h3>
          <p>Create one to start launching sessions.</p>
        </div>
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
                <td className="row-actions">
                  <button
                    type="button"
                    className="btn-danger-sm"
                    onClick={() => setConfirmDelete(n)}
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

      <Modal
        open={wizardOpen}
        onClose={() => setWizardOpen(false)}
        title="New profile"
        testId="profile-wizard"
        footer={
          <>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => setWizardOpen(false)}
            >
              Cancel
            </button>
            <button
              type="submit"
              form="profile-wizard-form"
              className="btn-primary"
              disabled={busy}
              data-testid="wizard-submit"
            >
              {busy ? "Creating…" : "Create profile"}
            </button>
          </>
        }
      >
        <form
          id="profile-wizard-form"
          className="form-grid"
          onSubmit={onCreate}
        >
          <label className="form-field">
            <span>Name</span>
            <input
              required
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="e.g. cdna4-tiny"
              data-testid="wizard-name"
            />
          </label>

          <div className="form-field">
            <span>Emulator backend</span>
            <Dropdown
              testId="wizard-emulator"
              ariaLabel="Emulator backend"
              value={form.emulator}
              onChange={(v) => setForm({ ...form, emulator: v, plugins: [] })}
              options={emulators.map((e) => ({
                value: e.name,
                label: e.name,
                description: e.description,
                badge: {
                  text: e.installed ? "installed" : "not installed",
                  tone: e.installed ? "ok" : "muted",
                },
              }))}
            />
            {selectedEmu && (
              <span className="form-help">
                {selectedEmu.description}
                {!selectedEmu.installed && (
                  <>
                    {" "}
                    <Pill tone="warn" testId="wizard-not-installed">
                      not installed
                    </Pill>
                  </>
                )}
              </span>
            )}
          </div>

          <div className="form-field">
            <span>Exec mode</span>
            <div className="segmented" role="radiogroup" aria-label="Exec mode">
              {(["Functional", "Clocked"] as const).map((m) => (
                <button
                  type="button"
                  key={m}
                  role="radio"
                  aria-checked={form.exec_mode === m}
                  className={`segmented-option${form.exec_mode === m ? " is-on" : ""}`}
                  data-testid={`wizard-mode-${m.toLowerCase()}`}
                  onClick={() => setForm({ ...form, exec_mode: m })}
                >
                  {m}
                </button>
              ))}
            </div>
          </div>

          <div className="form-field span-2">
            <span>Topology source</span>
            <div className="segmented" role="radiogroup" aria-label="Topology source">
              {(
                [
                  ["existing", "Existing"],
                  ["new", "New (inline)"],
                ] as const
              ).map(([v, label]) => (
                <button
                  type="button"
                  key={v}
                  role="radio"
                  aria-checked={form.topology_mode === v}
                  className={`segmented-option${form.topology_mode === v ? " is-on" : ""}`}
                  data-testid={`wizard-topology-mode-${v}`}
                  onClick={() => setForm({ ...form, topology_mode: v })}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {form.topology_mode === "existing" ? (
            <div className="form-field span-2">
              <span>Topology</span>
              <Dropdown
                testId="wizard-topology-pick"
                ariaLabel="Topology"
                value={form.topology_pick}
                onChange={(v) => setForm({ ...form, topology_pick: v })}
                options={topologies.map((t) => ({ value: t, label: t }))}
              />
              {topologies.length === 0 && (
                <span className="form-help">
                  No topologies yet. Switch to "New (inline)" or create one
                  on the Topologies page.
                </span>
              )}
            </div>
          ) : (
            <>
              <label className="form-field">
                <span>Nodes per rack</span>
                <input
                  type="number"
                  min={1}
                  value={form.num_nodes}
                  data-testid="wizard-nodes"
                  onChange={(e) =>
                    setForm({
                      ...form,
                      num_nodes: Math.max(1, +e.target.value || 1),
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
                  data-testid="wizard-gpus"
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
                  testId="wizard-agent"
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

              <label className="form-field span-2">
                <span>Save topology as (optional)</span>
                <input
                  value={form.topology_save_as}
                  data-testid="wizard-topology-save-name"
                  placeholder="leave blank to keep inline"
                  onChange={(e) =>
                    setForm({ ...form, topology_save_as: e.target.value })
                  }
                />
                <span className="form-help">
                  If set, the topology is also saved to{" "}
                  <code>&lt;MIRAGE_CONFIG&gt;/topology/</code> and the profile
                  references it by name.
                </span>
              </label>
            </>
          )}

          <div className="form-field span-2">
            <span>Plugins</span>
            <MultiSelect
              testId="wizard-plugins"
              values={form.plugins}
              onChange={(plugins) => setForm({ ...form, plugins })}
              options={(selectedEmu?.available_plugins ?? []).map((p) => ({
                value: p,
                label: p,
              }))}
              emptyHint={
                <span className="muted">
                  {selectedEmu
                    ? `${selectedEmu.name} doesn't expose pluggable slots in the registry.`
                    : "Pick an emulator first."}
                </span>
              }
            />
          </div>

          <label className="form-field span-2">
            <span>Description</span>
            <textarea
              rows={3}
              value={form.description}
              data-testid="wizard-description"
              onChange={(e) => setForm({ ...form, description: e.target.value })}
              placeholder="What is this profile for?"
            />
          </label>

          <div className="form-field span-2">
            <span>Containerization</span>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={form.containerize}
                data-testid="wizard-containerize"
                onChange={(e) =>
                  setForm({ ...form, containerize: e.target.checked })
                }
              />
              <span>Run every node inside a container</span>
            </label>
            <span className="form-help">
              When enabled, each node of a session runs in its own
              container on a shared virtual network.
            </span>
          </div>

          {form.containerize && (
            <>
              <label className="form-field span-2">
                <span>Image</span>
                <input
                  required
                  value={form.image}
                  data-testid="wizard-image"
                  placeholder="e.g. rocm/dev-ubuntu-22.04:latest"
                  onChange={(e) =>
                    setForm({ ...form, image: e.target.value })
                  }
                />
              </label>

              <div className="form-field">
                <span>Provider</span>
                <Dropdown
                  testId="wizard-provider"
                  ariaLabel="Container provider"
                  value={form.provider}
                  onChange={(v) =>
                    setForm({ ...form, provider: v as "" | "podman" | "docker" })
                  }
                  options={[
                    { value: "", label: "Auto-detect" },
                    { value: "podman", label: "podman" },
                    { value: "docker", label: "docker" },
                  ]}
                />
              </div>

              <div className="form-field span-2">
                <span>Bind mounts</span>
                {form.mounts.length === 0 && (
                  <span className="form-help">
                    No mounts. Add one to share a host directory with every
                    node container.
                  </span>
                )}
                {form.mounts.map((m, i) => (
                  <div className="mount-row" key={i} data-testid={`wizard-mount-${i}`}>
                    <input
                      aria-label={`Mount ${i} host path`}
                      placeholder="host path"
                      value={m.host}
                      data-testid={`wizard-mount-host-${i}`}
                      onChange={(e) => {
                        const mounts = [...form.mounts];
                        mounts[i] = { ...mounts[i], host: e.target.value };
                        setForm({ ...form, mounts });
                      }}
                    />
                    <input
                      aria-label={`Mount ${i} container path`}
                      placeholder="container path (defaults to host)"
                      value={m.container}
                      data-testid={`wizard-mount-container-${i}`}
                      onChange={(e) => {
                        const mounts = [...form.mounts];
                        mounts[i] = { ...mounts[i], container: e.target.value };
                        setForm({ ...form, mounts });
                      }}
                    />
                    <label className="checkbox-row">
                      <input
                        type="checkbox"
                        checked={m.read_only}
                        aria-label={`Mount ${i} read-only`}
                        data-testid={`wizard-mount-ro-${i}`}
                        onChange={(e) => {
                          const mounts = [...form.mounts];
                          mounts[i] = {
                            ...mounts[i],
                            read_only: e.target.checked,
                          };
                          setForm({ ...form, mounts });
                        }}
                      />
                      <span>ro</span>
                    </label>
                    <button
                      type="button"
                      className="btn-danger-sm"
                      aria-label={`Remove mount ${i}`}
                      data-testid={`wizard-mount-remove-${i}`}
                      onClick={() =>
                        setForm({
                          ...form,
                          mounts: form.mounts.filter((_, j) => j !== i),
                        })
                      }
                    >
                      Remove
                    </button>
                  </div>
                ))}
                <button
                  type="button"
                  className="btn-secondary-sm"
                  data-testid="wizard-mount-add"
                  onClick={() =>
                    setForm({
                      ...form,
                      mounts: [
                        ...form.mounts,
                        { host: "", container: "", read_only: false },
                      ],
                    })
                  }
                >
                  + Add mount
                </button>
              </div>
            </>
          )}
        </form>
      </Modal>

      <Modal
        open={confirmDelete !== null}
        onClose={() => setConfirmDelete(null)}
        title={`Delete "${confirmDelete}"?`}
        testId="confirm-delete-profile"
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
              data-testid="confirm-delete-profile-confirm"
            >
              Delete profile
            </button>
          </>
        }
      >
        <p>
          This removes the profile definition from disk. Any active sessions
          using it will keep running.
        </p>
      </Modal>
    </div>
  );
}
