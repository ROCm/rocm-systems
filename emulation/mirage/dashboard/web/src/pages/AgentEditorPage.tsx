import { useEffect, useRef, useState, useCallback } from "react";
import { useNavigate, useParams } from "react-router-dom";
import * as api from "../api/client";
import type { AgentDef } from "../api/types";
import { useToast } from "../components/ui/Toast";
import { createEditor, type TopologyEditor } from "../editor/editor";
import { exportConfig } from "../editor/export";
import { importConfig } from "../editor/import";
import "../editor/editor.css";

import defaultConfig from "../editor/default-config.json";

export function AgentEditorPage() {
  const { name: routeName } = useParams();
  const navigate = useNavigate();
  const toast = useToast();
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<TopologyEditor | null>(null);
  /// Holds the original AgentDef so unmapped `vm.*` fields survive
  /// round-trips. `null` for brand-new agents.
  const originalRef = useRef<AgentDef | null>(null);
  const [jsonOutput, setJsonOutput] = useState<string>("");
  const [showJson, setShowJson] = useState(false);
  const [busy, setBusy] = useState(false);
  const [name, setName] = useState(routeName ?? "");
  const fileInputRef = useRef<HTMLInputElement>(null);

  const loadAgent = useCallback(async (agentName: string) => {
    if (!editorRef.current) return;
    try {
      const agent = await api.getAgent(agentName);
      originalRef.current = agent;
      const vm = agent.vm as Record<string, unknown>;
      const arch = (vm?.arch as string) ?? "cdna4";
      const config = {
        max_ticks: 100000,
        num_threads: 1,
        exec_mode: "functional",
        vm: { arch },
        topology: agent.topology,
      };
      await importConfig(editorRef.current.editor, config as any);
      await editorRef.current.arrange();
      setName(agentName);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      toast.error(`Load failed: ${msg}`);
    }
  }, [toast]);

  // Build the editor once.
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    let ed: TopologyEditor | null = null;
    let cancelled = false;
    createEditor(el).then(async (e) => {
      if (cancelled) {
        e.destroy();
        return;
      }
      ed = e;
      editorRef.current = e;
      // Preload the default CDNA4 config so the canvas isn't empty.
      await importConfig(e.editor, defaultConfig as any);
      await e.arrange();
      // If we were opened with a specific agent, load it on top.
      if (routeName) {
        await loadAgent(routeName);
      }
    });
    return () => {
      cancelled = true;
      ed?.destroy();
      editorRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const handleLoadFile = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !editorRef.current) return;
    try {
      const text = await file.text();
      const config = JSON.parse(text);
      // If the file looks like an AgentDef ({ vm, topology }) without
      // the editor's sim-config wrapper, fold it into the expected
      // shape so the editor can ingest it.
      const looksLikeAgent =
        config && typeof config === "object" && "vm" in config && "topology" in config
        && !("max_ticks" in config);
      const editorInput = looksLikeAgent
        ? {
            max_ticks: 100000,
            num_threads: 1,
            exec_mode: "functional",
            vm: { arch: (config.vm?.arch as string) ?? "cdna4" },
            topology: config.topology,
          }
        : config;
      if (looksLikeAgent) {
        originalRef.current = config as AgentDef;
      }
      await importConfig(editorRef.current.editor, editorInput);
      await editorRef.current.arrange();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setJsonOutput(`Import error: ${msg}`);
      setShowJson(true);
    }
    e.target.value = "";
  }, []);

  const handleExport = useCallback(() => {
    if (!editorRef.current) return;
    try {
      const config = exportConfig(editorRef.current.editor);
      setJsonOutput(JSON.stringify(config, null, 2));
      setShowJson(true);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setJsonOutput(`Error: ${msg}`);
      setShowJson(true);
    }
  }, []);

  const handleArrange = useCallback(() => {
    editorRef.current?.arrange();
  }, []);

  const handleCopyJson = useCallback(() => {
    navigator.clipboard.writeText(jsonOutput);
  }, [jsonOutput]);

  /// Build an `AgentDef` from the editor state. If we have an
  /// `original` we preserve every `vm.*` field except `arch`, which
  /// is the only thing the editor actually exposes today.
  function buildAgent(): AgentDef {
    if (!editorRef.current) throw new Error("editor not ready");
    const cfg = exportConfig(editorRef.current.editor);
    const baseVm: Record<string, unknown> = {
      ...(originalRef.current?.vm ?? {}),
    };
    baseVm.arch = (cfg.vm as { arch: string }).arch;
    return {
      vm: baseVm,
      topology: cfg.topology as unknown as Record<string, unknown>,
    };
  }

  async function doSave(target: string) {
    setBusy(true);
    try {
      const agent = buildAgent();
      await api.putAgent(target, agent);
      toast.success(`Agent "${target}" saved`);
      navigate(`/agents/edit/${encodeURIComponent(target)}`, { replace: true });
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      toast.error(`Save failed: ${msg}`);
    } finally {
      setBusy(false);
    }
  }

  const handleSave = useCallback(async () => {
    const target = name.trim();
    if (!target) {
      const prompted = window.prompt("Save agent as:")?.trim();
      if (!prompted) return;
      setName(prompted);
      return doSave(prompted);
    }
    return doSave(target);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name]);

  const handleSaveAs = useCallback(async () => {
    const prompted = window.prompt("Save agent as:", name)?.trim();
    if (!prompted) return;
    setName(prompted);
    return doSave(prompted);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [name]);

  return (
    <div className="topology-editor-page">
      <div className="topology-toolbar">
        <h2>
          Agent Editor
          {name && (
            <>
              {" "}
              — <code data-testid="agent-editor-name">{name}</code>
            </>
          )}
        </h2>
        <div className="toolbar-actions">
          <button
            type="button"
            onClick={() => navigate("/agents")}
            className="btn btn-secondary"
          >
            ← Agents
          </button>
          <button
            type="button"
            onClick={handleLoadFile}
            className="btn btn-secondary"
          >
            Load file
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          <button
            type="button"
            onClick={handleArrange}
            className="btn btn-secondary"
          >
            Auto Arrange
          </button>
          <button
            type="button"
            onClick={handleExport}
            className="btn btn-secondary"
          >
            Export JSON
          </button>
          <button
            type="button"
            onClick={handleSaveAs}
            className="btn btn-secondary"
            disabled={busy}
            data-testid="agent-editor-save-as"
          >
            Save as…
          </button>
          <button
            type="button"
            onClick={handleSave}
            className="btn btn-primary"
            disabled={busy}
            data-testid="agent-editor-save"
          >
            {busy ? "Saving…" : "Save"}
          </button>
        </div>
      </div>
      <div className="topology-editor-container">
        <div ref={containerRef} className="rete-editor" />
        {showJson && (
          <div className="json-panel">
            <div className="json-panel-header">
              <span>Exported Config</span>
              <div>
                <button onClick={handleCopyJson} className="btn btn-small">
                  Copy
                </button>
                <button
                  onClick={() => setShowJson(false)}
                  className="btn btn-small"
                >
                  Close
                </button>
              </div>
            </div>
            <pre className="json-output">{jsonOutput}</pre>
          </div>
        )}
      </div>
    </div>
  );
}
