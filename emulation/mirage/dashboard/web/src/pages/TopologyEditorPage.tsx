import { useEffect, useRef, useState, useCallback } from "react";
import { createEditor, type TopologyEditor } from "../editor/editor";
import { exportConfig } from "../editor/export";
import { importConfig } from "../editor/import";
import "../editor/editor.css";

import defaultConfig from "../editor/default-config.json";

export function TopologyEditorPage() {
  const containerRef = useRef<HTMLDivElement>(null);
  const editorRef = useRef<TopologyEditor | null>(null);
  const [jsonOutput, setJsonOutput] = useState<string>("");
  const [showJson, setShowJson] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    let ed: TopologyEditor | null = null;
    let cancelled = false;
    createEditor(el).then(async (e) => {
      if (cancelled) { e.destroy(); return; }
      ed = e;
      editorRef.current = e;
      // Preload the CDNA4 config
      await importConfig(e.editor, defaultConfig as any);
      await e.arrange();
    });

    return () => {
      cancelled = true;
      ed?.destroy();
      editorRef.current = null;
    };
  }, []);

  const handleLoadConfig = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  const handleFileChange = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !editorRef.current) return;
    try {
      const text = await file.text();
      const config = JSON.parse(text);
      await importConfig(editorRef.current.editor, config);
      await editorRef.current.arrange();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      setJsonOutput(`Import error: ${msg}`);
      setShowJson(true);
    }
    // Reset the input so the same file can be loaded again
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

  return (
    <div className="topology-editor-page">
      <div className="topology-toolbar">
        <h2>GPU Topology Editor</h2>
        <div className="toolbar-actions">
          <button onClick={handleLoadConfig} className="btn btn-secondary">
            Load Config
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept=".json"
            style={{ display: "none" }}
            onChange={handleFileChange}
          />
          <button onClick={handleArrange} className="btn btn-secondary">
            Auto Arrange
          </button>
          <button onClick={handleExport} className="btn btn-primary">
            Export JSON
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
