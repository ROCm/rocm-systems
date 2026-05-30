import { useEffect, useRef } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import "@xterm/xterm/css/xterm.css";
import { getTerminalExecId, terminalAttachUrl } from "../api/client";

interface Props {
  terminalId: string;
  onClose?: () => void;
  onDead?: () => void;
}

/** WebSocket terminal protocol over the daemon's generated `attach` endpoint.
 *
 *  Frames are JSON text messages shaped exactly like the Rust
 *  `AttachInput`/`AttachOutput`/`AttachReply` types.
 *  Client → server: `{ "stream": <number[]> }` where the array is the
 *  UTF-8 bytes of the keystroke(s).
 *  Server → client:
 *    `{ "type": "output", "data": { "is_stdout": bool, "output": <number[]> } }`
 *    `{ "type": "reply",  "data": { "exit_code": <number> } }`
 *    `{ "type": "error",  "message": <string> }`
 */

function encodeInput(data: string): string {
  const bytes = Array.from(new TextEncoder().encode(data));
  return JSON.stringify({ stream: bytes });
}

export function TerminalView({ terminalId, onClose, onDead }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const termRef = useRef<Terminal | null>(null);
  const fitRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  // Stash callbacks in refs so the effect below can run exactly once per
  // terminalId without tearing down the WebSocket whenever the parent
  // re-renders with a new `onDead`/`onClose` lambda.
  const onDeadRef = useRef(onDead);
  const onCloseRef = useRef(onClose);
  useEffect(() => {
    onDeadRef.current = onDead;
    onCloseRef.current = onClose;
  }, [onDead, onClose]);

  useEffect(() => {
    if (!containerRef.current) return;

    let cancelled = false;

    const term = new Terminal({
      cursorBlink: true,
      fontSize: 14,
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
      theme: {
        background: "#1a1a1a",
        foreground: "#e0e0e0",
        cursor: "#ED1C24",
        selectionBackground: "rgba(237, 28, 36, 0.3)",
      },
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(containerRef.current);
    fit.fit();
    term.focus();

    termRef.current = term;
    fitRef.current = fit;

    const execId = getTerminalExecId(terminalId);
    if (!execId) {
      term.write(`\r\n\x1b[31m[unknown terminal '${terminalId}']\x1b[0m\r\n`);
      return () => {
        term.dispose();
      };
    }

    const ws = new WebSocket(terminalAttachUrl(execId));
    wsRef.current = ws;

    ws.onmessage = (ev) => {
      if (cancelled) return;
      try {
        const msg = JSON.parse(ev.data as string);
        if (msg.type === "output") {
          const bytes = msg.data?.output as number[] | undefined;
          if (bytes && bytes.length) {
            const text = new TextDecoder().decode(new Uint8Array(bytes));
            term.write(text);
          }
        } else if (msg.type === "reply") {
          term.write("\r\n\x1b[31m[terminal exited]\x1b[0m\r\n");
          onDeadRef.current?.();
        } else if (msg.type === "error") {
          term.write(`\r\n\x1b[31m[attach error: ${msg.message}]\x1b[0m\r\n`);
          onDeadRef.current?.();
        }
      } catch {
        // malformed frame; ignore
      }
    };

    ws.onclose = () => {
      if (cancelled) return;
      term.write("\r\n\x1b[33m[disconnected]\x1b[0m\r\n");
      onDeadRef.current?.();
    };

    term.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(encodeInput(data));
      }
    });

    const onResize = () => {
      fit.fit();
    };
    window.addEventListener("resize", onResize);

    return () => {
      cancelled = true;
      window.removeEventListener("resize", onResize);
      if (ws.readyState === WebSocket.OPEN) ws.close();
      term.dispose();
    };
  }, [terminalId]);

  return (
    <div className="terminal-container">
      <div className="terminal-header">
        <span className="terminal-title">{terminalId}</span>
        {onClose && (
          <button className="btn-danger-sm" onClick={onClose}>
            Close
          </button>
        )}
      </div>
      <div
        ref={containerRef}
        className="terminal-xterm"
        onClick={() => termRef.current?.focus()}
      />
    </div>
  );
}
